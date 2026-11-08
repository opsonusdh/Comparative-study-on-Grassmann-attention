#!/usr/bin/env python3
"""Recursive, one-way, incremental Kaggle -> Google Drive sync.

Copies every eligible file under the DeepSeek SWA + CSA + HCA baseline run directory to:
    My Drive/loop_llm/deepseek_baseline_runs/DeepSeek_SWA_CSA_HCA/<relative path>

The local tree contains checkpoints/ (latest.pt, metrics.jsonl, run_config.json)
and visualizations/ (step_*.png plus any associated JSON/support files).

- Preserves the complete directory structure (for the DeepSeek baseline run).
- Skips files whose MD5 matches the remote binary file.
- Updates existing files in-place; does not create version-named snapshots.
- Creates missing destination folders.
- Never downloads from Drive and never deletes remote files.
- Takes a stable local temporary snapshot before hashing/uploading, so files
  appended/replaced during a sync are retried or skipped rather than uploaded
  from an inconsistent read.
- Existing same-name remote files are all updated in-place. Google Drive permits
  duplicate names; this script does not reject them or create another copy.
- If duplicate same-name folders exist, the most recently modified folder is
  selected deterministically and a warning is printed; other folder trees are
  left untouched.

Run manually from a Kaggle notebook cell. Requires kaggle_secrets and pydrive2.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from kaggle_secrets import UserSecretsClient
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


# ============================================================================
# CONFIGURATION
# ============================================================================

# This must be the parent directory containing the two folders written by
# your DeepSeek SWA + CSA + HCA training command: checkpoints/ and visualizations/.
LOCAL_ROOT = Path(
    "/kaggle/working/loop_llm/deepseek_baseline_runs/DeepSeek_SWA_CSA_HCA"
)

# Remote layout:
# My Drive/loop_llm/deepseek_baseline_runs/DeepSeek_SWA_CSA_HCA/...
DRIVE_ROOT_NAME = "loop_llm"
DRIVE_DESTINATION_PATH = (
    "deepseek_baseline_runs",
    "DeepSeek_SWA_CSA_HCA",
)

# Sync recursively. This includes checkpoints/latest.pt, checkpoints/metrics.jsonl,
# checkpoints/run_config.json, every step_*.png, and visualization sidecar files.

# Safety: do not upload common transient/partial files. Hidden files are not
# automatically excluded; add unwanted names/patterns explicitly below.
EXCLUDED_DIR_NAMES = {".git", "__pycache__"}
EXCLUDED_FILE_SUFFIXES = {".tmp", ".part", ".lock", ".incomplete"}
EXCLUDED_FILE_NAMES: set[str] = set()

# Set True to print the plan without uploading. Default False performs sync.
DRY_RUN = False

# Verify every successful upload by fetching its remote MD5 checksum.
VERIFY_UPLOADS = True

# If a file changes while being copied to a temporary snapshot, retry this many
# times. If it keeps changing (e.g. a rapidly appended JSONL file), skip it and
# report it rather than upload a potentially inconsistent snapshot.
SNAPSHOT_ATTEMPTS = 4
SNAPSHOT_RETRY_DELAY_SECONDS = 0.5

# PyDrive2 authenticated credential JSON stored in Kaggle Secrets.
GDRIVE_SECRET_NAME = "GDRIVE_CREDENTIALS_JSON"

FOLDER_MIME = "application/vnd.google-apps.folder"


# ============================================================================
# LOCAL FILE DISCOVERY AND SNAPSHOTS
# ============================================================================

def is_excluded_file(path: Path) -> bool:
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    if path.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
        return True
    # Avoid uploading editor/OS temporary files.
    if path.name.startswith((".~", "#")) or path.name.endswith("~"):
        return True
    return False


def discover_local_files(root: Path) -> list[Path]:
    """Return eligible regular files recursively, sorted by relative path."""
    files: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(current_root)
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in EXCLUDED_DIR_NAMES
            and not (current / name).is_symlink()
        )
        for filename in filenames:
            path = current / filename
            if path.is_symlink() or not path.is_file() or is_excluded_file(path):
                continue
            files.append(path)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def stable_snapshot(source: Path, temp_dir: Path) -> Path:
    """Copy source to a temporary file only if it stayed unchanged while read."""
    snapshot = temp_dir / "upload_snapshot.bin"
    last_problem = "file changed while snapshot was being copied"

    for attempt in range(1, SNAPSHOT_ATTEMPTS + 1):
        try:
            before = source.stat()
            shutil.copyfile(source, snapshot)
            after = source.stat()
            copied = snapshot.stat()

            unchanged = (
                before.st_size == after.st_size == copied.st_size
                and before.st_mtime_ns == after.st_mtime_ns
                and before.st_ino == after.st_ino
            )
            if unchanged:
                return snapshot

            last_problem = (
                f"changed during snapshot (size {before.st_size}->{after.st_size}, "
                f"mtime_ns {before.st_mtime_ns}->{after.st_mtime_ns})"
            )
        except FileNotFoundError:
            last_problem = "file was replaced or removed during snapshot"

        if attempt < SNAPSHOT_ATTEMPTS:
            time.sleep(SNAPSHOT_RETRY_DELAY_SECONDS)

    raise RuntimeError(last_problem)


def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# ============================================================================
# GOOGLE DRIVE HELPERS
# ============================================================================

def q_escape(value: str) -> str:
    """Escape a string literal for Drive API v2 q expressions."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def list_children(drive: GoogleDrive, parent_id: str) -> list[Any]:
    """List every direct child, allowing PyDrive2 to follow all pages.

    Important: PyDrive2 GetList() automatically iterates pages only when
    maxResults is omitted. Supplying maxResults=1000 makes it fetch one page.
    """
    query = f"'{q_escape(parent_id)}' in parents and trashed = false"
    return drive.ListFile({
        "q": query,
        # Include the fields needed for safe matching and checksum comparison.
        # Do not pass maxResults: PyDrive2 only auto-paginates when it is omitted.
        "fields": "items(id,title,mimeType,md5Checksum,modifiedDate),nextPageToken",
    }).GetList()


def _newest_drive_item(items: list[Any]) -> Any:
    """Choose a deterministic folder when Drive contains duplicate folder names."""
    return max(
        items,
        key=lambda item: (str(item.get("modifiedDate", "")), str(item.get("id", ""))),
    )


def index_children(
    children: list[Any],
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    """Index folders by name and retain *all* same-name files.

    Drive permits duplicate names. For files, every matching remote copy is
    updated in-place by the sync loop. For duplicate folders, the newest folder
    is selected to avoid syncing into an arbitrary tree; other folder trees are
    not modified.
    """
    folder_groups: dict[str, list[Any]] = defaultdict(list)
    file_groups: dict[str, list[Any]] = defaultdict(list)

    for item in children:
        title = item.get("title")
        if not title:
            continue
        if item.get("mimeType") == FOLDER_MIME:
            folder_groups[title].append(item)
        else:
            file_groups[title].append(item)

    folders: dict[str, Any] = {}
    for name, items in folder_groups.items():
        if len(items) > 1:
            selected = _newest_drive_item(items)
            print(
                f"[Drive][WARN] {len(items)} folders named {name!r}; "
                f"using most recently modified folder id={selected.get('id')}"
            )
            folders[name] = selected
        else:
            folders[name] = items[0]

    files: dict[str, list[Any]] = dict(file_groups)
    for name, items in files.items():
        if len(items) > 1:
            print(
                f"[Drive][INFO] {len(items)} existing files named {name!r}; "
                "all copies will be overwritten in-place if content differs."
            )

    return folders, files


def create_folder(drive: GoogleDrive, name: str, parent_id: str) -> Any:
    folder = drive.CreateFile({
        "title": name,
        "mimeType": FOLDER_MIME,
        "parents": [{"id": parent_id}],
    })
    folder.Upload()
    return folder


def ensure_folder(
    drive: GoogleDrive,
    name: str,
    parent_id: str,
    cached_children: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> Any:
    folders, files = cached_children if cached_children is not None else index_children(
        list_children(drive, parent_id)
    )

    if name in files:
        raise RuntimeError(
            f"Cannot create/find destination folder {name!r}: a file with that name "
            "already exists in its parent folder."
        )
    if name in folders:
        return folders[name]

    print(f"[Drive] Creating folder: {name}")
    return create_folder(drive, name, parent_id)


def authenticate() -> GoogleDrive:
    print("[Drive] Authenticating with Google Drive...")
    secret = UserSecretsClient().get_secret(GDRIVE_SECRET_NAME)
    if not secret or not secret.strip():
        raise RuntimeError(f"Kaggle Secret {GDRIVE_SECRET_NAME!r} is empty or missing.")

    credentials_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        ) as temp_credentials:
            temp_credentials.write(secret)
            credentials_path = temp_credentials.name

        gauth = GoogleAuth()
        gauth.LoadCredentialsFile(credentials_path)

        if gauth.credentials is None:
            raise RuntimeError(
                f"{GDRIVE_SECRET_NAME} does not contain usable authorized PyDrive2 credentials."
            )

        if gauth.access_token_expired:
            print("[Drive] Access token expired; refreshing credentials...")
            gauth.Refresh()
        else:
            gauth.Authorize()

        drive = GoogleDrive(gauth)
        print("[Drive] Authentication ready.")
        return drive
    finally:
        if credentials_path:
            try:
                os.unlink(credentials_path)
            except OSError:
                pass


def upload_or_update(
    drive: GoogleDrive,
    snapshot: Path,
    local_name: str,
    remote_file: Any | None,
    parent_id: str,
    expected_md5: str,
) -> str:
    """Upload a stable snapshot, updating an existing file in place."""
    if remote_file is not None:
        remote_mime = remote_file.get("mimeType", "")
        if remote_mime.startswith("application/vnd.google-apps."):
            raise RuntimeError(
                f"A Google Workspace-native file named {local_name!r} already exists. "
                "Refusing to replace it with a binary training artifact. Rename or "
                "move the remote document first."
            )

        remote_md5 = remote_file.get("md5Checksum")
        if remote_md5 and remote_md5.lower() == expected_md5.lower():
            return "unchanged"

        remote_file.SetContentFile(str(snapshot))
        remote_file.Upload()
        result = "updated"
    else:
        mime_type = mimetypes.guess_type(local_name)[0] or "application/octet-stream"
        remote_file = drive.CreateFile({
            "title": local_name,
            "mimeType": mime_type,
            "parents": [{"id": parent_id}],
        })
        remote_file.SetContentFile(str(snapshot))
        remote_file.Upload()
        result = "new"

    if VERIFY_UPLOADS:
        remote_file.FetchMetadata(fields="md5Checksum")
        uploaded_md5 = remote_file.get("md5Checksum")
        if not uploaded_md5 or uploaded_md5.lower() != expected_md5.lower():
            raise RuntimeError(
                f"Remote checksum verification failed for {local_name!r}. "
                f"Expected {expected_md5}, received {uploaded_md5!r}."
            )

    return result


# ============================================================================
# MAIN SYNC
# ============================================================================

def main() -> None:
    root = LOCAL_ROOT.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Local sync directory not found: {root}")

    checkpoints_dir = root / "checkpoints"
    visualizations_dir = root / "visualizations"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoints_dir}. "
            "Make sure --checkpoint-dir points inside LOCAL_ROOT/checkpoints."
        )
    if not visualizations_dir.is_dir():
        raise FileNotFoundError(
            f"Visualization directory not found: {visualizations_dir}. "
            "Make sure --viz-dir points inside LOCAL_ROOT/visualizations."
        )

    checkpoint_file = checkpoints_dir / "latest.pt"
    if not checkpoint_file.is_file() or checkpoint_file.stat().st_size == 0:
        raise RuntimeError(
            f"No usable checkpoint at {checkpoint_file}. "
            "Wait until the trainer has completed at least one checkpoint save."
        )

    step_images = sorted(visualizations_dir.glob("step_*.png"))
    if not step_images:
        raise RuntimeError(
            f"No step_*.png visualizations found in {visualizations_dir}. "
            "Wait until the trainer has completed at least one visualization save."
        )

    local_files = discover_local_files(root)
    if not local_files:
        raise RuntimeError(f"No eligible files found under {root}")

    rels = {path.relative_to(root).as_posix() for path in local_files}
    required_payloads = {
        "checkpoints/latest.pt",
        "checkpoints/metrics.jsonl",
        "checkpoints/run_config.json",
    }
    missing_metadata = sorted(
        name for name in required_payloads - {
            "checkpoints/latest.pt"
        } if name not in rels
    )
    if missing_metadata:
        print(
            "[WARN] The following metadata files are not present yet and will not be uploaded: "
            + ", ".join(missing_metadata),
            flush=True,
        )

    total_bytes = sum(path.stat().st_size for path in local_files)
    print("=" * 72)
    print("GOOGLE DRIVE RECURSIVE DELTA SYNC — DEEPSEEK BASELINE")
    print("=" * 72)
    print(f"Local source : {root}")
    remote_path = "/".join((DRIVE_ROOT_NAME, *DRIVE_DESTINATION_PATH))
    print(f"Remote target: My Drive/{remote_path}/")
    print(f"Files found  : {len(local_files)}")
    print(f"Total size   : {total_bytes / (1024 ** 3):.3f} GiB")
    print(f"Dry run      : {DRY_RUN}")
    print("Deletes      : disabled")
    print("Downloads    : disabled")

    if DRY_RUN:
        for path in local_files:
            print(f"[PLAN] {path.relative_to(root).as_posix()} ({path.stat().st_size:,} bytes)")
        print("Dry run complete; nothing was uploaded.")
        return

    drive = authenticate()

    # Create/find the full nested destination, preserving the local run tree.
    drive_destination = ensure_folder(drive, DRIVE_ROOT_NAME, "root")
    for folder_name in DRIVE_DESTINATION_PATH:
        drive_destination = ensure_folder(
            drive, folder_name, drive_destination["id"]
        )

    counters = {"new": 0, "updated": 0, "unchanged": 0, "skipped": 0}
    skipped_details: list[tuple[str, str]] = []
    bytes_uploaded = 0

    # Cache remote children per local directory to avoid repeatedly listing
    # the same Drive directory for each file.
    folder_cache: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    folder_cache[""] = index_children(list_children(drive, drive_destination["id"]))

    def ensure_relative_directory(relative_dir: Path) -> str:
        """Ensure each component of a local relative directory exists on Drive."""
        current_parent_id = drive_destination["id"]
        parts = relative_dir.parts if str(relative_dir) not in ("", ".") else ()
        accumulated: list[str] = []

        for part in parts:
            accumulated.append(part)
            rel_key = Path(*accumulated).as_posix()
            if rel_key in folder_cache:
                # Folder is already represented by the cache entry; resolve its
                # ID from its parent cache without another list request.
                parent_key = Path(*accumulated[:-1]).as_posix() if len(accumulated) > 1 else ""
                parent_folders, _ = folder_cache[parent_key]
                current_parent_id = parent_folders[part]["id"]
                continue

            parent_key = Path(*accumulated[:-1]).as_posix() if len(accumulated) > 1 else ""
            parent_folders, parent_files = folder_cache[parent_key]
            if part in parent_files:
                raise RuntimeError(
                    f"A file named {part!r} blocks creation of a required folder "
                    f"at {rel_key!r} on Google Drive."
                )

            remote_folder = parent_folders.get(part)
            if remote_folder is None:
                print(f"[Drive] Creating directory: {rel_key}/")
                remote_folder = create_folder(drive, part, current_parent_id)
                # Keep the parent cache current.
                parent_folders[part] = remote_folder

            current_parent_id = remote_folder["id"]
            folder_cache[rel_key] = index_children(list_children(drive, current_parent_id))

        return current_parent_id

    for index, local_path in enumerate(local_files, start=1):
        relative_path = local_path.relative_to(root)
        relative_parent = relative_path.parent
        relative_name = relative_path.as_posix()
        parent_id = ensure_relative_directory(relative_parent)

        parent_key = relative_parent.as_posix() if str(relative_parent) != "." else ""
        _, remote_files = folder_cache[parent_key]
        matching_remote_files = remote_files.get(local_path.name, [])

        print(f"\n[{index}/{len(local_files)}] {relative_name}")
        try:
            with tempfile.TemporaryDirectory(prefix="gdrive_sync_") as temp_name:
                temp_dir = Path(temp_name)
                snapshot = stable_snapshot(local_path, temp_dir)
                digest = md5_file(snapshot)

                # If Drive already contains same-name duplicates, update every
                # existing copy rather than failing or creating yet another one.
                if matching_remote_files:
                    per_copy_results = []
                    for duplicate_index, remote_file in enumerate(matching_remote_files, start=1):
                        result = upload_or_update(
                            drive=drive,
                            snapshot=snapshot,
                            local_name=local_path.name,
                            remote_file=remote_file,
                            parent_id=parent_id,
                            expected_md5=digest,
                        )
                        counters[result] += 1
                        per_copy_results.append(result)
                        if result in ("new", "updated"):
                            bytes_uploaded += snapshot.stat().st_size

                    updated_count = per_copy_results.count("updated")
                    unchanged_count = per_copy_results.count("unchanged")
                    if updated_count:
                        print(
                            f"[OK] updated {updated_count}/{len(matching_remote_files)} "
                            f"existing Drive copies in-place: {relative_name}"
                        )
                    else:
                        print(
                            f"[SKIP] all {unchanged_count} existing Drive copies "
                            f"already match: {relative_name}"
                        )
                else:
                    result = upload_or_update(
                        drive=drive,
                        snapshot=snapshot,
                        local_name=local_path.name,
                        remote_file=None,
                        parent_id=parent_id,
                        expected_md5=digest,
                    )
                    counters[result] += 1
                    if result in ("new", "updated"):
                        bytes_uploaded += snapshot.stat().st_size
                    print(f"[OK] {result}: {relative_name}")

                    # Re-list to capture the Drive ID returned/assigned by Drive
                    # and ensure later operations see a complete entry.
                    folder_cache[parent_key] = index_children(list_children(drive, parent_id))

        except Exception as exc:
            counters["skipped"] += 1
            skipped_details.append((relative_name, str(exc)))
            print(f"[SKIP/ERROR] {relative_name}: {exc}")

    print("\n" + "=" * 72)
    print("GOOGLE DRIVE DELTA SYNC SUMMARY")
    print("=" * 72)
    print(f"New files       : {counters['new']}")
    print(f"Updated files   : {counters['updated']}")
    print(f"Unchanged files : {counters['unchanged']}")
    print(f"Skipped/errors  : {counters['skipped']}")
    print(f"Uploaded bytes  : {bytes_uploaded / (1024 ** 2):.2f} MiB")
    print(f"Remote location : My Drive/{'/'.join((DRIVE_ROOT_NAME, *DRIVE_DESTINATION_PATH))}/")
    print("Duplicate policy: update every same-name file copy in-place")
    print("Remote deletion : none")
    print("Drive download  : none")

    if skipped_details:
        print("\nFiles requiring attention:")
        for name, reason in skipped_details:
            print(f"  - {name}: {reason}")
        raise RuntimeError(
            f"Sync finished with {len(skipped_details)} skipped/error file(s). "
            "Review the messages above and rerun after resolving them."
        )

    print("\nAll eligible files synchronized successfully.")


if __name__ == "__main__":
    main()
