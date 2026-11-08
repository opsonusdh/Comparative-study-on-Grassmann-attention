#!/usr/bin/env python3
"""Google Drive persistence for LoopLLM Kaggle runs.

The training code owns the local working tree.  This module owns persistence:

    Google Drive/loop_llm/  <->  /kaggle/working/loop_llm/

On startup the remote tree is restored locally. During training a single
background worker incrementally mirrors only files that were created, changed,
or deleted locally since the last successful synchronization.

Authentication is intentionally Kaggle-native:
- Add a Kaggle Secret named ``GDRIVE_CREDENTIALS_JSON`` whose value is the
  PyDrive2 ``credentials.json`` produced by the included local helper.
- The secret is never written into the source tree. A temporary local copy is
  created under ``/kaggle/working`` and removed when the process exits.

PyDrive2 is imported lazily so local CPU tests and normal non-Kaggle runs do not
need Google dependencies installed.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

DRIVE_FOLDER_NAME = "loop_llm"
DEFAULT_SECRET_NAME = "GDRIVE_CREDENTIALS_JSON"
DEFAULT_MAX_RETRIES = 3
ATOMIC_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".npy",
    ".npz",
}
SNAPSHOT_COPY_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".log",
    ".txt",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
}


@dataclass
class SyncResult:
    uploaded: int = 0
    created: int = 0
    deleted: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_uploaded: int = 0


class DriveSyncError(RuntimeError):
    """Raised when Google Drive persistence cannot be established safely."""


class KaggleDriveSync:
    """Synchronize a local LoopLLM working tree with a Drive folder.

    ``restore()`` is synchronous and must complete before a resumed checkpoint
    is loaded. ``start()`` launches one worker; ``request_sync()`` is cheap and
    coalesces overlapping requests. ``close(final_sync=True)`` performs one last
    sync and waits for the worker to exit.
    """

    def __init__(
        self,
        local_root: Path,
        *,
        remote_folder_name: str = DRIVE_FOLDER_NAME,
        secret_name: str = DEFAULT_SECRET_NAME,
        manifest_path: Optional[Path] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.local_root = Path(local_root).resolve()
        self.remote_folder_name = remote_folder_name
        self.secret_name = secret_name
        self.manifest_path = Path(manifest_path or (self.local_root.parent / ".loop_llm_drive_manifest.json"))
        self.max_retries = max(1, int(max_retries))

        self.local_root.mkdir(parents=True, exist_ok=True)
        self.manifest: Dict[str, Dict[str, Any]] = self._load_manifest()
        self._drive = None
        self._remote_root_id: Optional[str] = None
        self._remote_index: Dict[str, Dict[str, Any]] = {}

        self._event = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._requested_generation = 0
        self._completed_generation = 0
        self._last_error: Optional[BaseException] = None
        self._closed = False
        self._stage_root = self.local_root.parent / ".loop_llm_drive_stage"
        self._stage_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Authentication / Drive discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _get_kaggle_secret(label: str) -> str:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore
        except ImportError as exc:
            raise DriveSyncError(
                "kaggle_secrets is unavailable. This Drive backend is intended "
                "to run inside a Kaggle Notebook."
            ) from exc

        try:
            value = UserSecretsClient().get_secret(label)
        except Exception as exc:
            raise DriveSyncError(
                f"Could not read Kaggle Secret '{label}'. Add and attach that "
                "secret in Add-ons -> Secrets."
            ) from exc
        if not value or not value.strip():
            raise DriveSyncError(f"Kaggle Secret '{label}' is empty.")
        return value

    def _authenticate(self) -> None:
        if self._drive is not None:
            return

        print("[DriveSync] Authenticating with the configured Kaggle Secret...", flush=True)
        try:
            from pydrive2.auth import GoogleAuth  # type: ignore
            from pydrive2.drive import GoogleDrive  # type: ignore
        except ImportError as exc:
            raise DriveSyncError(
                "PyDrive2 is not installed. In the Kaggle notebook run "
                "`%pip install -q PyDrive2` once, then restart the session."
            ) from exc

        credentials_json = self._get_kaggle_secret(self.secret_name)
        credentials_path = self.local_root.parent / ".loop_llm_gdrive_credentials.json"
        credentials_path.write_text(credentials_json, encoding="utf-8")
        try:
            settings = {
                "save_credentials": False,
                "get_refresh_token": True,
                "oauth_scope": ["https://www.googleapis.com/auth/drive"],
            }
            gauth = GoogleAuth(settings=settings)
            gauth.LoadCredentialsFile(str(credentials_path))
            if gauth.credentials is None:
                raise DriveSyncError(
                    "The stored Google Drive credential is not a PyDrive2 credential file. "
                    "Use scripts/create_drive_credentials.py to create one."
                )
            if gauth.access_token_expired:
                try:
                    gauth.Refresh()
                except Exception as exc:
                    raise DriveSyncError(
                        "The saved Google Drive refresh token is expired or invalid. "
                        "Regenerate credentials with scripts/create_drive_credentials.py."
                    ) from exc
            else:
                gauth.Authorize()
            self._drive = GoogleDrive(gauth)
            print("[DriveSync] Google Drive authentication initialized.", flush=True)
        finally:
            try:
                credentials_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _find_or_create_root(self) -> str:
        self._authenticate()
        if self._remote_root_id:
            return self._remote_root_id

        q = (
            f"title = '{self.remote_folder_name.replace(chr(39), chr(92) + chr(39))}' "
            "and mimeType = 'application/vnd.google-apps.folder' "
            "and trashed = false and 'root' in parents"
        )
        matches = self._drive.ListFile({"q": q, "maxResults": 100}).GetList()
        if len(matches) > 1:
            matches = sorted(matches, key=lambda item: item.get("modifiedDate", ""), reverse=True)
            raise DriveSyncError(
                f"Multiple My Drive root folders named '{self.remote_folder_name}' were found. "
                "Rename the duplicates so exactly one remains."
            )
        if matches:
            self._remote_root_id = str(matches[0]["id"])
            return self._remote_root_id

        folder = self._drive.CreateFile(
            {
                "title": self.remote_folder_name,
                "mimeType": "application/vnd.google-apps.folder",
                "parents": [{"id": "root"}],
            }
        )
        folder.Upload()
        self._remote_root_id = str(folder["id"])
        return self._remote_root_id

    # ------------------------------------------------------------------
    # Local/remote indexing
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        if not self.manifest_path.exists():
            return {}
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)}
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.manifest_path)

    @staticmethod
    def _relative(root: Path, path: Path) -> str:
        return path.relative_to(root).as_posix()

    def _scan_local(self) -> Dict[str, Dict[str, Any]]:
        current: Dict[str, Dict[str, Any]] = {}
        if not self.local_root.exists():
            return current
        for path in self.local_root.rglob("*"):
            if not path.is_file():
                continue
            rel = self._relative(self.local_root, path)
            try:
                st = path.stat()
            except FileNotFoundError:
                continue
            current[rel] = {
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
                "inode": int(getattr(st, "st_ino", 0)),
            }
        return current

    def _list_children(self, parent_id: str) -> list[dict[str, Any]]:
        items = self._drive.ListFile(
            {"q": f"'{parent_id}' in parents and trashed = false", "maxResults": 1000}
        ).GetList()
        return [dict(item) for item in items]

    def _index_remote_tree(self, root_id: str) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        stack: list[tuple[str, str]] = [(root_id, "")]
        folders_scanned = 0
        while stack:
            parent_id, prefix = stack.pop()
            folders_scanned += 1
            print(
                f"[DriveSync] Listing remote folder {folders_scanned} "
                f"({prefix or '<root>'}); indexed files so far: {len(index):,}.",
                flush=True,
            )
            for item in self._list_children(parent_id):
                name = str(item.get("title", ""))
                if not name or name in {".", ".."}:
                    continue
                rel = f"{prefix}/{name}" if prefix else name
                item_id = str(item.get("id", ""))
                if item.get("mimeType") == "application/vnd.google-apps.folder":
                    stack.append((item_id, rel))
                    continue
                index[rel] = {
                    "id": item_id,
                    "title": name,
                    "size": int(item.get("fileSize", 0) or 0),
                    "md5": item.get("md5Checksum"),
                    "mimeType": item.get("mimeType"),
                }
        return index

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self) -> None:
        """Restore the complete remote tree before training/resume starts.

        Use a process-unique staging directory so separate invocations cannot
        collide on one fixed ``.restore_tmp`` path. Swap the old tree aside only
        after all downloads succeed, and roll it back if the final rename fails.
        The trainer additionally ensures only rank 0 calls restore under torchrun.
        """
        with self._lock:
            print("[DriveSync] Resolving remote folder and listing its files...", flush=True)
            root_id = self._find_or_create_root()
            self._remote_index = self._index_remote_tree(root_id)
            print(
                f"[DriveSync] Remote listing complete: {len(self._remote_index):,} files to restore.",
                flush=True,
            )
            unique_suffix = f"{os.getpid()}-{threading.get_ident()}"
            tmp_root = self.local_root.parent / f".{self.local_root.name}.restore_tmp-{unique_suffix}"
            backup_root = self.local_root.parent / f".{self.local_root.name}.restore_backup-{unique_suffix}"
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
            if backup_root.exists():
                shutil.rmtree(backup_root)
            tmp_root.mkdir(parents=True, exist_ok=True)
            try:
                total_files = len(self._remote_index)
                for file_index, (rel, item) in enumerate(sorted(self._remote_index.items()), start=1):
                    if file_index == 1 or file_index % 10 == 0 or file_index == total_files:
                        print(
                            f"[DriveSync] Downloading file {file_index}/{total_files}: {rel}",
                            flush=True,
                        )
                    mime = str(item.get("mimeType") or "")
                    if mime.startswith("application/vnd.google-apps."):
                        raise DriveSyncError(
                            f"Drive item '{rel}' is a native Google document and cannot be restored "
                            "as a regular file."
                        )
                    destination = tmp_root / Path(rel)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    file_obj = self._drive.CreateFile({"id": item["id"]})
                    self._download_with_retries(file_obj, destination, rel)

                moved_old_tree = False
                if self.local_root.exists():
                    self.local_root.replace(backup_root)
                    moved_old_tree = True
                try:
                    tmp_root.replace(self.local_root)
                except Exception:
                    if moved_old_tree and backup_root.exists() and not self.local_root.exists():
                        backup_root.replace(self.local_root)
                    raise
                if moved_old_tree:
                    shutil.rmtree(backup_root, ignore_errors=True)

                self.manifest = {}
                local = self._scan_local()
                for rel, meta in local.items():
                    remote = self._remote_index.get(rel, {})
                    self.manifest[rel] = {
                        "drive_id": remote.get("id"),
                        **meta,
                        "drive_md5": remote.get("md5"),
                    }
                self._save_manifest()
                print("[DriveSync] Restore finished; local tree and manifest are ready.", flush=True)
            except Exception:
                shutil.rmtree(tmp_root, ignore_errors=True)
                raise

    def _download_with_retries(self, file_obj: Any, destination: Path, rel: str) -> None:
        last: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                file_obj.GetContentFile(str(destination))
                return
            except Exception as exc:
                last = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise DriveSyncError(f"Failed to restore '{rel}' after {self.max_retries} attempts: {last}") from last

    # ------------------------------------------------------------------
    # Upload / deletion
    # ------------------------------------------------------------------

    def _ensure_remote_folder(self, path_parts: Iterable[str], root_id: str) -> str:
        parent_id = root_id
        prefix = ""
        for name in path_parts:
            prefix = f"{prefix}/{name}" if prefix else name
            cached = self._remote_index.get(prefix)
            if cached and cached.get("mimeType") == "application/vnd.google-apps.folder":
                parent_id = str(cached["id"])
                continue

            q = (
                f"title = '{name.replace(chr(39), chr(92) + chr(39))}' "
                "and mimeType = 'application/vnd.google-apps.folder' "
                f"and trashed = false and '{parent_id}' in parents"
            )
            matches = self._drive.ListFile({"q": q, "maxResults": 20}).GetList()
            if len(matches) > 1:
                raise DriveSyncError(f"Multiple Drive folders exist at '{prefix}'.")
            if matches:
                folder_id = str(matches[0]["id"])
            else:
                folder = self._drive.CreateFile(
                    {
                        "title": name,
                        "mimeType": "application/vnd.google-apps.folder",
                        "parents": [{"id": parent_id}],
                    }
                )
                folder.Upload()
                folder_id = str(folder["id"])
            self._remote_index[prefix] = {
                "id": folder_id,
                "title": name,
                "mimeType": "application/vnd.google-apps.folder",
            }
            parent_id = folder_id
        return parent_id

    def _stage_file(self, source: Path, rel: str) -> Path:
        staged = self._stage_root / rel
        staged.parent.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower()
        if suffix in ATOMIC_EXTENSIONS:
            try:
                staged.unlink(missing_ok=True)
                os.link(source, staged)
                return staged
            except OSError:
                pass
        shutil.copy2(source, staged)
        return staged

    def _upload_file(self, source: Path, rel: str, drive_id: Optional[str]) -> None:
        root_id = self._find_or_create_root()
        parent_parts = Path(rel).parts[:-1]
        parent_id = self._ensure_remote_folder(parent_parts, root_id)
        staged = self._stage_file(source, rel)
        staged_stat = staged.stat()
        try:
            file_obj = self._drive.CreateFile({"id": drive_id}) if drive_id else self._drive.CreateFile({})
            if not drive_id:
                file_obj["title"] = Path(rel).name
                file_obj["parents"] = [{"id": parent_id}]
            file_obj.SetContentFile(str(staged))
            file_obj.Upload()
            drive_id = str(file_obj["id"])
            self.manifest[rel] = {
                "drive_id": drive_id,
                "size": int(staged_stat.st_size),
                "mtime_ns": int(staged_stat.st_mtime_ns),
                "drive_md5": file_obj.get("md5Checksum"),
            }
            self._remote_index[rel] = {
                "id": drive_id,
                "title": Path(rel).name,
                "size": int(staged_stat.st_size),
                "md5": file_obj.get("md5Checksum"),
                "mimeType": file_obj.get("mimeType"),
            }
        finally:
            try:
                staged.unlink(missing_ok=True)
            except OSError:
                pass

    def _trash_remote(self, drive_id: str) -> None:
        file_obj = self._drive.CreateFile({"id": drive_id})
        file_obj.Trash()

    def _should_upload(self, rel: str, meta: Dict[str, Any]) -> bool:
        old = self.manifest.get(rel)
        if old is None:
            return True
        return (
            int(old.get("size", -1)) != int(meta["size"])
            or int(old.get("mtime_ns", -1)) != int(meta["mtime_ns"])
            or int(old.get("inode", -1)) != int(meta.get("inode", -1))
        )

    def sync_once(self) -> SyncResult:
        with self._lock:
            self._authenticate()
            root_id = self._find_or_create_root()
            self._remote_index = self._index_remote_tree(root_id)
            local = self._scan_local()
            result = SyncResult()

            # Upload/create local files. Each successful file is committed to
            # the manifest immediately, so a later failure does not cause the
            # successful upload to be redundantly retried.
            for rel, meta in sorted(local.items()):
                if not self._should_upload(rel, meta):
                    result.skipped += 1
                    continue
                remote = self._remote_index.get(rel, {})
                drive_id = remote.get("id") or self.manifest.get(rel, {}).get("drive_id")
                last: Optional[BaseException] = None
                for attempt in range(1, self.max_retries + 1):
                    try:
                        self._upload_file(self.local_root / Path(rel), rel, str(drive_id) if drive_id else None)
                        if remote:
                            result.uploaded += 1
                        else:
                            result.created += 1
                        result.bytes_uploaded += int(meta["size"])
                        self._save_manifest()
                        last = None
                        break
                    except Exception as exc:
                        last = exc
                        # If the stale Drive id is gone, retry once as a create.
                        if drive_id:
                            drive_id = None
                        if attempt < self.max_retries:
                            time.sleep(2 ** (attempt - 1))
                if last is not None:
                    result.failed += 1
                    print(f"[DriveSync] FAILED upload {rel}: {last}", flush=True)

            # Deletions are propagated only for paths known to have existed in
            # the prior synchronized manifest. A fresh restore establishes the
            # complete baseline before this code ever considers deletion.
            local_set = set(local)
            for rel, old in sorted(list(self.manifest.items())):
                if rel in local_set:
                    continue
                drive_id = old.get("drive_id")
                if not drive_id:
                    self.manifest.pop(rel, None)
                    continue
                try:
                    self._trash_remote(str(drive_id))
                    self.manifest.pop(rel, None)
                    self._remote_index.pop(rel, None)
                    result.deleted += 1
                    self._save_manifest()
                except Exception as exc:
                    result.failed += 1
                    print(f"[DriveSync] FAILED delete {rel}: {exc}", flush=True)

            # Final manifest write also captures unchanged/newly observed local
            # metadata so the next sync has a clean baseline.
            for rel, meta in local.items():
                if rel in self.manifest:
                    self.manifest[rel].update(meta)
            self._save_manifest()
            return result

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Drive sync worker is already closed.")
            if self._worker and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(target=self._worker_main, name="loopllm-drive-sync", daemon=True)
            self._worker.start()

    def request_sync(self) -> int:
        with self._lock:
            if self._closed:
                return self._completed_generation
            self._requested_generation += 1
            generation = self._requested_generation
            self._event.set()
            return generation

    def sync_now(self, timeout: Optional[float] = None) -> bool:
        generation = self.request_sync()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            worker = self._worker
        if worker is None:
            try:
                self.sync_once()
                with self._lock:
                    self._completed_generation = max(self._completed_generation, generation)
                    self._last_error = None
                return True
            except Exception as exc:
                with self._lock:
                    self._last_error = exc
                    self._completed_generation = max(self._completed_generation, generation)
                print(f"[DriveSync] synchronization failed: {exc}", flush=True)
                return False

        while True:
            with self._lock:
                if self._completed_generation >= generation:
                    return self._last_error is None
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.25)

    def _worker_main(self) -> None:
        while True:
            self._event.wait()
            self._event.clear()
            while True:
                with self._lock:
                    target = self._requested_generation
                    stopping = self._stop.is_set()
                    if target <= self._completed_generation:
                        if stopping:
                            return
                        break
                try:
                    result = self.sync_once()
                    with self._lock:
                        self._completed_generation = target
                        self._last_error = None if result.failed == 0 else DriveSyncError(
                            f"{result.failed} file operation(s) failed"
                        )
                    print(
                        f"[DriveSync] sync complete: uploaded={result.uploaded}, "
                        f"created={result.created}, deleted={result.deleted}, "
                        f"skipped={result.skipped}, failed={result.failed}, "
                        f"bytes={result.bytes_uploaded:,}",
                        flush=True,
                    )
                except Exception as exc:
                    with self._lock:
                        self._completed_generation = target
                        self._last_error = exc
                    print(f"[DriveSync] sync failed: {exc}", flush=True)

                with self._lock:
                    newer = self._requested_generation > self._completed_generation
                    stopping = self._stop.is_set()
                if newer:
                    continue
                if stopping:
                    return
                break

    def close(self, *, final_sync: bool = True, timeout: Optional[float] = 600.0) -> None:
        with self._lock:
            if self._closed:
                return

        # Do the final synchronization before marking the worker closed;
        # request_sync() intentionally ignores requests after _closed=True.
        if final_sync:
            self.sync_now(timeout=timeout)

        with self._lock:
            self._closed = True
            self._stop.set()
            self._event.set()
            worker = self._worker

        if worker and worker.is_alive():
            remaining = None if timeout is None else max(0.0, timeout)
            worker.join(timeout=remaining)
        shutil.rmtree(self._stage_root, ignore_errors=True)
