from pathlib import Path

from drive_sync import KaggleDriveSync

FOLDER_MIME = "application/vnd.google-apps.folder"


class FakeFile(dict):
    def __init__(self, drive, *, file_id, title, parent_id=None, mime_type="text/plain", content=b""):
        super().__init__(id=file_id, title=title, mimeType=mime_type, parents=[{"id": parent_id}] if parent_id else [])
        self.drive = drive
        self.content = bytes(content)
        self.trashed = False

    def GetContentFile(self, destination):
        Path(destination).write_bytes(self.content)

    def SetContentFile(self, source):
        self._pending = Path(source).read_bytes()

    def Upload(self):
        if "_pending" in self.__dict__:
            self.content = self._pending
            del self._pending
        self.drive.nodes[self["id"]] = self
        self["fileSize"] = str(len(self.content))
        self["md5Checksum"] = "fake-md5"
        return self

    def Trash(self):
        self.trashed = True
        self.drive.nodes.pop(self["id"], None)


class FakeList:
    def __init__(self, items):
        self.items = items

    def GetList(self):
        return list(self.items)


class FakeDrive:
    def __init__(self):
        self.nodes = {}
        self.next_id = 100

    def ListFile(self, query):
        q = query.get("q", "")
        items = []
        for item in self.nodes.values():
            if item.trashed:
                continue
            if "'root' in parents" in q:
                if item.get("title") == "loop_llm" and item.get("mimeType") == FOLDER_MIME:
                    items.append(item)
                continue
            marker = "'"
            parent_id = None
            if "' in parents" in q:
                parent_id = q.split("'", 2)[1]
            if parent_id and item.get("parents") == [{"id": parent_id}]:
                items.append(item)
        return FakeList(items)

    def CreateFile(self, metadata=None):
        metadata = dict(metadata or {})
        if metadata.get("id"):
            return self.nodes[metadata["id"]]
        file_id = str(self.next_id)
        self.next_id += 1
        parent_id = (metadata.get("parents") or [{"id": None}])[0]["id"]
        item = FakeFile(
            self,
            file_id=file_id,
            title=metadata.get("title", ""),
            parent_id=parent_id,
            mime_type=metadata.get("mimeType", "text/plain"),
        )
        self.nodes[file_id] = item
        return item


def make_sync(tmp_path):
    drive = FakeDrive()
    root = FakeFile(drive, file_id="root-loop", title="loop_llm", mime_type=FOLDER_MIME, parent_id="root")
    drive.nodes[root["id"]] = root
    sync = KaggleDriveSync(tmp_path / "loop_llm")
    sync._drive = drive
    sync._remote_root_id = root["id"]
    sync._authenticate = lambda: None
    return sync, drive, root


def add_remote_file(drive, root, path_parts, content):
    parent = root["id"]
    for part in path_parts[:-1]:
        existing = next(
            (x for x in drive.nodes.values() if x.get("title") == part and x.get("parents") == [{"id": parent}] and x.get("mimeType") == FOLDER_MIME),
            None,
        )
        if existing is None:
            existing = FakeFile(drive, file_id=str(drive.next_id), title=part, parent_id=parent, mime_type=FOLDER_MIME)
            drive.next_id += 1
            drive.nodes[existing["id"]] = existing
        parent = existing["id"]
    item = FakeFile(drive, file_id=str(drive.next_id), title=path_parts[-1], parent_id=parent, content=content)
    drive.next_id += 1
    drive.nodes[item["id"]] = item
    return item


def test_restore_builds_complete_local_tree_and_baseline(tmp_path):
    sync, drive, root = make_sync(tmp_path)
    add_remote_file(drive, root, ("rfc_512x8", "latest.pt"), b"checkpoint")
    add_remote_file(drive, root, ("visualizations", "step_0001.png"), b"png")

    sync.restore()

    assert (sync.local_root / "rfc_512x8/latest.pt").read_bytes() == b"checkpoint"
    assert (sync.local_root / "visualizations/step_0001.png").read_bytes() == b"png"
    result = sync.sync_once()
    assert result.skipped == 2
    assert result.failed == 0


def test_sync_creates_changes_and_propagates_deletion(tmp_path):
    sync, drive, root = make_sync(tmp_path)
    original = add_remote_file(drive, root, ("rfc_512x8", "latest.pt"), b"old")
    add_remote_file(drive, root, ("old.log",), b"log")
    sync.restore()

    (sync.local_root / "rfc_512x8/latest.pt").write_bytes(b"new")
    (sync.local_root / "new.jsonl").write_text("row\n", encoding="utf-8")
    (sync.local_root / "old.log").unlink()

    result = sync.sync_once()
    assert result.uploaded == 1
    assert result.created == 1
    assert result.deleted == 1
    assert result.failed == 0
    assert drive.nodes[original["id"]].content == b"new"


def test_sync_generation_requests_coalesce_without_spawning_threads(tmp_path):
    sync, _, _ = make_sync(tmp_path)
    g1 = sync.request_sync()
    g2 = sync.request_sync()
    g3 = sync.request_sync()
    assert (g1, g2, g3) == (1, 2, 3)
    assert sync._requested_generation == 3
    sync.close(final_sync=False)


def test_background_worker_can_complete_final_sync_and_exit(tmp_path):
    sync, drive, root = make_sync(tmp_path)
    (sync.local_root / "checkpoint.json").parent.mkdir(parents=True, exist_ok=True)
    (sync.local_root / "checkpoint.json").write_text("{}", encoding="utf-8")
    sync.start()
    assert sync.sync_now(timeout=5.0)
    sync.close(final_sync=True, timeout=5.0)
    assert any(item.get("title") == "checkpoint.json" for item in drive.nodes.values())
    assert sync._worker is None or not sync._worker.is_alive()
