import importlib
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


class DummyTask:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_file_path(self):
        return self.file_path


@pytest.mark.unit
def test_file_ops_manage_task_files_and_archives(monkeypatch, tmp_path):
    file_ops_module = importlib.import_module("core.lib.common.file_ops")
    task = DummyTask("payload.bin")
    temp_file = tmp_path / "temp" / "payload.bin"
    final_file = tmp_path / "payload.bin"

    monkeypatch.setattr(
        file_ops_module.Context,
        "get_temporary_file_path",
        staticmethod(lambda file_name: str(tmp_path / "temp" / file_name)),
    )
    temp_file.parent.mkdir(parents=True, exist_ok=True)

    file_ops_module.FileOps.save_task_file_in_temp(task, b"temp-data")
    assert temp_file.read_bytes() == b"temp-data"
    file_ops_module.FileOps.remove_task_file_in_temp(task)
    assert not temp_file.exists()

    task.file_path = str(final_file)
    file_ops_module.FileOps.save_task_file(task, b"final-data")
    assert final_file.read_bytes() == b"final-data"
    file_ops_module.FileOps.remove_task_file(task)
    assert not final_file.exists()

    data_dir = tmp_path / "dataset"
    nested_file = data_dir / "nested" / "frame.txt"
    nested_file.parent.mkdir(parents=True, exist_ok=True)
    nested_file.write_text("frame-001", encoding="utf-8")

    archive_path = file_ops_module.FileOps.zip_directory(str(tmp_path), "dataset.zip", str(data_dir))
    assert Path(archive_path).exists()

    extract_dir = tmp_path / "unzipped"
    file_ops_module.FileOps.unzip_file(archive_path, str(extract_dir))
    assert (extract_dir / "nested" / "frame.txt").read_text(encoding="utf-8") == "frame-001"

    file_ops_module.FileOps.clear_directory(str(extract_dir))
    assert extract_dir.exists() and list(extract_dir.iterdir()) == []


@pytest.mark.unit
def test_file_cleaner_deletes_only_expired_matching_files(tmp_path):
    file_ops_module = importlib.import_module("core.lib.common.file_ops")
    old_tmp = tmp_path / "old.tmp"
    fresh_tmp = tmp_path / "fresh.tmp"
    note_file = tmp_path / "note.txt"

    for path in (old_tmp, fresh_tmp, note_file):
        path.write_text(path.name, encoding="utf-8")

    now_ts = datetime.now(tz=timezone.utc).timestamp()
    Path(old_tmp).touch()
    Path(fresh_tmp).touch()
    Path(note_file).touch()
    old_expired_mtime = now_ts - 20
    fresh_mtime = now_ts - 2
    for path, mtime in ((old_tmp, old_expired_mtime), (fresh_tmp, fresh_mtime), (note_file, old_expired_mtime)):
        Path(path).touch()
        import os

        os.utime(path, (mtime, mtime))

    cleaner = file_ops_module.FileCleaner(
        tmp_path,
        poll_seconds=0.01,
        ttl_seconds=5,
        pattern="*.tmp",
        max_delete_per_round=1,
    )
    cleaner._clean_once()

    assert not old_tmp.exists()
    assert fresh_tmp.exists()
    assert note_file.exists()


@pytest.mark.unit
def test_file_cleaner_supports_timestamp_conversion_and_thread_lifecycle(monkeypatch, tmp_path):
    file_ops_module = importlib.import_module("core.lib.common.file_ops")

    created_threads = []

    class DummyThread:
        def __init__(self, target=None, name=None, daemon=None):
            self.target = target
            self.name = name
            self.daemon = daemon
            self.started = False
            self.joined = False
            created_threads.append(self)

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started and not self.joined

        def join(self, timeout=None):
            self.joined = True

    monkeypatch.setattr(file_ops_module.threading, "Thread", DummyThread)

    cleaner = file_ops_module.FileCleaner(
        tmp_path,
        poll_seconds=0.01,
        ttl_seconds=None,
        recursive=True,
        expiry_resolver=lambda path: datetime.now(tz=timezone.utc) - timedelta(seconds=1)
        if path.name == "expired.log"
        else None,
    )

    cleaner.start()
    cleaner.start()
    cleaner.stop()

    assert len(created_threads) == 1
    assert created_threads[0].name == "FolderCleaner"
    assert created_threads[0].daemon is True
    assert created_threads[0].started is True
    assert created_threads[0].joined is True

    aware_ts = file_ops_module.FileCleaner._to_timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc))
    naive_ts = file_ops_module.FileCleaner._to_timestamp(datetime(2024, 1, 1))
    numeric_ts = file_ops_module.FileCleaner._to_timestamp(12.5)

    assert aware_ts == naive_ts
    assert numeric_ts == 12.5
    with pytest.raises(TypeError):
        file_ops_module.FileCleaner._to_timestamp("invalid")


@pytest.mark.unit
def test_file_ops_cover_directory_guards_temp_dirs_and_invalid_archives(monkeypatch, tmp_path):
    file_ops_module = importlib.import_module("core.lib.common.file_ops")

    existing_file = tmp_path / "existing.txt"
    existing_file.write_text("payload", encoding="utf-8")

    with pytest.raises(AssertionError, match="is a FILE"):
        file_ops_module.FileOps.create_directory(str(existing_file))

    removable_dir = tmp_path / "removable"
    removable_dir.mkdir()
    (removable_dir / "child.txt").write_text("child", encoding="utf-8")
    file_ops_module.FileOps.remove_file(str(removable_dir))
    assert not removable_dir.exists()

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    created_temp_dir = file_ops_module.FileOps.create_temp_directory("runtime-cache")
    assert Path(created_temp_dir).is_dir()

    file_ops_module.FileOps.remove_file(str(tmp_path / "missing.bin"))

    with pytest.raises(ValueError, match="Invalid zip file path"):
        file_ops_module.FileOps.unzip_file(str(tmp_path / "missing.txt"), str(tmp_path / "out"))


@pytest.mark.unit
def test_file_cleaner_logs_missing_and_non_directory_targets(monkeypatch, tmp_path):
    file_ops_module = importlib.import_module("core.lib.common.file_ops")
    warnings = []

    monkeypatch.setattr(file_ops_module.LOGGER, "warning", lambda message: warnings.append(message))

    missing_cleaner = file_ops_module.FileCleaner(tmp_path / "missing-folder", ttl_seconds=1)
    missing_cleaner._clean_once()

    plain_file = tmp_path / "plain.txt"
    plain_file.write_text("payload", encoding="utf-8")
    file_cleaner = file_ops_module.FileCleaner(plain_file, ttl_seconds=1)
    file_cleaner._clean_once()

    assert any("Folder not found" in message for message in warnings)
    assert any("Not a directory" in message for message in warnings)
