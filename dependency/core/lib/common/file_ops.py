import os
import shutil
import tempfile
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable, Optional, Union

from .context import Context
from .log import LOGGER


class FileOps:
    @staticmethod
    def save_task_file_in_temp(task, file_data):
        file_path = Context.get_temporary_file_path(task.get_file_path())
        with open(file_path, 'wb') as buffer:
            buffer.write(file_data)

    @staticmethod
    def remove_task_file_in_temp(task):
        file_path = Context.get_temporary_file_path(task.get_file_path())
        FileOps.remove_file(file_path)

    @staticmethod
    def clear_temp_directory():
        temp_dir = Context.get_temporary_file_path('')
        FileOps.clear_directory(temp_dir)

    @staticmethod
    def save_task_file(task, file_data):
        file_path = task.get_file_path()
        with open(file_path, 'wb') as buffer:
            buffer.write(file_data)

    @staticmethod
    def remove_task_file(task):
        file_path = task.get_file_path()
        FileOps.remove_file(file_path)

    @staticmethod
    def remove_file(file_path):
        if not file_path or not os.path.exists(file_path):
            return

        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)

    @staticmethod
    def create_temp_directory(prefix):
        tmp_dir = os.path.join(tempfile.gettempdir(), prefix)
        FileOps.create_directory(tmp_dir)
        return tmp_dir

    @staticmethod
    def create_directory(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            assert os.path.isdir(dir_path), f'Path "{dir_path}" is a FILE'

    @staticmethod
    def clear_directory(dir_path):
        if dir_path and os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

    @staticmethod
    def zip_directory(dir_path, zip_name, data_dir):
        """
        Create a .zip archive from the contents of `data_dir`, placing the archive in `dir_path`
        and naming it `zip_name`.

        Args:
            dir_path: Directory where the resulting .zip file will be stored.
            zip_name: Name of the .zip file to create (with or without the .zip extension).
            data_dir: Source directory whose contents will be archived.

        Returns:
            Absolute path to the created .zip file.
        """
        dir_path = os.path.abspath(dir_path)
        zip_path = os.path.join(dir_path, zip_name)
        base_name, _ = os.path.splitext(zip_path)
        shutil.make_archive(base_name, 'zip', root_dir=data_dir)
        return zip_path

    @staticmethod
    def unzip_file(zip_path, extract_dir):
        """
        Extracts a .zip file to the specified directory.

        Args:
            zip_path: Path to the .zip file to extract.
            extract_dir: Directory where the contents will be extracted.

        Returns:
            None
        """
        if not os.path.exists(zip_path) or not zip_path.lower().endswith('.zip'):
            raise ValueError(f'Invalid zip file path: {zip_path}')

        FileOps.create_directory(extract_dir)
        shutil.unpack_archive(zip_path, extract_dir, 'zip')


class FileCleaner:
    """
    Polling for cleaning expired files in the background thread.
    - The thread is a daemon, which does not block the main process from exiting.
    - All exceptions are caught to prevent the thread from exiting abnormally.
    """
    Expiry = Union[datetime, int, float]

    def __init__(
            self,
            folder: Union[str, Path],
            poll_seconds: float = 30.0,
            ttl_seconds: Optional[float] = 3600.0,
            recursive: bool = False,
            pattern: Optional[str] = None,  # e.g. "*.tmp"
            expiry_resolver: Optional[Callable[[Path], Optional[Expiry]]] = None,
            max_delete_per_round: int = 200,
    ):
        """
        folder: Target folder
        poll_seconds: Poll interval
        ttl_seconds: Uniform TTL (seconds). If expiry_resolver is provided, it can be set to None
        recursive: Whether to recursively process subdirectories
        pattern: Optional glob filter (only process matching files)
        expiry_resolver: Expiry time resolution function for each file; return None to indicate "do not process/expire"
        max_delete_per_round: Maximum number of files to delete per round, to avoid excessive cleaning that may affect IO
        """
        self.folder = Path(folder)
        self.poll_seconds = float(poll_seconds)
        self.ttl_seconds = None if ttl_seconds is None else float(ttl_seconds)
        self.recursive = bool(recursive)
        self.pattern = pattern
        self.expiry_resolver = expiry_resolver
        self.max_delete_per_round = int(max_delete_per_round)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if self.ttl_seconds is None and self.expiry_resolver is None:
            raise ValueError("At least one of 'ttl_seconds' and 'expiration_resolver' is need to "
                             "provide to determine the expiration time")

    def start(self) -> None:
        """Start background cleaning thread (repeated calls will not start again)"""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="FolderCleaner", daemon=True)
        self._thread.start()
        LOGGER.info(f"[FileCleaner] started: folder={self.folder} poll={self.poll_seconds}")

    def stop(self, join: bool = True, timeout: Optional[float] = 5.0) -> None:
        """Stop thread"""
        self._stop_event.set()
        if join and self._thread:
            self._thread.join(timeout=timeout)
        LOGGER.info("[FileCleaner] stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._stop_event.wait(self.poll_seconds):
                    break
                self._clean_once()
            except Exception as e:
                LOGGER.warning("[FileCleaner] Unexpected error in cleaner loop, continue next round.")
                LOGGER.exception(e)
                time.sleep(min(1.0, self.poll_seconds))

    def _clean_once(self) -> None:
        folder = self.folder
        if not folder.exists():
            LOGGER.warning(f"[FileCleaner] Folder not found: {folder}")
            return
        if not folder.is_dir():
            LOGGER.warning(f"[FileCleaner] Not a directory: {folder}")
            return

        now_ts = time.time()
        deleted = 0
        scanned = 0

        paths_iter = self._iter_files(folder)

        for p in paths_iter:
            if self._stop_event.is_set():
                break
            scanned += 1

            # Lightweight control: Maximum of N deletions per round
            if deleted >= self.max_delete_per_round:
                break

            try:
                if not p.is_file():
                    continue
                if self.pattern and not p.match(self.pattern):
                    continue

                if self._is_expired(p, now_ts):
                    self._safe_delete(p)
                    deleted += 1

            except FileNotFoundError:
                continue
            except PermissionError:
                LOGGER.warning(f"[FileCleaner] Permission denied: {p}")
                continue
            except Exception as e:
                LOGGER.warning(f"[FileCleaner] Error processing file: {p}")
                LOGGER.exception(e)
                continue

    def _iter_files(self, folder: Path):
        if self.recursive:
            if self.pattern:
                yield from folder.rglob(self.pattern)
            else:
                yield from folder.rglob("*")
        else:
            yield from folder.iterdir()

    def _is_expired(self, path: Path, now_ts: float) -> bool:
        # Use custom expiry resolver first
        if self.expiry_resolver is not None:
            expiry = self.expiry_resolver(path)
            if expiry is None:
                return False
            exp_ts = self._to_timestamp(expiry)
            return now_ts >= exp_ts

        # Uniform TTL：mtime + ttl
        st = path.stat()
        exp_ts = st.st_mtime + (self.ttl_seconds or 0.0)
        return now_ts >= exp_ts

    @staticmethod
    def _to_timestamp(expiry: Expiry) -> float:
        if isinstance(expiry, (int, float)):
            return float(expiry)
        if isinstance(expiry, datetime):
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            return expiry.timestamp()
        raise TypeError(f"Unsupported expiry type: {type(expiry)}")

    def _safe_delete(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except Exception as e:
            LOGGER.warning(f"[FileCleaner] Failed to delete: {path}")
            LOGGER.exception(e)
