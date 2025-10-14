import os
from datetime import datetime
from contextlib import contextmanager

try:
    import fcntl  # POSIX file locking
except Exception:  # pragma: no cover
    fcntl = None

from .time_estimation import Timer
from core.lib.common import FileOps, Context


class OverheadEstimator:
    def __init__(self, method_name, save_dir, agent_id=0):

        self.method_name = method_name
        self.timer = Timer(f'Runtime Overhead of {method_name}')
        self.overhead_file = Context.get_file_path(os.path.join(save_dir, f'{method_name}_Overhead.txt'))
        self.latest_overhead = 0
        self.agent_id = agent_id
        # ensure directory and header exist; do NOT truncate existing logs implicitly
        self._ensure_file_initialized()

    @contextmanager
    def _lock_file(self, file_obj):
        """Cross-process exclusive lock (best-effort)."""
        if fcntl is None:
            # On non-POSIX platforms we skip locking; on Linux/macOS this path won't be taken.
            yield
            return
        fd = file_obj.fileno()
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)

    @contextmanager
    def _lock_file_shared(self, file_obj):
        """Cross-process shared lock for readers (best-effort)."""
        if fcntl is None:
            yield
            return
        fd = file_obj.fileno()
        fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)

    def __enter__(self):
        self.timer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.__exit__(exc_type, exc_val, exc_tb)
        self.latest_overhead = self.timer.get_elapsed_time()
        self.write_overhead(self.latest_overhead)

    def get_latest_overhead(self):
        return self.latest_overhead

    def get_average_overhead(self):
        """
        Compute the average overhead from the log file.
        Compatible with both the new CSV format and legacy plain-number format.
        Returns 0.0 when no valid records exist.
        """
        if not os.path.exists(self.overhead_file):
            return 0.0
        durations = []
        with open(self.overhead_file, 'r') as f:
            with self._lock_file_shared(f):
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # skip comments and header line
                    if line.startswith('#') or line.lower().startswith('agent_id'):
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    # CSV format: agent_id,timestamp,start_time,end_time,duration_seconds
                    if len(parts) >= 5:
                        try:
                            durations.append(float(parts[-1]))
                        except ValueError:
                            continue
                    else:
                        # legacy format: a single float per line
                        try:
                            durations.append(float(line))
                        except ValueError:
                            continue
        return sum(durations) / len(durations) if durations else 0.0

    def write_overhead(self, overhead):
        """
        Append a record with human-readable timestamps and duration in seconds.
        Safe for concurrent writers on POSIX platforms.
        """
        # ensure directory and header exist
        self._ensure_file_initialized()
        # prefer timer times if available
        start_ts = getattr(self.timer, 'start_time', None)
        end_ts = getattr(self.timer, 'end_time', None)
        # fall back to now if not available
        now = datetime.now()
        ts_str = self._format_dt(datetime.fromtimestamp(end_ts)) if end_ts else self._format_dt(now)
        start_str = self._format_dt(datetime.fromtimestamp(start_ts)) if start_ts else ''
        end_str = self._format_dt(datetime.fromtimestamp(end_ts)) if end_ts else ''
        line = f"{self.agent_id},{ts_str},{start_str},{end_str},{float(overhead):.6f}\n"
        # open and lock for append
        with open(self.overhead_file, 'a') as f:
            with self._lock_file(f):
                f.write(line)
                f.flush()
                os.fsync(f.fileno())

    def clear(self):
        """
        Reset the overhead log by recreating the file with a descriptive header.
        Use this explicitly if you want to start a fresh log. Thread/process-safe.
        """
        self.latest_overhead = 0
        # ensure directory exists
        dir_path = os.path.dirname(self.overhead_file)
        if dir_path:
            FileOps.create_directory(dir_path)
        # write header and column names with exclusive lock
        # open with write mode to truncate
        with open(self.overhead_file, 'w') as f:
            with self._lock_file(f):
                created = self._format_dt(datetime.now())
                f.write(f"# Overhead Log for {self.method_name}\n")
                f.write(f"# Created: {created}\n")
                f.write("# Columns: agent_id,timestamp,start_time,end_time,duration_seconds\n")
                # CSV header line for easy parsing
                f.write("agent_id,timestamp,start_time,end_time,duration_seconds\n")
                f.flush()
                os.fsync(f.fileno())

    def _ensure_file_initialized(self):
        dir_path = os.path.dirname(self.overhead_file)
        if dir_path:
            FileOps.create_directory(dir_path)
        # Create file if missing, or ensure header exists if empty, under lock
        if not os.path.exists(self.overhead_file):
            # create and write header
            with open(self.overhead_file, 'w') as f:
                with self._lock_file(f):
                    created = self._format_dt(datetime.now())
                    f.write(f"# Overhead Log for {self.method_name}\n")
                    f.write(f"# Created: {created}\n")
                    f.write("# Columns: agent_id,timestamp,start_time,end_time,duration_seconds\n")
                    f.write("agent_id,timestamp,start_time,end_time,duration_seconds\n")
                    f.flush()
                    os.fsync(f.fileno())
        else:
            # If the file exists but is empty, write header
            if os.path.getsize(self.overhead_file) == 0:
                with open(self.overhead_file, 'r+') as f:
                    with self._lock_file(f):
                        # Double-check size under lock
                        f.seek(0, os.SEEK_END)
                        if f.tell() == 0:
                            f.seek(0)
                            created = self._format_dt(datetime.now())
                            f.write(f"# Overhead Log for {self.method_name}\n")
                            f.write(f"# Created: {created}\n")
                            f.write("# Columns: agent_id,timestamp,start_time,end_time,duration_seconds\n")
                            f.write("agent_id,timestamp,start_time,end_time,duration_seconds\n")
                            f.flush()
                            os.fsync(f.fileno())

    @staticmethod
    def _format_dt(dt: datetime) -> str:
        return dt.strftime('%Y-%m-%d %H:%M:%S.%f')
