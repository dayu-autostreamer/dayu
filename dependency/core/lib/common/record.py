import os
import csv
import json
import time
from typing import Optional, Dict, Any, List


class Recorder:
    """
    Generic training recorder:
    - Holds a file handle
    - Writes a dict as one line to the file on each log()
    - Supports two formats: CSV (default) and JSONL

    Typical usage:
        rec = Recorder("logs/deployment_train.csv", fmt="csv")
        rec.log(step=1, epoch=0, avg_return=0.5, loss_pi=1.23)
        rec.log(step=2, epoch=0, avg_return=0.6, loss_pi=1.10)
        rec.close()

    Or with context manager:
        with Recorder("logs/train.jsonl", fmt="jsonl") as rec:
            rec.log(step=1, reward=0.5)
            rec.log(step=2, reward=0.8)
    """

    def __init__(
        self,
        filepath: str,
        fmt: str = "csv",               # "csv" or "jsonl"
        fieldnames: Optional[List[str]] = None,
        overwrite: bool = True,
        add_timestamp: bool = True,     # whether to automatically add a wall_time field
        flush_every: int = 1,           # flush every N writes
    ):
        """
        Args:
            filepath: Path to the log file
            fmt: "csv" or "jsonl"
            fieldnames: For CSV, the column names. If None, they will be inferred
                from the first logged row.
            overwrite: If True, overwrite the file; if False, append.
            add_timestamp: Whether to automatically add a wall_time field (UNIX timestamp).
            flush_every: Flush to disk every `flush_every` written rows (reduces frequent I/O).
        """
        self.filepath = filepath
        self.fmt = fmt.lower()
        assert self.fmt in ("csv", "jsonl"), f"Unsupported format: {fmt}"

        self.fieldnames = fieldnames[:] if fieldnames is not None else None
        self.add_timestamp = add_timestamp
        self.flush_every = max(1, int(flush_every))
        self._counter = 0  # used to control flushes

        # Ensure containing directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        mode = "w" if overwrite else "a"
        newline = "" if self.fmt == "csv" else None
        self._f = open(filepath, mode, newline=newline, encoding="utf-8")

        self._writer = None  # csv.DictWriter for CSV; not used for JSONL

        if self.fmt == "csv" and not overwrite and self.fieldnames is None:
            # In append mode, one might want to read the existing header from file.
            # For simplicity, require fieldnames to be provided when appending.
            raise ValueError(
                "When append (overwrite=False) and fmt='csv', you must provide fieldnames "
                "so that new rows match existing header."
            )

        # If CSV and fieldnames are already provided, write the header immediately
        if self.fmt == "csv" and self.fieldnames is not None:
            self._writer = csv.DictWriter(self._f, fieldnames=self.fieldnames)
            if overwrite:
                self._writer.writeheader()

    # Allow use as: with Recorder(...) as rec: rec.log(...)
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _ensure_csv_writer(self, row: Dict[str, Any]):
        """
        For CSV:
        - If fieldnames are not set: infer them from the row keys on the first log()
        - Create a DictWriter and write the header
        """
        if self._writer is not None:
            return

        if self.fieldnames is None:
            # Auto-infer columns: sort keys to make the logging order stable
            self.fieldnames = sorted(row.keys())

        self._writer = csv.DictWriter(self._f, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def log(self, **kwargs: Any):
        """
        Record one row. You can call directly:
            rec.log(step=10, loss=0.1, reward=0.5)
        Or pass a prepared dict (see log_dict).

        - Automatically adds wall_time if add_timestamp=True
        - For CSV: only fields listed in fieldnames will be written
        """
        self.log_dict(kwargs)

    def log_dict(self, data: Dict[str, Any]):
        """
        Record a single dict as one row.
        """
        if not isinstance(data, dict):
            raise TypeError(f"log_dict expects a dict, got {type(data)}")

        row = dict(data)  # shallow copy to avoid mutating the original dict

        if self.add_timestamp and "wall_time" not in row:
            row["wall_time"] = time.time()

        if self.fmt == "jsonl":
            # One JSON object per line
            self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            # CSV
            self._ensure_csv_writer(row)
            # Only write keys present in fieldnames; ignore extra keys
            filtered = {k: row.get(k, "") for k in self.fieldnames}
            self._writer.writerow(filtered)

        self._counter += 1
        if self._counter % self.flush_every == 0:
            self._f.flush()

    def flush(self):
        """Manually flush the underlying file."""
        if self._f and not self._f.closed:
            self._f.flush()

    def close(self):
        """Close the underlying file."""
        if self._f and not self._f.closed:
            self._f.flush()
            self._f.close()

    def __del__(self):
        # As a safety measure, attempt to close the file when garbage collected
        try:
            self.close()
        except Exception:
            pass
