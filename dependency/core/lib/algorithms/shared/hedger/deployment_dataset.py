import glob
import os
import random
import threading
from typing import Dict, List, Optional

import torch


class DeploymentTransitionWriter:
    """Append deployment macro-transitions into small torch-save shards."""

    def __init__(self, root_dir: str, shard_size: int = 32, clear_existing: bool = False):
        self.root_dir = os.path.abspath(root_dir)
        self.shard_size = max(1, int(shard_size))
        self._buffer: List[dict] = []
        self._lock = threading.Lock()
        self._count = 0
        self._shard_idx = 0
        os.makedirs(self.root_dir, exist_ok=True)
        if clear_existing:
            for path in glob.glob(os.path.join(self.root_dir, "shard_*.pt")):
                os.remove(path)
        else:
            self._resume_counters()

    @property
    def count(self) -> int:
        with self._lock:
            return int(self._count)

    def append(self, transition: dict) -> int:
        with self._lock:
            self._buffer.append(_detach_transition(transition))
            self._count += 1
            count = int(self._count)
            if len(self._buffer) >= self.shard_size:
                self._flush_locked()
            return count

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        self.flush()

    def _resume_counters(self) -> None:
        files = sorted(glob.glob(os.path.join(self.root_dir, "shard_*.pt")))
        self._shard_idx = len(files)
        total = 0
        for path in files:
            try:
                payload = torch.load(path, map_location="cpu")
                total += len(payload.get("transitions", []))
            except Exception:
                continue
        self._count = total

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        path = os.path.join(self.root_dir, f"shard_{self._shard_idx:06d}.pt")
        payload = {
            "schema_version": 1,
            "count": len(self._buffer),
            "transitions": self._buffer,
        }
        torch.save(payload, path)
        self._buffer = []
        self._shard_idx += 1


class DeploymentTransitionDataset:
    """In-memory deployment macro-transition dataset for offline/replay RL."""

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.transitions: List[dict] = []
        self.reload()

    def __len__(self) -> int:
        return len(self.transitions)

    def reload(self) -> None:
        transitions: List[dict] = []
        for path in sorted(glob.glob(os.path.join(self.root_dir, "shard_*.pt"))):
            payload = torch.load(path, map_location="cpu")
            shard_transitions = payload.get("transitions", [])
            if isinstance(shard_transitions, list):
                transitions.extend(shard_transitions)
        self.transitions = transitions

    def sample(self, batch_size: int) -> List[dict]:
        batch_size = max(1, int(batch_size))
        if not self.transitions:
            return []
        if len(self.transitions) >= batch_size:
            return random.sample(self.transitions, batch_size)
        return [random.choice(self.transitions) for _ in range(batch_size)]


def _detach_transition(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _detach_transition(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_detach_transition(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach_transition(v) for v in value)
    return value
