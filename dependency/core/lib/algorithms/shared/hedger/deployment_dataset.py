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
        bad_bucket = [tr for tr in self.transitions if transition_quality_bucket(tr) == "bad"]
        normal_bucket = [tr for tr in self.transitions if transition_quality_bucket(tr) != "bad"]
        if bad_bucket:
            bad_ratio = len(bad_bucket) / max(1, len(self.transitions))
            bad_target_ratio = min(0.30, bad_ratio * 2.5)
            bad_target = max(1, int(round(batch_size * bad_target_ratio)))
        else:
            bad_target = 0
        if normal_bucket:
            bad_target = min(batch_size - 1, bad_target)
        else:
            bad_target = min(batch_size, bad_target)
        normal_target = batch_size - bad_target
        batch: List[dict] = []
        batch.extend(_sample_from_bucket(bad_bucket, bad_target))
        batch.extend(_sample_from_bucket(normal_bucket, normal_target))
        if len(batch) < batch_size:
            batch.extend(_sample_from_bucket(self.transitions, batch_size - len(batch)))
        random.shuffle(batch)
        return batch


def transition_quality_bucket(transition: dict) -> str:
    if not isinstance(transition, dict):
        return "unknown"
    quality_bucket = str(transition.get("collect_quality_bucket", "") or "").strip().lower()
    if quality_bucket in {"good", "mid", "bad"}:
        return quality_bucket
    return "bad" if _is_bad_transition(transition) else "unknown"


def summarize_transition_quality(transitions: List[dict]) -> Dict[str, float]:
    count = len(transitions)
    buckets = {"good": 0, "mid": 0, "bad": 0, "unknown": 0}
    rewards = []
    slo_values = []
    p95_values = []
    risk_values = []
    queue_values = []
    hotspot_values = []
    guard_count = 0
    for tr in transitions:
        bucket = transition_quality_bucket(tr)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        rewards.append(_as_float(tr.get("reward"), 0.0))
        metrics = tr.get("metrics") or {}
        slo_values.append(_as_float(metrics.get("e2e_slo_violation"), 0.0))
        p95_values.append(_as_float(metrics.get("e2e_latency_p95"), 0.0))
        risk_values.append(_as_float(tr.get("collect_predicted_risk"), 0.0))
        queue_values.append(_as_float(tr.get("collect_candidate_queue_pressure"), 0.0))
        hotspot_values.append(_as_float(tr.get("collect_candidate_hotspot_cost"), 0.0))
        if _truthy(tr.get("feedback_guard_interrupted")) or _truthy(metrics.get("feedback_guard_interrupted")):
            guard_count += 1
    return {
        "offline_batch_good": float(buckets.get("good", 0)),
        "offline_batch_mid": float(buckets.get("mid", 0)),
        "offline_batch_bad": float(buckets.get("bad", 0)),
        "offline_batch_unknown": float(buckets.get("unknown", 0)),
        "offline_batch_bad_ratio": float(buckets.get("bad", 0)) / max(1, count),
        "offline_batch_reward_mean": _mean_or_zero(rewards),
        "offline_batch_slo_violation_mean": _mean_or_zero(slo_values),
        "offline_batch_latency_p95_mean": _mean_or_zero(p95_values),
        "offline_batch_guard_interrupted": float(guard_count),
        "offline_batch_collect_risk_mean": _mean_or_zero(risk_values),
        "offline_batch_queue_pressure_mean": _mean_or_zero(queue_values),
        "offline_batch_hotspot_cost_mean": _mean_or_zero(hotspot_values),
    }


def _sample_from_bucket(bucket: List[dict], count: int) -> List[dict]:
    count = max(0, int(count))
    if count <= 0 or not bucket:
        return []
    if len(bucket) >= count:
        return random.sample(bucket, count)
    return [random.choice(bucket) for _ in range(count)]


def _is_bad_transition(transition: dict) -> bool:
    if not isinstance(transition, dict):
        return False
    quality_bucket = str(transition.get("collect_quality_bucket", "") or "").strip().lower()
    if quality_bucket == "bad":
        return True
    metrics = transition.get("metrics") or {}
    reward_breakdown = transition.get("reward_breakdown") or {}

    if (
            _truthy(transition.get("feedback_guard_interrupted"))
            or _truthy(metrics.get("feedback_guard_interrupted"))
    ):
        return True
    if _as_float(metrics.get("e2e_slo_violation"), 0.0) >= 0.8:
        return True
    if _as_float(metrics.get("e2e_latency_p95"), 0.0) >= 12.0:
        return True
    if _as_float(metrics.get("latency_guard_max_queue"), 0.0) >= 16.0:
        return True
    if _as_float(transition.get("collect_candidate_queue_pressure"), 0.0) >= 0.65:
        return True
    hotspot_cost = max(
        _as_float(reward_breakdown.get("active_pair_hotspot_cost"), 0.0),
        _as_float(reward_breakdown.get("executed_active_pair_hotspot_cost"), 0.0),
        _as_float(transition.get("collect_candidate_hotspot_cost"), 0.0),
    )
    if hotspot_cost >= 0.08:
        return True
    return False


def _mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _truthy(value) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_float(value, default: float = 0.0) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        value = value.detach().cpu().reshape(-1)[0].item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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
