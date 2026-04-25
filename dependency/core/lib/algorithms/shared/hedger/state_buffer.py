"""
Thread-safe state buffer used by Hedger.

Writers continuously append observations through `add_*` APIs.
Readers fetch consistent fixed-horizon snapshots through `get_*` APIs.

Key properties
--------------
1. Thread safety: all accesses are protected by an internal lock.
2. Blocking snapshots: `get_state()` can wait until the buffer is ready.
3. Multi-consumer friendly: readers do not consume data, so deployment and
   offloading loops can read concurrently at different cadences.
4. Fixed-horizon tensors: dynamic sequences are padded or truncated to
   `seq_len`.
"""

import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BufferWaitCfg:
    """
    Waiting policy for producing a consistent snapshot.

    min_dynamic_len:
        Minimum required length for each dynamic sequence buffer, such as
        bandwidth, utilization, and task-complexity history.

    require_full_seq:
        If True, every dynamic buffer must satisfy `len(buf) >= seq_len`.
        If False, only `len(buf) >= min_dynamic_len` is required and the
        snapshot builder will pad to `seq_len`.

    timeout_s:
        Maximum number of seconds to wait. None means wait forever.
    """
    min_dynamic_len: int = 1
    require_full_seq: bool = True
    timeout_s: Optional[float] = 5.0


class StateBuffer:
    DEFAULT_LAN_BANDWIDTH_MBPS = 100.0
    LATENCY_SHORT_ALPHA = 0.30
    LATENCY_LONG_ALPHA = 0.10
    LATENCY_DEV_ALPHA = 0.20
    LATENCY_PAIR_CONF_DENOM = 5.0
    LATENCY_BASELINE_CONF_DENOM = 10.0
    LATENCY_TASK_FRESHNESS_TAU = 150.0
    LATENCY_DEPLOYMENT_FRESHNESS_TAU = 3.0

    QUEUE_SHORT_ALPHA = 0.35
    QUEUE_LONG_ALPHA = 0.10
    QUEUE_BUSY_ALPHA = 0.20
    QUEUE_PAIR_CONF_DENOM = 5.0
    QUEUE_BASELINE_CONF_DENOM = 10.0
    QUEUE_TIME_FRESHNESS_TAU_S = 10.0
    QUEUE_DEPLOYMENT_FRESHNESS_TAU = 3.0

    def __init__(self, max_capacity: int, logical_topology, physical_topology,
                 fixed_lan_bandwidth_mbps: float = DEFAULT_LAN_BANDWIDTH_MBPS):
        self.max_capacity = int(max_capacity)
        self.logical_topology = logical_topology
        self.physical_topology = physical_topology
        self.fixed_lan_bandwidth_mbps = max(0.0, float(fixed_lan_bandwidth_mbps))
        self._service_to_idx = {self.logical_topology[i]: i for i in range(len(self.logical_topology))}
        self._device_to_idx = {self.physical_topology[i]: i for i in range(len(self.physical_topology))}

        # All buffers are guarded by this lock/condition pair.
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)

        # Monotonic version number incremented on every append/update.
        self._version = 0
        self._task_observation_version = 0
        self._offloading_reward_min_task_version = 0
        self._offloading_reward_active_deployment_version = None

        # Readiness flags for static features.
        self._static_logic_ready = False
        self._static_phys_ready = False

        self._init_buffer()

    # -----------------------
    # Initialization
    # -----------------------
    def _init_buffer(self):
        """Initialize all buffers from the topology sizes."""
        num_services = self.logical_topology.node_num
        num_devices = self.physical_topology.node_num

        # Static logical features.
        self.model_flops_buffer: List[float] = [0.0 for _ in range(num_services)]
        self.model_memory_buffer: List[float] = [0.0 for _ in range(num_services)]

        # Dynamic logical features, one time series per service.
        self.task_complexity_buffer: List[List[float]] = [[] for _ in range(num_services)]
        self.task_end_to_end_latency_buffer: List[Tuple[int, float, Optional[int]]] = []

        # Static physical features.
        self.gpu_flops_buffer: List[float] = [0.0 for _ in range(num_devices)]
        self.memory_capacity_buffer: List[float] = [0.0 for _ in range(num_devices)]
        self.device_role_buffer: List[int] = []
        for device_idx in range(num_devices):
            if device_idx == self.physical_topology.cloud_idx:
                self.device_role_buffer.append(1)
            else:
                self.device_role_buffer.append(0)  # 0: edge, 1: cloud

        # Dynamic physical features, one time series per device.
        self.bandwidth_buffer: List[List[float]] = [[] for _ in range(num_devices)]
        self.gpu_utilization_buffer: List[List[float]] = [[] for _ in range(num_devices)]
        self.memory_utilization_buffer: List[List[float]] = [[] for _ in range(num_devices)]

        # Task-level observations and offloading rewards carry deployment
        # versions so delayed tasks are not credited to a newer deployment.
        self.task_observation_version_buffer: List[Tuple[int, Optional[int]]] = []
        self.offloading_reward_buffer: List[Dict[str, Any]] = []

        # Shared service-device latency pair table plus hierarchical backoff baselines.
        self.latency_pair_short = np.zeros((num_services, num_devices), dtype=np.float32)
        self.latency_pair_long = np.zeros((num_services, num_devices), dtype=np.float32)
        self.latency_pair_dev = np.zeros((num_services, num_devices), dtype=np.float32)
        self.latency_pair_last = np.zeros((num_services, num_devices), dtype=np.float32)
        self.latency_pair_obs_count = np.zeros((num_services, num_devices), dtype=np.int32)
        self.latency_pair_last_task_v = np.zeros((num_services, num_devices), dtype=np.int64)
        self.latency_pair_last_dep_v = np.full((num_services, num_devices), -1, dtype=np.int64)

        self.latency_service_short = np.zeros(num_services, dtype=np.float32)
        self.latency_service_long = np.zeros(num_services, dtype=np.float32)
        self.latency_service_dev = np.zeros(num_services, dtype=np.float32)
        self.latency_service_obs_count = np.zeros(num_services, dtype=np.int32)

        self.latency_device_short = np.zeros(num_devices, dtype=np.float32)
        self.latency_device_long = np.zeros(num_devices, dtype=np.float32)
        self.latency_device_dev = np.zeros(num_devices, dtype=np.float32)
        self.latency_device_obs_count = np.zeros(num_devices, dtype=np.int32)

        self.latency_global_short = 0.0
        self.latency_global_long = 0.0
        self.latency_global_dev = 0.0
        self.latency_global_obs_count = 0

        # Independent queue pair table with wall-clock freshness.
        self.queue_pair_short = np.zeros((num_services, num_devices), dtype=np.float32)
        self.queue_pair_long = np.zeros((num_services, num_devices), dtype=np.float32)
        self.queue_pair_busy = np.zeros((num_services, num_devices), dtype=np.float32)
        self.queue_pair_last = np.zeros((num_services, num_devices), dtype=np.float32)
        self.queue_pair_obs_count = np.zeros((num_services, num_devices), dtype=np.int32)
        self.queue_pair_last_t = np.zeros((num_services, num_devices), dtype=np.float64)
        self.queue_pair_last_dep_v = np.full((num_services, num_devices), -1, dtype=np.int64)

        self.queue_service_short = np.zeros(num_services, dtype=np.float32)
        self.queue_service_long = np.zeros(num_services, dtype=np.float32)
        self.queue_service_busy = np.zeros(num_services, dtype=np.float32)
        self.queue_service_obs_count = np.zeros(num_services, dtype=np.int32)

        self.queue_device_short = np.zeros(num_devices, dtype=np.float32)
        self.queue_device_long = np.zeros(num_devices, dtype=np.float32)
        self.queue_device_busy = np.zeros(num_devices, dtype=np.float32)
        self.queue_device_obs_count = np.zeros(num_devices, dtype=np.int32)

        self.queue_global_short = 0.0
        self.queue_global_long = 0.0
        self.queue_global_busy = 0.0
        self.queue_global_obs_count = 0

        # Track whether each static entry has been observed at least once.
        self._logic_flops_seen = [False for _ in range(num_services)]
        self._logic_memory_seen = [False for _ in range(num_services)]
        self._logic_memory_edge_seen = [False for _ in range(num_services)]
        self._logic_memory_edge_max = [0.0 for _ in range(num_services)]
        self._logic_memory_cloud_seen = [False for _ in range(num_services)]
        self._logic_memory_cloud_max = [0.0 for _ in range(num_services)]
        self._phys_flops_seen = [False for _ in range(num_devices)]
        self._phys_memory_seen = [False for _ in range(num_devices)]

    # -----------------------
    # Internal helpers
    # -----------------------
    def _bump_version(self):
        self._version += 1
        self._cond.notify_all()

    def _trim_inplace(self, seq: List[Any]):
        """Trim a list in place so that it stays within `max_capacity`."""
        if self.max_capacity <= 0:
            return
        if len(seq) > self.max_capacity:
            del seq[: len(seq) - self.max_capacity]

    @staticmethod
    def _normalize_deployment_version(deployment_version) -> Optional[int]:
        if deployment_version is None:
            return None
        try:
            return int(deployment_version)
        except (TypeError, ValueError):
            return None

    def _matching_offloading_rewards_locked(self, deployment_version=None) -> List[float]:
        target_version = self._normalize_deployment_version(deployment_version)
        rewards = []
        for item in self.offloading_reward_buffer:
            item_version = item.get("deployment_version")
            if target_version is not None and item_version != target_version:
                continue
            rewards.append(float(item["reward"]))
        return rewards

    def _resolve_service_index(self, service_name: str) -> Optional[int]:
        return self._service_to_idx.get(service_name)

    def _resolve_device_index(self, device_name: str) -> Optional[int]:
        return self._device_to_idx.get(device_name)

    @staticmethod
    def _merge_static_service_scalar(current: float, new_value: float) -> float:
        """
        Merge device-agnostic service-level static scalars.

        When a caller does not provide device provenance, keeping the maximum
        avoids arrival-order dependent overwrites and remains conservative.
        """
        return max(float(current), float(new_value))

    def _refresh_effective_model_memory(self, s_idx: int):
        """
        Refresh the logical memory requirement after a device-aware update.

        Deployment capacity checks mainly decide whether an edge node can host a
        service. Cloud-side pod requests can be much larger than edge-side pod
        requests, so cloud observations are used only as a startup fallback until
        at least one edge observation for the service is available.
        """
        if self._logic_memory_edge_seen[s_idx]:
            self.model_memory_buffer[s_idx] = float(self._logic_memory_edge_max[s_idx])
        elif self._logic_memory_cloud_seen[s_idx]:
            self.model_memory_buffer[s_idx] = float(self._logic_memory_cloud_max[s_idx])
        self._logic_memory_seen[s_idx] = (
            self._logic_memory_seen[s_idx]
            or self._logic_memory_edge_seen[s_idx]
            or self._logic_memory_cloud_seen[s_idx]
        )

    def _pad_trunc_1d(self, x: List[float], seq_len: int, pad_mode: str = "edge") -> List[float]:
        """
        Pad or truncate a 1D sequence to `seq_len`.

        pad_mode:
          - "edge": pad with the last element, or 0 if the sequence is empty
          - "zero": pad with 0
        """
        if seq_len <= 0:
            return []
        if len(x) >= seq_len:
            return x[-seq_len:]
        # Pad the missing time steps.
        if pad_mode == "zero" or len(x) == 0:
            pad_val = 0.0
        else:
            pad_val = float(x[-1])
        pad = [pad_val] * (seq_len - len(x))
        return pad + x

    def _ready_for_snapshot(self, seq_len: int, wait_cfg: BufferWaitCfg) -> bool:
        """Return whether the buffer is ready to produce a snapshot."""
        if not (self._static_logic_ready and self._static_phys_ready):
            return False

        min_len = max(0, int(wait_cfg.min_dynamic_len))

        # Check dynamic buffer lengths.
        def min_dyn_len(bufs: List[List[float]]) -> int:
            if not bufs:
                return 0
            return min((len(b) for b in bufs), default=0)

        # First satisfy the minimum dynamic-length requirement.
        if min_dyn_len(self.bandwidth_buffer) < min_len:
            return False
        if min_dyn_len(self.gpu_utilization_buffer) < min_len:
            return False
        if min_dyn_len(self.memory_utilization_buffer) < min_len:
            return False
        if min_dyn_len(self.task_complexity_buffer) < min_len:
            return False
        if wait_cfg.require_full_seq:
            if min_dyn_len(self.bandwidth_buffer) < seq_len:
                return False
            if min_dyn_len(self.gpu_utilization_buffer) < seq_len:
                return False
            if min_dyn_len(self.memory_utilization_buffer) < seq_len:
                return False
            if min_dyn_len(self.task_complexity_buffer) < seq_len:
                return False
        return True

    @staticmethod
    def _sanitize_non_negative(value: Any) -> Optional[float]:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return max(0.0, value)

    @staticmethod
    def _ema_update_latency_state(
            short_value: float,
            long_value: float,
            dev_value: float,
            count: int,
            x: float,
            alpha_short: float,
            alpha_long: float,
            alpha_dev: float,
    ) -> Tuple[float, float, float, int]:
        if count <= 0:
            return float(x), float(x), 0.0, 1
        new_short = (1.0 - alpha_short) * float(short_value) + alpha_short * float(x)
        new_long = (1.0 - alpha_long) * float(long_value) + alpha_long * float(x)
        new_dev = (1.0 - alpha_dev) * float(dev_value) + alpha_dev * abs(float(x) - new_short)
        return float(new_short), float(new_long), float(new_dev), int(count) + 1

    @staticmethod
    def _ema_update_queue_state(
            short_value: float,
            long_value: float,
            busy_value: float,
            count: int,
            x: float,
            busy: float,
            alpha_short: float,
            alpha_long: float,
            alpha_busy: float,
    ) -> Tuple[float, float, float, int]:
        if count <= 0:
            return float(x), float(x), float(busy), 1
        new_short = (1.0 - alpha_short) * float(short_value) + alpha_short * float(x)
        new_long = (1.0 - alpha_long) * float(long_value) + alpha_long * float(x)
        new_busy = (1.0 - alpha_busy) * float(busy_value) + alpha_busy * float(busy)
        return float(new_short), float(new_long), float(new_busy), int(count) + 1

    @staticmethod
    def _confidence(counts: np.ndarray, denom: float) -> np.ndarray:
        counts = counts.astype(np.float32)
        return counts / (counts + float(denom))

    @staticmethod
    def _service_device_baseline(
            service_values: np.ndarray,
            service_counts: np.ndarray,
            device_values: np.ndarray,
            device_counts: np.ndarray,
            global_value: float,
            denom: float,
    ) -> np.ndarray:
        service_conf = StateBuffer._confidence(service_counts, denom).reshape(-1, 1)
        device_conf = StateBuffer._confidence(device_counts, denom).reshape(1, -1)
        numerator = (
            service_conf * service_values.reshape(-1, 1)
            + device_conf * device_values.reshape(1, -1)
            + float(global_value)
        )
        return numerator / (service_conf + device_conf + 1.0)

    @staticmethod
    def _freshness_from_versions(
            last_versions: np.ndarray,
            counts: np.ndarray,
            current_version: Optional[int],
            tau: float,
    ) -> np.ndarray:
        if current_version is None:
            return np.where(counts > 0, 1.0, 0.0).astype(np.float32)
        delta = np.maximum(0.0, float(current_version) - last_versions.astype(np.float32))
        freshness = np.exp(-delta / float(tau))
        freshness = np.where(counts > 0, freshness, 0.0)
        return freshness.astype(np.float32)

    @staticmethod
    def _freshness_from_time(
            last_timestamps: np.ndarray,
            counts: np.ndarray,
            now_monotonic: float,
            tau_s: float,
    ) -> np.ndarray:
        delta = np.maximum(0.0, float(now_monotonic) - last_timestamps.astype(np.float32))
        freshness = np.exp(-delta / float(tau_s))
        freshness = np.where(counts > 0, freshness, 0.0)
        return freshness.astype(np.float32)

    def _service_names(self) -> List[str]:
        return [self.logical_topology[i] for i in range(self.logical_topology.node_num)]

    def _device_names(self) -> List[str]:
        return [self.physical_topology[i] for i in range(self.physical_topology.node_num)]

    def _build_pair_snapshot_locked(
            self,
            current_deployment_version: Optional[int] = None,
            now_monotonic: Optional[float] = None,
    ) -> Dict[str, Any]:
        current_task_version = int(self._task_observation_version)
        current_deployment_version = self._normalize_deployment_version(current_deployment_version)
        if now_monotonic is None:
            now_monotonic = time.monotonic()

        latency_base_short = self._service_device_baseline(
            self.latency_service_short,
            self.latency_service_obs_count,
            self.latency_device_short,
            self.latency_device_obs_count,
            self.latency_global_short,
            self.LATENCY_BASELINE_CONF_DENOM,
        )
        latency_base_long = self._service_device_baseline(
            self.latency_service_long,
            self.latency_service_obs_count,
            self.latency_device_long,
            self.latency_device_obs_count,
            self.latency_global_long,
            self.LATENCY_BASELINE_CONF_DENOM,
        )
        latency_base_dev = self._service_device_baseline(
            self.latency_service_dev,
            self.latency_service_obs_count,
            self.latency_device_dev,
            self.latency_device_obs_count,
            self.latency_global_dev,
            self.LATENCY_BASELINE_CONF_DENOM,
        )

        latency_pair_conf = self._confidence(self.latency_pair_obs_count, self.LATENCY_PAIR_CONF_DENOM)
        latency_fresh_task = self._freshness_from_versions(
            self.latency_pair_last_task_v,
            self.latency_pair_obs_count,
            current_task_version,
            self.LATENCY_TASK_FRESHNESS_TAU,
        )
        latency_fresh_dep = self._freshness_from_versions(
            self.latency_pair_last_dep_v,
            self.latency_pair_obs_count,
            current_deployment_version,
            self.LATENCY_DEPLOYMENT_FRESHNESS_TAU,
        )
        latency_rel_off = latency_pair_conf * latency_fresh_task
        latency_rel_dep = latency_pair_conf * latency_fresh_task * latency_fresh_dep

        latency_pred_short_off = latency_rel_off * self.latency_pair_short + (1.0 - latency_rel_off) * latency_base_short
        latency_pred_long_off = latency_rel_off * self.latency_pair_long + (1.0 - latency_rel_off) * latency_base_long
        latency_pred_last_off = latency_rel_off * self.latency_pair_last + (1.0 - latency_rel_off) * latency_base_short
        latency_pred_dev_off = latency_rel_off * self.latency_pair_dev + (1.0 - latency_rel_off) * latency_base_dev

        latency_pred_short_dep = latency_rel_dep * self.latency_pair_short + (1.0 - latency_rel_dep) * latency_base_short
        latency_pred_long_dep = latency_rel_dep * self.latency_pair_long + (1.0 - latency_rel_dep) * latency_base_long
        latency_pred_dev_dep = latency_rel_dep * self.latency_pair_dev + (1.0 - latency_rel_dep) * latency_base_dev

        deployment_latency_features = np.stack([
            latency_pred_long_dep - latency_base_long,
            latency_pred_short_dep - latency_base_short,
            (latency_pred_short_dep - latency_base_short) - (latency_pred_long_dep - latency_base_long),
            latency_pred_dev_dep,
            latency_rel_dep,
        ], axis=-1).astype(np.float32)
        offloading_latency_features = np.stack([
            latency_pred_short_off - latency_base_short,
            latency_pred_long_off - latency_base_long,
            latency_pred_last_off - latency_pred_short_off,
            (latency_pred_short_off - latency_base_short) - (latency_pred_long_off - latency_base_long),
            latency_pred_dev_off,
            latency_rel_off,
        ], axis=-1).astype(np.float32)

        queue_base_short = self._service_device_baseline(
            self.queue_service_short,
            self.queue_service_obs_count,
            self.queue_device_short,
            self.queue_device_obs_count,
            self.queue_global_short,
            self.QUEUE_BASELINE_CONF_DENOM,
        )
        queue_base_long = self._service_device_baseline(
            self.queue_service_long,
            self.queue_service_obs_count,
            self.queue_device_long,
            self.queue_device_obs_count,
            self.queue_global_long,
            self.QUEUE_BASELINE_CONF_DENOM,
        )
        queue_base_busy = self._service_device_baseline(
            self.queue_service_busy,
            self.queue_service_obs_count,
            self.queue_device_busy,
            self.queue_device_obs_count,
            self.queue_global_busy,
            self.QUEUE_BASELINE_CONF_DENOM,
        )

        queue_pair_conf = self._confidence(self.queue_pair_obs_count, self.QUEUE_PAIR_CONF_DENOM)
        queue_fresh_time = self._freshness_from_time(
            self.queue_pair_last_t,
            self.queue_pair_obs_count,
            float(now_monotonic),
            self.QUEUE_TIME_FRESHNESS_TAU_S,
        )
        queue_fresh_dep = self._freshness_from_versions(
            self.queue_pair_last_dep_v,
            self.queue_pair_obs_count,
            current_deployment_version,
            self.QUEUE_DEPLOYMENT_FRESHNESS_TAU,
        )
        queue_rel = queue_pair_conf * queue_fresh_time * queue_fresh_dep

        queue_pred_short = queue_rel * self.queue_pair_short + (1.0 - queue_rel) * queue_base_short
        queue_pred_long = queue_rel * self.queue_pair_long + (1.0 - queue_rel) * queue_base_long
        queue_pred_busy = queue_rel * self.queue_pair_busy + (1.0 - queue_rel) * queue_base_busy
        queue_pred_last = queue_rel * self.queue_pair_last + (1.0 - queue_rel) * queue_base_short

        deployment_queue_features = np.stack([
            queue_pred_long - queue_base_long,
            queue_pred_short - queue_base_short,
            queue_pred_busy,
            queue_rel,
        ], axis=-1).astype(np.float32)
        offloading_queue_features = np.stack([
            queue_pred_short - queue_base_short,
            queue_pred_long - queue_base_long,
            queue_pred_busy,
            queue_pred_last - queue_pred_short,
            queue_rel,
        ], axis=-1).astype(np.float32)

        return {
            "service_names": self._service_names(),
            "device_names": self._device_names(),
            "current_task_version": current_task_version,
            "current_deployment_version": current_deployment_version,
            "monotonic_time": float(now_monotonic),
            "latency_pair_snapshot": {
                "pair_short": self.latency_pair_short.tolist(),
                "pair_long": self.latency_pair_long.tolist(),
                "pair_dev": self.latency_pair_dev.tolist(),
                "pair_last": self.latency_pair_last.tolist(),
                "pair_count": self.latency_pair_obs_count.tolist(),
                "pair_last_task_v": self.latency_pair_last_task_v.tolist(),
                "pair_last_dep_v": self.latency_pair_last_dep_v.tolist(),
                "service_baseline": {
                    "short": self.latency_service_short.tolist(),
                    "long": self.latency_service_long.tolist(),
                    "dev": self.latency_service_dev.tolist(),
                    "count": self.latency_service_obs_count.tolist(),
                },
                "device_baseline": {
                    "short": self.latency_device_short.tolist(),
                    "long": self.latency_device_long.tolist(),
                    "dev": self.latency_device_dev.tolist(),
                    "count": self.latency_device_obs_count.tolist(),
                },
                "global_baseline": {
                    "short": float(self.latency_global_short),
                    "long": float(self.latency_global_long),
                    "dev": float(self.latency_global_dev),
                    "count": int(self.latency_global_obs_count),
                },
            },
            "queue_pair_snapshot": {
                "pair_short": self.queue_pair_short.tolist(),
                "pair_long": self.queue_pair_long.tolist(),
                "pair_busy": self.queue_pair_busy.tolist(),
                "pair_last": self.queue_pair_last.tolist(),
                "pair_count": self.queue_pair_obs_count.tolist(),
                "pair_last_t": self.queue_pair_last_t.tolist(),
                "pair_last_dep_v": self.queue_pair_last_dep_v.tolist(),
                "service_baseline": {
                    "short": self.queue_service_short.tolist(),
                    "long": self.queue_service_long.tolist(),
                    "busy": self.queue_service_busy.tolist(),
                    "count": self.queue_service_obs_count.tolist(),
                },
                "device_baseline": {
                    "short": self.queue_device_short.tolist(),
                    "long": self.queue_device_long.tolist(),
                    "busy": self.queue_device_busy.tolist(),
                    "count": self.queue_device_obs_count.tolist(),
                },
                "global_baseline": {
                    "short": float(self.queue_global_short),
                    "long": float(self.queue_global_long),
                    "busy": float(self.queue_global_busy),
                    "count": int(self.queue_global_obs_count),
                },
            },
            "pair_feature_snapshot": {
                "deployment_latency": deployment_latency_features.tolist(),
                "offloading_latency": offloading_latency_features.tolist(),
                "deployment_queue": deployment_queue_features.tolist(),
                "offloading_queue": offloading_queue_features.tolist(),
            },
        }

    def _mark_static_logic_ready(self):
        self._static_logic_ready = all(self._logic_flops_seen) and all(self._logic_memory_seen)

    def _mark_static_phys_ready(self):
        self._static_phys_ready = all(self._phys_flops_seen) and all(self._phys_memory_seen)

    # -----------------------
    # Write APIs
    # -----------------------
    def add_model_flops(self, service_or_values, flops: Optional[float] = None):
        with self._cond:
            if flops is None:
                if isinstance(service_or_values, dict):
                    for service_name, value in service_or_values.items():
                        s_idx = self._resolve_service_index(service_name)
                        if s_idx is None:
                            continue
                        self.model_flops_buffer[s_idx] = self._merge_static_service_scalar(
                            self.model_flops_buffer[s_idx],
                            value,
                        )
                        self._logic_flops_seen[s_idx] = True
                else:
                    flops_list = list(service_or_values)
                    assert len(flops_list) == len(self.model_flops_buffer), \
                        f"model flops length mismatch: {len(flops_list)} vs {len(self.model_flops_buffer)}"
                    self.model_flops_buffer = list(map(float, flops_list))
                    self._logic_flops_seen = [True for _ in self.model_flops_buffer]
            else:
                s_idx = self._resolve_service_index(service_or_values)
                if s_idx is None:
                    return
                self.model_flops_buffer[s_idx] = self._merge_static_service_scalar(
                    self.model_flops_buffer[s_idx],
                    flops,
                )
                self._logic_flops_seen[s_idx] = True
            self._mark_static_logic_ready()
            self._bump_version()

    def add_model_memory(self, service_or_values, memory: Optional[float] = None):
        with self._cond:
            if memory is None:
                if isinstance(service_or_values, dict):
                    for service_name, value in service_or_values.items():
                        s_idx = self._resolve_service_index(service_name)
                        if s_idx is None:
                            continue
                        self.model_memory_buffer[s_idx] = self._merge_static_service_scalar(
                            self.model_memory_buffer[s_idx],
                            value,
                        )
                        self._logic_memory_seen[s_idx] = True
                else:
                    memory_list = list(service_or_values)
                    assert len(memory_list) == len(self.model_memory_buffer), \
                        f"model memory length mismatch: {len(memory_list)} vs {len(self.model_memory_buffer)}"
                    self.model_memory_buffer = list(map(float, memory_list))
                    self._logic_memory_seen = [True for _ in self.model_memory_buffer]
            else:
                s_idx = self._resolve_service_index(service_or_values)
                if s_idx is None:
                    return
                self.model_memory_buffer[s_idx] = self._merge_static_service_scalar(
                    self.model_memory_buffer[s_idx],
                    memory,
                )
                self._logic_memory_seen[s_idx] = True
            self._mark_static_logic_ready()
            self._bump_version()

    def add_model_memory_from_device(self, device_name: str, service_name: str, memory: float):
        """
        Update a service memory requirement while preserving device provenance.

        The state model exposes one `model_mem` scalar per logical service, while
        the monitor reports memory from pods deployed on concrete devices. Edge
        observations are the best signal for edge placement feasibility; cloud
        observations are retained only as fallback for services that have not yet
        been observed on any edge node.
        """
        with self._cond:
            s_idx = self._resolve_service_index(service_name)
            d_idx = self._resolve_device_index(device_name)
            if s_idx is None or d_idx is None:
                return

            value = max(0.0, float(memory))
            if d_idx == self.physical_topology.cloud_idx:
                self._logic_memory_cloud_max[s_idx] = max(self._logic_memory_cloud_max[s_idx], value)
                self._logic_memory_cloud_seen[s_idx] = True
            else:
                self._logic_memory_edge_max[s_idx] = max(self._logic_memory_edge_max[s_idx], value)
                self._logic_memory_edge_seen[s_idx] = True

            self._refresh_effective_model_memory(s_idx)
            self._mark_static_logic_ready()
            self._bump_version()

    def add_task_complexity(self, service_name: str, complexity: float):
        """Append a task-complexity observation for a service."""
        with self._cond:
            s_idx = self._resolve_service_index(service_name)
            if s_idx is None:
                return
            self.task_complexity_buffer[s_idx].append(float(complexity))
            self._trim_inplace(self.task_complexity_buffer[s_idx])
            self._bump_version()

    def add_task_latency_pair(
            self,
            service_name: str,
            device_name: str,
            latency: float,
            task_version: Optional[int] = None,
            deployment_version=None,
    ):
        with self._cond:
            s_idx = self._resolve_service_index(service_name)
            d_idx = self._resolve_device_index(device_name)
            value = self._sanitize_non_negative(latency)
            if s_idx is None or d_idx is None or value is None:
                return

            x = float(np.log1p(value))
            dep_version = self._normalize_deployment_version(deployment_version)
            if task_version is None:
                task_version = self._task_observation_version
            task_version = int(task_version)

            short, long_, dev, count = self._ema_update_latency_state(
                self.latency_pair_short[s_idx, d_idx],
                self.latency_pair_long[s_idx, d_idx],
                self.latency_pair_dev[s_idx, d_idx],
                int(self.latency_pair_obs_count[s_idx, d_idx]),
                x,
                self.LATENCY_SHORT_ALPHA,
                self.LATENCY_LONG_ALPHA,
                self.LATENCY_DEV_ALPHA,
            )
            self.latency_pair_short[s_idx, d_idx] = short
            self.latency_pair_long[s_idx, d_idx] = long_
            self.latency_pair_dev[s_idx, d_idx] = dev
            self.latency_pair_last[s_idx, d_idx] = x
            self.latency_pair_obs_count[s_idx, d_idx] = count
            self.latency_pair_last_task_v[s_idx, d_idx] = task_version
            self.latency_pair_last_dep_v[s_idx, d_idx] = -1 if dep_version is None else dep_version

            short, long_, dev, count = self._ema_update_latency_state(
                self.latency_service_short[s_idx],
                self.latency_service_long[s_idx],
                self.latency_service_dev[s_idx],
                int(self.latency_service_obs_count[s_idx]),
                x,
                self.LATENCY_SHORT_ALPHA,
                self.LATENCY_LONG_ALPHA,
                self.LATENCY_DEV_ALPHA,
            )
            self.latency_service_short[s_idx] = short
            self.latency_service_long[s_idx] = long_
            self.latency_service_dev[s_idx] = dev
            self.latency_service_obs_count[s_idx] = count

            short, long_, dev, count = self._ema_update_latency_state(
                self.latency_device_short[d_idx],
                self.latency_device_long[d_idx],
                self.latency_device_dev[d_idx],
                int(self.latency_device_obs_count[d_idx]),
                x,
                self.LATENCY_SHORT_ALPHA,
                self.LATENCY_LONG_ALPHA,
                self.LATENCY_DEV_ALPHA,
            )
            self.latency_device_short[d_idx] = short
            self.latency_device_long[d_idx] = long_
            self.latency_device_dev[d_idx] = dev
            self.latency_device_obs_count[d_idx] = count

            short, long_, dev, count = self._ema_update_latency_state(
                self.latency_global_short,
                self.latency_global_long,
                self.latency_global_dev,
                int(self.latency_global_obs_count),
                x,
                self.LATENCY_SHORT_ALPHA,
                self.LATENCY_LONG_ALPHA,
                self.LATENCY_DEV_ALPHA,
            )
            self.latency_global_short = short
            self.latency_global_long = long_
            self.latency_global_dev = dev
            self.latency_global_obs_count = count
            self._bump_version()

    def add_task_end_to_end_latency(self, latency: float, deployment_version=None):
        """Append one end-to-end latency observation for a completed task."""
        with self._cond:
            self._task_observation_version += 1
            normalized_deployment_version = self._normalize_deployment_version(deployment_version)
            self.task_end_to_end_latency_buffer.append(
                (self._task_observation_version, float(latency), normalized_deployment_version)
            )
            self._trim_inplace(self.task_end_to_end_latency_buffer)
            self.task_observation_version_buffer.append(
                (self._task_observation_version, normalized_deployment_version)
            )
            self._trim_inplace(self.task_observation_version_buffer)
            self._bump_version()

    def add_gpu_flops(self, device_or_values, flops: Optional[float] = None):
        with self._cond:
            if flops is None:
                if isinstance(device_or_values, dict):
                    for device_name, value in device_or_values.items():
                        d_idx = self._resolve_device_index(device_name)
                        if d_idx is None:
                            continue
                        self.gpu_flops_buffer[d_idx] = float(value)
                        self._phys_flops_seen[d_idx] = True
                else:
                    flops_list = list(device_or_values)
                    assert len(flops_list) == len(self.gpu_flops_buffer), \
                        f"gpu flops length mismatch: {len(flops_list)} vs {len(self.gpu_flops_buffer)}"
                    self.gpu_flops_buffer = list(map(float, flops_list))
                    self._phys_flops_seen = [True for _ in self.gpu_flops_buffer]
            else:
                d_idx = self._resolve_device_index(device_or_values)
                if d_idx is None:
                    return
                self.gpu_flops_buffer[d_idx] = float(flops)
                self._phys_flops_seen[d_idx] = True
            self._mark_static_phys_ready()
            self._bump_version()

    def add_memory_capacity(self, device_or_values, capacity: Optional[float] = None):
        with self._cond:
            if capacity is None:
                if isinstance(device_or_values, dict):
                    for device_name, value in device_or_values.items():
                        d_idx = self._resolve_device_index(device_name)
                        if d_idx is None:
                            continue
                        self.memory_capacity_buffer[d_idx] = float(value)
                        self._phys_memory_seen[d_idx] = True
                else:
                    capacity_list = list(device_or_values)
                    assert len(capacity_list) == len(self.memory_capacity_buffer), \
                        f"memory capacity length mismatch: {len(capacity_list)} vs {len(self.memory_capacity_buffer)}"
                    self.memory_capacity_buffer = list(map(float, capacity_list))
                    self._phys_memory_seen = [True for _ in self.memory_capacity_buffer]
            else:
                d_idx = self._resolve_device_index(device_or_values)
                if d_idx is None:
                    return
                self.memory_capacity_buffer[d_idx] = float(capacity)
                self._phys_memory_seen[d_idx] = True
            self._mark_static_phys_ready()
            self._bump_version()

    def add_bandwidths(self, wan_bandwidth: float):
        """
        Append one WAN observation and expand it into the per-node bandwidth view.

        Hedger models edge-side LAN bandwidth as a fixed constant and only treats
        the cloud link as dynamic. Therefore each appended sample is expanded as:

            - every edge node -> fixed LAN bandwidth
            - cloud node -> measured WAN bandwidth

        The monitor reports a single `available_bandwidth` scalar, which is the
        only dynamic input consumed here.
        """
        with self._cond:
            wan_value = float(wan_bandwidth)

            for d_idx in range(len(self.bandwidth_buffer)):
                if d_idx == self.physical_topology.cloud_idx:
                    bw = wan_value
                else:
                    bw = self.fixed_lan_bandwidth_mbps
                self.bandwidth_buffer[d_idx].append(float(bw))
                self._trim_inplace(self.bandwidth_buffer[d_idx])
            self._bump_version()

    def add_gpu_utilization(self, device_name: str, util: float):
        with self._cond:
            d_idx = self._resolve_device_index(device_name)
            if d_idx is None:
                return
            self.gpu_utilization_buffer[d_idx].append(float(util))
            self._trim_inplace(self.gpu_utilization_buffer[d_idx])
            self._bump_version()

    def add_memory_utilization(self, device_name: str, util: float):
        with self._cond:
            d_idx = self._resolve_device_index(device_name)
            if d_idx is None:
                return
            self.memory_utilization_buffer[d_idx].append(float(util))
            self._trim_inplace(self.memory_utilization_buffer[d_idx])
            self._bump_version()

    def add_queue_lengths(
            self,
            device_name: str,
            queue_lengths: Dict[str, Any],
            deployment_version=None,
            now_monotonic: Optional[float] = None,
    ):
        with self._cond:
            d_idx = self._resolve_device_index(device_name)
            if d_idx is None or not isinstance(queue_lengths, dict) or not queue_lengths:
                return
            dep_version = self._normalize_deployment_version(deployment_version)
            if now_monotonic is None:
                now_monotonic = time.monotonic()

            updated = False
            for service_name, queue_length in queue_lengths.items():
                s_idx = self._resolve_service_index(service_name)
                value = self._sanitize_non_negative(queue_length)
                if s_idx is None or value is None:
                    continue
                x = float(np.log1p(value))
                busy = 1.0 if value > 0.0 else 0.0

                short, long_, busy_ema, count = self._ema_update_queue_state(
                    self.queue_pair_short[s_idx, d_idx],
                    self.queue_pair_long[s_idx, d_idx],
                    self.queue_pair_busy[s_idx, d_idx],
                    int(self.queue_pair_obs_count[s_idx, d_idx]),
                    x,
                    busy,
                    self.QUEUE_SHORT_ALPHA,
                    self.QUEUE_LONG_ALPHA,
                    self.QUEUE_BUSY_ALPHA,
                )
                self.queue_pair_short[s_idx, d_idx] = short
                self.queue_pair_long[s_idx, d_idx] = long_
                self.queue_pair_busy[s_idx, d_idx] = busy_ema
                self.queue_pair_last[s_idx, d_idx] = x
                self.queue_pair_obs_count[s_idx, d_idx] = count
                self.queue_pair_last_t[s_idx, d_idx] = float(now_monotonic)
                self.queue_pair_last_dep_v[s_idx, d_idx] = -1 if dep_version is None else dep_version

                short, long_, busy_ema, count = self._ema_update_queue_state(
                    self.queue_service_short[s_idx],
                    self.queue_service_long[s_idx],
                    self.queue_service_busy[s_idx],
                    int(self.queue_service_obs_count[s_idx]),
                    x,
                    busy,
                    self.QUEUE_SHORT_ALPHA,
                    self.QUEUE_LONG_ALPHA,
                    self.QUEUE_BUSY_ALPHA,
                )
                self.queue_service_short[s_idx] = short
                self.queue_service_long[s_idx] = long_
                self.queue_service_busy[s_idx] = busy_ema
                self.queue_service_obs_count[s_idx] = count

                short, long_, busy_ema, count = self._ema_update_queue_state(
                    self.queue_device_short[d_idx],
                    self.queue_device_long[d_idx],
                    self.queue_device_busy[d_idx],
                    int(self.queue_device_obs_count[d_idx]),
                    x,
                    busy,
                    self.QUEUE_SHORT_ALPHA,
                    self.QUEUE_LONG_ALPHA,
                    self.QUEUE_BUSY_ALPHA,
                )
                self.queue_device_short[d_idx] = short
                self.queue_device_long[d_idx] = long_
                self.queue_device_busy[d_idx] = busy_ema
                self.queue_device_obs_count[d_idx] = count

                short, long_, busy_ema, count = self._ema_update_queue_state(
                    self.queue_global_short,
                    self.queue_global_long,
                    self.queue_global_busy,
                    int(self.queue_global_obs_count),
                    x,
                    busy,
                    self.QUEUE_SHORT_ALPHA,
                    self.QUEUE_LONG_ALPHA,
                    self.QUEUE_BUSY_ALPHA,
                )
                self.queue_global_short = short
                self.queue_global_long = long_
                self.queue_global_busy = busy_ema
                self.queue_global_obs_count = count
                updated = True

            if updated:
                self._bump_version()

    def add_offloading_reward(self, reward: float, task_version: Optional[int] = None,
                              deployment_version=None) -> bool:
        """Append an offloading reward, usually from the offloading loop."""
        with self._cond:
            if task_version is not None and int(task_version) <= self._offloading_reward_min_task_version:
                return False
            target_version = self._normalize_deployment_version(deployment_version)
            if (
                    self._offloading_reward_active_deployment_version is not None
                    and target_version != self._offloading_reward_active_deployment_version
            ):
                return False
            self.offloading_reward_buffer.append({
                "reward": float(reward),
                "task_version": int(task_version) if task_version is not None else None,
                "deployment_version": target_version,
            })
            self._trim_inplace(self.offloading_reward_buffer)
            self._bump_version()
            return True

    def clear_offloading_rewards(self, deployment_version=None):
        """Start a fresh deployment-feedback reward window."""
        with self._cond:
            self.offloading_reward_buffer.clear()
            self._offloading_reward_min_task_version = self._task_observation_version
            self._offloading_reward_active_deployment_version = self._normalize_deployment_version(
                deployment_version
            )
            self._bump_version()

    def get_task_observation_version(self) -> int:
        """Return the number of completed-task latency observations received so far."""
        with self._lock:
            return int(self._task_observation_version)

    def get_task_end_to_end_latency_stats(
            self,
            since_version: Optional[int] = None,
            deployment_version=None,
            last_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Return latency stats over completed-task end-to-end observations.

        `since_version` is exclusive. It is used by the offloading loop to
        compute reward from tasks completed after the sampled action.
        """
        target_version = self._normalize_deployment_version(deployment_version)
        with self._lock:
            records = []
            for version, latency, record_deployment_version in self.task_end_to_end_latency_buffer:
                if since_version is not None and int(version) <= int(since_version):
                    continue
                if target_version is not None and record_deployment_version != target_version:
                    continue
                records.append((int(version), float(latency), record_deployment_version))

            if last_k is not None:
                records = records[-max(1, int(last_k)):]

            if not records:
                return {
                    "mean": 0.0,
                    "latest": 0.0,
                    "count": 0,
                    "latencies": [],
                    "latest_version": int(self._task_observation_version),
                }

            latencies = [record[1] for record in records]
            return {
                "mean": float(np.mean(latencies)),
                "latest": float(latencies[-1]),
                "count": len(latencies),
                "latencies": latencies,
                "latest_version": int(records[-1][0]),
            }

    def get_task_observation_deployment_summary(self, since_version: int) -> Dict[str, Any]:
        """Summarize deployment versions among task observations after `since_version`."""
        since_version = int(since_version)
        with self._lock:
            records = [
                deployment_version
                for version, deployment_version in self.task_observation_version_buffer
                if version > since_version
            ]
            counts: Dict[Optional[int], int] = {}
            for deployment_version in records:
                counts[deployment_version] = counts.get(deployment_version, 0) + 1
            dominant_version = None
            dominant_count = 0
            if counts:
                dominant_version, dominant_count = max(counts.items(), key=lambda item: item[1])
            return {
                "current_version": int(self._task_observation_version),
                "count": len(records),
                "deployment_version_counts": dict(counts),
                "dominant_deployment_version": dominant_version,
                "dominant_count": dominant_count,
                "unique_deployment_versions": len(counts),
                "all_same_deployment_version": len(counts) == 1,
            }

    def wait_for_offloading_rewards(self, min_count: int, timeout_s: Optional[float] = None,
                                    deployment_version=None) -> int:
        """
        Wait until at least `min_count` offloading rewards are available.

        The returned value is the current reward count. Callers can decide
        whether the available feedback is enough for a training transition.
        """
        min_count = max(0, int(min_count))
        with self._cond:
            current_count = len(self._matching_offloading_rewards_locked(deployment_version))
            if current_count >= min_count:
                return current_count

            start_t = time.time()
            while current_count < min_count:
                if timeout_s is None:
                    self._cond.wait(timeout=0.5)
                    current_count = len(self._matching_offloading_rewards_locked(deployment_version))
                    continue

                elapsed = time.time() - start_t
                remaining = float(timeout_s) - elapsed
                if remaining <= 0:
                    break
                self._cond.wait(timeout=min(0.5, remaining))
                current_count = len(self._matching_offloading_rewards_locked(deployment_version))

            return current_count

    # -----------------------
    # Read APIs
    # -----------------------
    def get_state(
            self,
            seq_len: int,
            wait_cfg: Optional[BufferWaitCfg] = None,
            pad_mode: str = "edge",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        logic_feats, phys_feats, _ = self.get_state_bundle(
            seq_len=seq_len,
            wait_cfg=wait_cfg,
            pad_mode=pad_mode,
        )
        return logic_feats, phys_feats

    def get_state_bundle(
            self,
            seq_len: int,
            wait_cfg: Optional[BufferWaitCfg] = None,
            pad_mode: str = "edge",
            current_deployment_version: Optional[int] = None,
            now_monotonic: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Return a snapshot of logical and physical features as Python lists.

        The returned dictionaries match the format that Hedger later converts
        into torch tensors:

        `logic_feats`:
            `model_flops`: `(Ms,)`
            `model_mem`: `(Ms,)`
            `task_complexity_seq`: `(Ms, T)`

        `phys_feats`:
            `gpu_flops`: `(Np,)`
            `role_id`: `(Np,)`
            `mem_capacity`: `(Np,)`
            `bandwidth_seq`: `(Np, T)`
            `gpu_util_seq`: `(Np, T)`
            `mem_util_seq`: `(Np, T)`
        """
        if wait_cfg is None:
            wait_cfg = BufferWaitCfg()

        seq_len = int(seq_len)
        assert seq_len > 0, f"seq_len must be > 0, got {seq_len}"

        with self._cond:
            start_t = time.time()
            while not self._ready_for_snapshot(seq_len, wait_cfg):
                if wait_cfg.timeout_s is None:
                    self._cond.wait(timeout=0.5)
                    continue
                elapsed = time.time() - start_t
                if elapsed >= float(wait_cfg.timeout_s):
                    break
                self._cond.wait(timeout=min(0.5, float(wait_cfg.timeout_s) - elapsed))

            Ms = len(self.model_flops_buffer)
            logic_task_complexity = [
                self._pad_trunc_1d(self.task_complexity_buffer[i], seq_len, pad_mode=pad_mode)
                for i in range(Ms)
            ]
            logic_feats = {
                "model_flops": list(self.model_flops_buffer),
                "model_mem": list(self.model_memory_buffer),
                "task_complexity_seq": logic_task_complexity,
            }

            Np = len(self.gpu_flops_buffer)
            phys_bandwidth = [
                self._pad_trunc_1d(self.bandwidth_buffer[i], seq_len, pad_mode=pad_mode)
                for i in range(Np)
            ]
            phys_gpu_util = [
                self._pad_trunc_1d(self.gpu_utilization_buffer[i], seq_len, pad_mode=pad_mode)
                for i in range(Np)
            ]
            phys_mem_util = [
                self._pad_trunc_1d(self.memory_utilization_buffer[i], seq_len, pad_mode=pad_mode)
                for i in range(Np)
            ]
            phys_feats = {
                "gpu_flops": list(self.gpu_flops_buffer),
                "role_id": list(self.device_role_buffer),
                "mem_capacity": list(self.memory_capacity_buffer),
                "bandwidth_seq": phys_bandwidth,
                "gpu_util_seq": phys_gpu_util,
                "mem_util_seq": phys_mem_util,
            }

            pair_snapshot = self._build_pair_snapshot_locked(
                current_deployment_version=current_deployment_version,
                now_monotonic=now_monotonic,
            )
            pair_snapshot["logic_snapshot"] = logic_feats
            pair_snapshot["phys_snapshot"] = phys_feats
            return logic_feats, phys_feats, pair_snapshot

    def get_offloading_reward_stats(self, last_k: int = 1, deployment_version=None) -> Dict[str, float]:
        """Return the mean and std of the last `last_k` offloading rewards."""
        with self._lock:
            last_k = max(1, int(last_k))
            rewards = self._matching_offloading_rewards_locked(deployment_version)
            if len(rewards) == 0:
                return {"mean": 0.0, "std": 0.0, "count": 0}
            arr = np.array(rewards[-last_k:], dtype=np.float32)
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "count": int(arr.size),
            }
