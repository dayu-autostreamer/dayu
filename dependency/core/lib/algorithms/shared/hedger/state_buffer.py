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
        bandwidth, utilization, and latency history.

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
        self.task_latency_buffer: List[List[float]] = [[] for _ in range(num_services)]

        # Static physical features.
        self.gpu_flops_buffer: List[float] = [0.0 for _ in range(num_devices)]
        self.memory_capacity_buffer: List[float] = [0.0 for _ in range(num_devices)]
        self.device_role_buffer: List[int] = []
        for device_idx in range(num_devices):
            if device_idx == self.physical_topology.source_idx:
                self.device_role_buffer.append(0)
            elif device_idx == self.physical_topology.cloud_idx:
                self.device_role_buffer.append(2)
            else:
                self.device_role_buffer.append(1)  # 0: source edge, 1: other edge, 2: cloud

        # Dynamic physical features, one time series per device.
        self.bandwidth_buffer: List[List[float]] = [[] for _ in range(num_devices)]
        self.gpu_utilization_buffer: List[List[float]] = [[] for _ in range(num_devices)]
        self.memory_utilization_buffer: List[List[float]] = [[] for _ in range(num_devices)]

        # Offloading reward history, used as aggregated feedback for deployment.
        self.offloading_reward_buffer: List[float] = []

        # Track whether each static entry has been observed at least once.
        self._logic_flops_seen = [False for _ in range(num_services)]
        self._logic_memory_seen = [False for _ in range(num_services)]
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

    def _resolve_service_index(self, service_name: str) -> Optional[int]:
        return self._service_to_idx.get(service_name)

    def _resolve_device_index(self, device_name: str) -> Optional[int]:
        return self._device_to_idx.get(device_name)

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
        if min_dyn_len(self.task_latency_buffer) < min_len:
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
            if min_dyn_len(self.task_latency_buffer) < seq_len:
                return False

        return True

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
                        self.model_flops_buffer[s_idx] = float(value)
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
                self.model_flops_buffer[s_idx] = float(flops)
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
                        self.model_memory_buffer[s_idx] = float(value)
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
                self.model_memory_buffer[s_idx] = float(memory)
                self._logic_memory_seen[s_idx] = True
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

    def add_task_latency(self, service_name: str, latency: float):
        """Append a task-latency observation for a service."""
        with self._cond:
            s_idx = self._resolve_service_index(service_name)
            if s_idx is None:
                return
            self.task_latency_buffer[s_idx].append(float(latency))
            self._trim_inplace(self.task_latency_buffer[s_idx])
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

    def add_offloading_reward(self, reward: float):
        """Append an offloading reward, usually from the offloading loop."""
        with self._cond:
            self.offloading_reward_buffer.append(float(reward))
            self._trim_inplace(self.offloading_reward_buffer)
            self._bump_version()

    # -----------------------
    # Read APIs
    # -----------------------
    def get_state(
            self,
            seq_len: int,
            wait_cfg: Optional[BufferWaitCfg] = None,
            pad_mode: str = "edge",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Return a snapshot of logical and physical features as Python lists.

        The returned dictionaries match the format that Hedger later converts
        into torch tensors:

        `logic_feats`:
            `model_flops`: `(Ms,)`
            `model_mem`: `(Ms,)`
            `task_complexity_seq`: `(Ms, T)`
            `hist_latency_seq`: `(Ms, T)`

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
                    # Timeout reached: return a best-effort snapshot.
                    break
                # Wait for the next update.
                self._cond.wait(timeout=min(0.5, float(wait_cfg.timeout_s) - elapsed))

            # Build the logical snapshot.
            Ms = len(self.model_flops_buffer)
            logic_task_complexity = [
                self._pad_trunc_1d(self.task_complexity_buffer[i], seq_len, pad_mode=pad_mode)
                for i in range(Ms)
            ]
            logic_latency = [
                self._pad_trunc_1d(self.task_latency_buffer[i], seq_len, pad_mode=pad_mode)
                for i in range(Ms)
            ]
            logic_feats = {
                "model_flops": list(self.model_flops_buffer),
                "model_mem": list(self.model_memory_buffer),
                "task_complexity_seq": logic_task_complexity,
                "hist_latency_seq": logic_latency,
            }

            # Build the physical snapshot.
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

            return logic_feats, phys_feats

    def get_offloading_reward_stats(self, last_k: int = 1) -> Dict[str, float]:
        """Return the mean and std of the last `last_k` offloading rewards."""
        with self._lock:
            last_k = max(1, int(last_k))
            if len(self.offloading_reward_buffer) == 0:
                return {"mean": 0.0, "std": 0.0}
            arr = np.array(self.offloading_reward_buffer[-last_k:], dtype=np.float32)
            return {"mean": float(arr.mean()), "std": float(arr.std())}
