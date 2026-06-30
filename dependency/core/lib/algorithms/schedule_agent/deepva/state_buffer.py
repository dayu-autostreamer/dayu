import math
import threading
import time
from collections import deque

import numpy as np


class StateBuffer:
    """Runtime state used by the plain DeepVA DRL baseline.

    The state is a service-device pair tensor.  It intentionally avoids
    Hedger-only graph encoders and hand-built QK scores while still exposing
    the basic signals a normal DRL policy needs to make deployment/offloading
    decisions.
    """

    FEATURE_NAMES = (
        "prev_deployed",
        "prev_target",
        "complexity",
        "arrival_rate",
        "model_flops",
        "model_memory",
        "gpu_flops",
        "mem_available",
        "cpu_usage",
        "queue_length",
        "runtime",
        "runtime_confidence",
        "bandwidth",
        "is_cloud",
    )

    DEFAULT_NORMALIZERS = {
        "complexity": 20.0,
        "arrival_rate": 5.0,
        "model_flops": 200.0,
        "model_memory": 8.0,
        "gpu_flops": 10000.0,
        "mem_available": 16.0,
        "queue_length": 10.0,
        "runtime": 3.0,
        "bandwidth": 100.0,
        "delay": 3.0,
    }

    def __init__(
        self,
        service_names,
        device_list,
        cloud_device=None,
        delay_window_size=30,
        runtime_alpha=0.25,
        normalizers=None,
    ):
        self.service_names = [str(name) for name in service_names]
        self.device_list = [str(device) for device in device_list]
        self.cloud_device = str(cloud_device) if cloud_device else None
        self.num_services = len(self.service_names)
        self.num_devices = len(self.device_list)
        self.service_to_idx = {name: idx for idx, name in enumerate(self.service_names)}
        self.device_to_idx = {name: idx for idx, name in enumerate(self.device_list)}

        self.normalizers = dict(self.DEFAULT_NORMALIZERS)
        if isinstance(normalizers, dict):
            for key, value in normalizers.items():
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    self.normalizers[key] = value

        self.delay_buffer = deque(maxlen=int(delay_window_size))
        self.arrival_times = deque(maxlen=200)
        self.runtime_alpha = float(runtime_alpha)

        self.deployment_mask = np.zeros((self.num_services, self.num_devices), dtype=np.float32)
        self.offloading_targets = np.zeros(self.num_services, dtype=np.int64)

        self.service_complexity = np.ones(self.num_services, dtype=np.float32)
        self.model_flops = np.zeros(self.num_services, dtype=np.float32)
        self.model_memory = np.zeros(self.num_services, dtype=np.float32)

        self.cpu_usage = np.zeros(self.num_devices, dtype=np.float32)
        self.memory_usage = np.zeros(self.num_devices, dtype=np.float32)
        self.gpu_flops = np.zeros(self.num_devices, dtype=np.float32)
        self.memory_capacity = np.zeros(self.num_devices, dtype=np.float32)
        self.bandwidth = np.zeros(self.num_devices, dtype=np.float32)

        self.queue_lengths = np.zeros((self.num_services, self.num_devices), dtype=np.float32)
        self.runtime_ewma = np.zeros((self.num_services, self.num_devices), dtype=np.float32)
        self.runtime_counts = np.zeros((self.num_services, self.num_devices), dtype=np.float32)

        self.lock = threading.RLock()

    @property
    def feature_dim(self):
        return len(self.FEATURE_NAMES)

    @staticmethod
    def _safe_float(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default if default is None else float(default)
        if not math.isfinite(value):
            return default if default is None else float(default)
        return float(value)

    @staticmethod
    def _normalize_usage(value):
        value = StateBuffer._safe_float(value, 0.0)
        if value > 1.0:
            value = value / 100.0
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _mean_obj_num(raw):
        if raw is None:
            return None
        try:
            arr = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.size == 0:
            return None
        value = float(np.mean(arr))
        return value if math.isfinite(value) and value > 0 else None

    def _resource_service_value(self, resource, key, service_name, default=0.0):
        value = resource.get(key) if isinstance(resource, dict) else None
        if isinstance(value, dict):
            return self._safe_float(value.get(service_name), default)
        return default

    def update_deployment(self, deployment_mask, offloading_targets=None):
        with self.lock:
            mask = np.asarray(deployment_mask, dtype=np.float32)
            if mask.shape != (self.num_services, self.num_devices):
                raise ValueError(
                    f"DeepVA deployment mask shape mismatch: {mask.shape} vs "
                    f"{(self.num_services, self.num_devices)}"
                )
            self.deployment_mask = (mask > 0).astype(np.float32)
            if offloading_targets is not None:
                targets = np.asarray(offloading_targets, dtype=np.int64)
                if targets.shape != (self.num_services,):
                    raise ValueError(
                        f"DeepVA offloading target shape mismatch: {targets.shape} vs {(self.num_services,)}"
                    )
                self.offloading_targets = np.clip(targets, 0, max(0, self.num_devices - 1))

    def update_scenario(self, scenario):
        if not isinstance(scenario, dict):
            return
        now = time.time()
        complexity = self._mean_obj_num(scenario.get("obj_num"))
        delay = self._safe_float(scenario.get("delay"), None)
        with self.lock:
            self.arrival_times.append(now)
            if delay is not None and delay >= 0:
                self.delay_buffer.append(float(delay))
            if complexity is not None:
                # The scheduler scenario is source-level.  Use it as a fallback
                # demand signal until per-service feedback arrives.
                self.service_complexity = (
                    0.8 * self.service_complexity + 0.2 * float(complexity)
                ).astype(np.float32)

    def update_resource(self, device, resource):
        device = str(device)
        if device not in self.device_to_idx or not isinstance(resource, dict):
            return
        d_idx = self.device_to_idx[device]
        with self.lock:
            self.cpu_usage[d_idx] = self._normalize_usage(resource.get("cpu_usage", 0.0))
            self.memory_usage[d_idx] = self._normalize_usage(resource.get("memory_usage", 0.0))
            self.gpu_flops[d_idx] = self._safe_float(resource.get("gpu_flops"), self.gpu_flops[d_idx])
            self.memory_capacity[d_idx] = self._safe_float(
                resource.get("memory_capacity"), self.memory_capacity[d_idx]
            )
            bandwidth = self._safe_float(resource.get("available_bandwidth"), self.bandwidth[d_idx])
            if bandwidth > 0:
                self.bandwidth[d_idx] = bandwidth

            queue_lengths = resource.get("queue_length")
            if isinstance(queue_lengths, dict):
                for service_name, value in queue_lengths.items():
                    s_idx = self.service_to_idx.get(str(service_name))
                    if s_idx is not None:
                        self.queue_lengths[s_idx, d_idx] = max(0.0, self._safe_float(value, 0.0))
            elif queue_lengths is not None:
                self.queue_lengths[:, d_idx] = max(0.0, self._safe_float(queue_lengths, 0.0))

            for service_name, s_idx in self.service_to_idx.items():
                model_flops = self._resource_service_value(resource, "model_flops", service_name, 0.0)
                model_memory = self._resource_service_value(resource, "model_memory", service_name, 0.0)
                if model_flops > 0:
                    self.model_flops[s_idx] = max(self.model_flops[s_idx], model_flops)
                if model_memory > 0:
                    self.model_memory[s_idx] = max(self.model_memory[s_idx], model_memory)

    def add_task_feedback(self, service_name, device, real_execute_time, complexity=None):
        service_name = str(service_name)
        device = str(device)
        s_idx = self.service_to_idx.get(service_name)
        d_idx = self.device_to_idx.get(device)
        if s_idx is None or d_idx is None:
            return
        runtime = self._safe_float(real_execute_time, 0.0)
        if runtime <= 0:
            return
        complexity_value = self._safe_float(complexity, 0.0)
        with self.lock:
            count = self.runtime_counts[s_idx, d_idx]
            old = self.runtime_ewma[s_idx, d_idx]
            self.runtime_ewma[s_idx, d_idx] = runtime if count <= 0 else (
                (1.0 - self.runtime_alpha) * old + self.runtime_alpha * runtime
            )
            self.runtime_counts[s_idx, d_idx] = count + 1.0
            if complexity_value > 0:
                self.service_complexity[s_idx] = (
                    0.75 * self.service_complexity[s_idx] + 0.25 * complexity_value
                )

    def add_task_delay(self, delay):
        delay = self._safe_float(delay, -1.0)
        if delay < 0:
            return
        with self.lock:
            self.delay_buffer.append(float(delay))

    def _arrival_rate(self):
        if len(self.arrival_times) < 2:
            return 0.0
        now = time.time()
        window_s = 60.0
        recent = [ts for ts in self.arrival_times if now - ts <= window_s]
        if len(recent) < 2:
            return 0.0
        duration = max(1.0, max(recent) - min(recent))
        return float(len(recent) / duration)

    def _log_norm(self, values, key):
        denom = math.log1p(self.normalizers[key])
        if denom <= 0:
            denom = 1.0
        return np.clip(np.log1p(np.maximum(values, 0.0)) / denom, 0.0, 5.0)

    def get_state(self):
        with self.lock:
            S, D = self.num_services, self.num_devices
            state = np.zeros((S, D, self.feature_dim), dtype=np.float32)
            feature_idx = {name: idx for idx, name in enumerate(self.FEATURE_NAMES)}

            state[:, :, feature_idx["prev_deployed"]] = self.deployment_mask
            for s_idx, target in enumerate(self.offloading_targets):
                if 0 <= int(target) < D:
                    state[s_idx, int(target), feature_idx["prev_target"]] = 1.0

            complexity = np.clip(self.service_complexity / self.normalizers["complexity"], 0.0, 5.0)
            arrival_rate = min(5.0, self._arrival_rate() / self.normalizers["arrival_rate"])
            model_flops = self._log_norm(self.model_flops, "model_flops")
            model_memory = self._log_norm(self.model_memory, "model_memory")
            gpu_flops = self._log_norm(self.gpu_flops, "gpu_flops")
            mem_capacity = self.memory_capacity * (1.0 - np.clip(self.memory_usage, 0.0, 1.0))
            mem_available = self._log_norm(mem_capacity, "mem_available")
            queue = np.clip(self.queue_lengths / self.normalizers["queue_length"], 0.0, 5.0)
            runtime = np.clip(self.runtime_ewma / self.normalizers["runtime"], 0.0, 5.0)
            runtime_conf = np.clip(np.log1p(self.runtime_counts) / math.log1p(20.0), 0.0, 1.0)

            bandwidth = self.bandwidth.copy()
            if np.any(bandwidth > 0):
                fallback = float(np.max(bandwidth))
                bandwidth[bandwidth <= 0] = fallback
            bandwidth = np.clip(bandwidth / self.normalizers["bandwidth"], 0.0, 5.0)

            for s_idx in range(S):
                state[s_idx, :, feature_idx["complexity"]] = complexity[s_idx]
                state[s_idx, :, feature_idx["arrival_rate"]] = arrival_rate
                state[s_idx, :, feature_idx["model_flops"]] = model_flops[s_idx]
                state[s_idx, :, feature_idx["model_memory"]] = model_memory[s_idx]
            for d_idx, device in enumerate(self.device_list):
                state[:, d_idx, feature_idx["gpu_flops"]] = gpu_flops[d_idx]
                state[:, d_idx, feature_idx["mem_available"]] = mem_available[d_idx]
                state[:, d_idx, feature_idx["cpu_usage"]] = self.cpu_usage[d_idx]
                state[:, d_idx, feature_idx["bandwidth"]] = bandwidth[d_idx]
                if self._is_cloud_device(device):
                    state[:, d_idx, feature_idx["is_cloud"]] = 1.0

            state[:, :, feature_idx["queue_length"]] = queue
            state[:, :, feature_idx["runtime"]] = runtime
            state[:, :, feature_idx["runtime_confidence"]] = runtime_conf
            return state

    def _is_cloud_device(self, device):
        if self.cloud_device and device == self.cloud_device:
            return True
        return "cloud" in str(device).lower()

    def get_average_delay(self):
        with self.lock:
            return float(np.mean(self.delay_buffer)) if self.delay_buffer else 0.0

    def get_memory_overage(self, deployment_mask=None):
        with self.lock:
            mask = self.deployment_mask if deployment_mask is None else np.asarray(deployment_mask, dtype=np.float32)
            capacity = self.memory_capacity * (1.0 - np.clip(self.memory_usage, 0.0, 1.0))
            used = np.matmul(mask.T, self.model_memory)
            over = np.maximum(used - capacity, 0.0)
            denom = np.maximum(capacity, 1.0)
            return float(np.mean(over / denom)) if over.size else 0.0

    def get_selected_queue_mean(self, offloading_targets=None):
        with self.lock:
            targets = self.offloading_targets if offloading_targets is None else np.asarray(offloading_targets)
            vals = []
            for s_idx, d_idx in enumerate(targets):
                if 0 <= int(d_idx) < self.num_devices:
                    vals.append(float(self.queue_lengths[s_idx, int(d_idx)]))
            return float(np.mean(vals)) if vals else 0.0

    def get_summary(self):
        with self.lock:
            return {
                "avg_delay": self.get_average_delay(),
                "arrival_rate": self._arrival_rate(),
                "complexity_mean": float(np.mean(self.service_complexity)) if self.service_complexity.size else 0.0,
                "queue_mean": float(np.mean(self.queue_lengths)) if self.queue_lengths.size else 0.0,
                "queue_max": float(np.max(self.queue_lengths)) if self.queue_lengths.size else 0.0,
                "runtime_mean": float(np.mean(self.runtime_ewma)) if self.runtime_ewma.size else 0.0,
                "runtime_confidence_mean": float(np.mean(np.clip(np.log1p(self.runtime_counts) / math.log1p(20.0), 0.0, 1.0))),
                "model_flops_mean": float(np.mean(self.model_flops)) if self.model_flops.size else 0.0,
                "model_memory_mean": float(np.mean(self.model_memory)) if self.model_memory.size else 0.0,
                "gpu_flops_mean": float(np.mean(self.gpu_flops)) if self.gpu_flops.size else 0.0,
                "memory_capacity_mean": float(np.mean(self.memory_capacity)) if self.memory_capacity.size else 0.0,
                "memory_overage": self.get_memory_overage(),
            }

    def is_ready(self):
        return self.num_services > 0 and self.num_devices > 0
