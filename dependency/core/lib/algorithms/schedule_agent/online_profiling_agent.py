import abc
import random
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from core.lib.common import ClassFactory, ClassType, Context, ConfigLoader, TaskConstant, LOGGER
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent

__all__ = ('OnlineProfilingAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='online_profiling')
class OnlineProfilingAgent(BaseAgent, abc.ABC):
    """
    Online Profiling Agent that selects execution devices based on bandwidth and online profiled latency data.
    - If bandwidth > threshold n, all stages execute on cloud
    - If bandwidth <= threshold n, probabilistically select edge devices using weighted latency
      (latency * service_importance_weight; lower effective latency -> higher probability)
    - Resource `queue_length` per device: if > 5, that device's relative score is halved before
      probabilities are re-normalized.
    - Updates latency profile every x minutes based on actual execution data
    - For services in ``per_obj_services``, recorded latency is
      real_execute_time / mean(obj_num) (aligned with LatencyMatrixAgent).
    - At each redeployment, the mean obj_num of each per_obj_service over the recent
      time window is used as that service's importance weight, so busier second-stage
      services are treated as more important in both offloading selection and deployment.
    """

    _OFFLOAD_QUEUE_LEN_HALVE_GT = 5
    _OFFLOAD_QUEUE_LEN_HALVE_FACTOR = 0.5

    _DEFAULT_PER_OBJ_SERVICES = (
        'exposure-identification',
        'category-identification',
        'license-plate-recognition',
    )

    def __init__(self, system, agent_id: int, configuration=None, bandwidth_threshold=None,
                 latency_profile=None, profile_update_interval=5, service_importance_weights=None,
                 per_obj_services=None):
        super().__init__(system, agent_id)

        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.system = system

        if configuration is None or isinstance(configuration, dict):
            self.default_configuration = configuration
        elif isinstance(configuration, str):
            self.default_configuration = ConfigLoader.load(Context.get_file_path(configuration))
        else:
            raise TypeError(f'Input "configuration" must be of type str or dict, get type {type(configuration)}')

        # Bandwidth threshold configuration.
        if bandwidth_threshold is None:
            self.bandwidth_threshold = 5.0  # Default value.
        elif isinstance(bandwidth_threshold, (int, float)):
            self.bandwidth_threshold = float(bandwidth_threshold)
        else:
            raise TypeError(f'Input "bandwidth_threshold" must be of type int or float, get type {type(bandwidth_threshold)}')

        # Initial latency observations collected offline and used as seeds.
        # Format: {service_name: {device_name: latency}}
        if latency_profile is None:
            self.latency_profile = {}
        elif isinstance(latency_profile, dict):
            self.latency_profile = latency_profile.copy()
        elif isinstance(latency_profile, str):
            self.latency_profile = ConfigLoader.load(Context.get_file_path(latency_profile))
        else:
            raise TypeError(f'Input "latency_profile" must be of type str or dict, get type {type(latency_profile)}')

        if service_importance_weights is None:
            self.service_importance_weights = {}
        elif isinstance(service_importance_weights, dict):
            self.service_importance_weights = {str(k): float(v) for k, v in service_importance_weights.items()}
        elif isinstance(service_importance_weights, str):
            self.service_importance_weights = {
                str(k): float(v) for k, v in ConfigLoader.load(Context.get_file_path(service_importance_weights)).items()
            }
        else:
            raise TypeError(
                f'Input "service_importance_weights" must be of type str or dict, get type {type(service_importance_weights)}'
            )

        if per_obj_services is None:
            self.per_obj_services = frozenset(self._DEFAULT_PER_OBJ_SERVICES)
        elif isinstance(per_obj_services, (list, tuple, set, frozenset)):
            self.per_obj_services = frozenset(str(s) for s in per_obj_services)
        else:
            raise TypeError(
                f'Input "per_obj_services" must be a list/tuple/set or None, get type {type(per_obj_services)}'
            )

        # Profiling update interval in minutes.
        if isinstance(profile_update_interval, (int, float)):
            self.profile_update_interval = float(profile_update_interval)
        else:
            raise TypeError(f'Input "profile_update_interval" must be of type int or float, get type {type(profile_update_interval)}')

        # Data structure that stores online execution latency samples.
        # {service_name: {device_name: [(timestamp, latency), ...]}}
        self.execution_records = defaultdict(lambda: defaultdict(deque))

        # Samples of object counts for per-object services, used to refresh
        # importance weights.
        # {service_name: deque([(timestamp, avg_obj_num), ...])}
        self.obj_num_records = defaultdict(deque)

        # Lock used for thread-safe profile updates.
        self.profile_lock = threading.Lock()

        # Last profile update timestamp.
        self.last_update_time = datetime.now()

        self.latest_offloading_policy = {}  # Latest offloading policy snapshot for redeployment.
        self.overhead_estimator = OverheadEstimator('OnlineProfiling', 'scheduler/online_profiling', agent_id=self.agent_id)

        LOGGER.info(f'[Online Profiling Agent] Initialized with bandwidth threshold: {self.bandwidth_threshold}')
        LOGGER.info(f'[Online Profiling Agent] Initial latency profile: {self.latency_profile}')
        LOGGER.info(f'[Online Profiling Agent] Service importance weights: {self.service_importance_weights}')
        LOGGER.info(f'[Online Profiling Agent] Per-object latency services: {sorted(self.per_obj_services)}')
        LOGGER.info(f'[Online Profiling Agent] Profile update interval: {self.profile_update_interval} minutes')

    def _extract_avg_obj_num(self, service) -> Optional[float]:
        """Extract mean(obj_num) from scenario_data and return None if invalid."""
        scenario = service.get_scenario_data()
        if not isinstance(scenario, dict):
            return None
        obj_num_raw = scenario.get('obj_num')
        if obj_num_raw is None:
            return None
        try:
            arr = np.asarray(obj_num_raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if arr.size == 0:
            return None
        avg = float(np.mean(arr))
        return avg if avg > 0 else None

    def _latency_for_profile(self, service_name: str, service) -> Optional[float]:
        """
        Latency value written into the online profile.

        Normal services use ``real_execute_time`` directly.
        Per-object services use ``real_execute_time / mean(obj_num)`` to stay
        aligned with ``LatencyMatrixAgent``.
        Return ``None`` when normalization is impossible so the sample is
        skipped.
        """
        real_exe_time = service.get_real_execute_time()
        if real_exe_time <= 0:
            return None

        if service_name not in self.per_obj_services:
            return float(real_exe_time)

        avg_obj_num = self._extract_avg_obj_num(service)
        if avg_obj_num is None:
            LOGGER.debug(f'[Online Profiling Agent] Skip per-obj norm for {service_name}: invalid obj_num')
            return None

        return float(real_exe_time / avg_obj_num)

    def _importance_weight(self, service_name: str) -> float:
        w = self.service_importance_weights.get(str(service_name), 1.0)
        return float(w) if w > 0 else 1.0

    @staticmethod
    def _resource_queue_length(resource, service_name: str) -> float:
        if not isinstance(resource, dict):
            return 0.0
        ql = resource.get('queue_length')
        if ql is None:
            return 0.0
        if isinstance(ql, (int, float)):
            return float(ql)
        if isinstance(ql, dict):
            if service_name in ql:
                try:
                    return float(ql[service_name])
                except (TypeError, ValueError):
                    pass
            vals = []
            for v in ql.values():
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            return max(vals) if vals else 0.0
        return 0.0

    def _device_queue_length(self, device: str, service_name: str) -> float:
        resource_table = self.system.get_scheduler_resource()
        if not resource_table or device not in resource_table:
            return 0.0
        return self._resource_queue_length(resource_table[device], str(service_name))

    def get_bandwidth(self, source_device=None):
        """Read the current bandwidth value from the scheduler resource table."""
        resource_table = self.system.get_scheduler_resource()
        if not resource_table:
            return None

        # Prefer the source device if its bandwidth is available.
        if source_device and source_device in resource_table:
            resource = resource_table[source_device]
            if isinstance(resource, dict) and 'available_bandwidth' in resource:
                bandwidth = resource['available_bandwidth']
                if bandwidth != -1 and bandwidth != 0:
                    return bandwidth

        # Otherwise scan devices and return the first valid bandwidth value.
        for device, resource in resource_table.items():
            if isinstance(resource, dict) and 'available_bandwidth' in resource:
                bandwidth = resource['available_bandwidth']
                if bandwidth != -1 and bandwidth != 0:
                    return bandwidth
        return None

    def get_current_deployment(self):
        """
        Get the current service deployment view used by the scheduler.

        The latest redeployment result is preferred. Before the first
        redeployment round, fall back to the fixed initial deployment policy
        when that policy is available.
        """
        if hasattr(self.redeployment_policy, 'policy') and self.redeployment_policy.policy:
            return self.redeployment_policy.policy

        fixed_policy = getattr(self.initial_deployment_policy, 'fixed_policy', None)
        if isinstance(fixed_policy, dict) and fixed_policy:
            return fixed_policy

        return {}

    @staticmethod
    def _normalize_deployed_devices(raw_devices):
        if raw_devices is None:
            return []
        if isinstance(raw_devices, str):
            return [raw_devices]
        if isinstance(raw_devices, (list, tuple, set, frozenset)):
            return [str(device) for device in raw_devices]
        return []

    def _get_executable_deployed_devices(self, service_name, deployment_view, all_edge_devices):
        """
        Return the executable edge devices for a service in scheduler order.

        A device is considered executable only if:
          1. The current deployment view says the service is deployed there.
          2. The device is still part of the current edge-device set.
        """
        edge_devices = [str(device) for device in (all_edge_devices or [])]
        if not edge_devices:
            return []

        deployed = set(self._normalize_deployed_devices(deployment_view.get(service_name, [])))
        if not deployed:
            return []

        return [device for device in edge_devices if device in deployed]

    def record_execution(self, service_name, device_name, latency):
        """
        Record a service execution latency sample on a device.

        Args:
            service_name: Service name.
            device_name: Device name.
            latency: Execution latency in seconds.
        """
        if service_name == TaskConstant.START.value or service_name == TaskConstant.END.value:
            return

        with self.profile_lock:
            current_time = datetime.now()
            self.execution_records[service_name][device_name].append((current_time, latency))
            LOGGER.debug(f'[Online Profiling Agent] Recorded execution: service={service_name}, '
                        f'device={device_name}, latency={latency:.4f}s')

    def update_latency_profile(self):
        """
        Refresh the latency profile using recent execution records.

        Only service-device pairs with samples inside the configured time
        window are updated.
        """
        with self.profile_lock:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(minutes=self.profile_update_interval)

            updated_count = 0

            # Walk through every recorded service-device pair.
            for service_name in self.execution_records:
                if service_name not in self.latency_profile:
                    self.latency_profile[service_name] = {}

                for device_name in self.execution_records[service_name]:
                    records = self.execution_records[service_name][device_name]

                    # Keep only samples that fall inside the active time window.
                    valid_records = [(ts, lat) for ts, lat in records if ts >= cutoff_time]

                    if valid_records:
                        # Compute the average latency over valid samples.
                        avg_latency = sum(lat for _, lat in valid_records) / len(valid_records)

                        # Update the latency profile entry.
                        old_latency = self.latency_profile[service_name].get(device_name, None)
                        self.latency_profile[service_name][device_name] = avg_latency
                        updated_count += 1

                        LOGGER.info(f'[Online Profiling Agent] Updated latency: service={service_name}, '
                                   f'device={device_name}, old={old_latency}, new={avg_latency:.4f}s, '
                                   f'samples={len(valid_records)}')

                    # Drop stale samples outside the time window.
                    self.execution_records[service_name][device_name] = deque(valid_records)

            self.last_update_time = current_time
            LOGGER.info(f'[Online Profiling Agent] Profile update completed. Updated {updated_count} entries.')
            LOGGER.info(f'[Online Profiling Agent] Current latency profile: {self.latency_profile}')

    def select_device_by_latency(self, service_name, deployed_devices):
        """
        Select a device probabilistically using weighted execution cost.

        The score is based on ``latency * importance_weight`` and then reduced
        for devices whose queue length exceeds the threshold before
        renormalization.

        Args:
            service_name: Service name.
            deployed_devices: Devices where the service is currently deployed.

        Returns:
            The selected device.
        """
        if not deployed_devices:
            LOGGER.warning(f'[Online Profiling Agent] Service {service_name} has no deployed devices, using cloud')
            return self.cloud_device

        # Read the latency profile under the lock.
        with self.profile_lock:
            latency_profile_copy = self.latency_profile.copy()

        # Fall back to a random deployed device when the service has no profile.
        if service_name not in latency_profile_copy:
            LOGGER.warning(f'[Online Profiling Agent] Service {service_name} has no latency profile, random selection')
            return random.choice(deployed_devices)

        service_latency = latency_profile_copy[service_name]

        # Collect latency entries for the currently deployed devices.
        device_latencies = {}
        for device in deployed_devices:
            if device in service_latency:
                device_latencies[device] = service_latency[device]
            else:
                LOGGER.warning(f'[Online Profiling Agent] Device {device} has no latency data for service {service_name}')

        # Fall back again if none of the deployed devices has latency data.
        if not device_latencies:
            LOGGER.warning(f'[Online Profiling Agent] No latency data for any deployed device, random selection')
            return random.choice(deployed_devices)

        weight = self._importance_weight(service_name)
        base_scores = {}
        for device, latency in device_latencies.items():
            cost = (latency * weight) if latency > 0 else 1e-9
            cost = max(cost, 1e-12)
            base_scores[device] = 1.0 / cost

        adjusted_scores = {}
        for device, base in base_scores.items():
            ql = self._device_queue_length(device, service_name)
            factor = (
                self._OFFLOAD_QUEUE_LEN_HALVE_FACTOR
                if ql > self._OFFLOAD_QUEUE_LEN_HALVE_GT
                else 1.0
            )
            adjusted_scores[device] = base * factor

        total_score = sum(adjusted_scores.values())
        if total_score == 0:
            return random.choice(list(device_latencies.keys()))

        probabilities = {device: s / total_score for device, s in adjusted_scores.items()}

        # Sample a device according to the normalized probabilities.
        devices = list(probabilities.keys())
        probs = list(probabilities.values())
        selected_device = random.choices(devices, weights=probs, k=1)[0]

        LOGGER.debug(f'[Online Profiling Agent] Service {service_name} selection probabilities: {probabilities}, selected: {selected_device}')

        return selected_device

    def get_schedule_plan(self, info):
        if self.default_configuration is None:
            return None

        with self.overhead_estimator:
            configuration = self.default_configuration.copy()
            policy = {}
            policy.update(configuration)

            cloud_device = self.cloud_device
            source_edge_device = info['source_device']
            all_edge_devices = info['all_edge_devices']
            dag = info['dag']

            # Get the current bandwidth.
            bandwidth = self.get_bandwidth(source_device=source_edge_device)
            LOGGER.info(f'[Online Profiling Agent] Current bandwidth: {bandwidth}, threshold: {self.bandwidth_threshold}')

            # Get the current service deployment snapshot.
            current_deployment = self.get_current_deployment()
            LOGGER.info(f'[Online Profiling Agent] Current deployment: {current_deployment}')

            # Build the offloading policy.
            offloading_policy = {}

            # Decide the execution device service by service.
            for service_name in dag:
                # Keep the terminal node on the cloud device.
                if service_name == TaskConstant.END.value:
                    dag[service_name]['service']['execute_device'] = cloud_device
                    offloading_policy[service_name] = cloud_device
                    continue

                if service_name == TaskConstant.START.value:
                    # Keep the start node on the source device.
                    dag[service_name]['service']['execute_device'] = source_edge_device
                    offloading_policy[service_name] = source_edge_device
                    continue

                # Send all stages to cloud when bandwidth is above the threshold.
                if bandwidth is not None and bandwidth > self.bandwidth_threshold:
                    execute_device = cloud_device
                else:
                    # Select only from devices where the service is both
                    # deployed and currently executable.
                    deployed_devices = self._get_executable_deployed_devices(
                        service_name=service_name,
                        deployment_view=current_deployment,
                        all_edge_devices=all_edge_devices,
                    )

                    if deployed_devices:
                        execute_device = self.select_device_by_latency(service_name, deployed_devices)
                    else:
                        execute_device = cloud_device
                        LOGGER.warning(
                            f'[Online Profiling Agent] Service {service_name} has no deployed executable '
                            f'edge device, falling back to cloud'
                        )

                dag[service_name]['service']['execute_device'] = execute_device
                offloading_policy[service_name] = execute_device

            # Keep the latest offloading policy for later inspection/use.
            self.latest_offloading_policy = offloading_policy.copy()
            LOGGER.info(f'[Online Profiling Agent] Latest offloading policy: {offloading_policy}')

            policy.update({'dag': dag})
        return policy

    def get_latest_offloading_policy(self):
        """Return the latest offloading policy snapshot."""
        return self.latest_offloading_policy.copy()

    def get_current_importance_weights(self) -> dict:
        """Return a read-only snapshot of current service importance weights."""
        with self.profile_lock:
            return dict(self.service_importance_weights)

    def _update_importance_weights(self):
        """
        Update service importance weights using recent per-object samples.

        The background thread calls this together with
        ``update_latency_profile()``.
        - For services with valid samples, the new weight is the mean
          ``avg_obj_num`` inside the time window.
        - For services without valid samples, keep the previous weight.
        """
        with self.profile_lock:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(minutes=self.profile_update_interval)

            updated_weights = dict(self.service_importance_weights)
            updated_count = 0

            for service_name in self.per_obj_services:
                records = self.obj_num_records.get(service_name)
                if not records:
                    continue

                valid = [(ts, v) for ts, v in records if ts >= cutoff_time]
                # Remove samples outside the active window.
                self.obj_num_records[service_name] = deque(valid)

                if not valid:
                    continue

                avg_weight = float(np.mean([v for _, v in valid]))
                if avg_weight > 0:
                    old_w = updated_weights.get(service_name, 1.0)
                    updated_weights[service_name] = avg_weight
                    updated_count += 1
                    LOGGER.info(
                        f'[Online Profiling Agent] Importance weight updated: '
                        f'{service_name}: {old_w:.4f} -> {avg_weight:.4f} '
                        f'(samples={len(valid)})'
                    )

            self.service_importance_weights = updated_weights
            LOGGER.info(
                f'[Online Profiling Agent] Importance weights update completed. '
                f'Updated {updated_count} entries. '
                f'Current weights: {self.service_importance_weights}'
            )

    def run(self):
        """
        Background thread that periodically refreshes:
          1. ``latency_profile`` from execution-latency samples
          2. ``service_importance_weights`` from per-object count samples
        """
        LOGGER.info(f'[Online Profiling Agent] Background profiling update thread started')

        while True:
            try:
                time.sleep(self.profile_update_interval * 60)

                self.update_latency_profile()
                self._update_importance_weights()

            except Exception as e:
                LOGGER.error(f'[Online Profiling Agent] Error in profiling update thread: {e}')
                LOGGER.exception(e)

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        """
        Extract execution data from a finished task and record it.

        Args:
            task: Finished task object.
        """
        try:
            # Get the task DAG.
            dag = task.get_dag()
            if not dag:
                return

            # Walk through all service nodes in the DAG.
            for service_name in dag.nodes:
                service = dag.get_node(service_name).service

                # Record service execution device; per-object services use
                # real_exe / mean(obj_num) as the profile sample.
                device_name = service.get_execute_device()
                latency_val = self._latency_for_profile(service_name, service)

                if (service_name != TaskConstant.START.value and
                        service_name != TaskConstant.END.value and
                        latency_val is not None and
                        device_name):
                    self.record_execution(service_name, device_name, latency_val)

                # Per-object services also record avg_obj_num so the importance
                # weights can be refreshed later.
                if service_name in self.per_obj_services:
                    avg_obj = self._extract_avg_obj_num(service)
                    if avg_obj is not None:
                        with self.profile_lock:
                            self.obj_num_records[service_name].append(
                                (datetime.now(), avg_obj)
                            )
                        LOGGER.debug(
                            f'[Online Profiling Agent] Recorded obj_num: '
                            f'service={service_name}, avg_obj_num={avg_obj:.2f}'
                        )

        except Exception as e:
            LOGGER.error(f'[Online Profiling Agent] Error updating task execution records: {e}')
            LOGGER.exception(e)

    def get_schedule_overhead(self):
        offloading_overhead = self.overhead_estimator.get_latest_overhead()
        redeployment_overhead = 0.0
        if hasattr(self.redeployment_policy, 'get_redeployment_overhead'):
            redeployment_overhead = self.redeployment_policy.get_redeployment_overhead()
        return offloading_overhead + redeployment_overhead
