import abc
import random
import threading
from collections import defaultdict

import numpy as np

from core.lib.common import ClassFactory, ClassType, Context, ConfigLoader, TaskConstant, LOGGER
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent

__all__ = ('LatencyMatrixCollectorAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='latency_matrix_collector')
class LatencyMatrixCollectorAgent(BaseAgent, abc.ABC):
    """
    Latency Matrix Agent.

    Profiles the real execution latency for every (service, device) pair and
    builds an N×M matrix where N = len(services), M = len(devices).

    For services listed in ``per_obj_services`` the recorded value is
    real_exe_time / mean(obj_num) (per-object processing time).

    Cloud devices are never tested.  Profiling continues until every cell has
    accumulated at least ``min_samples`` valid measurements.

    Parameters (passed via SCH_AGENT_PARAMETERS)
    --------------------------------------------
    configuration : dict | str | None
        Video/encoding configuration forwarded to the schedule plan.
    services : list[str]
        Service names that form the matrix rows.
    devices : list[str]
        Edge device names that form the matrix columns (cloud excluded).
    per_obj_services : list[str]
        Subset of services for which latency is divided by mean(obj_num).
    min_samples : int  (default 3)
        Minimum valid samples required before a cell is finalised.
    """

    def __init__(self, system, agent_id: int,
                 configuration=None,
                 services=None,
                 devices=None,
                 per_obj_services=None,
                 min_samples=3):
        super().__init__(system, agent_id)

        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.system = system

        # --- video configuration ---
        if configuration is None or isinstance(configuration, dict):
            self.default_configuration = configuration
        elif isinstance(configuration, str):
            self.default_configuration = ConfigLoader.load(
                Context.get_file_path(configuration))
        else:
            raise TypeError(
                f'configuration must be dict or str, got {type(configuration)}')

        # --- services ---
        if services is None:
            raise ValueError('[Latency Matrix Agent] "services" parameter is required')
        if not isinstance(services, list) or not services:
            raise TypeError(f'"services" must be a non-empty list, got {type(services)}')
        self.services = [str(s) for s in services]

        # --- devices ---
        if devices is None:
            raise ValueError('[Latency Matrix Agent] "devices" parameter is required')
        if not isinstance(devices, list) or not devices:
            raise TypeError(f'"devices" must be a non-empty list, got {type(devices)}')
        self.devices = [str(d) for d in devices]

        # --- per-object services ---
        if per_obj_services is None:
            self.per_obj_services = frozenset()
        elif isinstance(per_obj_services, (list, tuple, set, frozenset)):
            self.per_obj_services = frozenset(str(s) for s in per_obj_services)
        else:
            raise TypeError(
                f'"per_obj_services" must be a list/set, got {type(per_obj_services)}')

        # --- min_samples ---
        try:
            self.min_samples = max(1, int(min_samples))
        except (TypeError, ValueError):
            raise TypeError(f'min_samples must be int, got {type(min_samples)}')

        # raw latency samples: {service: {device: [latency, ...]}}
        self._sample_buffer = defaultdict(lambda: defaultdict(list))

        # finalised averages: {service: {device: float}}
        self._measured_matrix = defaultdict(dict)

        self._matrix_complete = False
        self._lock = threading.Lock()

        self.overhead_estimator = OverheadEstimator(
            'LatencyMatrix', 'scheduler/latency_matrix',
            agent_id=self.agent_id)

        total_cells = len(self.services) * len(self.devices)
        LOGGER.info(
            f'[Latency Matrix Agent] Initialized. '
            f'min_samples={self.min_samples}, '
            f'services({len(self.services)})={self.services}, '
            f'devices({len(self.devices)})={self.devices}, '
            f'per_obj_services={sorted(self.per_obj_services)}, '
            f'total_cells={total_cells}')

    # ------------------------------------------------------------------
    # Public helpers (used by the redeployment policy)
    # ------------------------------------------------------------------

    def get_measured_matrix(self) -> dict:
        """Return a snapshot of the finalised latency matrix (thread-safe)."""
        with self._lock:
            return {svc: dict(devs)
                    for svc, devs in self._measured_matrix.items()}

    def is_matrix_complete(self) -> bool:
        with self._lock:
            return self._matrix_complete

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_latency(self, service_name, service):
        """
        Return the latency value to record for this service execution.

        For per_obj_services: real_exe_time / mean(obj_num).
        Returns None when the value is invalid or indeterminate.
        """
        real_exe_time = service.get_real_execute_time()
        if real_exe_time <= 0:
            return None

        if service_name not in self.per_obj_services:
            return real_exe_time

        # Per-object services: need obj_num from scenario data
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

        avg_obj_num = float(np.mean(arr))
        if avg_obj_num <= 0:
            return None

        return real_exe_time / avg_obj_num

    def _unmeasured_pairs(self) -> list:
        """Return [(service, device), ...] for cells not yet finalised."""
        pairs = []
        for svc in self.services:
            measured_devs = self._measured_matrix.get(svc, {})
            for dev in self.devices:
                if dev not in measured_devs:
                    pairs.append((svc, dev))
        return pairs

    def _log_matrix(self):
        """Write the current matrix state to the scheduler log."""
        col_w = 14
        svc_w = 32
        header = f"{'service':<{svc_w}}" + ''.join(
            f'{d:<{col_w}}' for d in self.devices)
        rows = [
            '',
            '[Latency Matrix Agent] ===== LATENCY MATRIX (real_exe_time, seconds) =====',
            header,
            '-' * (svc_w + col_w * len(self.devices)),
        ]
        for svc in self.services:
            label = svc + (' (per-obj)' if svc in self.per_obj_services else '')
            row = f'{label:<{svc_w}}'
            for dev in self.devices:
                val = self._measured_matrix.get(svc, {}).get(dev)
                row += f"{'N/A':<{col_w}}" if val is None else f'{val:.6f}      '
            rows.append(row)

        measured = sum(
            1 for svc in self.services for dev in self.devices
            if dev in self._measured_matrix.get(svc, {}))
        total = len(self.services) * len(self.devices)
        rows.append(f'Progress: {measured}/{total}')
        rows.append('=' * (svc_w + col_w * len(self.devices)))
        LOGGER.info('\n'.join(rows))

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def get_schedule_plan(self, info):
        if self.default_configuration is None:
            return None

        with self.overhead_estimator:
            policy = dict(self.default_configuration)

            cloud_device = self.cloud_device
            source_edge_device = info['source_device']
            dag = info['dag']

            # Snapshot of current deployment from redeployment policy
            current_deployment: dict = {}
            if (hasattr(self.redeployment_policy, 'policy') and
                    self.redeployment_policy.policy):
                current_deployment = self.redeployment_policy.policy

            with self._lock:
                measured_snapshot = {
                    svc: set(devs.keys())
                    for svc, devs in self._measured_matrix.items()
                }

            for service_name in dag:
                if service_name == TaskConstant.END.value:
                    dag[service_name]['service']['execute_device'] = cloud_device
                    continue

                if service_name == TaskConstant.START.value:
                    dag[service_name]['service']['execute_device'] = source_edge_device
                    continue

                deployed_devices = current_deployment.get(service_name, [])

                # Prefer devices that haven't been measured yet for this service
                already_measured = measured_snapshot.get(service_name, set())
                unmeasured_deployed = [
                    d for d in deployed_devices if d not in already_measured]

                if unmeasured_deployed:
                    execute_device = random.choice(unmeasured_deployed)
                elif deployed_devices:
                    execute_device = random.choice(deployed_devices)
                else:
                    execute_device = cloud_device

                dag[service_name]['service']['execute_device'] = execute_device

            policy['dag'] = dag
        return policy

    def update_task(self, task):
        """
        Called after each task completes.  Extracts latency measurements from
        the task's DAG and accumulates them.  Marks a (service, device) cell
        as finalised once min_samples valid samples have been collected.
        """
        try:
            dag = task.get_dag()
            if not dag:
                return

            newly_finalised = []

            with self._lock:
                for service_name, node in dag.nodes.items():
                    if service_name in (TaskConstant.START.value,
                                        TaskConstant.END.value):
                        continue

                    service = node.service
                    device_name = service.get_execute_device()

                    if not device_name or device_name == self.cloud_device:
                        continue

                    if (service_name not in self.services or
                            device_name not in self.devices):
                        continue

                    # Skip if already finalised
                    if device_name in self._measured_matrix.get(service_name, {}):
                        continue

                    latency = self._extract_latency(service_name, service)
                    if latency is None:
                        continue

                    self._sample_buffer[service_name][device_name].append(latency)
                    samples = self._sample_buffer[service_name][device_name]

                    LOGGER.debug(
                        f'[Latency Matrix Agent] Sample recorded: '
                        f'service={service_name}, device={device_name}, '
                        f'latency={latency:.6f}s '
                        f'({"per-obj " if service_name in self.per_obj_services else ""}'
                        f'count={len(samples)}/{self.min_samples})')

                    if len(samples) >= self.min_samples:
                        avg = float(np.mean(samples))
                        self._measured_matrix[service_name][device_name] = avg
                        newly_finalised.append((service_name, device_name, avg))

                if newly_finalised:
                    for svc, dev, avg in newly_finalised:
                        label = 'per-obj ' if svc in self.per_obj_services else ''
                        LOGGER.info(
                            f'[Latency Matrix Agent] CELL FINALISED: '
                            f'service={svc}, device={dev}, '
                            f'avg_{label}latency={avg:.6f}s '
                            f'(samples={len(self._sample_buffer[svc][dev])})')

                    self._log_matrix()

                    total_cells = len(self.services) * len(self.devices)
                    if not self._unmeasured_pairs() and not self._matrix_complete:
                        self._matrix_complete = True
                        LOGGER.info(
                            f'[Latency Matrix Agent] '
                            f'*** ALL {total_cells} CELLS MEASURED — MATRIX COMPLETE ***')
                        self._log_matrix()

        except Exception as exc:
            LOGGER.error(f'[Latency Matrix Agent] Error in update_task: {exc}')
            LOGGER.exception(exc)

    def run(self):
        pass

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()
