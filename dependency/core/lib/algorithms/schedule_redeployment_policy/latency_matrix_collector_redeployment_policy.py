import abc

from .base_redeployment_policy import BaseRedeploymentPolicy
from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('LatencyMatrixCollectorRedeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='latency_matrix_collector')
class LatencyMatrixCollectorRedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    """
    Redeployment policy for latency matrix profiling.

    At each scheduling round it reads the set of already-finalised
    (service, device) pairs from the companion LatencyMatrixAgent and
    computes the next deployment plan that maximises coverage of the
    remaining unmeasured pairs, subject to:

        • each device hosts at most device_service_limits[device] services
        • every service that exists in the task DAG must be deployed on at
          least one device (so the agent always has somewhere to send tasks)

    Once all N×M cells are measured the policy continues to return a stable
    deployment (each service on one device) so that the system keeps running.
    Deployment switching is gated: once a plan is active it will not be
    replaced until every (service, device) pair in that plan has been
    finalised (reached min_samples).  This prevents the system from
    churning through deployments before useful measurements accumulate.

    Parameters (passed via SCH_REDEPLOYMENT_POLICY_PARAMETERS)
    ----------------------------------------------------------
    services : list[str]
        Ordered list of service names to profile.
    devices : list[str]
        Ordered list of edge device names to profile on
        (cloud is excluded automatically).
    device_service_limits : dict[str, int]
        Maximum number of services each device may host simultaneously.
        Missing devices fall back to ``default_device_limit``.
    default_device_limit : int  (default 2)
        Fallback limit for devices not listed in device_service_limits.
    """

    def __init__(self, system, agent_id,
                 services=None,
                 devices=None,
                 device_service_limits=None,
                 default_device_limit=2,
                 **kwargs):
        self.system = system
        self.agent_id = agent_id
        self.policy = None

        # --- services ---
        if services is None:
            raise ValueError('[Latency Matrix Redeployment] "services" parameter is required')
        if not isinstance(services, list) or not services:
            raise TypeError(f'"services" must be a non-empty list, got {type(services)}')
        self.services = [str(s) for s in services]

        # --- devices ---
        if devices is None:
            raise ValueError('[Latency Matrix Redeployment] "devices" parameter is required')
        if not isinstance(devices, list) or not devices:
            raise TypeError(f'"devices" must be a non-empty list, got {type(devices)}')
        self.devices = [str(d) for d in devices]

        # --- per-device service limits ---
        try:
            self.default_device_limit = max(1, int(default_device_limit))
        except (TypeError, ValueError):
            raise TypeError(
                f'"default_device_limit" must be int, got {type(default_device_limit)}')

        if device_service_limits is None:
            self.device_service_limits = {}
        elif isinstance(device_service_limits, dict):
            self.device_service_limits = {
                str(dev): max(1, int(limit))
                for dev, limit in device_service_limits.items()
            }
        else:
            raise TypeError(
                f'"device_service_limits" must be a dict, got {type(device_service_limits)}')

        total_cells = len(self.services) * len(self.devices)
        LOGGER.info(
            f'[Latency Matrix Redeployment] Initialized. '
            f'services({len(self.services)})={self.services}, '
            f'devices({len(self.devices)})={self.devices}, '
            f'limits={self._limits_summary()}, '
            f'total_cells={total_cells}')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _limits_summary(self) -> dict:
        return {dev: self._get_limit(dev) for dev in self.devices}

    def _get_limit(self, device: str) -> int:
        return self.device_service_limits.get(device, self.default_device_limit)

    def _all_current_pairs_measured(self, measured_matrix: dict) -> bool:
        """
        Return True only when every (service, device) pair that is listed in
        the current deployment plan has been finalised in measured_matrix.

        An empty / None plan is treated as "not yet started", which triggers
        the very first deployment computation.
        """
        if not self.policy:
            return True
        for svc, devs in self.policy.items():
            for dev in devs:
                if dev not in measured_matrix.get(svc, {}):
                    return False
        return True

    def _get_measured_matrix(self) -> dict:
        """
        Retrieve the finalised latency matrix from the agent.
        Returns {service_name: {device_name: latency}}.
        """
        try:
            if (hasattr(self.system, 'schedule_table') and
                    self.agent_id in self.system.schedule_table):
                agent = self.system.schedule_table[self.agent_id]
                if hasattr(agent, 'get_measured_matrix'):
                    return agent.get_measured_matrix()
        except Exception as exc:
            LOGGER.warning(
                f'[Latency Matrix Redeployment] '
                f'Could not read measured matrix from agent: {exc}')
        return {}

    def _compute_deployment(self, measured_matrix: dict, dag_services: list) -> dict:
        """
        Greedily assign services to devices to cover unmeasured pairs.

        Algorithm
        ---------
        1. Collect all unmeasured (service, device) pairs restricted to
           services that are present in the current DAG.
        2. Sort services by the number of unmeasured devices they still have
           (fewest first — most constrained first).
        3. For each service, assign it to the first unmeasured device that
           still has a free slot, consuming one slot on that device.
        4. Second pass: any service that was not assigned in step 3 (all its
           pairs are already measured) gets a fallback assignment to any
           device with a remaining free slot.
        5. If no free slots remain, the service is assigned to the
           least-loaded device (overflow warning logged).

        Returns
        -------
        {service_name: [device_name, ...]}
        """
        # Restrict to services that actually appear in the current DAG
        services = [s for s in self.services if s in dag_services]

        # Initialise per-device remaining slots using per-device limits
        device_slots = {dev: self._get_limit(dev) for dev in self.devices}
        deploy_plan = {svc: [] for svc in services}

        def unmeasured_devices(svc):
            measured = measured_matrix.get(svc, {})
            return [d for d in self.devices if d not in measured]

        # Sort: services with fewer unmeasured devices first (most constrained)
        sorted_services = sorted(services, key=lambda s: len(unmeasured_devices(s)))

        unassigned = []
        for svc in sorted_services:
            candidates = [
                d for d in unmeasured_devices(svc)
                if device_slots[d] > 0
            ]
            if candidates:
                chosen = candidates[0]
                deploy_plan[svc].append(chosen)
                device_slots[chosen] -= 1
                LOGGER.debug(
                    f'[Latency Matrix Redeployment] '
                    f'Profiling assignment: {svc} → {chosen} '
                    f'(remaining slot: {device_slots[chosen]}/{self._get_limit(chosen)})')
            else:
                unassigned.append(svc)

        # Second pass: services whose all pairs are measured (or no free slot above)
        for svc in unassigned:
            assigned = False
            for dev in self.devices:
                if device_slots[dev] > 0:
                    deploy_plan[svc].append(dev)
                    device_slots[dev] -= 1
                    LOGGER.info(
                        f'[Latency Matrix Redeployment] '
                        f'Fallback assignment (all pairs measured): {svc} → {dev}')
                    assigned = True
                    break
            if not assigned:
                # No free slots at all — pick the device with the most total capacity
                # (least relative load)
                least = min(
                    self.devices,
                    key=lambda d: self._get_limit(d) - device_slots[d])
                deploy_plan[svc].append(least)
                LOGGER.warning(
                    f'[Latency Matrix Redeployment] '
                    f'Overflow assignment (no free slots): {svc} → {least}')

        total_cells = len(self.services) * len(self.devices)
        unmeasured_count = sum(
            1 for s in self.services for d in self.devices
            if d not in measured_matrix.get(s, {}))
        LOGGER.info(
            f'[Latency Matrix Redeployment] '
            f'Unmeasured pairs remaining: {unmeasured_count}/{total_cells}')
        LOGGER.info(
            f'[Latency Matrix Redeployment] Deploy plan: {deploy_plan}')
        LOGGER.info(
            f'[Latency Matrix Redeployment] Remaining device slots: {device_slots}')

        return deploy_plan

    # ------------------------------------------------------------------
    # BaseRedeploymentPolicy interface
    # ------------------------------------------------------------------

    def __call__(self, info) -> dict:
        source_id = info['source']['id']
        dag = info['dag']

        # Services present in this DAG (excluding _start / _end)
        dag_services = [
            svc for svc in dag
            if svc not in ('_start', '_end')
        ]

        measured_matrix = self._get_measured_matrix()

        # Keep the current plan until every pair in it is fully measured.
        if not self._all_current_pairs_measured(measured_matrix):
            pending = [
                (svc, dev)
                for svc, devs in self.policy.items()
                for dev in devs
                if dev not in measured_matrix.get(svc, {})
            ]
            LOGGER.debug(
                f'[Latency Matrix Redeployment] '
                f'(source {source_id}) Waiting for {len(pending)} pair(s): '
                f'{pending}  — keeping current plan.')
            return self.policy

        deploy_plan = self._compute_deployment(measured_matrix, dag_services)

        LOGGER.info(
            f'[Latency Matrix Redeployment] '
            f'(source {source_id}) New policy: {deploy_plan}')

        self.policy = deploy_plan
        return deploy_plan
