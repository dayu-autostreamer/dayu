import abc
import copy

from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER, ConfigLoader, Context, TaskConstant
from core.lib.estimation import OverheadEstimator

__all__ = ('OnlineProfilingRedeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='online_profiling')
class OnlineProfilingRedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    """
    Online Profiling Redeployment Policy that deploys services greedily based on online profiled latency data.
    - Task complexity: mean(latency * service_importance_weight) across devices
    - Device capability: mean(latency * service_importance_weight) over services on that device (lower is better)
    - Greedy deployment: harder (higher weighted-cost) tasks to more capable devices
    - Constraints:
      - Each device has a maximum number of services it can host
      - Each service can be deployed on a maximum number of devices
      - No duplicate service on the same device
    - Uses online updated latency profile from the agent
    """

    def __init__(self, system, agent_id, latency_profile=None, device_service_limits=None,
                 service_replica_count=None, default_service_limit=None, default_replica_count=None,
                 service_importance_weights=None, **kwargs):

        self.system = system
        self.agent_id = agent_id

        # Initial latency observations collected offline and used as seeds.
        # Format: {service_name: {device_name: latency}}
        if latency_profile is None:
            self.initial_latency_profile = {}
        elif isinstance(latency_profile, dict):
            self.initial_latency_profile = latency_profile.copy()
        elif isinstance(latency_profile, str):
            self.initial_latency_profile = ConfigLoader.load(Context.get_file_path(latency_profile))
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

        # Per-device service-count limits.
        # Format: {device_name: max_service_count}
        if device_service_limits is None:
            self.device_service_limits = {}
        elif isinstance(device_service_limits, dict):
            self.device_service_limits = {str(device): int(limit) for device, limit in device_service_limits.items()}
        else:
            raise TypeError(f'Input "device_service_limits" must be of type dict, get type {type(device_service_limits)}')

        # Per-service replica/device-count limits.
        # Format: {service_name: max_device_count}
        if service_replica_count is None:
            self.service_replica_count = {}
        elif isinstance(service_replica_count, dict):
            self.service_replica_count = {str(service): int(count) for service, count in service_replica_count.items()}
        else:
            raise TypeError(f'Input "service_replica_count" must be of type dict, get type {type(service_replica_count)}')

        # Default service-count limit for each device.
        if default_service_limit is None:
            self.default_service_limit = 2
        elif isinstance(default_service_limit, (int, float)):
            self.default_service_limit = int(default_service_limit)
        else:
            raise TypeError(f'Input "default_service_limit" must be of type int or float, get type {type(default_service_limit)}')

        self.policy = None

        # Default replica-count limit for each service.
        if default_replica_count is None:
            self.default_replica_count = 2
        elif isinstance(default_replica_count, (int, float)):
            self.default_replica_count = int(default_replica_count)
        else:
            raise TypeError(f'Input "default_replica_count" must be of type int or float, get type {type(default_replica_count)}')

        LOGGER.info(f'[Online Profiling Redeployment] Initialized with initial latency profile: {self.initial_latency_profile}')
        LOGGER.info(f'[Online Profiling Redeployment] Service importance weights: {self.service_importance_weights}')
        LOGGER.info(f'[Online Profiling Redeployment] Device service limits: {self.device_service_limits}, default: {self.default_service_limit}')
        LOGGER.info(f'[Online Profiling Redeployment] Service replica count: {self.service_replica_count}, default: {self.default_replica_count}')
        self.overhead_estimator = OverheadEstimator(
            'OnlineProfilingRedeployment',
            'scheduler/online_profiling',
            agent_id=self.agent_id,
        )

    def _importance_weight(self, service_name: str) -> float:
        w = self.service_importance_weights.get(str(service_name), 1.0)
        return float(w) if w > 0 else 1.0

    def get_importance_weights_from_agent(self) -> dict:
        """
        Read the latest service-importance snapshot maintained by the agent.

        This is the read-only counterpart of ``get_latency_profile_from_agent``.
        If the lookup fails, fall back to the weights currently stored by this
        policy.
        """
        try:
            if hasattr(self.system, 'schedule_table') and self.agent_id in self.system.schedule_table:
                agent = self.system.schedule_table[self.agent_id]
                if hasattr(agent, 'get_current_importance_weights'):
                    weights = agent.get_current_importance_weights()
                    LOGGER.debug(
                        f'[Online Profiling Redeployment] Importance weights from agent: {weights}'
                    )
                    return weights
        except Exception as e:
            LOGGER.warning(
                f'[Online Profiling Redeployment] Failed to get importance weights from agent: {e}'
            )
        LOGGER.debug('[Online Profiling Redeployment] Using existing importance weights')
        return dict(self.service_importance_weights)

    def get_latency_profile_from_agent(self):
        """
        Read the online-updated latency profile from the scheduler agent.

        Fall back to the initial latency profile if the agent profile is not
        available.

        Returns:
            latency_profile: {service_name: {device_name: latency}}
        """
        try:
            # Read the matching agent from the system schedule table.
            if hasattr(self.system, 'schedule_table') and self.agent_id in self.system.schedule_table:
                agent = self.system.schedule_table[self.agent_id]
                if hasattr(agent, 'latency_profile'):
                    # Return the online-updated latency profile from the agent.
                    latency_profile = copy.deepcopy(agent.latency_profile)
                    LOGGER.debug(f'[Online Profiling Redeployment] Retrieved online latency profile from agent: {latency_profile}')
                    return latency_profile
        except Exception as e:
            LOGGER.warning(f'[Online Profiling Redeployment] Failed to get latency profile from agent: {e}')

        # Fall back to the initial latency profile when agent access fails.
        LOGGER.debug(f'[Online Profiling Redeployment] Using initial latency profile')
        return copy.deepcopy(self.initial_latency_profile)

    def calculate_task_complexity(self, latency_profile):
        """
        Compute service complexity as the average weighted latency across devices.

        Args:
            latency_profile: {service_name: {device_name: latency}}

        Returns:
            {service_name: complexity}
        """
        task_complexity = {}

        for service_name, device_latencies in latency_profile.items():
            if device_latencies:
                w = self._importance_weight(service_name)
                weighted = [lat * w for lat in device_latencies.values()]
                task_complexity[service_name] = sum(weighted) / len(weighted)
            else:
                task_complexity[service_name] = 0.0

        return task_complexity

    def calculate_device_capability(self, latency_profile, available_devices):
        """
        Compute device capability as the average weighted latency across services.

        Args:
            latency_profile: {service_name: {device_name: latency}}
            available_devices: List of available devices.

        Returns:
            {device_name: capability_score}
        """
        device_capability = {}

        for device in available_devices:
            latencies = []
            for service_name, device_latencies in latency_profile.items():
                if device in device_latencies:
                    w = self._importance_weight(service_name)
                    latencies.append(device_latencies[device] * w)

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                device_capability[device] = avg_latency
            else:
                # Use a large score when no data exists, meaning weaker capability.
                device_capability[device] = float('inf')

        return device_capability

    def get_device_service_limit(self, device_name):
        """Get the maximum number of services allowed on a device."""
        device_name = str(device_name)
        return self.device_service_limits.get(device_name, self.default_service_limit)

    def get_service_replica_count(self, service_name):
        """Get the maximum replica count allowed for a service."""
        service_name = str(service_name)
        return self.service_replica_count.get(service_name, self.default_replica_count)

    def greedy_deployment(self, latency_profile, dag, available_devices):
        """
        Greedy deployment: place harder services on stronger devices first.

        Args:
            latency_profile: {service_name: {device_name: latency}}
            dag: Task DAG.
            available_devices: Available edge devices.

        Returns:
            {service_name: [device1, device2, ...]}
        """
        # Compute service complexity and device capability scores.
        task_complexity = self.calculate_task_complexity(latency_profile)
        device_capability = self.calculate_device_capability(latency_profile, available_devices)

        # Sort services by descending complexity.
        sorted_services = sorted(task_complexity.items(), key=lambda x: x[1], reverse=True)

        # Sort devices by ascending score; lower weighted latency means stronger.
        sorted_devices = sorted(device_capability.items(), key=lambda x: x[1])

        LOGGER.info(f'[Online Profiling Redeployment] Task complexity (descending): {sorted_services}')
        LOGGER.info(f'[Online Profiling Redeployment] Device capability (ascending, lower is better): {sorted_devices}')

        # Initialize the deployment plan and per-device service counters.
        deploy_plan = {}
        device_service_count = {device: 0 for device in available_devices}

        # Greedy placement.
        for service_name, complexity in sorted_services:
            # Skip services that are not present in the current DAG.
            if service_name not in dag:
                continue

            max_replicas = self.get_service_replica_count(service_name)
            deploy_plan[service_name] = []
            deployed_count = 0

            # Try to place replicas on the strongest devices first.
            for device_name, capability in sorted_devices:
                # Stop when the service has reached its replica budget.
                if deployed_count >= max_replicas:
                    break

                # Check whether the device still has capacity.
                device_limit = self.get_device_service_limit(device_name)
                if device_service_count[device_name] >= device_limit:
                    continue

                # Skip devices without latency data for this service.
                if service_name not in latency_profile or device_name not in latency_profile[service_name]:
                    LOGGER.debug(f'[Online Profiling Redeployment] No latency data for service {service_name} on device {device_name}, skipping')
                    continue

                # Place the service on this device.
                deploy_plan[service_name].append(device_name)
                device_service_count[device_name] += 1
                deployed_count += 1

                LOGGER.debug(f'[Online Profiling Redeployment] Deployed service {service_name} to device {device_name} '
                           f'(complexity: {complexity:.4f}, capability: {capability:.4f}, '
                           f'device usage: {device_service_count[device_name]}/{device_limit})')

            # Ensure every service is deployed to at least one device.
            if not deploy_plan[service_name]:
                # Pick the first device that still has remaining capacity.
                for device_name, capability in sorted_devices:
                    device_limit = self.get_device_service_limit(device_name)
                    if device_service_count[device_name] < device_limit:
                        deploy_plan[service_name].append(device_name)
                        device_service_count[device_name] += 1
                        LOGGER.warning(f'[Online Profiling Redeployment] Service {service_name} forced to device {device_name} '
                                     f'(no latency data but needed deployment)')
                        break

        LOGGER.info(f'[Online Profiling Redeployment] Final device service count: {device_service_count}')

        return deploy_plan

    def __call__(self, info):
        """
        Generate a redeployment plan using online profiling data.

        Before each redeployment round, pull the latest importance weights
        derived from recent object-count statistics from the agent and apply
        them to the greedy placement decision.
        """
        with self.overhead_estimator:
            source_id = info['source']['id']
            dag = info['dag']
            node_set = info['node_set']

            # Step 1: Sync importance weights derived from recent obj_num data.
            updated_weights = self.get_importance_weights_from_agent()
            self.service_importance_weights = updated_weights

            # Step 2: Pull the latest online latency profile from the agent.
            latency_profile = self.get_latency_profile_from_agent()

            # Step 3: Keep edge devices only and run greedy placement.
            cloud_device = getattr(self.system, 'cloud_device', None)
            available_devices = [node for node in node_set if node != cloud_device]

            if not available_devices:
                LOGGER.warning(f'[Online Profiling Redeployment] (source {source_id}) No edge devices available')
                deploy_plan = {
                    service_name: []
                    for service_name in dag
                    if service_name not in (TaskConstant.START.value, TaskConstant.END.value)
                }
            else:
                deploy_plan = self.greedy_deployment(latency_profile, dag, available_devices)

            LOGGER.info(f'[Online Profiling Redeployment] (source {source_id}) Deploy policy: {deploy_plan}')

            self.policy = deploy_plan

        return deploy_plan

    def get_redeployment_overhead(self):
        return self.overhead_estimator.get_latest_overhead()
