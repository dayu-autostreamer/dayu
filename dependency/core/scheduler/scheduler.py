import threading

from core.lib.common import Context, LOGGER, ResourceLockManager
from core.lib.scheduling.deployment_plan import validate_plan
from core.lib.runtime import RuntimeContext

from .runtime_directory import RuntimeDirectoryError, create_runtime_directory_store
from .task_lease import create_task_lease_store


class Scheduler:
    def __init__(self, runtime_context=None, runtime_directory=None, task_lease_store=None):
        self.schedule_table = {}
        self.resource_table = {}

        self.resource_lock_manager = ResourceLockManager()
        self._runtime_state_lock = threading.RLock()

        self.runtime_context = runtime_context or RuntimeContext.get_default()
        initial_directory = runtime_directory
        if initial_directory is None:
            initial_directory = self.runtime_context.bootstrap.get("runtime_directory")
        if initial_directory is None:
            bootstrap_routes = self.runtime_context.bootstrap.get("runtime_routes")
            if bootstrap_routes:
                initial_directory = {
                    "install_id": self.runtime_context.install_id,
                    "revision": self.runtime_context.directory_revision,
                    "routes": bootstrap_routes,
                }
        if runtime_directory is not None and hasattr(runtime_directory, "snapshot_model"):
            self.runtime_directory = runtime_directory
        else:
            self.runtime_directory = create_runtime_directory_store(
                self.runtime_context,
                initial=initial_directory,
            )
        self.task_leases = task_lease_store or create_task_lease_store(self.runtime_context)

        self.cloud_device = self.runtime_context.cloud_node or self.runtime_context.local_node

        self.config_extraction = Context.get_algorithm('SCH_CONFIG_EXTRACTION')
        self.scenario_retrieval = Context.get_algorithm('SCH_SCENARIO_RETRIEVAL')
        self.policy_retrieval = Context.get_algorithm('SCH_POLICY_RETRIEVAL')
        self.startup_policy = Context.get_algorithm('SCH_STARTUP_POLICY')

        self.extract_necessary_configuration_setting()

    def extract_necessary_configuration_setting(self):
        self.config_extraction(self)

    def get_scenario_from_task(self, task):
        return self.scenario_retrieval(task)

    def get_policy_from_task(self, task):
        return self.policy_retrieval(task)

    def get_startup_policy(self, info):
        return self.startup_policy(info)

    def add_scheduler_agent(self, source_id):
        agent = Context.get_algorithm('SCH_AGENT', system=self, agent_id=source_id)
        threading.Thread(target=agent.run).start()
        self.schedule_table[source_id] = agent

    def register_schedule_table(self, source_id):
        if source_id in self.schedule_table:
            return
        self.add_scheduler_agent(source_id)

    def get_schedule_plan(self, info):
        source_id = info['source_id']
        agent = self.schedule_table[source_id]

        plan = agent.get_schedule_plan(info)

        if plan is None:
            LOGGER.debug('No schedule plan, use startup policy')
            plan = self.get_startup_policy(info)

        # LOGGER.info(f'[Schedule Plan] Source {source_id}: {plan}')

        return plan

    def runtime_directory_snapshot(self):
        return self.runtime_directory.snapshot()

    def runtime_directory_revision(self):
        return self.runtime_directory.snapshot_model().revision

    def runtime_routes(self, component=None, target_node=None, logical_service=None):
        return [
            route.to_dict()
            for route in self.runtime_directory.snapshot_model().find(
                component=component,
                target_node=target_node,
                logical_service=logical_service,
            )
        ]

    def runtime_service_nodes(self):
        return self.runtime_directory.snapshot_model().processor_deployment()

    def runtime_nodes_for_service(self, logical_service):
        return list(self.runtime_service_nodes().get(str(logical_service), []))

    def resolve_runtime_route(self, component, target_node=None, logical_service=None):
        return self.runtime_directory.snapshot_model().resolve(
            component=component,
            target_node=target_node,
            logical_service=logical_service,
        ).to_dict()

    def compact_runtime_routes(self, plan, source_device=""):
        return self.runtime_directory.compact_routes_for_plan(
            plan,
            source_device=source_device,
            cloud_node=self.cloud_device,
        )

    def replace_runtime_directory(self, directory, expected_revision):
        with self._runtime_state_lock:
            return self.runtime_directory.replace(directory, expected_revision)

    def propose_runtime_directory(self, directory, base_revision, proposal_id=None, ttl_seconds=60.0):
        with self._runtime_state_lock:
            return self.runtime_directory.propose(
                directory,
                base_revision=base_revision,
                proposal_id=proposal_id,
                ttl_seconds=ttl_seconds,
            )

    def commit_runtime_directory(self, proposal_id, expected_revision):
        with self._runtime_state_lock:
            return self.runtime_directory.commit(proposal_id, expected_revision)

    def reject_runtime_directory(self, proposal_id, reason=""):
        with self._runtime_state_lock:
            return self.runtime_directory.reject(proposal_id, reason)

    def clear_runtime_directory(self, install_id):
        with self._runtime_state_lock:
            return self.runtime_directory.clear(install_id)

    def acquire_task_lease(self, revision, root_uuid, ttl_seconds=60.0):
        with self._runtime_state_lock:
            return self.task_leases.acquire(
                revision=revision,
                root_uuid=root_uuid,
                active_revision=self.runtime_directory_revision(),
                ttl_seconds=ttl_seconds,
            )

    def renew_task_lease(self, revision, root_uuid, ttl_seconds=60.0):
        return self.task_leases.renew(revision, root_uuid, ttl_seconds=ttl_seconds)

    def release_task_lease(self, revision, root_uuid):
        return self.task_leases.release(revision, root_uuid)

    def count_task_leases(self, revision):
        return self.task_leases.count(revision)

    def update_scheduler_scenario(self, task):
        source_id = task.get_source_id()
        if source_id not in self.schedule_table:
            LOGGER.warning(f'Scheduler agent for source {source_id} not exists!')
            return False
        scenario = self.get_scenario_from_task(task)
        policy = self.get_policy_from_task(task)
        agent = self.schedule_table[source_id]
        agent.update_scenario(scenario)
        agent.update_policy(policy)
        agent.update_task(task)
        # LOGGER.info(f'[Update Scenario] Source {source_id}: {scenario}')
        return True

    def register_resource_table(self, device):
        if device in self.resource_table:
            return
        self.resource_table[device] = {}

    def update_scheduler_resource(self, info):
        device = info['device']
        resource = info['resource']
        self.resource_table[device] = resource

        for source_id in self.schedule_table:
            agent = self.schedule_table[source_id]
            agent.update_resource(device, resource)

        # LOGGER.info(f'[Update Resource] Device {device}: {resource}')

    def get_scheduler_resource(self):
        return self.resource_table

    async def get_resource_lock(self, info):
        return await self.resource_lock_manager.acquire_lock(
            info['resource'], info['device']
        )

    def get_source_node_selection_plan(self, source_id, data):
        agent = self.schedule_table[source_id]
        plan = agent.get_source_selection_plan(data)
        return plan

    def get_initial_deployment_plan(self, source_id, data):
        agent = self.schedule_table[source_id]
        plan = agent.get_initial_deployment_plan(data)
        try:
            return validate_plan(plan, data, cloud_node=self.cloud_device)
        except ValueError as exc:
            raise RuntimeDirectoryError(str(exc)) from exc

    def get_redeployment_plan(self, source_id, data):
        agent = self.schedule_table[source_id]
        plan = agent.get_redeployment_plan(data)
        try:
            return validate_plan(plan, data, cloud_node=self.cloud_device)
        except ValueError as exc:
            raise RuntimeDirectoryError(str(exc)) from exc

    @staticmethod
    def _normalize_generation_decision(decision):
        if isinstance(decision, bool):
            return {
                "generate": bool(decision),
                "reason": "agent_bool",
            }
        if not isinstance(decision, dict):
            return {
                "generate": True,
                "reason": "default_allow_invalid_decision",
            }

        generate = decision.get("generate", decision.get("allow", True))
        normalized = dict(decision)
        normalized["generate"] = bool(generate)
        normalized.setdefault("reason", "agent_decision")
        return normalized

    def should_generate(self, source_id, data):
        agent = self.schedule_table[source_id]
        hook = getattr(agent, "should_generate", None)
        if not callable(hook):
            return {
                "generate": True,
                "reason": "default_allow_no_hook",
            }
        decision = self._normalize_generation_decision(hook(data))
        revision = self.runtime_directory_revision()
        for action in decision.get("actions") or []:
            if not isinstance(action, dict) or action.get("type") != "clear_processor_queues":
                continue
            devices = action.get("target_devices") or action.get("devices") or []
            if isinstance(devices, str):
                devices = [devices]
            routes = []
            for device in devices:
                routes.extend(self.runtime_routes(component="controller", target_node=str(device)))
                routes.extend(self.runtime_routes(component="processor", target_node=str(device)))
            action["runtime_directory_revision"] = revision
            action["runtime_routes"] = routes
        return decision

    def get_schedule_overhead(self):
        overheads = []
        for source_id in self.schedule_table:
            agent = self.schedule_table[source_id]
            overheads.append(agent.get_schedule_overhead())

        return sum(overheads) / len(overheads) if overheads else 0
