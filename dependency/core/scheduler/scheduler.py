import threading

from core.lib.common import Context, LOGGER, ResourceLockManager, KubeConfig, TaskConstant
from core.lib.network import NodeInfo


class Scheduler:
    def __init__(self):
        self.schedule_table = {}
        self.resource_table = {}

        self.resource_lock_manager = ResourceLockManager()

        self.cloud_device = NodeInfo.get_cloud_node()

        self.config_extraction = Context.get_algorithm('SCH_CONFIG_EXTRACTION')
        self.scenario_extraction = Context.get_algorithm('SCH_SCENARIO_EXTRACTION')
        self.policy_extraction = Context.get_algorithm('SCH_POLICY_EXTRACTION')
        self.startup_policy = Context.get_algorithm('SCH_STARTUP_POLICY')

        self.extract_configuration()

    def extract_configuration(self):
        self.config_extraction(self)

    def get_startup_policy(self, info):
        return self.startup_policy(info)

    def add_scheduler_agent(self, source_id):
        agent = Context.get_algorithm('SCH_AGENT', system=self, agent_id=source_id)
        threading.Thread(target=agent.run).start()
        self.schedule_table[source_id] = agent

    def extract_scenario(self, task):
        return self.scenario_extraction(task)

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

        if not KubeConfig.check_services_running():
            plan = self.set_backup_offloading_plan(plan)
            LOGGER.debug('[Backup Offloading Plan] Processor not in running state (maybe in redeployment), '
                         'using backup offloading plan (all cloud).')

        LOGGER.info(f'[Schedule Plan] Source {source_id}: {plan}')

        return plan

    def set_backup_offloading_plan(self, plan):
        for service_name in plan['dag']:
            if service_name != TaskConstant.START.value:
                plan['dag'][service_name]['service']['execute_device'] = self.cloud_device

        return plan

    def update_scheduler_scenario(self, task):
        source_id = task.get_source_id()
        if source_id not in self.schedule_table:
            LOGGER.warning(f'Scheduler agent for source {source_id} not exists!')
            return
        scenario = self.extract_scenario(task)
        policy = self.policy_extraction(task)
        agent = self.schedule_table[source_id]
        agent.update_scenario(scenario)
        agent.update_policy(policy)
        agent.update_task(task)
        LOGGER.info(f'[Update Scenario] Source {source_id}: {scenario}')

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

        LOGGER.info(f'[Update Resource] Device {device}: {resource}')

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
        return plan

    def get_redeployment_plan(self, source_id, data):
        agent = self.schedule_table[source_id]
        plan = agent.get_redeployment_plan(data)
        return plan

    def get_schedule_overhead(self):
        overheads = []
        for source_id in self.schedule_table:
            agent = self.schedule_table[source_id]
            overheads.append(agent.get_schedule_overhead())

        return sum(overheads) / len(overheads) if overheads else 0
