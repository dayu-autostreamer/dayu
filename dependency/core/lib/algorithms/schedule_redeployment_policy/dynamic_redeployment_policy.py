import abc
import copy
import time
import threading

from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER, KubeConfig, ConfigLoader, Context

__all__ = ('DynamicRedeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='dynamic')
class DynamicRedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    """
    Dynamic Redeployment Policy that redeploys services periodically based on the latest offloading strategy.
    - Redeploys every X minutes
    - Uses the latest offloading policy from the agent
    - Enforces configurable service limits per device
    """

    def __init__(self, system, agent_id, redeployment_interval_minutes=None, device_service_limits=None, default_service_limit=None, **kwargs):

        self.system = system
        self.agent_id = agent_id

        # 只允许从配置文件中读取policy，若没提供policy则为None
        if 'policy' in kwargs and kwargs['policy'] is not None:
            policy = kwargs['policy']
            if isinstance(policy, dict):
                self.policy = policy
            elif isinstance(policy, str):
                self.policy = ConfigLoader.load(Context.get_file_path(policy))
            else:
                raise TypeError(f'Input "policy" must be of type str or dict, get type {type(policy)}')
            LOGGER.info(f'[Dynamic Redeployment] Using policy from config file: {self.policy}')
        else:
            self.policy = None
            LOGGER.info(f'[Dynamic Redeployment] No policy supplied in config, default is None')

        LOGGER.info(f'[Dynamic Redeployment] Initialized with policy: {self.policy}')

        # 重部署间隔（分钟）
        if redeployment_interval_minutes is None:
            self.redeployment_interval_minutes = 5  # 默认5分钟
        elif isinstance(redeployment_interval_minutes, (int, float)):
            self.redeployment_interval_minutes = float(redeployment_interval_minutes)
        else:
            raise TypeError(f'Input "redeployment_interval_minutes" must be of type int or float, get type {type(redeployment_interval_minutes)}')

        self.redeployment_interval_seconds = self.redeployment_interval_minutes * 60

        # 设备服务数量限制配置
        if device_service_limits is None:
            self.device_service_limits = {}
        elif isinstance(device_service_limits, dict):
            # 确保所有值都是整数
            self.device_service_limits = {str(device): int(limit) for device, limit in device_service_limits.items()}
        else:
            raise TypeError(f'Input "device_service_limits" must be of type dict, get type {type(device_service_limits)}')

        # 默认服务数量限制
        if default_service_limit is None:
            self.default_service_limit = 2  # 默认每个设备最多2个服务
        elif isinstance(default_service_limit, (int, float)):
            self.default_service_limit = int(default_service_limit)
        else:
            raise TypeError(f'Input "default_service_limit" must be of type int or float, get type {type(default_service_limit)}')

        self.latest_offloading_policy = {}  # 存储最新的offloading策略
        self.last_redeployment_time = time.time()
        self.lock = threading.Lock()

        LOGGER.info(f'[Dynamic Redeployment] Initialized with device service limits: {self.device_service_limits}, default limit: {self.default_service_limit}')

    def update_latest_offloading_policy(self, offloading_policy):
        """更新最新的offloading策略"""
        # 如果offloading_policy中某个服务的设备为'cloud.kubeedge'，则不更新
        if offloading_policy and any(device == 'cloud.kubeedge' for device in offloading_policy.values()):
            LOGGER.debug(f'[Dynamic Redeployment] Offloading policy contains device "cloud.kubeedge", not updating latest offloading policy.')
            return
        with self.lock:
            self.latest_offloading_policy = copy.deepcopy(offloading_policy) if offloading_policy else {}
            LOGGER.debug(f'[Dynamic Redeployment] Updated latest offloading policy: {self.latest_offloading_policy}')

    def count_services_per_device(self, deploy_plan):
        """统计每个设备上部署的service数量"""
        device_service_count = {}
        for service_name, devices in deploy_plan.items():
            if not isinstance(devices, list):
                continue
            for device in devices:
                if device not in device_service_count:
                    device_service_count[device] = 0
                device_service_count[device] += 1
        return device_service_count

    def get_device_service_limit(self, device_name):
        """获取指定设备的最大服务数量限制"""
        device_name = str(device_name)
        # 如果设备在配置中，使用配置的限制；否则使用默认限制
        return self.device_service_limits.get(device_name, self.default_service_limit)

    def check_deployment_constraint(self, deploy_plan):
        """
        检查重部署策略是否满足约束条件
        如果某个设备上部署的service数量超过该设备的最大限制，则返回False
        如果某个服务的部署列表为空，也返回False
        """
        # 检查服务的部署列表是否为空
        for service_name, devices in deploy_plan.items():
            if not isinstance(devices, list) or len(devices) == 0:
                LOGGER.warning(f'[Dynamic Redeployment] Service {service_name} has no assigned devices, constraint violated.')
                return False

        device_service_count = self.count_services_per_device(deploy_plan)

        for device, count in device_service_count.items():
            max_limit = self.get_device_service_limit(device)
            if count > max_limit:
                LOGGER.warning(f'[Dynamic Redeployment] Device {device} has {count} services (exceeds limit {max_limit}), constraint violated')
                return False

        return True

    def convert_offloading_to_deployment_plan(self, offloading_policy, dag, node_set):
        """将offloading策略转换为deployment plan格式"""
        deploy_plan = {}

        # offloading策略格式: {service_name: device}
        # deployment plan格式: {service_name: [device1, device2, ...]}

        for service_name, device in offloading_policy.items():
            if device in node_set:
                deploy_plan[service_name] = [device]
            else:
                deploy_plan[service_name] = []

        # 如果deploy_plan的某个服务的部署列表为空列表，则返回原来的policy
        for service_name, devices in deploy_plan.items():
            if not devices:  # devices为空列表
                LOGGER.warning(f"[Dynamic Redeployment] Service {service_name} has empty deployment list, returning policy instead.")
                return copy.deepcopy(self.policy) if self.policy is not None else {}

        return deploy_plan

    def should_redeploy(self):
        """判断是否应该进行重部署"""
        current_time = time.time()
        time_since_last = current_time - self.last_redeployment_time

        if time_since_last >= self.redeployment_interval_seconds:
            return True
        return False

    def get_latest_offloading_from_agent(self, source_id):
        """从agent获取最新的offloading策略"""
        # 通过system获取对应的agent
        if not hasattr(self.system, 'schedule_table'):
            return None

        if source_id not in self.system.schedule_table:
            return None

        agent = self.system.schedule_table[source_id]

        # 检查agent是否有get_latest_offloading_policy方法
        if hasattr(agent, 'get_latest_offloading_policy'):
            return agent.get_latest_offloading_policy()

        return None

    def __call__(self, info):
        """
        生成重部署计划
        如果不需要更新部署策略，返回policy；如果需要更新，则更新policy并返回
        """
        source_id = info['source']['id']
        dag = info['dag']
        node_set = info['node_set']

        # 检查是否应该进行重部署
        if not self.should_redeploy():
            # 不需要更新，直接返回policy（类似fixed_redeployment_policy.py的处理方式）
            if self.policy is None:
                raise RuntimeError("[Dynamic Redeployment] No policy supplied for returning deployment plan.")
            else:
                deploy_plan = copy.deepcopy(self.policy)

            LOGGER.debug(f'[Dynamic Redeployment] (source {source_id}) Not time for redeployment yet, using policy: {deploy_plan}')
            return deploy_plan

        # 需要更新部署策略，从agent获取最新的offloading策略
        latest_offloading = self.get_latest_offloading_from_agent(source_id)

        # 如果从agent获取失败，尝试从本地存储获取
        if not latest_offloading:
            with self.lock:
                latest_offloading = copy.deepcopy(self.latest_offloading_policy)

        # 转换为deployment plan格式
        deploy_plan = self.convert_offloading_to_deployment_plan(latest_offloading, dag, node_set)

        # 检查约束条件，如果不满足则等待
        max_wait_attempts = 10  # 最多等待10次
        wait_attempt = 0

        while not self.check_deployment_constraint(deploy_plan) and wait_attempt < max_wait_attempts:
            LOGGER.info(f'[Dynamic Redeployment] (source {source_id}) Deployment plan violates constraint, waiting for new offloading policy (attempt {wait_attempt + 1}/{max_wait_attempts})')

            # 等待一小段时间后重新获取最新的offloading策略
            time.sleep(0.5)

            # 先从agent获取，如果失败再从本地存储获取
            latest_offloading = self.get_latest_offloading_from_agent(source_id)
            if not latest_offloading:
                with self.lock:
                    latest_offloading = copy.deepcopy(self.latest_offloading_policy)

            if latest_offloading:
                deploy_plan = self.convert_offloading_to_deployment_plan(latest_offloading, dag, node_set)

            wait_attempt += 1

        if not self.check_deployment_constraint(deploy_plan):
            LOGGER.warning(f'[Dynamic Redeployment] (source {source_id}) Could not find valid deployment plan after {max_wait_attempts} attempts, using policy')
            # 如果等待后仍不满足，使用policy
            if self.policy is None:
                raise RuntimeError("[Dynamic Redeployment] No policy supplied to fallback for deployment.")
            else:
                deploy_plan = copy.deepcopy(self.policy)


            LOGGER.info(f'[Dynamic Redeployment] (source {source_id}) Deploy policy: {deploy_plan}')
            return deploy_plan

        # 成功生成部署计划后，更新policy并更新最后重部署时间
        with self.lock:
            self.policy = copy.deepcopy(deploy_plan)
        self.last_redeployment_time = time.time()

        LOGGER.info(f'[Dynamic Redeployment] (source {source_id}) Deploy policy: {deploy_plan}')

        return deploy_plan
