import abc
import copy
import time
import threading

from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER, KubeConfig

__all__ = ('DeepVARedeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='deepva')
class DeepVARedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    """
    DeepVA重部署策略：基于DQN智能体的决策进行服务重部署
    每90秒执行一次重部署（与智能体的决策周期同步）
    """

    def __init__(self, system, agent_id, redeployment_interval=90):
        """
        Args:
            system: 调度系统
            agent_id: 智能体ID
            redeployment_interval: 重部署间隔（秒）
        """
        self.system = system
        self.agent_id = agent_id
        self.redeployment_interval = redeployment_interval
        
        # 上次重部署时间
        self.last_redeployment_time = time.time()
        
        # 上一次的部署计划（用于在非重部署时间返回，保持配置不变）
        self.last_deploy_plan = None
        
        # 线程锁
        self.lock = threading.Lock()
        
        LOGGER.info(f'[DeepVA Redeployment] Initialized with interval={redeployment_interval}s')

    def should_redeploy(self):
        """判断是否应该进行重部署"""
        current_time = time.time()
        
        with self.lock:
            if self.last_redeployment_time is None:
                # 第一次调用，进行重部署
                self.last_redeployment_time = current_time
                return True
            
            elapsed = current_time - self.last_redeployment_time
            
            if elapsed >= self.redeployment_interval:
                # 到达重部署时间
                self.last_redeployment_time = current_time
                return True
        
        return False

    def get_deployment_from_agent(self, source_id):
        """
        从agent获取当前部署决策
        
        Args:
            source_id: 数据源ID
            
        Returns:
            deployment: (num_services,) array 或 None
        """
        try:
            # 从调度系统获取对应的agent
            if source_id not in self.system.schedule_table:
                LOGGER.warning(f'[DeepVA Redeployment] Agent for source {source_id} not found')
                return None
            
            agent = self.system.schedule_table[source_id]
            
            # 检查agent是否有get_current_deployment方法
            if hasattr(agent, 'get_current_deployment'):
                deployment = agent.get_current_deployment()
                return deployment
            else:
                LOGGER.warning(f'[DeepVA Redeployment] Agent does not have get_current_deployment method')
                return None
        except Exception as e:
            LOGGER.error(f'[DeepVA Redeployment] Error getting deployment from agent: {str(e)}')
            return None

    def get_service_names_from_agent(self, source_id):
        """
        从agent获取服务名称列表，确保顺序一致
        
        Args:
            source_id: 数据源ID
            
        Returns:
            service_names: 服务名称列表，或None
        """
        try:
            if source_id not in self.system.schedule_table:
                return None
            
            agent = self.system.schedule_table[source_id]
            
            if hasattr(agent, 'service_names') and agent.service_names is not None:
                return agent.service_names
            
            return None
        except Exception as e:
            LOGGER.warning(f'[DeepVA Redeployment] Error getting service names: {str(e)}')
            return None

    def convert_deployment_to_plan(self, deployment, dag, node_set, source_id):
        """
        将部署数组转换为部署计划字典
        
        Args:
            deployment: (num_services,) array，每个元素是设备索引
            dag: DAG字典，{service_name: service_info}
            node_set: 可用节点集合
            source_id: 数据源ID
            
        Returns:
            deploy_plan: {service_name: [device_name]}
        """
        deploy_plan = {}
        
        # 获取设备列表（按索引顺序）
        device_list = self.system.device_list
        
        # 优先从agent获取服务列表（确保顺序一致）
        service_list = self.get_service_names_from_agent(source_id)
        if service_list is None:
            service_list = list(dag.keys())
        
        # 构建部署计划
        for service_idx, service_name in enumerate(service_list):
            if service_idx < len(deployment):
                device_idx = int(deployment[service_idx])
                if 0 <= device_idx < len(device_list):
                    device_name = device_list[device_idx]
                    
                    # 检查设备是否在可用节点集合中
                    if device_name in node_set:
                        deploy_plan[service_name] = [device_name]
                    else:
                        # 如果设备不在可用集合中，使用第一个可用设备
                        LOGGER.warning(f'[DeepVA Redeployment] Device {device_name} not in node_set, '
                                      f'using first available device')
                        deploy_plan[service_name] = [list(node_set)[0]]
                else:
                    # 无效的设备索引，使用第一个可用设备
                    LOGGER.warning(f'[DeepVA Redeployment] Invalid device index {device_idx}, '
                                  f'using first available device')
                    deploy_plan[service_name] = [list(node_set)[0]]
            else:
                # 服务数量不匹配，使用第一个可用设备
                deploy_plan[service_name] = [list(node_set)[0]]
        
        return deploy_plan

    def get_default_deployment(self, dag, node_set):
        """
        获取默认部署方案（将服务均匀分布到各个设备上，考虑设备容量限制）
        
        Args:
            dag: DAG字典
            node_set: 可用节点集合
            
        Returns:
            deploy_plan: {service_name: [device_name]}
        """
        deploy_plan = {}
        
        if not node_set:
            # 没有可用节点，使用cloud
            for service_name in dag.keys():
                deploy_plan[service_name] = ['cloud']
            return deploy_plan
        
        # 获取设备列表和容量限制
        device_list = self.system.device_list
        device_service_limits = self.system.device_service_limits
        
        # 只使用node_set中存在的设备
        available_devices = [dev for dev in device_list if dev in node_set]
        
        if not available_devices:
            # 如果没有匹配的设备，使用node_set的第一个
            for service_name in dag.keys():
                deploy_plan[service_name] = [list(node_set)[0]]
            return deploy_plan
        
        # 获取每个设备的容量限制
        device_capacities = {}
        for device in available_devices:
            device_idx = device_list.index(device)
            device_capacities[device] = device_service_limits[device_idx]
        
        # 统计每个设备当前分配的服务数
        device_counts = {device: 0 for device in available_devices}
        
        # 将服务均匀分布到各个设备
        service_names = list(dag.keys())
        for service_name in service_names:
            # 找到当前分配服务数最少且未达到容量限制的设备
            available_devices_sorted = sorted(
                [dev for dev in available_devices if device_counts[dev] < device_capacities[dev]],
                key=lambda d: device_counts[d]
            )
            
            if available_devices_sorted:
                # 选择负载最轻的设备
                selected_device = available_devices_sorted[0]
                deploy_plan[service_name] = [selected_device]
                device_counts[selected_device] += 1
            else:
                # 所有设备都达到容量限制，选择第一个设备（可能超限）
                LOGGER.warning(f'[DeepVA Redeployment] All devices at capacity, '
                             f'assigning {service_name} to {available_devices[0]}')
                deploy_plan[service_name] = [available_devices[0]]
                device_counts[available_devices[0]] += 1
        
        LOGGER.info(f'[DeepVA Redeployment] Default deployment distribution: {device_counts}')
        
        return deploy_plan

    def __call__(self, info):
        """
        生成重部署计划
        
        Args:
            info: dict，包含source、dag、node_set等信息
            
        Returns:
            deploy_plan: {service_name: [device_name]}
        """
        source_id = info['source']['id']
        dag = info['dag']
        node_set = info['node_set']
        
        # 检查是否应该进行重部署
        if not self.should_redeploy():
            # 不需要重部署，返回上一次的部署计划（保持配置不变）
            if self.last_deploy_plan is not None:
                LOGGER.debug(f'[DeepVA Redeployment] (source {source_id}) Not time for redeployment yet, '
                            f'returning last deploy plan')
                return self.last_deploy_plan
            else:
                # 第一次调用且未到重部署时间，返回默认配置
                default_plan = self.get_default_deployment(dag, node_set)
                self.last_deploy_plan = copy.deepcopy(default_plan)
                LOGGER.debug(f'[DeepVA Redeployment] (source {source_id}) Not time for redeployment yet, '
                            f'no previous plan, returning default deployment: {default_plan}')
                return default_plan
        
        # 从agent获取部署决策
        deployment = self.get_deployment_from_agent(source_id)
        
        if deployment is None:
            # 无法获取部署决策，使用默认方案
            LOGGER.warning(f'[DeepVA Redeployment] (source {source_id}) Cannot get deployment from agent, '
                          f'using default deployment')
            deploy_plan = self.get_default_deployment(dag, node_set)
        else:
            # 转换为部署计划
            deploy_plan = self.convert_deployment_to_plan(deployment, dag, node_set, source_id)
        
        # 保存本次部署计划，供下次非重部署时间返回
        self.last_deploy_plan = copy.deepcopy(deploy_plan)
        
        LOGGER.info(f'[DeepVA Redeployment] (source {source_id}) Deploy plan: {deploy_plan}')
        
        return deploy_plan
