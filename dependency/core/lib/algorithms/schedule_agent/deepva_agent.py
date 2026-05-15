import abc
import os
import threading
import time
import numpy as np

from core.lib.common import ClassFactory, ClassType, LOGGER, FileOps, Context, ConfigLoader
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent

__all__ = ('DeepVAAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='deepva')
class DeepVAAgent(BaseAgent, abc.ABC):
    """
    DeepVA: 基于深度强化学习的服务部署和任务卸载调度器
    使用DQN网络进行决策
    """

    def __init__(self, system,
                 agent_id: int,
                 mode: str = 'inference',
                 model_dir: str = 'model',
                 load_model: bool = False,
                 load_model_episode: int = 0,
                 redeployment_interval: int = 90,
                 reward_collection_window: int = 45,
                 learning_rate: float = 0.003,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.96,
                 target_update_freq: int = 5,
                 hidden_dim: int = 128,
                 replay_buffer_size: int = 5000,
                 batch_size: int = 32,
                 update_interval: int = 1,
                 update_after: int = 15,
                 save_interval: int = 10,
                 total_steps: int = 100,
                 configuration=None):  # 2.5小时 / 90秒 = 100步
        super().__init__(system, agent_id)

        from .deepva import DQNAgent, ReplayBuffer, StateBuffer

        self.agent_id = agent_id
        self.system = system
        self.mode = mode
        
        # 处理configuration参数
        if configuration is None or isinstance(configuration, dict):
            self.configuration = configuration
        elif isinstance(configuration, str):
            self.configuration = ConfigLoader.load(Context.get_file_path(configuration))
        else:
            raise TypeError(f'Input "configuration" must be of type str or dict, get type {type(configuration)}')

        # 从系统获取配置
        self.num_services = system.num_services
        self.num_devices = system.num_devices
        self.device_service_limits = system.device_service_limits
        self.device_list = system.device_list
        
        # 服务名称列表（将在第一次get_schedule_plan时初始化）
        self.service_names = None

        # 时间参数
        self.redeployment_interval = redeployment_interval  # 重部署间隔（秒）
        self.reward_collection_window = reward_collection_window  # 奖励收集窗口（秒）

        # 状态缓冲区
        self.state_buffer = StateBuffer(self.num_services, self.num_devices, 
                                       delay_window_size=10)

        # DQN智能体
        self.dqn_agent = DQNAgent(
            num_services=self.num_services,
            num_devices=self.num_devices,
            device_service_limits=self.device_service_limits,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            target_update_freq=target_update_freq,
            hidden_dim=hidden_dim
        )

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)

        # 训练参数
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.update_after = update_after
        self.save_interval = save_interval
        self.total_steps = total_steps

        # 模型保存路径
        self.model_dir = Context.get_file_path(os.path.join('scheduler/deepva', model_dir, f'agent_{self.agent_id}'))
        FileOps.create_directory(self.model_dir)
        if load_model:
            self.dqn_agent.load(self.model_dir, load_model_episode)

        # 当前部署方案和调度计划
        # 初始化为均匀分布的默认部署
        self.current_deployment = self._get_initial_deployment()
        self.state_buffer.update_deployment(self.current_deployment)
        self.schedule_plan = None
        
        # 部署方案的线程锁
        self.deployment_lock = threading.Lock()

        # 时延收集
        self.delay_collection = []
        self.delay_collection_start_time = None

        # 奖励日志
        self.reward_file = Context.get_file_path(os.path.join('scheduler/deepva', 'reward.txt'))
        FileOps.remove_file(self.reward_file)

        # 开销估计器
        self.overhead_estimator = OverheadEstimator('DeepVA', 'scheduler/deepva', agent_id=self.agent_id)

        LOGGER.info(f'[DeepVA Agent {self.agent_id}] Initialized with mode={mode}, '
                   f'num_services={self.num_services}, num_devices={self.num_devices}')

    def _get_initial_deployment(self):
        """
        获取初始部署方案（均匀分布到各设备）
        
        Returns:
            deployment: (num_services,) numpy array
        """
        deployment = np.zeros(self.num_services, dtype=int)
        device_counts = np.zeros(self.num_devices, dtype=int)
        
        for service_id in range(self.num_services):
            # 找到负载最轻且未超限的设备
            best_device = 0
            min_count = float('inf')
            
            for device_id in range(self.num_devices):
                if device_counts[device_id] < self.device_service_limits[device_id]:
                    if device_counts[device_id] < min_count:
                        min_count = device_counts[device_id]
                        best_device = device_id
            
            deployment[service_id] = best_device
            device_counts[best_device] += 1
        
        LOGGER.info(f'[DeepVA Agent {self.agent_id}] Initial deployment: {deployment}')
        return deployment

    def get_current_state(self):
        """获取当前环境状态"""
        while not self.state_buffer.is_ready():
            LOGGER.info(f'[DeepVA Agent {self.agent_id}] Waiting for state buffer to be ready...')
            time.sleep(1)
        
        state = self.state_buffer.get_state(self.device_list)
        return state

    def reset_env(self):
        """重置环境"""
        # 清空时延收集
        self.delay_collection = []
        self.delay_collection_start_time = time.time()
        
        state = self.get_current_state()
        return state

    def step_env(self, action):
        """
        执行动作，等待环境反馈
        
        Args:
            action: (num_services,) numpy array，部署方案
            
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否结束（DQN中始终为False）
            info: 额外信息
        """
        # 更新当前部署方案
        with self.deployment_lock:
            self.current_deployment = action
            self.state_buffer.update_deployment(action)
        
        # 触发重部署（通过redeployment_policy）
        # 注意：实际的重部署会由redeployment_policy处理
        
        # 等待reward_collection_window秒，收集时延数据
        LOGGER.info(f'[DeepVA Agent {self.agent_id}] Waiting {self.reward_collection_window}s to collect rewards...')
        time.sleep(self.reward_collection_window)
        
        # 计算奖励
        reward = self.calculate_reward()
        
        # 等待剩余时间直到redeployment_interval
        remaining_time = self.redeployment_interval - self.reward_collection_window
        if remaining_time > 0:
            LOGGER.info(f'[DeepVA Agent {self.agent_id}] Waiting {remaining_time}s until next decision...')
            time.sleep(remaining_time)
        
        # 获取下一个状态
        next_state = self.get_current_state()
        
        done = False
        info = {}
        
        return next_state, reward, done, info

    def calculate_reward(self):
        """
        计算奖励：reward = -avg_delay/5 + 1，并clip到(-1, 1)
        
        Returns:
            reward: float
        """
        avg_delay = self.state_buffer.get_average_delay()
        
        # reward = -avg_delay/5 + 1
        reward = -avg_delay / 5.0 + 1.0
        
        # clip到(-1, 1)
        reward = np.clip(reward, -1.0, 1.0)
        
        LOGGER.info(f'[DeepVA Agent {self.agent_id}] Reward: avg_delay={avg_delay:.4f}, reward={reward:.4f}')
        
        # 记录到文件
        with open(self.reward_file, 'a') as f:
            f.write(f'avg_delay:{avg_delay:.4f} reward:{reward:.4f}\n')
        
        return reward

    def train_dqn_agent(self):
        """训练DQN智能体"""
        LOGGER.info(f'[DeepVA Agent {self.agent_id}] Start training DQN agent...')
        
        state = self.reset_env()
        
        for step in range(self.total_steps):
            LOGGER.info(f'[DeepVA Agent {self.agent_id}] Training step {step}/{self.total_steps}')
            
            with self.overhead_estimator:
                # 选择动作
                action = self.dqn_agent.select_action(state, deterministic=False)
            
            # 执行动作
            next_state, reward, done, info = self.step_env(action)
            
            # 添加到replay buffer
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            LOGGER.info(f'[DeepVA Agent {self.agent_id}] Step {step}: reward={reward:.4f}, '
                       f'buffer_size={len(self.replay_buffer)}, epsilon={self.dqn_agent.epsilon:.4f}')
            
            # 更新状态
            state = next_state
            
            # 训练网络
            if step >= self.update_after and step % self.update_interval == 0:
                loss = self.dqn_agent.train(self.replay_buffer, self.batch_size)
                LOGGER.info(f'[DeepVA Agent {self.agent_id}] Training DQN: loss={loss:.4f}')
            
            # 保存模型
            if step % self.save_interval == 0 and step > 0:
                self.dqn_agent.save(self.model_dir, step)
                LOGGER.info(f'[DeepVA Agent {self.agent_id}] Model saved at step {step}')
        
        # 最终保存
        self.dqn_agent.save(self.model_dir, self.total_steps)
        LOGGER.info(f'[DeepVA Agent {self.agent_id}] Training completed!')

    def inference_dqn_agent(self):
        """推理模式：使用训练好的模型进行决策"""
        LOGGER.info(f'[DeepVA Agent {self.agent_id}] Start inference mode...')
        
        state = self.reset_env()
        step = 0
        
        while True:
            LOGGER.info(f'[DeepVA Agent {self.agent_id}] Inference step {step}')
            
            with self.overhead_estimator:
                # 选择动作（确定性策略）
                action = self.dqn_agent.select_action(state, deterministic=True)
            
            # 执行动作
            next_state, reward, done, info = self.step_env(action)
            
            LOGGER.info(f'[DeepVA Agent {self.agent_id}] Inference step {step}: reward={reward:.4f}')
            
            # 更新状态
            state = next_state
            step += 1

    def update_scenario(self, scenario):
        """
        更新场景信息（从Distributor获取）
        
        Args:
            scenario: dict，包含任务时延等信息
        """
        try:
            if 'delay' in scenario:
                task_delay = scenario['delay']
                self.state_buffer.update_delay(task_delay)
                LOGGER.debug(f'[DeepVA Agent {self.agent_id}] Updated delay: {task_delay}')
        except Exception as e:
            LOGGER.warning(f'[DeepVA Agent {self.agent_id}] Error updating scenario: {str(e)}')

    def update_resource(self, device, resource):
        """
        更新设备资源信息
        
        Args:
            device: 设备标识
            resource: dict，包含CPU、内存、带宽等信息
        """
        try:
            cpu_load = resource.get('cpu_usage', 0)
            memory_load = resource.get('memory_usage', 0)
            bandwidth = resource.get('available_bandwidth', 0)
            
            self.state_buffer.update_resource(device, cpu_load, memory_load, bandwidth)
            
            LOGGER.debug(f'[DeepVA Agent {self.agent_id}] Updated resource for {device}: '
                        f'cpu={cpu_load}, mem={memory_load}, bw={bandwidth}')
        except Exception as e:
            LOGGER.warning(f'[DeepVA Agent {self.agent_id}] Error updating resource: {str(e)}')

    def update_policy(self, policy):
        """更新策略（预留接口）"""
        pass

    def update_task(self, task):
        """更新任务信息（预留接口）"""
        pass

    def get_schedule_plan(self, info):
        """
        获取调度计划（用于任务offloading）
        根据当前部署方案生成offloading决策
        
        Args:
            info: dict，包含dag等信息
            
        Returns:
            schedule_plan: 调度方案（包含dag字段），或None（使用startup policy）
        """
        with self.deployment_lock:
            if self.current_deployment is None:
                return None
            
            # 复制当前部署方案，避免在处理过程中被修改
            deployment_copy = self.current_deployment.copy()
        
        # 获取DAG
        dag = info.get('dag', {})
        if not dag:
            return None
        
        # 初始化服务名称列表（第一次调用时）
        if self.service_names is None:
            self.service_names = list(dag.keys())
            LOGGER.info(f'[DeepVA Agent {self.agent_id}] Service names initialized: {self.service_names}')
        
        # 根据部署方案设置每个服务的执行设备
        for service_idx, service_name in enumerate(self.service_names):
            if service_name in dag and service_idx < len(deployment_copy):
                device_idx = int(deployment_copy[service_idx])
                if 0 <= device_idx < len(self.device_list):
                    device_name = self.device_list[device_idx]
                    
                    # 设置该服务的执行设备
                    if 'service' in dag[service_name]:
                        dag[service_name]['service']['execute_device'] = device_name
                        LOGGER.debug(f'[DeepVA Agent {self.agent_id}] Service {service_name} -> {device_name}')
        
        # 构建返回的policy
        policy = {'dag': dag}
        
        # 如果配置了configuration，添加到policy中
        if self.configuration is not None:
            policy.update(self.configuration.copy())
        
        return policy

    def get_current_deployment(self):
        """获取当前部署方案（供redeployment_policy使用）"""
        with self.deployment_lock:
            if self.current_deployment is not None:
                return self.current_deployment.copy()
            return None

    def get_schedule_overhead(self):
        """获取调度开销"""
        return self.overhead_estimator.get_latest_overhead()

    def run(self):
        """主运行函数"""
        if self.mode == 'train':
            self.train_dqn_agent()
        elif self.mode == 'inference':
            self.inference_dqn_agent()
        else:
            raise ValueError(f'Invalid mode: {self.mode}, only support ["train", "inference"]')

