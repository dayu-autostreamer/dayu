from typing import List, Optional
import threading
import random
import torch
import time
import os
import glob

from core.lib.common import LOGGER, FileOps, Context

from .topology_encoder import TopologyEncoders
from .ppo_agent import HedgerOffloadingPPO, HedgerDeploymentPPO
from .hedger_config import from_partial_dict, OffloadingConstraintCfg, DeploymentConstraintCfg, LogicalTopology, \
    PhysicalTopology

__all__ = ('Hedger',)


class Hedger:
    def __init__(self, network_params: dict, hyper_params: dict, agent_params: dict):
        self.encoder_network_params = network_params['topology_encoder']
        self.offloading_network_params = network_params['offloading_agent']
        self.deployment_network_params = network_params['deployment_agent']

        self.mode = hyper_params['mode']
        self.device = torch.device(hyper_params['device'])
        self.seed = hyper_params['seed']
        self.deployment_interval = hyper_params['deployment_interval']
        self.offloading_interval = hyper_params['offloading_interval']
        self.update_epochs = hyper_params['update_epochs']
        self.total_steps = hyper_params['total_steps']
        self.model_dir = Context.get_file_path(hyper_params['model_dir'])
        self.load_model = hyper_params['load_model']
        self.save_interval = hyper_params['save_interval']
        self.load_epoch = hyper_params['load_epoch']

        self.train_deployment_flag = hyper_params.get("train_deployment", True)
        self.train_offloading_flag = hyper_params.get("train_offloading", True)

        self.offloading_rollout_len = hyper_params.get("offloading_rollout_len", 32)
        self.deployment_rollout_len = hyper_params.get("deployment_rollout_len", 8)
        self.offloading_batch_size = hyper_params.get("offloading_batch_size", 16)
        self.deployment_batch_size = hyper_params.get("deployment_batch_size", 4)

        self.offloading_agent_params = agent_params['offloading_agent']
        self.deployment_agent_params = agent_params['deployment_agent']

        self.deployment_thread_stop_event = threading.Event()
        self.offloading_thread_stop_event = threading.Event()

        self.physical_topology = None
        self.logical_topology = None

        self.shared_topology_encoder = None
        self.deployment_agent = None
        self.offloading_agent = None

        self._deployment_update_steps = 0
        self._offloading_update_steps = 0
        # Global training epoch counter (increments when PPO updates are performed per loop)
        self._epoch = 0

        self.register_topology_encoder()
        self.register_deployment_agent()
        self.register_offloading_agent()

        self._data_lock = threading.Lock()

        FileOps.create_directory(self.model_dir)
        if self.load_model:
            self.load_checkpoint(epoch=self.load_epoch)

        self.deployment_decision = None
        self.offloading_decision = None

        self.initial_deployment_plan = None
        self.deployment_plan = None
        self.offloading_plan = None

        self.deployment_transitions: List[dict] = []
        self.offloading_transitions: List[dict] = []

        self.prev_deploy_mask = None

        threading.Thread(target=self.run, daemon=True).start()

    def set_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def register_topology_encoder(self):
        if self.shared_topology_encoder:
            return

        self.shared_topology_encoder = TopologyEncoders(
            d_model=self.encoder_network_params['encoder_emb_dim'],
            heads=self.encoder_network_params['logic_gat_heads'],
            num_roles=self.encoder_network_params['phys_role_num'],
            role_emb_dim=self.encoder_network_params['phys_role_emb_dim'],
            dropout=self.encoder_network_params['encoder_dropout'],
        ).to(self.device)

    def register_deployment_agent(self):
        if self.deployment_agent:
            return

        assert self.shared_topology_encoder, 'Shared topology encoder must be registered before deployment agent.'

        self.deployment_agent = HedgerDeploymentPPO(
            encoder=self.shared_topology_encoder,
            d_model=self.encoder_network_params['encoder_emb_dim'],
            actor_lr=self.deployment_agent_params['actor_lr'],
            critic_lr=self.deployment_agent_params['critic_lr'],
            gamma=self.deployment_agent_params['gamma'],
            lamda=self.deployment_agent_params['lamda'],
            clip_eps=self.deployment_agent_params['clip_eps'],
            update_encoder=self.deployment_agent_params['update_encoder'],
            constraint_cfg=from_partial_dict(DeploymentConstraintCfg, self.deployment_agent_params),
        ).to(self.device)

    def register_offloading_agent(self):
        if self.offloading_agent:
            return

        assert self.shared_topology_encoder, 'Shared topology encoder must be registered before offloading agent.'

        self.offloading_agent = HedgerOffloadingPPO(
            encoder=self.shared_topology_encoder,
            d_model=self.encoder_network_params['encoder_emb_dim'],
            actor_lr=self.offloading_agent_params['actor_lr'],
            critic_lr=self.offloading_agent_params['critic_lr'],
            gamma=self.offloading_agent_params['gamma'],
            lamda=self.offloading_agent_params['lamda'],
            clip_eps=self.offloading_agent_params['clip_eps'],
            update_encoder=self.offloading_agent_params['update_encoder'],
            constraint_cfg=from_partial_dict(OffloadingConstraintCfg, self.offloading_agent_params),
        ).to(self.device)

    def register_physical_topology(self, edge_nodes, source_device):
        if self.physical_topology:
            return

        self.physical_topology = PhysicalTopology(edge_nodes, source_device)

        LOGGER.debug(f'[Hedger] Registered physical topology, nodes: {self.physical_topology.nodes}, '
                     f'links: {self.physical_topology.links}')

    def register_logical_topology(self, dag):
        if self.logical_topology:
            return

        self.logical_topology = LogicalTopology(dag)

        LOGGER.debug(f'[Hedger] Registered logical topology, nodes: {self.logical_topology.service_list}, '
                     f'links: {self.logical_topology.links}')

    def register_initial_deployment(self, deployment_plan):
        assert self.logical_topology, "Logical topology must be registered before registering initial deployment."
        assert self.physical_topology, "Physical topology must be registered before registering initial deployment."

        if self.initial_deployment_plan:
            return
        self.initial_deployment_plan = deployment_plan
        self.deployment_plan = deployment_plan

        # update deploy mask
        self.prev_deploy_mask = self._map_deployment_plan_to_deployment_mask(deployment_plan)

    def get_offloading_plan(self):
        return self.offloading_plan

    def get_initial_deployment_plan(self):
        return self.initial_deployment_plan

    def get_redeployment_plan(self):
        return self.deployment_plan

    def _collect_deployment_state(self, prev_deploy_mask: Optional[torch.Tensor]):
        pass

    def _collect_offloading_state(self):
        pass

    def inference_hedger(self):
        LOGGER.info('[Hedger] Hedger is running in inference mode..')

        logic_links = torch.tensor(self.logical_topology.links,
                                   dtype=torch.long, device=self.device).t().contiguous()
        phys_links = torch.tensor(self.physical_topology.links,
                                  dtype=torch.long, device=self.device).t().contiguous()

        self.prev_deploy_mask = torch.zeros((len(self.logical_topology), len(self.physical_topology)),
                                            dtype=torch.bool, device=self.device)
        self.prev_deploy_mask[:, self.physical_topology.cloud_idx] = True

        threading.Thread(target=self.inference_deployment_agent).start()
        threading.Thread(target=self.inference_offloading_agent).start()

    def inference_deployment_agent(self):
        LOGGER.info('[Hedger Deployment] Hedger Deployment Agent start inference.')

    def inference_offloading_agent(self):
        LOGGER.info('[Hedger Offloading] Hedger Offloading Agent start inference.')

    def train_hedger(self):
        assert self.logical_topology is not None, "Logical topology must be registered before training."
        assert self.physical_topology is not None, "Physical topology must be registered before training."

        LOGGER.info('[Hedger] Hedger is running in training mode..')
        self.set_seed()

        logic_links = torch.tensor(self.logical_topology.links,
                                   dtype=torch.long, device=self.device).t().contiguous()
        phys_links = torch.tensor(self.physical_topology.links,
                                  dtype=torch.long, device=self.device).t().contiguous()

        LOGGER.info(f"Logical graph edges: {logic_links.size(1)}, physical graph edges: {phys_links.size(1)}")

        if not self.prev_deploy_mask:
            LOGGER.warning('No previous deployment mask found, initialize to pure cloud deployment.')
            self.prev_deploy_mask = torch.zeros((len(self.logical_topology), len(self.physical_topology)),
                                                dtype=torch.bool, device=self.device)
            self.prev_deploy_mask[:, self.physical_topology.cloud_idx] = True

        threading.Thread(target=self.train_deployment_agent, daemon=True).start()
        threading.Thread(target=self.train_offloading_agent, daemon=True).start()

        while True:
            try:
                updates_in_tick = 0

                # Offloading PPO update
                if self.train_offloading_flag and len(self.offloading_transitions) >= self.offloading_rollout_len:
                    with self._data_lock:
                        off_transitions = self.offloading_transitions[:self.offloading_rollout_len]
                        del self.offloading_transitions[:self.offloading_rollout_len]

                    self.offloading_agent.ppo_update(off_transitions,
                                                     epochs=self.update_epochs,
                                                     batch_size=self.offloading_batch_size)
                    self._offloading_update_steps += 1
                    updates_in_tick += 1
                    LOGGER.info(f"[Offloading] PPO update #{self._offloading_update_steps}, "
                                f"used {len(off_transitions)} transitions.")

                # Deployment PPO update
                if self.train_deployment_flag and len(self.deployment_transitions) >= self.deployment_rollout_len:
                    with self._data_lock:
                        dep_transitions = self.deployment_transitions[:self.deployment_rollout_len]
                        del self.deployment_transitions[:self.deployment_rollout_len]

                    self.deployment_agent.ppo_update(dep_transitions,
                                                     epochs=self.update_epochs,
                                                     batch_size=self.deployment_batch_size)
                    self._deployment_update_steps += 1
                    updates_in_tick += 1
                    LOGGER.info(f"[Deployment] PPO update #{self._deployment_update_steps}, "
                                f"used {len(dep_transitions)} transitions.")

                # Save checkpoint
                if updates_in_tick > 0:
                    prev_epoch = self._epoch
                    self._epoch += updates_in_tick
                    # 当 _epoch 跨过 save_interval 的倍数时保存一次
                    if (prev_epoch // self.save_interval) != (self._epoch // self.save_interval):
                        try:
                            self.save_checkpoint(epoch=self._epoch)
                        except Exception as e:
                            LOGGER.warning(f'[Hedger] Failed to save checkpoint (epoch {self._epoch}): {e}')
                            LOGGER.exception(e)

                if self._epoch > self.total_steps:
                    LOGGER.info("Reached max_epoch, stop training.")
                    break

                time.sleep(0.5)
            except Exception as e:
                LOGGER.exception(f"Error in Hedger training loop: {e}")
                continue

        self.deployment_thread_stop_event.set()
        self.offloading_thread_stop_event.set()
        LOGGER.info('[Hedger] Training of Hedger finished.')

    def train_deployment_agent(self):
        """
        部署 agent 的采样线程：
        - 周期性地收集当前拓扑状态（由 _collect_deployment_state 实现）
        - 调用 deployment_agent.policy 采样新的部署 mask
        - 让环境根据部署运行一段时间，统计指标，计算 reward
        - 将 transition 放入 self.deployment_transitions，等待主线程 PPO 更新
        """
        if not self.train_deployment_flag:
            LOGGER.info("train_deployment_flag=False, deployment training thread will not run.")
            return

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting deployment training."

        LOGGER.info('[Hedger Deployment] Hedger Deployment Agent start training.')

        # 静态 edge_index，可以一直复用
        logic_edge_index = torch.tensor(self.logical_topology.links, dtype=torch.long,
                                        device=self.device).t().contiguous()
        phys_edge_index = torch.tensor(self.physical_topology.links, dtype=torch.long,
                                       device=self.device).t().contiguous()

        step = 0

        while not self.deployment_thread_stop_event.is_set():
            try:
                logic_feats, phys_feats, metrics, done = self._collect_deployment_state(self.prev_deploy_mask)
            except Exception as e:
                LOGGER.exception(f"Error in collect_deployment_state: {e}")
                time.sleep(1.0)
                continue

            if logic_feats is None or phys_feats is None:
                # No valid data, skip
                time.sleep(self.deployment_interval)
                continue

            # map features to device
            logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
            phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
            prev_deploy_mask_dev = self.prev_deploy_mask.to(self.device) if self.prev_deploy_mask is not None else None

            with torch.no_grad():
                # 采样新的部署策略
                deploy_mask, logp, ent, value, aux = self.deployment_agent.policy(
                    logic_edge_index=logic_edge_index,
                    logic_feats=logic_feats_dev,
                    phys_edge_index=phys_edge_index,
                    phys_feats=phys_feats_dev,
                    topo_order=None,  # 内部会根据 logical edge index 推一个拓扑顺序
                    prev_deploy_mask=prev_deploy_mask_dev
                )

            # 根据指标 + RL 辅助信息计算 reward
            reward = self._compute_deployment_reward(metrics, aux)

            # 构造 transition，并全部搬到 CPU，以避免跨线程的 device 问题
            tr = {
                "logic_edge_index": logic_edge_index.cpu(),
                "logic_feats": {k: v.cpu() for k, v in logic_feats_dev.items()},
                "phys_edge_index": phys_edge_index.cpu(),
                "phys_feats": {k: v.cpu() for k, v in phys_feats_dev.items()},
                "deploy_mask": deploy_mask.cpu(),
                "topo_order": None,  # evaluate 时也可以重新 topo 排序
                "prev_deploy_mask": self.prev_deploy_mask.cpu() if self.prev_deploy_mask is not None else None,
                "logp": logp.detach().cpu(),
                "value": value.detach().cpu(),
                "reward": float(reward),
                "done": bool(done),
            }

            with self._data_lock:
                self.deployment_transitions.append(tr)

            # 更新“上一轮部署”，供下一次 residual 内存计算
            self.prev_deploy_mask = deploy_mask.detach().cpu()

            step += 1
            # 根据部署周期控制采样频率
            time.sleep(self.deployment_interval)

        LOGGER.info("Deployment training thread stopped.")

    def train_offloading_agent(self):
        """
        卸载 agent 的采样线程：
        - 以较短间隔（每次任务到来 / 小时间窗）收集当前拓扑与系统状态
        - 调用 offloading_agent.policy 决策每个 service 的执行设备
        - 环境执行该策略，返回该窗口内的 latency / SLO / cloud 使用等指标
        - 计算 reward，并将 transition 放入 self.offloading_transitions
        """
        if not self.train_offloading_flag:
            LOGGER.info("train_offloading_flag=False, offloading training thread will not run.")
            return

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting offloading training."

        LOGGER.info('[Hedger Offloading] Hedger Offloading Agent start training.')

        logic_edge_index = torch.tensor(self.logical_topology.links, dtype=torch.long,
                                        device=self.device).t().contiguous()
        phys_edge_index = torch.tensor(self.physical_topology.links, dtype=torch.long,
                                       device=self.device).t().contiguous()

        step = 0
        while not self.offloading_thread_stop_event.is_set():
            try:
                logic_feats, phys_feats, static_mask, metrics, done = self._collect_offloading_state()
            except Exception as e:
                LOGGER.exception(f"Error in collect_offloading_state: {e}")
                time.sleep(0.1)
                continue

            if logic_feats is None or phys_feats is None or static_mask is None:
                time.sleep(self.offloading_interval)
                continue

            logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
            phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
            static_mask_dev = static_mask.to(self.device)

            with torch.no_grad():
                actions, logp, ent, value, aux = self.offloading_agent.policy(
                    logic_edge_index=logic_edge_index,
                    logic_feats=logic_feats_dev,
                    phys_edge_index=phys_edge_index,
                    phys_feats=phys_feats_dev,
                    static_mask=static_mask_dev,
                    topo_order=None,
                )

            reward = self._compute_offloading_reward(metrics, aux)

            tr = {
                "logic_edge_index": logic_edge_index.cpu(),
                "logic_feats": {k: v.cpu() for k, v in logic_feats_dev.items()},
                "phys_edge_index": phys_edge_index.cpu(),
                "phys_feats": {k: v.cpu() for k, v in phys_feats_dev.items()},
                "actions": actions.cpu(),
                "static_mask": static_mask_dev.cpu(),
                "topo_order": None,
                "logp": logp.detach().cpu(),
                "value": value.detach().cpu(),
                "reward": float(reward),
                "done": bool(done),
            }

            with self._data_lock:
                self.offloading_transitions.append(tr)

            step += 1
            time.sleep(self.offloading_interval)

        LOGGER.info("Offloading training thread stopped.")

    def _compute_offloading_reward(self, metrics, aux) -> float:
        """
        Offloading 的 reward：以“负延迟 + 惩罚项”的形式定义。
        metrics 中应包含一些基础性能指标，aux 来自 offloading policy 内部（switch/relax 的惩罚）。
        """
        metrics = metrics or {}

        # 从 metrics 中提取关键指标
        latency_ms = float(metrics.get("latency_ms", metrics.get("avg_latency_ms", 0.0)))
        slo_v = float(metrics.get("slo_violation", metrics.get("slo_violation_rate", 0.0)))
        cloud_frac = float(metrics.get("cloud_fraction", 0.0))

        # 权重从 hyper_params 里读，如果没有则用默认
        w_lat = float(self.hyper_params.get("reward_off_latency_weight", 1e-3))
        w_slo = float(self.hyper_params.get("reward_off_slo_weight", 1.0))
        w_cloud = float(self.hyper_params.get("reward_off_cloud_weight", 0.1))

        # 基本 reward：倾向于降低延迟、降低 SLO 违约、降低上云比例
        reward = 0.0
        reward -= w_lat * latency_ms
        reward -= w_slo * slo_v
        reward -= w_cloud * cloud_frac

        # 来自约束的附加代价（切换次数 + relax 次数）
        aux_cost = float(aux.get("aux_cost", 0.0))
        reward -= aux_cost

        return reward

    def _compute_deployment_reward(self, metrics, aux) -> float:
        """
        Deployment 的 reward：
        - 利用宏观性能指标（平均延迟、SLO violation、cloud 使用）
        - 加上来自 offloading 的汇总反馈（avg_offloading_reward）
        - 再加上部署变更成本 + 容量 relax 惩罚，实现“自上而下 + 自下而上的双向反馈”
        """
        metrics = metrics or {}

        avg_latency_ms = float(metrics["avg_latency_ms"])
        slo_v = float(metrics["slo_violation_rate"])
        cloud_frac = float(metrics["cloud_fraction"])
        avg_off_r = float(metrics["avg_offloading_reward"])
        deploy_change_cost = float(metrics["deploy_change_cost"])

        # 权重可以在 hyper_params 中配置
        w_lat = float(self.hyper_params.get("reward_dep_latency_weight", 1e-3))
        w_slo = float(self.hyper_params.get("reward_dep_slo_weight", 1.0))
        w_cloud = float(self.hyper_params.get("reward_dep_cloud_weight", 0.1))
        w_change = float(self.hyper_params.get("reward_dep_change_weight", 0.1))
        w_off = float(self.hyper_params.get("reward_dep_offload_weight", 0.1))

        reward = 0.0
        # 惩罚高延迟 / 高 SLO violation / 高 cloud 使用
        reward -= w_lat * avg_latency_ms
        reward -= w_slo * slo_v
        reward -= w_cloud * cloud_frac
        reward -= w_change * deploy_change_cost

        # 利用下层 agent 的平均 reward 作为“自下而上”的反馈信号
        reward += w_off * avg_off_r

        # 容量 relax 罚项（DeploymentConstraintCfg.penalty_capacity_relax）
        cap_relax_cnt = float(aux.get("capacity_relax_cnt", 0.0))
        penalty_capacity_relax = float(self.deployment_agent.cfg.penalty_capacity_relax)
        reward -= penalty_capacity_relax * cap_relax_cnt

        return reward

    def _epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.model_dir, f'hedger_ckpt_epoch_{epoch}.pt')

    def _latest_epoch_checkpoint_path(self) -> Optional[str]:
        pattern = os.path.join(self.model_dir, 'hedger_ckpt_epoch_*.pt')
        files = glob.glob(pattern)
        if not files:
            return None

        # parse epoch number
        def parse_epoch(p):
            try:
                base = os.path.basename(p)
                # hedger_ckpt_epoch_{n}.pt
                num = base.replace('hedger_ckpt_epoch_', '').replace('.pt', '')
                return int(num)
            except Exception:
                return -1

        files = sorted(files, key=parse_epoch, reverse=True)
        return files[0] if files else None

    def save_checkpoint(self, epoch: Optional[int] = None):
        """Save encoder + both agents and their optimizers into a single file, labeled by epoch."""
        if epoch is None:
            epoch = self._epoch
        path = self._epoch_checkpoint_path(epoch)
        ckpt = {
            'encoder': self.shared_topology_encoder.state_dict(),
            'deployment_agent': self.deployment_agent.state_dict(),
            'offloading_agent': self.offloading_agent.state_dict(),
            'deployment_actor_opt': self.deployment_agent.actor_opt.state_dict(),
            'deployment_critic_opt': self.deployment_agent.critic_opt.state_dict(),
            'offloading_actor_opt': self.offloading_agent.actor_opt.state_dict(),
            'offloading_critic_opt': self.offloading_agent.critic_opt.state_dict(),
            'meta': {
                'time': time.time(),
                'seed': self.seed,
                'deployment_updates': self._deployment_update_steps,
                'offloading_updates': self._offloading_update_steps,
                'device': str(self.device),
                'epoch': epoch,
            }
        }
        torch.save(ckpt, path)
        LOGGER.info(f'[Hedger] Checkpoint (epoch={epoch}) saved to {path}')

    def load_checkpoint(self, epoch: int = None):
        """
        Load encoder + agents + optimizers.
        - epoch: specific epoch number to load checkpoint from.
                (If `epoch` is not provided, the latest epoch-based checkpoint is used.)
        """
        if epoch is not None:
            ep_path = self._epoch_checkpoint_path(int(epoch))
            if os.path.exists(ep_path):
                target_path = ep_path
            else:
                LOGGER.warning(f'[Hedger] Epoch checkpoint not found at {ep_path}, falling back to latest epoch file.')
                target_path = self._latest_epoch_checkpoint_path()
        else:
            target_path = self._latest_epoch_checkpoint_path()

        if not target_path or not os.path.exists(target_path):
            LOGGER.warning(f'[Hedger] No checkpoint found to load in {self.model_dir}.')
            return

        ckpt = torch.load(target_path, map_location=self.device)
        # Load models first
        self.shared_topology_encoder.load_state_dict(ckpt.get('encoder', {}))
        # Agents keep a shared reference to the encoder; load their specific heads
        self.deployment_agent.load_state_dict(ckpt.get('deployment_agent', {}), strict=False)
        self.offloading_agent.load_state_dict(ckpt.get('offloading_agent', {}), strict=False)
        # Then load optimizer states (after modules are loaded)
        if 'deployment_actor_opt' in ckpt:
            self.deployment_agent.actor_opt.load_state_dict(ckpt['deployment_actor_opt'])
            self._move_optimizer_state(self.deployment_agent.actor_opt, self.device)
        if 'deployment_critic_opt' in ckpt:
            self.deployment_agent.critic_opt.load_state_dict(ckpt['deployment_critic_opt'])
            self._move_optimizer_state(self.deployment_agent.critic_opt, self.device)
        if 'offloading_actor_opt' in ckpt:
            self.offloading_agent.actor_opt.load_state_dict(ckpt['offloading_actor_opt'])
            self._move_optimizer_state(self.offloading_agent.actor_opt, self.device)
        if 'offloading_critic_opt' in ckpt:
            self.offloading_agent.critic_opt.load_state_dict(ckpt['offloading_critic_opt'])
            self._move_optimizer_state(self.offloading_agent.critic_opt, self.device)

        meta = ckpt.get('meta', {})
        self._deployment_update_steps = meta.get('deployment_updates', 0)
        self._offloading_update_steps = meta.get('offloading_updates', 0)
        self._epoch = int(meta.get('epoch', self._epoch))

        LOGGER.info(f'[Hedger] Checkpoint loaded from {target_path} (epoch={self._epoch})')

    @staticmethod
    def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device):
        """Ensure optimizer state tensors are on the same device as the model."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


    def _map_deployment_plan_to_deployment_mask(self, deployment_plan:dict):
        """
        Convert deployment plan dict to deployment mask tensor.
        deployment_plan: dict mapping service_name to device_name
        Returns:
            deploy_mask: Tensor of shape (num_services, num_devices), bool
        """
        num_services = len(self.logical_topology)
        num_devices = len(self.physical_topology)
        deploy_mask = torch.zeros((num_services, num_devices), dtype=torch.bool, device=self.device)

        for service_name, device_name in deployment_plan.items():
            s_idx = self.logical_topology.index(service_name)
            d_idx = self.physical_topology.index(device_name)
            deploy_mask[s_idx, d_idx] = True

        return deploy_mask

    @property
    def _ready_for_run(self):
        return self.physical_topology and self.logical_topology

    def run(self):
        while not self._ready_for_run:
            LOGGER.debug('[Hedger] Waiting for physical/logical topology information to start run Hedger..')
            time.sleep(0.5)

        if self.mode == 'train':
            self.train_hedger()
        elif self.mode == 'inference':
            self.inference_hedger()
        else:
            raise ValueError(f'Unsupported mode {self.mode} for Hedger, only "train" and "inference" are supported.')
