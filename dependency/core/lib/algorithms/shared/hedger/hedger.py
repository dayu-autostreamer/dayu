import copy
from typing import List, Optional
import threading
import random
import torch
import time
import os
import glob

from core.lib.common import LOGGER, FileOps, Context, Recorder

from .topology_encoder import TopologyEncoders
from .ppo_agent import HedgerOffloadingPPO, HedgerDeploymentPPO
from .hedger_config import from_partial_dict, OffloadingConstraintCfg, DeploymentConstraintCfg, LogicalTopology, \
    PhysicalTopology
from .state_buffer import StateBuffer

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

        self.load_encoder_flag = hyper_params.get("load_encoder", True)
        self.load_deployment_agent_flag = hyper_params.get("load_deployment_agent", True)
        self.load_offloading_agent_flag = hyper_params.get("load_offloading_agent", True)

        self.load_optimizer_flag = hyper_params.get("load_optimizer", True)

        self.reset_steps_on_load = hyper_params.get("reset_steps_on_load", False)

        self.train_deployment_flag = hyper_params.get("train_deployment", True)
        self.train_offloading_flag = hyper_params.get("train_offloading", True)

        self.offloading_rollout_len = hyper_params.get("offloading_rollout_len", 32)
        self.deployment_rollout_len = hyper_params.get("deployment_rollout_len", 8)
        self.offloading_batch_size = hyper_params.get("offloading_batch_size", 16)
        self.deployment_batch_size = hyper_params.get("deployment_batch_size", 4)

        self.max_state_buffer_size = hyper_params.get("max_state_buffer_size", 1000)

        self.offloading_agent_params = agent_params['offloading_agent']
        self.deployment_agent_params = agent_params['deployment_agent']

        self.deployment_thread_stop_event = threading.Event()
        self.offloading_thread_stop_event = threading.Event()

        self.physical_topology = None
        self.logical_topology = None

        self.shared_topology_encoder = None
        self.deployment_agent = None
        self.offloading_agent = None

        self.state_buffer = None

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

        self.initial_deployment_plan = None
        self.deployment_plan = None
        self.offloading_plan = None

        self.deployment_transitions: List[dict] = []
        self.offloading_transitions: List[dict] = []

        self.dep_recorder = None
        self.off_recorder = None

        self.cur_deploy_mask = None

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

    def register_state_buffer(self):
        if self.state_buffer:
            return

        assert self.logical_topology, "Logical topology must be registered before registering state buffer."
        assert self.physical_topology, "Physical topology must be registered before registering state buffer."

        self.state_buffer = StateBuffer(self.max_state_buffer_size,
                                        logical_topology=self.logical_topology,
                                        physical_topology=self.physical_topology)

    def register_initial_deployment(self, deployment_plan):
        assert self.logical_topology, "Logical topology must be registered before registering initial deployment."
        assert self.physical_topology, "Physical topology must be registered before registering initial deployment."

        if self.initial_deployment_plan:
            return
        self.initial_deployment_plan = deployment_plan
        self.deployment_plan = deployment_plan

        # update deploy mask
        self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(deployment_plan)

    def get_offloading_plan(self):
        return self.offloading_plan

    def get_initial_deployment_plan(self):
        return self.initial_deployment_plan

    def get_redeployment_plan(self):
        self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(self.deployment_plan)
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

        self.cur_deploy_mask = torch.zeros((len(self.logical_topology), len(self.physical_topology)),
                                           dtype=torch.bool, device=self.device)
        self.cur_deploy_mask[:, self.physical_topology.cloud_idx] = True

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

        if not self.cur_deploy_mask:
            LOGGER.warning('No previous deployment mask found, initialize to pure cloud deployment.')
            self.cur_deploy_mask = torch.zeros((len(self.logical_topology), len(self.physical_topology)),
                                               dtype=torch.bool, device=self.device)
            self.cur_deploy_mask[:, self.physical_topology.cloud_idx] = True

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
                    # Save once when _epoch crosses a multiple of save_interval
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
        Sampling thread for deploying the agent:
            - Collect current topology state periodically
            - Call deployment_agent.policy to sample the new deployment mask
            - Let the environment run for a period of time based on deployment, count metrics, and calculate reward
            - Put transition into self.deployment_transitions and wait for the main thread PPO update
        """
        if not self.train_deployment_flag:
            LOGGER.info("train_deployment_flag=False, deployment training thread will not run.")
            return

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting deployment training."

        LOGGER.info('[Hedger Deployment] Hedger Deployment Agent start training.')

        self.dep_recorder = Recorder(
            "deployment_train.csv",
            fmt="csv",
            fieldnames=["step", "epoch", "dep_updates", "dep_reward", "avg_off_reward",
                        "dep_change_cost", "cap_relax_cnt", "dep_offload_weight",
                        "dep_change_weight", "cap_relax_weight"],
            overwrite=True,
            flush_every=1,
        )

        # 静态 edge_index，可以一直复用
        logic_edge_index = torch.tensor(self.logical_topology.links, dtype=torch.long,
                                        device=self.device).t().contiguous()
        phys_edge_index = torch.tensor(self.physical_topology.links, dtype=torch.long,
                                       device=self.device).t().contiguous()

        step = 0
        deployment_time_ticket = 0

        prev_deploy_mask = copy.deepcopy(self.cur_deploy_mask)
        logic_feats, phys_feats, _, _ = self._collect_deployment_state()

        while not self.deployment_thread_stop_event.is_set():
            # map features to device
            logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
            phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
            prev_deploy_mask_dev = prev_deploy_mask.to(self.device) if prev_deploy_mask is not None else None

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
            deploy_plan = self._map_deployment_mask_to_deployment_plan(deploy_mask)
            self.deployment_plan = deploy_plan

            time_ticket = time.time()
            if deployment_time_ticket == 0:
                time.sleep(self.deployment_interval)
            else:
                elapsed = time_ticket - deployment_time_ticket
                time.sleep(max(0, self.deployment_interval - elapsed))
            deployment_time_ticket = time_ticket

            new_logic_feats, new_phys_feats, metrics, done = self._collect_deployment_state()

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
                "prev_deploy_mask": prev_deploy_mask.cpu() if prev_deploy_mask is not None else None,
                "logp": logp.detach().cpu(),
                "value": value.detach().cpu(),
                "reward": float(reward),
                "done": bool(done),
            }

            with self._data_lock:
                self.deployment_transitions.append(tr)

            logic_feats = new_logic_feats
            phys_feats = new_phys_feats
            prev_deploy_mask = deploy_mask.detach().cpu()

            self.dep_recorder.log(
                step=step,
                epoch=self._epoch,
                dep_updates=self._deployment_update_steps,
                dep_reward=reward,
                avg_off_reward=metrics["avg_offloading_reward"],
                dep_change_cost=metrics["deploy_change_cost"],
                cap_relax_cnt=aux["capacity_relax_cnt"],
                dep_offload_weight=self.deployment_agent_params["reward_dep_offload_weight"],
                dep_change_weight=self.deployment_agent_params["reward_dep_change_weight"],
                cap_relax_weight=self.deployment_agent_params["penalty_capacity_relax"],
            )

            step += 1

        self.dep_recorder.close()
        LOGGER.info("Deployment training thread stopped.")

    def train_offloading_agent(self):
        """
        Uninstall the agent's sampling thread:
            - Collect current topology and system state at short intervals (each task arrival/small time window)
            - Call offloading_agent.policy to determine the execution device for each service
            - The environment implements this policy and returns metrics such as latency / SLO / cloud usage in this window
            - Calculate reward and put transition into self.offloading_transitions
        """
        if not self.train_offloading_flag:
            LOGGER.info("train_offloading_flag=False, offloading training thread will not run.")
            return

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting offloading training."

        LOGGER.info('[Hedger Offloading] Hedger Offloading Agent start training.')

        self.off_recorder = Recorder(
            "offloading_train.csv",
            fmt="csv",
            fieldnames=["step", "epoch", "off_updates", "off_reward", "latency",
                        "slo_violation", "cloud_fraction", "aux_cost", "off_latency_weight",
                        "off_slo_weight", "off_cloud_weight", "switch_cnt", "relax_cnt",
                        "switch_weight", "relax_weight"],
            overwrite=True,
            flush_every=10,
        )

        logic_edge_index = torch.tensor(self.logical_topology.links, dtype=torch.long,
                                        device=self.device).t().contiguous()
        phys_edge_index = torch.tensor(self.physical_topology.links, dtype=torch.long,
                                       device=self.device).t().contiguous()

        step = 0
        offloading_time_ticket = 0
        static_mask = copy.deepcopy(self.cur_deploy_mask)
        logic_feats, phys_feats, _, _ = self._collect_offloading_state()
        while not self.offloading_thread_stop_event.is_set():
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
                offloading_plan = self._map_offloading_mask_to_offloading_plan(actions)
                self.offloading_plan = offloading_plan

            time_ticket = time.time()
            if offloading_time_ticket == 0:
                time.sleep(self.offloading_interval)
            else:
                elapsed = time_ticket - offloading_time_ticket
                time.sleep(max(0, self.offloading_interval - elapsed))
            offloading_time_ticket = time_ticket

            new_logic_feats, new_phys_feats, metrics, done = self._collect_offloading_state()

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

            logic_feats = new_logic_feats
            phys_feats = new_phys_feats

            self.off_recorder.log(
                step=step,
                epoch=self._epoch,
                off_updates=self._offloading_update_steps,
                off_reward=reward,
                latency=metrics["latency"],
                slo_violation=metrics["slo_violation"],
                cloud_fraction=metrics["cloud_fraction"],
                aux_cost=aux["aux_cost"],
                off_latency_weight=self.offloading_agent_params["reward_off_latency_weight"],
                off_slo_weight=self.offloading_agent_params["reward_off_slo_weight"],
                off_cloud_weight=self.offloading_agent_params["reward_off_cloud_weight"],
                switch_cnt=aux["switch_cnt"],
                relax_cnt=aux["relax_cnt"],
                switch_weight=self.offloading_agent_params["penalty_switch"],
                relax_weight=self.offloading_agent_params["penalty_relax"],
            )

            step += 1

        self.off_recorder.close()
        LOGGER.info("Offloading training thread stopped.")

    def _compute_offloading_reward(self, metrics, aux) -> float:
        """
        Offloading agent reward:
        Defined in the form of "negative delay + penalty".
        Metrics should include some basic performance indicators, aux from within the offloading policy (penalty of switch/relax).
        """
        metrics = metrics or {}

        # Extract key metrics from metrics
        latency = float(metrics["latency"])
        slo_v = float(metrics["slo_violation"])
        cloud_frac = float(metrics["cloud_fraction"])

        # The weight is read from hyper_params. If not, use the default.
        w_lat = float(self.offloading_agent_params["reward_off_latency_weight"])
        w_slo = float(self.offloading_agent_params["reward_off_slo_weight"])
        w_cloud = float(self.offloading_agent_params["reward_off_cloud_weight"])

        # Basic reward: Tending to reduce latency, reduce SLO breaches, reduce cloud adoption ratio
        reward = 0.0
        reward -= w_lat * latency
        reward -= w_slo * slo_v
        reward -= w_cloud * cloud_frac

        # Additional Cost from Constraints (Number of Switches + Number of Relaxations)
        aux_cost = float(aux["aux_cost"])
        reward -= aux_cost

        return reward

    def _compute_deployment_reward(self, metrics, aux) -> float:
        """
        Deployment agent reward:
            - Utilizing macro performance indicators (average latency, SLO violation, cloud usage)
            - Plus aggregated feedback from offloading (avg_offloading_reward)
            - Plus deployment change cost + capacity relax penalty, achieving "top-down + bottom-up two-way feedback"
        """
        metrics = metrics or {}

        avg_off_r = float(metrics["avg_offloading_reward"])
        deploy_change_cost = float(metrics["deploy_change_cost"])

        w_change = float(self.deployment_agent_params["reward_dep_change_weight"])
        w_off = float(self.deployment_agent_params["reward_dep_offload_weight"])

        reward = 0.0
        reward -= w_change * deploy_change_cost

        # Using the average reward of the lower-level agent as the "bottom-up" feedback signal
        reward += w_off * avg_off_r

        # Penalty capacity relaxation
        cap_relax_cnt = float(aux["capacity_relax_cnt"])
        penalty_capacity_relax = float(self.deployment_agent_params["penalty_capacity_relax"])
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
        Based on the switches in hyper_params, it allows:
            - Load only the encoder
            - Load only a specific agent
            - Choose whether to load optimizer state
            - Choose whether to reset epoch / update step counter
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
        LOGGER.info(f'[Hedger] Loading checkpoint from {target_path}')

        # Load encoder parameters
        if self.load_encoder_flag:
            enc_state = ckpt.get('encoder', None)
            if enc_state:
                self.shared_topology_encoder.load_state_dict(enc_state)
                LOGGER.info('[Hedger] Loaded encoder state from checkpoint.')
            else:
                LOGGER.warning('[Hedger] No encoder state found in checkpoint.')
        else:
            LOGGER.info('[Hedger] Skipping encoder loading per config (load_encoder=False).')

        # Load dual agent parameters
        # Deployment agent
        if self.load_deployment_agent_flag:
            dep_state = ckpt.get('deployment_agent', None)
            if dep_state:
                self.deployment_agent.load_state_dict(dep_state, strict=False)
                LOGGER.info('[Hedger] Loaded deployment agent state from checkpoint.')
            else:
                LOGGER.warning('[Hedger] No deployment_agent state found in checkpoint.')
        else:
            LOGGER.info('[Hedger] Skipping deployment agent loading per config (load_deployment_agent=False).')

        # Offloading agent
        if self.load_offloading_agent_flag:
            off_state = ckpt.get('offloading_agent', None)
            if off_state:
                self.offloading_agent.load_state_dict(off_state, strict=False)
                LOGGER.info('[Hedger] Loaded offloading agent state from checkpoint.')
            else:
                LOGGER.warning('[Hedger] No offloading_agent state found in checkpoint.')
        else:
            LOGGER.info('[Hedger] Skipping offloading agent loading per config (load_offloading_agent=False).')

        # Load optimizer status
        if self.load_optimizer_flag:
            # Load the corresponding optimizers for those agents that actually loaded
            if self.load_deployment_agent_flag and 'deployment_actor_opt' in ckpt:
                self.deployment_agent.actor_opt.load_state_dict(ckpt['deployment_actor_opt'])
                self._move_optimizer_state(self.deployment_agent.actor_opt, self.device)
                LOGGER.info('[Hedger] Loaded deployment actor optimizer state.')

            if self.load_deployment_agent_flag and 'deployment_critic_opt' in ckpt:
                self.deployment_agent.critic_opt.load_state_dict(ckpt['deployment_critic_opt'])
                self._move_optimizer_state(self.deployment_agent.critic_opt, self.device)
                LOGGER.info('[Hedger] Loaded deployment critic optimizer state.')

            if self.load_offloading_agent_flag and 'offloading_actor_opt' in ckpt:
                self.offloading_agent.actor_opt.load_state_dict(ckpt['offloading_actor_opt'])
                self._move_optimizer_state(self.offloading_agent.actor_opt, self.device)
                LOGGER.info('[Hedger] Loaded offloading actor optimizer state.')

            if self.load_offloading_agent_flag and 'offloading_critic_opt' in ckpt:
                self.offloading_agent.critic_opt.load_state_dict(ckpt['offloading_critic_opt'])
                self._move_optimizer_state(self.offloading_agent.critic_opt, self.device)
                LOGGER.info('[Hedger] Loaded offloading critic optimizer state.')
        else:
            LOGGER.info('[Hedger] Skipping optimizer states loading per config (load_optimizer=False).')

        # Load / Reset counter
        meta = ckpt.get('meta', {})

        if self.reset_steps_on_load:
            # Reset counters in Phase 2/3 training
            self._deployment_update_steps = 0
            self._offloading_update_steps = 0
            self._epoch = 0
            LOGGER.info('[Hedger] Reset update counters and epoch to 0 per config (reset_steps_on_load=True).')
        else:
            self._deployment_update_steps = meta.get('deployment_updates', 0)
            self._offloading_update_steps = meta.get('offloading_updates', 0)
            self._epoch = int(meta.get('epoch', self._epoch))
            LOGGER.info(f'[Hedger] Restored counters from checkpoint: '
                        f'dep_updates={self._deployment_update_steps}, '
                        f'off_updates={self._offloading_update_steps}, '
                        f'epoch={self._epoch}')

        LOGGER.info(f'[Hedger] Checkpoint loaded from {target_path}')

    @staticmethod
    def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device):
        """Ensure optimizer state tensors are on the same device as the model."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def _map_deployment_plan_to_deployment_mask(self, deployment_plan: dict):
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

    def _map_deployment_mask_to_deployment_plan(self, deploy_mask: torch.Tensor):
        """
        Convert deployment mask tensor to deployment plan dict.
        deploy_mask: Tensor of shape (num_services, num_devices), bool
        Returns:
            deployment_plan: dict mapping service_name to device_name
        """
        deployment_plan = {}
        num_services = deploy_mask.size(0)
        num_devices = deploy_mask.size(1)

        for s_idx in range(num_services):
            for d_idx in range(num_devices):
                if deploy_mask[s_idx, d_idx]:
                    service_name = self.logical_topology[s_idx]
                    device_name = self.physical_topology[d_idx]
                    if service_name not in deployment_plan:
                        deployment_plan[service_name] = [device_name]
                    else:
                        deployment_plan[service_name].append(device_name)

        return deployment_plan

    def _map_offloading_mask_to_offloading_plan(self, offloading_mask: torch.Tensor):
        """
        Convert offloading mask tensor to offloading plan dict.
        offloading_mask: Tensor of shape (num_services,), int (device indices)
        Returns:
            offloading_plan: dict mapping service_name to device_name
        """
        offloading_plan = {}
        num_services = offloading_mask.size(0)

        for s_idx in range(num_services):
            d_idx = offloading_mask[s_idx].item()
            service_name = self.logical_topology[s_idx]
            device_name = self.physical_topology[d_idx]
            offloading_plan[service_name] = device_name

        return offloading_plan

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
