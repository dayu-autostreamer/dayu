from typing import List, Optional
import threading
import random
import torch
import time
import os
import glob

from core.lib.common import LOGGER, FileOps, Context

from .topology_encoder import TopologyEncoders
from .ppo_agent import HedgerOffloadPPO, HedgerDeploymentPPO
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

        self.offloading_agent_params = agent_params['offloading_agent']
        self.deployment_agent_params = agent_params['deployment_agent']

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

        FileOps.create_directory(self.model_dir)
        if self.load_model:
            self.load_checkpoint(epoch=self.load_epoch)

        self.deployment_decision = None
        self.offloading_decision = None

        self.deployment_transitions: List[dict] = []
        self.offloading_transitions: List[dict] = []

        self.prev_deploy_mask = None

        self.run()

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

        self.offloading_agent = HedgerOffloadPPO(
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

    def register_logical_topology(self, dag):
        if self.logical_topology:
            return

        self.logical_topology = LogicalTopology(dag)

    def get_offloading_decision(self):
        return self.offloading_decision

    def get_initial_deployment_decision(self):
        return self.deployment_decision

    def get_redeployment_decision(self):
        return self.deployment_decision

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
        LOGGER.info('[Hedger] Hedger is running in training mode..')
        self.set_seed()

        logic_links = torch.tensor(self.logical_topology.links,
                                   dtype=torch.long, device=self.device).t().contiguous()
        phys_links = torch.tensor(self.physical_topology.links,
                                  dtype=torch.long, device=self.device).t().contiguous()

        self.prev_deploy_mask = torch.zeros((len(self.logical_topology), len(self.physical_topology)),
                                            dtype=torch.bool, device=self.device)
        self.prev_deploy_mask[:, self.physical_topology.cloud_idx] = True

        threading.Thread(target=self.train_deployment_agent, daemon=True).start()
        threading.Thread(target=self.train_offloading_agent, daemon=True).start()

        while True:
            time.sleep(1)

            updates_in_tick = 0
            if len(self.offloading_transitions) >= 32:
                off_transitions = self.offloading_transitions.copy()
                self.offloading_transitions.clear()
                self.offloading_agent.ppo_update(off_transitions,
                                                 epochs=self.update_epochs, batch_size=16)
                self._offloading_update_steps += 1
                updates_in_tick += 1

            if len(self.deployment_transitions) >= 8:
                dep_transitions = self.deployment_transitions.copy()
                self.deployment_transitions.clear()
                self.deployment_agent.ppo_update(dep_transitions,
                                                 epochs=self.update_epochs, batch_size=4)
                self._deployment_update_steps += 1
                updates_in_tick += 1

            # Epoch-based checkpointing: epoch counts number of PPO updates across agents
            if updates_in_tick > 0:
                prev_epoch = self._epoch
                self._epoch += updates_in_tick
                # Save when crossing a multiple of save_interval
                if (prev_epoch // self.save_interval) != (self._epoch // self.save_interval):
                    try:
                        self.save_checkpoint(epoch=self._epoch)
                    except Exception as e:
                        LOGGER.warning(f'[Hedger] Failed to save checkpoint (epoch {self._epoch}): {e}')
                        LOGGER.exception(e)

        LOGGER.info('[Hedger] Training of Hedger finished.')

    def train_deployment_agent(self):
        LOGGER.info('[Hedger Deployment] Hedger Deployment Agent start training.')

    def train_offloading_agent(self):
        LOGGER.info('[Hedger Offloading] Hedger Offloading Agent start training.')

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
