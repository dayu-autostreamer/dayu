import threading
import random
import torch
import time

from core.lib.common import LOGGER

from topology_encoder import TopologyEncoders
from ppo_agent import HedgerOffloadPPO, HedgerDeploymentPPO
from hedger_config import from_partial_dict, OffloadingConstraintCfg, DeploymentConstraintCfg, LogicalTopology, \
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

        self.offloading_agent_params = agent_params['offloading_agent']
        self.deployment_agent_params = agent_params['deployment_agent']

        self.physical_topology = None
        self.logical_topology = None

        self.shared_topology_encoder = None
        self.deployment_agent = None
        self.offloading_agent = None

        self.register_topology_encoder()
        self.register_deployment_agent()
        self.register_offloading_agent()

        self.deployment_decision = None
        self.offloading_decision = None

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
        if self.offloading_agent:
            return

        assert self.shared_topology_encoder, 'Shared topology encoder must be registered before deployment agent.'

        self.deployment_agent = HedgerDeploymentPPO(
            encoder=self.shared_topology_encoder,
            d_model=self.encoder_network_params['encoder_emb_dim'],
            actor_lr=self.deployment_agent['actor_lr'],
            critic_lr=self.deployment_agent['critic_lr'],
            gamma=self.deployment_agent['gamma'],
            lamda=self.deployment_agent['lamda'],
            clip_eps=self.deployment_agent['clip_eps'],
            update_encoder=self.deployment_agent['update_encoder'],
            constraint_cfg=from_partial_dict(DeploymentConstraintCfg, self.deployment_agent_params),
        ).to(self.device)

    def register_offloading_agent(self):
        if self.offloading_agent:
            return

        assert self.shared_topology_encoder, 'Shared topology encoder must be registered before offloading agent.'

        self.offloading_agent = HedgerOffloadPPO(
            encoder=self.shared_topology_encoder,
            d_model=self.encoder_network_params['encoder_emb_dim'],
            actor_lr=self.offloading_agent['actor_lr'],
            critic_lr=self.offloading_agent['critic_lr'],
            gamma=self.offloading_agent['gamma'],
            lamda=self.offloading_agent['lamda'],
            clip_eps=self.offloading_agent['clip_eps'],
            update_encoder=self.offloading_agent['update_encoder'],
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

    def inference_deployment_agent(self):
        LOGGER.info('[Hedger Deployment] Hedger Deployment Agent start inference.')

    def inference_offloading_agent(self):
        LOGGER.info('[Hedger Offloading] Hedger Offloading Agent start inference.')

    def train_deployment_agent(self):
        LOGGER.info('[Hedger Deployment] Hedger Deployment Agent start training.')

    def train_offloading_agent(self):
        LOGGER.info('[Hedger Offloading] Hedger Offloading Agent start training.')

    @property
    def _ready_for_run(self):
        return self.physical_topology and self.logical_topology

    def run(self):
        while not self._ready_for_run:
            LOGGER.debug('[Hedger] Waiting for physical/logical topology information to start run Hedger..')
            time.sleep(0.5)

        if self.mode == 'train':
            LOGGER.info('[Hedger] Hedger is running in training mode..')
            self.set_seed()
            threading.Thread(self.train_deployment_agent).start()
            threading.Thread(self.train_offloading_agent).start()
        elif self.mode == 'inference':
            LOGGER.info('[Hedger] Hedger is running in inference mode..')
            threading.Thread(self.inference_deployment_agent).start()
            threading.Thread(self.inference_offloading_agent).start()
        else:
            raise ValueError(f'Unsupported mode {self.mode} for Hedger, only "train" and "inference" are supported.')
