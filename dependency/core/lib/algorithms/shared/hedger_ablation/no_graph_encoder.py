from core.lib.algorithms.shared.hedger import Hedger

from .no_graph_topology_encoder import NoGraphTopologyEncoders


class HedgerNoGraphEncoder(Hedger):
    """Topology-encoder ablation that removes graph message passing."""

    def register_topology_encoder(self):
        if self.shared_topology_encoder:
            return

        self.shared_topology_encoder = NoGraphTopologyEncoders(
            d_model=self.encoder_cfg.embedding_dim,
            num_roles=getattr(self.encoder_cfg, "physical_role_count", 2),
            role_emb_dim=getattr(self.encoder_cfg, "physical_role_embedding_dim", 8),
            dropout=self.encoder_cfg.dropout,
        ).to(self.device)

    def _load_encoder_state(self, state_dict: dict) -> None:
        self._load_state_dict_compatible(
            self.shared_topology_encoder,
            state_dict,
            "no_graph_encoder",
        )
