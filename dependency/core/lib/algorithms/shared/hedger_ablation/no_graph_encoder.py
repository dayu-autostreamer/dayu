from core.lib.algorithms.shared.hedger import Hedger
from core.lib.common import LOGGER

from .no_graph_topology_encoder import NoGraphTopologyEncoders


class HedgerNoGraphEncoder(Hedger):
    """Ablation that removes learned service/device topology embeddings."""

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
        if state_dict:
            LOGGER.info(
                "[HedgerNoGraphEncoder][Checkpoint] Skip encoder state loading because "
                "this ablation has no trainable topology encoder."
            )
