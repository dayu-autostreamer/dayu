import threading
import time

from core.lib.common import LOGGER
from core.lib.algorithms.shared.hedger import Hedger

from .deployment_support import HedgerHeuristicDeploymentMixin


class HedgerOffloadingOnly(HedgerHeuristicDeploymentMixin, Hedger):
    """Train Hedger offloading while replacing deployment PPO with a heuristic."""

    def get_initial_deployment_plan(self):
        return self.set_heuristic_deployment_plan(default_deployment=self.initial_deployment_plan, mark_version=False)

    def get_redeployment_plan(self):
        return self.set_heuristic_deployment_plan(default_deployment=self.initial_deployment_plan, mark_version=True)

    def inference_hedger(self):
        assert self.logical_topology is not None and self.physical_topology is not None and self.state_buffer is not None
        if self.checkpoint_cfg.load.enabled and self._loaded_checkpoint_path is None:
            raise RuntimeError(
                "[HedgerOffloadingOnly][Inference] Checkpoint loading was enabled but no checkpoint was loaded."
            )
        if not self.checkpoint_cfg.load.enabled:
            raise RuntimeError(
                "[HedgerOffloadingOnly][Inference] checkpoint.load.enabled must be true because "
                "the offloading policy is learned."
            )

        LOGGER.info(
            f"[HedgerOffloadingOnly][Inference] Start: "
            f"{self._summarize_runtime_config()}, {self._summarize_topology()}"
        )
        self.set_seed()
        self.shared_topology_encoder.eval()
        self.offloading_agent.eval()
        self.deployment_thread_stop_event.clear()
        self.offloading_thread_stop_event.clear()

        if self.cur_deploy_mask is None:
            self.set_heuristic_deployment_plan(default_deployment=self.initial_deployment_plan, mark_version=False)
        elif self.deployment_plan is None:
            self.deployment_plan = self._map_deployment_mask_to_deployment_plan(self._current_deploy_mask())

        worker = threading.Thread(target=self.inference_offloading_agent, daemon=True)
        worker.start()
        while not self.offloading_thread_stop_event.is_set():
            if not worker.is_alive():
                LOGGER.warning("[HedgerOffloadingOnly][Inference] Offloading worker stopped unexpectedly.")
                break
            time.sleep(0.5)
        self.deployment_thread_stop_event.set()
        self.offloading_thread_stop_event.set()
