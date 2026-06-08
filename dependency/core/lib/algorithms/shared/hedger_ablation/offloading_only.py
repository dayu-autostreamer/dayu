from dataclasses import replace

from core.lib.common import LOGGER
from core.lib.algorithms.shared.hedger import Hedger

from .deployment_support import HedgerHeuristicDeploymentMixin


class HedgerOffloadingOnly(HedgerHeuristicDeploymentMixin, Hedger):
    """Use learned Hedger offloading while replacing deployment PPO with a heuristic."""

    def get_initial_deployment_plan(self):
        return self.set_heuristic_deployment_plan(default_deployment=self.initial_deployment_plan, mark_version=False)

    def get_redeployment_plan(self):
        return self.set_heuristic_deployment_plan(default_deployment=self.initial_deployment_plan, mark_version=True)

    def inference_hedger(self):
        if not self.inference_cfg.run_offloading_worker:
            raise RuntimeError(
                "[HedgerOffloadingOnly][Inference] inference.run_offloading_worker must be true because "
                "the offloading policy is the learned component in this ablation."
            )
        if self.inference_cfg.run_deployment_worker:
            LOGGER.info(
                "[HedgerOffloadingOnly][Inference] Disable learned deployment worker; "
                "the deployment policies serve heuristic deployment decisions."
            )
            self.inference_cfg = replace(self.inference_cfg, run_deployment_worker=False)
        if self.cur_deploy_mask is None:
            self.set_heuristic_deployment_plan(default_deployment=self.initial_deployment_plan, mark_version=False)
        return super().inference_hedger()
