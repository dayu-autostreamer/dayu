import threading

from topology_encoder import TopologyEncoders
from ppo_agent import HedgerOffloadPPO, HedgerDeploymentPPO

__all__ = ('Hedger',)


class Hedger:
    def __init__(self):
        self.shared_topology_encoder = None
        self.deployment_agent = None
        self.offloading_agent = None

    def get_offloading_decision(self):
        pass

    def get_initial_deployment_decision(self):
        pass

    def get_redeployment_decision(self):
        pass

    def run_deployment(self):
        pass

    def run_offloading(self):
        pass

    def run(self):
        threading.Thread(target=self.run_deployment).start()
        threading.Thread(target=self.run_offloading).start()
