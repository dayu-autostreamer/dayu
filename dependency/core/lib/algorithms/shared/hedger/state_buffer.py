class StateBuffer:

    LAN_BANDWIDTH = 1000  # in Mbps

    def __init__(self, max_capacity, logical_topology=None, physical_topology=None):
        self.max_capacity = max_capacity
        self.logical_topology = logical_topology
        self.physical_topology = physical_topology

        self.model_flops_buffer = {}
        self.model_memory_buffer = {}
        self.task_complexity_buffer = {}
        self.task_latency_buffer = {}
        self.gpu_flops_buffer = {}
        self.memory_capacity_buffer = {}
        self.device_role_buffer = {}
        self.bandwidth_buffer = {}
        self.gpu_utilization_buffer = {}
        self.memory_utilization_buffer = {}

        self.init_buffer()

    def init_buffer(self):
        self.model_flops_buffer = [0.0 for _ in range(self.logical_topology.node_num)]
        self.model_memory_buffer = [0.0 for _ in range(self.logical_topology.node_num)]
        self.task_complexity_buffer = [[] for _ in range(self.logical_topology.node_num)]
        self.task_latency_buffer = [[] for _ in range(self.logical_topology.node_num)]
        self.gpu_flops_buffer = [0.0 for _ in range(self.physical_topology.node_num)]
        self.memory_capacity_buffer = [0.0 for _ in range(self.physical_topology.node_num)]
        self.device_role_buffer = [0]+[1]*(self.physical_topology.node_num-2)+[2]
        self.bandwidth_buffer = [[] for _ in range(self.physical_topology.node_num)]
        self.gpu_utilization_buffer = [[] for _ in range(self.physical_topology.node_num)]
        self.memory_utilization_buffer = [[] for _ in range(self.physical_topology.node_num)]

    def add_model_flops(self, service, flops):
        self.model_flops_buffer[self.logical_topology.index(service)] = flops

    def add_model_memory(self, service, memory):
        self.model_memory_buffer[self.logical_topology.index(service)] = memory

    def add_task_complexity(self, service, complexity):
        self.task_complexity_buffer[self.logical_topology.index(service)].append(complexity)

    def add_task_latency(self, service, latency):
        self.task_latency_buffer[self.logical_topology.index(service)].append(latency)

    def add_gpu_flops(self, device, flops):
        self.gpu_flops_buffer[self.physical_topology.index(device)] = flops

    def add_memory_capacity(self, device, capacity):
        self.memory_capacity_buffer[self.physical_topology.index(device)] = capacity

    def add_bandwidths(self, bandwidth):
        self.bandwidth_buffer[self.physical_topology.cloud_idx].append(bandwidth)
        for i in range(self.physical_topology.node_num-1):
            self.bandwidth_buffer[i].append(self.LAN_BANDWIDTH)

    def add_gpu_utilization(self, device, utilization):
        self.gpu_utilization_buffer[self.physical_topology.index(device)].append(utilization)

    def add_memory_utilization(self, device, utilization):
        self.memory_utilization_buffer[self.physical_topology.index(device)].append(utilization)
