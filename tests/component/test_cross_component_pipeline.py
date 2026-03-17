import copy
import importlib
import json
import threading
from pathlib import Path
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient

from core.lib.common import Queue
from core.lib.content import Task


GeneratorBase = importlib.import_module("core.generator.generator").Generator


def build_single_service_task():
    dag = Task.extract_dag_from_dag_deployment(
        {
            "face-detection": {
                "service": {"service_name": "face-detection", "execute_device": "edge-node"},
                "next_nodes": [],
            }
        }
    )
    return Task(
        source_id=0,
        task_id=0,
        source_device="edge-node",
        all_edge_devices=["edge-node"],
        dag=dag,
        metadata={"buffer_size": 1},
        raw_metadata={"buffer_size": 1},
        file_path="payload.bin",
    )


class DummyGenerator(GeneratorBase):
    def run(self):
        raise NotImplementedError


class FakeScheduler:
    def __init__(self):
        self.schedule_calls = []
        self.resource_table = {}
        self.scenario_tasks = []
        self.resource_updates = []

    def register_schedule_table(self, source_id):
        return None

    def get_schedule_plan(self, info):
        self.schedule_calls.append(copy.deepcopy(info))
        return {
            "dag": {
                "face-detection": {
                    "service": {"service_name": "face-detection", "execute_device": "edge-node"},
                    "next_nodes": [],
                }
            },
            "buffer_size": info["meta_data"]["buffer_size"],
        }

    def get_schedule_overhead(self):
        return 0.0

    def update_scheduler_scenario(self, task):
        self.scenario_tasks.append(task)

    def register_resource_table(self, device):
        self.resource_table.setdefault(device, {})

    def update_scheduler_resource(self, info):
        self.resource_updates.append(copy.deepcopy(info))
        self.resource_table[info["device"]] = info["resource"]

    def get_scheduler_resource(self):
        return self.resource_table

    async def get_resource_lock(self, info):
        return info["device"]

    def get_source_node_selection_plan(self, source_id, data):
        return data["node_set"][0]

    def get_initial_deployment_plan(self, source_id, data):
        return {"edge-node": ["face-detection"]}

    def get_redeployment_plan(self, source_id, data):
        return {"edge-node": ["face-detection"]}


class FakeProcessor:
    def __call__(self, task):
        task.set_current_content({"service": task.get_flow_index(), "detections": 1})
        task.add_scenario({"obj_num": 1})
        return task

    @property
    def flops(self):
        return 128.0


class FakeMonitorWorker(threading.Thread):
    def __init__(self, system, metric_name, value):
        super().__init__(daemon=True)
        self.system = system
        self.metric_name = metric_name
        self.value = value

    def run(self):
        self.system.resource_info[self.metric_name] = self.value


class ComponentRouter:
    def __init__(self, scheduler_server, controller_server, processor_server, distributor_server):
        self.scheduler_server = scheduler_server
        self.controller_server = controller_server
        self.processor_server = processor_server
        self.distributor_server = distributor_server

        self.scheduler_client = TestClient(scheduler_server.app)
        self.controller_client = TestClient(controller_server.app)
        self.processor_client = TestClient(processor_server.app)
        self.distributor_client = TestClient(distributor_server.app)

        self.client_by_port = {
            "9001": self.scheduler_client,
            "9002": self.controller_client,
            "9003": self.distributor_client,
            "9004": self.processor_client,
        }

    def close(self):
        self.scheduler_client.close()
        self.controller_client.close()
        self.processor_client.close()
        self.distributor_client.close()

    def request(self, url, method=None, no_decode=False, binary=True, **kwargs):
        method = method or "GET"
        parsed = urlparse(url)
        port = str(parsed.port)
        client = self.client_by_port[port]
        path = parsed.path or "/"
        response = client.request(method, path, **kwargs)
        assert response.status_code == 200, f"{method} {url} failed: {response.status_code} {response.text}"

        if port == "9004" and path == "/predict_local":
            while not self.processor_server.task_queue.empty():
                queued_task = self.processor_server.task_queue.get()
                processed_task = self.processor_server.process_task_service(queued_task)
                if processed_task:
                    self.processor_server.send_result_back_to_controller(processed_task)

        if no_decode:
            return response
        return response.json() if binary else response.content.decode("utf-8")


@pytest.mark.component
def test_generator_controller_processor_distributor_scheduler_pipeline(mounted_runtime, monkeypatch, tmp_path):
    generator_module = importlib.import_module("core.generator.generator")
    controller_module = importlib.import_module("core.controller.controller")
    controller_server_module = importlib.import_module("core.controller.controller_server")
    task_coordinator_module = importlib.import_module("core.controller.task_coordinator")
    network_module = importlib.import_module("core.lib.network")
    processor_server_module = importlib.import_module("core.processor.processor_server")
    distributor_module = importlib.import_module("core.distributor.distributor")
    distributor_server_module = importlib.import_module("core.distributor.distributor_server")
    scheduler_server_module = importlib.import_module("core.scheduler.scheduler_server")

    monkeypatch.chdir(tmp_path)
    Path("payload.bin").write_bytes(b"frame-bytes")

    monkeypatch.setenv("NODE_NAME", "edge-node")
    monkeypatch.setenv("ALL_EDGE_DEVICES", "['edge-node']")
    monkeypatch.setenv("REQUEST_SCHEDULING_INTERVAL", "1")
    monkeypatch.setenv("GUNICORN_PORT", "9004")
    monkeypatch.setenv("DISPLAY", "True")
    monkeypatch.setenv("MONITORS", "[]")

    for module in (
        generator_module,
        controller_module,
        task_coordinator_module,
        processor_server_module,
        distributor_module,
    ):
        monkeypatch.setattr(module.NodeInfo, "get_local_device", staticmethod(lambda: "edge-node"))
        monkeypatch.setattr(module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-node"))
        monkeypatch.setattr(module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: hostname))

    monkeypatch.setattr(controller_module.KubeConfig, "get_service_nodes_dict", staticmethod(lambda: {}))
    monkeypatch.setattr(scheduler_server_module.KubeConfig, "get_service_nodes_dict", staticmethod(lambda: {}))
    monkeypatch.setattr(
        network_module.PortInfo,
        "force_refresh",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr(
        network_module.PortInfo,
        "get_component_port",
        staticmethod(
            lambda component: {
                "controller": 9002,
                "distributor": 9003,
                "scheduler": 9001,
                "redis": 6379,
            }[component]
        ),
    )
    monkeypatch.setattr(
        network_module.PortInfo,
        "get_service_ports_dict",
        staticmethod(lambda device: {"face-detection": 9004}),
    )

    monkeypatch.setattr(processor_server_module.ProcessorServer, "loop_process", lambda self: None)
    monkeypatch.setattr(scheduler_server_module, "Scheduler", FakeScheduler)

    fake_queue = Queue()

    def fake_get_algorithm(algorithm, al_name=None, **kwargs):
        if algorithm == "GEN_BSO":
            return lambda system: {
                "source_id": system.source_id,
                "meta_data": system.raw_meta_data,
                "source_device": system.local_device,
                "all_edge_devices": system.all_edge_devices,
                "dag": Task.extract_dag_deployment_from_dag(system.task_dag),
            }
        if algorithm == "GEN_ASO":
            def after_schedule(system, scheduler_response):
                dag = Task.extract_dag_from_dag_deployment(scheduler_response["plan"]["dag"])
                dag.get_start_node().service.set_execute_device(system.local_device)
                dag.get_end_node().service.set_execute_device("cloud-node")
                system.task_dag = dag
                system.service_deployment = scheduler_response["deployment"]
                system.meta_data.update({"buffer_size": scheduler_response["plan"]["buffer_size"]})
            return after_schedule
        if algorithm == "GEN_BSTO":
            return lambda system, task: None
        if algorithm == "GEN_GETTER":
            return lambda system: None
        if algorithm == "PROCESSOR":
            return FakeProcessor()
        if algorithm == "PRO_QUEUE":
            return fake_queue
        raise AssertionError(f"Unexpected algorithm request: {algorithm}")

    monkeypatch.setattr(generator_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    monkeypatch.setattr(processor_server_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))

    scheduler_server = scheduler_server_module.SchedulerServer()
    controller_server = controller_server_module.ControllerServer()
    processor_server = processor_server_module.ProcessorServer()
    distributor_server = distributor_server_module.DistributorServer()

    router = ComponentRouter(scheduler_server, controller_server, processor_server, distributor_server)
    for module in (generator_module, controller_module, processor_server_module, distributor_module):
        monkeypatch.setattr(module, "http_request", router.request)

    try:
        generator = DummyGenerator(
            source_id=0,
            metadata={"buffer_size": 1},
            task_dag=Task.extract_dict_from_dag(build_single_service_task().get_dag()),
        )
        generator.request_schedule_policy()
        task = generator.generate_task(
            task_id=0,
            task_dag=generator.task_dag,
            service_deployment=generator.service_deployment,
            meta_data=generator.meta_data,
            compressed_path="payload.bin",
            hash_codes=[],
        )
        generator.record_total_start_ts(task)
        generator.submit_task_to_controller(task)

        query_response = router.distributor_client.get("/all_result")
        assert query_response.status_code == 200
        assert query_response.json()["size"] == 1

        stored_task = Task.deserialize(query_response.json()["result"][0])
        assert stored_task.get_current_service_info()[0] == "_end"
        assert stored_task.get_last_content() == {"service": "face-detection", "detections": 1}

        assert scheduler_server.scheduler.schedule_calls, "Generator should request a schedule plan"
        assert len(scheduler_server.scheduler.scenario_tasks) == 1
        scenario_task = scheduler_server.scheduler.scenario_tasks[0]
        assert scenario_task.get_scenario_data("face-detection") == {"obj_num": 1}
    finally:
        router.close()


@pytest.mark.component
def test_monitor_reports_resource_state_to_scheduler(mounted_runtime, monkeypatch):
    monitor_module = importlib.import_module("core.monitor.monitor")
    scheduler_server_module = importlib.import_module("core.scheduler.scheduler_server")

    monkeypatch.setenv("NODE_NAME", "edge-node")
    monkeypatch.setenv("INTERVAL", "0")
    monkeypatch.setenv("MONITORS", "['cpu_usage', 'memory_usage']")

    monkeypatch.setattr(monitor_module.NodeInfo, "get_local_device", staticmethod(lambda: "edge-node"))
    monkeypatch.setattr(monitor_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-node"))
    monkeypatch.setattr(monitor_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: hostname))
    monkeypatch.setattr(
        monitor_module.PortInfo,
        "get_component_port",
        staticmethod(lambda component: {"scheduler": 9001}[component]),
    )
    monkeypatch.setattr(scheduler_server_module, "Scheduler", FakeScheduler)

    def fake_get_algorithm(algorithm, al_name=None, **kwargs):
        if algorithm == "MON_PRAM":
            values = {"cpu_usage": 0.51, "memory_usage": 0.73}
            return lambda: FakeMonitorWorker(kwargs["system"], al_name, values[al_name])
        raise AssertionError(f"Unexpected algorithm request: {algorithm}")

    monkeypatch.setattr(monitor_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))

    scheduler_server = scheduler_server_module.SchedulerServer()
    scheduler_client = TestClient(scheduler_server.app)

    def dispatch(url, method=None, no_decode=False, binary=True, **kwargs):
        parsed = urlparse(url)
        response = scheduler_client.request(method or "GET", parsed.path or "/", **kwargs)
        assert response.status_code == 200, response.text
        if no_decode:
            return response
        return response.json() if binary else response.content.decode("utf-8")

    monkeypatch.setattr(monitor_module, "http_request", dispatch)

    try:
        monitor = monitor_module.Monitor()
        monitor.monitor_resource()
        monitor.send_resource_state_to_scheduler()
        assert scheduler_server.scheduler.resource_table == {
            "edge-node": {"cpu_usage": 0.51, "memory_usage": 0.73}
        }
    finally:
        scheduler_client.close()
