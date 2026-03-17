import importlib
import gzip
import json

import pytest
from fastapi.testclient import TestClient

from core.lib.common import Queue
from core.lib.content import Task
from core.lib.estimation import TimeEstimator


def build_task(flow_index="face-detection", execute_device="edge-node", file_path="payload.bin"):
    dag = Task.extract_dag_from_dag_deployment(
        {
            "face-detection": {
                "service": {"service_name": "face-detection", "execute_device": execute_device},
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
        flow_index=flow_index,
        metadata={"buffer_size": 1},
        raw_metadata={"buffer_size": 1},
        file_path=file_path,
    )


class FakeScheduler:
    def __init__(self):
        self.schedule_calls = []
        self.resource_table = {}
        self.resource_locks = {}
        self.scenario_tasks = []

    def register_schedule_table(self, source_id):
        return None

    def get_schedule_plan(self, info):
        self.schedule_calls.append(info)
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
        return 0.0123

    def update_scheduler_scenario(self, task):
        self.scenario_tasks.append(task)

    def register_resource_table(self, device):
        self.resource_table.setdefault(device, {})

    def update_scheduler_resource(self, info):
        self.resource_table[info["device"]] = info["resource"]

    def get_scheduler_resource(self):
        return self.resource_table

    async def get_resource_lock(self, info):
        self.resource_locks.setdefault(info["resource"], info["device"])
        return self.resource_locks[info["resource"]]

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
        return 321.0


@pytest.mark.integration
def test_scheduler_server_covers_schedule_resource_and_deployment_contracts(monkeypatch):
    scheduler_server_module = importlib.import_module("core.scheduler.scheduler_server")
    monkeypatch.setattr(scheduler_server_module, "Scheduler", FakeScheduler)
    monkeypatch.setattr(
        scheduler_server_module.KubeConfig,
        "get_service_nodes_dict",
        staticmethod(lambda: {"face-detection": ["edge-node"]}),
    )

    server = scheduler_server_module.SchedulerServer()

    with TestClient(server.app) as client:
        payload = {
            "source_id": 7,
            "meta_data": {"buffer_size": 2},
            "source_device": "edge-node",
            "all_edge_devices": ["edge-node"],
            "dag": {
                "face-detection": {
                    "service": {"service_name": "face-detection", "execute_device": "edge-node"},
                    "next_nodes": [],
                }
            },
        }
        schedule_response = client.request(
            "GET",
            "/schedule",
            data={"data": json.dumps(payload)},
        )
        assert schedule_response.status_code == 200
        assert schedule_response.json()["plan"]["buffer_size"] == 2
        assert schedule_response.json()["deployment"] == {"face-detection": ["edge-node"]}

        assert client.get("/overhead").json() == 0.0123

        resource_payload = {"device": "edge-node", "resource": {"cpu_usage": 0.42}}
        post_resource = client.post("/resource", data={"data": json.dumps(resource_payload)})
        assert post_resource.status_code == 200
        assert client.get("/resource").json() == {"edge-node": {"cpu_usage": 0.42}}

        lock_response = client.request(
            "GET",
            "/resource_lock",
            data={"data": json.dumps({"resource": "camera-0", "device": "edge-node"})},
        )
        assert lock_response.status_code == 200
        assert lock_response.json() == {"holder": "edge-node"}

        select_response = client.request(
            "GET",
            "/source_nodes_selection",
            data={"data": json.dumps([{"source": {"id": 1}, "node_set": ["edge-node"], "dag": {}}])},
        )
        assert select_response.status_code == 200
        assert select_response.json() == {"plan": {"1": "edge-node"}}

        initial_response = client.request(
            "GET",
            "/initial_deployment",
            data={"data": json.dumps([{"source": {"id": 1}, "node_set": ["edge-node"], "dag": {}}])},
        )
        assert initial_response.status_code == 200
        assert initial_response.json() == {"plan": {"edge-node": ["face-detection"]}}

        redeployment_response = client.request(
            "GET",
            "/redeployment",
            data={"data": json.dumps([{"source": {"id": 1}, "node_set": ["edge-node"], "dag": {}}])},
        )
        assert redeployment_response.status_code == 200
        assert redeployment_response.json() == {"plan": {"edge-node": ["face-detection"]}}


@pytest.mark.integration
def test_processor_server_exposes_queue_processing_and_return_contract(mounted_runtime, monkeypatch, tmp_path):
    processor_server_module = importlib.import_module("core.processor.processor_server")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(processor_server_module.ProcessorServer, "loop_process", lambda self: None)
    monkeypatch.setattr(
        processor_server_module.NodeInfo,
        "get_local_device",
        staticmethod(lambda: "edge-node"),
    )
    monkeypatch.setattr(
        processor_server_module.NodeInfo,
        "hostname2ip",
        staticmethod(lambda hostname: hostname),
    )
    monkeypatch.setattr(
        processor_server_module.PortInfo,
        "get_component_port",
        staticmethod(lambda component: 9002),
    )

    fake_queue = Queue()

    def fake_get_algorithm(algorithm, al_name=None, **kwargs):
        if algorithm == "PROCESSOR":
            return FakeProcessor()
        if algorithm == "PRO_QUEUE":
            return fake_queue
        raise AssertionError(f"Unexpected algorithm request: {algorithm}")

    monkeypatch.setattr(processor_server_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    monkeypatch.setenv("GUNICORN_PORT", "9004")

    server = processor_server_module.ProcessorServer()
    task = build_task(file_path="processor-input.bin")

    with TestClient(server.app) as client:
        local_response = client.post("/predict_local", data={"data": task.serialize()})
        assert local_response.status_code == 200
        assert fake_queue.size() == 1

        queued_task = fake_queue.get()
        processed_task = server.process_task_service(queued_task)
        assert processed_task.get_current_content() == {"service": "face-detection", "detections": 1}
        assert processed_task.get_scenario_data("face-detection") == {"obj_num": 1}

        with open("processor-input.bin", "wb") as fh:
            fh.write(b"payload")
        with open("processor-input.bin", "rb") as fh:
            return_response = client.post(
                "/predict_and_return",
                data={"data": task.serialize()},
                files={"file": ("processor-input.bin", fh, "application/octet-stream")},
            )
        assert return_response.status_code == 200
        returned_task = Task.deserialize(return_response.json())
        assert returned_task.get_current_content() == {"service": "face-detection", "detections": 1}
        assert client.get("/queue_length").json() == 0
        assert client.get("/model_flops").json() == 321.0


@pytest.mark.integration
def test_distributor_server_persists_records_and_queries_incrementally(monkeypatch, tmp_path):
    distributor_server_module = importlib.import_module("core.distributor.distributor_server")
    distributor_module = importlib.import_module("core.distributor.distributor")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        distributor_module.NodeInfo,
        "get_cloud_node",
        staticmethod(lambda: "cloud-node"),
    )
    monkeypatch.setattr(
        distributor_module.NodeInfo,
        "hostname2ip",
        staticmethod(lambda hostname: hostname),
    )
    monkeypatch.setattr(
        distributor_module.PortInfo,
        "get_component_port",
        staticmethod(lambda component: 9001),
    )

    scheduler_calls = []
    monkeypatch.setattr(
        distributor_module,
        "http_request",
        lambda url, method=None, **kwargs: scheduler_calls.append((url, method, kwargs)) or None,
    )

    server = distributor_server_module.DistributorServer()
    task = build_task(flow_index="_end", file_path="distributor-output.bin")
    TimeEstimator.record_dag_ts(task, is_end=False, sub_tag="transmit")

    with TestClient(server.app) as client:
        with open("distributor-output.bin", "wb") as fh:
            fh.write(b"payload")
        with open("distributor-output.bin", "rb") as fh:
            distribute_response = client.post(
                "/distribute",
                data={"data": task.serialize()},
                files={"file": ("distributor-output.bin", fh, "application/octet-stream")},
            )
        assert distribute_response.status_code == 200
        assert scheduler_calls, "Distributor should forward scenario updates to scheduler"

        result_response = client.request("GET", "/result", json={"time_ticket": 0, "size": 10})
        assert result_response.status_code == 200
        payload = result_response.json()
        assert payload["size"] == 1
        restored_task = Task.deserialize(payload["result"][0])
        assert restored_task.get_task_id() == 0

        all_response = client.get("/all_result")
        assert all_response.status_code == 200
        assert all_response.json()["size"] == 1

        export_response = client.get("/export_result_log")
        assert export_response.status_code == 200
        exported_tasks = json.loads(gzip.decompress(export_response.content).decode("utf-8"))
        assert len(exported_tasks) == 1
        assert exported_tasks[0]["task_id"] == 0
        assert client.get("/all_result").json()["size"] == 1
        assert client.get("/is_database_empty").json() is False
