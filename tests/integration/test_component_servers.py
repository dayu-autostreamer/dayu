import importlib
import gzip
import json

import pytest
from fastapi.testclient import TestClient

from core.lib.common import Queue, FileOps
from core.lib.content import Task
from core.lib.estimation import TimeEstimator
from core.lib.runtime import RuntimeContext
from core.scheduler.runtime_directory import RuntimeDirectoryConflict, RuntimeDirectoryError


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
        return True

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
        return {"face-detection": ["edge-node"]}

    def get_redeployment_plan(self, source_id, data):
        return {"face-detection": ["edge-node"]}

    def should_generate(self, source_id, data):
        return {"generate": data.get("allow", True), "reason": "fake_scheduler"}

    @staticmethod
    def runtime_service_nodes():
        return {"face-detection": ["edge-node"]}

    @staticmethod
    def runtime_directory_revision():
        return 3

    @staticmethod
    def clear_runtime_directory(install_id):
        if not install_id:
            raise RuntimeDirectoryError("runtime directory clear requires install_id")
        if install_id != "install-integration":
            raise RuntimeDirectoryConflict(
                "runtime directory install_id does not match clear request"
            )
        return {
            "cleared": True,
            "install_id": install_id,
            "previous_revision": 3,
        }

    @staticmethod
    def compact_runtime_routes(plan, source_device=""):
        common = {
            "target_node": "edge-node",
            "deployment_revision": 3,
            "install_id": "install-integration",
        }
        return [
            {
                **common,
                "component": "controller",
                "runtime_id": "controller-edge-node-r3",
                "fqdn": "controller-edge-node-r3.dayu.svc.cluster.local",
                "port": 9002,
                "runtime_service_uid": "controller-runtime-uid",
                "service_uid": "controller-service-uid",
                "endpoint_pod_uid": "controller-pod-uid",
            },
            {
                **common,
                "component": "processor",
                "logical_service": "face-detection",
                "runtime_id": "processor-face-detection-edge-node-r3",
                "fqdn": "processor-face-detection-edge-node-r3.dayu.svc.cluster.local",
                "port": 9004,
                "runtime_service_uid": "processor-runtime-uid",
                "service_uid": "processor-service-uid",
                "endpoint_pod_uid": "processor-pod-uid",
            },
        ]


def test_scheduler_deployment_plan_merge_has_one_service_to_nodes_contract():
    scheduler_server_module = importlib.import_module("core.scheduler.scheduler_server")
    plan = {}

    scheduler_server_module.SchedulerServer._merge_deployment_plan(
        plan, {"detector": ["edge-b", "edge-a"]}, allowed_services={"detector"},
    )
    scheduler_server_module.SchedulerServer._merge_deployment_plan(
        plan, {"detector": ["edge-a"], "tracker": ["edge-b"]},
        allowed_services={"detector", "tracker"},
    )

    assert plan == {
        "detector": ["edge-a", "edge-b"],
        "tracker": ["edge-b"],
    }
    with pytest.raises(RuntimeDirectoryError, match="JSON node list"):
        scheduler_server_module.SchedulerServer._merge_deployment_plan(
            {}, {"detector": "edge-a"}, allowed_services={"detector"},
        )
    with pytest.raises(RuntimeDirectoryError, match="outside the current DAG"):
        scheduler_server_module.SchedulerServer._merge_deployment_plan(
            {}, {"detector": ["edge-a"], "unknown": ["edge-a"]},
            allowed_services={"detector"},
        )
    with pytest.raises(RuntimeDirectoryError, match="omitted current DAG services"):
        scheduler_server_module.SchedulerServer._merge_deployment_plan(
            {}, {"detector": ["edge-a"]},
            allowed_services={"detector", "tracker"},
        )


class FakeProcessor:
    def __call__(self, task):
        task.set_current_content(
            {
                "service": task.get_flow_index(),
                "outputs": {
                    "bbox": [
                        {
                            "frame_index": 0,
                            "items": [{"bbox": [0, 0, 1, 1], "score": 1.0, "label": "object", "object_id": 1}],
                        }
                    ]
                },
                "profile": {
                    "frame_count": 1,
                },
            }
        )
        task.add_scenario({"obj_num": 1})
        return task

    @property
    def flops(self):
        return 321.0


@pytest.mark.integration
def test_scheduler_server_covers_schedule_resource_and_deployment_contracts(monkeypatch):
    scheduler_server_module = importlib.import_module("core.scheduler.scheduler_server")
    monkeypatch.setattr(scheduler_server_module, "Scheduler", FakeScheduler)

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
        assert schedule_response.json()["runtime_directory_revision"] == 3
        assert {route["component"] for route in schedule_response.json()["runtime_routes"]} == {
            "controller", "processor",
        }

        assert client.get("/overhead").json() == 0.0123

        admission_response = client.request(
            "GET",
            "/generation_admission",
            data={"data": json.dumps({"source_id": 7, "allow": False})},
        )
        assert admission_response.status_code == 200
        assert admission_response.json() == {"generate": False, "reason": "fake_scheduler"}

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

        deployment_source = [{
            "source": {"id": 1},
            "node_set": ["edge-node"],
            "dag": {"face-detection": {}},
        }]
        initial_response = client.request(
            "GET",
            "/initial_deployment",
            data={"data": json.dumps(deployment_source)},
        )
        assert initial_response.status_code == 200
        assert initial_response.json() == {"plan": {"face-detection": ["edge-node"]}}

        redeployment_response = client.request(
            "GET",
            "/redeployment",
            data={"data": json.dumps(deployment_source)},
        )
        assert redeployment_response.status_code == 200
        assert redeployment_response.json() == {"plan": {"face-detection": ["edge-node"]}}

        clear_response = client.request(
            "DELETE",
            "/runtime-directory",
            data={"data": json.dumps({"install_id": "install-integration"})},
        )
        assert clear_response.status_code == 200
        assert clear_response.json() == {
            "cleared": True,
            "install_id": "install-integration",
            "previous_revision": 3,
        }

        missing_identity = client.request(
            "DELETE",
            "/runtime-directory",
            data={"data": json.dumps({})},
        )
        assert missing_identity.status_code == 422

        wrong_identity = client.request(
            "DELETE",
            "/runtime-directory",
            data={"data": json.dumps({"install_id": "another-install"})},
        )
        assert wrong_identity.status_code == 409


@pytest.mark.integration
def test_processor_server_exposes_queue_processing_and_return_contract(mounted_runtime, monkeypatch, tmp_path):
    processor_server_module = importlib.import_module("core.processor.processor_server")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(processor_server_module.ProcessorServer, "loop_process", lambda self: None)
    runtime_context = RuntimeContext({"local_node": "edge-node"})
    monkeypatch.setattr(
        processor_server_module.RuntimeContext,
        "get_default",
        staticmethod(lambda: runtime_context),
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
    renewed = []
    class Lease:
        @staticmethod
        def keepalive(task):
            class Guard:
                def __enter__(self):
                    renewed.append(task.get_root_uuid())

                def __exit__(self, *_args):
                    return False

            return Guard()

    server.runtime_lease_client = Lease()
    task = build_task(file_path="processor-input.bin")

    with TestClient(server.app) as client:
        local_response = client.post("/predict_local", data={"data": task.serialize()})
        assert local_response.status_code == 200
        assert fake_queue.size() == 1

        queued_task = fake_queue.get()
        processed_task = server.process_task_service(queued_task)
        assert processed_task.get_current_content() == {
            "service": "face-detection",
            "outputs": {
                "bbox": [
                    {
                        "frame_index": 0,
                        "items": [{"bbox": [0, 0, 1, 1], "score": 1.0, "label": "object", "object_id": 1}],
                    }
                ]
            },
            "profile": {
                "frame_count": 1,
            },
        }
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
        assert returned_task.get_current_content() == {
            "service": "face-detection",
            "outputs": {
                "bbox": [
                    {
                        "frame_index": 0,
                        "items": [{"bbox": [0, 0, 1, 1], "score": 1.0, "label": "object", "object_id": 1}],
                    }
                ]
            },
            "profile": {
                "frame_count": 1,
            },
        }
        assert client.get("/queue_length").json() == 0
        assert client.get("/model_flops").json() == 321.0
        assert len(renewed) == 2


@pytest.mark.integration
def test_distributor_server_persists_records_and_queries_incrementally(monkeypatch, tmp_path):
    distributor_server_module = importlib.import_module("core.distributor.distributor_server")
    distributor_module = importlib.import_module("core.distributor.distributor")
    monkeypatch.chdir(tmp_path)
    runtime_context = RuntimeContext({
        "cloud_node": "cloud-node",
        "endpoints": {
            "scheduler": {
                "component": "scheduler",
                "target_node": "cloud-node",
                "fqdn": "scheduler-cloud.dayu.svc.cluster.local",
                "port": 9001,
            }
        },
    })
    monkeypatch.setattr(
        distributor_module.RuntimeContext,
        "get_default",
        staticmethod(lambda: runtime_context),
    )

    scheduler_calls = []
    def fake_scheduler_request(url, method=None, **kwargs):
        scheduler_calls.append((url, method, kwargs))
        if url.endswith("/scenario"):
            return {"accepted": True}
        if url.endswith("/runtime-directory/task-leases"):
            payload = json.loads(kwargs["data"]["data"])
            return {
                "revision": payload["revision"],
                "root_uuid": payload["root_uuid"],
                "released": True,
            }
        return None

    monkeypatch.setattr(distributor_module, "http_request", fake_scheduler_request)

    server = distributor_server_module.DistributorServer()
    task = build_task(flow_index="_end", file_path="distributor-output.bin")
    task.set_runtime_directory_revision(1)
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
        assert any(url.endswith("/runtime-directory/task-leases") for url, _, _ in scheduler_calls)

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


@pytest.mark.integration
def test_controller_server_accepts_health_submit_and_return_contracts(mounted_runtime, monkeypatch, tmp_path):
    controller_server_module = importlib.import_module("core.controller.controller_server")
    monkeypatch.chdir(tmp_path)

    submitted_tasks = []
    returned_tasks = []
    transmit_records = []
    execute_records = []

    class FakeController:
        def check_processor_health(self, request=None):
            return True

        @staticmethod
        def record_transmit_ts(task, is_end=False):
            transmit_records.append((task.get_task_id(), is_end, task.get_file_path()))

        @staticmethod
        def record_execute_ts(task, is_end=False):
            execute_records.append((task.get_task_id(), is_end))

        def submit_task(self, task):
            submitted_tasks.append(task)

        def process_return(self, task):
            returned_tasks.append(task)

    monkeypatch.setattr(controller_server_module, "Controller", FakeController)

    server = controller_server_module.ControllerServer()
    task = build_task(file_path="controller-input.bin")

    with TestClient(server.app) as client:
        health_response = client.post("/check")
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "ok"}

        submit_response = client.post(
            "/submit_task",
            data={"data": task.serialize()},
            files={"file": ("controller-input.bin", b"controller-payload", "application/octet-stream")},
        )
        assert submit_response.status_code == 200
        assert submitted_tasks and submitted_tasks[0].get_task_id() == task.get_task_id()
        assert transmit_records == [(task.get_task_id(), True, "controller-input.bin")]

        temp_file_path = FileOps.get_task_file_in_temp(task)
        with open(temp_file_path, "rb") as fh:
            assert fh.read() == b"controller-payload"

        return_response = client.post("/process_return_task", data={"data": task.serialize()})
        assert return_response.status_code == 200
        assert returned_tasks and returned_tasks[0].get_task_id() == task.get_task_id()
        assert execute_records == [(task.get_task_id(), True)]
