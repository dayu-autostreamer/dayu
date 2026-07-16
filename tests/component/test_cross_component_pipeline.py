import copy
import importlib
import json
import threading
from pathlib import Path
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI, Form, Response
from fastapi.testclient import TestClient

from core.lib.common import Context, Queue, FileOps
from core.lib.content import Task
from core.lib.runtime import RuntimeContext


GeneratorBase = importlib.import_module("core.generator.generator").Generator


def build_pipeline_runtime_context():
    return RuntimeContext({
        "install_id": "install-component",
        "runtime_directory_revision": 1,
        "local_node": "edge-node",
        "cloud_node": "cloud-node",
        "nodes": {
            "edge-node": {"role": "edge", "address": "edge-node"},
            "cloud-node": {"role": "cloud", "address": "cloud-node"},
        },
        "endpoints": {
            "scheduler": {
                "component": "scheduler", "target_node": "cloud-node",
                "fqdn": "scheduler-cloud.dayu.svc.cluster.local", "port": 9001,
            },
            "distributor": {
                "component": "distributor", "target_node": "cloud-node",
                "fqdn": "distributor-cloud.dayu.svc.cluster.local", "port": 9003,
            },
            "redis": {
                "component": "redis", "target_node": "cloud-node",
                "fqdn": "redis-cloud.dayu.svc.cluster.local", "port": 6379,
            },
        },
    })


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
        self.leases = set()
        self.lease_operations = []

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
        return True

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
        return {"face-detection": ["edge-node"]}

    def get_redeployment_plan(self, source_id, data):
        return {"face-detection": ["edge-node"]}

    def should_generate(self, source_id, data):
        return {"generate": True, "reason": "fake_scheduler"}

    @staticmethod
    def runtime_service_nodes():
        return {"face-detection": ["edge-node"]}

    @staticmethod
    def runtime_directory_revision():
        return 1

    @staticmethod
    def compact_runtime_routes(plan, source_device=""):
        common = {
            "target_node": "edge-node",
            "deployment_revision": 1,
            "install_id": "install-component",
        }
        return [
            {
                **common,
                "component": "controller",
                "runtime_id": "controller-edge-node-r1",
                "fqdn": "controller-edge-node-r1.dayu.svc.cluster.local",
                "port": 9002,
                "runtime_service_uid": "controller-runtime-uid",
                "service_uid": "controller-service-uid",
                "endpoint_pod_uid": "controller-pod-uid",
            },
            {
                **common,
                "component": "processor",
                "logical_service": "face-detection",
                "runtime_id": "processor-face-detection-edge-node-r1",
                "fqdn": "processor-face-detection-edge-node-r1.dayu.svc.cluster.local",
                "port": 9004,
                "runtime_service_uid": "processor-runtime-uid",
                "service_uid": "processor-service-uid",
                "endpoint_pod_uid": "processor-pod-uid",
            },
        ]

    def schedule_runtime_state(self, plan, source_device=""):
        return {
            "revision": self.runtime_directory_revision(),
            "hash": "component-runtime-directory",
            "deployment": self.runtime_service_nodes(),
            "routes": self.compact_runtime_routes(plan, source_device),
        }

    def acquire_task_lease(self, revision, root_uuid, ttl_seconds=60.0):
        self.lease_operations.append(("acquire", int(revision), str(root_uuid)))
        self.leases.add((int(revision), str(root_uuid)))
        return {
            "revision": int(revision), "root_uuid": str(root_uuid), "expires_at": 9999999999.0,
            "valid_for_seconds": float(ttl_seconds),
        }

    def renew_task_lease(self, revision, root_uuid, ttl_seconds=60.0):
        self.lease_operations.append(("renew", int(revision), str(root_uuid)))
        assert (int(revision), str(root_uuid)) in self.leases
        return {
            "revision": int(revision), "root_uuid": str(root_uuid), "expires_at": 9999999999.0,
            "valid_for_seconds": float(ttl_seconds),
        }

    def release_task_lease(self, revision, root_uuid):
        self.lease_operations.append(("release", int(revision), str(root_uuid)))
        self.leases.discard((int(revision), str(root_uuid)))
        return {"revision": int(revision), "root_uuid": str(root_uuid), "released": True}

    def count_task_leases(self, revision):
        return sum(1 for item_revision, _ in self.leases if item_revision == int(revision))


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
        return 128.0


class StreamAwareProcessor:
    def __call__(self, task):
        payload = Path(FileOps.get_task_file_in_temp(task)).read_bytes().decode("utf-8")
        task.set_current_content(
            {
                "service": task.get_flow_index(),
                "outputs": {
                    "text": [
                        {
                            "frame_index": None,
                            "items": [{"text": payload, "frames": task.get_hash_data()}],
                        }
                    ]
                },
                "profile": {
                    "frame_count": len(task.get_hash_data()),
                },
            }
        )
        task.add_scenario({"obj_num": len(task.get_hash_data()), "payload": payload})
        return task

    @property
    def flops(self):
        return 256.0


class FakeMonitorWorker(threading.Thread):
    def __init__(self, system, metric_name, value):
        super().__init__(daemon=True)
        self.system = system
        self.metric_name = metric_name
        self.value = value

    def run(self):
        self.system.resource_info[self.metric_name] = self.value


class FakeStreamDataSource:
    def __init__(self):
        self.source_requests = []
        self.pending_payload = b""
        self.batch_id = 0
        self.app = FastAPI()

        @self.app.get("/stream-0/source")
        async def get_source_data(data: str = Form(...)):
            request_payload = json.loads(data)
            self.source_requests.append(copy.deepcopy(request_payload))

            buffer_size = request_payload["meta_data"]["buffer_size"]
            start_frame = self.batch_id * buffer_size
            frames_index = list(range(start_frame, start_frame + buffer_size))

            self.pending_payload = f"stream-batch-{self.batch_id}".encode("utf-8")
            self.batch_id += 1
            return frames_index

        @self.app.get("/stream-0/file")
        async def get_source_file():
            return Response(content=self.pending_payload, media_type="application/octet-stream")


class ComponentRouter:
    def __init__(self, scheduler_server, controller_server, processor_server, distributor_server, source_app=None):
        self.scheduler_server = scheduler_server
        self.controller_server = controller_server
        self.processor_server = processor_server
        self.distributor_server = distributor_server
        self.source_app = source_app

        self.scheduler_client = TestClient(scheduler_server.app)
        self.controller_client = TestClient(controller_server.app)
        self.processor_client = TestClient(processor_server.app)
        self.distributor_client = TestClient(distributor_server.app)
        self.source_client = TestClient(source_app) if source_app else None

        self.client_by_port = {
            "9001": self.scheduler_client,
            "9002": self.controller_client,
            "9003": self.distributor_client,
            "9004": self.processor_client,
        }
        if self.source_client:
            self.client_by_port["9010"] = self.source_client

    def close(self):
        self.scheduler_client.close()
        self.controller_client.close()
        self.processor_client.close()
        self.distributor_client.close()
        if self.source_client:
            self.source_client.close()

    def request(self, url, method=None, no_decode=False, binary=True, **kwargs):
        method = method or "GET"
        # RuntimeLeaseClient exposes transport retry semantics that are not a
        # TestClient request argument; the in-process router is deterministic.
        kwargs.pop("retry", None)
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
def test_parallel_branches_keep_one_artifact_and_retry_ready_join(mounted_runtime):
    controller_module = importlib.import_module("core.controller.controller")
    processor_server_module = importlib.import_module("core.processor.processor_server")
    dag = Task.extract_dag_from_dag_deployment({
        "fast": {
            "service": {"service_name": "fast", "execute_device": "edge-node"},
            "next_nodes": ["join"],
        },
        "slow": {
            "service": {"service_name": "slow", "execute_device": "edge-node"},
            "next_nodes": ["join"],
        },
        "join": {
            "service": {"service_name": "join", "execute_device": "edge-node"},
            "next_nodes": [],
        },
    })
    root = Task(
        source_id=0,
        task_id=9,
        source_device="edge-node",
        all_edge_devices=["edge-node"],
        dag=dag,
        metadata={"buffer_size": 1},
        raw_metadata={"buffer_size": 1},
        file_path="shared.mp4",
        runtime_directory_revision=1,
    )
    FileOps.save_task_file_in_temp(root, b"immutable-video")
    fast, slow = root.step_to_next_stage()
    assert FileOps.get_task_file_in_temp(fast) == FileOps.get_task_file_in_temp(slow)

    class ReadingProcessor:
        def __call__(self, task):
            payload = Path(FileOps.get_task_file_in_temp(task)).read_bytes()
            task.set_current_content({
                "service": task.get_flow_index(),
                "outputs": {"text": [{"frame_index": None, "items": [{"text": payload.decode()}]}]},
                "profile": {"frame_count": 1},
            })
            return task

    processor = object.__new__(processor_server_module.ProcessorServer)
    processor.processor = ReadingProcessor()
    fast = processor.process_task_service(fast)
    assert Path(FileOps.get_task_file_in_temp(root)).read_bytes() == b"immutable-video"

    class Barrier:
        def __init__(self):
            self.tasks = {}
            self.completed = 0

        def arrive(self, task, joint_service_name, required_count):
            self.tasks[task.get_past_flow_index()] = Task.deserialize(task.serialize())
            return list(self.tasks.values()) if len(self.tasks) == required_count else None

        def complete(self, root_uuid, joint_service_name):
            self.completed += 1
            self.tasks.clear()

    controller = object.__new__(controller_module.Controller)
    controller.task_coordinator = Barrier()
    downstream_attempts = []
    downstream_results = []

    def submit_merged(task):
        downstream_attempts.append(task)
        if len(downstream_attempts) == 1:
            return False
        downstream_results.append(task)
        return True

    controller.submit_task = submit_merged
    assert controller.process_return(fast) is True

    slow = processor.process_task_service(slow)
    assert slow.get_current_content()["outputs"]["text"][0]["items"][0]["text"] == "immutable-video"
    assert controller.process_return(slow) is False
    assert len(controller.task_coordinator.tasks) == 2
    assert Path(FileOps.get_task_file_in_temp(root)).read_bytes() == b"immutable-video"

    assert controller.process_return(Task.deserialize(slow.serialize())) is True
    assert controller.task_coordinator.completed == 1
    assert len(downstream_results) == 1
    assert downstream_results[0].get_service("fast").get_content_data() is not None
    assert downstream_results[0].get_service("slow").get_content_data() is not None


@pytest.mark.component
def test_generator_controller_processor_distributor_scheduler_pipeline(mounted_runtime, monkeypatch, tmp_path):
    generator_module = importlib.import_module("core.generator.generator")
    controller_module = importlib.import_module("core.controller.controller")
    controller_server_module = importlib.import_module("core.controller.controller_server")
    processor_server_module = importlib.import_module("core.processor.processor_server")
    distributor_module = importlib.import_module("core.distributor.distributor")
    distributor_server_module = importlib.import_module("core.distributor.distributor_server")
    scheduler_server_module = importlib.import_module("core.scheduler.scheduler_server")
    delivery_module = importlib.import_module("core.lib.network.delivery")

    monkeypatch.chdir(tmp_path)
    Path("payload.bin").write_bytes(b"frame-bytes")

    monkeypatch.setenv("NODE_NAME", "edge-node")
    monkeypatch.setenv("ALL_EDGE_DEVICES", "['edge-node']")
    monkeypatch.setenv("REQUEST_SCHEDULING_INTERVAL", "1")
    monkeypatch.setenv("GUNICORN_PORT", "9004")
    monkeypatch.setenv("DISPLAY", "True")
    monkeypatch.setenv("MONITORS", "[]")

    runtime_context = build_pipeline_runtime_context()
    monkeypatch.setattr(
        RuntimeContext, "get_default", staticmethod(lambda: runtime_context)
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
    for module in (generator_module, controller_module, distributor_module):
        monkeypatch.setattr(module, "http_request", router.request)
    monkeypatch.setattr(delivery_module, "http_request", router.request)
    distributor_server.distributor.runtime_lease_client.requester = router.request

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
        assert stored_task.get_last_content() == {
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

        assert scheduler_server.scheduler.schedule_calls, "Generator should request a schedule plan"
        assert len(scheduler_server.scheduler.scenario_tasks) == 1
        scenario_task = scheduler_server.scheduler.scenario_tasks[0]
        assert scenario_task.get_scenario_data("face-detection") == {"obj_num": 1}
        assert scheduler_server.scheduler.lease_operations == [
            ("acquire", 1, stored_task.get_root_uuid()),
            ("renew", 1, stored_task.get_root_uuid()),
            ("release", 1, stored_task.get_root_uuid()),
        ]
    finally:
        router.close()


@pytest.mark.component
def test_stream_data_flows_from_datasource_to_processing_and_storage(mounted_runtime, monkeypatch, tmp_path):
    generator_module = importlib.import_module("core.generator.generator")
    video_generator_module = importlib.import_module("core.generator.video_generator")
    controller_module = importlib.import_module("core.controller.controller")
    controller_server_module = importlib.import_module("core.controller.controller_server")
    http_video_getter_module = importlib.import_module("core.lib.algorithms.data_getter.http_video_getter")
    processor_server_module = importlib.import_module("core.processor.processor_server")
    distributor_module = importlib.import_module("core.distributor.distributor")
    distributor_server_module = importlib.import_module("core.distributor.distributor_server")
    scheduler_server_module = importlib.import_module("core.scheduler.scheduler_server")
    delivery_module = importlib.import_module("core.lib.network.delivery")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NODE_NAME", "edge-node")
    monkeypatch.setenv("ALL_EDGE_DEVICES", "['edge-node']")
    monkeypatch.setenv("REQUEST_SCHEDULING_INTERVAL", "1")
    monkeypatch.setenv("GUNICORN_PORT", "9004")
    monkeypatch.setenv("DISPLAY", "True")
    monkeypatch.setenv("GEN_FILTER_NAME", "simple")
    monkeypatch.setenv("GEN_PROCESS_NAME", "simple")
    monkeypatch.setenv("GEN_COMPRESS_NAME", "simple")

    runtime_context = build_pipeline_runtime_context()
    monkeypatch.setattr(
        RuntimeContext, "get_default", staticmethod(lambda: runtime_context)
    )

    monkeypatch.setattr(processor_server_module.ProcessorServer, "loop_process", lambda self: None)
    monkeypatch.setattr(scheduler_server_module, "Scheduler", FakeScheduler)
    monkeypatch.setattr(http_video_getter_module.time, "sleep", lambda _: None)

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
                if scheduler_response is None:
                    system.service_deployment = {"edge-node": ["face-detection"]}
                    return
                dag = Task.extract_dag_from_dag_deployment(scheduler_response["plan"]["dag"])
                dag.get_start_node().service.set_execute_device(system.local_device)
                dag.get_end_node().service.set_execute_device("cloud-node")
                system.task_dag = dag
                system.service_deployment = {"edge-node": ["face-detection"]}
                system.meta_data.update({"buffer_size": scheduler_response["plan"]["buffer_size"]})
            return after_schedule
        if algorithm == "GEN_BSTO":
            return lambda system, task: None
        if algorithm == "GEN_GETTER":
            return http_video_getter_module.HttpVideoGetter()
        if algorithm == "GEN_GETTER_FILTER":
            return lambda system: True
        if algorithm in {"GEN_FILTER", "GEN_PROCESS", "GEN_COMPRESS"}:
            return object()
        if algorithm == "PROCESSOR":
            return StreamAwareProcessor()
        if algorithm == "PRO_QUEUE":
            return fake_queue
        raise AssertionError(f"Unexpected algorithm request: {algorithm}")

    monkeypatch.setattr(generator_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    monkeypatch.setattr(video_generator_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    monkeypatch.setattr(processor_server_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))

    scheduler_server = scheduler_server_module.SchedulerServer()
    controller_server = controller_server_module.ControllerServer()
    processor_server = processor_server_module.ProcessorServer()
    distributor_server = distributor_server_module.DistributorServer()
    source_server = FakeStreamDataSource()

    router = ComponentRouter(
        scheduler_server,
        controller_server,
        processor_server,
        distributor_server,
        source_app=source_server.app,
    )
    for module in (
        generator_module,
        http_video_getter_module,
        controller_module,
        distributor_module,
    ):
        monkeypatch.setattr(module, "http_request", router.request)
    monkeypatch.setattr(delivery_module, "http_request", router.request)
    distributor_server.distributor.runtime_lease_client.requester = router.request

    class BoundedVideoGenerator(video_generator_module.VideoGenerator):
        def run_stream(self, rounds):
            self.after_schedule_operation(self, None)
            self.request_schedule_policy()

            for _ in range(rounds):
                assert self.getter_filter(self)
                self.data_getter(self)

                if self.cumulative_scheduling_frame_count > \
                        self.request_scheduling_interval * self.raw_meta_data.get("fps", 0):
                    self.request_schedule_policy()
                    self.cumulative_scheduling_frame_count = 0

    try:
        generator = BoundedVideoGenerator(
            source_id=0,
            source_url="http://stream-host:9010/stream-0",
            source_metadata={
                "buffer_size": 1,
                "fps": 2,
                "resolution": "720p",
                "encoding": "mp4v",
            },
            dag=Task.extract_dict_from_dag(build_single_service_task().get_dag()),
        )
        generator.run_stream(rounds=3)

        query_response = router.distributor_client.get("/all_result")
        assert query_response.status_code == 200
        assert query_response.json()["size"] == 3

        stored_tasks = [Task.deserialize(task_json) for task_json in query_response.json()["result"]]
        assert [task.get_task_id() for task in stored_tasks] == [0, 1, 2]
        assert [task.get_last_content()["outputs"]["text"][0]["items"][0]["text"] for task in stored_tasks] == [
            "stream-batch-0",
            "stream-batch-1",
            "stream-batch-2",
        ]
        assert [task.get_last_content()["outputs"]["text"][0]["items"][0]["frames"] for task in stored_tasks] == [
            [0],
            [1],
            [2],
        ]

        assert len(source_server.source_requests) == 3
        assert all(request["gen_filter_name"] == "simple" for request in source_server.source_requests)
        assert all(request["gen_process_name"] == "simple" for request in source_server.source_requests)
        assert all(request["gen_compress_name"] == "simple" for request in source_server.source_requests)

        assert len(scheduler_server.scheduler.schedule_calls) == 2
        assert len(scheduler_server.scheduler.scenario_tasks) == 3
        assert scheduler_server.scheduler.scenario_tasks[-1].get_scenario_data("face-detection") == {
            "obj_num": 1,
            "payload": "stream-batch-2",
        }
        assert scheduler_server.scheduler.lease_operations == [
            operation
            for task in stored_tasks
            for operation in (
                ("acquire", 1, task.get_root_uuid()),
                ("renew", 1, task.get_root_uuid()),
                ("release", 1, task.get_root_uuid()),
            )
        ]
    finally:
        router.close()


@pytest.mark.component
def test_monitor_reports_resource_state_to_scheduler(mounted_runtime, monkeypatch):
    monitor_module = importlib.import_module("core.monitor.monitor")
    scheduler_server_module = importlib.import_module("core.scheduler.scheduler_server")

    monkeypatch.setenv("NODE_NAME", "edge-node")
    monkeypatch.setenv("INTERVAL", "1")
    monkeypatch.setenv("MONITORS", "['cpu_usage', 'memory_usage']")

    runtime_context = RuntimeContext({
        "local_node": "edge-node",
        "cloud_node": "cloud-node",
        "nodes": {
            "edge-node": {"role": "edge"},
            "cloud-node": {"role": "cloud"},
        },
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
        monitor_module.RuntimeContext,
        "get_default",
        staticmethod(lambda: runtime_context),
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
