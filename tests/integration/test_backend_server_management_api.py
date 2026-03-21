import asyncio
import copy
import gzip
import importlib
import json

import pytest
from fastapi.testclient import TestClient


def make_dag():
    return {
        "_start": ["face-detection"],
        "face-detection": {"id": "face-detection", "prev": [], "succ": []},
    }


class FakeStreamResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.headers = {"content-length": str(len(payload))}
        self.closed = False

    def iter_content(self, chunk_size=8192):
        for idx in range(0, len(self._payload), chunk_size):
            yield self._payload[idx: idx + chunk_size]

    def close(self):
        self.closed = True
        return None


class FakeBackendCoreManagement:
    def __init__(self):
        self.namespace = "dayu-test"
        self.schedulers = [{"id": "fixed", "name": "Fixed Policy"}]
        self.services = [
            {
                "id": "face-detection",
                "name": "face detection",
                "description": "face detection",
                "input": "frame",
                "output": "bbox",
            }
        ]
        self.dags = [{"dag_id": 1, "dag_name": "face-pipeline", "dag": make_dag()}]
        self.source_configs = [
            {
                "source_label": "source-config-0",
                "source_name": "demo",
                "source_type": "video",
                "source_mode": "http_video",
                "source_list": [
                    {"id": 0, "name": "camera-0", "url": "http://camera-0/live", "metadata": {"fps": 25}},
                    {"id": 1, "name": "camera-1", "url": "http://camera-1/live", "metadata": {"fps": 30}},
                ],
            }
        ]
        self.source_open = False
        self.source_label = ""
        self.inner_datasource = True
        self.task_results = {}
        self.customized_source_result_visualization_configs = {}
        self.resource_url = None
        self.install_state = False
        self.install_result = (True, "ok")
        self.install_exception = None
        self.uninstall_result = (True, "ok")
        self.run_get_result_called = False
        self.clear_yaml_docs_calls = 0
        self.applied_templates = []
        self.datasource_config_to_return = {
            "source_name": "uploaded",
            "source_type": "video",
            "source_mode": "http_video",
            "source_list": [{"name": "camera-upload", "url": "http://uploaded/live", "metadata": {"fps": 20}}],
        }
        self.visualization_config_to_return = [
            {
                "name": "Latency",
                "type": "curve",
                "variables": ["delay"],
                "size": 1,
                "hook_name": "delay",
            }
        ]
        self.export_stream = FakeStreamResponse(gzip.compress(json.dumps([{"task_id": 1}]).encode("utf-8")))

    def parse_base_info(self):
        return None

    def check_pods_running_state(self):
        return True

    def check_dag(self, dag):
        return True, "ok"

    def find_scheduler_policy_by_id(self, policy_id):
        return next((policy for policy in self.schedulers if policy["id"] == policy_id), None)

    def find_dag_by_id(self, dag_id):
        for dag in self.dags:
            if dag["dag_id"] == dag_id:
                return dag["dag"]
        return None

    def find_datasource_configuration_by_label(self, label):
        return next((config for config in self.source_configs if config["source_label"] == label), None)

    def check_node_exist(self, node):
        return node in {"edgex1", "edgex2"}

    def parse_and_apply_templates(self, policy, source_deploy):
        if self.install_exception:
            raise self.install_exception
        self.applied_templates.append({"policy": policy, "source_deploy": copy.deepcopy(source_deploy)})
        return self.install_result

    def parse_and_delete_templates(self):
        return self.uninstall_result

    def clear_yaml_docs(self):
        self.clear_yaml_docs_calls += 1

    def check_install_state(self):
        return self.install_state

    def get_source_ids(self):
        config = self.find_datasource_configuration_by_label(self.source_label)
        return [] if not config else [source["id"] for source in config["source_list"]]

    def run_get_result(self):
        self.run_get_result_called = True

    def fetch_visualization_data(self, source_id):
        return [{"source_id": source_id, "value": f"result-{source_id}"}]

    def get_system_parameters(self):
        return {"namespace": self.namespace}

    def get_result_visualization_config(self, source_id):
        return [{"id": 0, "source_id": source_id}]

    def get_system_visualization_config(self):
        return [{"name": "CPU Usage"}]

    def check_datasource_config(self, config_path):
        return copy.deepcopy(self.datasource_config_to_return)

    def fill_datasource_config(self, config):
        config = copy.deepcopy(config)
        config["source_label"] = "source-config-1"
        config["source_list"][0]["id"] = 0
        return config

    def check_visualization_config(self, config_path):
        return copy.deepcopy(self.visualization_config_to_return)

    def get_edge_nodes(self):
        return [{"name": "edgex1"}, {"name": "edgex2"}]

    def get_resource_url(self):
        self.resource_url = "http://scheduler/resource"

    def get_log_file_name(self):
        return None

    def open_result_log_export_stream(self):
        return self.export_stream


@pytest.fixture
def management_backend(monkeypatch, tmp_path):
    backend_server_module = importlib.import_module("backend_server")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(backend_server_module, "BackendCore", FakeBackendCoreManagement)
    monkeypatch.setattr(backend_server_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        backend_server_module.KubeHelper,
        "get_service_info",
        staticmethod(lambda service_name, namespace: [{"hostname": "edge-a", "age": "1m"}]),
    )
    monkeypatch.setattr(
        backend_server_module,
        "http_request",
        lambda address, method=None, **kwargs: {"edge-a": {"available_bandwidth": 12.34}}
        if address == "http://scheduler/resource"
        else None,
    )

    backend = backend_server_module.BackendServer()
    with TestClient(backend.app) as client:
        started_targets = []
        original_thread = backend_server_module.threading.Thread

        class DummyThread:
            def __init__(self, target):
                self.target = target

            def start(self):
                started_targets.append(self.target)

        def thread_factory(*args, target=None, **kwargs):
            if target == backend.server.run_get_result:
                return DummyThread(target)
            return original_thread(*args, target=target, **kwargs)

        monkeypatch.setattr(backend_server_module.threading, "Thread", thread_factory)
        yield backend_server_module, backend, client, started_targets


@pytest.mark.integration
def test_backend_server_covers_install_and_datasource_management_flows(management_backend):
    _, backend, client, _ = management_backend

    assert client.get("/datasource").json()[0]["source_label"] == "source-config-0"

    upload_response = client.post(
        "/datasource",
        files={"file": ("datasource.yaml", b"source_name: uploaded\n", "application/x-yaml")},
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["state"] == "success"
    assert backend.server.source_configs[-1]["source_label"] == "source-config-1"

    backend.server.datasource_config_to_return = None
    failed_upload = client.post(
        "/datasource",
        files={"file": ("broken.yaml", b"broken: true\n", "application/x-yaml")},
    )
    assert failed_upload.json()["state"] == "fail"

    delete_result = asyncio.run(backend.delete_datasource_info(json.dumps({"source_label": "source-config-1"}).encode()))
    assert delete_result["state"] == "success"
    missing_delete_result = asyncio.run(
        backend.delete_datasource_info(json.dumps({"source_label": "missing"}).encode())
    )
    assert missing_delete_result["state"] == "fail"

    invalid_policy = asyncio.run(
        backend.install_service(
            json.dumps(
                {
                    "source_config_label": "source-config-0",
                    "policy_id": "missing",
                    "source": [{"id": 0, "dag_selected": 1, "node_selected": ["edgex1"]}],
                }
            ).encode()
        )
    )
    assert invalid_policy["state"] == "fail"

    invalid_node = asyncio.run(
        backend.install_service(
            json.dumps(
                {
                    "source_config_label": "source-config-0",
                    "policy_id": "fixed",
                    "source": [
                        {"id": 0, "dag_selected": 1, "node_selected": ["missing-node"]},
                        {"id": 1, "dag_selected": 1, "node_selected": ["edgex1"]},
                    ],
                }
            ).encode()
        )
    )
    assert invalid_node["state"] == "fail"

    install_result = asyncio.run(
        backend.install_service(
            json.dumps(
                {
                    "source_config_label": "source-config-0",
                    "policy_id": "fixed",
                    "source": [
                        {"id": 0, "dag_selected": 1, "node_selected": ["edgex1"]},
                        {"id": 1, "dag_selected": 1, "node_selected": ["edgex2"]},
                    ],
                }
            ).encode()
        )
    )
    assert install_result["state"] == "success"
    assert len(backend.server.applied_templates) == 1
    assert backend.server.applied_templates[0]["source_deploy"][0]["source"]["source_mode"] == "http_video"

    backend.server.install_exception = RuntimeError("boom")
    failed_install = asyncio.run(
        backend.install_service(
            json.dumps(
                {
                    "source_config_label": "source-config-0",
                    "policy_id": "fixed",
                    "source": [
                        {"id": 0, "dag_selected": 1, "node_selected": ["edgex1"]},
                        {"id": 1, "dag_selected": 1, "node_selected": ["edgex2"]},
                    ],
                }
            ).encode()
        )
    )
    assert failed_install["state"] == "fail"

    uninstall_result = asyncio.run(backend.uninstall_service())
    assert uninstall_result["state"] == "success"
    assert backend.server.clear_yaml_docs_calls == 1

    backend.server.uninstall_result = (False, "still running")
    failed_uninstall = asyncio.run(backend.uninstall_service())
    assert failed_uninstall["state"] == "fail"


@pytest.mark.integration
def test_backend_server_covers_query_state_visualization_and_service_info(management_backend):
    _, backend, client, started_targets = management_backend

    missing_query = asyncio.run(backend.submit_query(json.dumps({"source_label": "missing"}).encode()))
    assert missing_query["state"] == "fail"

    open_query = asyncio.run(backend.submit_query(json.dumps({"source_label": "source-config-0"}).encode()))
    assert open_query["state"] == "success"
    assert backend.server.source_open is True
    assert sorted(backend.server.task_results.keys()) == [0, 1]
    assert len(started_targets) == 1

    assert client.get("/query_state").json() == {"state": "open", "source_label": "source-config-0"}
    assert client.get("/source_list").json() == [
        {"id": 0, "label": "camera-0"},
        {"id": 1, "label": "camera-1"},
    ]
    assert client.get("/task_result").json() == {
        "0": [{"source_id": 0, "value": "result-0"}],
        "1": [{"source_id": 1, "value": "result-1"}],
    }
    datasource_state = client.get("/datasource_state").json()
    assert datasource_state["state"] == "open"
    assert datasource_state["source_mode"] == "http_video"

    upload_viz = client.post(
        "/result_visualization_config/3",
        files={"file": ("visualization.yaml", b"- name: Latency\n", "application/x-yaml")},
    )
    assert upload_viz.json()["state"] == "success"
    assert backend.server.customized_source_result_visualization_configs[3][0]["name"] == "Latency"

    backend.server.visualization_config_to_return = None
    failed_upload_viz = client.post(
        "/result_visualization_config/3",
        files={"file": ("broken.yaml", b"invalid\n", "application/x-yaml")},
    )
    assert failed_upload_viz.json()["state"] == "fail"

    service_info = asyncio.run(backend.get_service_info("face-detection"))
    assert service_info == [{"hostname": "edge-a", "age": "1m", "bandwidth": "12.34Mbps"}]
    assert asyncio.run(backend.get_service_info("null")) == []

    stop_result = asyncio.run(backend.stop_query())
    assert stop_result["state"] == "success"
    assert client.get("/query_state").json() == {"state": "close", "source_label": ""}
    assert client.get("/task_result").json() == {}

    backend.server.source_open = True
    backend.server.source_label = "missing"
    assert client.get("/datasource_state").json() == {"state": "close"}

    reset_response = client.post("/reset_datasource")
    assert reset_response.status_code == 200
    assert backend.server.source_open is False


@pytest.mark.integration
def test_backend_server_covers_delete_dag_and_install_state_routes(management_backend):
    _, backend, client, _ = management_backend

    delete_result = asyncio.run(backend.delete_dag_workflow(json.dumps({"dag_id": 1}).encode()))
    assert delete_result["state"] == "success"
    missing_delete = asyncio.run(backend.delete_dag_workflow(json.dumps({"dag_id": 1}).encode()))
    assert missing_delete["state"] == "fail"

    backend.server.install_state = True
    assert client.get("/install_state").json() == {"state": "install"}
    backend.server.install_state = False
    assert client.get("/install_state").json() == {"state": "uninstall"}


@pytest.mark.integration
def test_backend_server_covers_disabled_query_state_and_service_info_fallbacks(management_backend, monkeypatch):
    backend_server_module, backend, client, _ = management_backend

    backend.server.inner_datasource = False
    assert client.get("/query_state").json() == {"state": "disabled", "source_label": ""}

    backend.server.resource_url = None
    monkeypatch.setattr(backend.server, "get_resource_url", lambda: None)
    assert asyncio.run(backend.get_service_info("face-detection")) == []

    monkeypatch.setattr(
        backend_server_module.KubeHelper,
        "get_service_info",
        staticmethod(lambda service_name, namespace: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    assert asyncio.run(backend.get_service_info("face-detection")) == []

    monkeypatch.setattr(backend.server, "open_result_log_export_stream", lambda: None)
    unavailable_log = client.get("/download_log")
    assert unavailable_log.status_code == 503
    assert unavailable_log.json() == {"detail": "Result log export is temporarily unavailable"}
