import asyncio
import copy
import gzip
import importlib
import json
import threading
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from runtime_model import RuntimeCleanupResource, RuntimeUninstallProgress

INSTALL_ID = "11111111-1111-4111-8111-111111111111"


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
                "input": ["frame"],
                "output": ["bbox"],
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
        self._query_generation = 0
        self.customized_source_result_visualization_configs = {}
        self.resource_url = None
        self.scheduler_resource = {
            "cloud-a": {"available_bandwidth": -1},
            "edge-a": {"available_bandwidth": -1},
            "edge-probe": {"available_bandwidth": 12.34},
        }
        self.runtime_metrics_snapshot = {
            "processor-face-detection-edge-a-r1": {
                "uid": "processor-pod-uid",
                "logical_service": "face-detection",
                "node": "edge-a",
                "node_info": {"address": "10.0.0.8"},
                "usage": {"processor": {"cpu": "25m", "memory": "64Mi"}},
                "resource_usage": {
                    "cpu": {
                        "status": "available",
                        "usage_millicores": 25.0,
                        "reference_millicores": 4000.0,
                        "utilization_percent": 0.625,
                        "basis": "node_allocatable",
                    },
                    "memory": {
                        "status": "available",
                        "usage_bytes": 64 * 1024 ** 2,
                        "reference_bytes": 8 * 1024 ** 3,
                        "utilization_percent": 0.78125,
                        "basis": "node_allocatable",
                    },
                },
                "created_at": "2026-07-12T00:00:00Z",
            }
        }
        self.resource_sampled_at = 1.0
        self.resource_stale = False
        self.runtime_metrics_sampled_at = 1.0
        self.install_state = False
        self._pending_install_id = ""
        self._current_install_id = INSTALL_ID
        self.runtime_orchestrator = type("RuntimeView", (), {
            "current_session": lambda _runtime: (
                SimpleNamespace(
                    install_id=self._current_install_id,
                    phase="active",
                    operation_id="operation-test",
                    updated_at="2026-07-16T00:00:00Z",
                    active_directory_revision=1,
                    active=(),
                    pending=(),
                    retirement=None,
                    cleanup=(),
                    last_error="",
                ) if self.install_state else None
            ),
        })()
        self.install_result = (True, "ok")
        self.install_exception = None
        self.uninstall_result = (True, "ok")
        self.uninstall_expected_ids = []
        self.run_get_result_called = False
        self.close_query_calls = 0
        self.query_lock = threading.Lock()
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

    @staticmethod
    def service_io_labels(service, field):
        service_id = service.get("id") or service.get("service") or "<unknown>"
        value = service.get(field)
        if not isinstance(value, list):
            return None, f"Service '{service_id}' field '{field}' must be a list of type labels"
        if any(not isinstance(item, str) or not item for item in value):
            return None, f"Service '{service_id}' field '{field}' must contain non-empty string labels"
        return value, None

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

    def parse_and_apply_templates(self, policy, source_deploy, source_label="", install_id=""):
        if self.install_exception:
            raise self.install_exception
        self.applied_templates.append({
            "policy": policy,
            "source_deploy": copy.deepcopy(source_deploy),
            "source_label": source_label,
            "install_id": install_id,
        })
        if self.install_result[0]:
            self._current_install_id = install_id
            self.install_state = True
        return self.install_result

    def parse_and_delete_templates(self, expected_install_id=""):
        self.uninstall_expected_ids.append(expected_install_id)
        if expected_install_id:
            try:
                if str(uuid.UUID(expected_install_id)) != expected_install_id:
                    raise ValueError
            except (ValueError, TypeError, AttributeError):
                return False, "install_id must be a canonical UUID"
        self.close_query()
        return self.uninstall_result

    def management_lifecycle_snapshot(self):
        session = self.runtime_orchestrator.current_session()
        pending = (
            {
                "install_id": self._pending_install_id,
                "phase": "preparing-install",
                "operation_id": "pending-operation",
            }
            if self._pending_install_id else None
        )
        return session, pending, bool(session and session.phase == "active"), ""

    def get_source_ids(self):
        config = self.find_datasource_configuration_by_label(self.source_label)
        return [] if not config else [source["id"] for source in config["source_list"]]

    def run_get_result(self):
        self.run_get_result_called = True

    def open_query(self, source_label):
        if not self.find_datasource_configuration_by_label(source_label):
            return False, "Datasource configuration not exists"
        if self.source_open:
            if self.source_label == source_label:
                return True, "Datasource is already open"
            return False, "Another datasource is already open, please close it first"
        self.source_open = True
        self.source_label = source_label
        self._query_generation += 1
        self.task_results = {source_id: object() for source_id in self.get_source_ids()}
        self.run_get_result()
        return True, "Datasource open successfully"

    def close_query(self):
        self.close_query_calls += 1
        self._query_generation += 1
        self.source_open = False
        self.source_label = ""
        self.task_results.clear()
        self.customized_source_result_visualization_configs.clear()
        return True, "Datasource close successfully"

    def query_snapshot(self, include_queues=False):
        with self.query_lock:
            return {
                "open": self.source_open,
                "source_label": self.source_label,
                "generation": self._query_generation,
                "queues": dict(self.task_results) if include_queues else None,
            }

    def is_query_generation_active(self, generation):
        with self.query_lock:
            return self.source_open and generation == self._query_generation

    def fetch_visualization_data(self, source_id, task_queue=None):
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
        config["source_label"] = f"source-config-{len(self.source_configs)}"
        config["source_list"][0]["id"] = 0
        return config

    def check_visualization_config(self, config_path):
        return copy.deepcopy(self.visualization_config_to_return)

    def get_edge_nodes(self):
        return [{"name": "edgex1"}, {"name": "edgex2"}]

    def get_runtime_telemetry(self, logical_service=""):
        metrics = copy.deepcopy(self.runtime_metrics_snapshot)
        if logical_service:
            metrics = {
                name: metric for name, metric in metrics.items()
                if metric.get("logical_service") == logical_service
            }
        return {
            "install_id": "install-test" if self.install_state else None,
            "directory_revision": 1 if self.install_state else None,
            "resource": copy.deepcopy(self.scheduler_resource),
            "scheduling_overhead": None,
            "runtime_metrics": metrics,
            "scheduler_sampled_at": None,
            "resource_sampled_at": self.resource_sampled_at,
            "resource_stale": self.resource_stale,
            "runtime_metrics_sampled_at": self.runtime_metrics_sampled_at,
        }

    def get_log_file_name(self):
        return None

    def open_result_log_export_stream(self):
        return self.export_stream


@pytest.fixture
def management_backend(monkeypatch, tmp_path):
    backend_server_module = importlib.import_module("backend_server")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(backend_server_module, "BackendCore", FakeBackendCoreManagement)
    backend = backend_server_module.BackendServer()
    with TestClient(backend.app) as client:
        yield backend_server_module, backend, client, []


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

    multi_upload_response = client.post(
        "/datasource",
        files=[
            ("files", ("datasource-a.yaml", b"source_name: uploaded-a\n", "application/x-yaml")),
            ("files", ("datasource-b.yaml", b"source_name: uploaded-b\n", "application/x-yaml")),
        ],
    )
    assert multi_upload_response.status_code == 200
    assert multi_upload_response.json()["state"] == "success"
    assert len(multi_upload_response.json()["results"]) == 2
    assert backend.server.source_configs[-1]["source_label"] == "source-config-3"

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
                    "install_id": INSTALL_ID,
                    "source_config_label": "source-config-0",
                    "policy_id": "missing",
                    "source": [{"id": 0, "dag_selected": 1, "node_selected": ["edgex1"]}],
                }
            ).encode()
        )
    )
    assert invalid_policy["state"] == "fail"

    install_result = client.post(
        "/install",
        json={
            "install_id": INSTALL_ID,
            "source_config_label": "source-config-0",
            "policy_id": "fixed",
            "source": [
                {"id": 0, "dag_selected": 1, "node_selected": ["edgex1"]},
                {"id": 1, "dag_selected": 1, "node_selected": ["edgex2"]},
            ],
        },
    ).json()
    assert install_result == {
        "state": "success",
        "msg": "Install services successfully",
    }
    installed_state = client.get("/install_state").json()
    assert installed_state["install_id"] == INSTALL_ID
    assert installed_state["ready"] is True
    assert len(backend.server.applied_templates) == 1
    assert backend.server.applied_templates[0]["source_deploy"][0]["source"]["source_mode"] == "http_video"
    assert backend.server.applied_templates[0]["source_label"] == "source-config-0"

    backend.server.install_exception = RuntimeError("boom")
    failed_install = asyncio.run(
        backend.install_service(
            json.dumps(
                {
                    "install_id": INSTALL_ID,
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

    backend.server.uninstall_result = (True, "Uninstall services started")
    uninstall_result = client.post(
        "/stop_service",
        json={"install_id": INSTALL_ID},
    ).json()
    assert uninstall_result == {
        "state": "success",
        "msg": "Uninstall services started",
    }
    assert backend.server.uninstall_expected_ids[-1] == INSTALL_ID
    assert backend.server.close_query_calls == 1

    backend.server.uninstall_result = (False, "still running")
    failed_uninstall = asyncio.run(backend.uninstall_service(None))
    assert failed_uninstall["state"] == "fail"
    assert backend.server.close_query_calls == 2


@pytest.mark.integration
def test_backend_server_covers_query_state_visualization_and_service_info(management_backend):
    _, backend, client, _ = management_backend
    backend.server.install_state = True

    missing_query = asyncio.run(backend.submit_query(json.dumps({"source_label": "missing"}).encode()))
    assert missing_query["state"] == "fail"

    open_query = asyncio.run(backend.submit_query(json.dumps({"source_label": "source-config-0"}).encode()))
    assert open_query["state"] == "success"
    assert backend.server.source_open is True
    assert sorted(backend.server.task_results.keys()) == [0, 1]
    assert backend.server.run_get_result_called is True

    duplicate_query = asyncio.run(backend.submit_query(json.dumps({"source_label": "source-config-0"}).encode()))
    assert duplicate_query["state"] == "success"
    assert backend.server.run_get_result_called is True

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
    assert service_info == [{
        "ip": "10.0.0.8",
        "hostname": "edge-a",
        "cpu": {
            "status": "available",
            "usage_millicores": 25.0,
            "reference_millicores": 4000.0,
            "utilization_percent": 0.625,
            "basis": "node_allocatable",
        },
        "memory": {
            "status": "available",
            "usage_bytes": 64 * 1024 ** 2,
            "reference_bytes": 8 * 1024 ** 3,
            "utilization_percent": 0.78125,
            "basis": "node_allocatable",
        },
        "bandwidth": {
            "status": "available",
            "mbps": 12.34,
            "probe_node": "edge-probe",
        },
        "age": "2026-07-12T00:00:00Z",
    }]
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
    assert client.get("/install_state").json() == {
        "state": "install",
        "phase": "active",
        "ready": True,
        "install_id": INSTALL_ID,
        "install_pending": False,
        "operation_id": "operation-test",
        "updated_at": "2026-07-16T00:00:00Z",
        "active_directory_revision": 1,
        "active_runtime_count": 0,
        "pending_runtime_count": 0,
        "cleanup_runtime_count": 0,
        "cleanup": None,
        "retirement_revision": 0,
        "retirement_deadline": None,
        "last_error": "",
    }

    backend.server._pending_install_id = INSTALL_ID
    assert client.get("/install_state").json()["ready"] is False
    backend.server._pending_install_id = ""
    backend.server.install_state = False
    assert client.get("/install_state").json() == {
        "state": "uninstall",
        "phase": "uninstalled",
        "ready": False,
        "install_id": "",
        "install_pending": False,
        "operation_id": "",
        "updated_at": "",
        "active_directory_revision": 0,
        "active_runtime_count": 0,
        "pending_runtime_count": 0,
        "cleanup_runtime_count": 0,
        "cleanup": None,
        "retirement_revision": 0,
        "retirement_deadline": None,
        "last_error": "",
    }

    backend.server._pending_install_id = INSTALL_ID
    pending_state = client.get("/install_state").json()
    assert pending_state["install_pending"] is True
    assert pending_state["install_id"] == INSTALL_ID
    assert pending_state["phase"] == "preparing-install"


@pytest.mark.integration
def test_install_state_projects_admission_and_local_projection_failures(
        management_backend,
):
    _, backend, _, _ = management_backend
    backend.server.install_state = True
    session = backend.server.runtime_orchestrator.current_session()

    preparing_stop = backend._install_state_response(
        session,
        {
            "kind": "stop",
            "install_id": INSTALL_ID,
            "phase": "preparing-uninstall",
            "operation_id": "stop-operation",
        },
        local_ready=False,
    )
    assert preparing_stop["state"] == "install"
    assert preparing_stop["phase"] == "preparing-uninstall"
    assert preparing_stop["operation_id"] == "stop-operation"
    assert preparing_stop["install_pending"] is False
    assert preparing_stop["ready"] is False

    targetless_stop = backend._install_state_response(
        None,
        {
            "kind": "stop",
            "install_id": "",
            "phase": "preparing-uninstall",
            "operation_id": "global-stop-operation",
        },
    )
    assert targetless_stop["state"] == "uninstall"
    assert targetless_stop["phase"] == "preparing-uninstall"
    assert targetless_stop["install_id"] == ""
    assert targetless_stop["operation_id"] == "global-stop-operation"
    assert targetless_stop["install_pending"] is False

    cancelling_install = backend._install_state_response(
        session,
        {
            "kind": "install",
            "install_id": INSTALL_ID,
            "phase": "cancelling-install",
            "operation_id": "cancel-operation",
        },
        local_ready=False,
    )
    assert cancelling_install["phase"] == "cancelling-install"
    assert cancelling_install["operation_id"] == "cancel-operation"
    assert cancelling_install["install_pending"] is True

    projection_failure = backend._install_state_response(
        session,
        local_ready=False,
        local_error="local runtime activation failed: bind failed",
    )
    assert projection_failure["state"] == "install"
    assert projection_failure["phase"] == "failed"
    assert projection_failure["ready"] is False
    assert projection_failure["last_error"].endswith("bind failed")


@pytest.mark.integration
def test_install_state_reports_delayed_cleanup_without_releasing_session_ownership(
        management_backend,
):
    _, backend, _, _ = management_backend
    backend.server.install_state = True
    session = copy.copy(backend.server.runtime_orchestrator.current_session())
    session.phase = "finalizing-uninstall"
    session.uninstall = RuntimeUninstallProgress(
        started_at="2020-01-01T00:00:00+00:00",
        last_progress_at="2020-01-01T00:01:00+00:00",
        deletion_submitted=True,
        remaining=(RuntimeCleanupResource(
            kind="Pod",
            name="processor-pod",
            uid="pod-uid",
            node="edge-a",
            deletion_timestamp="2020-01-01T00:00:30+00:00",
            finalizers=("example.io/cleanup",),
        ),),
    )

    state = backend._install_state_response(session)

    assert state["state"] == "install"
    assert state["phase"] == "finalizing-uninstall"
    assert state["ready"] is False
    assert state["install_id"] == INSTALL_ID
    assert state["cleanup"]["status"] == "delayed"
    assert state["cleanup"]["remaining_by_kind"] == {"Pod": 1}
    assert state["cleanup"]["affected_nodes"] == ["edge-a"]
    assert state["cleanup"]["blocking_objects"][0]["finalizers"] == [
        "example.io/cleanup",
    ]
    assert state["cleanup"]["truncated_count"] == 0


@pytest.mark.integration
def test_install_state_bounds_cleanup_details_without_losing_aggregate_count(
        management_backend,
):
    _, backend, _, _ = management_backend
    backend.server.install_state = True
    session = copy.copy(backend.server.runtime_orchestrator.current_session())
    session.phase = "finalizing-uninstall"
    session.uninstall = RuntimeUninstallProgress(
        started_at="2026-07-16T00:00:00+00:00",
        last_progress_at="2026-07-16T00:01:00+00:00",
        deletion_submitted=True,
        remaining=tuple(
            RuntimeCleanupResource(
                kind="Pod",
                name=f"processor-{index}",
                uid=f"pod-uid-{index}",
            )
            for index in range(30)
        ),
    )

    cleanup = backend._install_state_response(session)["cleanup"]

    assert cleanup["remaining_count"] == 30
    assert cleanup["remaining_by_kind"] == {"Pod": 30}
    assert len(cleanup["blocking_objects"]) == 25
    assert cleanup["truncated_count"] == 5


@pytest.mark.integration
def test_install_and_uninstall_reject_malformed_json_objects(management_backend):
    _, _, client, _ = management_backend

    install = client.post("/install", json=[])
    uninstall = client.post("/stop_service", json=[])

    assert install.json() == {
        "state": "fail",
        "msg": "Install services failed: invalid request body",
    }
    assert uninstall.json() == {
        "state": "fail",
        "msg": "Uninstall services failed: invalid request body",
    }

    invalid_identity = client.post(
        "/stop_service",
        json={"install_id": "not-an-install-id"},
    )
    assert invalid_identity.json() == {
        "state": "fail",
        "msg": "Uninstall services failed: install_id must be a canonical UUID",
    }


@pytest.mark.integration
def test_implicit_query_failure_is_a_warning_not_runtime_rollback(
        management_backend, monkeypatch,
):
    _, backend, client, _ = management_backend
    backend.server.inner_datasource = False
    monkeypatch.setattr(
        backend.server,
        "open_query",
        lambda _source_label: (False, "datasource temporarily unavailable"),
    )

    response = client.post(
        "/install",
        json={
            "install_id": INSTALL_ID,
            "source_config_label": "source-config-0",
            "policy_id": "fixed",
            "source": [
                {"dag_selected": 1, "node_selected": ["edgex1"]},
                {"dag_selected": 1, "node_selected": ["edgex2"]},
            ],
        },
    ).json()

    assert response["state"] == "success"
    assert "warning" in response
    assert "datasource temporarily unavailable" in response["warning"]
    assert "install_state" not in response
    assert backend.server.install_state is True


@pytest.mark.integration
def test_long_uninstall_does_not_block_install_state_event_loop(management_backend, monkeypatch):
    _, backend, _, _ = management_backend
    started = threading.Event()
    release = threading.Event()

    def slow_uninstall(expected_install_id=""):
        started.set()
        assert release.wait(2)
        return True, "ok"

    monkeypatch.setattr(backend.server, "parse_and_delete_templates", slow_uninstall)

    async def exercise():
        uninstall = asyncio.create_task(backend.uninstall_service(None))
        assert await asyncio.to_thread(started.wait, 1)
        state = await asyncio.wait_for(backend.get_install_state(), timeout=0.2)
        release.set()
        result = await uninstall
        return state, result

    state, result = asyncio.run(exercise())

    assert state["state"] == "uninstall"
    assert result["state"] == "success"


@pytest.mark.integration
def test_session_snapshot_reads_are_offloaded_from_async_management_handlers(
        management_backend, monkeypatch,
):
    _, backend, _, _ = management_backend
    main_thread = threading.get_ident()
    caller_threads = []
    export_threads = []
    session = SimpleNamespace(
        install_id=INSTALL_ID,
        phase="active",
        operation_id="operation-test",
        updated_at="2026-07-16T00:00:00Z",
        active_directory_revision=1,
        active=(),
        pending=(),
        retirement=None,
        cleanup=(),
        last_error="",
    )

    def current_session():
        caller_threads.append(threading.get_ident())
        return session

    monkeypatch.setattr(
        backend.server.runtime_orchestrator,
        "current_session",
        current_session,
    )
    monkeypatch.setattr(
        backend.server,
        "open_result_log_export_stream",
        lambda: (
            export_threads.append(threading.get_ident()),
            FakeStreamResponse(gzip.compress(b"[]")),
        )[1],
    )

    asyncio.run(backend.get_install_state())
    asyncio.run(backend.download_log())
    asyncio.run(backend.get_service_info("face-detection"))
    asyncio.run(backend.get_task_result())

    # service_info is now a pure telemetry-cache read and no longer consults
    # lifecycle state or Kubernetes through RuntimeOrchestrator.
    assert len(caller_threads) == 2
    assert all(thread_id != main_thread for thread_id in caller_threads)
    assert len(export_threads) == 1
    assert export_threads[0] != main_thread


@pytest.mark.integration
def test_backend_server_covers_disabled_query_state_and_service_info_fallbacks(management_backend, monkeypatch):
    _, backend, client, _ = management_backend
    backend.server.install_state = True

    backend.server.inner_datasource = False
    assert client.get("/query_state").json() == {"state": "disabled", "source_label": ""}

    backend.server.scheduler_resource = None
    without_scheduler_resource = asyncio.run(backend.get_service_info("face-detection"))
    assert without_scheduler_resource[0]["hostname"] == "edge-a"
    assert without_scheduler_resource[0]["bandwidth"] == {
        "status": "unavailable",
        "mbps": None,
        "probe_node": "",
    }

    monkeypatch.setattr(
        backend.server,
        "get_runtime_telemetry",
        lambda logical_service="": (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert asyncio.run(backend.get_service_info("face-detection")) == []

    monkeypatch.setattr(backend.server, "open_result_log_export_stream", lambda: None)
    unavailable_log = client.get("/download_log")
    assert unavailable_log.status_code == 503
    assert unavailable_log.json() == {"detail": "Result log export is temporarily unavailable"}


@pytest.mark.integration
def test_service_info_projects_one_probe_to_every_service_row(management_backend):
    _, backend, _, _ = management_backend
    backend.server.install_state = True
    backend.server.runtime_metrics_snapshot["processor-face-detection-edge-b-r1"] = {
        **copy.deepcopy(next(iter(backend.server.runtime_metrics_snapshot.values()))),
        "uid": "processor-pod-b-uid",
        "node": "edge-b",
        "node_info": {"address": "10.0.0.9"},
    }

    service_info = asyncio.run(backend.get_service_info("face-detection"))

    assert [item["hostname"] for item in service_info] == ["edge-a", "edge-b"]
    expected = {
        "status": "available",
        "mbps": 12.34,
        "probe_node": "edge-probe",
    }
    assert [item["bandwidth"] for item in service_info] == [expected, expected]
    assert service_info[0]["bandwidth"] is not service_info[1]["bandwidth"]


@pytest.mark.integration
def test_service_info_exposes_collecting_and_stale_bandwidth_states(management_backend):
    _, backend, _, _ = management_backend
    backend.server.install_state = True

    backend.server.scheduler_resource = None
    backend.server.resource_sampled_at = None
    collecting = asyncio.run(backend.get_service_info("face-detection"))[0]
    assert collecting["bandwidth"]["status"] == "collecting"

    backend.server.scheduler_resource = {"edge-probe": {"available_bandwidth": 12.34}}
    backend.server.resource_sampled_at = 1.0
    backend.server.resource_stale = True
    stale = asyncio.run(backend.get_service_info("face-detection"))[0]
    assert stale["bandwidth"] == {
        "status": "stale",
        "mbps": 12.34,
        "probe_node": "edge-probe",
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    ("resource", "has_sample", "expected"),
    [
        (
            {
                "cloud": {"available_bandwidth": -1},
                "edge-a": {"available_bandwidth": 0},
                "edge-b": {"available_bandwidth": True},
                "edge-c": {"available_bandwidth": float("nan")},
                "edge-d": {"available_bandwidth": float("inf")},
            },
            True,
            {"status": "unavailable", "mbps": None, "probe_node": ""},
        ),
        (
            {
                "edge-b": {"available_bandwidth": 8.0},
                "edge-a": {"available_bandwidth": 7.0},
            },
            True,
            {"status": "ambiguous", "mbps": None, "probe_node": ""},
        ),
        (
            {},
            False,
            {"status": "collecting", "mbps": None, "probe_node": ""},
        ),
    ],
)
def test_shared_bandwidth_fails_closed_for_invalid_or_conflicting_probes(
        management_backend, resource, has_sample, expected,
):
    backend_server_module, _, _, _ = management_backend

    assert backend_server_module._shared_bandwidth(resource, has_sample) == expected


@pytest.mark.integration
def test_service_resource_detail_recomputes_percent_and_fails_closed(management_backend):
    backend_server_module, _, _, _ = management_backend

    assert backend_server_module._resource_detail(
        {
            "status": "available",
            "usage_millicores": 100.0,
            "reference_millicores": 1000.0,
            "utilization_percent": 99.0,
            "basis": "node_allocatable",
        },
        "cpu",
        has_sample=True,
    ) == {
        "status": "available",
        "usage_millicores": 100.0,
        "reference_millicores": 1000.0,
        "utilization_percent": 10.0,
        "basis": "node_allocatable",
    }
    assert backend_server_module._resource_detail(
        {
            "status": "stale",
            "usage_bytes": float("nan"),
            "reference_bytes": 0,
            "utilization_percent": float("inf"),
            "basis": "unknown",
        },
        "memory",
        has_sample=True,
    ) == {
        "status": "unavailable",
        "usage_bytes": None,
        "reference_bytes": None,
        "utilization_percent": None,
        "basis": "",
    }
    assert backend_server_module._resource_detail(None, "cpu", has_sample=False)["status"] == "collecting"
