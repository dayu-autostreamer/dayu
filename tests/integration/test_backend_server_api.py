import importlib

import pytest
from fastapi.testclient import TestClient


class FakeBackendCore:
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
            },
            {
                "id": "gender-classification",
                "name": "gender classification",
                "description": "gender classification",
                "input": "bbox",
                "output": "text",
            },
        ]
        self.dags = []
        self.source_configs = [
            {
                "source_label": "source-config-0",
                "source_type": "video",
                "source_mode": "http_video",
                "source_list": [
                    {"id": 0, "name": "camera-0"},
                    {"id": 1, "name": "camera-1"},
                ],
            }
        ]
        self.source_open = False
        self.source_label = ""
        self.inner_datasource = True

    def parse_base_info(self):
        return None

    def check_pods_running_state(self):
        return True

    def check_dag(self, dag):
        return True, "ok"

    def find_datasource_configuration_by_label(self, label):
        for config in self.source_configs:
            if config["source_label"] == label:
                return config
        return None

    def get_edge_nodes(self):
        return [{"name": "edgex1"}]

    def get_system_parameters(self):
        return {"namespace": self.namespace}

    def get_result_visualization_config(self, source_id):
        return [{"id": 0, "source_id": source_id}]

    def get_system_visualization_config(self):
        return [{"name": "CPU Usage"}]

    def check_install_state(self):
        return False


@pytest.fixture
def backend_client(monkeypatch):
    backend_server_module = importlib.import_module("backend_server")
    monkeypatch.setattr(backend_server_module, "BackendCore", FakeBackendCore)
    monkeypatch.setattr(
        backend_server_module.KubeHelper,
        "check_pod_name",
        staticmethod(lambda name, namespace: name == "face-detection"),
    )

    backend = backend_server_module.BackendServer()
    with TestClient(backend.app) as client:
        yield backend, client


@pytest.mark.integration
def test_policy_and_installed_service_routes_are_exposed(backend_client):
    _, client = backend_client

    policy_response = client.get("/policy")
    assert policy_response.status_code == 200
    assert policy_response.json() == [{"policy_id": "fixed", "policy_name": "Fixed Policy"}]

    installed_response = client.get("/installed_service")
    assert installed_response.status_code == 200
    assert installed_response.json() == ["face-detection"]


@pytest.mark.integration
def test_get_all_services_is_idempotent(backend_client):
    _, client = backend_client

    first_response = client.get("/service")
    second_response = client.get("/service")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json() == second_response.json()
    service_map = {service["id"]: service for service in first_response.json()}
    assert service_map["face-detection"]["description"].endswith("(in:frame, out:bbox)")
    assert service_map["face-detection"]["description"].count("(in:") == 1


@pytest.mark.integration
def test_dag_and_query_related_routes_return_expected_payloads(backend_client):
    backend, client = backend_client

    dag_payload = {
        "dag_name": "face-pipeline",
        "dag": {
            "_start": ["face-detection"],
            "face-detection": {
                "id": "face-detection",
                "prev": [],
                "succ": ["gender-classification"],
            },
            "gender-classification": {
                "id": "gender-classification",
                "prev": ["face-detection"],
                "succ": [],
            },
        },
    }

    create_response = client.post("/dag_workflow", json=dag_payload)
    assert create_response.status_code == 200
    assert create_response.json()["state"] == "success"

    dag_response = client.get("/dag_workflow")
    assert dag_response.status_code == 200
    assert dag_response.json()[0]["dag_name"] == "face-pipeline"

    backend.server.source_open = True
    backend.server.source_label = "source-config-0"
    source_list_response = client.get("/source_list")
    assert source_list_response.status_code == 200
    assert source_list_response.json() == [
        {"id": 0, "label": "camera-0"},
        {"id": 1, "label": "camera-1"},
    ]

    assert client.get("/system_parameters").json() == {"namespace": "dayu-test"}
    assert client.get("/result_visualization_config/1").json() == [{"id": 0, "source_id": 1}]
    assert client.get("/system_visualization_config").json() == [{"name": "CPU Usage"}]
