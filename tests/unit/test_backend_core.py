import copy
import importlib

import pytest


def make_valid_dag():
    return {
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
    }


@pytest.fixture
def backend_core_instance(mounted_runtime, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "check_pod_name",
        staticmethod(lambda *args, **kwargs: False),
    )
    return backend_core_module.BackendCore()


@pytest.mark.unit
def test_check_dag_validates_service_input_output_contracts(backend_core_instance):
    valid_state, valid_msg = backend_core_instance.check_dag(make_valid_dag())
    assert valid_state is True
    assert valid_msg == "DAG validation passed"

    invalid_dag = {
        "_start": ["gender-classification"],
        "gender-classification": {
            "id": "gender-classification",
            "prev": [],
            "succ": ["face-detection"],
        },
        "face-detection": {
            "id": "face-detection",
            "prev": ["gender-classification"],
            "succ": [],
        },
    }

    invalid_state, invalid_msg = backend_core_instance.check_dag(invalid_dag)
    assert invalid_state is False
    assert "Node connection mismatch" in invalid_msg


@pytest.mark.unit
def test_extract_service_from_source_deployment_merges_edge_nodes(backend_core_instance):
    source_deploy = [
        {
            "source": {"id": 0, "name": "camera-0"},
            "node_set": ["edgex1"],
            "dag": make_valid_dag(),
        },
        {
            "source": {"id": 1, "name": "camera-1"},
            "node_set": ["edgex2"],
            "dag": make_valid_dag(),
        },
    ]

    service_dict = backend_core_instance.extract_service_from_source_deployment(source_deploy)

    assert set(service_dict.keys()) == {"face-detection", "gender-classification"}
    assert set(service_dict["face-detection"]["node"]) == {"edgex1", "edgex2"}
    assert set(service_dict["gender-classification"]["node"]) == {"edgex1", "edgex2"}
    assert "_start" not in source_deploy[0]["dag"]
    assert source_deploy[0]["dag"]["face-detection"]["id"] == "face-detection"
    assert source_deploy[0]["dag"]["gender-classification"]["prev"] == ["face-detection"]


@pytest.mark.unit
def test_has_significant_changes_ignores_non_deployment_fields():
    backend_core_module = importlib.import_module("backend_core")

    old_doc = {
        "apiVersion": "sedna.io/v1alpha1",
        "kind": "JointMultiEdgeService",
        "metadata": {"name": "processor-face-detection-edgex1"},
        "spec": {
            "edgeWorker": [
                {
                    "logLevel": {"level": "INFO"},
                    "file": {"paths": ["/data/dayu-files/processor/face-detection/"]},
                    "template": {
                        "spec": {
                            "nodeName": "edgex1",
                            "dnsPolicy": "ClusterFirstWithHostNet",
                            "serviceAccountName": "worker-admin",
                            "containers": [
                                {
                                    "image": "repo:5000/dayuhub/face-detection:v1.3",
                                    "ports": [{"containerPort": 9000}],
                                    "env": [{"name": "PROCESSOR_NAME", "value": "detector"}],
                                }
                            ],
                        }
                    },
                }
            ]
        },
    }
    new_doc = copy.deepcopy(old_doc)
    worker = new_doc["spec"]["edgeWorker"][0]
    worker["logLevel"]["level"] = "DEBUG"
    worker["file"]["paths"] = ["/tmp/other-path"]
    worker["template"]["spec"]["containers"][0]["env"] = [{"name": "PROCESSOR_NAME", "value": "detector-v2"}]

    assert backend_core_module.BackendCore.has_significant_changes(old_doc, new_doc) is False

    new_doc["spec"]["edgeWorker"][0]["template"]["spec"]["containers"][0]["image"] = (
        "repo:5000/dayuhub/face-detection:v1.4"
    )
    assert backend_core_module.BackendCore.has_significant_changes(old_doc, new_doc) is True
