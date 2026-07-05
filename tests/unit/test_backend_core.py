import ast
import copy
import gzip
import importlib
import json
from pathlib import Path

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

    fan_in_dag = {
        "_start": ["traffic-object-detection", "road-context-segmentation"],
        "traffic-object-detection": {
            "id": "traffic-object-detection",
            "prev": [],
            "succ": [
                "vehicle-reidentification-tracking",
                "vehicle-attribute-recognition",
            ],
        },
        "road-context-segmentation": {
            "id": "road-context-segmentation",
            "prev": [],
            "succ": ["vehicle-trajectory-prediction"],
        },
        "vehicle-reidentification-tracking": {
            "id": "vehicle-reidentification-tracking",
            "prev": ["traffic-object-detection"],
            "succ": ["vehicle-trajectory-prediction"],
        },
        "vehicle-attribute-recognition": {
            "id": "vehicle-attribute-recognition",
            "prev": ["traffic-object-detection"],
            "succ": ["vehicle-trajectory-prediction"],
        },
        "vehicle-trajectory-prediction": {
            "id": "vehicle-trajectory-prediction",
            "prev": [
                "road-context-segmentation",
                "vehicle-reidentification-tracking",
                "vehicle-attribute-recognition",
            ],
            "succ": [],
        },
    }

    fan_in_state, fan_in_msg = backend_core_instance.check_dag(fan_in_dag)
    assert fan_in_state is True
    assert fan_in_msg == "DAG validation passed"

    generic_shape_dag = {
        "_start": ["car-detection"],
        "car-detection": {
            "id": "car-detection",
            "prev": [],
            "succ": ["gender-classification"],
        },
        "gender-classification": {
            "id": "gender-classification",
            "prev": ["car-detection"],
            "succ": [],
        },
    }
    generic_state, generic_msg = backend_core_instance.check_dag(generic_shape_dag)
    assert generic_state is True
    assert generic_msg == "DAG validation passed"

    allowed_io_labels = {
        "frame",
        "bbox",
        "text",
        "segmentation",
        "track",
        "attribute",
        "trajectory",
        "pose",
        "graph",
    }
    for service in backend_core_instance.services:
        assert set(service["input"]).issubset(allowed_io_labels)
        assert set(service["output"]).issubset(allowed_io_labels)

    face_service = backend_core_instance.find_service_by_id("face-detection")
    original_input = face_service["input"]
    face_service["input"] = "frame"
    strict_state, strict_msg = backend_core_instance.check_dag(make_valid_dag())
    assert strict_state is False
    assert "must be a list" in strict_msg
    face_service["input"] = original_input


@pytest.mark.unit
def test_structured_traffic_example_dag_and_templates_remain_flexible(backend_core_instance, mounted_runtime):
    from core.lib.common import YamlOps

    repo_root = Path(__file__).resolve().parents[2]
    example = YamlOps.read_yaml(repo_root / "config" / "application_dags" / "traffic_risk_monitoring.dag")
    assert example["format"] == "dayu.application-dag"
    assert example["version"] == 1
    assert example["dag_name"] == "traffic risk monitoring"
    assert set(example["layout"]["nodes"]) == set(example["dag"]) - {"_start"}

    state, msg = backend_core_instance.check_dag(example["dag"])
    assert state is True
    assert msg == "DAG validation passed"

    structured_services = {
        "traffic-object-detection",
        "road-context-segmentation",
        "traffic-signal-recognition",
        "vehicle-reidentification-tracking",
        "vehicle-attribute-recognition",
        "vehicle-trajectory-prediction",
        "pedestrian-cyclist-pose-estimation",
        "pedestrian-cyclist-intent-recognition",
        "traffic-risk-graph-inference",
    }
    assert structured_services.issubset(set(example["dag"]) - {"_start"})

    for service_id in structured_services:
        service = backend_core_instance.find_service_by_id(service_id)
        processor_yaml = YamlOps.read_yaml(mounted_runtime / "processor" / service["yaml"])
        env = {
            item["name"]: item["value"]
            for item in processor_yaml["pod-template"]["env"]
        }
        assert env["PROCESSOR_NAME"] == "structured_processor"
        assert ast.literal_eval(env["INPUT_SERVICES"]) == []


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
                    "mounts": [
                        {
                            "source": {
                                "type": "hostPath",
                                "hostPath": {
                                    "path": "processor/face-detection/",
                                    "pathType": "Directory",
                                    "prefix": "/data/dayu-files",
                                },
                            },
                            "target": {},
                            "envName": "DEFAULT_MOUNT_PATH",
                        }
                    ],
                    "template": {
                        "spec": {
                            "nodeName": "edgex1",
                            "dnsPolicy": "ClusterFirstWithHostNet",
                            "serviceAccountName": "worker-admin",
                            "containers": [
                                {
                                    "image": "repo:5000/dayuhub/face-detection:v1.4",
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
    worker["mounts"][0]["source"]["hostPath"]["path"] = "processor/other-path/"
    worker["template"]["spec"]["containers"][0]["env"] = [{"name": "PROCESSOR_NAME", "value": "detector-v2"}]

    assert backend_core_module.BackendCore.has_significant_changes(old_doc, new_doc) is False

    new_doc["spec"]["edgeWorker"][0]["template"]["spec"]["containers"][0]["image"] = (
        "repo:5000/dayuhub/face-detection:v1.5"
    )
    assert backend_core_module.BackendCore.has_significant_changes(old_doc, new_doc) is True


@pytest.mark.unit
def test_system_log_export_uses_repeatable_snapshot_files(backend_core_instance, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    backend_core_instance.installed_running_state = True
    backend_core_instance.install_state = True
    monkeypatch.setattr(
        backend_core_instance,
        "prepare_system_visualizations_data",
        lambda: [{"id": 0, "data": {"cpu_usage": 0.42}}],
    )

    backend_core_instance.get_system_parameters()
    backend_core_instance.get_system_parameters()

    export_path = Path(backend_core_instance.create_system_log_export_file())
    try:
        with gzip.open(export_path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        assert len(payload) == 2
        assert payload[0]["data"][0]["data"]["cpu_usage"] == 0.42
    finally:
        export_path.unlink(missing_ok=True)

    second_export_path = Path(backend_core_instance.create_system_log_export_file())
    try:
        with gzip.open(second_export_path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        assert len(payload) == 2
    finally:
        second_export_path.unlink(missing_ok=True)
