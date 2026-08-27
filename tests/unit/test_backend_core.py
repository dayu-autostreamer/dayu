import gzip
import json
from pathlib import Path
from types import SimpleNamespace

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
def backend_core_instance(mounted_runtime):
    from backend_core import BackendCore

    return BackendCore()


def test_check_dag_validates_service_input_output_contracts(backend_core_instance):
    assert backend_core_instance.check_dag(make_valid_dag()) == (
        True, "DAG validation passed"
    )

    invalid_dag = {
        "_start": ["gender-classification"],
        "gender-classification": {
            "id": "gender-classification", "prev": [], "succ": ["face-detection"],
        },
        "face-detection": {
            "id": "face-detection", "prev": ["gender-classification"], "succ": [],
        },
    }
    state, message = backend_core_instance.check_dag(invalid_dag)
    assert state is False
    assert "Node connection mismatch" in message

    face_service = backend_core_instance.find_service_by_id("face-detection")
    original = face_service["input"]
    face_service["input"] = "frame"
    state, message = backend_core_instance.check_dag(make_valid_dag())
    assert state is False
    assert "must be a list" in message
    face_service["input"] = original


def test_structured_traffic_catalog_and_example_dag_are_consistent(
        backend_core_instance, mounted_runtime,
):
    from core.lib.common import YamlOps

    repo_root = Path(__file__).resolve().parents[2]
    example = YamlOps.read_yaml(
        repo_root / "config" / "application_dags" / "driving_risk_perception.dag"
    )
    assert example["format"] == "dayu.application-dag"
    assert example["version"] == 1
    assert backend_core_instance.check_dag(example["dag"]) == (
        True, "DAG validation passed"
    )

    structured_services = set(example["dag"]) - {"_start"}
    for service_id in structured_services:
        service = backend_core_instance.find_service_by_id(service_id)
        assert service is not None
        processor_yaml = YamlOps.read_yaml(mounted_runtime / "processor" / service["yaml"])
        env = {item["name"]: item["value"] for item in processor_yaml["pod-template"]["env"]}
        assert env["PROCESSOR_NAME"] == "structured_processor"


def test_system_log_export_uses_repeatable_snapshots(
        backend_core_instance, monkeypatch, tmp_path,
):
    monkeypatch.chdir(tmp_path)
    backend_core_instance.system_log_store_path = str(tmp_path / "system.jsonl")
    session = SimpleNamespace(
        phase="active",
        install_id="install-a",
        active_directory_revision=1,
    )
    backend_core_instance.runtime_orchestrator = SimpleNamespace(
        current_session=lambda: session,
    )
    backend_core_instance._bound_runtime_key = ("install-a", 1)
    monkeypatch.setattr(
        backend_core_instance,
        "prepare_system_visualizations_data",
        lambda: [{"id": 0, "data": {"cpu_usage": 0.42}}],
    )

    backend_core_instance.get_system_parameters()
    backend_core_instance.get_system_parameters()

    for _ in range(2):
        export_path = Path(backend_core_instance.create_system_log_export_file())
        try:
            with gzip.open(export_path, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
            assert len(payload) == 2
            assert payload[0]["data"][0]["data"]["cpu_usage"] == 0.42
        finally:
            export_path.unlink(missing_ok=True)
