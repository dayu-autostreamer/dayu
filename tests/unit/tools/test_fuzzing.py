"""Replay contract fuzz recipes without requiring the native Atheris engine."""

import importlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
FUZZ = ROOT / "tests" / "fuzz"
TARGETS = ("dag", "schedule_plan", "runtime_routes")
pytestmark = pytest.mark.unit


@pytest.fixture
def targets(monkeypatch):
    monkeypatch.syspath_prepend(str(FUZZ))
    return {name: importlib.import_module("fuzz_" + name) for name in TARGETS}


@pytest.mark.parametrize(
    "seed", sorted((FUZZ / "corpus").glob("*/*.json")), ids=lambda seed: str(seed.relative_to(FUZZ))
)
def test_committed_corpus(targets, seed):
    targets[seed.parent.name].check_input(seed.read_bytes())


@pytest.mark.parametrize("name", TARGETS)
def test_structured_mutations_are_deterministic_bounded_and_replayable(targets, name):
    target = targets[name]
    data = b"{}"
    for seed in range(200):
        mutated = target.mutate(data, 8192, seed)
        assert mutated == target.mutate(data, 8192, seed)
        assert len(mutated) <= 8192
        assert isinstance(json.loads(mutated), dict)
        target.check_input(mutated)
        data = mutated
    for size in (0, 1, 2, 8, 32):
        assert len(target.mutate(data, size, 1)) <= size


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\xff",
        b"null",
        b"[]",
        b"{" * 9000,
        b"[" * 2000 + b"]" * 2000,
        b'{"nodes":[],"edges":[null,{},["a",1]],"placements":[null],"port":{}}',
    ],
)
@pytest.mark.parametrize("name", TARGETS)
def test_malformed_recipes_still_exercise_a_valid_baseline(targets, name, data):
    targets[name].check_input(data)


@pytest.mark.parametrize("shape", ("snake", "camel", "nested", "url"))
@pytest.mark.parametrize("port", (1, 65535))
def test_endpoint_shapes_and_port_boundaries(targets, shape, port):
    targets["runtime_routes"].check_input(json.dumps({"shape": shape, "port": port, "path": "/api/v1"}).encode())


@pytest.mark.parametrize(
    "field",
    (
        "runtime_id",
        "runtime_service_uid",
        "service_uid",
        "endpoint_pod_uid",
        "deployment_revision",
        "fqdn",
        "target_node",
        "component",
        "logical_service",
    ),
)
def test_missing_exact_route_identity(targets, field):
    targets["runtime_routes"].check_input(json.dumps({"mode": "missing", "field": field}).encode())


@pytest.mark.parametrize(
    "field",
    (
        "runtime_id",
        "runtime_service_uid",
        "service_uid",
        "endpoint_pod_uid",
        "deployment_revision",
        "fqdn",
        "port",
        "protocol",
        "base_path",
    ),
)
def test_conflicting_exact_routes(targets, field):
    targets["runtime_routes"].check_input(json.dumps({"mode": "conflict", "field": field}).encode())


def test_unexpected_production_exceptions_are_not_swallowed(targets, monkeypatch):
    from core.lib.scheduling import deployment_plan

    def unexpected(*args, **kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(deployment_plan, "validate_plan", unexpected)
    with pytest.raises(RuntimeError, match="unexpected failure"):
        targets["schedule_plan"].check_input(b'{"mode":"outside"}')


def test_oracle_detects_incorrect_production_result(targets, monkeypatch):
    from core.lib.scheduling import dag

    monkeypatch.setattr(dag, "topological_order", lambda value: [])
    with pytest.raises(AssertionError):
        targets["dag"].check_input(b"{}")


def test_runner_preserves_process_failure_and_log(tmp_path):
    from tools.run_fuzz import run_process

    log = tmp_path / "fuzz.log"
    code = run_process([sys.executable, "-c", 'print("crash evidence"); raise SystemExit(7)'], log, 5, dict(os.environ))
    assert code == 7
    assert "crash evidence" in log.read_text()


def test_runner_terminates_a_stalled_process(tmp_path):
    from tools.run_fuzz import run_process

    log = tmp_path / "fuzz.log"
    code = run_process([sys.executable, "-c", "import time; time.sleep(30)"], log, 0.1, dict(os.environ))
    assert code == 124
    assert "wall-clock timeout" in log.read_text()


def test_runner_keeps_failure_status_across_targets_and_isolates_environment(monkeypatch, tmp_path):
    from tools import run_fuzz

    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    monkeypatch.setenv("PROCESSOR_SERVICE_NAME", "processor-live")
    monkeypatch.setenv("DAYU_SCHEDULER_ENDPOINT", "http://live.example")
    seen = []

    def simulated_engine(command, log_path, timeout, env):
        assert "PYTHONOPTIMIZE" not in env
        assert "DAYU_SCHEDULER_ENDPOINT" not in env
        assert env["PROCESSOR_SERVICE_NAME"] == ""
        seen.append(log_path.parent.name)
        log_path.write_text("simulated engine result\n")
        return 7 if log_path.parent.name == "schedule_plan" else 0

    original = {path: path.read_bytes() for path in (FUZZ / "corpus").glob("*/*.json")}
    monkeypatch.setattr(run_fuzz, "run_process", simulated_engine)
    assert run_fuzz.main(["--output", str(tmp_path)]) == 1
    assert seen == list(TARGETS)
    assert json.loads((tmp_path / "schedule_plan" / "run.json").read_text())["returncode"] == 7
    assert all(path.read_bytes() == data for path, data in original.items())


@pytest.mark.parametrize(
    "argv",
    [
        ["--seconds", "0"],
        ["--seconds", "3601"],
        ["--replay", __file__],
        ["--target", "dag", "--replay", "/nonexistent/input"],
        ["--output", str(FUZZ / "corpus")],
    ],
)
def test_runner_rejects_unbounded_or_unsafe_arguments(argv):
    from tools.run_fuzz import main

    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
