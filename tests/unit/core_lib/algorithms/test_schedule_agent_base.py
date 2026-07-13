import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


BASE_AGENT_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "dependency"
    / "core"
    / "lib"
    / "algorithms"
    / "schedule_agent"
    / "base_agent.py"
)


def load_base_agent_module():
    module_name = "dayu_test_schedule_agent_base"
    spec = importlib.util.spec_from_file_location(module_name, BASE_AGENT_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


@pytest.mark.unit
def test_base_agent_initializes_policy_hooks_and_delegates_plans(monkeypatch):
    base_agent_module = load_base_agent_module()
    policy_calls = []

    def fake_get_algorithm(name, **kwargs):
        policy_calls.append((name, kwargs["system"], kwargs["agent_id"]))
        if name == "SCH_SELECTION_POLICY":
            return lambda info: {"policy": name, "info": info}
        return lambda info: {"detector": ["edge-a"]}

    monkeypatch.setattr(base_agent_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))

    system = SimpleNamespace(cloud_device="cloud-a")
    agent = base_agent_module.BaseAgent(system=system, agent_id=7)

    assert policy_calls == [
        ("SCH_SELECTION_POLICY", system, 7),
        ("SCH_INITIAL_DEPLOYMENT_POLICY", system, 7),
        ("SCH_REDEPLOYMENT_POLICY", system, 7),
    ]
    assert agent.get_source_selection_plan({"source": 1}) == {
        "policy": "SCH_SELECTION_POLICY",
        "info": {"source": 1},
    }
    deployment_info = {
        "dag": {"detector": {}},
        "node_set": ["edge-a"],
    }
    assert agent.get_initial_deployment_plan(deployment_info) == {
        "detector": ["edge-a"],
    }
    assert agent.get_redeployment_plan(deployment_info) == {
        "detector": ["edge-a"],
    }

    agent.redeployment_policy = lambda info: {
        "detector": ["edge-a"], "stale-service": ["edge-a"],
    }
    with pytest.raises(ValueError, match="outside the current DAG"):
        agent.get_redeployment_plan(deployment_info)
    assert agent.should_generate({"source_id": 7}) == {
        "generate": True,
        "reason": "default_allow",
    }
    assert agent.get_schedule_overhead() == 0


@pytest.mark.unit
def test_base_agent_default_abstract_contracts_raise_not_implemented(monkeypatch):
    base_agent_module = load_base_agent_module()
    monkeypatch.setattr(
        base_agent_module.Context,
        "get_algorithm",
        staticmethod(lambda name, **kwargs: lambda info: info),
    )
    agent = base_agent_module.BaseAgent(system=SimpleNamespace(), agent_id=1)

    with pytest.raises(NotImplementedError):
        agent()
    with pytest.raises(NotImplementedError):
        agent.update_scenario({})
    with pytest.raises(NotImplementedError):
        agent.update_resource("edge-a", {})
    with pytest.raises(NotImplementedError):
        agent.update_policy({})
    with pytest.raises(NotImplementedError):
        agent.update_task({})
    with pytest.raises(NotImplementedError):
        agent.get_schedule_plan({})
    with pytest.raises(NotImplementedError):
        agent.run()
