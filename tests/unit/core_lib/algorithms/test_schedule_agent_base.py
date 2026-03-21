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
        return lambda info, name=name: {"policy": name, "info": info}

    monkeypatch.setattr(base_agent_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))

    agent = base_agent_module.BaseAgent(system="scheduler-system", agent_id=7)

    assert policy_calls == [
        ("SCH_SELECTION_POLICY", "scheduler-system", 7),
        ("SCH_INITIAL_DEPLOYMENT_POLICY", "scheduler-system", 7),
        ("SCH_REDEPLOYMENT_POLICY", "scheduler-system", 7),
    ]
    assert agent.get_source_selection_plan({"source": 1}) == {
        "policy": "SCH_SELECTION_POLICY",
        "info": {"source": 1},
    }
    assert agent.get_initial_deployment_plan({"dag": "a"}) == {
        "policy": "SCH_INITIAL_DEPLOYMENT_POLICY",
        "info": {"dag": "a"},
    }
    assert agent.get_redeployment_plan({"resource": "b"}) == {
        "policy": "SCH_REDEPLOYMENT_POLICY",
        "info": {"resource": "b"},
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
