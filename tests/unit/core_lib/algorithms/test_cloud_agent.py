import copy
from types import SimpleNamespace

import pytest

import core.lib.algorithms.schedule_agent as schedule_agent_package
from core.lib.algorithms.schedule_agent.cloud_agent import CloudAgent
from core.lib.common import ClassFactory, ClassType, TaskConstant


@pytest.mark.unit
def test_cloud_agent_remains_registered_without_optional_heavy_agents():
    assert schedule_agent_package.CloudAgent is CloudAgent
    assert ClassFactory.get_cls(ClassType.SCH_AGENT, "cloud") is CloudAgent


@pytest.mark.unit
def test_cloud_agent_uses_injected_cloud_node_and_preserves_input_dag():
    configuration = {"resolution": "720p", "fps": 6}
    agent = CloudAgent(
        SimpleNamespace(cloud_device="cloud-a"),
        agent_id=7,
        configuration=configuration,
    )
    dag = {
        TaskConstant.START.value: {"service": {"execute_device": "stale"}},
        "detector": {"service": {}},
        "tracker": {"service": {}},
    }
    original = copy.deepcopy(dag)

    plan = agent.get_schedule_plan({"dag": dag, "source_device": "edge-a"})

    assert plan == {
        "resolution": "720p",
        "fps": 6,
        "dag": {
            TaskConstant.START.value: {"service": {"execute_device": "edge-a"}},
            "detector": {"service": {"execute_device": "cloud-a"}},
            "tracker": {"service": {"execute_device": "cloud-a"}},
        },
    }
    assert dag == original
    assert agent.get_schedule_overhead() == 0


@pytest.mark.unit
def test_cloud_agent_rejects_missing_cloud_identity_and_malformed_dag():
    with pytest.raises(ValueError, match="system.cloud_device"):
        CloudAgent(SimpleNamespace(cloud_device=""), agent_id=1)

    agent = CloudAgent(SimpleNamespace(cloud_device="cloud-a"), agent_id=1)
    with pytest.raises(ValueError, match="malformed"):
        agent.get_schedule_plan({"dag": {"detector": {}}, "source_device": "edge-a"})
