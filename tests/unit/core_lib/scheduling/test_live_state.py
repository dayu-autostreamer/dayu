from types import SimpleNamespace

import pytest

from core.lib.scheduling import SchedulingSnapshotScope
from core.lib.scheduling.live_state import (
    active_deployment_for_dag,
    get_live_snapshot,
    live_resources,
    require_active_plan,
)


def _dag():
    return {
        "_start": {},
        "detector": {},
        "classifier": {},
        "_end": {},
    }


@pytest.mark.unit
def test_live_state_uses_one_revision_for_deployment_and_resources():
    snapshot = {
        "runtime_directory_revision": 7,
        "deployment": {
            "detector": ["edge-a", "cloud-a"],
            "classifier": ["cloud-a"],
        },
        "resources": {
            "edge-a": {"cpu_usage": 20},
            "cloud-a": {"cpu_usage": 30},
            "stale-edge": {"cpu_usage": 99},
        },
        "resource_runtime_revision": {
            "edge-a": 7,
            "cloud-a": 7,
            "stale-edge": 6,
        },
    }
    scopes = []

    def get_scheduling_snapshot(scope):
        scopes.append(scope)
        return snapshot

    system = SimpleNamespace(get_scheduling_snapshot=get_scheduling_snapshot)
    live = get_live_snapshot(system)
    live["deployment"]["detector"].append("mutated")
    assert "mutated" not in snapshot["deployment"]["detector"]

    captured, deployment = active_deployment_for_dag(system, _dag())
    assert deployment["detector"] == ["cloud-a", "edge-a"]
    assert live_resources(captured) == {
        "edge-a": {"cpu_usage": 20},
        "cloud-a": {"cpu_usage": 30},
    }
    assert require_active_plan(
        {"detector": "edge-a", "classifier": "cloud-a"},
        deployment,
    ) == {"detector": "edge-a", "classifier": "cloud-a"}
    with pytest.raises(ValueError, match="not active"):
        require_active_plan({"detector": "edge-b"}, deployment)

    assert scopes == [
        SchedulingSnapshotScope.LIVE,
        SchedulingSnapshotScope.LIVE,
    ]


@pytest.mark.unit
def test_live_state_fails_closed_before_directory_is_ready():
    system = SimpleNamespace(
        get_scheduling_snapshot=lambda scope: {
            "runtime_directory_revision": 0,
            "deployment": {"detector": ["edge-a"]},
        }
    )
    with pytest.raises(ValueError, match="not ready"):
        get_live_snapshot(system)
    assert live_resources({
        "runtime_directory_revision": 0,
        "resources": {"edge-a": {"cpu_usage": 10}},
        "resource_runtime_revision": {"edge-a": 0},
    }) == {}
