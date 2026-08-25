import pytest

from core.lib.scheduling import (
    SchedulingSnapshotScope,
    deployment_from_snapshot,
    normalize_scheduling_snapshot_scope,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, SchedulingSnapshotScope.COMMITTED),
        (SchedulingSnapshotScope.LIVE, SchedulingSnapshotScope.LIVE),
        (" live ", SchedulingSnapshotScope.LIVE),
        ("COMMITTED", SchedulingSnapshotScope.COMMITTED),
    ],
)
def test_normalize_scheduling_snapshot_scope_accepts_supported_values(
    value,
    expected,
):
    assert normalize_scheduling_snapshot_scope(value) is expected


@pytest.mark.unit
def test_normalize_scheduling_snapshot_scope_rejects_unknown_values():
    with pytest.raises(
        ValueError,
        match=r"unsupported scheduling snapshot scope 'future'.*live, committed",
    ):
        normalize_scheduling_snapshot_scope("future")


@pytest.mark.unit
def test_deployment_from_snapshot_normalizes_service_replicas():
    assert deployment_from_snapshot({
        "deployment": {
            "detect": "edge-a",
            "classify": ["edge-c", "edge-b", "edge-b", ""],
            "invalid": 3,
        },
    }) == {
        "detect": ["edge-a"],
        "classify": ["edge-b", "edge-c"],
    }


@pytest.mark.unit
@pytest.mark.parametrize("snapshot", [None, {}, {"deployment": []}])
def test_deployment_from_snapshot_requires_a_fixed_deployment(snapshot):
    with pytest.raises(ValueError, match="no fixed deployment"):
        deployment_from_snapshot(snapshot)
