import copy

import pytest

from core.lib.scheduling import (
    queue_waiting_counts,
    service_waiting_count,
    snapshot_queue_states,
)


@pytest.mark.unit
def test_queue_state_helpers_extract_only_structured_waiting_counts():
    resource = {
        "queue_state": {
            "detector": {"waiting_count": 3, "busy": True},
            "classifier": {"waiting_count": "2.5", "busy": False},
            "negative": {"waiting_count": -4},
            "invalid": {"waiting_count": "unknown"},
            "missing": {"busy": False},
        }
    }

    assert queue_waiting_counts(resource) == {
        "detector": 3.0,
        "classifier": 2.5,
        "negative": 0.0,
    }
    assert service_waiting_count(resource, "classifier") == 2.5
    assert service_waiting_count(resource, "missing") == 0.0


@pytest.mark.unit
def test_queue_state_helpers_do_not_accept_the_removed_queue_length_contract():
    resource = {"queue_length": {"detector": 9}}

    assert queue_waiting_counts(resource) == {}
    assert service_waiting_count(resource, "detector") == 0.0


@pytest.mark.unit
def test_snapshot_queue_states_filters_revision_and_scheduler_staleness():
    snapshot = {
        "captured_at": 10.0,
        "runtime_directory_revision": 4,
        "resources": {
            "edge-a": {"queue_state": {"detect": {
                "busy": True,
                "observed_at": 100.0,
            }}},
            "edge-b": {"queue_state": {"detect": {"busy": False}}},
            "old-edge": {"queue_state": {"detect": {"busy": True}}},
        },
        "resource_received_at": {"edge-a": 7.0, "edge-b": 9.5},
        "resource_runtime_revision": {
            "edge-a": 4,
            "edge-b": 4,
            "old-edge": 3,
        },
    }
    original = copy.deepcopy(snapshot)

    states = snapshot_queue_states(snapshot, max_age_s=1.0)

    assert states == {("detect", "edge-b"): {
        "busy": False,
        "_age_s": 0.5,
    }}
    assert snapshot == original


@pytest.mark.unit
def test_snapshot_queue_states_treats_missing_receive_time_as_current():
    states = snapshot_queue_states({
        "captured_at": 10.0,
        "runtime_directory_revision": 1,
        "resources": {
            "edge-a": {"queue_state": {"detect": {
                "busy": True,
                "observed_at": 1.0,
            }}},
        },
        "resource_received_at": {},
        "resource_runtime_revision": {"edge-a": 1},
    })

    assert states[("detect", "edge-a")]["_age_s"] == 0.0
