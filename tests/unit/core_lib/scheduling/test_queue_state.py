import pytest

from core.lib.scheduling import queue_waiting_counts, service_waiting_count


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
