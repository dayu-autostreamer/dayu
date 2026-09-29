import importlib

import pytest


scenario_profile_module = importlib.import_module(
    "core.lib.algorithms.scenario_extraction.structured_profile_extraction"
)
task_queue_module = importlib.import_module("core.lib.algorithms.task_queue")


@pytest.mark.unit
def test_structured_profile_extraction_only_returns_profile_dictionaries():
    extractor = scenario_profile_module.StructuredProfileExtraction()
    profile = {"frame_count": 5}

    assert extractor({"profile": profile}, object()) is profile
    assert extractor({"profile": {"frame_count": 0, "extra": "ignored-by-validator"}}, object()) == {
        "frame_count": 0,
        "extra": "ignored-by-validator",
    }
    assert extractor({"profile": ["bad"]}, object()) == {}
    assert extractor({"outputs": {}}, object()) == {}
    assert extractor("not-a-result", object()) == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    "queue_factory",
    [
        task_queue_module.SimpleQueue,
        lambda: task_queue_module.LimitQueue(max_size=10),
    ],
)
def test_task_queues_support_preview_bounded_drain_and_clear(queue_factory):
    queue = queue_factory()

    assert queue.get() is None
    queue.put("one")
    queue.put("two")
    queue.put("three")

    assert queue.get_all_without_drop() == ["one", "two", "three"]
    assert queue.size() == 3
    assert queue.drain(max_count=2) == ["one", "two"]
    assert queue.get_all_without_drop() == ["three"]
    assert queue.empty() is False

    queue.put("four")
    assert queue.drain() == ["three", "four"]
    assert queue.empty() is True

    queue.put("five")
    queue.clear()
    assert queue.empty() is True
    assert queue.get() is None
