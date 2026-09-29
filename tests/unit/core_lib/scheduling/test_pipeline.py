import copy

import pytest

from core.lib.scheduling.pipeline import (
    apply_pipeline_partition,
    materialize_pipeline_policy,
    pipeline_entries,
    pipeline_partition_index,
    rematerialize_pipeline_policy,
)


def _node(name, previous, following, device=""):
    return {
        "service": {
            "service_name": name,
            "execute_device": device,
            "scenario": {"marker": name},
            "tmp_data": {"preserve": True},
        },
        "prev_nodes": list(previous),
        "next_nodes": list(following),
        "extension": {"node": name},
    }


def _pipeline_dag(first_device="", second_device=""):
    return {
        "_start": _node("_start", [], ["detector"], "source-old"),
        "detector": _node(
            "detector", ["_start"], ["classifier"], first_device
        ),
        "classifier": _node(
            "classifier", ["detector"], ["_end"], second_device
        ),
        "_end": _node("_end", ["classifier"], [], "cloud-old"),
    }


@pytest.mark.unit
def test_partition_index_preserves_cloud_split_and_edge_semantics():
    original = _pipeline_dag()
    original_copy = copy.deepcopy(original)

    entries = pipeline_entries(original)
    assert [entry["service_name"] for entry in entries] == [
        "detector",
        "classifier",
        "_end",
    ]

    all_cloud = apply_pipeline_partition(original, 0, "edge-a", "cloud-a")
    split = apply_pipeline_partition(original, 1, "edge-a", "cloud-a")
    all_edge = apply_pipeline_partition(original, 2, "edge-a", "cloud-a")

    assert all_cloud["detector"]["service"]["execute_device"] == "cloud-a"
    assert all_cloud["classifier"]["service"]["execute_device"] == "cloud-a"
    assert split["detector"]["service"]["execute_device"] == "edge-a"
    assert split["classifier"]["service"]["execute_device"] == "cloud-a"
    assert all_edge["detector"]["service"]["execute_device"] == "edge-a"
    assert all_edge["classifier"]["service"]["execute_device"] == "edge-a"
    assert all_edge["_start"]["service"]["execute_device"] == "edge-a"
    assert all_edge["_end"]["service"]["execute_device"] == "cloud-a"

    assert all_edge["classifier"]["extension"] == {"node": "classifier"}
    assert all_edge["classifier"]["service"]["scenario"] == {
        "marker": "classifier"
    }
    assert original == original_copy

    assert pipeline_partition_index(all_cloud, "edge-a", "cloud-a") == 0
    assert pipeline_partition_index(split, "edge-a", "cloud-a") == 1
    assert pipeline_partition_index(all_edge, "edge-a", "cloud-a") == 2


@pytest.mark.unit
def test_pipeline_policy_rematerializes_current_dag_and_drops_input_fields():
    stored = materialize_pipeline_policy(
        {
            "resolution": "720p",
            "pipeline": ["obsolete"],
            "partition_index": 99,
        },
        _pipeline_dag(),
        1,
        "edge-a",
        "cloud-a",
    )
    assert "pipeline" not in stored
    assert "partition_index" not in stored

    current = _pipeline_dag()
    current["detector"]["service"]["scenario"] = {"request": "current"}
    rebound = rematerialize_pipeline_policy(
        stored,
        current,
        "edge-a",
        "cloud-a",
    )

    assert rebound["resolution"] == "720p"
    assert rebound["dag"]["detector"]["service"]["scenario"] == {
        "request": "current"
    }
    assert pipeline_partition_index(
        rebound["dag"], "edge-a", "cloud-a"
    ) == 1


@pytest.mark.unit
def test_pipeline_rejects_invalid_shapes_and_partitions():
    branching = {
        "_start": _node("_start", [], ["left", "right"]),
        "left": _node("left", ["_start"], ["_end"]),
        "right": _node("right", ["_start"], ["_end"]),
        "_end": _node("_end", ["left", "right"], []),
    }
    with pytest.raises(ValueError, match="supports only a pipeline"):
        pipeline_entries(branching)
    with pytest.raises(ValueError, match="explicit _start and _end"):
        pipeline_entries({"detector": _node("detector", [], [])})

    inconsistent = _pipeline_dag()
    inconsistent["classifier"]["prev_nodes"] = []
    with pytest.raises(ValueError, match="exactly one matching input"):
        pipeline_entries(inconsistent)

    mismatched_service = _pipeline_dag()
    mismatched_service["classifier"]["service"]["service_name"] = "other"
    with pytest.raises(ValueError, match="mismatched service name"):
        pipeline_entries(mismatched_service)

    with pytest.raises(TypeError, match="must be an integer"):
        apply_pipeline_partition(_pipeline_dag(), 1.5, "edge-a", "cloud-a")
    with pytest.raises(ValueError, match="outside"):
        apply_pipeline_partition(_pipeline_dag(), 3, "edge-a", "cloud-a")

    non_monotonic = _pipeline_dag("cloud-a", "edge-a")
    with pytest.raises(ValueError, match="non-monotonic"):
        pipeline_partition_index(non_monotonic, "edge-a", "cloud-a")
