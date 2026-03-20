import importlib
from types import SimpleNamespace

import pytest

from core.lib.common import Counter, TaskConstant
from core.lib.content import Task


after_schedule_module = importlib.import_module("core.lib.algorithms.after_schedule_operation.casva_operation")
simple_filter_module = importlib.import_module("core.lib.algorithms.frame_filter.simple_filter")
task_queue_module = importlib.import_module("core.lib.algorithms.task_queue.limit_queue")


def service_entry(name, *, execute_device="", next_nodes=None, prev_nodes=None):
    return {
        "service": {
            "service_name": name,
            "execute_device": execute_device,
        },
        "next_nodes": next_nodes or [],
        "prev_nodes": prev_nodes or [],
    }


def build_task():
    dag = Task.extract_dag_from_dict(
        {
            "detector": service_entry("detector", execute_device="edge-a", next_nodes=["classifier"]),
            "classifier": service_entry("classifier", execute_device="edge-b"),
        }
    )
    return Task(
        source_id=1,
        task_id=2,
        source_device="edge-a",
        all_edge_devices=["edge-a", "edge-b"],
        dag=dag,
        metadata={"fps": 10, "buffer_size": 2},
        raw_metadata={"fps": 20, "buffer_size": 2},
        file_path="payload.mp4",
    )


@pytest.mark.unit
def test_simple_filter_covers_same_skip_and_remain_modes():
    simple_filter = simple_filter_module.SimpleFilter()
    assert simple_filter.get_fps_adjust_mode(20, 20) == ("same", 0, 0)
    assert simple_filter.get_fps_adjust_mode(20, 12) == ("skip", 2, 0)
    assert simple_filter.get_fps_adjust_mode(20, 4) == ("remain", 0, 5)

    Counter.reset_all_counts()
    remain_system = SimpleNamespace(raw_meta_data={"fps": 20}, meta_data={"fps": 4})
    assert [simple_filter(remain_system, object()) for _ in range(5)] == [False, False, False, False, True]


@pytest.mark.unit
def test_limit_queue_discards_oldest_half_when_capacity_is_reached():
    queue = task_queue_module.LimitQueue(max_size=2)
    assert queue.empty() is True

    queue.put("one")
    queue.put("two")
    queue.put("three")

    assert queue.size() == 2
    assert queue.get() == "two"
    assert queue.get() == "three"
    assert queue.get() is None
    assert queue.empty() is True


@pytest.mark.unit
def test_casva_after_schedule_operation_keeps_local_execution_without_scheduler_and_preserves_explicit_qp(monkeypatch):
    task = build_task()
    system = SimpleNamespace(
        task_dag=task.get_dag(),
        local_device="edge-a",
        meta_data={"fps": 10},
        service_deployment={},
    )
    operation = after_schedule_module.CASVAASOperation()

    operation(system, None)
    assert all(node.service.get_execute_device() == "edge-a" for node in system.task_dag.nodes.values())

    dag_deployment = Task.extract_dag_deployment_from_dag(build_task().get_dag())
    dag_deployment["detector"]["service"]["execute_device"] = "edge-b"
    monkeypatch.setattr(after_schedule_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-a"))
    system.meta_data = {"fps": 10, "qp": 31}

    operation(system, {"plan": {"fps": 5, "dag": dag_deployment}, "deployment": {"detector": ["edge-b"]}})

    assert system.meta_data["fps"] == 5
    assert system.meta_data["qp"] == 31
    assert system.service_deployment == {"detector": ["edge-b"]}
    assert system.task_dag.get_node("detector").service.get_execute_device() == "edge-b"
    assert system.task_dag.get_end_node().service.get_execute_device() == "cloud-a"
    assert system.task_dag.get_start_node().service.get_execute_device() == "edge-a"
