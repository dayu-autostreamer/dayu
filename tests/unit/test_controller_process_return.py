import importlib

import pytest

from core.lib.content import Task


def build_parallel_branch_task(branch_name, value, root_uuid="root-task-0"):
    dag = Task.extract_dag_from_dag_deployment(
        {
            "detector-a": {
                "service": {"service_name": "detector-a", "execute_device": "edge-node"},
                "next_nodes": ["join"],
            },
            "detector-b": {
                "service": {"service_name": "detector-b", "execute_device": "edge-node"},
                "next_nodes": ["join"],
            },
            "join": {
                "service": {"service_name": "join", "execute_device": "edge-node"},
                "next_nodes": [],
            },
        }
    )
    task = Task(
        source_id=0,
        task_id=0,
        source_device="edge-node",
        all_edge_devices=["edge-node"],
        dag=dag,
        flow_index=branch_name,
        past_flow_index="_start",
        metadata={"buffer_size": 1},
        raw_metadata={"buffer_size": 1},
        file_path="payload.bin",
        root_uuid=root_uuid,
        runtime_directory_revision=1,
    )
    task.set_current_content({"branch": branch_name, "value": value})
    task.add_scenario({"branch": value})
    return task


class RecordingTaskCoordinator:
    def __init__(self):
        self.stored_tasks = []
        self.completed = []

    def arrive(self, task, joint_service_name, required_count):
        by_branch = {stored.get_past_flow_index(): stored for stored in self.stored_tasks}
        by_branch[task.get_past_flow_index()] = task
        self.stored_tasks = list(by_branch.values())
        return list(self.stored_tasks) if len(self.stored_tasks) == required_count else None

    def complete(self, root_uuid, joint_service_name):
        self.completed.append((root_uuid, joint_service_name))
        self.stored_tasks = []


@pytest.mark.unit
def test_process_return_waits_until_all_parallel_branches_arrive():
    controller_module = importlib.import_module("core.controller.controller")
    controller = object.__new__(controller_module.Controller)
    controller.task_coordinator = RecordingTaskCoordinator()

    submitted_tasks = []
    controller.submit_task = lambda task: submitted_tasks.append(task) or True

    first_branch_task = build_parallel_branch_task("detector-a", "left-branch")

    accepted = controller.process_return(first_branch_task)

    assert accepted is True
    assert submitted_tasks == []
    assert len(controller.task_coordinator.stored_tasks) == 1
    assert controller.task_coordinator.stored_tasks[0].get_flow_index() == "join"
    assert controller.task_coordinator.stored_tasks[0].get_past_flow_index() == "detector-a"


@pytest.mark.unit
def test_process_return_merges_parallel_branch_results_before_submitting():
    controller_module = importlib.import_module("core.controller.controller")
    controller = object.__new__(controller_module.Controller)
    controller.task_coordinator = RecordingTaskCoordinator()

    submitted_tasks = []
    controller.submit_task = lambda task: submitted_tasks.append(task) or True

    first_branch_task = build_parallel_branch_task("detector-a", "left-branch")
    second_branch_task = build_parallel_branch_task("detector-b", "right-branch")

    assert controller.process_return(first_branch_task) is True

    accepted = controller.process_return(second_branch_task)

    assert accepted is True
    assert len(submitted_tasks) == 1
    assert controller.task_coordinator.completed == [(second_branch_task.get_root_uuid(), "join")]

    merged_task = submitted_tasks[0]
    assert merged_task.get_flow_index() == "join"
    assert merged_task.get_service("detector-a").get_content_data() == {
        "branch": "detector-a",
        "value": "left-branch",
    }
    assert merged_task.get_service("detector-b").get_content_data() == {
        "branch": "detector-b",
        "value": "right-branch",
    }
    assert merged_task.get_service("detector-a").get_scenario_data() == {"branch": "left-branch"}
    assert merged_task.get_service("detector-b").get_scenario_data() == {"branch": "right-branch"}


@pytest.mark.unit
def test_process_return_keeps_ready_barrier_when_downstream_rejects_delivery():
    controller_module = importlib.import_module("core.controller.controller")
    controller = object.__new__(controller_module.Controller)
    controller.task_coordinator = RecordingTaskCoordinator()
    controller.submit_task = lambda task: False
    first = build_parallel_branch_task("detector-a", "left")
    second = build_parallel_branch_task("detector-b", "right")

    assert controller.process_return(first) is True
    assert controller.process_return(second) is False

    assert len(controller.task_coordinator.stored_tasks) == 2
    assert controller.task_coordinator.completed == []


@pytest.mark.unit
def test_merge_task_preserves_completed_ancestors_from_nested_join():
    dag_deployment = {
        "left": {
            "service": {"service_name": "left", "execute_device": "edge-node"},
            "next_nodes": ["nested-leaf"],
        },
        "right": {
            "service": {"service_name": "right", "execute_device": "edge-node"},
            "next_nodes": ["nested-join", "final-join"],
        },
        "nested-leaf": {
            "service": {"service_name": "nested-leaf", "execute_device": "edge-node"},
            "next_nodes": ["nested-join"],
        },
        "nested-join": {
            "service": {"service_name": "nested-join", "execute_device": "edge-node"},
            "next_nodes": ["final-join"],
        },
        "final-join": {
            "service": {"service_name": "final-join", "execute_device": "edge-node"},
            "next_nodes": [],
        },
    }

    def build_task(past_flow_index, completed):
        task = Task(
            source_id=0,
            task_id=0,
            source_device="edge-node",
            all_edge_devices=["edge-node"],
            dag=Task.extract_dag_from_dag_deployment(dag_deployment),
            flow_index="final-join",
            past_flow_index=past_flow_index,
            metadata={"buffer_size": 1},
            raw_metadata={"buffer_size": 1},
            file_path="payload.bin",
            root_uuid="nested-root",
            runtime_directory_revision=1,
        )
        for service_name, duration in completed.items():
            task.get_service(service_name).set_real_execute_time(duration)
        return task

    final_base = build_task("right", {"right": 2.0})
    nested_branch = build_task(
        "nested-join",
        {
            "left": 1.0,
            "right": 2.0,
            "nested-leaf": 3.0,
            "nested-join": 4.0,
        },
    )

    final_base.merge_task(nested_branch)

    assert final_base.get_service("left").get_real_execute_time() == 1.0
    assert final_base.get_service("right").get_real_execute_time() == 2.0
    assert final_base.get_service("nested-leaf").get_real_execute_time() == 3.0
    assert final_base.get_service("nested-join").get_real_execute_time() == 4.0
