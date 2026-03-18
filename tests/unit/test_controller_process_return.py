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
    )
    task.set_current_content({"branch": branch_name, "value": value})
    task.add_scenario({"branch": value})
    return task


class RecordingTaskCoordinator:
    def __init__(self):
        self.stored_tasks = []

    def store_task_data(self, task, joint_service_name):
        self.stored_tasks.append(task)
        return len(self.stored_tasks)

    def retrieve_task_data(self, root_uuid, joint_service_name, required_count):
        return list(self.stored_tasks)


@pytest.mark.unit
def test_process_return_waits_until_all_parallel_branches_arrive():
    controller_module = importlib.import_module("core.controller.controller")
    controller = object.__new__(controller_module.Controller)
    controller.task_coordinator = RecordingTaskCoordinator()

    submitted_tasks = []
    controller.submit_task = lambda task: submitted_tasks.append(task) or "execute"

    first_branch_task = build_parallel_branch_task("detector-a", "left-branch")

    actions = controller.process_return(first_branch_task)

    assert actions == ["wait"]
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
    controller.submit_task = lambda task: submitted_tasks.append(task) or "execute"

    first_branch_task = build_parallel_branch_task("detector-a", "left-branch")
    second_branch_task = build_parallel_branch_task("detector-b", "right-branch")

    assert controller.process_return(first_branch_task) == ["wait"]

    actions = controller.process_return(second_branch_task)

    assert actions == ["execute"]
    assert len(submitted_tasks) == 1

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
