import importlib

import pytest

from core.lib.common import NameMaintainer, TaskConstant
from core.lib.content import DAG, Service, Task
from core.lib.solver import PathSolver


dag_module = importlib.import_module("core.lib.content.dag")
time_estimation_module = importlib.import_module("core.lib.estimation.time_estimation")


def service_entry(name, *, execute_device="", next_nodes=None, prev_nodes=None):
    return {
        "service": {
            "service_name": name,
            "execute_device": execute_device,
        },
        "next_nodes": next_nodes or [],
        "prev_nodes": prev_nodes or [],
    }


def build_branching_task():
    dag = Task.extract_dag_from_dict(
        {
            "a": service_entry("a", execute_device="edge-a", next_nodes=["join"]),
            "b": service_entry("b", execute_device="edge-b", next_nodes=["join"]),
            "join": service_entry("join", execute_device="cloud-a"),
        }
    )
    return Task(
        source_id=1,
        task_id=2,
        source_device="edge-a",
        all_edge_devices=["edge-a", "edge-b"],
        dag=dag,
        metadata={"buffer_size": 2},
        raw_metadata={"buffer_size": 2},
        file_path="payload.bin",
    )


@pytest.mark.unit
def test_service_supports_roundtrip_and_time_ticket_guards():
    service = Service(
        "detector",
        execute_device="edge-a",
        transmit_time=0.2,
        execute_time=0.5,
        real_execute_time=0.4,
        content={"boxes": 1},
        scenario={"obj_num": 1},
        temp={"seed": 1},
    )

    service.add_scenario({"velocity": 2})
    assert service.get_service_total_time() == pytest.approx(0.7)
    service.record_time_ticket("queue_start", 1.0)
    with pytest.raises(AssertionError, match="already exists"):
        service.record_time_ticket("queue_start", 2.0)

    service.erase_time_ticket("queue_start")
    with pytest.raises(AssertionError, match="does not exist"):
        service.erase_time_ticket("queue_start")

    with pytest.raises(AssertionError, match="negative"):
        service.set_transmit_time(-1)
    with pytest.raises(AssertionError, match="negative"):
        service.set_execute_time(-1)
    with pytest.raises(AssertionError, match="negative"):
        service.set_real_execute_time(-1)

    restored = Service.deserialize(service.serialize())
    assert restored.to_dict() == service.to_dict()
    assert restored == Service("detector")
    assert repr(restored) == "detector"


@pytest.mark.unit
def test_dag_validation_repairs_missing_edges_and_rejects_invalid_graphs():
    start = TaskConstant.START.value
    end = TaskConstant.END.value

    dag = DAG.from_dict(
        {
            start: service_entry(start, next_nodes=["decode"]),
            "decode": service_entry("decode", execute_device="edge-a", next_nodes=["infer"]),
            "infer": service_entry("infer", execute_device="cloud-a"),
            end: service_entry(end, prev_nodes=["infer"]),
        }
    )

    dag.validate_dag()
    assert dag.get_prev_nodes("decode") == [start]
    assert dag.get_prev_nodes("infer") == ["decode"]
    assert dag.get_next_nodes("infer") == [end]
    assert dag.check_is_pipeline() is True
    assert DAG.deserialize(dag.serialize()).to_dict() == dag.to_dict()

    cycle = DAG.from_dict(
        {
            start: service_entry(start, next_nodes=["a"]),
            "a": service_entry("a", next_nodes=["b"]),
            "b": service_entry("b", next_nodes=["a", end]),
            end: service_entry(end),
        }
    )
    with pytest.raises(ValueError, match="Cycle detected"):
        cycle.validate_dag()

    disconnected = DAG.from_dict(
        {
            start: service_entry(start, next_nodes=["a"]),
            "a": service_entry("a", next_nodes=[end]),
            end: service_entry(end),
            "x": service_entry("x", next_nodes=["y"]),
            "y": service_entry("y"),
        }
    )
    with pytest.raises(ValueError, match="disconnected components"):
        disconnected.validate_dag()


@pytest.mark.unit
def test_path_solver_finds_shortest_all_and_weighted_longest_paths():
    start = TaskConstant.START.value
    end = TaskConstant.END.value
    dag = Task.extract_dag_from_dict(
        {
            "b": service_entry("b", next_nodes=["d"]),
            "c": service_entry("c", next_nodes=[end]),
            "d": service_entry("d", next_nodes=[end]),
        }
    )
    dag.get_node(start).service.set_execute_time(0)
    dag.get_node("b").service.set_execute_time(1)
    dag.get_node("c").service.set_execute_time(100)
    dag.get_node("d").service.set_execute_time(1000)
    dag.get_node(end).service.set_execute_time(0)

    solver = PathSolver(dag)
    assert solver.get_shortest_path(start, end) == [start, "c", end]
    assert sorted(solver.get_all_paths(start, end)) == sorted([[start, "b", "d", end], [start, "c", end]])

    weight = lambda service: service.get_execute_time()
    assert solver.get_weighted_shortest_path(start, end, weight) == (100, [start, "c", end])
    assert solver.get_weighted_longest_path(start, end, weight) == (1001, [start, "b", "d", end])

    no_path = DAG.from_dict(
        {
            start: service_entry(start, next_nodes=["a"]),
            "a": service_entry("a"),
            end: service_entry(end),
        }
    )
    with pytest.raises(ValueError, match="No path exists"):
        PathSolver(no_path).get_shortest_path(start, end)


@pytest.mark.unit
def test_task_pipeline_conversion_stage_navigation_and_branch_merge():
    pipeline = [
        {"service_name": "decode", "execute_device": "edge-a"},
        {"service_name": "infer", "execute_device": "cloud-a"},
    ]
    dag = Task.extract_dag_from_pipeline_deployment(pipeline)
    assert Task.extract_pipeline_deployment_from_dag(dag) == pipeline
    assert Task.extract_pipeline_deployment_from_dag_deployment(
        Task.extract_dag_deployment_from_pipeline_deployment(pipeline)
    ) == pipeline

    with pytest.raises(ValueError, match="not a pipeline structure"):
        Task.extract_pipeline_deployment_from_dag(build_branching_task().get_dag())

    task = build_branching_task()
    branches = {branch.get_flow_index(): branch for branch in task.step_to_next_stage()}
    assert set(branches) == {"a", "b"}
    assert branches["a"].get_past_flow_index() == TaskConstant.START.value

    branches["a"].set_current_content({"branch": "a"})
    branches["b"].set_current_content({"branch": "b"})
    assert branches["a"].get_parallel_info_for_merge() == [
        {"joint_service": "join", "parallel_services": ["a", "b"]}
    ]

    join_from_a = branches["a"].step_to_next_stage()[0]
    join_from_b = branches["b"].step_to_next_stage()[0]
    join_from_a.merge_task(join_from_b)
    assert join_from_a.get_service("b").get_content_data() == {"branch": "b"}

    roundtrip = Task.deserialize(join_from_a.serialize())
    assert roundtrip.to_dict() == join_from_a.to_dict()


@pytest.mark.unit
def test_task_delay_calculation_time_tickets_and_estimator_helpers(monkeypatch):
    task = Task(
        source_id=3,
        task_id=4,
        source_device="edge-a",
        all_edge_devices=["edge-a"],
        dag=Task.extract_dag_from_dict({"decode": service_entry("decode", execute_device="edge-a")}),
        metadata={"buffer_size": 2},
        raw_metadata={"buffer_size": 2},
        file_path="payload.bin",
    )

    task.set_flow_index("decode")
    task.save_transmit_time(0.5)
    task.save_execute_time(1.0)
    task.save_real_execute_time(0.8)

    time_values = iter([10.0, 12.0, 13.0, 16.0])
    monkeypatch.setattr(time_estimation_module.time, "time", lambda: next(time_values))

    assert time_estimation_module.TimeEstimator.record_task_ts(task, "total_start_time") == 0
    assert time_estimation_module.TimeEstimator.record_dag_ts(task, is_end=False, sub_tag="transmit") == 0
    assert time_estimation_module.TimeEstimator.record_dag_ts(task, is_end=True, sub_tag="transmit") == pytest.approx(1.0)
    assert time_estimation_module.TimeEstimator.record_task_ts(task, "total_end_time") == 0

    prefix = NameMaintainer.get_time_ticket_tag_prefix(task)
    task.get_tmp_data()[f"{prefix}:total_end_time"] = 16.0
    task.set_flow_index(TaskConstant.END.value)

    assert task.get_real_end_to_end_time() == pytest.approx(6.0)
    assert task.calculate_total_time() == pytest.approx(1.5)
    assert task.calculate_cloud_edge_transmit_time() == pytest.approx(0.5)
    assert "total delay:1.5000s" in task.get_delay_info()
    assert "real end-to-end delay:6.0000s" in task.get_delay_info()

    del task.get_tmp_data()[f"{prefix}:total_end_time"]
    with pytest.raises(ValueError, match="ending lacks"):
        task.get_real_end_to_end_time()


@pytest.mark.unit
def test_node_and_dag_cover_error_branches_and_structural_helpers():
    node = dag_module.Node(Service("decode"))
    node.add_prev_node(Service(TaskConstant.START.value))
    node.add_next_node(Service("infer"))

    restored = dag_module.Node.deserialize(node.serialize())
    assert restored.to_dict() == node.to_dict()
    assert repr(restored) == "decode"

    dag = DAG()
    dag.add_node(Service("decode", execute_device="edge-a"))
    dag.set_node_service("decode", Service("decode", execute_device="cloud-a"))
    assert dag.get_node("decode").service.get_execute_device() == "cloud-a"

    with pytest.raises(ValueError, match='already exists'):
        dag.add_node(Service("decode"))
    with pytest.raises(TypeError, match="Expected input type"):
        dag.add_node("decode")
    with pytest.raises(KeyError, match="does not exist"):
        dag.get_node("missing")
    with pytest.raises(KeyError, match="does not exist"):
        dag.get_next_nodes("missing")
    with pytest.raises(KeyError, match="does not exist"):
        dag.get_prev_nodes("missing")
    with pytest.raises(ValueError, match="Start node"):
        dag.get_start_node()
    with pytest.raises(ValueError, match="End node"):
        dag.get_end_node()

    start = TaskConstant.START.value
    end = TaskConstant.END.value
    pipeline_like = DAG.from_dict(
        {
            start: service_entry(start, next_nodes=["decode"]),
            "decode": service_entry("decode", next_nodes=[end]),
            end: service_entry(end, prev_nodes=["decode"]),
        }
    )
    assert "decode -> [_end]" in repr(pipeline_like)

    with pytest.raises(ValueError, match='already exists'):
        pipeline_like.add_start_node(Service(start))
    with pytest.raises(ValueError, match='already exists'):
        pipeline_like.add_end_node(Service(end))

    invalid_child = DAG.from_dict(
        {
            start: service_entry(start, next_nodes=["missing"]),
            end: service_entry(end),
        }
    )
    with pytest.raises(ValueError, match="child node doesn't exist"):
        invalid_child._check_edge_consistency()

    invalid_parent = DAG.from_dict(
        {
            start: service_entry(start),
            "decode": service_entry("decode", prev_nodes=["ghost"]),
            end: service_entry(end),
        }
    )
    with pytest.raises(ValueError, match="parent node doesn't exist"):
        invalid_parent._check_edge_consistency()
