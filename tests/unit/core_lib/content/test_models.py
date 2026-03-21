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
    restored.set_service_name("detector-v2")
    assert restored.get_service_name() == "detector-v2"
    assert hash(restored) == hash("detector-v2")
    assert repr(restored) == "detector-v2"


@pytest.mark.unit
def test_service_from_dict_supports_partial_payloads_and_non_service_equality():
    service = Service.from_dict(
        {
            "service_name": "tracker",
            "execute_data": {"execute_time": 1.5},
        }
    )

    assert service.get_service_name() == "tracker"
    assert service.get_execute_time() == 1.5
    assert service.get_execute_device() == ""
    assert service.get_transmit_time() == 0
    assert service.get_real_execute_time() == 0
    assert service.get_content_data() is None
    assert service.get_scenario_data() == {}
    assert service.get_tmp_data() == {}
    assert (service == "tracker") is False


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
def test_dag_helpers_cover_node_access_auto_connections_and_duplicate_edges():
    start = TaskConstant.START.value
    end = TaskConstant.END.value

    node = dag_module.Node(Service("decode"), prev_nodes=[start], next_nodes=["infer"])
    assert node.get_prev_nodes() == [start]
    assert node.get_next_nodes() == ["infer"]
    assert dag_module.Node.deserialize(node.serialize()).to_dict() == node.to_dict()

    dag = DAG()
    dag.add_edge(Service("decode"), Service("infer"))
    dag.add_start_node(Service(start))
    dag.add_end_node(Service(end))

    assert dag.get_start_node().service.get_service_name() == start
    assert dag.get_end_node().service.get_service_name() == end
    assert dag.get_prev_nodes("decode") == [start]
    assert dag.get_next_nodes("infer") == [end]

    with pytest.raises(ValueError, match='already exists in DAG'):
        dag.add_node(dag_module.Node(Service("decode")))
    with pytest.raises(ValueError, match='already exists'):
        dag.add_start_node(Service(start))
    with pytest.raises(ValueError, match='already exists'):
        dag.add_end_node(Service(end))
    with pytest.raises(TypeError, match="Expected input type Service or Node"):
        dag.add_node("invalid")

    boundaryless = DAG()
    with pytest.raises(ValueError, match='Start node'):
        boundaryless.get_start_node()
    with pytest.raises(ValueError, match='End node'):
        boundaryless.get_end_node()

    duplicate = DAG.from_dict(
        {
            start: service_entry(start, next_nodes=["decode"]),
            "decode": service_entry("decode", prev_nodes=[start], next_nodes=[end]),
            end: service_entry(end, prev_nodes=["decode"]),
        }
    )
    duplicate.add_edge(Service(start), Service("decode"))
    with pytest.raises(ValueError, match="Duplicate edge"):
        duplicate.validate_dag()


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
def test_task_service_and_content_helpers_cover_stage_updates_and_time_ticket_operations():
    pipeline = [
        {"service_name": "decode", "execute_device": "edge-a"},
        {"service_name": "infer", "execute_device": "cloud-a"},
    ]
    pipeline_task = Task(
        source_id=8,
        task_id=9,
        source_device="edge-a",
        all_edge_devices=["edge-a"],
        dag=Task.extract_dag_from_pipeline_deployment(pipeline),
    )
    assert pipeline_task.get_pipeline_deployment_info() == pipeline
    assert Task.extract_dict_from_dag(pipeline_task.get_dag()) == pipeline_task.get_dag().to_dict()

    task = build_branching_task()
    branches = {branch.get_flow_index(): branch for branch in task.step_to_next_stage()}
    branch = branches["a"]

    branch.add_hash_data("hash-1")
    branch.set_current_content({"boxes": 2})
    branch.set_scenario_data({"objects": 2})
    branch.add_scenario({"speed": 3})
    branch.save_transmit_time(0.2)
    branch.save_execute_time(0.4)
    branch.save_real_execute_time(0.3)
    branch.record_time_ticket_in_service("queue", is_end=False, time_ticket=1.0)
    branch.record_time_ticket_in_service("queue", is_end=True, time_ticket=1.2)

    assert branch.get_current_content() == {"boxes": 2}
    assert branch.get_current_service_info() == ("a", "edge-a")
    assert branch.get_current_stage_device() == "edge-a"

    branch.set_current_stage_device("edge-c")
    assert branch.get_current_stage_device() == "edge-c"

    assert branch.get_scenario_data("a") == {"objects": 2, "speed": 3}
    assert branch.get_first_scenario_data() == {"objects": 2, "speed": 3}
    assert branch.get_first_content() == {"boxes": 2}

    join_task = branch.step_to_next_stage()[0]
    assert join_task.get_prev_content() == {"boxes": 2}

    join_task.set_current_content({"merged": True})
    join_task.set_flow_index(TaskConstant.END.value)
    assert join_task.get_last_content() == {"merged": True}
    assert join_task.calculate_cloud_edge_transmit_time() == pytest.approx(0.2)

    updated_dag = Task.set_execute_device(join_task.get_dag(), "uniform-device")
    assert all(
        updated_dag.get_node(node_name).service.get_execute_device() == "uniform-device"
        for node_name in updated_dag.nodes
    )
    join_task.set_initial_execute_device("initial-device")
    assert all(
        join_task.get_dag().get_node(node_name).service.get_execute_device() == "initial-device"
        for node_name in join_task.get_dag().nodes
    )
    assert branch.get_hash_data() == ["hash-1"]

    join_task.record_time_ticket_in_service("queue", is_end=False, time_ticket=2.0)
    join_task.erase_time_ticket_in_service("queue", is_end=False)
    with pytest.raises(AssertionError, match="does not exist"):
        join_task.erase_time_ticket_in_service("queue", is_end=False)

    missing_start_task = Task(
        source_id=5,
        task_id=6,
        source_device="edge-a",
        all_edge_devices=["edge-a"],
        dag=Task.extract_dag_from_dict({"decode": service_entry("decode", execute_device="edge-a")}),
    )
    with pytest.raises(ValueError, match="starting lacks"):
        missing_start_task.get_real_end_to_end_time()

    cloned = join_task.fork_task(join_task.get_flow_index())
    assert cloned.get_flow_index() == join_task.get_flow_index()
    assert cloned.get_parent_uuid() == join_task.get_task_uuid()


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


@pytest.mark.unit
def test_dag_private_validators_cover_pipeline_edge_cases_and_manual_mismatch_guards():
    start = TaskConstant.START.value
    end = TaskConstant.END.value

    revisit_cycle = DAG.from_dict(
        {
            start: service_entry(start, next_nodes=["a"]),
            "a": service_entry("a", prev_nodes=[start], next_nodes=["a"]),
            end: service_entry(end),
        }
    )
    assert revisit_cycle.check_is_pipeline() is False

    broken_prev_link = DAG.from_dict(
        {
            start: service_entry(start, next_nodes=["a"]),
            "a": service_entry("a", prev_nodes=[], next_nodes=[end]),
            end: service_entry(end, prev_nodes=["a"]),
        }
    )
    assert broken_prev_link.check_is_pipeline() is False

    broken_end = DAG.from_dict(
        {
            start: service_entry(start, next_nodes=["a"]),
            "a": service_entry("a", prev_nodes=[start], next_nodes=[end]),
            end: service_entry(end, prev_nodes=["a"], next_nodes=["tail"]),
            "tail": service_entry("tail", prev_nodes=[end]),
        }
    )
    assert broken_end.check_is_pipeline() is False

    with pytest.raises(ValueError, match="Start node"):
        DAG()._check_start_end_node()

    with pytest.raises(ValueError, match="End node"):
        DAG.from_dict({start: service_entry(start)})._check_start_end_node()

    empty_dag = DAG()
    empty_dag._check_connectivity()

    class NonAppendingList(list):
        def append(self, value):
            return None

    mismatch_child = DAG.from_dict(
        {
            start: service_entry(start, next_nodes=["decode"]),
            "decode": service_entry("decode", prev_nodes=[], next_nodes=[end]),
            end: service_entry(end, prev_nodes=["decode"]),
        }
    )
    mismatch_child.get_node("decode").prev_nodes = NonAppendingList()
    with pytest.raises(ValueError, match="doesn't list"):
        mismatch_child._check_edge_consistency()

    mismatch_parent = DAG.from_dict(
        {
            start: service_entry(start, next_nodes=[]),
            "decode": service_entry("decode", prev_nodes=[start], next_nodes=[end]),
            end: service_entry(end, prev_nodes=["decode"]),
        }
    )
    mismatch_parent.get_node(start).next_nodes = NonAppendingList()
    with pytest.raises(ValueError, match="doesn't list"):
        mismatch_parent._check_edge_consistency()
