"""Regression coverage for the STEADY/UNSTEADY integration contracts."""

import ast
import copy
import importlib
import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from core.lib.algorithms.scenario_extraction.object_velocity_extraction import ObjectVelocityExtraction, Tracking
from core.lib.algorithms.schedule_agent.steady_agent.accuracy_calculation import AccuracyCalculation
from core.lib.algorithms.schedule_agent.steady_agent.accuracy_prediction import (
    AccuracyPrediction2fps,
    AccuracyPrediction2reso,
)
from core.lib.algorithms.schedule_agent.steady_agent.configuration import (
    constraint_combinations,
    constraint_values,
    normalize_parameters,
)
from core.lib.algorithms.schedule_agent.steady_agent.context_cluster import ContextCluster
from core.lib.algorithms.schedule_agent.steady_agent.context_record import ContextRecord
from core.lib.algorithms.schedule_agent.steady_agent.hook import SteadyAgent
from core.lib.algorithms.schedule_agent.steady_agent.integrated_safe_predictor import IntegratedSafePredictor
from core.lib.algorithms.schedule_agent.steady_agent.overall_scheduler import MicroFeedback
from core.lib.algorithms.schedule_agent.unsteady_agent.hook import UnsteadyAgent
from core.lib.common import ClassFactory, ClassType, Context, FileOps
from core.lib.content import Task
from core.lib.scheduling.pipeline import apply_pipeline_partition

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]
SERVICES = ["car-detection", "vehicle-classification"]
CONTEXT = {"band_Mbps": 2.1, "obj_size_norm": 0.1, "obj_num": 7.5, "obj_speed": 650}
RANGES = {
    "fps": [5, 15, 30],
    "resolution": ["360p", "720p", "1080p"],
    "buffer_size": [2, 4],
    "edge_serv_num": [0, 1, 2],
}


def template_parameters(name="steady"):
    template = yaml.safe_load((ROOT / "template" / "scheduler" / f"{name}.yaml").read_text())
    env = {item["name"]: item["value"] for item in template["pod-template"]["env"]}
    return env, ast.literal_eval(env["SCH_AGENT_PARAMETERS"])


def pipeline():
    names = ["_start", *SERVICES, "_end"]
    return {
        name: {
            "service": {"service_name": name, "execute_device": "edge-a" if index == 0 else "cloud-a"},
            "prev_nodes": names[index - 1 : index] if index else [],
            "next_nodes": names[index + 1 : index + 2],
            "custom_options": {"retain": name},
        }
        for index, name in enumerate(names)
    }


def task(task_id=1, resolution="360p", fps=15, partition=1):
    dag = apply_pipeline_partition(pipeline(), partition, "edge-a", "cloud-a")
    result = Task(
        source_id=7,
        task_id=task_id,
        source_device="edge-a",
        all_edge_devices=["edge-a"],
        dag=Task.extract_dag_from_dict(dag),
        file_path="frames.mp4",
        runtime_directory_revision=7,
        metadata={"resolution": resolution, "fps": fps, "buffer_size": 2, "encoding": "mp4v"},
        raw_metadata={"resolution": "1080p", "fps": 30},
        temp={"file_size": 0.1},
    )
    for name in SERVICES:
        service = result.get_service(name)
        service.set_execute_time(0.4)
        service.set_real_execute_time(0.2)
        service.set_transmit_time(0.1)
    result.get_service(SERVICES[0]).set_scenario_data({"obj_num": [7, 8], "obj_size": [0.1, 0.1], "obj_velocity": 650})
    size = [192, 108] if resolution == "1080p" else [64, 36]
    result.get_service(SERVICES[0]).set_content_data(
        {
            "service": SERVICES[0],
            "outputs": {
                "bbox": [
                    {"frame_index": index, "items": [{"bbox": [0, 0, *size], "score": 0.9, "label": "car"}]}
                    for index in range(2)
                ]
            },
            "profile": {"frame_count": 2},
        }
    )
    return result


@pytest.fixture
def profiles(tmp_path):
    for stage, name in enumerate(SERVICES):
        data = {}
        for index, resolution in enumerate(RANGES["resolution"], 1):
            for role, factor in (("edge", 3), ("cloud", 1)):
                data[f"execute_role={role}#resolution={resolution}"] = factor * index * 0.01 / (stage + 1)
        (tmp_path / f"{name}.json").write_text(json.dumps(data))
    sizes = {
        f"resolution={resolution}#fps={fps}#encoding=mp4v#buffer_size={size}": 0.001 * index * size
        for index, resolution in enumerate(RANGES["resolution"], 1)
        for fps in RANGES["fps"]
        for size in RANGES["buffer_size"]
    }
    (tmp_path / "file_size.json").write_text(json.dumps(sizes))
    return tmp_path


@pytest.fixture
def make_agent(monkeypatch, profiles):
    agents = []
    monkeypatch.setattr(Context, "get_file_path", staticmethod(lambda value: str(value)))

    def build(mode="steady", parameter_updates=None, sch_updates=None):
        name = "steady" if mode == "steady" else f"unsteady-{mode}"
        env, parameters = template_parameters(name)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        key = "steady_param" if mode == "steady" else "unsteady_param"
        params = parameters[key]
        params.update(kb_path=str(profiles), if_online_train=False, macro_update_interval=600)
        params.pop("steady_record_path", None)
        params.pop("correct_record_path", None)
        params.update(parameter_updates or {})
        sch_params = {"if_stop_record_in_single_cycle": False, **(sch_updates or {})}
        snapshot = {"runtime_directory_revision": 7, "deployment": {name: ["edge-a", "cloud-a"] for name in SERVICES}}
        system = SimpleNamespace(
            cloud_device="cloud-a",
            fps_list=RANGES["fps"],
            resolution_list=RANGES["resolution"],
            buffer_size_list=RANGES["buffer_size"],
            get_scheduling_snapshot=lambda **kwargs: snapshot,
        )
        agent_class = ClassFactory.get_cls(ClassType.SCH_AGENT, env["SCH_AGENT_NAME"])
        agent = agent_class(system, 7, sch_params, **{key: params})
        agents.append(agent)
        return agent, snapshot, params

    yield build
    for agent in agents:
        agent.stop()


def request():
    return {"source_id": 7, "source_device": "edge-a", "dag": pipeline(), "meta_data": {"fps": 30}}


@pytest.mark.parametrize("mode", ["steady", "macro", "micro"])
def test_registered_templates_preserve_dag_and_require_active_targets(make_agent, mode):
    agent, snapshot, parameters = make_agent(mode)
    before = copy.deepcopy(parameters)
    info = request()
    original = copy.deepcopy(info)
    plan = agent.get_schedule_plan(info)
    assert plan["fps"] == 30 and plan["buffer_size"] == 4
    assert info == original and parameters == before
    assert plan["dag"]["_start"]["service"]["execute_device"] == "edge-a"
    assert plan["dag"]["_end"]["service"]["execute_device"] == "cloud-a"
    for name in plan["dag"]:
        assert plan["dag"][name]["custom_options"] == original["dag"][name]["custom_options"]
    snapshot["deployment"][SERVICES[0]] = ["edge-a"]
    with pytest.raises(ValueError, match="not active"):
        agent.get_schedule_plan(info)
    assert agent.should_generate(info)["generate"] is True


@pytest.mark.parametrize("mode", ["steady", "macro", "micro"])
def test_reference_sampling_and_structured_feedback(make_agent, mode):
    agent, _, _ = make_agent(mode, {"acc_sample_interval": 2})
    agent.get_schedule_plan(request())
    sample = agent.get_schedule_plan(request())
    assert {key: sample[key] for key in ("fps", "resolution", "buffer_size")} == {
        "fps": 30,
        "resolution": "1080p",
        "buffer_size": 2,
    }
    agent.update_resource("edge-a", {"available_bandwidth": 2.1})
    measured = task()
    agent.update_task(measured)
    assert agent.get_conf_info_from_task(measured)["edge_serv_num"] == 1
    assert agent.get_context_info_from_task(measured) == CONTEXT
    assert agent.get_delay_from_task(measured) == pytest.approx(0.225)
    feedback = agent.get_task_info_from_task(measured)
    assert feedback["real_exe_detect"] == pytest.approx(0.1)
    assert feedback["classify_wait_delay"] == pytest.approx(0.1)
    assert feedback["real_trans"] == pytest.approx(0.05)
    reference = task(2, "1080p", 30, 0)
    configuration, observation = agent.get_conf_info_task_info_from_task(reference)
    assert configuration["resolution"] == "360p"
    assert observation["real_acc_reso"] == pytest.approx(1.0)
    reference.get_service(SERVICES[0]).set_content_data({"outputs": {"bbox": []}})
    assert AccuracyCalculation.get_real_acc(measured, reference) is None
    measured.get_service(SERVICES[0]).set_scenario_data({})
    assert agent.get_context_info_from_task(measured) is None
    measured.get_service(SERVICES[0]).set_scenario_data(None)
    assert agent.get_context_info_from_task(measured) is None


def test_rejects_unsupported_topology_and_source_changes(make_agent):
    agent, _, _ = make_agent()
    bad = request()
    bad["dag"]["_start"]["next_nodes"].append(SERVICES[1])
    with pytest.raises(ValueError, match="pipeline"):
        agent.get_schedule_plan(bad)
    agent.get_schedule_plan(request())
    changed = request()
    changed["source_device"] = "edge-b"
    with pytest.raises(ValueError, match="source device"):
        agent.get_schedule_plan(changed)


def test_constraint_sweep_advances_and_old_records_remain_readable(make_agent, tmp_path):
    settings = {
        "all_delay_cons_info": [{"value": 0.3, "adjust": 0.7}, {"value": 0.4, "adjust": 0.5}],
        "all_acc_cons_info": [{"value": 0.6, "adjust": 1.2}],
        "all_delay_weight_info": [{"value": 0.8, "adjust": 1}],
        "unit_logic_frame_num_max": 4,
    }
    agent, _, _ = make_agent(
        parameter_updates=settings,
        sch_updates={"if_stop_record_in_single_cycle": True, "record_path": str(tmp_path / "nested/context")},
    )
    agent.get_schedule_plan(request())
    measured = task()
    agent.update_task(measured)
    assert agent.cons_info_comb_idx == 1
    assert agent.overall_scheduler.macro_search.knowledge_base.delay_cons == pytest.approx(0.2)
    assert agent._constraints()["delay_cons"] == 0.4
    agent.update_task(task(2))
    assert agent.if_keep_record is False
    assert agent.get_schedule_plan(request())["dag"]
    path = next((tmp_path / "nested").iterdir())
    records = ContextRecord.read_record(path)
    assert [record.get_cons_table()["delay_cons"] for record in records] == [0.3, 0.4]
    legacy = json.loads(ContextRecord.serialize(records[0]))
    legacy.pop("cons_table")
    assert ContextRecord.deserialize(json.dumps(legacy)).get_cons_table() == {}


def test_single_constraint_legacy_spellings_do_not_mutate_parameters(make_agent):
    agent, _, params = make_agent(parameter_updates={"history_lenghth": 5, "context_anylze_type": 0})
    # Canonical keys take precedence when both are supplied.
    assert agent.init_param["history_length"] == 3
    assert params["history_lenghth"] == 5
    legacy = copy.deepcopy(params)
    del legacy["history_length"], legacy["context_analyze_type"], legacy["if_online_train"]
    normalized = normalize_parameters(legacy)
    assert normalized["history_length"] == 5
    assert normalized["context_analyze_type"] == 0
    assert normalized["if_online_train"] is True
    combinations = constraint_combinations(
        {"delay_cons": 0.3, "acc_cons": 0.6, "delay_weight": 0.8, "delay_cons_adjust": 0.7}
    )
    assert len(combinations) == 1
    assert constraint_values(combinations[0], adjusted=True)["delay_cons"] == pytest.approx(0.21)
    for name, value in (("delay_cons", 0), ("acc_cons", -1), ("delay_weight", 1.1)):
        with pytest.raises(ValueError):
            constraint_combinations({"delay_cons": 0.3, "acc_cons": 0.6, "delay_weight": 0.8, name: value})
    with pytest.raises(ValueError, match="nonempty"):
        constraint_combinations({"all_delay_weight_info": []})


@pytest.mark.parametrize(
    "family, class_name",
    [("adamec", "AdaMECPolicySearch"), ("gecko", "GeckoPolicySearch"), ("madeye", "MadEyePolicySearch")],
)
def test_existing_baselines_construct_real_predictor_without_new_parameters(profiles, family, class_name):
    _, settings = template_parameters()
    params = settings["steady_param"]
    corrector = copy.deepcopy(params["corrector_param"])
    corrector.pop("corrector_pool_threshold")
    module = importlib.import_module(f"core.lib.algorithms.schedule_agent.{family}_agent.policy_search")
    search = getattr(module, class_name)(
        kb_path=str(profiles),
        service_name_pipeline=SERVICES,
        knob_value_range_dict=RANGES,
        delay_cons=0.3,
        acc_cons=0.6,
        delay_weight=0.8,
        acc_weight=0.2,
        default_policy=params["default_policy"],
        raw_meta_data={"fps": 30},
        corrector_param=corrector,
        queue_param=params["queue_param"],
    )
    policy = search.get_schedule_plan(1, params["default_policy"], CONTEXT)
    assert all(policy[name] in values for name, values in RANGES.items())
    assert search.performance_predictor.corrected_predictor.context_cluster.cluster_threshold == 0


def test_role_profiles_correction_pool_and_recording(profiles, tmp_path):
    _, settings = template_parameters()
    params = settings["steady_param"]
    predictor = IntegratedSafePredictor(str(profiles), SERVICES, params["corrector_param"], params["queue_param"], 0.5)
    policy = {"fps": 15, "resolution": "720p", "buffer_size": 2, "edge_serv_num": 0}
    expected_transmission = math.exp(3.5905589868141545) * (0.004 / 2.1) ** 1.372854018944592 / 2
    path = tmp_path / "nested/prediction.jsonl"
    delay = predictor.delay_pre(CONTEXT, policy, 2, False, str(path))
    assert delay == pytest.approx(0.02 + 0.01 * 7.5 + expected_transmission)
    assert json.loads(path.read_text())["task_id_for_pre"] == 2
    for task_id in range(1, 4):
        predictor.update_corrector(
            CONTEXT,
            policy,
            {
                "task_id": task_id,
                "real_exe_detect": 0.04,
                "real_exe_classify": 0.15,
                "real_trans": expected_transmission,
                "real_acc_reso": 0.8,
            },
        )
    assert "2222" in predictor.corrected_predictor.corrector_pool
    assert math.isfinite(predictor.delay_pre(CONTEXT, policy, 4, True))
    assert predictor.acc_pre(CONTEXT, policy, True) > 0
    # A query receives its own window copy, never mutable shared pool state.
    windows = predictor.corrected_predictor.get_all_coeff_window_by_context(CONTEXT)
    windows["detect_coeff_window"][0].clear()
    assert predictor.corrected_predictor.corrector_pool["2222"]["detect_coeff_window"][0]


@pytest.mark.parametrize(
    "context, expected",
    [
        (CONTEXT, ("2222", {"band_Mbps": 1, "obj_size_norm": 0.1, "obj_num": 10, "obj_speed": 780}, 1)),
        (
            {"band_Mbps": 0.09, "obj_size_norm": 0.04, "obj_num": 0, "obj_speed": 0},
            ("0000", {"band_Mbps": 0, "obj_size_norm": 0.01, "obj_num": 1, "obj_speed": 260}, 1),
        ),
        (
            {**CONTEXT, "band_Mbps": 3.01},
            ("2222", {"band_Mbps": 1, "obj_size_norm": 0.1, "obj_num": 10, "obj_speed": 780}, 0),
        ),
        (None, (None, None, None)),
    ],
)
def test_context_cluster_preserves_research_bin_boundaries(context, expected):
    assert ContextCluster(0.5).process_context_for_cluster(context) == expected
    assert ContextCluster(0).process_context_for_cluster(context) == (None, None, None)


def test_accuracy_coefficients_and_feedback_step_boundaries():
    fps = AccuracyPrediction2fps()
    resolution = AccuracyPrediction2reso()
    config = {"fps": 5, "resolution": "720p"}
    assert fps.predict("car-detection", config, obj_speed=520) == pytest.approx(0.97 - 1.20 * math.exp(-0.78 * 5))
    assert resolution.predict("car-detection", config, obj_size=100000) == pytest.approx(
        0.99 - 0.47 * math.exp(-0.008 * 370)
    )
    _, parameters = template_parameters()
    feedback = MicroFeedback(parameters["steady_param"]["coeff_info"], RANGES)
    feedback.update_step_coeff(False, False)
    assert feedback.coeff_info["step_coeff"]["cur_value"] == 15
    feedback.update_step_coeff(False, True)
    assert feedback.coeff_info["step_coeff"]["cur_value"] == 10
    feedback.update_delay_loss_and_acc_loss(1, 0)
    feedback.update_all_knob_weight({key: -1 for key in RANGES}, {key: 1 for key in RANGES})
    policy = {"fps": 30, "resolution": "1080p", "buffer_size": 4, "edge_serv_num": 2}
    assert feedback.get_new_policy_by_feedback(policy) == {key: values[0] for key, values in RANGES.items()}
    assert policy["fps"] == 30


@pytest.mark.parametrize("mode", ["steady", "macro", "micro"])
def test_modes_keep_macro_choice_feedback_and_steady_hold_rule(make_agent, mode):
    agent, _, _ = make_agent(mode)
    agent.get_schedule_plan(request())
    scheduler = agent.overall_scheduler
    current = {"fps": 30, "resolution": "720p", "buffer_size": 4, "edge_serv_num": 1}
    macro_policy = {**current, "fps": 5}
    scheduler._macro_output = {
        "if_need_macro_search": True,
        "path_record": [{"policy": macro_policy, "loss": 0}],
        "context_history": {},
        "macro_search_delay": 0,
        "chosen_knob_list": list(RANGES),
        "delay_decrease_weight": {key: -1 for key in RANGES},
        "acc_increase_weight": {key: 1 for key in RANGES},
    }
    policy = scheduler.get_schedule_plan(1, current, CONTEXT, 0.21, 0.72)
    assert policy == (macro_policy if mode == "macro" else current)
    overloaded = scheduler.get_schedule_plan(2, current, CONTEXT, 2, 0.72)
    assert overloaded["fps"] == 5
    assert scheduler._thread is None if mode == "micro" else scheduler._thread.is_alive()
    agent.stop()
    assert scheduler._thread is None or not scheduler._thread.is_alive()


def test_macro_worker_and_training_worker_stop_cleanly(make_agent):
    agent, _, _ = make_agent(parameter_updates={"macro_update_interval": 0.001, "if_online_train": True})
    agent.get_schedule_plan(request())
    scheduler = agent.overall_scheduler
    knowledge = scheduler.macro_search.knowledge_base
    completed = threading.Event()
    original = scheduler._compute_macro

    def compute(*args):
        output = original(*args)
        completed.set()
        return output

    scheduler._compute_macro = compute
    agent.update_resource("edge-a", {"available_bandwidth": 2.1})
    agent.update_task(task())
    agent.get_schedule_plan(request())
    assert completed.wait(5), "macro search did not produce a decision"
    agent.stop()
    assert not scheduler._thread.is_alive()
    assert not knowledge._thread.is_alive()


def test_velocity_reads_temporary_file_releases_capture_and_handles_empty_features(monkeypatch, mounted_runtime):
    module = importlib.import_module("core.lib.algorithms.scenario_extraction.object_velocity_extraction")
    measured = task()
    frames = iter([(True, np.zeros((32, 32, 3), dtype=np.uint8)), (False, None)])
    paths, released = [], []

    class Capture:
        def __init__(self, path):
            paths.append(path)

        def read(self):
            return next(frames)

        def release(self):
            released.append(True)

    monkeypatch.setattr(module.cv2, "VideoCapture", Capture)
    extractor = ObjectVelocityExtraction()
    try:
        assert extractor(measured.get_first_content(), measured) == 0
    finally:
        extractor.stop()
    assert paths == [FileOps.get_task_file_in_temp(measured)]
    assert paths[0] != measured.get_file_path()
    assert released == [True]
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    assert Tracking().track_bbox(frame, [0, 0, 16, 16], frame)[0] is False
    assert Tracking().track_bbox(frame, [-10, -10, -1, -1], frame)[0] is False
