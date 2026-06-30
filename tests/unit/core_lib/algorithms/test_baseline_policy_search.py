from pathlib import Path
import importlib.util


SCHEDULE_AGENT_DIR = (
    Path(__file__).resolve().parents[4] / "dependency" / "core" / "lib" / "algorithms" / "schedule_agent"
)


def load_policy_search(module_name, relative_path, class_name):
    spec = importlib.util.spec_from_file_location(module_name, SCHEDULE_AGENT_DIR / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


AdaMECPolicySearch = load_policy_search("adamec_policy_search", "adamec/policy_search.py", "AdaMECPolicySearch")
GeckoPolicySearch = load_policy_search("gecko_policy_search", "gecko/policy_search.py", "GeckoPolicySearch")
MadEyePolicySearch = load_policy_search("madeye_policy_search", "madeye/policy_search.py", "MadEyePolicySearch")


class FakePredictor:
    def delay_pre(self, context_info, conf_info, latest_task_id, if_correct, record_path=None):
        return conf_info["buffer_size"] * 0.1

    def acc_pre(self, context_info, conf_info, if_correct):
        return 1.0

    def update_corrector(self, context_info, conf_info, task_info):
        pass


def build_adamec_search(adamec_param=None):
    return AdaMECPolicySearch(
        kb_path="unused",
        service_name_pipeline=["detector", "classifier"],
        knob_value_range_dict={
            "fps": [1, 5, 10, 20],
            "resolution": ["360p", "720p"],
            "buffer_size": [1, 2, 4],
            "edge_serv_num": [0, 1, 2],
        },
        delay_cons=0.15,
        acc_cons=0.5,
        delay_weight=1.0,
        acc_weight=1.0,
        default_policy={
            "fps": 10,
            "resolution": "720p",
            "buffer_size": 4,
            "edge_serv_num": 0,
        },
        raw_meta_data={"fps": 20},
        corrector_param={},
        queue_param={},
        search_param={"max_iterations": 8, "max_expanded_states": 64},
        adamec_param=adamec_param or {},
        performance_predictor=FakePredictor(),
    )


def build_gecko_search(gecko_param=None):
    return GeckoPolicySearch(
        kb_path="unused",
        service_name_pipeline=["detector", "classifier"],
        knob_value_range_dict={
            "fps": [1, 5, 10, 20],
            "resolution": ["360p", "720p"],
            "buffer_size": [1, 2, 4],
            "edge_serv_num": [0, 1, 2],
        },
        delay_cons=0.15,
        acc_cons=0.5,
        delay_weight=1.0,
        acc_weight=1.0,
        default_policy={
            "fps": 10,
            "resolution": "720p",
            "buffer_size": 4,
            "edge_serv_num": 0,
        },
        raw_meta_data={"fps": 20},
        corrector_param={},
        queue_param={},
        search_param={"max_iterations": 8, "max_expanded_states": 64},
        gecko_param=gecko_param or {},
        performance_predictor=FakePredictor(),
    )


def build_madeye_search(madeye_param=None):
    return MadEyePolicySearch(
        kb_path="unused",
        service_name_pipeline=["detector", "classifier"],
        knob_value_range_dict={
            "fps": [1, 5, 10, 20],
            "resolution": ["360p", "720p"],
            "buffer_size": [1, 2, 4],
            "edge_serv_num": [0, 1, 2],
        },
        delay_cons=0.15,
        acc_cons=0.5,
        delay_weight=1.0,
        acc_weight=1.0,
        default_policy={
            "fps": 10,
            "resolution": "720p",
            "buffer_size": 4,
            "edge_serv_num": 0,
        },
        raw_meta_data={"fps": 20},
        corrector_param={},
        queue_param={},
        search_param={"max_iterations": 8, "max_expanded_states": 64},
        madeye_param=madeye_param or {"candidate_pool_size": 4},
        performance_predictor=FakePredictor(),
    )


def test_madeye_keeps_candidate_pool_and_returns_clean_policy():
    search = build_madeye_search()
    current_policy = {
        "fps": 10,
        "resolution": "720p",
        "buffer_size": 4,
        "edge_serv_num": 0,
    }
    context_info = {"band_Mbps": 10, "obj_num": 1, "obj_size_norm": 0.1, "obj_speed": 0.5}
    policy = search.get_schedule_plan(
        cur_task_id=1,
        cur_policy=current_policy,
        context_info=context_info,
    )

    assert search.candidate_pool
    assert search.evaluate_loss(policy, context_info) <= search.evaluate_loss(current_policy, context_info)
    assert "loss" not in policy


def test_adamec_improves_from_current_policy_by_local_graph_search():
    search = build_adamec_search()
    current_policy = {
        "fps": 10,
        "resolution": "720p",
        "buffer_size": 4,
        "edge_serv_num": 0,
    }
    context_info = {"band_Mbps": 10, "obj_num": 1, "obj_size_norm": 0.1, "obj_speed": 0.5}
    policy = search.get_schedule_plan(
        cur_task_id=1,
        cur_policy=current_policy,
        context_info=context_info,
    )

    assert search.evaluate_loss(policy, context_info) < search.evaluate_loss(current_policy, context_info)
    assert "loss" not in policy


def test_gecko_applies_aimd_before_searching_remaining_knobs():
    search = build_gecko_search({"speed_threshold": 1.0, "slowdown_factor": 0.5})
    policy = search.get_schedule_plan(
        cur_task_id=1,
        cur_policy={
            "fps": 10,
            "resolution": "720p",
            "buffer_size": 4,
            "edge_serv_num": 0,
        },
        context_info={"band_Mbps": 10, "obj_num": 0, "obj_size_norm": 0.0, "obj_speed": 0.2},
    )

    assert policy["fps"] == 5
    assert policy["resolution"] == "360p"
    assert "loss" not in policy
