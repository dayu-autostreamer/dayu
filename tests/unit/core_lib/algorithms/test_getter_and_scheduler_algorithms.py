import copy
import importlib
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.lib.common import TaskConstant
from core.lib.content import Task


data_getter_base_module = importlib.import_module("core.lib.algorithms.data_getter.base_getter")
http_getter_module = importlib.import_module("core.lib.algorithms.data_getter.http_video_getter")
rtsp_getter_module = importlib.import_module("core.lib.algorithms.data_getter.rtsp_video_getter")
base_config_module = importlib.import_module("core.lib.algorithms.schedule_config_extraction.base_config_extraction")
config_extraction_module = importlib.import_module("core.lib.algorithms.schedule_config_extraction")
config_simple_module = importlib.import_module("core.lib.algorithms.schedule_config_extraction.simple_config_extraction")
config_fc_module = importlib.import_module("core.lib.algorithms.schedule_config_extraction.fc_config_extraction")
config_chameleon_module = importlib.import_module("core.lib.algorithms.schedule_config_extraction.chameleon_config_extraction")
config_casva_module = importlib.import_module("core.lib.algorithms.schedule_config_extraction.casva_config_extraction")
config_hedger_module = importlib.import_module("core.lib.algorithms.schedule_config_extraction.hedger_config_extraction")
config_hei_module = importlib.import_module("core.lib.algorithms.schedule_config_extraction.hei_config_extraction")
config_hei_drl_module = importlib.import_module("core.lib.algorithms.schedule_config_extraction.hei_drl_config_extraction")
base_policy_retrieval_module = importlib.import_module(
    "core.lib.algorithms.schedule_policy_retrieval.base_policy_extraction"
)
policy_retrieval_module = importlib.import_module("core.lib.algorithms.schedule_policy_retrieval")
base_scenario_module = importlib.import_module(
    "core.lib.algorithms.schedule_scenario_retrieval.base_scenario_retrieval"
)
scenario_retrieval_module = importlib.import_module("core.lib.algorithms.schedule_scenario_retrieval")
selection_base_module = importlib.import_module("core.lib.algorithms.schedule_selection_policy.base_selection_policy")
selection_policy_module = importlib.import_module("core.lib.algorithms.schedule_selection_policy")
random_selection_module = importlib.import_module("core.lib.algorithms.schedule_selection_policy.random_selection_policy")
base_startup_module = importlib.import_module("core.lib.algorithms.schedule_startup_policy.base_startup_policy")
startup_policy_module = importlib.import_module("core.lib.algorithms.schedule_startup_policy")


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
            "detector": service_entry("detector", execute_device="edge-a"),
        }
    )
    task = Task(
        source_id=3,
        task_id=4,
        source_device="edge-a",
        all_edge_devices=["edge-a", "edge-b"],
        dag=dag,
        metadata={"buffer_size": 2, "resolution": "720p", "fps": 10},
        raw_metadata={"buffer_size": 2, "resolution": "1080p", "fps": 20},
        file_path="clip.mp4",
    )
    task.get_service("detector").set_scenario_data({"obj_num": [1, 3]})
    task.get_service("detector").set_execute_time(0.6)
    task.get_service("detector").set_transmit_time(0.2)
    task.set_tmp_data({"file_size": 1.25, "file_dynamics": 0.4})
    task.set_flow_index(TaskConstant.END.value)
    return task


@pytest.mark.unit
def test_data_getter_base_and_http_video_getter_cover_success_and_exhaustion(monkeypatch, tmp_path):
    with pytest.raises(NotImplementedError):
        data_getter_base_module.BaseDataGetter()(SimpleNamespace())

    getter = http_getter_module.HttpVideoGetter()
    system = SimpleNamespace(
        source_id=5,
        video_data_source="http://datasource",
        meta_data={"fps": 10, "buffer_size": 2},
        raw_meta_data={"fps": 20},
        cumulative_scheduling_frame_count=0,
        task_dag=build_task().get_dag(),
        service_deployment={"detector": ["edge-a"]},
    )

    monkeypatch.setattr(
        http_getter_module.Context,
        "get_parameter",
        staticmethod(lambda key: {"GEN_FILTER_NAME": "simple", "GEN_PROCESS_NAME": "simple", "GEN_COMPRESS_NAME": "simple"}.get(key)),
    )
    monkeypatch.chdir(tmp_path)

    class FakeResponse:
        def __init__(self, content):
            self.content = content

    exhausted_calls = []
    monkeypatch.setattr(
        http_getter_module,
        "http_request",
        lambda url, method=None, **kwargs: exhausted_calls.append(url) or [],
    )
    assert getter.request_source_data(system, task_id=1) is False
    assert exhausted_calls == ["http://datasource/source"]
    assert http_getter_module.HttpVideoGetter.compute_cost_time(system, cost=0.05, actual_buffer_size=1) == pytest.approx(0.05)

    request_log = []

    def fake_http_request(url, method=None, no_decode=False, **kwargs):
        request_log.append((url, no_decode))
        if url.endswith("/source"):
            return ["hash-0", "hash-1"]
        return FakeResponse(b"video-bytes")

    monkeypatch.setattr(http_getter_module, "http_request", fake_http_request)
    assert getter.request_source_data(system, task_id=2) is True
    assert Path(getter.file_name).read_bytes() == b"video-bytes"
    assert getter.hash_codes == ["hash-0", "hash-1"]


@pytest.mark.unit
def test_http_video_getter_call_generates_tasks_and_cleans_files(monkeypatch, tmp_path):
    getter = http_getter_module.HttpVideoGetter()
    system = SimpleNamespace(
        source_id=5,
        video_data_source="http://datasource",
        meta_data={"fps": 10, "buffer_size": 2},
        raw_meta_data={"fps": 20},
        cumulative_scheduling_frame_count=0,
        task_dag=build_task().get_dag(),
        service_deployment={"detector": ["edge-a"]},
        generate_task=lambda task_id, dag, deployment, metadata, file_name, hash_codes: {
            "task_id": task_id,
            "file_name": file_name,
            "hashes": hash_codes,
        },
        submit_task_to_controller=lambda task: submitted.append(task),
    )
    submitted = []
    removed = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(getter, "request_source_data", lambda cur_system, task_id: setattr(getter, "hash_codes", ["hash-0", "hash-1"]) or setattr(getter, "file_name", "payload.mp4") or Path("payload.mp4").write_bytes(b"video") is None or True)
    monkeypatch.setattr(http_getter_module.Counter, "get_count", staticmethod(lambda name: 11))
    monkeypatch.setattr(http_getter_module.time, "time", lambda: 10.0)
    monkeypatch.setattr(http_getter_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(http_getter_module.FileOps, "remove_file", lambda file_path: removed.append(file_path))

    getter(system)

    assert system.cumulative_scheduling_frame_count == 4
    assert submitted == [{"task_id": 11, "file_name": "payload.mp4", "hashes": ["hash-0", "hash-1"]}]
    assert removed == ["payload.mp4"]


@pytest.mark.unit
def test_rtsp_video_getter_helpers_reconnect_and_dispatch(monkeypatch):
    getter = rtsp_getter_module.RtspVideoGetter()

    system = SimpleNamespace(
        frame_filter=lambda cur_system, frame: frame.sum() >= 0,
        frame_process=lambda cur_system, frame, src, dst: frame + 1,
        frame_compress=lambda cur_system, frames, file_name: ("compressed", len(frames), file_name),
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert bool(getter.filter_frame(system, frame)) is True
    assert np.array_equal(getter.process_frame(system, frame, "1080p", "720p"), frame + 1)
    assert getter.compress_frames(system, [frame], "clip.mp4") == ("compressed", 1, "clip.mp4")
    with pytest.raises(AssertionError, match="Frame buffer is not list or is empty"):
        getter.compress_frames(system, [], "clip.mp4")

    class DummyCapture:
        def __init__(self, opened, reads):
            self.opened = opened
            self.reads = iter(reads)

        def isOpened(self):
            return self.opened

        def read(self):
            return next(self.reads)

        def release(self):
            return None

    captures = iter(
        [
            DummyCapture(True, [(False, None)]),
            DummyCapture(True, [(True, np.ones((2, 2, 3), dtype=np.uint8))]),
        ]
    )
    monkeypatch.setattr(getter, "_open_capture", lambda url: next(captures))
    monkeypatch.setattr(rtsp_getter_module.time, "sleep", lambda seconds: None)

    frame = getter.get_one_frame(SimpleNamespace(source_id=1, video_data_source="rtsp://camera"))
    assert frame.shape == (2, 2, 3)

    generated = []
    removed = []
    monkeypatch.setattr(getter, "process_frame", lambda cur_system, frame, src, dst: frame)
    monkeypatch.setattr(getter, "compress_frames", lambda cur_system, frames, file_name: generated.append(("compress", len(frames), file_name)))
    monkeypatch.setattr(rtsp_getter_module.FileOps, "remove_file", lambda file_path: removed.append(file_path))
    run_system = SimpleNamespace(
        source_id=2,
        generate_task=lambda task_id, dag, deployment, metadata, file_name, hash_codes: {"task_id": task_id, "file": file_name},
        submit_task_to_controller=lambda task: generated.append(("submit", task)),
        raw_meta_data={"resolution": "1080p"},
        meta_data={"resolution": "720p"},
    )

    getter.generate_and_send_new_task(run_system, [frame], 7, build_task().get_dag(), {"detector": ["edge-a"]}, {"resolution": "720p"})
    assert any(event[0] == "compress" for event in generated)
    assert any(event[0] == "submit" for event in generated)
    assert removed and removed[0].endswith(".mp4")

    started_processes = []

    class DummyProcess:
        def __init__(self, target=None, args=None):
            self.target = target
            self.args = args or ()

        def start(self):
            started_processes.append((self.target, self.args))

    monkeypatch.setattr(rtsp_getter_module.multiprocessing, "Process", DummyProcess)
    monkeypatch.setattr(rtsp_getter_module.Counter, "get_count", staticmethod(lambda name: 8))
    getter.frame_buffer = [np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8)]
    dispatch_system = SimpleNamespace(
        source_id=2,
        meta_data={"buffer_size": 2, "fps": 10},
        raw_meta_data={"fps": 20},
        cumulative_scheduling_frame_count=0,
        task_dag=build_task().get_dag(),
        service_deployment={"detector": ["edge-a"]},
        video_data_source="rtsp://camera",
    )
    monkeypatch.setattr(getter, "get_one_frame", lambda cur_system: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(getter, "filter_frame", lambda cur_system, frame: True)
    getter(dispatch_system)
    assert dispatch_system.cumulative_scheduling_frame_count == 4
    assert started_processes
    assert getter.frame_buffer == []


@pytest.mark.unit
def test_scheduler_helper_algorithms_cover_base_contracts_selection_and_retrieval(monkeypatch):
    with pytest.raises(NotImplementedError):
        base_config_module.BaseConfigExtraction()(SimpleNamespace())
    with pytest.raises(NotImplementedError):
        base_policy_retrieval_module.BasePolicyRetrieval()(SimpleNamespace())
    with pytest.raises(NotImplementedError):
        base_scenario_module.BaseScenarioRetrieval()(SimpleNamespace())
    with pytest.raises(NotImplementedError):
        base_startup_module.BaseStartupPolicy()({"dag": {}})

    info = {"dag": {"detector": {}}, "source": {"id": 1}}
    assert startup_policy_module.FixedStartupPolicy()(info)["dag"] == {"detector": {}}

    task = build_task()
    task.get_service("detector").set_execute_time(0.8)
    assert policy_retrieval_module.SimplePolicyRetrieval()(task)["edge_device"] == "edge-a"
    assert scenario_retrieval_module.SimpleScenarioRetrieval()(task)["delay"] == pytest.approx(0.5)
    casva_scenario = scenario_retrieval_module.CASVAScenarioRetrieval()(task)
    assert casva_scenario["segment_size"] == 1.25
    assert casva_scenario["content_dynamics"] == 0.4

    monkeypatch.setattr(selection_base_module.NodeInfo, "get_all_edge_nodes", staticmethod(lambda: ["edge-a", "edge-b", "edge-c"]))
    selector = selection_base_module.BaseSelectionPolicy(scope="cluster")
    assert selector.get_candidate_node_set({"node_set": ["edge-a", "edge-b"], "all_edge_nodes": ["edge-b", "edge-c"]}) == [
        "edge-b",
        "edge-c",
    ]
    selector.scope = "unknown"
    assert selector.get_candidate_node_set({"node_set": ["edge-a", "edge-b"]}) == ["edge-a", "edge-b"]

    fixed_position = selection_policy_module.FixedSelectionPolicy(SimpleNamespace(), 1, fixed_value=1, fixed_type="position")
    assert fixed_position({"source": {"id": 1}, "node_set": ["edge-a", "edge-b"]}) == "edge-b"
    fixed_hostname = selection_policy_module.FixedSelectionPolicy(SimpleNamespace(), 1, fixed_value="edge-b", fixed_type="hostname")
    assert fixed_hostname({"source": {"id": 1}, "node_set": ["edge-a", "edge-b"]}) == "edge-b"

    random_selector = selection_policy_module.RandomSelectionPolicy(SimpleNamespace(), 1, scope="node_set")
    monkeypatch.setattr(random_selection_module.random, "choice", lambda seq: seq[-1])
    assert random_selector({"source": {"id": 1}, "node_set": ["edge-a", "edge-b"]}) == "edge-b"


@pytest.mark.unit
def test_schedule_config_extractors_load_expected_knobs_and_config_files(monkeypatch):
    captured_paths = []

    for module in (
        config_simple_module,
        config_fc_module,
        config_chameleon_module,
        config_casva_module,
        config_hedger_module,
        config_hei_module,
        config_hei_drl_module,
    ):
        monkeypatch.setattr(module.Context, "get_file_path", staticmethod(lambda relative_path: f"/runtime/{relative_path}"))

    def fake_read_yaml(path):
        captured_paths.append(path)
        if path.endswith("scheduler_config.yaml"):
            return {"fps": [5, 10], "resolution": ["480p", "720p"], "buffer_size": [1, 2], "qp": [20, 28]}
        return {"loaded_from": path}

    for module in (
        config_simple_module,
        config_fc_module,
        config_chameleon_module,
        config_casva_module,
        config_hedger_module,
        config_hei_module,
        config_hei_drl_module,
    ):
        monkeypatch.setattr(module.YamlOps, "read_yaml", staticmethod(fake_read_yaml))
    scheduler = SimpleNamespace()

    config_extraction_module.SimpleConfigExtraction()(scheduler)
    assert scheduler.schedule_knobs == ["resolution", "fps", "buffer_size", "pipeline"]

    config_extraction_module.FCConfigExtraction()(scheduler)
    assert scheduler.schedule_knobs == ["resolution"]

    config_extraction_module.ChameleonConfigExtraction()(scheduler)
    assert scheduler.schedule_knobs == ["resolution", "fps"]

    config_extraction_module.CASVAConfigExtraction("drl.yaml", "hyper.yaml")(scheduler)
    assert scheduler.qp_list == [20, 28]
    assert scheduler.drl_params == {"loaded_from": "/runtime/scheduler/casva/drl.yaml"}
    assert scheduler.hyper_params == {"loaded_from": "/runtime/scheduler/casva/hyper.yaml"}

    config_extraction_module.HedgerConfigExtraction("network.yaml", "hyper.yaml", "agent.yaml")(scheduler)
    assert scheduler.network_params == {"loaded_from": "/runtime/scheduler/hedger/network.yaml"}
    assert scheduler.agent_params == {"loaded_from": "/runtime/scheduler/hedger/agent.yaml"}

    config_extraction_module.HEIConfigExtraction("drl.yaml", "hyper.yaml")(scheduler)
    assert scheduler.monotonic_schedule_knobs == ["resolution", "fps", "buffer_size"]
    assert scheduler.non_monotonic_schedule_knobs == ["pipeline"]

    config_extraction_module.HEIDRLConfigExtraction("drl.yaml", "hyper.yaml")(scheduler)
    assert scheduler.drl_params == {"loaded_from": "/runtime/scheduler/hei-drl/drl.yaml"}
    assert scheduler.hyper_params == {"loaded_from": "/runtime/scheduler/hei-drl/hyper.yaml"}
    assert any(path.endswith("scheduler/hei-drl/hyper.yaml") for path in captured_paths)
