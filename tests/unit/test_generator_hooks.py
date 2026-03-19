import importlib
import json
from pathlib import Path

import pytest

from core.lib.content import Task


class StopGeneratorLoop(RuntimeError):
    pass


def build_dag_deployment(execute_device="edge-node"):
    return {
        "face-detection": {
            "service": {
                "service_name": "face-detection",
                "execute_device": execute_device,
            },
            "next_nodes": [],
        }
    }


def patch_generator_runtime(monkeypatch, generator_module, hooks, video_generator_module=None):
    def fake_get_algorithm(algorithm, al_name=None, **kwargs):
        try:
            return hooks[algorithm]
        except KeyError as exc:
            raise AssertionError(f"Unexpected algorithm request: {algorithm}") from exc

    monkeypatch.setattr(generator_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    if video_generator_module is not None:
        monkeypatch.setattr(video_generator_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))

    monkeypatch.setenv("ALL_EDGE_DEVICES", "['edge-node', 'edge-target']")
    monkeypatch.setenv("REQUEST_SCHEDULING_INTERVAL", "1")
    monkeypatch.setattr(generator_module.NodeInfo, "get_local_device", staticmethod(lambda: "edge-node"))
    monkeypatch.setattr(generator_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-node"))
    monkeypatch.setattr(generator_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: hostname))
    monkeypatch.setattr(
        generator_module.PortInfo,
        "get_component_port",
        staticmethod(lambda component: {"scheduler": 9001, "controller": 9002}[component]),
    )


@pytest.mark.unit
def test_generator_request_schedule_policy_and_generate_task_follow_hook_contracts(monkeypatch):
    generator_module = importlib.import_module("core.generator.generator")

    hook_calls = {}

    def before_schedule(system):
        hook_calls["before"] = {
            "source_id": system.source_id,
            "meta_data": system.raw_meta_data,
        }
        return hook_calls["before"]

    def after_schedule(system, scheduler_response):
        hook_calls["after"] = scheduler_response

    hooks = {
        "GEN_BSO": before_schedule,
        "GEN_ASO": after_schedule,
        "GEN_GETTER": lambda system: None,
        "GEN_BSTO": lambda system, task: None,
    }

    patch_generator_runtime(monkeypatch, generator_module, hooks)

    captured_request = {}

    def fake_http_request(url, method=None, data=None, **kwargs):
        captured_request.update(url=url, method=method, data=data)
        return {"plan": {"buffer_size": 2}}

    monkeypatch.setattr(generator_module, "http_request", fake_http_request)

    class DummyGenerator(generator_module.Generator):
        def run(self):
            raise NotImplementedError

    generator = DummyGenerator(
        source_id=7,
        metadata={"fps": 25, "buffer_size": 1},
        task_dag=build_dag_deployment(execute_device="edge-target"),
    )

    task = generator.generate_task(
        task_id=3,
        task_dag=generator.task_dag,
        service_deployment={"edge-target": ["face-detection"]},
        meta_data={"fps": 15},
        compressed_path="payload.bin",
        hash_codes=[11, 12, 13],
    )

    assert task.get_source_id() == 7
    assert task.get_metadata() == {"fps": 15}
    assert task.get_raw_metadata() == {"fps": 25, "buffer_size": 1}
    assert task.get_hash_data() == [11, 12, 13]

    generator.request_schedule_policy()

    assert captured_request["url"] == "http://cloud-node:9001/schedule"
    assert captured_request["method"] == "GET"
    assert json.loads(captured_request["data"]["data"]) == {
        "source_id": 7,
        "meta_data": {"fps": 25, "buffer_size": 1},
    }
    assert hook_calls["after"] == {"plan": {"buffer_size": 2}}


@pytest.mark.unit
def test_generator_submit_task_to_controller_invokes_bsto_records_timing_and_uploads_file(
    monkeypatch,
    tmp_path,
):
    generator_module = importlib.import_module("core.generator.generator")

    call_order = []

    def before_submit(system, task):
        call_order.append(("bsto", task.get_task_id()))

    hooks = {
        "GEN_BSO": lambda system: {},
        "GEN_ASO": lambda system, response: None,
        "GEN_GETTER": lambda system: None,
        "GEN_BSTO": before_submit,
    }

    patch_generator_runtime(monkeypatch, generator_module, hooks)

    uploaded = {}

    def fake_http_request(url, method=None, data=None, files=None, **kwargs):
        uploaded.update(url=url, method=method, data=data, files=files)

    monkeypatch.setattr(generator_module, "http_request", fake_http_request)

    class DummyGenerator(generator_module.Generator):
        def run(self):
            raise NotImplementedError

    generator = DummyGenerator(
        source_id=1,
        metadata={"fps": 10},
        task_dag=build_dag_deployment(execute_device="edge-target"),
    )

    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(b"payload")

    task = generator.generate_task(
        task_id=5,
        task_dag=Task.extract_dag_from_dag_deployment(build_dag_deployment(execute_device="edge-target")),
        service_deployment={"edge-target": ["face-detection"]},
        meta_data={"fps": 10},
        compressed_path=str(payload_path),
        hash_codes=[],
    )
    task.set_flow_index("face-detection")

    monkeypatch.setattr(
        generator,
        "record_transmit_start_ts",
        lambda cur_task: call_order.append(("record", cur_task.get_task_id())),
    )

    generator.submit_task_to_controller(task)

    file_name, file_handle, content_type = uploaded["files"]["file"]
    assert call_order == [("bsto", 5), ("record", 5)]
    assert uploaded["url"] == "http://edge-target:9002/submit_task"
    assert uploaded["method"] == "POST"
    assert uploaded["data"]["data"] == task.serialize()
    assert file_name == str(payload_path)
    assert file_handle.read() == b"payload"
    assert content_type == "multipart/form-data"
    file_handle.close()

    with pytest.raises(AssertionError, match="Task is empty when submit to controller"):
        generator.submit_task_to_controller(None)


@pytest.mark.unit
def test_video_generator_submit_records_total_start_before_parent_submit(monkeypatch):
    generator_module = importlib.import_module("core.generator.generator")
    video_generator_module = importlib.import_module("core.generator.video_generator")

    hooks = {
        "GEN_BSO": lambda system: {},
        "GEN_ASO": lambda system, response: None,
        "GEN_GETTER": lambda system: None,
        "GEN_BSTO": lambda system, task: None,
        "GEN_FILTER": object(),
        "GEN_PROCESS": object(),
        "GEN_COMPRESS": object(),
        "GEN_GETTER_FILTER": object(),
    }

    patch_generator_runtime(monkeypatch, generator_module, hooks, video_generator_module=video_generator_module)

    order = []

    monkeypatch.setattr(
        video_generator_module.VideoGenerator,
        "record_total_start_ts",
        staticmethod(lambda task: order.append("total")),
    )
    monkeypatch.setattr(
        generator_module.Generator,
        "submit_task_to_controller",
        lambda self, task: order.append("submit"),
    )

    generator = video_generator_module.VideoGenerator(
        source_id=3,
        source_url="http://source/video",
        source_metadata={"fps": 30},
        dag=build_dag_deployment(),
    )

    generator.submit_task_to_controller(object())

    assert order == ["total", "submit"]


@pytest.mark.unit
def test_video_generator_run_waits_for_health_skips_filtered_rounds_and_requests_schedule(monkeypatch):
    generator_module = importlib.import_module("core.generator.generator")
    video_generator_module = importlib.import_module("core.generator.video_generator")

    after_schedule_calls = []
    getter_filter_calls = []
    schedule_requests = []

    service_states = iter([False, True, True, True])
    health_states = iter([False, True])
    getter_states = iter([False, True])

    def after_schedule(system, scheduler_response):
        after_schedule_calls.append(scheduler_response)

    def data_getter(system):
        system.cumulative_scheduling_frame_count = (
            system.request_scheduling_interval * system.raw_meta_data["fps"] + 1
        )

    def getter_filter(system):
        getter_filter_calls.append(system.source_id)
        return next(getter_states)

    hooks = {
        "GEN_BSO": lambda system: {"source_id": system.source_id},
        "GEN_ASO": after_schedule,
        "GEN_GETTER": data_getter,
        "GEN_BSTO": lambda system, task: None,
        "GEN_FILTER": object(),
        "GEN_PROCESS": object(),
        "GEN_COMPRESS": object(),
        "GEN_GETTER_FILTER": getter_filter,
    }

    patch_generator_runtime(monkeypatch, generator_module, hooks, video_generator_module=video_generator_module)

    monkeypatch.setattr(
        video_generator_module.KubeConfig,
        "check_services_running",
        staticmethod(lambda: next(service_states)),
    )
    monkeypatch.setattr(
        video_generator_module.HealthChecker,
        "check_processors_health",
        staticmethod(lambda: next(health_states)),
    )
    monkeypatch.setattr(video_generator_module.time, "sleep", lambda *_args, **_kwargs: None)

    generator = video_generator_module.VideoGenerator(
        source_id=9,
        source_url="http://source/video",
        source_metadata={"fps": 5},
        dag=build_dag_deployment(),
    )

    def fake_request_schedule_policy():
        schedule_requests.append(generator.cumulative_scheduling_frame_count)
        raise StopGeneratorLoop

    monkeypatch.setattr(generator, "request_schedule_policy", fake_request_schedule_policy)

    with pytest.raises(StopGeneratorLoop):
        generator.run()

    assert after_schedule_calls == [None]
    assert getter_filter_calls == [9, 9]
    assert schedule_requests == [6]
