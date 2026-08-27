import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core.lib.common import TaskConstant
from core.lib.content import Task
from core.lib.runtime import RuntimeContext, RuntimeEndpoint


parameter_base_module = importlib.import_module("core.lib.algorithms.parameter_monitor.base_monitor")
available_bandwidth_module = importlib.import_module("core.lib.algorithms.parameter_monitor.available_bandwidth_monitor")
cpu_flops_module = importlib.import_module("core.lib.algorithms.parameter_monitor.cpu_flops_monitor")
cpu_usage_module = importlib.import_module("core.lib.algorithms.parameter_monitor.cpu_usage_monitor")
gpu_flops_module = importlib.import_module("core.lib.algorithms.parameter_monitor.gpu_flops_monitor")
memory_capacity_module = importlib.import_module("core.lib.algorithms.parameter_monitor.memory_capacity_monitor")
memory_usage_module = importlib.import_module("core.lib.algorithms.parameter_monitor.memory_usage_monitor")
model_flops_module = importlib.import_module("core.lib.algorithms.parameter_monitor.model_flops_monitor")
model_memory_module = importlib.import_module("core.lib.algorithms.parameter_monitor.model_memory_monitor")
queue_state_module = importlib.import_module("core.lib.algorithms.parameter_monitor.queue_state_monitor")
result_base_module = importlib.import_module("core.lib.algorithms.result_visualizer.base_visualizer")
result_curve_module = importlib.import_module("core.lib.algorithms.result_visualizer.curve_visualizer")
result_topology_module = importlib.import_module("core.lib.algorithms.result_visualizer.topology_visualizer")
dag_deployment_module = importlib.import_module("core.lib.algorithms.result_visualizer.dag_deployment_topology_visualizer")
dag_offloading_module = importlib.import_module("core.lib.algorithms.result_visualizer.dag_offloading_topology_visualizer")
e2e_delay_module = importlib.import_module("core.lib.algorithms.result_visualizer.end_to_end_delay_visualizer")
frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.frame_visualizer")
image_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.image_visualizer")
multiple_roi_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.multiple_roi_frame_visualizer")
multiple_object_number_module = importlib.import_module("core.lib.algorithms.result_visualizer.multiple_object_number_visualizer")
object_number_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.object_number_visualizer")
roi_frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.roi_frame_visualizer")
roi_label_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.roi_label_frame_visualizer")
service_delay_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.service_processing_delay_visualizer")
service_queue_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.service_queue_length_visualizer")
system_base_module = importlib.import_module("core.lib.algorithms.system_visualizer.base_visualizer")
system_curve_module = importlib.import_module("core.lib.algorithms.system_visualizer.curve_visualizer")
cpu_visualizer_module = importlib.import_module("core.lib.algorithms.system_visualizer.cpu_usage_visualizer")
memory_visualizer_module = importlib.import_module("core.lib.algorithms.system_visualizer.memory_usage_visualizer")
overhead_visualizer_module = importlib.import_module("core.lib.algorithms.system_visualizer.schedule_overhead_visualizer")


def content_profile(frame_count=1):
    return {
        "frame_count": frame_count,
    }


def service_entry(name, *, execute_device="", next_nodes=None, prev_nodes=None):
    return {
        "service": {
            "service_name": name,
            "execute_device": execute_device,
        },
        "next_nodes": next_nodes or [],
        "prev_nodes": prev_nodes or [],
    }


def build_visualization_task():
    dag = Task.extract_dag_from_dict(
        {
            "detector": service_entry("detector", execute_device="edge-a", next_nodes=["classifier"]),
            "classifier": service_entry("classifier", execute_device="cloud-a"),
        }
    )
    task = Task(
        source_id=1,
        task_id=9,
        source_device="edge-a",
        all_edge_devices=["edge-a", "edge-b"],
        dag=dag,
        metadata={"buffer_size": 2, "resolution": "720p"},
        raw_metadata={"buffer_size": 2, "resolution": "1080p"},
        file_path="sample.mp4",
        runtime_directory_revision=1,
        runtime_routes=[
            {"component": "processor", "logical_service": "detector", "target_node": "edge-a",
             "runtime_id": "processor-detector-edge-a-0", "fqdn": "detector-a", "port": 9000},
            {"component": "processor", "logical_service": "detector", "target_node": "edge-b",
             "runtime_id": "processor-detector-edge-b-0", "fqdn": "detector-b", "port": 9000},
            {"component": "processor", "logical_service": "classifier", "target_node": "edge-b",
             "runtime_id": "processor-classifier-edge-b-0", "fqdn": "classifier-b", "port": 9000},
        ],
    )
    task.get_service("detector").set_content_data(
        {
            "service": "detector",
            "outputs": {
                "bbox": [
                    {
                        "frame_index": 0,
                        "items": [
                            {"bbox": [1, 1, 5, 5], "score": 0.9, "label": "car", "object_id": 1}
                        ],
                    }
                ]
            },
            "profile": content_profile(),
        }
    )
    task.get_service("detector").set_scenario_data({"obj_num": [2, 4]})
    task.get_service("classifier").set_content_data(
        {
            "service": "classifier",
            "outputs": {
                "text": [
                    {
                        "frame_index": 0,
                        "items": [
                            {"text": "car", "source_object_id": 1, "bbox": [1, 1, 5, 5], "score": 0.9}
                        ],
                    }
                ]
            },
            "profile": content_profile(),
        }
    )
    task.get_service("detector").set_execute_time(0.4)
    task.get_service("detector").set_transmit_time(0.1)
    task.get_service("classifier").set_execute_time(0.6)
    task.get_service("classifier").set_transmit_time(0.2)
    task.set_flow_index(TaskConstant.END.value)
    return task


@pytest.mark.unit
def test_base_visualizer_and_monitor_contracts_raise_or_update_resources():
    with pytest.raises(NotImplementedError):
        result_base_module.BaseVisualizer(variables=["x"])(SimpleNamespace())
    with pytest.raises(NotImplementedError):
        result_curve_module.CurveVisualizer(variables=["x"])(SimpleNamespace())
    with pytest.raises(NotImplementedError):
        result_topology_module.TopologyVisualizer(variables=["x"])(SimpleNamespace())
    with pytest.raises(NotImplementedError):
        system_base_module.BaseVisualizer(variables=["x"])()
    with pytest.raises(NotImplementedError):
        system_curve_module.CurveVisualizer(variables=["x"])()

    class DemoMonitor(parameter_base_module.BaseMonitor):
        def __init__(self, system):
            super().__init__(system)
            self.name = "demo"

        def get_parameter_value(self):
            return 42

    endpoints = [
        RuntimeEndpoint(component="processor", target_node="edge-a", logical_service="detector", fqdn="detector", port=31000),
        RuntimeEndpoint(component="processor", target_node="edge-a", logical_service="face", fqdn="face", port=31001),
    ]
    system = SimpleNamespace(
        resource_info={}, local_device="edge-a",
        runtime_routes=lambda component=None, target_node=None: endpoints,
    )
    monitor = DemoMonitor(system)
    thread = monitor()
    thread.run()
    assert system.resource_info == {"demo": 42}


@pytest.mark.unit
def test_local_and_remote_parameter_monitors_collect_expected_values(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "cpu_percent", lambda: 12.5)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=61.5, total=8e9))

    endpoints = [
        RuntimeEndpoint(component="processor", target_node="edge-a", logical_service="detector", fqdn="detector", port=31000),
        RuntimeEndpoint(component="processor", target_node="edge-a", logical_service="face", fqdn="face", port=31001),
    ]
    system = SimpleNamespace(
        resource_info={}, local_device="edge-a",
        runtime_routes=lambda component=None, target_node=None: endpoints,
    )
    assert cpu_usage_module.CPUUsageMonitor(system).get_parameter_value() == 12.5
    assert memory_usage_module.MemoryUsageMonitor(system).get_parameter_value() == 0.615
    assert memory_capacity_module.MemoryCapacityMonitor(system).get_parameter_value() == 8.0

    monkeypatch.setattr(model_flops_module, "http_request", lambda address, method=None, **kwargs: 3e9)
    assert model_flops_module.ModelFlopsMonitor(system).get_parameter_value() == {"detector": 3.0, "face": 3.0}

    processor_state = {
        "waiting_count": 7,
        "waiting_tasks": [{"task_uuid": "task-2"}],
        "busy": True,
        "running_elapsed_s": 0.5,
        "running_phase": "processing",
        "phase_elapsed_s": 0.25,
        "capacity": 1,
        "sequence": 4,
        "running_task": {"task_uuid": "task-1"},
        "observed_at": 123.0,
    }
    monkeypatch.setattr(queue_state_module, "http_request", lambda address, method=None, **kwargs: processor_state)
    queue_monitor = queue_state_module.QueueStateMonitor(system)
    queue_states = queue_monitor.get_parameter_value()
    assert queue_states["detector"] == {
        **processor_state,
        "runtime": {
            "component": "processor",
            "target_node": "edge-a",
            "logical_service": "detector",
            "runtime_id": "",
            "runtime_service_uid": "",
            "service_uid": "",
            "endpoint_pod_uid": "",
            "deployment_revision": 0,
        },
    }
    assert queue_states["face"]["runtime"]["logical_service"] == "face"
    queue_monitor.run_monitor(system)
    assert system.resource_info["queue_state"]["detector"]["waiting_count"] == 7
    assert system.resource_info["queue_state"]["detector"]["busy"] is True

    monkeypatch.setattr(
        model_memory_module,
        "http_request",
        lambda address, method=None, timeout=None: 1_000_000_000 if "face" in address else 5_000_000_000,
    )
    model_memory_monitor = model_memory_module.ModelMemoryMonitor(system)
    assert model_memory_monitor.get_parameter_value() == {"face": 1.0, "detector": 5.0}


@pytest.mark.unit
def test_cpu_gpu_and_bandwidth_monitors_cover_success_and_fallback_paths(monkeypatch):
    system = SimpleNamespace(resource_info={})

    monkeypatch.setattr(
        cpu_flops_module.CPUFlopsMonitor,
        "parse_lscpu",
        staticmethod(
            lambda: {
                "flags": ["avx2"],
                "sockets": 1,
                "cores_per_socket": 4,
                "threads_per_core": 2,
                "max_mhz": 1000.0,
                "model_name": "unit-cpu",
            }
        ),
    )
    cpu_monitor = cpu_flops_module.CPUFlopsMonitor(system)
    assert cpu_monitor.get_parameter_value() > 0

    monkeypatch.setattr(
        cpu_flops_module.CPUFlopsMonitor,
        "parse_lscpu",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("lscpu missing"))),
    )
    assert cpu_flops_module.CPUFlopsMonitor(system).get_parameter_value() == 0

    fake_cuda = SimpleNamespace(
        Device=type(
            "DeviceFactory",
            (),
            {
                "count": staticmethod(lambda: 1),
                "__call__": staticmethod(
                    lambda idx: SimpleNamespace(
                        name=lambda: "RTX",
                        compute_capability=lambda: (8, 6),
                        get_attribute=lambda attr: 2 if attr == "MULTIPROCESSOR_COUNT" else 1_000_000,
                    )
                ),
                "MULTIPROCESSOR_COUNT": "MULTIPROCESSOR_COUNT",
                "CLOCK_RATE": "CLOCK_RATE",
            },
        )(),
        device_attribute=SimpleNamespace(MULTIPROCESSOR_COUNT="MULTIPROCESSOR_COUNT", CLOCK_RATE="CLOCK_RATE"),
    )
    monkeypatch.setattr(gpu_flops_module.GPUFlopsMonitor, "load_pycuda", staticmethod(lambda: fake_cuda))
    monkeypatch.setattr(gpu_flops_module.GPUFlopsMonitor, "is_jetson_device", staticmethod(lambda: False))
    assert gpu_flops_module.GPUFlopsMonitor(system).get_parameter_value() > 0

    monkeypatch.setattr(
        gpu_flops_module.GPUFlopsMonitor,
        "get_device_fp32_flops",
        lambda self, is_jetson=False: (_ for _ in ()).throw(RuntimeError("no gpu")),
    )
    assert gpu_flops_module.GPUFlopsMonitor(system).get_parameter_value() is None

    started_threads = []

    class DummyThread:
        def __init__(self, target=None, args=None):
            self.target = target
            self.args = args or ()

        def start(self):
            started_threads.append(self.args)

    fake_iperf3 = SimpleNamespace(
        Client=lambda: SimpleNamespace(
            duration=None,
            server_hostname=None,
            port=None,
            protocol=None,
            run=lambda: SimpleNamespace(error=None, sent_Mbps=88.0),
        ),
        Server=object,
    )
    monkeypatch.setitem(sys.modules, "iperf3", fake_iperf3)

    monkeypatch.setattr(available_bandwidth_module.Context, "get_parameter", staticmethod(lambda key: 9000))
    monkeypatch.setattr(available_bandwidth_module.threading, "Thread", DummyThread)
    server_context = RuntimeContext({"local_node": "cloud-a", "cloud_node": "cloud-a", "nodes": {"cloud-a": {"role": "cloud"}}})
    server_system = SimpleNamespace(resource_info={}, local_device="cloud-a", runtime_context=server_context)
    server_monitor = available_bandwidth_module.AvailableBandwidthMonitor(server_system)
    assert server_monitor.get_parameter_value() == -1
    assert started_threads == [(9000,)]

    monkeypatch.setattr(
        available_bandwidth_module,
        "http_request",
        lambda address, method=None, data=None, **kwargs: {"holder": "edge-a"},
    )
    client_context = RuntimeContext({
        "local_node": "edge-a", "cloud_node": "cloud-a",
        "nodes": {"edge-a": {"role": "edge"}, "cloud-a": {"role": "cloud"}},
        "endpoints": {
            "monitor": {"component": "monitor", "target_node": "cloud-a", "fqdn": "10.0.0.1", "port": 5201},
            "scheduler": {"fqdn": "10.0.0.1", "port": 31000},
        },
    })
    client_system = SimpleNamespace(
        resource_info={}, local_device="edge-a", runtime_context=client_context,
        scheduler_endpoint=client_context.resolve_static_endpoint("scheduler"),
    )
    client_monitor = available_bandwidth_module.AvailableBandwidthMonitor(client_system)
    assert client_monitor.permitted_device == "edge-a"
    assert client_monitor.get_parameter_value() == 88.0
    client_monitor.permitted_device = "other-edge"
    assert client_monitor.get_parameter_value() == -1


@pytest.mark.unit
def test_system_visualizers_consume_prefetched_resource_and_overhead():
    cpu_visualizer = cpu_visualizer_module.CPUUsageVisualizer(variables=["edge-a", "edge-b"])
    assert cpu_visualizer(resource={"edge-a": {"cpu_usage": 33.0}}) == {
        "edge-a": 33.0,
        "edge-b": 0,
    }
    assert cpu_visualizer(resource={"edge-x": {"cpu_usage": 12.0}}) == {"edge-a": 0, "edge-b": 0}

    memory_visualizer = memory_visualizer_module.MemoryUsageVisualizer(variables=[])
    assert memory_visualizer(resource=None) == {"no device": 0}
    assert memory_visualizer(resource={"edge-a": {"memory_usage": 44.0}}) == {"edge-a": 44.0}

    overhead_visualizer = overhead_visualizer_module.ScheduleOverheadVisualizer(variables=["overhead"])
    assert overhead_visualizer(scheduling_overhead=0.125) == {"overhead": 125.0}


@pytest.mark.unit
def test_system_visualizers_cover_missing_snapshots_and_default_views():
    cpu_visualizer = cpu_visualizer_module.CPUUsageVisualizer(variables=["edge-a"])
    assert cpu_visualizer(resource=None) == {"edge-a": 0}
    cpu_visualizer.variables = []
    assert cpu_visualizer(resource={"edge-a": {"cpu_usage": 12.0}}) == {"edge-a": 12.0}

    memory_visualizer = memory_visualizer_module.MemoryUsageVisualizer(variables=["edge-a"])
    assert memory_visualizer(resource=None) == {"edge-a": 0}
    memory_visualizer.variables = []
    assert memory_visualizer(resource={"edge-a": {"memory_usage": 22.0}}) == {"edge-a": 22.0}

    overhead_visualizer = overhead_visualizer_module.ScheduleOverheadVisualizer(variables=["overhead"])
    assert overhead_visualizer() == {"overhead": 0}


@pytest.mark.unit
def test_image_helpers_validate_inputs_and_extract_first_frame(monkeypatch):
    import cv2

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert image_visualizer_module.ImageVisualizer.draw_bboxes(frame.copy(), [[1, 1, 4, 4]]).shape == frame.shape
    assert image_visualizer_module.ImageVisualizer.draw_bboxes_and_labels(frame.copy(), [[1, 1, 4, 4]], ["car"]).shape == frame.shape

    with pytest.raises(ValueError, match="numpy array"):
        image_visualizer_module.ImageVisualizer.draw_bboxes("bad-frame", [[1, 1, 4, 4]])
    with pytest.raises(ValueError, match="Bounding boxes must be"):
        image_visualizer_module.ImageVisualizer.draw_bboxes(frame.copy(), "bad-boxes")
    with pytest.raises(ValueError, match="Labels must be a list"):
        image_visualizer_module.ImageVisualizer.draw_bboxes_and_labels(frame.copy(), [[1, 1, 4, 4]], [])
    with pytest.raises(ValueError, match="non-empty string"):
        image_visualizer_module.ImageVisualizer.get_first_frame_from_video("")

    class DummyCap:
        def __init__(self, opened, reads):
            self.opened = opened
            self.reads = iter(reads)

        def isOpened(self):
            return self.opened

        def read(self):
            return next(self.reads)

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: DummyCap(True, [(True, frame)]))
    assert image_visualizer_module.ImageVisualizer.get_first_frame_from_video("demo.mp4").shape == frame.shape

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: DummyCap(False, []))
    with pytest.raises(ValueError, match="Failed to open video file"):
        image_visualizer_module.ImageVisualizer.get_first_frame_from_video("demo.mp4")


@pytest.mark.unit
def test_result_visualizers_render_task_data_and_fallback_images(monkeypatch):
    task = build_visualization_task()
    import cv2

    monkeypatch.setattr(frame_visualizer_module.EncodeOps, "encode_image", staticmethod(lambda image: "encoded"))
    monkeypatch.setattr(frame_visualizer_module.FrameVisualizer, "get_first_frame_from_video", staticmethod(lambda path: np.zeros((8, 8, 3), dtype=np.uint8)))
    assert frame_visualizer_module.FrameVisualizer(variables=["frame"])(task) == {"frame": "encoded"}

    monkeypatch.setattr(frame_visualizer_module.FrameVisualizer, "get_first_frame_from_video", staticmethod(lambda path: (_ for _ in ()).throw(RuntimeError("bad video"))))
    monkeypatch.setattr(cv2, "imread", lambda path: np.ones((4, 4, 3), dtype=np.uint8))
    assert frame_visualizer_module.FrameVisualizer(variables=["frame"])(task) == {"frame": "encoded"}

    drawn_boxes = []
    monkeypatch.setattr(roi_frame_visualizer_module.EncodeOps, "encode_image", staticmethod(lambda image: "roi-encoded"))
    monkeypatch.setattr(roi_frame_visualizer_module.ROIFrameVisualizer, "get_first_frame_from_video", staticmethod(lambda path: np.zeros((8, 8, 3), dtype=np.uint8)))
    monkeypatch.setattr(roi_frame_visualizer_module.ROIFrameVisualizer, "draw_bboxes", staticmethod(lambda image, boxes: drawn_boxes.append(list(boxes)) or image))
    assert roi_frame_visualizer_module.ROIFrameVisualizer(variables=["roi"], roi_service="detector")(task) == {"roi": "roi-encoded"}
    assert drawn_boxes == [[[1, 1, 5, 5]]]

    drawn_pairs = []
    monkeypatch.setattr(roi_label_visualizer_module.EncodeOps, "encode_image", staticmethod(lambda image: "label-encoded"))
    monkeypatch.setattr(roi_label_visualizer_module.ROILabelFrameVisualizer, "get_first_frame_from_video", staticmethod(lambda path: np.zeros((8, 8, 3), dtype=np.uint8)))
    monkeypatch.setattr(
        roi_label_visualizer_module.ROILabelFrameVisualizer,
        "draw_bboxes_and_labels",
        staticmethod(lambda image, boxes, labels: drawn_pairs.append((list(boxes), list(labels))) or image),
    )
    assert roi_label_visualizer_module.ROILabelFrameVisualizer(
        variables=["labeled"],
        roi_service="detector",
        label_service="classifier",
    )(task) == {"labeled": "label-encoded"}
    assert drawn_pairs == [([[1, 1, 5, 5]], ["car"])]

    drawn_multi = []
    monkeypatch.setattr(multiple_roi_visualizer_module.EncodeOps, "encode_image", staticmethod(lambda image: "multi-encoded"))
    monkeypatch.setattr(multiple_roi_visualizer_module.ROIFrameVisualizer, "get_first_frame_from_video", staticmethod(lambda path: np.zeros((8, 8, 3), dtype=np.uint8)))
    monkeypatch.setattr(
        multiple_roi_visualizer_module.ROIFrameVisualizer,
        "draw_bboxes",
        staticmethod(lambda image, boxes: drawn_multi.append(list(boxes)) or image),
    )
    assert multiple_roi_visualizer_module.ROIFrameVisualizer(variables=["multi"], roi_services=["detector"])(task) == {
        "multi": "multi-encoded"
    }
    assert drawn_multi == [[[1, 1, 5, 5]]]

    deployment = dag_deployment_module.DAGDeploymentTopologyVisualizer(variables=["topology"])(task)["topology"]
    assert "execute_device" not in deployment["detector"]["service"]
    assert deployment["detector"]["service"]["data"] == "edge-a\nedge-b"

    task.get_dag().get_end_node().service.set_execute_device("cloud-a")
    offloading = dag_offloading_module.DAGOffloadingTopologyVisualizer(variables=["offloading"])(task)["offloading"]
    assert offloading["detector"]["service"]["data"] == "edge-a"
    assert offloading["classifier"]["service"]["data"] == "cloud-a"

    assert e2e_delay_module.EndToEndDelayVisualizer(variables=["delay"])(task)["delay"] == pytest.approx(1.3)
    assert object_number_visualizer_module.ObjectNumberVisualizer(variables=["obj_num"])(task) == {"obj_num": 3.0}
    task.get_service("classifier").set_scenario_data({"obj_num": 5})
    assert multiple_object_number_module.MultipleObjectNumberVisualizer(
        variables=["detector", "classifier", "missing"]
    )(task) == {"detector": 3.0, "classifier": 5.0, "missing": 0.0}
    assert service_delay_visualizer_module.ServiceProcessingDelayVisualizer(
        variables=["detector", "classifier"]
    )(task) == {"detector": 0.4, "classifier": 0.6}


@pytest.mark.unit
def test_service_queue_length_visualizer_renders_replica_queue_bars():
    visualizer = service_queue_visualizer_module.ServiceQueueLengthVisualizer(variables=["detector", "classifier"])
    result = visualizer(
        build_visualization_task(),
        resource={
            "edge-a": {"queue_state": {"detector": {"waiting_count": 7}}},
            "edge-b": {
                "queue_state": {
                    "detector": {"waiting_count": 2},
                    "classifier": {"waiting_count": 5},
                }
            },
        },
    )

    assert [item["pod_name"] for item in result["detector"]] == [
        "processor-detector-edge-a-0",
        "processor-detector-edge-b-0",
    ]
    assert [item["queue_length"] for item in result["detector"]] == [7.0, 2.0]
    assert result["classifier"] == [
        {
            "device": "edge-b",
            "pod_name": "processor-classifier-edge-b-0",
            "replica_label": "edge-b/processor-classifier-edge-b-0",
            "queue_length": 5.0,
        }
    ]
