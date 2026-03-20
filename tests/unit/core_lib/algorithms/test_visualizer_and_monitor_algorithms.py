import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core.lib.common import TaskConstant
from core.lib.content import Task


parameter_base_module = importlib.import_module("core.lib.algorithms.parameter_monitor.base_monitor")
available_bandwidth_module = importlib.import_module("core.lib.algorithms.parameter_monitor.available_bandwidth_monitor")
cpu_flops_module = importlib.import_module("core.lib.algorithms.parameter_monitor.cpu_flops_monitor")
cpu_usage_module = importlib.import_module("core.lib.algorithms.parameter_monitor.cpu_usage_monitor")
gpu_flops_module = importlib.import_module("core.lib.algorithms.parameter_monitor.gpu_flops_monitor")
memory_capacity_module = importlib.import_module("core.lib.algorithms.parameter_monitor.memory_capacity_monitor")
memory_usage_module = importlib.import_module("core.lib.algorithms.parameter_monitor.memory_usage_monitor")
model_flops_module = importlib.import_module("core.lib.algorithms.parameter_monitor.model_flops_monitor")
model_memory_module = importlib.import_module("core.lib.algorithms.parameter_monitor.model_memory_monitor")
queue_length_module = importlib.import_module("core.lib.algorithms.parameter_monitor.queue_length_monitor")
result_base_module = importlib.import_module("core.lib.algorithms.result_visualizer.base_visualizer")
result_curve_module = importlib.import_module("core.lib.algorithms.result_visualizer.curve_visualizer")
result_topology_module = importlib.import_module("core.lib.algorithms.result_visualizer.topology_visualizer")
dag_deployment_module = importlib.import_module("core.lib.algorithms.result_visualizer.dag_deployment_topology_visualizer")
dag_offloading_module = importlib.import_module("core.lib.algorithms.result_visualizer.dag_offloading_topology_visualizer")
e2e_delay_module = importlib.import_module("core.lib.algorithms.result_visualizer.end_to_end_delay_visualizer")
frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.frame_visualizer")
image_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.image_visualizer")
multiple_roi_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.multiple_roi_frame_visualizer")
object_number_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.object_number_visualizer")
roi_frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.roi_frame_visualizer")
roi_label_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.roi_label_frame_visualizer")
service_delay_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.service_processing_delay_visualizer")
system_base_module = importlib.import_module("core.lib.algorithms.system_visualizer.base_visualizer")
system_curve_module = importlib.import_module("core.lib.algorithms.system_visualizer.curve_visualizer")
cpu_visualizer_module = importlib.import_module("core.lib.algorithms.system_visualizer.cpu_usage_visualizer")
memory_visualizer_module = importlib.import_module("core.lib.algorithms.system_visualizer.memory_usage_visualizer")
overhead_visualizer_module = importlib.import_module("core.lib.algorithms.system_visualizer.schedule_overhead_visualizer")


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
    )
    task.get_service("detector").set_content_data([([[1, 1, 5, 5]], [0.9])])
    task.get_service("detector").set_scenario_data({"obj_num": [2, 4]})
    task.get_service("classifier").set_content_data([(["car"], [0.9])])
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

    system = SimpleNamespace(resource_info={})
    monitor = DemoMonitor(system)
    thread = monitor()
    thread.run()
    assert system.resource_info == {"demo": 42}


@pytest.mark.unit
def test_local_and_remote_parameter_monitors_collect_expected_values(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "cpu_percent", lambda: 12.5)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(percent=61.5, total=8e9))

    system = SimpleNamespace(resource_info={})
    assert cpu_usage_module.CPUUsageMonitor(system).get_parameter_value() == 12.5
    assert memory_usage_module.MemoryUsageMonitor(system).get_parameter_value() == 61.5
    assert memory_capacity_module.MemoryCapacityMonitor(system).get_parameter_value() == 8.0

    monkeypatch.setattr(model_flops_module.NodeInfo, "get_local_device", staticmethod(lambda: "edge-a"))
    monkeypatch.setattr(model_flops_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.2"))
    monkeypatch.setattr(model_flops_module.PortInfo, "get_service_ports_dict", staticmethod(lambda device: {"detector": 31000}))
    monkeypatch.setattr(model_flops_module, "http_request", lambda address, method=None: 3e9)
    assert model_flops_module.ModelFlopsMonitor(system).get_parameter_value() == {"detector": 3.0}

    monkeypatch.setattr(queue_length_module.NodeInfo, "get_local_device", staticmethod(lambda: "edge-a"))
    monkeypatch.setattr(queue_length_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.2"))
    monkeypatch.setattr(queue_length_module.PortInfo, "get_service_ports_dict", staticmethod(lambda device: {"detector": 31000}))
    monkeypatch.setattr(queue_length_module, "http_request", lambda address, method=None: 7)
    assert queue_length_module.QueueLengthMonitor(system).get_parameter_value() == {"detector": 7}

    monkeypatch.setattr(model_memory_module.NodeInfo, "get_local_device", staticmethod(lambda: "edge-a"))
    monkeypatch.setattr(model_memory_module.KubeConfig, "force_refresh", staticmethod(lambda: None))
    monkeypatch.setattr(model_memory_module.KubeConfig, "get_pods_on_node", staticmethod(lambda device: ["processor-face-edge-a-0"]))
    monkeypatch.setattr(
        model_memory_module.KubeConfig,
        "get_pod_memory_from_metrics",
        staticmethod(lambda pods: {"processor-face-edge-a-0": 2_000_000_000}),
    )
    monkeypatch.setattr(model_memory_module.ServiceConfig, "map_pod_name_to_service", staticmethod(lambda pod: "face"))
    assert model_memory_module.ModelMemoryMonitor(system).get_parameter_value() == {"face": 2.0}


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
    assert gpu_flops_module.GPUFlopsMonitor(system).get_parameter_value() == 0

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

    monkeypatch.setattr(available_bandwidth_module.NodeInfo, "get_local_device", staticmethod(lambda: "cloud-a"))
    monkeypatch.setattr(available_bandwidth_module.NodeInfo, "get_node_role", staticmethod(lambda hostname: "cloud"))
    monkeypatch.setattr(available_bandwidth_module.Context, "get_parameter", staticmethod(lambda key: 9000))
    monkeypatch.setattr(available_bandwidth_module.threading, "Thread", DummyThread)
    server_monitor = available_bandwidth_module.AvailableBandwidthMonitor(system)
    assert server_monitor.get_parameter_value() == -1
    assert started_threads == [(9000,)]

    monkeypatch.setattr(available_bandwidth_module.NodeInfo, "get_local_device", staticmethod(lambda: "edge-a"))
    monkeypatch.setattr(available_bandwidth_module.NodeInfo, "get_node_role", staticmethod(lambda hostname: "edge"))
    monkeypatch.setattr(available_bandwidth_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.1"))
    monkeypatch.setattr(available_bandwidth_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-a"))
    monkeypatch.setattr(available_bandwidth_module.PortInfo, "get_component_port", staticmethod(lambda component: 5201))
    monkeypatch.setattr(available_bandwidth_module, "http_request", lambda address, method=None, data=None: {"holder": "edge-a"})
    client_monitor = available_bandwidth_module.AvailableBandwidthMonitor(system)
    assert client_monitor.permitted_device == "edge-a"
    assert client_monitor.get_parameter_value() == 88.0
    client_monitor.permitted_device = "other-edge"
    assert client_monitor.get_parameter_value() == -1


@pytest.mark.unit
def test_system_visualizers_fetch_resource_snapshots_and_overhead(monkeypatch):
    monkeypatch.setattr(cpu_visualizer_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-a"))
    monkeypatch.setattr(cpu_visualizer_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.1"))
    monkeypatch.setattr(cpu_visualizer_module.PortInfo, "get_component_port", staticmethod(lambda component: 31000))
    monkeypatch.setattr(cpu_visualizer_module, "http_request", lambda address, method=None: {"edge-a": {"cpu_usage": 33.0}})

    cpu_visualizer = cpu_visualizer_module.CPUUsageVisualizer(variables=["edge-a", "edge-b"])
    cpu_visualizer.get_resource_url()
    assert cpu_visualizer.resource_url == "http://10.0.0.1:31000/resource"
    assert cpu_visualizer() == {"edge-a": 33.0, "edge-b": 0}
    assert cpu_visualizer(resource={"edge-x": {"cpu_usage": 12.0}}) == {"edge-a": 0, "edge-b": 0}

    monkeypatch.setattr(memory_visualizer_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-a"))
    monkeypatch.setattr(memory_visualizer_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.1"))
    monkeypatch.setattr(memory_visualizer_module.PortInfo, "get_component_port", staticmethod(lambda component: 31000))
    memory_visualizer = memory_visualizer_module.MemoryUsageVisualizer(variables=[])
    assert memory_visualizer(resource=None) == {"no device": 0}
    assert memory_visualizer(resource={"edge-a": {"memory_usage": 44.0}}) == {"edge-a": 44.0}

    monkeypatch.setattr(overhead_visualizer_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-a"))
    monkeypatch.setattr(overhead_visualizer_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.1"))
    monkeypatch.setattr(overhead_visualizer_module.PortInfo, "get_component_port", staticmethod(lambda component: 31000))
    monkeypatch.setattr(overhead_visualizer_module, "http_request", lambda address, method=None: 0.125)
    overhead_visualizer = overhead_visualizer_module.ScheduleOverheadVisualizer(variables=["overhead"])
    assert overhead_visualizer() == {"overhead": 125.0}


@pytest.mark.unit
def test_system_visualizers_cover_missing_scheduler_ports_and_default_views(monkeypatch):
    monkeypatch.setattr(cpu_visualizer_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-a"))
    monkeypatch.setattr(cpu_visualizer_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.1"))
    monkeypatch.setattr(
        cpu_visualizer_module.PortInfo,
        "get_component_port",
        staticmethod(lambda component: (_ for _ in ()).throw(AssertionError("missing scheduler"))),
    )
    cpu_visualizer = cpu_visualizer_module.CPUUsageVisualizer(variables=["edge-a"])
    assert cpu_visualizer.request_resource_info() is None
    assert cpu_visualizer(resource=None) == {"edge-a": 0}
    cpu_visualizer.variables = []
    assert cpu_visualizer(resource={"edge-a": {"cpu_usage": 12.0}}) == {"edge-a": 12.0}

    monkeypatch.setattr(memory_visualizer_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-a"))
    monkeypatch.setattr(memory_visualizer_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.1"))
    monkeypatch.setattr(
        memory_visualizer_module.PortInfo,
        "get_component_port",
        staticmethod(lambda component: (_ for _ in ()).throw(AssertionError("missing scheduler"))),
    )
    memory_visualizer = memory_visualizer_module.MemoryUsageVisualizer(variables=["edge-a"])
    assert memory_visualizer.request_resource_info() is None
    assert memory_visualizer(resource=None) == {"edge-a": 0}
    memory_visualizer.variables = []
    assert memory_visualizer(resource={"edge-a": {"memory_usage": 22.0}}) == {"edge-a": 22.0}

    monkeypatch.setattr(overhead_visualizer_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-a"))
    monkeypatch.setattr(overhead_visualizer_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.1"))
    monkeypatch.setattr(
        overhead_visualizer_module.PortInfo,
        "get_component_port",
        staticmethod(lambda component: (_ for _ in ()).throw(AssertionError("missing scheduler"))),
    )
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

    monkeypatch.setattr(dag_deployment_module.KubeConfig, "get_nodes_for_service", staticmethod(lambda service: ["edge-a", "edge-b"]))
    deployment = dag_deployment_module.DAGDeploymentTopologyVisualizer(variables=["topology"])(task)["topology"]
    assert "execute_device" not in deployment["detector"]["service"]
    assert deployment["detector"]["service"]["data"] == "edge-a\nedge-b"

    monkeypatch.setattr(dag_offloading_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-a"))
    offloading = dag_offloading_module.DAGOffloadingTopologyVisualizer(variables=["offloading"])(task)["offloading"]
    assert offloading["detector"]["service"]["data"] == "edge-a"
    assert offloading["classifier"]["service"]["data"] == "cloud-a"

    assert e2e_delay_module.EndToEndDelayVisualizer(variables=["delay"])(task)["delay"] == pytest.approx(1.3)
    assert object_number_visualizer_module.ObjectNumberVisualizer(variables=["obj_num"])(task) == {"obj_num": 3.0}
    assert service_delay_visualizer_module.ServiceProcessingDelayVisualizer(
        variables=["detector", "classifier"]
    )(task) == {"detector": 0.4, "classifier": 0.6}
