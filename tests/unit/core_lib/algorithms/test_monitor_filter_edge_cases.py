import importlib
import glob
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


casva_bsto_module = importlib.import_module("core.lib.algorithms.before_submit_task_operation.casva_operation")
dynamic_filter_module = importlib.import_module("core.lib.algorithms.frame_filter.dynamic_filter")
available_bandwidth_module = importlib.import_module("core.lib.algorithms.parameter_monitor.available_bandwidth_monitor")
cpu_flops_module = importlib.import_module("core.lib.algorithms.parameter_monitor.cpu_flops_monitor")
gpu_flops_module = importlib.import_module("core.lib.algorithms.parameter_monitor.gpu_flops_monitor")
gpu_usage_module = importlib.import_module("core.lib.algorithms.parameter_monitor.gpu_usage_monitor")


@pytest.mark.unit
def test_casva_before_submit_operation_covers_ffmpeg_paths_and_frame_modes(monkeypatch, tmp_path):
    operation = casva_bsto_module.CASVABSTOperation()
    warnings = []
    exceptions = []
    moved = []

    monkeypatch.setattr(casva_bsto_module.LOGGER, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(casva_bsto_module.LOGGER, "exception", lambda message: exceptions.append(str(message)))
    monkeypatch.setattr(casva_bsto_module.LOGGER, "debug", lambda message: None)

    operation.modify_file_qp({}, tmp_path / "clip.mp4")

    monkeypatch.setattr(
        casva_bsto_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="ffmpeg failed"),
    )
    operation.modify_file_qp({"qp": 23}, tmp_path / "clip.mp4")

    monkeypatch.setattr(
        casva_bsto_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr=""),
    )
    monkeypatch.setattr(casva_bsto_module.shutil, "move", lambda src, dst: moved.append((src, dst)))
    operation.modify_file_qp({"qp": 20}, "final.mp4")

    monkeypatch.setattr(
        casva_bsto_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("ffmpeg missing")),
    )
    operation.modify_file_qp({"qp": 18}, "broken.mp4")

    monkeypatch.setattr(casva_bsto_module.os.path, "getsize", lambda file_path: 1024)
    assert operation.reprocess_data(
        "compressed.mp4",
        {"resolution": "720p", "fps": 10},
        {"resolution": "480p", "fps": 5},
    ) == pytest.approx(256.0)

    skip_filter = casva_bsto_module.CASVABSTOperation()
    assert [skip_filter.filter_frame(20, 15) for _ in range(4)] == [True, True, True, False]

    remain_filter = casva_bsto_module.CASVABSTOperation()
    assert [remain_filter.filter_frame(20, 4) for _ in range(5)] == [False, False, False, False, True]

    assert any("'qp' not found" in message for message in warnings)
    assert any("ffmpeg failed" in message for message in warnings)
    assert moved == [("tmp.mp4", "final.mp4")]
    assert exceptions == ["An error occurred while compressing broken.mp4: ffmpeg missing"]


@pytest.mark.unit
def test_dynamic_filter_transitions_cycles_and_processing_gates(monkeypatch):
    uniform_values = iter([1.0, 4.0, 10.0, 2.0, 5.0, 12.0])
    monkeypatch.setattr(dynamic_filter_module.random, "uniform", lambda start, end: next(uniform_values))
    monkeypatch.setattr(dynamic_filter_module.time, "time", lambda: 100.0)

    dynamic_filter = dynamic_filter_module.DynamicFilter(min_fps=1, max_fps=5, min_duration=4, max_duration=4)
    assert dynamic_filter.current_fps_range == {"min_fps": 1.0, "max_fps": 4.0}
    assert dynamic_filter.current_cycle_duration == 10.0

    dynamic_filter._update_cycle_state(109.5)
    assert dynamic_filter.is_transitioning is True
    assert dynamic_filter.next_fps_range == {"min_fps": 2.0, "max_fps": 5.0}
    assert dynamic_filter._calculate_target_fps(109.75) > 0

    dynamic_filter._update_cycle_state(110.1)
    assert dynamic_filter.is_transitioning is False
    assert dynamic_filter.current_fps_range == {"min_fps": 2.0, "max_fps": 5.0}
    assert dynamic_filter.current_cycle_duration == 12.0

    time_values = iter([200.0, 200.1, 200.8])
    monkeypatch.setattr(dynamic_filter_module.time, "time", lambda: next(time_values))
    monkeypatch.setattr(dynamic_filter, "_update_cycle_state", lambda current_time: None)
    monkeypatch.setattr(dynamic_filter, "_calculate_target_fps", lambda current_time: 2.0)
    assert dynamic_filter(SimpleNamespace(), np.zeros((2, 2, 3), dtype=np.uint8)) is True
    assert dynamic_filter(SimpleNamespace(), np.zeros((2, 2, 3), dtype=np.uint8)) is False
    assert dynamic_filter(SimpleNamespace(), np.zeros((2, 2, 3), dtype=np.uint8)) is True


@pytest.mark.unit
def test_available_bandwidth_monitor_retries_permission_and_handles_client_errors(monkeypatch):
    permission_monitor = available_bandwidth_module.AvailableBandwidthMonitor.__new__(
        available_bandwidth_module.AvailableBandwidthMonitor
    )
    permission_monitor.local_device = "edge-a"

    responses = iter([None, None, {"holder": "edge-a"}])
    monkeypatch.setattr(available_bandwidth_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-a"))
    monkeypatch.setattr(available_bandwidth_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.1"))
    monkeypatch.setattr(available_bandwidth_module.PortInfo, "get_component_port", staticmethod(lambda component: 5201))
    monkeypatch.setattr(available_bandwidth_module, "merge_address", lambda *args, **kwargs: "http://scheduler/lock")
    monkeypatch.setattr(available_bandwidth_module, "http_request", lambda *args, **kwargs: next(responses))
    permission_monitor.request_for_bandwidth_permission()
    assert permission_monitor.permitted_device == "edge-a"

    warnings = []
    exceptions = []
    monkeypatch.setattr(available_bandwidth_module.LOGGER, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(available_bandwidth_module.LOGGER, "exception", lambda message: exceptions.append(str(message)))

    class ErrorClient:
        def __init__(self):
            self.duration = None
            self.server_hostname = None
            self.port = None
            self.protocol = None

        def run(self):
            return SimpleNamespace(error="bandwidth error", sent_Mbps=99.0)

    client_monitor = available_bandwidth_module.AvailableBandwidthMonitor.__new__(
        available_bandwidth_module.AvailableBandwidthMonitor
    )
    client_monitor.is_server = False
    client_monitor.local_device = "edge-a"
    client_monitor.permitted_device = "edge-a"
    client_monitor.iperf3_server_ip = "10.0.0.2"
    client_monitor.iperf3_port = 5201
    monkeypatch.setitem(sys.modules, "iperf3", SimpleNamespace(Client=ErrorClient))
    assert client_monitor.get_parameter_value() == 0

    class ExplodingClient(ErrorClient):
        def run(self):
            raise RuntimeError("iperf crashed")

    monkeypatch.setitem(sys.modules, "iperf3", SimpleNamespace(Client=ExplodingClient))
    assert client_monitor.get_parameter_value() == 0
    assert any("bandwidth error" in message for message in warnings)
    assert any("iperf crashed" in message for message in exceptions)


@pytest.mark.unit
def test_cpu_gpu_flops_and_gpu_usage_monitors_cover_helper_fallbacks(monkeypatch, tmp_path):
    sample_lscpu = """
Model name: Unit CPU
Socket(s): 2
Core(s) per socket: 4
Thread(s) per core: 2
CPU max MHz: 3200.0
Flags: avx2 fma
"""
    monkeypatch.setattr(cpu_flops_module.subprocess, "check_output", lambda cmd, text=True: sample_lscpu)
    parsed = cpu_flops_module.CPUFlopsMonitor.parse_lscpu()
    assert parsed["model_name"] == "Unit CPU"
    assert parsed["sockets"] == 2
    assert parsed["cores_per_socket"] == 4

    monkeypatch.setattr(gpu_flops_module.os.path, "exists", lambda path: path == "/etc/nv_tegra_release")
    assert gpu_flops_module.GPUFlopsMonitor.is_jetson_device() is True
    monkeypatch.setattr(gpu_flops_module.os.path, "exists", lambda path: False)
    monkeypatch.setattr(gpu_flops_module.platform, "machine", lambda: "x86_64")
    assert gpu_flops_module.GPUFlopsMonitor.is_jetson_device() is False

    class FakeDevice:
        def __init__(self, capability=(8, 6)):
            self._capability = capability

        def name(self):
            return "RTX"

        def get_attribute(self, attr):
            if attr == "CLOCK_RATE":
                return 1_500_000
            return 2

        def compute_capability(self):
            return self._capability

    class FakeDeviceFactory:
        @staticmethod
        def count():
            return 1

        def __call__(self, idx):
            return FakeDevice()

    fake_cuda = SimpleNamespace(
        Device=FakeDeviceFactory(),
        device_attribute=SimpleNamespace(CLOCK_RATE="CLOCK_RATE", MULTIPROCESSOR_COUNT="MULTIPROCESSOR_COUNT"),
    )
    monkeypatch.setattr(gpu_flops_module.GPUFlopsMonitor, "load_pycuda", staticmethod(lambda: fake_cuda))
    monkeypatch.setattr(gpu_flops_module.GPUFlopsMonitor, "is_jetson_device", staticmethod(lambda: False))
    monitor = gpu_flops_module.GPUFlopsMonitor(SimpleNamespace(resource_info={}))
    assert monitor.get_parameter_value() > 0
    assert gpu_flops_module.GPUFlopsMonitor.calculate_flops(2, 128, 1_500_000, 2) > 0

    unsupported_cuda = SimpleNamespace(
        Device=type(
            "UnsupportedFactory",
            (),
            {
                "count": staticmethod(lambda: 1),
                "__call__": staticmethod(lambda idx: FakeDevice((9, 9))),
            },
        )(),
        device_attribute=SimpleNamespace(CLOCK_RATE="CLOCK_RATE", MULTIPROCESSOR_COUNT="MULTIPROCESSOR_COUNT"),
    )
    monkeypatch.setattr(gpu_flops_module.GPUFlopsMonitor, "load_pycuda", staticmethod(lambda: unsupported_cuda))
    with pytest.raises(Exception, match="Unsupported device"):
        monitor.get_device_fp32_flops(False)

    fake_pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=[12, 34][handle]),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvml() == 34

    monkeypatch.setattr(shutil, "which", lambda binary: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="40\n110\nbad\n", stderr=""),
    )
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvidia_smi() == 100
    monkeypatch.setattr(shutil, "which", lambda binary: None)
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvidia_smi() is None

    load_a = tmp_path / "gpu-load-a"
    load_b = tmp_path / "gpu-load-b"
    load_a.write_text("750", encoding="utf-8")
    load_b.write_text("50", encoding="utf-8")
    monkeypatch.setattr(glob, "glob", lambda pattern: [str(load_a), str(load_b)])
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_jetson_sysfs() == 75.0

    class DummyStdout:
        def readline(self):
            return "RAM 0/0MB GR3D_FREQ 55%"

    class DummyProc:
        def __init__(self):
            self.stdout = DummyStdout()

        def terminate(self):
            return None

    monkeypatch.setattr(shutil, "which", lambda binary: "/usr/bin/tegrastats")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: DummyProc())
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_tegrastats() == 55

    warnings = []
    monkeypatch.setattr(gpu_usage_module.LOGGER, "warning", lambda message: warnings.append(message))
    fallback_monitor = gpu_usage_module.GPUUsageMonitor(SimpleNamespace(resource_info={}))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_nvml", staticmethod(lambda: None))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_nvidia_smi", staticmethod(lambda timeout_sec=1.0: None))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_jetson_sysfs", staticmethod(lambda: None))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_tegrastats", staticmethod(lambda: None))
    assert fallback_monitor.get_parameter_value() == 0
    assert any("Unable to determine GPU usage" in message for message in warnings)
