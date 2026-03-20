import importlib
import glob
import shutil
import subprocess
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest


available_bandwidth_module = importlib.import_module("core.lib.algorithms.parameter_monitor.available_bandwidth_monitor")
cpu_flops_module = importlib.import_module("core.lib.algorithms.parameter_monitor.cpu_flops_monitor")
gpu_flops_module = importlib.import_module("core.lib.algorithms.parameter_monitor.gpu_flops_monitor")
gpu_usage_module = importlib.import_module("core.lib.algorithms.parameter_monitor.gpu_usage_monitor")


def lscpu_info(flags):
    return {
        "flags": flags,
        "sockets": 1,
        "cores_per_socket": 4,
        "threads_per_core": 2,
        "max_mhz": 1000.0,
        "model_name": "unit-cpu",
    }


@pytest.mark.unit
def test_cpu_flops_monitor_covers_avx512_neon_and_scalar_branches(monkeypatch):
    system = SimpleNamespace(resource_info={})

    monkeypatch.setattr(cpu_flops_module.CPUFlopsMonitor, "parse_lscpu", staticmethod(lambda: lscpu_info(["avx512f"])))
    avx512_value = cpu_flops_module.CPUFlopsMonitor(system).get_parameter_value()

    monkeypatch.setattr(cpu_flops_module.CPUFlopsMonitor, "parse_lscpu", staticmethod(lambda: lscpu_info(["neon"])))
    neon_value = cpu_flops_module.CPUFlopsMonitor(system).get_parameter_value()

    monkeypatch.setattr(cpu_flops_module.CPUFlopsMonitor, "parse_lscpu", staticmethod(lambda: lscpu_info([])))
    scalar_value = cpu_flops_module.CPUFlopsMonitor(system).get_parameter_value()

    assert avx512_value > neon_value
    assert neon_value == scalar_value
    assert scalar_value > 0


@pytest.mark.unit
def test_available_bandwidth_monitor_iperf_server_logs_runtime_errors(monkeypatch):
    warnings = []

    class FakeServer:
        def __init__(self):
            self.bind_address = "0.0.0.0"
            self.port = None
            self.calls = 0

        def run(self):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(error="temporary iperf warning")
            raise KeyboardInterrupt

    monkeypatch.setitem(sys.modules, "iperf3", SimpleNamespace(Server=FakeServer))
    monkeypatch.setattr(available_bandwidth_module.LOGGER, "debug", lambda message: None)
    monkeypatch.setattr(available_bandwidth_module.LOGGER, "warning", lambda message: warnings.append(message))

    with pytest.raises(KeyboardInterrupt):
        available_bandwidth_module.AvailableBandwidthMonitor.iperf_server(5201)

    assert warnings == ["temporary iperf warning"]


@pytest.mark.unit
def test_gpu_usage_monitor_returns_first_available_result_after_exceptions(monkeypatch):
    monitor = gpu_usage_module.GPUUsageMonitor(SimpleNamespace(resource_info={}))

    monkeypatch.setattr(
        gpu_usage_module.GPUUsageMonitor,
        "_get_usage_via_nvml",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("nvml failed"))),
    )
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_nvidia_smi", staticmethod(lambda timeout_sec=1.0: 61))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_jetson_sysfs", staticmethod(lambda: 17))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_tegrastats", staticmethod(lambda: 9))
    assert monitor.get_parameter_value() == 61

    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_nvml", staticmethod(lambda: None))
    monkeypatch.setattr(
        gpu_usage_module.GPUUsageMonitor,
        "_get_usage_via_nvidia_smi",
        staticmethod(lambda timeout_sec=1.0: (_ for _ in ()).throw(RuntimeError("nvidia-smi failed"))),
    )
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_jetson_sysfs", staticmethod(lambda: None))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_tegrastats", staticmethod(lambda: 18))
    assert monitor.get_parameter_value() == 18


@pytest.mark.unit
def test_gpu_usage_helper_methods_cover_empty_devices_invalid_cli_and_scaling(monkeypatch, tmp_path):
    fake_pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvml() == 0

    monkeypatch.setattr(shutil, "which", lambda binary: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="bad\n", stderr=""),
    )
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvidia_smi() is None

    high_load_file = tmp_path / "gpu-load"
    high_load_file.write_text("1500", encoding="utf-8")
    monkeypatch.setattr(glob, "glob", lambda pattern: [str(high_load_file)])
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_jetson_sysfs() == 100.0

    class DummyStdout:
        def readline(self):
            return "RAM 0/0MB"

    class DummyProc:
        def __init__(self):
            self.stdout = DummyStdout()

        def terminate(self):
            return None

    monkeypatch.setattr(shutil, "which", lambda binary: "/usr/bin/tegrastats")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: DummyProc())
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_tegrastats() is None


@pytest.mark.unit
def test_gpu_flops_monitor_covers_pycuda_loading_and_arm_jetson_detection(monkeypatch):
    init_calls = []
    fake_driver = ModuleType("pycuda.driver")
    fake_driver.init = lambda: init_calls.append("init")

    fake_pycuda = ModuleType("pycuda")
    fake_pycuda.driver = fake_driver

    monkeypatch.setitem(sys.modules, "pycuda", fake_pycuda)
    monkeypatch.setitem(sys.modules, "pycuda.driver", fake_driver)
    assert gpu_flops_module.GPUFlopsMonitor.load_pycuda() is fake_driver
    assert init_calls == ["init"]

    monkeypatch.setattr(gpu_flops_module.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        gpu_flops_module.os.path,
        "exists",
        lambda path: path == "/sys/module/tegra_fuse/parameters/tegra_chip_id",
    )
    assert gpu_flops_module.GPUFlopsMonitor.is_jetson_device() is True
