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
    assert monitor.get_parameter_value() == 0.61

    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_nvml", staticmethod(lambda: None))
    monkeypatch.setattr(
        gpu_usage_module.GPUUsageMonitor,
        "_get_usage_via_nvidia_smi",
        staticmethod(lambda timeout_sec=1.0: (_ for _ in ()).throw(RuntimeError("nvidia-smi failed"))),
    )
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_jetson_sysfs", staticmethod(lambda: None))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_tegrastats", staticmethod(lambda: 18))
    assert monitor._read_instantaneous_percent() == 18
    assert 0.60 <= monitor.get_parameter_value() <= 0.61


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


def configure_jetson_sysfs(monkeypatch, tmp_path, *, soc_id="35", mask="0xf", cur_freq=1_173_000_000):
    soc_file = tmp_path / "soc_id"
    soc_file.write_text(soc_id, encoding="utf-8")
    gpu_root = tmp_path / "17000000.gpu"
    devfreq = gpu_root / "devfreq" / "17000000.gpu"
    devfreq.mkdir(parents=True)
    (gpu_root / "tpc_fs_mask").write_text(mask, encoding="utf-8")
    (devfreq / "cur_freq").write_text(str(cur_freq), encoding="utf-8")
    (devfreq / "max_freq").write_text("1173000000", encoding="utf-8")
    monkeypatch.setattr(gpu_flops_module.GPUFlopsMonitor, "JETSON_SOC_ID_PATHS", (str(soc_file),))
    monkeypatch.setattr(gpu_flops_module.GPUFlopsMonitor, "JETSON_DEVFREQ_PATTERNS", (str(devfreq),))
    return gpu_root, devfreq


@pytest.mark.unit
def test_gpu_flops_monitor_uses_jetson_sysfs_without_pycuda(monkeypatch, tmp_path):
    gpu_root, devfreq = configure_jetson_sysfs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        gpu_flops_module.GPUFlopsMonitor,
        "is_jetson_device",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        gpu_flops_module.GPUFlopsMonitor,
        "load_pycuda",
        staticmethod(lambda: (_ for _ in ()).throw(AssertionError("PyCUDA must not load on Jetson"))),
    )
    monitor = gpu_flops_module.GPUFlopsMonitor(SimpleNamespace(resource_info={}))

    first = monitor.get_parameter_value()
    (devfreq / "cur_freq").write_text("586500000", encoding="utf-8")
    second = monitor.get_parameter_value()

    # T234 mask 0xf = 4 active TPCs; GA10B has 2 SM/TPC and 128 FP32 lanes/SM.
    assert first == pytest.approx(2402.304)
    assert second == pytest.approx(first / 2.0)
    assert monitor._jetson_profile["capability"] == (8, 7)
    assert bin(monitor._parse_tpc_mask((gpu_root / "tpc_fs_mask").read_text())).count("1") == 4


@pytest.mark.unit
@pytest.mark.parametrize(
    ("soc_id", "mask", "expected_gflops"),
    [
        ("33", "0x1", 256.0),   # T210/GM20B: 1 TPC * 1 SM * 128 lanes
        ("24", "0x3", 512.0),   # T186/GP10B: 2 TPC * 1 SM * 128 lanes
        ("25", "0xf", 1024.0),  # T194/GV11B: 4 TPC * 2 SM * 64 lanes
        ("35", "0xff", 4096.0),  # T234/GA10B: 8 TPC * 2 SM * 128 lanes
    ],
)
def test_gpu_flops_monitor_jetson_soc_profiles(monkeypatch, tmp_path, soc_id, mask, expected_gflops):
    configure_jetson_sysfs(
        monkeypatch,
        tmp_path,
        soc_id=soc_id,
        mask=mask,
        cur_freq=1_000_000_000,
    )
    monkeypatch.setattr(
        gpu_flops_module.GPUFlopsMonitor,
        "is_jetson_device",
        staticmethod(lambda: True),
    )

    monitor = gpu_flops_module.GPUFlopsMonitor(SimpleNamespace(resource_info={}))
    assert monitor.get_parameter_value() == pytest.approx(expected_gflops)


@pytest.mark.unit
def test_gpu_flops_monitor_rejects_topology_that_changes_during_both_sampling_attempts(monkeypatch, tmp_path):
    gpu_root, _ = configure_jetson_sysfs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        gpu_flops_module.GPUFlopsMonitor,
        "is_jetson_device",
        staticmethod(lambda: True),
    )
    original_read = gpu_flops_module.GPUFlopsMonitor._read_text
    mask_samples = iter(["0xf", "0x3", "0xf", "0x3"])

    def read_with_changing_mask(path):
        if path == str(gpu_root / "tpc_fs_mask"):
            return next(mask_samples)
        return original_read(path)

    monkeypatch.setattr(
        gpu_flops_module.GPUFlopsMonitor,
        "_read_text",
        staticmethod(read_with_changing_mask),
    )
    monkeypatch.setattr(gpu_flops_module.LOGGER, "warning", lambda message: None)

    monitor = gpu_flops_module.GPUFlopsMonitor(SimpleNamespace(resource_info={}))
    assert monitor.get_parameter_value() is None


@pytest.mark.unit
def test_gpu_flops_monitor_reuses_cached_device_meta_without_reloading(monkeypatch):
    monitor = gpu_flops_module.GPUFlopsMonitor(SimpleNamespace(resource_info={}))
    cached_meta = [{
        "idx": 0,
        "name": "cached-gpu",
        "max_freq_khz": 1_000_000.0,
        "capability": (8, 6),
        "sm_count": 1,
        "fp32_cores_per_sm": 128,
    }]
    monitor._device_meta = cached_meta

    def fail_loader():
        raise AssertionError("loader should not run")

    monkeypatch.setattr(
        gpu_flops_module.GPUFlopsMonitor,
        "load_pycuda",
        staticmethod(fail_loader),
    )

    assert monitor._get_device_meta() is cached_meta
    assert monitor._get_device_meta() is cached_meta


@pytest.mark.unit
def test_gpu_flops_monitor_normalizes_clock_units_and_caps_to_max(monkeypatch):
    monitor = gpu_flops_module.GPUFlopsMonitor(SimpleNamespace(resource_info={}))
    monitor._device_meta = [{
        "idx": 0,
        "name": "desktop-test-gpu",
        "max_freq_khz": 1_000_000.0,
        "capability": (8, 6),
        "sm_count": 1,
        "fp32_cores_per_sm": 128,
    }]
    monkeypatch.setattr(
        gpu_flops_module.GPUFlopsMonitor,
        "_get_current_clocks_via_nvml",
        staticmethod(lambda: [1_500]),  # NVML reports MHz; monitor caps to max PyCUDA clock.
    )

    assert gpu_flops_module.GPUFlopsMonitor._clock_value_to_khz(1_500) == 1_500_000
    assert gpu_flops_module.GPUFlopsMonitor._clock_value_to_khz(1_500_000) == 1_500_000
    assert gpu_flops_module.GPUFlopsMonitor._clock_value_to_khz(1_500_000_000) == 1_500_000
    assert monitor.get_parameter_value() == pytest.approx(
        gpu_flops_module.GPUFlopsMonitor.calculate_flops(1, 128, 1_000_000) / 1e9
    )


@pytest.mark.unit
def test_gpu_flops_monitor_omits_repeated_failures_and_recovers(monkeypatch, tmp_path):
    configure_jetson_sysfs(monkeypatch, tmp_path, soc_id="999")
    monkeypatch.setattr(
        gpu_flops_module.GPUFlopsMonitor,
        "is_jetson_device",
        staticmethod(lambda: True),
    )
    warnings = []
    infos = []
    monkeypatch.setattr(gpu_flops_module.LOGGER, "warning", warnings.append)
    monkeypatch.setattr(gpu_flops_module.LOGGER, "info", infos.append)
    monitor = gpu_flops_module.GPUFlopsMonitor(SimpleNamespace(resource_info={}))

    assert monitor.get_parameter_value() is None
    assert monitor.get_parameter_value() is None
    assert len(warnings) == 1

    (tmp_path / "soc_id").write_text("35", encoding="utf-8")
    assert monitor.get_parameter_value() == pytest.approx(2402.304)
    assert infos == ["GPU FLOPS monitor recovered"]


@pytest.mark.unit
def test_gpu_flops_monitor_keeps_last_valid_sample_when_sysfs_is_temporarily_unavailable(monkeypatch, tmp_path):
    _, devfreq = configure_jetson_sysfs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        gpu_flops_module.GPUFlopsMonitor,
        "is_jetson_device",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(gpu_flops_module.LOGGER, "warning", lambda message: None)
    timestamps = iter([10.0, 20.0, 26.0])
    monkeypatch.setattr(gpu_flops_module.time, "monotonic", lambda: next(timestamps))
    system = SimpleNamespace(resource_info={})
    monitor = gpu_flops_module.GPUFlopsMonitor(system)

    monitor.run_monitor(system)
    assert system.resource_info["gpu_flops"] == pytest.approx(2402.304)

    (devfreq / "cur_freq").write_text("0", encoding="utf-8")
    monitor.run_monitor(system)
    assert system.resource_info["gpu_flops"] == pytest.approx(2402.304)

    monitor.run_monitor(system)
    assert "gpu_flops" not in system.resource_info
