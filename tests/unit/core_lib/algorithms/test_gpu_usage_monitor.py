import builtins
import glob
import importlib
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


gpu_usage_module = importlib.import_module("core.lib.algorithms.parameter_monitor.gpu_usage_monitor")


@pytest.mark.unit
def test_gpu_usage_monitor_returns_zero_and_warns_when_all_backends_fail(monkeypatch):
    warnings = []
    monitor = gpu_usage_module.GPUUsageMonitor(SimpleNamespace(resource_info={}))

    monkeypatch.setattr(gpu_usage_module.LOGGER, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_nvml", staticmethod(lambda: None))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_nvidia_smi", staticmethod(lambda timeout_sec=1.0: None))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_jetson_sysfs", staticmethod(lambda: None))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_tegrastats", staticmethod(lambda: None))

    assert monitor.get_parameter_value() == 0
    assert warnings == ["[GPUUsage] Unable to determine GPU usage, returning 0."]


@pytest.mark.unit
def test_gpu_usage_helpers_cover_import_errors_nvml_success_and_cli_edges(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("no pynvml installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvml() is None

    shutdown_calls = []

    class FakePynvml:
        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlShutdown():
            shutdown_calls.append("shutdown")
            raise RuntimeError("shutdown failed")

        @staticmethod
        def nvmlDeviceGetCount():
            return 2

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):
            return index

        @staticmethod
        def nvmlDeviceGetUtilizationRates(handle):
            return SimpleNamespace(gpu=(44, 91)[handle])

    monkeypatch.setattr(builtins, "__import__", real_import)
    monkeypatch.setitem(sys.modules, "pynvml", FakePynvml)
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvml() == 91
    assert shutdown_calls == ["shutdown"]

    monkeypatch.setattr(shutil, "which", lambda binary: None)
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvidia_smi() is None

    monkeypatch.setattr(shutil, "which", lambda binary: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="failed"),
    )
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvidia_smi() is None

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="120\n40\n", stderr=""),
    )
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvidia_smi() == 100

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("nvidia-smi unavailable")),
    )
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvidia_smi() is None


@pytest.mark.unit
def test_gpu_usage_helpers_cover_jetson_sysfs_and_tegrastats_paths(monkeypatch, tmp_path):
    percent_file = tmp_path / "gpu-percent.load"
    percent_file.write_text("88", encoding="utf-8")
    per_mille_file = tmp_path / "gpu-per-mille.load"
    per_mille_file.write_text("450", encoding="utf-8")
    overflow_file = tmp_path / "gpu-overflow.load"
    overflow_file.write_text("1500", encoding="utf-8")
    empty_file = tmp_path / "gpu-empty.load"
    empty_file.write_text("", encoding="utf-8")
    invalid_file = tmp_path / "gpu-invalid.load"
    invalid_file.write_text("not-a-number", encoding="utf-8")

    monkeypatch.setattr(
        glob,
        "glob",
        lambda pattern: [
            str(percent_file),
            str(per_mille_file),
            str(overflow_file),
            str(empty_file),
            str(invalid_file),
            str(percent_file),
        ],
    )
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_jetson_sysfs() == 100.0

    class DummyStdout:
        def readline(self):
            return "RAM 0/0MB GR3D_FREQ 88% cpu"

    class DummyProc:
        def __init__(self):
            self.stdout = DummyStdout()

        def terminate(self):
            raise RuntimeError("terminate failed")

    monkeypatch.setattr(shutil, "which", lambda binary: "/usr/bin/tegrastats")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: DummyProc())
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_tegrastats() == 88

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("tegrastats unavailable")),
    )
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_tegrastats() is None

    monkeypatch.setattr(shutil, "which", lambda binary: None)
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_tegrastats() is None

    class EmptyStdoutProc:
        def __init__(self):
            self.stdout = SimpleNamespace(readline=lambda: "")

        def terminate(self):
            return None

    monkeypatch.setattr(shutil, "which", lambda binary: "/usr/bin/tegrastats")
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: EmptyStdoutProc())
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_tegrastats() is None

    monkeypatch.setattr(glob, "glob", lambda pattern: [])
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_jetson_sysfs() is None

    class NoDevicePynvml:
        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlShutdown():
            return None

        @staticmethod
        def nvmlDeviceGetCount():
            return 0

    monkeypatch.setitem(sys.modules, "pynvml", NoDevicePynvml)
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvml() == 0

    class BrokenPynvml:
        @staticmethod
        def nvmlInit():
            raise RuntimeError("init failed")

    monkeypatch.setitem(sys.modules, "pynvml", BrokenPynvml)
    assert gpu_usage_module.GPUUsageMonitor._get_usage_via_nvml() is None


@pytest.mark.unit
def test_gpu_usage_monitor_returns_first_non_none_backend_value(monkeypatch):
    monitor = gpu_usage_module.GPUUsageMonitor(SimpleNamespace(resource_info={}))

    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_nvml", staticmethod(lambda: 0))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_nvidia_smi", staticmethod(lambda timeout_sec=1.0: 66))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_jetson_sysfs", staticmethod(lambda: 77))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_tegrastats", staticmethod(lambda: 88))
    assert monitor.get_parameter_value() == 0

    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_nvml", staticmethod(lambda: None))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_nvidia_smi", staticmethod(lambda timeout_sec=1.0: None))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_jetson_sysfs", staticmethod(lambda: 27))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_tegrastats", staticmethod(lambda: 88))
    assert monitor.get_parameter_value() == 27

    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_jetson_sysfs", staticmethod(lambda: None))
    monkeypatch.setattr(gpu_usage_module.GPUUsageMonitor, "_get_usage_via_tegrastats", staticmethod(lambda: 18))
    assert monitor.get_parameter_value() == 18


@pytest.mark.unit
def test_gpu_usage_monitor_ignores_backend_exceptions_and_keeps_fallback_order(monkeypatch):
    warnings = []
    monitor = gpu_usage_module.GPUUsageMonitor(SimpleNamespace(resource_info={}))

    monkeypatch.setattr(gpu_usage_module.LOGGER, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(
        gpu_usage_module.GPUUsageMonitor,
        "_get_usage_via_nvml",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("nvml failed"))),
    )
    monkeypatch.setattr(
        gpu_usage_module.GPUUsageMonitor,
        "_get_usage_via_nvidia_smi",
        staticmethod(lambda timeout_sec=1.0: (_ for _ in ()).throw(RuntimeError("smi failed"))),
    )
    monkeypatch.setattr(
        gpu_usage_module.GPUUsageMonitor,
        "_get_usage_via_jetson_sysfs",
        staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("sysfs failed"))),
    )
    monkeypatch.setattr(
        gpu_usage_module.GPUUsageMonitor,
        "_get_usage_via_tegrastats",
        staticmethod(lambda: 33),
    )

    assert monitor.get_parameter_value() == 33
    assert warnings == []
