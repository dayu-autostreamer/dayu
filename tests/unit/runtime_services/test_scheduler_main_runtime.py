import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


SCHEDULER_MAIN_PATH = (
    Path(__file__).resolve().parents[3] / "components" / "scheduler" / "main.py"
)
SCHEDULER_DOCKERFILE_PATH = (
    Path(__file__).resolve().parents[3] / "build" / "scheduler.Dockerfile"
)


def install_fake_scheduler_runtime(monkeypatch, *, port=19040):
    run_calls = []
    constructed_servers = []
    app_token = object()

    uvicorn_module = ModuleType("uvicorn")
    uvicorn_module.run = lambda *args, **kwargs: run_calls.append((args, kwargs))

    scheduler_module = ModuleType("core.scheduler")

    class FakeSchedulerServer:
        def __init__(self):
            constructed_servers.append(self)
            self.app = app_token

    scheduler_module.SchedulerServer = FakeSchedulerServer

    core_module = ModuleType("core")
    lib_module = ModuleType("core.lib")
    common_module = ModuleType("core.lib.common")

    class FakeContext:
        @staticmethod
        def get_parameter(name, default=None, direct=True):
            return port if name == "GUNICORN_PORT" else default

    common_module.Context = FakeContext
    core_module.lib = lib_module
    core_module.scheduler = scheduler_module
    lib_module.common = common_module

    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)
    monkeypatch.setitem(sys.modules, "core", core_module)
    monkeypatch.setitem(sys.modules, "core.lib", lib_module)
    monkeypatch.setitem(sys.modules, "core.lib.common", common_module)
    monkeypatch.setitem(sys.modules, "core.scheduler", scheduler_module)

    return app_token, run_calls, constructed_servers


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_scheduler_main_exposes_side_effect_free_app_factory(monkeypatch):
    app_token, run_calls, constructed_servers = install_fake_scheduler_runtime(
        monkeypatch
    )

    module = load_module_from_path(
        SCHEDULER_MAIN_PATH,
        "_test_scheduler_main_runtime",
    )

    assert constructed_servers == []
    assert module.create_app() is app_token
    assert len(constructed_servers) == 1
    assert run_calls == []


@pytest.mark.unit
def test_scheduler_main_runs_one_uvicorn_worker_when_executed(monkeypatch):
    app_token, run_calls, constructed_servers = install_fake_scheduler_runtime(
        monkeypatch,
        port=29090,
    )

    runpy.run_path(str(SCHEDULER_MAIN_PATH), run_name="__main__")

    assert run_calls == [((app_token,), {
        "host": "0.0.0.0",
        "port": 29090,
        "workers": 1,
    })]
    assert len(constructed_servers) == 1


@pytest.mark.unit
def test_scheduler_image_uses_the_direct_uvicorn_entrypoint():
    dockerfile = SCHEDULER_DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert 'CMD ["python3", "main.py"]' in dockerfile
    assert "gunicorn" not in dockerfile.lower()
    assert not (SCHEDULER_MAIN_PATH.parent / "gunicorn.conf.py").exists()
