import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


BACKEND_MAIN_PATH = Path(__file__).resolve().parents[3] / "backend" / "main.py"


def install_fake_backend_runtime(monkeypatch, *, port=19010):
    run_calls = []
    app_token = object()

    uvicorn_module = ModuleType("uvicorn")
    uvicorn_module.run = lambda *args, **kwargs: run_calls.append((args, kwargs))

    backend_server_module = ModuleType("backend_server")

    class FakeBackendServer:
        def __init__(self):
            self.app = app_token

    backend_server_module.BackendServer = FakeBackendServer

    core_module = ModuleType("core")
    lib_module = ModuleType("core.lib")
    common_module = ModuleType("core.lib.common")

    class FakeContext:
        @staticmethod
        def get_parameter(name, default=None, direct=True):
            return port if name == "GUNICORN_PORT" else default

    common_module.Context = FakeContext
    core_module.lib = lib_module
    lib_module.common = common_module

    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)
    monkeypatch.setitem(sys.modules, "backend_server", backend_server_module)
    monkeypatch.setitem(sys.modules, "core", core_module)
    monkeypatch.setitem(sys.modules, "core.lib", lib_module)
    monkeypatch.setitem(sys.modules, "core.lib.common", common_module)

    return app_token, run_calls


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_backend_main_exposes_app_without_starting_server(monkeypatch):
    app_token, run_calls = install_fake_backend_runtime(monkeypatch)

    module = load_module_from_path(BACKEND_MAIN_PATH, "_test_backend_main_runtime")

    assert module.app is app_token
    assert run_calls == []


@pytest.mark.unit
def test_backend_main_runs_uvicorn_with_context_port_when_executed_as_script(monkeypatch):
    app_token, run_calls = install_fake_backend_runtime(monkeypatch, port=28080)

    runpy.run_path(str(BACKEND_MAIN_PATH), run_name="__main__")

    assert run_calls == [((app_token,), {"host": "0.0.0.0", "port": 28080})]
