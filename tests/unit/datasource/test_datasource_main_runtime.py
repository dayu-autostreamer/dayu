import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


DATASOURCE_MAIN_PATH = Path(__file__).resolve().parents[3] / "datasource" / "main.py"


def install_fake_datasource_runtime(monkeypatch):
    events = []
    datasource_server_module = ModuleType("datasource_server")

    class FakeDataSource:
        def __init__(self):
            events.append("init")

        def run(self):
            events.append("run")

    datasource_server_module.DataSource = FakeDataSource
    monkeypatch.setitem(sys.modules, "datasource_server", datasource_server_module)
    return events


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_datasource_main_invokes_datasource_run(monkeypatch):
    events = install_fake_datasource_runtime(monkeypatch)
    module = load_module_from_path(DATASOURCE_MAIN_PATH, "_test_datasource_main_runtime")

    module.main()

    assert events == ["init", "run"]


@pytest.mark.unit
def test_datasource_main_executes_main_when_run_as_script(monkeypatch):
    events = install_fake_datasource_runtime(monkeypatch)

    runpy.run_path(str(DATASOURCE_MAIN_PATH), run_name="__main__")

    assert events == ["init", "run"]
