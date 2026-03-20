import importlib.util
import sys
from pathlib import Path

import pytest


PROCESSOR_INIT_PATH = (
    Path(__file__).resolve().parents[3]
    / "dependency"
    / "core"
    / "processor"
    / "__init__.py"
)


def load_processor_package(monkeypatch, package_name, import_behavior):
    original_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if package == package_name and name.startswith("."):
            return import_behavior(name)
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    spec = importlib.util.spec_from_file_location(
        package_name,
        PROCESSOR_INIT_PATH,
        submodule_search_locations=[str(PROCESSOR_INIT_PATH.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(package_name, None)
    return module


@pytest.mark.unit
def test_processor_package_skips_optional_dependencies_with_warning(monkeypatch):
    def import_behavior(name):
        if name == ".processor_server":
            return type("ServerModule", (), {"ProcessorServer": object})
        raise ModuleNotFoundError("optional missing", name="ultralytics")

    with pytest.warns(RuntimeWarning, match="Skip loading optional processor"):
        module = load_processor_package(monkeypatch, "dayu_test_processor_pkg_optional", import_behavior)

    assert module.__all__ == ["ProcessorServer"]


@pytest.mark.unit
def test_processor_package_re_raises_core_import_errors(monkeypatch):
    def import_behavior(name):
        raise ModuleNotFoundError("core missing", name="core.lib.content")

    with pytest.raises(ModuleNotFoundError, match="core missing"):
        load_processor_package(monkeypatch, "dayu_test_processor_pkg_core", import_behavior)
