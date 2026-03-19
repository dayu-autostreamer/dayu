import importlib
import importlib.util
import pkgutil
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest


class_factory_module = importlib.import_module("core.lib.common.class_factory")
context_module = importlib.import_module("core.lib.common.context")

ClassFactory = class_factory_module.ClassFactory
ClassType = class_factory_module.ClassType
Context = context_module.Context
FileNotMountedError = context_module.FileNotMountedError


@pytest.fixture(autouse=True)
def restore_class_registry():
    snapshot = {
        type_name: registry.copy()
        for type_name, registry in ClassFactory.__registry__.items()
    }
    yield
    ClassFactory.__registry__.clear()
    ClassFactory.__registry__.update(
        {type_name: registry.copy() for type_name, registry in snapshot.items()}
    )


def load_algorithms_module(monkeypatch, packages, import_side_effect):
    module_path = (
        Path(__file__).resolve().parents[2]
        / "dependency"
        / "core"
        / "lib"
        / "algorithms"
        / "__init__.py"
    )
    module_name = f"_test_algorithms_loader_{uuid4().hex}"

    monkeypatch.setattr(pkgutil, "iter_modules", lambda path: iter(packages))
    monkeypatch.setattr(importlib, "import_module", import_side_effect)

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
        submodule_search_locations=[str(module_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec.loader.exec_module(module)

    return module, caught


@pytest.mark.unit
def test_class_factory_register_supports_alias_lookup_and_duplicate_guards():
    type_name = f"test_type_{uuid4().hex}"

    @ClassFactory.register(type_name, alias="demo")
    class DemoHook:
        pass

    assert ClassFactory.is_exists(type_name)
    assert ClassFactory.is_exists(type_name, "demo")
    assert ClassFactory.get_cls(type_name, "demo") is DemoHook

    with pytest.raises(ValueError, match="Cannot register duplicate class"):

        @ClassFactory.register(type_name, alias="demo")
        class DuplicateHook:
            pass

    with pytest.raises(ValueError, match="can't find class\\. class type="):
        ClassFactory.get_cls(type_name)

    with pytest.raises(ValueError, match="can't find class type"):
        ClassFactory.get_cls(type_name, "missing")


@pytest.mark.unit
def test_class_factory_register_cls_and_register_from_package_skip_private_symbols():
    type_name = f"package_type_{uuid4().hex}"

    class PackageClass:
        pass

    def helper():
        return "ok"

    package = SimpleNamespace(
        PackageClass=PackageClass,
        helper=helper,
        _hidden=object(),
        value=3,
    )

    ClassFactory.register_from_package(package, type_name)

    assert ClassFactory.get_cls(type_name, "PackageClass") is PackageClass
    assert ClassFactory.get_cls(type_name, "helper") is helper
    assert not ClassFactory.is_exists(type_name, "_hidden")
    assert not ClassFactory.is_exists(type_name, "value")

    class ExtraClass:
        pass

    ClassFactory.register_cls(ExtraClass, type_name, alias="extra")
    assert ClassFactory.get_cls(type_name, "extra") is ExtraClass

    with pytest.raises(ValueError, match="Cannot register duplicate class"):
        ClassFactory.register_cls(ExtraClass, type_name, alias="extra")


@pytest.mark.unit
def test_context_resolves_algorithms_with_env_parameters_and_literal_fallbacks(monkeypatch):
    type_name = f"hook_type_{uuid4().hex}"
    monkeypatch.setattr(ClassType, "TEST_HOOK", type_name, raising=False)

    class DemoAlgorithm:
        def __init__(self, alpha=0, beta=0):
            self.alpha = alpha
            self.beta = beta

    ClassFactory.register_cls(DemoAlgorithm, type_name, alias="demo")
    monkeypatch.setenv("TEST_HOOK_NAME", "demo")
    monkeypatch.setenv("TEST_HOOK_PARAMETERS", "{'alpha': 1}")

    instance = Context.get_algorithm("test_hook", beta=2)

    assert isinstance(instance, DemoAlgorithm)
    assert (instance.alpha, instance.beta) == (1, 2)
    assert Context.get_algorithm_info("TEST_HOOK", None, beta=3) == {
        "method": "demo",
        "param": {"alpha": 1, "beta": 3},
    }

    monkeypatch.setenv("BROKEN_LITERAL", "{oops")
    assert Context.get_parameter("BROKEN_LITERAL", direct=False) == "{oops"

    monkeypatch.delenv("TEST_HOOK_NAME", raising=False)
    assert Context.get_algorithm("test_hook") is None


@pytest.mark.unit
def test_context_file_path_resolution_handles_mounted_volumes_and_temp_files(monkeypatch, tmp_path):
    mount_prefix = tmp_path / "mnt"
    raw_prefix = tmp_path / "raw"
    volume0 = raw_prefix / "templates"
    volume1 = raw_prefix / "datasets"

    monkeypatch.setenv("DATA_PATH_PREFIX", str(mount_prefix))
    monkeypatch.setenv("FILE_PREFIX", str(raw_prefix))
    monkeypatch.setenv("VOLUME_NUM", "3")
    monkeypatch.setenv("VOLUME_0", str(volume0))
    monkeypatch.setenv("VOLUME_1", str(volume1))

    assert Context.get_file_path(1) == str(mount_prefix / "volume1")
    assert Context.get_file_path(str(volume1 / "clips" / "clip.mp4")) == str(
        mount_prefix / "volume1" / "clips" / "clip.mp4"
    )
    assert Context.get_file_path("datasets/clips/clip.mp4") == str(
        mount_prefix / "volume1" / "clips" / "clip.mp4"
    )

    with pytest.raises(IndexError, match="out of range"):
        Context.get_file_path(5)

    with pytest.raises(FileNotMountedError, match="not mounted"):
        Context.get_file_path("/outside/runtime.mp4")

    monkeypatch.setenv("VOLUME_NUM", "2")
    monkeypatch.setenv("VOLUME_0", str(volume0))
    assert Context.get_file_path("relative/file.txt") == str(
        mount_prefix / "volume0" / "relative" / "file.txt"
    )

    temp_path = Path(Context.get_temporary_file_path("payload.bin"))
    assert temp_path == mount_prefix / "volume1" / "temp_files" / "payload.bin"
    assert temp_path.parent.exists()

    monkeypatch.setenv("VOLUME_NUM", "1")
    with pytest.raises(FileNotMountedError, match="No file directory is mounted"):
        Context.get_file_path("anything.txt")


@pytest.mark.unit
def test_context_get_instance_uses_module_globals_and_reports_missing_classes(monkeypatch):
    class DummyService:
        def __init__(self, alpha, beta=0):
            self.alpha = alpha
            self.beta = beta

    monkeypatch.setattr(context_module, "DummyService", DummyService, raising=False)
    monkeypatch.setenv("DUMMYSERVICE_PARAMETERS", "{'beta': 4}")

    instance = Context.get_instance("DummyService", alpha=1)

    assert isinstance(instance, DummyService)
    assert (instance.alpha, instance.beta) == (1, 4)

    with pytest.raises(ValueError, match="Class 'MissingService' is not defined or imported."):
        Context.get_instance("MissingService")


@pytest.mark.unit
def test_algorithm_autoloader_registers_packages_and_skips_optional_dependencies(monkeypatch):
    loaded_package = ModuleType("good_pkg")

    def fake_import_module(name, package=None):
        if name == ".good_pkg":
            return loaded_package
        if name == ".optional_pkg":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        raise AssertionError(f"Unexpected import: {name!r} from {package!r}")

    module, caught = load_algorithms_module(
        monkeypatch,
        packages=[
            (None, "good_pkg", True),
            (None, "optional_pkg", True),
            (None, "plain_file", False),
        ],
        import_side_effect=fake_import_module,
    )

    assert module.good_pkg is loaded_package
    assert module.__all__ == ["good_pkg"]
    assert len(caught) == 1
    assert "optional_pkg" in str(caught[0].message)
    assert "torch" in str(caught[0].message)


@pytest.mark.unit
def test_algorithm_autoloader_reraises_missing_core_dependencies(monkeypatch):
    def fake_import_module(name, package=None):
        raise ModuleNotFoundError(
            "No module named 'core.lib.common'",
            name="core.lib.common",
        )

    with pytest.raises(ModuleNotFoundError, match="core\\.lib\\.common"):
        load_algorithms_module(
            monkeypatch,
            packages=[(None, "broken_pkg", True)],
            import_side_effect=fake_import_module,
        )
