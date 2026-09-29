import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.lib.algorithms import loader


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
ALGORITHM_ROOT = REPOSITORY_ROOT / "dependency" / "core" / "lib" / "algorithms"
HOOK_FAMILIES = (
    "after_schedule_operation",
    "before_schedule_operation",
    "before_submit_task_operation",
    "data_getter",
    "data_getter_filter",
    "frame_compress",
    "frame_filter",
    "frame_process",
    "parameter_monitor",
    "result_visualizer",
    "scenario_extraction",
    "schedule_agent",
    "schedule_config_extraction",
    "schedule_initial_deployment_policy",
    "schedule_policy_retrieval",
    "schedule_redeployment_policy",
    "schedule_scenario_retrieval",
    "schedule_selection_policy",
    "schedule_startup_policy",
    "system_visualizer",
    "task_queue",
)

HOOK_PACKAGE_SUFFIXES = {
    "schedule_agent": "_agent",
    "schedule_config_extraction": "_config_extraction",
    "schedule_initial_deployment_policy": "_initial_deployment_policy",
    "schedule_redeployment_policy": "_redeployment_policy",
}


@pytest.mark.unit
def test_hook_loader_imports_public_files_and_explicit_package_entries(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "base.py").write_text("", encoding="utf-8")
    (tmp_path / "visible.py").write_text("", encoding="utf-8")
    (tmp_path / "_private.py").write_text("", encoding="utf-8")
    package = tmp_path / "family"
    package.mkdir()
    (package / "hook.py").write_text("", encoding="utf-8")
    ignored = tmp_path / "implementation_only"
    ignored.mkdir()
    (ignored / "__init__.py").write_text("", encoding="utf-8")

    main = object()
    variant = object()
    visible = object()
    modules = {
        "example.base": SimpleNamespace(),
        "example.family.hook": SimpleNamespace(
            __all__=("Main", "Variant"),
            Main=main,
            Variant=variant,
        ),
        "example.visible": SimpleNamespace(
            __all__=("Visible",),
            Visible=visible,
        ),
    }
    imported = []

    def import_module(name):
        imported.append(name)
        return modules[name]

    monkeypatch.setattr(loader.importlib, "import_module", import_module)

    exports = loader.load_hooks("example", [str(tmp_path)])

    assert imported == [
        "example.base",
        "example.family.hook",
        "example.visible",
    ]
    assert exports == {
        "Main": main,
        "Variant": variant,
        "Visible": visible,
    }


@pytest.mark.unit
def test_hook_loader_only_skips_missing_external_optional_dependencies(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "optional.py").write_text("", encoding="utf-8")

    def missing_external(_):
        raise ModuleNotFoundError(
            "No module named 'optional_runtime'",
            name="optional_runtime",
        )

    monkeypatch.setattr(loader.importlib, "import_module", missing_external)

    with pytest.warns(
        RuntimeWarning,
        match="optional example hook .*optional_runtime",
    ):
        assert loader.load_hooks(
            "example",
            [str(tmp_path)],
            optional_dependencies=True,
            warning_subject="example hook",
        ) == {}

    def missing_internal(_):
        raise ModuleNotFoundError(
            "No module named 'core.missing'",
            name="core.missing",
        )

    monkeypatch.setattr(loader.importlib, "import_module", missing_internal)
    with pytest.raises(ModuleNotFoundError, match="core.missing"):
        loader.load_hooks(
            "example",
            [str(tmp_path)],
            optional_dependencies=True,
        )


@pytest.mark.unit
def test_repository_hook_families_have_no_undiscovered_source_entries():
    violations = []

    for family_name in HOOK_FAMILIES:
        family = ALGORITHM_ROOT / family_name
        init_source = (family / "__init__.py").read_text(encoding="utf-8")
        if "load_hooks" not in init_source:
            violations.append(f"{family_name}/__init__.py does not use load_hooks")

        for entry in family.iterdir():
            if (
                entry.is_file()
                and entry.suffix == ".py"
                and entry.name != "__init__.py"
                and entry.stem.startswith("_")
            ):
                violations.append(f"{family_name}/{entry.name} is not discovered")

            if (
                entry.is_dir()
                and any(entry.rglob("*.py"))
                and not (entry / "hook.py").is_file()
            ):
                violations.append(f"{family_name}/{entry.name} has no hook.py")

            package_suffix = HOOK_PACKAGE_SUFFIXES.get(family_name)
            if (
                entry.is_dir()
                and (entry / "hook.py").is_file()
                and package_suffix
                and not entry.name.endswith(package_suffix)
            ):
                violations.append(
                    f"{family_name}/{entry.name} does not end with "
                    f"{package_suffix}"
                )

            if (
                entry.is_dir()
                and (entry / "hook.py").is_file()
                and (family / f"{entry.name}.py").is_file()
            ):
                violations.append(
                    f"{family_name}/{entry.name} conflicts with "
                    f"{entry.name}.py"
                )

    assert violations == []


@pytest.mark.unit
def test_algorithm_package_initializers_are_lightweight():
    violations = []

    for family_name in HOOK_FAMILIES:
        family = ALGORITHM_ROOT / family_name
        for package in family.iterdir():
            hook_path = package / "hook.py"
            init_path = package / "__init__.py"
            if not hook_path.is_file() or not init_path.is_file():
                continue
            tree = ast.parse(init_path.read_text(encoding="utf-8"))
            statements = list(tree.body)
            if (
                statements
                and isinstance(statements[0], ast.Expr)
                and isinstance(statements[0].value, ast.Constant)
                and isinstance(statements[0].value.value, str)
            ):
                statements.pop(0)
            if statements:
                violations.append(str(init_path.relative_to(REPOSITORY_ROOT)))

    assert violations == []


@pytest.mark.unit
def test_hook_catalog_algorithm_module_paths_exist():
    catalog = (REPOSITORY_ROOT / "docs" / "hooks" / "catalog.md").read_text(
        encoding="utf-8"
    )
    paths = set(re.findall(
        r"`(dependency/core/lib/algorithms/[^`]+\.py)`",
        catalog,
    ))

    assert paths
    assert [path for path in sorted(paths) if not (REPOSITORY_ROOT / path).is_file()] == []
