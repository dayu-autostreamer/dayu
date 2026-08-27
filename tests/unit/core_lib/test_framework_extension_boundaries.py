import ast
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CORE_ROOT = REPOSITORY_ROOT / "dependency" / "core"
COMPOSITION_ROOT = CORE_ROOT / "__init__.py"
FRAMEWORK_ROOTS = (
    COMPOSITION_ROOT,
    REPOSITORY_ROOT / "components",
    CORE_ROOT / "backend",
    CORE_ROOT / "controller",
    CORE_ROOT / "distributor",
    CORE_ROOT / "generator",
    CORE_ROOT / "monitor",
    CORE_ROOT / "processor",
    CORE_ROOT / "scheduler",
    CORE_ROOT / "lib" / "common",
    CORE_ROOT / "lib" / "content",
    CORE_ROOT / "lib" / "estimation",
    CORE_ROOT / "lib" / "network",
    CORE_ROOT / "lib" / "runtime",
    CORE_ROOT / "lib" / "scheduling",
    CORE_ROOT / "lib" / "solver",
)
SCHEDULER_ROOTS = (
    REPOSITORY_ROOT / "components" / "scheduler",
    CORE_ROOT / "scheduler",
)
CONCRETE_EXTENSION_NAMES = (
    "fragsplice",
    "hedger",
    "ibdash",
    "distream",
    "dtodrl",
    "casva",
    "chameleon",
    "cevas",
    "deepva",
    "adamec",
    "madeye",
)


def framework_python_files():
    for root in FRAMEWORK_ROOTS:
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from root.rglob("*.py")


@pytest.mark.unit
def test_framework_does_not_name_concrete_extensions():
    violations = []
    for path in framework_python_files():
        content = path.read_text(encoding="utf-8")
        lowered = content.lower()
        names = [name for name in CONCRETE_EXTENSION_NAMES if name in lowered]
        if names:
            violations.append(
                f"{path.relative_to(REPOSITORY_ROOT)} names {', '.join(names)}"
            )

    assert violations == []


@pytest.mark.unit
def test_scheduler_treats_extension_actions_as_opaque_data():
    scheduler_source = (
        CORE_ROOT / "scheduler" / "scheduler.py"
    ).read_text(encoding="utf-8")

    assert "clear_processor_queues" not in scheduler_source


@pytest.mark.unit
def test_scheduler_has_no_datasource_transport_dependency():
    violations = []
    forbidden_endpoints = {"/source", "/shared_file", "/file"}

    for root in SCHEDULER_ROOTS:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    modules = []
                for module in modules:
                    if module == "datasource" or module.startswith("datasource."):
                        violations.append(f"{path}: imports {module}")
                    if ".data_getter" in module:
                        violations.append(f"{path}: imports {module}")

                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value in forbidden_endpoints:
                        violations.append(
                            f"{path}: embeds datasource endpoint {node.value}"
                        )
                elif isinstance(node, (ast.Name, ast.Attribute)):
                    symbol = node.id if isinstance(node, ast.Name) else node.attr
                    if symbol == "video_data_source":
                        violations.append(f"{path}: references {symbol}")

    assert violations == []
