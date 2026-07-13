import ast
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOTS = (
    PROJECT_ROOT / "dependency" / "core",
    PROJECT_ROOT / "datasource",
    PROJECT_ROOT / "components",
)
FORBIDDEN_SYMBOLS = {"KubeConfig", "NodeInfo", "PortConfig", "PortInfo"}
FORBIDDEN_RUNTIME_OPERATIONS = {
    "delete_error_processor_pods",
    "delete_error_processor_pods_if_needed",
    "_delete_error_processor_pods_if_needed",
}


@pytest.mark.unit
def test_runtime_python_has_no_kubernetes_client_or_discovery_helpers():
    violations = []
    for path in (
        path
        for root in RUNTIME_ROOTS
        for path in root.rglob("*.py")
    ):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "kubernetes" or alias.name.startswith("kubernetes."):
                        violations.append(f"{path}: imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "kubernetes" or module.startswith("kubernetes."):
                    violations.append(f"{path}: imports from {module}")
                for alias in node.names:
                    if alias.name in FORBIDDEN_SYMBOLS:
                        violations.append(f"{path}: imports {alias.name}")
            elif isinstance(node, ast.Name):
                if node.id in FORBIDDEN_SYMBOLS | FORBIDDEN_RUNTIME_OPERATIONS:
                    violations.append(f"{path}: references {node.id}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in FORBIDDEN_RUNTIME_OPERATIONS:
                    violations.append(f"{path}: defines {node.name}")
            elif isinstance(node, ast.Attribute):
                if node.attr == "force_refresh":
                    violations.append(f"{path}: calls force_refresh")
                elif node.attr in FORBIDDEN_RUNTIME_OPERATIONS:
                    violations.append(f"{path}: references {node.attr}")

        for legacy_module in (
            "core.lib.common.kube",
            "core.lib.network.node",
            "core.lib.network.port",
        ):
            if legacy_module in source:
                violations.append(f"{path}: references {legacy_module}")

    assert violations == []


@pytest.mark.unit
def test_runtime_requirements_do_not_install_kubernetes_client():
    offenders = []
    for path in (
        path
        for root in RUNTIME_ROOTS
        for path in root.rglob("requirements*.txt")
    ):
        for line in path.read_text(encoding="utf-8").splitlines():
            requirement = line.split("#", 1)[0].strip().lower()
            if requirement.startswith("kubernetes"):
                offenders.append(f"{path}: {line.strip()}")
    assert offenders == []
