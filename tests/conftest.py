import importlib
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
DEPENDENCY_DIR = REPO_ROOT / "dependency"
DATASOURCE_DIR = REPO_ROOT / "datasource"

for path in (str(BACKEND_DIR), str(DEPENDENCY_DIR), str(DATASOURCE_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("NAMESPACE", "dayu")


REQUIRED_ML_MODULES = ("torch", "torch_geometric")


def pytest_addoption(parser):
    parser.addoption(
        "--require-ml-dependencies",
        action="store_true",
        default=False,
        help=(
            "Fail before collection unless the real PyTorch and "
            "PyTorch Geometric runtimes are importable."
        ),
    )


def pytest_sessionstart(session):
    if not session.config.getoption("--require-ml-dependencies"):
        return

    import_errors = []
    for module_name in REQUIRED_ML_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - runner environment gate
            import_errors.append(
                f"{module_name}: {type(exc).__name__}: {exc}"
            )

    if import_errors:
        details = "; ".join(import_errors)
        raise pytest.UsageError(
            "Required ML test dependencies are unavailable or broken: "
            f"{details}"
        )


@pytest.fixture(autouse=True)
def reset_counters():
    from core.lib.common import Counter

    Counter.reset_all_counts()
    yield
    Counter.reset_all_counts()


@pytest.fixture
def mounted_runtime(monkeypatch, tmp_path):
    data_path_prefix = tmp_path / "runtime"
    volume0 = data_path_prefix / "volume0"
    volume1 = data_path_prefix / "volume1"
    temp_dir = volume1 / "temp_files"

    shutil.copytree(REPO_ROOT / "template", volume0)
    temp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_PATH_PREFIX", str(data_path_prefix))
    monkeypatch.setenv("DEFAULT_MOUNT_PATH", str(volume0))
    monkeypatch.setenv("TEMP_PATH", str(temp_dir))
    monkeypatch.setenv("NAMESPACE", "dayu")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    return volume0
