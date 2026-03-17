import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
DEPENDENCY_DIR = REPO_ROOT / "dependency"

for path in (str(BACKEND_DIR), str(DEPENDENCY_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("DELETE_TEMP_FILES", "False")
os.environ.setdefault("LOG_LEVEL", "INFO")


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

    shutil.copytree(REPO_ROOT / "template", volume0)
    volume1.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("DATA_PATH_PREFIX", str(data_path_prefix))
    monkeypatch.setenv("FILE_PREFIX", str(REPO_ROOT / "template"))
    monkeypatch.setenv("VOLUME_NUM", "2")
    monkeypatch.setenv("NAMESPACE", "dayu")
    monkeypatch.setenv("DELETE_TEMP_FILES", "False")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    return volume0
