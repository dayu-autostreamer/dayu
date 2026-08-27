import shutil
from pathlib import Path

import pytest

from tools import validate_build_matrix


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.unit
def test_validate_build_matrix_passes_for_repository_sources():
    errors, summary = validate_build_matrix.validate_build_matrix(REPO_ROOT)

    assert errors == []
    assert "backend" in summary["bake_targets"]
    assert "traffic-signal-recognition" in summary["bake_targets"]
    assert "default" in summary["bake_groups"]
    assert "build/backend.Dockerfile" in summary["dockerfiles"]
    assert "traffic-signal-recognition" in summary["template_images"]


@pytest.mark.unit
def test_validate_build_matrix_normalizes_image_references():
    assert validate_build_matrix.normalize_image_reference("scheduler:shy") == "scheduler"
    assert validate_build_matrix.normalize_image_reference("repo:5000/dayuhub/generator:v1.4") == "generator"
    assert validate_build_matrix.normalize_image_reference("ghcr.io/dayu/processor@sha256:abc") == "processor"


@pytest.mark.unit
def test_validate_build_matrix_reports_unlisted_dockerfile(tmp_path):
    for name in ("build", "template"):
        shutil.copytree(REPO_ROOT / name, tmp_path / name)
    shutil.copy2(REPO_ROOT / "docker-bake.hcl", tmp_path / "docker-bake.hcl")

    extra_dockerfile = tmp_path / "build" / "orphan.Dockerfile"
    extra_dockerfile.write_text("FROM scratch\n", encoding="utf-8")

    errors, _ = validate_build_matrix.validate_build_matrix(tmp_path)

    assert f"build Dockerfile is not referenced by docker-bake.hcl: build/{extra_dockerfile.name}" in errors
