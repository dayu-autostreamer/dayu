import os
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DAYU_SCRIPT = REPOSITORY_ROOT / "dayu.sh"


def _write_executable(path, content):
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_stop_continues_cleanup_when_backend_uninstall_fails(tmp_path):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to exercise dayu.sh")

    template = tmp_path / "template"
    template.mkdir()
    (template / "base.yaml").write_text("namespace: dayu-test\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    kubectl_log = tmp_path / "kubectl.log"

    _write_executable(
        fake_bin / "yq",
        """#!/bin/sh
case "$2" in
  .namespace) echo dayu-test ;;
  .log-level) echo INFO ;;
  .backend-rbac.service-account) echo dayu-backend ;;
  .backend-rbac.role) echo dayu-backend-role ;;
  .backend-rbac.role-binding) echo dayu-backend-binding ;;
  .backend-rbac.cluster-role) echo dayu-backend-cluster-role ;;
  .backend-rbac.cluster-role-binding) echo dayu-backend-cluster-binding ;;
  .support-crd-meta.api-version) echo sedna.io/v1alpha1 ;;
  .support-crd-meta.kind) echo JointMultiEdgeService ;;
  .default-image-meta.registry) echo registry.example ;;
  .default-image-meta.repository) echo dayu ;;
  .default-image-meta.tag) echo test ;;
  .default-file-mount-prefix) echo /data/dayu-files ;;
  .datasource.use-simulation) echo false ;;
  .datasource.data-root) echo /data/source ;;
  .datasource.node) echo edge-a ;;
  .datasource.play-mode) echo cycle ;;
  .log-export.system.retention-records) echo 0 ;;
  .log-export.system.compact-interval) echo 200 ;;
  *) echo "unexpected yq query: $2" >&2; exit 1 ;;
esac
""",
    )
    _write_executable(
        fake_bin / "kubectl",
        """#!/bin/sh
printf '%s\n' "$*" >> "$KUBECTL_LOG"
case "$*" in
  "get nodes "*) echo "cloud-a 10.0.0.1" ;;
  *"get runtimeservices.sedna.io "*) echo "scheduler-test" ;;
  *"get svc backend-cloud "*"jsonpath="*) echo "30080" ;;
esac
exit 0
""",
    )
    _write_executable(
        fake_bin / "curl",
        """#!/bin/sh
printf '%s\n' '{"state":"fail","msg":"scheduler unavailable"}'
""",
    )

    environment = os.environ.copy()
    environment.update({
        "ACTION": "stop",
        "TEMPLATE": str(template),
        "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
        "KUBECTL_LOG": str(kubectl_log),
        "GRACEFUL_STOP_WAIT_SEC": "1",
        "SVC_WAIT_SEC": "0",
        "MESH_WAIT_SEC": "0",
        "POD_WAIT_SEC": "0",
        "NS_WAIT_SEC": "0",
    })

    result = subprocess.run(
        [bash, str(DAYU_SCRIPT)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Backend graceful uninstall did not finish cleanly" in result.stdout
    assert "Backend graceful uninstall failed; continue with system cleanup" in result.stdout
    assert "DAYU system stop successfully" in result.stdout

    calls = kubectl_log.read_text(encoding="utf-8").splitlines()
    runtime_delete = next(
        index for index, call in enumerate(calls)
        if call.startswith("delete runtimeservices.sedna.io ")
    )
    namespace_delete = next(
        index for index, call in enumerate(calls)
        if call.startswith("delete namespace dayu-test ")
    )
    assert runtime_delete < namespace_delete


def test_stop_interface_does_not_require_a_force_flag():
    script = DAYU_SCRIPT.read_text(encoding="utf-8")

    assert "FORCE_RUNTIME_STOP" not in script


def test_datasource_backend_url_uses_absolute_cluster_service_dns():
    script = DAYU_SCRIPT.read_text(encoding="utf-8")

    assert 'http://backend-cloud.$NAMESPACE.svc.cluster.local.:8000' in script
