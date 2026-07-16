import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DAYU_SCRIPT = REPOSITORY_ROOT / "dayu.sh"


def test_graceful_stop_binds_command_and_completion_to_install_identity():
    source = DAYU_SCRIPT.read_text(encoding="utf-8")

    assert 'stop_payload="{\\"install_id\\":\\"${install_id}\\"}"' in source
    assert '--data "${stop_payload}"' in source
    assert '"${current_install_id}" != "${install_id}"' in source


def _write_executable(path, content):
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _run_stop_scenario(
        tmp_path,
        *,
        install_states=(),
        stop_response='{"state":"success","msg":"accepted"}',
        app_resources="",
        namespace_exists=True,
        namespace_query_error=False,
        namespace_delete_mode="remove",
        final_namespace_query_error=False,
):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to exercise dayu.sh")

    template = tmp_path / "template"
    template.mkdir()
    (template / "base.yaml").write_text("namespace: dayu-test\n", encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    kubectl_log = tmp_path / "kubectl.log"
    timeout_log = tmp_path / "timeout.log"
    curl_log = tmp_path / "curl.log"
    namespace_state = tmp_path / "namespace.exists"
    namespace_delete_attempted = tmp_path / "namespace.delete-attempted"
    install_state_dir = tmp_path / "install-states"
    install_state_dir.mkdir()
    install_state_counter = tmp_path / "install-state.counter"
    fake_time_counter = tmp_path / "time.counter"

    if namespace_exists:
        namespace_state.write_text("present\n", encoding="utf-8")
    for index, response in enumerate(install_states, start=1):
        (install_state_dir / str(index)).write_text(response, encoding="utf-8")
    if install_states:
        (install_state_dir / "default").write_text(install_states[-1], encoding="utf-8")

    _write_executable(
        fake_bin / "yq",
        """#!/bin/sh
if [ "${3:-}" = "-" ]; then
  key="${2#.}"
  "$TEST_PYTHON" -c '
import json
import sys

value = json.load(sys.stdin)[sys.argv[1]]
if not isinstance(value, str):
    raise TypeError("install-state field must be a string")
print(value)
' "$key"
  exit $?
fi

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
  *"get namespace dayu-test --ignore-not-found=true -o name"*)
    if [ "$NAMESPACE_QUERY_ERROR" = "true" ]; then
      exit 1
    fi
    if [ "$FINAL_NAMESPACE_QUERY_ERROR" = "true" ] && [ -f "$NAMESPACE_DELETE_ATTEMPTED" ]; then
      exit 1
    fi
    if [ -f "$NAMESPACE_STATE_FILE" ]; then
      echo "namespace/dayu-test"
    fi
    ;;
  *"get runtimeservices.sedna.io "*) printf '%s' "$APP_RESOURCES" ;;
  *"get svc backend-cloud "*"jsonpath="*) echo "30080" ;;
  *"delete namespace dayu-test "*)
    : > "$NAMESPACE_DELETE_ATTEMPTED"
    if [ "$NAMESPACE_DELETE_MODE" = "remove" ]; then
      rm -f "$NAMESPACE_STATE_FILE"
      exit 0
    fi
    exit 1
    ;;
esac
exit 0
""",
    )
    _write_executable(
        fake_bin / "timeout",
        """#!/bin/sh
printf '%s\n' "$*" >> "$TIMEOUT_LOG"
case "$*" in
  *"kubectl --request-timeout=10s delete endpointslices "*) exit 124 ;;
esac
shift
exec "$@"
""",
    )
    _write_executable(
        fake_bin / "curl",
        """#!/bin/sh
printf '%s\n' "$*" >> "$CURL_LOG"
case "$*" in
  *"/install_state")
    count=0
    if [ -f "$INSTALL_STATE_COUNTER" ]; then
      count=$(cat "$INSTALL_STATE_COUNTER")
    fi
    count=$((count + 1))
    printf '%s\n' "$count" > "$INSTALL_STATE_COUNTER"
    response_file="$INSTALL_STATE_DIR/$count"
    if [ ! -f "$response_file" ]; then
      response_file="$INSTALL_STATE_DIR/default"
    fi
    if [ -f "$response_file" ]; then
      cat "$response_file"
    fi
    ;;
  *"/stop_service") printf '%s\n' "$BACKEND_STOP_RESPONSE" ;;
esac
""",
    )
    _write_executable(
        fake_bin / "date",
        """#!/bin/sh
if [ "${1:-}" = "+%s" ]; then
  value=0
  if [ -f "$FAKE_TIME_COUNTER" ]; then
    value=$(cat "$FAKE_TIME_COUNTER")
  fi
  value=$((value + 1))
  printf '%s\n' "$value" > "$FAKE_TIME_COUNTER"
  printf '%s\n' "$value"
  exit 0
fi
exec /bin/date "$@"
""",
    )
    _write_executable(fake_bin / "sleep", "#!/bin/sh\nexit 0\n")

    environment = os.environ.copy()
    environment.update({
        "ACTION": "stop",
        "TEMPLATE": str(template),
        "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
        "TEST_PYTHON": sys.executable,
        "KUBECTL_LOG": str(kubectl_log),
        "TIMEOUT_LOG": str(timeout_log),
        "CURL_LOG": str(curl_log),
        "NAMESPACE_STATE_FILE": str(namespace_state),
        "NAMESPACE_DELETE_ATTEMPTED": str(namespace_delete_attempted),
        "NAMESPACE_QUERY_ERROR": str(namespace_query_error).lower(),
        "NAMESPACE_DELETE_MODE": namespace_delete_mode,
        "FINAL_NAMESPACE_QUERY_ERROR": str(final_namespace_query_error).lower(),
        "APP_RESOURCES": app_resources,
        "INSTALL_STATE_DIR": str(install_state_dir),
        "INSTALL_STATE_COUNTER": str(install_state_counter),
        "BACKEND_STOP_RESPONSE": stop_response,
        "FAKE_TIME_COUNTER": str(fake_time_counter),
        "GRACEFUL_STOP_WAIT_SEC": "2",
        "WAIT_EDGEMESH_RULES": "true",
        "MESH_WAIT_SEC": "0",
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
    return SimpleNamespace(
        result=result,
        kubectl_calls=(kubectl_log.read_text(encoding="utf-8").splitlines()
                       if kubectl_log.exists() else []),
        timeout_calls=(timeout_log.read_text(encoding="utf-8").splitlines()
                       if timeout_log.exists() else []),
        curl_calls=(curl_log.read_text(encoding="utf-8").splitlines()
                    if curl_log.exists() else []),
    )


def test_stop_continues_cleanup_when_backend_uninstall_fails(tmp_path):
    scenario = _run_stop_scenario(
        tmp_path,
        install_states=('nonsense',),
        stop_response='{"state":"fail","msg":"scheduler unavailable"}',
        app_resources="scheduler-test",
    )

    result = scenario.result
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Backend graceful uninstall did not finish cleanly" in result.stdout
    assert "Backend graceful uninstall failed; continue with system cleanup" in result.stdout
    assert "DAYU system stop successfully" in result.stdout

    calls = scenario.kubectl_calls
    runtime_delete = next(
        index for index, call in enumerate(calls)
        if "delete runtimeservices.sedna.io " in call
    )
    namespace_delete = next(
        index for index, call in enumerate(calls)
        if "delete namespace dayu-test " in call
    )
    assert runtime_delete < namespace_delete

    bounded_calls = scenario.timeout_calls
    assert any(
        call.startswith("6 kubectl --request-timeout=5s get runtimeservices.sedna.io ")
        for call in bounded_calls
    )
    assert any(
        call.startswith("6 kubectl --request-timeout=5s get svc backend-cloud ")
        for call in bounded_calls
    )
    assert any(
        call.startswith("6 kubectl --request-timeout=5s get pods -A ")
        for call in bounded_calls
    )
    assert any(
        call.startswith(
            "11 kubectl --request-timeout=10s delete runtimeservices.sedna.io "
        ) and "--wait=false" in call
        for call in bounded_calls
    )
    assert any(
        "kubectl --request-timeout=10s delete clusterrolebinding " in call
        and "--wait=false" in call
        for call in bounded_calls
    )
    assert any(
        "kubectl --request-timeout=10s delete endpointslices " in call
        for call in bounded_calls
    )
    assert any(
        call.startswith("10 kubectl --request-timeout=10s delete namespace dayu-test ")
        and "--wait=true" in call
        and "--timeout=0s" in call
        for call in bounded_calls
    )


def test_stop_cancels_pending_install_even_without_runtime_services(tmp_path):
    install_id = "11111111-1111-4111-8111-111111111111"
    scenario = _run_stop_scenario(
        tmp_path,
        install_states=(
            f'{{"state":"uninstall","phase":"preparing-install","install_id":"{install_id}"}}',
            '{"state":"uninstall","phase":"uninstalled","install_id":""}',
        ),
        app_resources="",
    )

    assert scenario.result.returncode == 0, scenario.result.stdout + scenario.result.stderr
    stop_call = next(call for call in scenario.curl_calls if "/stop_service" in call)
    assert f'{{"install_id":"{install_id}"}}' in stop_call
    assert "Backend graceful uninstall finished successfully" in scenario.result.stdout


def test_stop_skips_global_runtime_stop_when_backend_reports_no_active_lifecycle(tmp_path):
    scenario = _run_stop_scenario(
        tmp_path,
        install_states=('{"state":"uninstall","phase":"uninstalled","install_id":""}',),
    )

    assert scenario.result.returncode == 0, scenario.result.stdout + scenario.result.stderr
    assert "No managed runtime or install admission is active" in scenario.result.stdout
    assert not any("/stop_service" in call for call in scenario.curl_calls)


@pytest.mark.parametrize(
    "invalid_snapshot",
    [
        "",
        "<html>bad gateway</html>",
        '{"state":"uninstall"}',
        '{"state":"uninstall","phase":"uninstalling","install_id":"not-a-canonical-uuid"}',
    ],
)
def test_stop_does_not_treat_invalid_install_state_as_target_completion(
        tmp_path, invalid_snapshot,
):
    install_id = "11111111-1111-4111-8111-111111111111"
    scenario = _run_stop_scenario(
        tmp_path,
        install_states=(
            f'{{"state":"install","phase":"active","install_id":"{install_id}"}}',
            invalid_snapshot,
        ),
    )

    assert scenario.result.returncode == 0, scenario.result.stdout + scenario.result.stderr
    assert "Backend graceful uninstall exceeded" in scenario.result.stdout
    assert "Backend graceful uninstall finished successfully" not in scenario.result.stdout
    assert "Backend graceful uninstall failed; continue with system cleanup" in scenario.result.stdout


def test_targeted_stop_completes_when_a_replacement_install_id_is_observed(tmp_path):
    install_id = "11111111-1111-4111-8111-111111111111"
    replacement_id = "22222222-2222-4222-8222-222222222222"
    scenario = _run_stop_scenario(
        tmp_path,
        install_states=(
            f'{{"state":"install","phase":"active","install_id":"{install_id}"}}',
            f'{{"state":"uninstall","phase":"preparing-install","install_id":"{replacement_id}"}}',
        ),
    )

    assert scenario.result.returncode == 0, scenario.result.stdout + scenario.result.stderr
    assert "Backend graceful uninstall finished successfully" in scenario.result.stdout


@pytest.mark.parametrize("phase", ["cancelling-install", "preparing-uninstall"])
def test_targeted_stop_waits_while_the_same_install_id_is_owned(tmp_path, phase):
    install_id = "11111111-1111-4111-8111-111111111111"
    scenario = _run_stop_scenario(
        tmp_path,
        install_states=(
            f'{{"state":"uninstall","phase":"preparing-install","install_id":"{install_id}"}}',
            f'{{"state":"uninstall","phase":"{phase}","install_id":"{install_id}"}}',
        ),
    )

    assert scenario.result.returncode == 0, scenario.result.stdout + scenario.result.stderr
    assert "Backend graceful uninstall exceeded" in scenario.result.stdout
    assert "Backend graceful uninstall finished successfully" not in scenario.result.stdout


@pytest.mark.parametrize(
    ("poll_snapshot", "graceful_finished"),
    [
        ('{"state":"uninstall","phase":"cancelling-install","install_id":"11111111-1111-4111-8111-111111111111"}', False),
        ('{"state":"uninstall","phase":"uninstalled","install_id":""}', True),
    ],
)
def test_global_stop_requires_a_fully_uninstalled_empty_identity(
        tmp_path, poll_snapshot, graceful_finished,
):
    scenario = _run_stop_scenario(
        tmp_path,
        install_states=("<html>state unavailable</html>", poll_snapshot),
    )

    assert scenario.result.returncode == 0, scenario.result.stdout + scenario.result.stderr
    stop_call = next(call for call in scenario.curl_calls if "/stop_service" in call)
    assert "--data" not in stop_call
    assert ("Backend graceful uninstall finished successfully" in scenario.result.stdout) is graceful_finished


def test_stop_is_idempotent_when_namespace_is_already_absent_and_cleans_cluster_rbac(tmp_path):
    scenario = _run_stop_scenario(tmp_path, namespace_exists=False)

    assert scenario.result.returncode == 0, scenario.result.stdout + scenario.result.stderr
    assert "DAYU system is already stopped" in scenario.result.stdout
    assert scenario.curl_calls == []
    assert any("delete clusterrolebinding dayu-backend-cluster-binding-dayu-test" in call
               for call in scenario.kubectl_calls)
    assert any("delete clusterrole dayu-backend-cluster-role-dayu-test" in call
               for call in scenario.kubectl_calls)


def test_stop_fails_when_initial_namespace_state_cannot_be_verified(tmp_path):
    scenario = _run_stop_scenario(tmp_path, namespace_query_error=True)

    assert scenario.result.returncode != 0
    assert "Unable to verify namespace 'dayu-test'" in scenario.result.stdout
    assert not any("delete namespace dayu-test" in call for call in scenario.kubectl_calls)


@pytest.mark.parametrize(
    ("delete_mode", "final_query_error", "expected_message"),
    [
        ("keep", False, "still exists; DAYU system stop is incomplete"),
        ("remove", True, "Unable to verify that namespace 'dayu-test' was removed"),
    ],
)
def test_stop_fails_when_final_namespace_removal_cannot_be_confirmed(
        tmp_path, delete_mode, final_query_error, expected_message,
):
    scenario = _run_stop_scenario(
        tmp_path,
        install_states=('{"state":"uninstall","phase":"uninstalled","install_id":""}',),
        stop_response='{"state":"fail","msg":"unavailable"}',
        namespace_delete_mode=delete_mode,
        final_namespace_query_error=final_query_error,
    )

    assert scenario.result.returncode != 0
    assert expected_message in scenario.result.stdout
    assert "DAYU system stop successfully" not in scenario.result.stdout


def test_stop_interface_does_not_require_a_force_flag():
    script = DAYU_SCRIPT.read_text(encoding="utf-8")

    assert "FORCE_RUNTIME_STOP" not in script
    assert 'GRACEFUL_STOP_WAIT_SEC:-60' in script
    assert "_wait_empty" not in script
    assert "SVC_WAIT_SEC" not in script
    assert "POD_WAIT_SEC" not in script


def test_backend_observer_rbac_uses_only_the_list_verbs_called_by_backend():
    script = DAYU_SCRIPT.read_text(encoding="utf-8")

    assert 'resources: ["nodes"]\n    verbs: ["list"]' in script
    assert 'resources: ["pods"]\n    verbs: ["list"]' in script
    assert 'apiGroups: ["metrics.k8s.io"]\n    resources: ["pods"]\n    verbs: ["list"]' in script


def test_support_layer_backend_clients_use_absolute_cluster_service_dns():
    script = DAYU_SCRIPT.read_text(encoding="utf-8")

    assert script.count(
        'http://backend-cloud.$NAMESPACE.svc.cluster.local.:8000'
    ) >= 2
    assert "value: 'http://$CLOUD_IP:$BACKEND_PORT'" not in script
