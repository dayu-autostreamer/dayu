import copy
import importlib
from types import SimpleNamespace

import pytest


def _processor_doc(name):
    return {
        "apiVersion": "sedna.io/v1alpha1",
        "kind": "JointMultiEdgeService",
        "metadata": {"name": name},
        "spec": {"edgeWorker": []},
    }


class FakeThread:
    def __init__(self, target):
        self.target = target
        self.started = False

    def start(self):
        self.started = True


@pytest.fixture
def backend_core_instance(mounted_runtime, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "check_pod_name",
        staticmethod(lambda *args, **kwargs: False),
    )
    return backend_core_module.BackendCore()


@pytest.mark.unit
def test_parse_and_apply_templates_runs_two_stage_install_and_starts_cycle_thread(backend_core_instance, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    source_deploy = [{"source": {"id": 0, "name": "camera-0"}, "dag": {}, "node_set": ["edge1"]}]
    first_docs = [_processor_doc("scheduler"), _processor_doc("controller")]
    second_docs = [_processor_doc("generator"), _processor_doc("processor-face")]
    saved_docs = []
    install_calls = []
    created_threads = []

    backend_core_instance.template_helper = SimpleNamespace(
        load_policy_apply_yaml=lambda policy: {"scheduler": {"policy": policy["id"]}},
        load_application_apply_yaml=lambda service_dict: service_dict,
        finetune_yaml_parameters=lambda yaml_dict, deploy, scopes: copy.deepcopy(
            first_docs if scopes == ["scheduler", "distributor", "monitor", "controller"] else second_docs
        ),
    )
    monkeypatch.setattr(
        backend_core_instance,
        "extract_service_from_source_deployment",
        lambda deploy: {"face-detection": {"yaml": "face.yaml", "node": ["edge1"]}},
    )
    monkeypatch.setattr(
        backend_core_instance,
        "install_yaml_templates",
        lambda docs: install_calls.append(copy.deepcopy(docs)) or (True, ""),
    )
    monkeypatch.setattr(
        backend_core_instance,
        "save_component_yaml",
        lambda docs: saved_docs.append(copy.deepcopy(docs)),
    )
    monkeypatch.setattr(backend_core_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        backend_core_module.threading,
        "Thread",
        lambda target: created_threads.append(FakeThread(target)) or created_threads[-1],
    )

    result, msg = backend_core_instance.parse_and_apply_templates({"id": "fixed"}, copy.deepcopy(source_deploy))

    assert (result, msg) == (True, "Install services successfully")
    assert install_calls == [first_docs, second_docs]
    assert saved_docs == [first_docs, first_docs + second_docs]
    assert backend_core_instance.installed_running_state is True
    assert backend_core_instance.is_cycle_deploy is True
    assert backend_core_instance.yaml_dict == {
        "scheduler": {"policy": "fixed"},
        "processor": {"face-detection": {"yaml": "face.yaml", "node": ["edge1"]}},
    }
    assert created_threads and created_threads[0].target == backend_core_instance.run_cycle_deploy
    assert created_threads[0].started is True


@pytest.mark.unit
def test_parse_and_apply_templates_handles_first_stage_timeout(backend_core_instance, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    first_docs = [_processor_doc("scheduler")]
    saved_docs = []

    backend_core_instance.template_helper = SimpleNamespace(
        load_policy_apply_yaml=lambda policy: {"scheduler": {"policy": policy["id"]}},
        load_application_apply_yaml=lambda service_dict: service_dict,
        finetune_yaml_parameters=lambda yaml_dict, deploy, scopes: copy.deepcopy(first_docs),
    )
    monkeypatch.setattr(backend_core_instance, "extract_service_from_source_deployment", lambda deploy: {})
    monkeypatch.setattr(
        backend_core_instance,
        "install_yaml_templates",
        lambda docs: (_ for _ in ()).throw(backend_core_module.timeout_exceptions.FunctionTimedOut()),
    )
    monkeypatch.setattr(
        backend_core_instance,
        "save_component_yaml",
        lambda docs: saved_docs.append(copy.deepcopy(docs)),
    )

    result, msg = backend_core_instance.parse_and_apply_templates({"id": "fixed"}, [])

    assert (result, msg) == (False, "first-stage install timeout after 100 seconds")
    assert saved_docs == [first_docs]
    assert backend_core_instance.installed_running_state is False
    assert backend_core_instance.is_cycle_deploy is False


@pytest.mark.unit
def test_parse_and_apply_templates_handles_second_stage_exception(backend_core_instance, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    first_docs = [_processor_doc("scheduler")]
    second_docs = [_processor_doc("processor-face")]
    saved_docs = []
    install_count = {"count": 0}

    backend_core_instance.template_helper = SimpleNamespace(
        load_policy_apply_yaml=lambda policy: {"scheduler": {"policy": policy["id"]}},
        load_application_apply_yaml=lambda service_dict: service_dict,
        finetune_yaml_parameters=lambda yaml_dict, deploy, scopes: copy.deepcopy(
            first_docs if scopes == ["scheduler", "distributor", "monitor", "controller"] else second_docs
        ),
    )
    monkeypatch.setattr(backend_core_instance, "extract_service_from_source_deployment", lambda deploy: {})

    def install_yaml_templates(docs):
        install_count["count"] += 1
        if install_count["count"] == 2:
            raise RuntimeError("boom")
        return True, ""

    monkeypatch.setattr(backend_core_instance, "install_yaml_templates", install_yaml_templates)
    monkeypatch.setattr(
        backend_core_instance,
        "save_component_yaml",
        lambda docs: saved_docs.append(copy.deepcopy(docs)),
    )
    monkeypatch.setattr(backend_core_module.time, "sleep", lambda seconds: None)

    result, msg = backend_core_instance.parse_and_apply_templates({"id": "fixed"}, [])

    assert (result, msg) == (False, "unexpected system error, please refer to logs in backend")
    assert install_count["count"] == 2
    assert saved_docs == [first_docs, first_docs + second_docs]
    assert backend_core_instance.installed_running_state is False
    assert backend_core_instance.is_cycle_deploy is False


@pytest.mark.unit
def test_parse_and_delete_templates_waits_for_lock_and_handles_timeout(backend_core_instance, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    sleep_calls = []
    backend_core_instance.uninstall_lock = True
    backend_core_instance.is_cycle_deploy = True

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        backend_core_instance.uninstall_lock = False

    monkeypatch.setattr(backend_core_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(backend_core_instance, "read_component_yaml", lambda: [_processor_doc("processor-face")])
    monkeypatch.setattr(
        backend_core_instance,
        "uninstall_yaml_templates",
        lambda docs: (_ for _ in ()).throw(backend_core_module.timeout_exceptions.FunctionTimedOut()),
    )

    result, msg = backend_core_instance.parse_and_delete_templates()

    assert (result, msg) == (False, "timeout after 200 seconds")
    assert sleep_calls[0] == 0.5
    assert backend_core_instance.is_cycle_deploy is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("original_docs", "operate_outcome", "expected"),
    [
        (None, None, (False, "")),
        ([_processor_doc("processor-face")], (False, "apply failed"), (False, "apply failed")),
        ([_processor_doc("processor-face")], RuntimeError("broken"), (False, "unexpected system error, please refer to logs in backend")),
        ([_processor_doc("processor-face")], (True, ""), (True, "")),
    ],
)
def test_parse_and_redeploy_services_handles_state_and_failures(
    backend_core_instance,
    monkeypatch,
    original_docs,
    operate_outcome,
    expected,
):
    update_docs = [_processor_doc("processor-face"), _processor_doc("processor-new")]
    monkeypatch.setattr(backend_core_instance, "read_component_yaml", lambda: copy.deepcopy(original_docs))

    if original_docs is not None:
        monkeypatch.setattr(
            backend_core_instance,
            "check_and_update_docs_list",
            lambda current_docs, new_docs: (
                current_docs + [_processor_doc("processor-new")],
                [_processor_doc("processor-new")],
                [_processor_doc("processor-face")],
                [],
            ),
        )

        def operate_processors(docs_to_update, docs_to_add, docs_to_delete):
            if isinstance(operate_outcome, Exception):
                raise operate_outcome
            return operate_outcome

        monkeypatch.setattr(backend_core_instance, "operate_processors", operate_processors)

    assert backend_core_instance.parse_and_redeploy_services(update_docs) == expected


@pytest.mark.unit
def test_operate_processors_waits_for_deleted_and_started_pods(backend_core_instance, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    delete_checks = iter([True, False])
    start_checks = iter([False, True])
    observed = {"delete": [], "start": []}
    sleep_calls = []

    monkeypatch.setattr(
        backend_core_instance,
        "update_processors",
        lambda docs: (True, "", ["processor-update"]),
    )
    monkeypatch.setattr(
        backend_core_instance,
        "install_processors",
        lambda docs: (True, "", ["processor-add"]),
    )
    monkeypatch.setattr(
        backend_core_instance,
        "uninstall_processors",
        lambda docs: (True, "", ["processor-delete"]),
    )
    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "check_pods_with_string_exists",
        staticmethod(
            lambda namespace, include_str_list=None: observed["delete"].append(list(include_str_list)) or next(delete_checks)
        ),
    )
    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "check_specific_pods_running",
        staticmethod(
            lambda namespace, names: observed["start"].append(list(names)) or next(start_checks)
        ),
    )
    monkeypatch.setattr(backend_core_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    result = backend_core_instance.operate_processors(
        [_processor_doc("processor-update")],
        [_processor_doc("processor-add")],
        [_processor_doc("processor-delete")],
    )

    assert result == (True, "")
    assert observed["delete"] == [["processor-delete", "processor-update"], ["processor-delete", "processor-update"]]
    assert observed["start"] == [["processor-add", "processor-update"], ["processor-add", "processor-update"]]
    assert sleep_calls == [1, 1]


@pytest.mark.unit
def test_processor_resource_helpers_filter_support_components_and_surface_errors(backend_core_instance, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    applied_docs = []
    deleted_docs = []
    yaml_docs = [_processor_doc("backend"), _processor_doc("processor-face")]

    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "apply_custom_resources",
        staticmethod(lambda docs: applied_docs.append([doc["metadata"]["name"] for doc in docs]) or True),
    )
    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "delete_custom_resources",
        staticmethod(lambda docs: deleted_docs.append([doc["metadata"]["name"] for doc in docs]) or True),
    )

    assert backend_core_instance.update_processors(copy.deepcopy(yaml_docs)) == (True, "", ["processor-face"])
    assert backend_core_instance.install_processors(copy.deepcopy(yaml_docs)) == (True, "", ["processor-face"])
    assert backend_core_instance.uninstall_processors(copy.deepcopy(yaml_docs)) == (True, "", ["processor-face"])
    assert applied_docs == [["processor-face"], ["processor-face"]]
    assert deleted_docs == [["processor-face"], ["processor-face"]]

    assert backend_core_instance.install_processors([_processor_doc("backend")]) == (
        True,
        "no processors need to be installed.",
        [],
    )

    monkeypatch.setattr(backend_core_module.KubeHelper, "apply_custom_resources", staticmethod(lambda docs: False))
    monkeypatch.setattr(backend_core_module.KubeHelper, "delete_custom_resources", staticmethod(lambda docs: False))

    assert backend_core_instance.update_processors([_processor_doc("processor-face")]) == (
        False,
        "kubernetes api error.",
        [],
    )
    assert backend_core_instance.install_processors([_processor_doc("processor-face")]) == (
        False,
        "kubernetes api error.",
        [],
    )
    assert backend_core_instance.uninstall_processors([_processor_doc("processor-face")]) == (
        False,
        "kubernetes api error.",
        [],
    )


@pytest.mark.unit
def test_yaml_template_helpers_wait_for_cluster_state(backend_core_instance, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    running_checks = iter([False, True])
    install_checks = iter([True, False])
    sleep_calls = []

    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "apply_custom_resources",
        staticmethod(lambda docs: True),
    )
    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "delete_custom_resources",
        staticmethod(lambda docs: True),
    )
    monkeypatch.setattr(
        backend_core_instance,
        "check_pods_running_state",
        lambda: next(running_checks),
    )
    monkeypatch.setattr(
        backend_core_instance,
        "check_install_state",
        lambda: next(install_checks),
    )
    monkeypatch.setattr(backend_core_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    backend_core_instance.installed_running_state = True

    assert backend_core_instance.install_yaml_templates(None) == (
        False,
        "yaml data is lost, fail to install resources",
    )
    assert backend_core_instance.uninstall_yaml_templates(None) == (
        False,
        "yaml docs is lost, fail to delete resources",
    )
    assert backend_core_instance.install_yaml_templates([_processor_doc("processor-face")]) == (True, "")
    assert backend_core_instance.uninstall_yaml_templates([_processor_doc("processor-face")]) == (True, "")
    assert backend_core_instance.installed_running_state is False
    assert sleep_calls == [1, 1]


@pytest.mark.unit
def test_backend_core_validation_and_url_helpers_cover_guard_branches(backend_core_instance, monkeypatch, tmp_path):
    backend_core_module = importlib.import_module("backend_core")
    backend_core_instance.inner_datasource = True

    inner_datasource_path = tmp_path / "inner_datasource.yaml"
    inner_datasource_path.write_text(
        "\n".join(
            [
                "source_name: demo",
                "source_type: video",
                "source_mode: http_video",
                "source_list:",
                "  - name: camera-a",
                "    dir: /data/camera-a",
                "    metadata: {fps: 25}",
            ]
        ),
        encoding="utf-8",
    )
    axis_visualization_path = tmp_path / "viz.yaml"
    axis_visualization_path.write_text(
        "\n".join(
            [
                "- name: Delay",
                "  type: curve",
                "  variables: [delay]",
                "  size: 1",
                "  x_axis: timestamp",
                "  y_axis: value",
                "  hook_name: delay_chart",
                "  hook_params: \"{}\"",
            ]
        ),
        encoding="utf-8",
    )

    assert backend_core_instance.check_datasource_config(str(tmp_path / "missing.txt")) is None
    assert backend_core_instance.check_visualization_config(str(tmp_path / "missing.txt")) is None
    assert backend_core_instance.check_datasource_config(str(inner_datasource_path))["source_list"][0]["dir"] == "/data/camera-a"
    assert backend_core_instance.check_visualization_config(str(axis_visualization_path))[0]["x_axis"] == "timestamp"

    monkeypatch.setattr(
        backend_core_module.PortInfo,
        "get_component_port",
        staticmethod(lambda component: (_ for _ in ()).throw(RuntimeError(f"missing {component}"))),
    )
    monkeypatch.setattr(backend_core_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloudx1"))
    monkeypatch.setattr(backend_core_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.8"))

    backend_core_instance.get_resource_url()
    backend_core_instance.get_result_url()
    backend_core_instance.get_log_url()

    assert backend_core_instance.resource_url is None
    assert backend_core_instance.result_url is None
    assert backend_core_instance.log_fetch_url is None
    assert backend_core_instance.get_file_result("artifact.bin") == ""
    assert backend_core_instance.open_result_log_export_stream() is None


@pytest.mark.unit
def test_run_get_result_retries_missing_and_empty_batches_before_success(backend_core_instance, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    url_states = iter(["missing", "empty", "success"])
    http_payloads = []
    parsed_batches = []

    def get_result_url():
        state = next(url_states)
        if state == "missing":
            backend_core_instance.result_url = None
            backend_core_instance.result_file_url = None
        else:
            backend_core_instance.result_url = "http://cloud/result"
            backend_core_instance.result_file_url = "http://cloud/file"

    def fake_http_request(url, method=None, json=None, **kwargs):
        http_payloads.append(copy.deepcopy(json))
        if len(http_payloads) == 1:
            return None
        return {"time_ticket": 7, "result": ["task-result"]}

    monkeypatch.setattr(backend_core_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(backend_core_instance, "get_result_url", get_result_url)
    monkeypatch.setattr(backend_core_module, "http_request", fake_http_request)
    monkeypatch.setattr(
        backend_core_instance,
        "parse_task_result",
        lambda results: parsed_batches.append(results) or setattr(backend_core_instance, "is_get_result", False),
    )

    backend_core_instance.is_get_result = True
    backend_core_instance.run_get_result()

    assert http_payloads == [{"time_ticket": 0, "size": 0}, {"time_ticket": 0, "size": 0}]
    assert parsed_batches == [["task-result"]]
    assert backend_core_instance.result_url == "http://cloud/result"
    assert backend_core_instance.result_file_url == "http://cloud/file"


@pytest.mark.unit
def test_run_cycle_deploy_skips_missing_config_then_updates_after_success(backend_core_instance, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    sleep_calls = []
    finetune_calls = []
    updated_docs = []
    redeploy_docs = [_processor_doc("processor-face")]

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        if seconds == 5 and sleep_calls.count(5) > 1:
            backend_core_instance.yaml_dict = {"scheduler": {"policy": "fixed"}}
            backend_core_instance.source_deploy = [{"source": {"id": 0}, "dag": {}, "node_set": ["edge1"]}]
        if seconds == 1 and backend_core_instance.yaml_dict:
            backend_core_instance.is_cycle_deploy = False

    backend_core_instance.is_cycle_deploy = True
    backend_core_instance.yaml_dict = None
    backend_core_instance.source_deploy = None
    backend_core_instance.template_helper = SimpleNamespace(
        finetune_yaml_parameters=lambda yaml_dict, source_deploy, scopes: finetune_calls.append(tuple(scopes)) or copy.deepcopy(redeploy_docs)
    )

    monkeypatch.setattr(backend_core_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(backend_core_instance, "check_pods_running_state", lambda: True)
    monkeypatch.setattr(
        backend_core_instance,
        "parse_and_redeploy_services",
        lambda docs: (True, ""),
    )
    monkeypatch.setattr(
        backend_core_instance,
        "update_component_yaml",
        lambda docs: updated_docs.append(copy.deepcopy(docs)),
    )

    backend_core_instance.run_cycle_deploy()

    assert sleep_calls[:3] == [5, 1, 5]
    assert finetune_calls == [("processor",)]
    assert updated_docs == [redeploy_docs]
    assert backend_core_instance.uninstall_lock is False
