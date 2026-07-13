from runtime_model import RuntimeSlot
from runtime_renderer import FORBIDDEN_RUNTIME_ENV
from template_helper import TemplateHelper


def _helper(mounted_runtime):
    return TemplateHelper(str(mounted_runtime))


def _source_deploy():
    return [{
        "source": {
            "id": 0,
            "name": "camera-a",
            "url": "http://datasource/video0",
            "source_mode": "http_video",
            "source_type": "video",
            "metadata": {"fps": 25},
        },
        "node_set": ["edge-a", "edge-b"],
        "dag": {
            "_start": ["face-detection"],
            "face-detection": {
                "id": "face-detection", "prev": [], "succ": [],
            },
        },
    }]


def _assert_runtime_service(rendered, install_id="install-e2e"):
    manifest = rendered.manifest
    assert manifest["apiVersion"] == "sedna.io/v1alpha1"
    assert manifest["kind"] == "RuntimeService"
    assert manifest["metadata"]["labels"]["dayu.io/install-id"] == install_id
    assert manifest["spec"]["installID"] == install_id
    pod_spec = manifest["spec"]["podTemplate"]["spec"]
    assert pod_spec["automountServiceAccountToken"] is False
    assert "nodeName" not in pod_spec
    assert "serviceAccountName" not in pod_spec
    env_names = {
        item["name"]
        for container in pod_spec["containers"]
        for item in container.get("env", [])
    }
    assert env_names.isdisjoint(FORBIDDEN_RUNTIME_ENV)


def test_declared_policy_and_processor_catalog_is_fully_loadable(mounted_runtime):
    helper = _helper(mounted_runtime)
    base = helper.load_base_info()

    assert base["default-cloud-processor-backup"] is False
    assert base["scheduler-policies"]
    assert base["services"]
    for policy in base["scheduler-policies"]:
        assert set(helper.load_policy_apply_yaml(policy)) == {
            "scheduler", "generator", "controller", "distributor", "monitor",
        }
    loaded = helper.load_application_apply_yaml({
        service["id"]: {
            "yaml": service["yaml"], "node": ["edge-a"],
        }
        for service in base["services"]
    })
    assert len(loaded) == len(base["services"])
    assert all(item["service"]["pod-template"]["image"] for item in loaded.values())


def test_end_to_end_compilation_produces_tokenless_runtime_services(mounted_runtime):
    helper = _helper(mounted_runtime)
    base = helper.load_base_info()
    policy = next(item for item in base["scheduler-policies"] if item["id"] == "fixed")
    policy_templates = helper.load_policy_apply_yaml(policy)
    normalized, service_dict = helper.normalize_source_deploy(_source_deploy())
    processors = helper.load_application_apply_yaml(service_dict)
    renderer = helper.create_runtime_renderer("install-e2e")

    rendered = []
    for component in ("scheduler", "controller", "distributor", "monitor"):
        logical = policy_templates[component]
        position = "cloud" if logical["position"] in {"cloud", "both"} else "edge"
        rendered.append(renderer.render(
            logical,
            RuntimeSlot(component, "cloud-a" if position == "cloud" else "edge-a", position),
            revision=7,
        ))
    rendered.extend(renderer.render_generator_sources(
        policy_templates["generator"], normalized, revision=7,
        selected_nodes={"0": "edge-a"},
    ))
    logical_processor = processors["face-detection"]["service"]
    rendered.extend([
        renderer.render(
            logical_processor,
            RuntimeSlot("processor", node, "edge", logical_service="face-detection"),
            revision=7,
        )
        for node in service_dict["face-detection"]["node"]
    ])

    assert {item.unit.slot.component for item in rendered} == {
        "scheduler", "generator", "controller", "distributor", "monitor", "processor",
    }
    assert all(item.unit.runtime_revision == 7 for item in rendered)
    assert len({item.unit.runtime_id for item in rendered}) == len(rendered)
    for item in rendered:
        _assert_runtime_service(item)


def test_generator_selection_is_independent_from_processor_placement(mounted_runtime):
    helper = _helper(mounted_runtime)
    policy = next(
        item for item in helper.load_base_info()["scheduler-policies"]
        if item["id"] == "fixed"
    )
    generator_template = helper.load_policy_apply_yaml(policy)["generator"]
    normalized, services = helper.normalize_source_deploy(_source_deploy())

    rendered = helper.render_generator_runtime_services(
        generator_template,
        normalized,
        revision=3,
        install_id="install-e2e",
        selected_nodes={"0": "edge-c"},
    )[0]

    assert rendered.unit.slot.target_node == "edge-c"
    assert services["face-detection"]["node"] == ["edge-a", "edge-b"]
    env = {
        item["name"]: item["value"]
        for item in rendered.manifest["spec"]["podTemplate"]["spec"]["containers"][0]["env"]
    }
    assert env["ALL_EDGE_DEVICES"] == "['edge-a', 'edge-b']"
    _assert_runtime_service(rendered)
