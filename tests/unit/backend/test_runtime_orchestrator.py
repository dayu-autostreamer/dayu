import copy
import json
import math
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from runtime_model import (
    RuntimeCleanupRef,
    RuntimeEndpoint,
    RuntimeSession,
    RuntimeSlot,
    RuntimeUnit,
    canonical_hash,
)
from runtime_orchestrator import (
    RuntimeOperationCancelled,
    RuntimeOrchestrationError,
    RuntimeOrchestrator,
    RuntimePreflightError,
    RuntimeRetirementPending,
)
from runtime_service_client import RuntimeServiceCancelled, RuntimeServiceClient
from runtime_session_store import StoredRuntimeSession


HASH = "b" * 64
INSTALL_ID = "11111111-1111-4111-8111-111111111111"


def source_deploy(*services):
    return [{
        "source": {
            "id": 1,
            "url": "http://camera/live",
            "source_mode": "http_video",
            "source_type": "video",
        },
        "node_set": ["edge-a", "edge-b"],
        "dag": {
            service: {"id": service, "prev": [], "succ": []}
            for service in services
        },
    }]


def inventory():
    return {
        "cloud-a": {
            "name": "cloud-a", "role": "cloud", "address": "10.0.0.1",
            "ready": True, "labels": {},
        },
        "edge-a": {
            "name": "edge-a", "role": "edge", "address": "10.0.0.2",
            "ready": True, "labels": {},
        },
        "edge-b": {
            "name": "edge-b", "role": "edge", "address": "10.0.0.3",
            "ready": True, "labels": {},
        },
    }


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class FakeCluster:
    def __init__(self, nodes=None, agents_ok=True, managed_nodes=None, events=None):
        self.nodes = copy.deepcopy(nodes or inventory())
        self.agents_ok = agents_ok
        self.managed_nodes = None if managed_nodes is None else set(managed_nodes)
        self.events = events if events is not None else []
        self.inventory_calls = 0
        self.inventory_timeouts = []
        self.preflight_calls = []
        self.metric_calls = []

    def node_inventory(self, request_timeout_seconds=None):
        self.inventory_calls += 1
        self.inventory_timeouts.append(request_timeout_seconds)
        return copy.deepcopy(self.nodes)

    def validate_managed_agents(self, targets):
        targets = tuple(sorted(targets))
        self.preflight_calls.append(targets)
        if self.agents_ok:
            ready = set(targets) if self.managed_nodes is None else set(targets) & self.managed_nodes
            missing = sorted(set(targets) - ready)
            return {
                "ok": not missing,
                "agents": {
                    name: {
                        "missing_nodes": missing,
                        "not_ready_nodes": [],
                        "ready_nodes": sorted(ready),
                    }
                    for name in ("sedna_lc", "edgemesh_agent")
                },
            }
        return {
            "ok": False,
            "agents": {
                "sedna_lc": {"missing_nodes": [targets[-1]], "not_ready_nodes": []},
                "edgemesh_agent": {"missing_nodes": [], "not_ready_nodes": [targets[-1]]},
            },
        }

    def runtime_metrics(
        self, refs, node_inventory=None, request_timeout_seconds=None,
    ):
        self.metric_calls.append((
            copy.deepcopy(refs),
            copy.deepcopy(node_inventory),
            request_timeout_seconds,
        ))
        return {}


class FakeRenderer:
    ENDPOINT_COMPONENTS = {"scheduler", "distributor", "controller", "processor"}

    def __init__(self, install_id):
        self.install_id = install_id

    @staticmethod
    def _has_endpoint(slot):
        return slot.component in FakeRenderer.ENDPOINT_COMPONENTS or (
            slot.component == "monitor" and slot.position == "cloud"
        )

    def render(self, logical_template, slot, revision, extra_env=None, container_overrides=None):
        runtime_id = slot.runtime_name(revision, self.install_id)
        container = {
            "name": slot.component,
            "env": [
                {"name": str(name), "value": str(value)}
                for name, value in sorted((extra_env or {}).items())
            ],
        }
        image = (logical_template.get("pod-template") or {}).get("image")
        if image:
            container["image"] = image
        spec = {
            "installID": self.install_id,
            "deploymentRevision": revision,
            "component": slot.component,
            "targetNode": slot.target_node,
            "podTemplate": {
                "spec": {
                    "containers": [container],
                },
            },
        }
        if slot.logical_service:
            spec["logicalService"] = slot.logical_service
        endpoint = None
        if self._has_endpoint(slot):
            spec["endpoint"] = {"port": 9000}
            endpoint = RuntimeEndpoint(f"{runtime_id}.dayu.svc.cluster.local", 9000)
        unit = RuntimeUnit(
            slot=slot,
            runtime_id=runtime_id,
            runtime_revision=revision,
            spec_hash=canonical_hash(spec),
            endpoint=endpoint,
        )
        return SimpleNamespace(
            unit=unit,
            manifest={
                "apiVersion": "sedna.io/v1alpha1",
                "kind": "RuntimeService",
                "metadata": {
                    "name": runtime_id,
                    "namespace": "dayu",
                    "labels": {
                        "app.kubernetes.io/managed-by": "dayu-backend",
                        "dayu.io/install-id": self.install_id,
                    },
                },
                "spec": spec,
            },
        )

    def render_generator_sources(
            self, logical_template, sources, revision, selected_nodes=None, common_env=None,
    ):
        rendered = []
        for source_info in sources:
            source = source_info["source"]
            source_id = str(source["id"])
            slot = RuntimeSlot(
                "generator", selected_nodes[source_id], "edge", source_id=source_id,
            )
            rendered.append(self.render(
                logical_template, slot, revision,
                extra_env={**(common_env or {}), "SOURCE_ID": source_id},
            ))
        return rendered


class FakeTemplateHelper:
    def __init__(
            self, clock=None, default_cloud_processor_backup=False,
            source_selection_scope="selected_edge_nodes",
    ):
        self.clock = clock
        self.default_cloud_processor_backup = default_cloud_processor_backup
        self.source_selection_scope = source_selection_scope

    def load_base_info(self):
        return {
            "namespace": "dayu",
            "default-cloud-processor-backup": self.default_cloud_processor_backup,
            "datasource": {"node": "edge-a"},
            "runtime": {
                "activation-timeout-seconds": 5,
                "operation-timeout-seconds": 30,
                "inventory-ttl-seconds": 1,
                "retirement-grace-seconds": 10,
                "lease-ttl-seconds": 30,
            },
        }

    def load_policy_apply_yaml(self, policy):
        templates = {
            component: {"component": component}
            for component in (
                "scheduler", "generator", "controller", "distributor", "monitor",
            )
        }
        templates["scheduler"]["pod-template"] = {
            "env": [{
                "name": "SCH_SELECTION_POLICY_PARAMETERS",
                "value": repr({"scope": self.source_selection_scope}),
            }],
        }
        templates["monitor"]["pod-template"] = {"image": "monitor:v1"}
        return templates

    @staticmethod
    def normalize_source_deploy(sources):
        normalized = copy.deepcopy(sources)
        services = {}
        for source_info in normalized:
            for service in source_info["dag"]:
                services.setdefault(service, {
                    "service_name": service,
                    "yaml": f"{service}.yaml",
                    "node": list(source_info["node_set"]),
                })
        return normalized, services

    @staticmethod
    def load_application_apply_yaml(services):
        result = copy.deepcopy(services)
        for item in result.values():
            item["service"] = {"pod-template": {"image": "processor:test"}}
        return result

    @staticmethod
    def process_image(image):
        return image

    @staticmethod
    def specify_jetpack_image(image, major):
        return f"{image}-jp{major}"

    @staticmethod
    def create_runtime_renderer(install_id):
        return FakeRenderer(install_id)


def observed_runtime(unit):
    status = {
        "observedGeneration": 1,
        "observedRevision": unit.runtime_revision,
        "observedSpecHash": HASH,
        "podRef": {"name": f"{unit.runtime_id}-pod", "uid": f"pod-{unit.runtime_id}"},
        "conditions": [
            {"type": "SpecAccepted", "status": "True"},
            {"type": "Ready", "status": "True"},
            {"type": "Activated", "status": "True"},
        ],
    }
    if unit.endpoint is not None:
        status["endpoint"] = {
            "serviceRef": {"name": unit.runtime_id, "uid": f"service-{unit.runtime_id}"},
            "dnsName": f"observed-{unit.runtime_id}.dayu.svc.cluster.local",
            "port": unit.endpoint.port,
        }
    return {
        "metadata": {
            "name": unit.runtime_id,
            "namespace": "dayu",
            "generation": 1,
            "uid": f"runtime-{unit.runtime_id}",
        },
        "spec": {"deploymentRevision": unit.runtime_revision},
        "status": status,
    }


class FakeRuntimeClient:
    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.created = {}
        self.wait_batches = []
        self.deleted = []
        self.delete_many_failures = {}

    def create(self, manifest, request_timeout_seconds=None):
        name = manifest["metadata"]["name"]
        self.created[name] = copy.deepcopy(manifest)
        self.events.append(f"create:{manifest['spec']['component']}")
        return copy.deepcopy(manifest)

    def get(self, name, request_timeout_seconds=None):
        return copy.deepcopy(self.created[name])

    def wait_for_conditions(self, expectations, **kwargs):
        units = tuple(expectations.values())
        self.wait_batches.append(tuple(unit.slot.component for unit in units))
        self.events.append("activate:" + ",".join(unit.slot.component for unit in units))
        return {unit.runtime_id: observed_runtime(unit) for unit in units}

    @staticmethod
    def bind_observed_unit(unit, obj):
        return RuntimeServiceClient.bind_observed_unit(unit, obj)

    def delete(self, name, uid=None, **kwargs):
        component = (self.created.get(name, {}).get("spec") or {}).get("component", "unknown")
        self.deleted.append((name, uid, component))
        self.events.append(f"delete:{component}")
        self.created.pop(name, None)
        return True

    def delete_many(self, identities, **kwargs):
        components = {
            (self.created.get(name, {}).get("spec") or {}).get("component", "unknown")
            for name in identities
        }
        for component in components:
            if self.delete_many_failures.get(component, 0):
                self.delete_many_failures[component] -= 1
                raise RuntimeError(f"transient {component} batch deletion failure")
        for name in sorted(identities):
            self.delete(name, uid=identities[name])
        return True

    def list(self, label_selector=None, request_timeout_seconds=None):
        install_id = ""
        for item in str(label_selector or "").split(","):
            if item.startswith("dayu.io/install-id="):
                install_id = item.split("=", 1)[1]
        items = []
        for manifest in self.created.values():
            obj = copy.deepcopy(manifest)
            metadata = obj.setdefault("metadata", {})
            if install_id and (metadata.get("labels") or {}).get("dayu.io/install-id") != install_id:
                continue
            metadata["uid"] = f"runtime-{metadata['name']}"
            items.append(obj)
        return {"metadata": {"resourceVersion": "1"}, "items": items}


class FakeSessionStore:
    def __init__(self):
        self.stored = None
        self.load_calls = 0
        self.revision = 0
        self.saved_phases = []
        self.deleted = False
        self.delete_failures = 0

    def load(self):
        self.load_calls += 1
        return self.stored

    def compare_and_swap(self, session, expected_resource_version):
        current = self.stored.resource_version if self.stored else None
        assert expected_resource_version == current
        self.revision += 1
        self.stored = StoredRuntimeSession(session, str(self.revision), "session-uid")
        self.saved_phases.append(session.phase)
        return self.stored

    def delete(self, expected_resource_version=None):
        assert self.stored is not None
        assert expected_resource_version == self.stored.resource_version
        if self.delete_failures:
            self.delete_failures -= 1
            raise RuntimeError("transient ConfigMap deletion failure")
        self.deleted = True
        self.stored = None
        return True


class UncertainSessionStore(FakeSessionStore):
    def __init__(self):
        super().__init__()
        self.lose_next_response = False
        self.fail_next_load = False
        self.replace_after_lost_response = None

    def load(self):
        if self.fail_next_load:
            self.fail_next_load = False
            raise RuntimeError("ConfigMap read unavailable")
        return super().load()

    def compare_and_swap(self, session, expected_resource_version):
        stored = super().compare_and_swap(session, expected_resource_version)
        if not self.lose_next_response:
            return stored
        self.lose_next_response = False
        if self.replace_after_lost_response is not None:
            self.revision += 1
            self.stored = StoredRuntimeSession(
                self.replace_after_lost_response,
                str(self.revision),
                "replacement-session-uid",
            )
        raise RuntimeError("ConfigMap write response lost")


class FakeScheduler:
    def __init__(
            self, initial_plan, events=None, initial_put_ack=True,
            wall_clock=None,
    ):
        self.source_plan = {"1": "edge-a"}
        self.initial_plan = copy.deepcopy(initial_plan)
        self.redeployment_plan = copy.deepcopy(initial_plan)
        self.events = events if events is not None else []
        self.directory = None
        self.proposals = {}
        self.calls = []
        self.initial_put_ack = initial_put_ack
        self.lease_failures = 0
        self.lease_counts = {}
        self.retirements = {}
        self.wall_clock = wall_clock or time.time
        self.clear_failures = 0
        self.clear_ack = True

    @staticmethod
    def _payload(kwargs):
        data = kwargs.get("data")
        return json.loads(data["data"]) if data else None

    def _lease_status(self, revision):
        revision = int(revision)
        retirement = self.retirements.get(revision)
        count = int(self.lease_counts.get(revision, 0))
        if (
            retirement is not None
            and not retirement["retired"]
            and self.wall_clock() >= retirement["deadline"]
        ):
            retirement["retired"] = True
            retirement["revoked_count"] += count
            self.lease_counts[revision] = 0
            count = 0
        return {
            "revision": revision,
            "count": count,
            "deadline": retirement["deadline"] if retirement else None,
            "retired": bool(retirement and retirement["retired"]),
            "revoked_count": retirement["revoked_count"] if retirement else 0,
        }

    def __call__(self, url, method, **kwargs):
        path = "/" + url.split("/", 3)[-1] if "/" in url.split(":9000", 1)[-1] else ""
        # Splitting at the authority is less surprising for FQDNs containing dots.
        path = url.split(":9000", 1)[1] or "/"
        payload = self._payload(kwargs)
        self.calls.append((method, path, payload, kwargs.get("params")))
        self.events.append(f"scheduler:{method}:{path}")

        if path == "/source_nodes_selection":
            return {"plan": copy.deepcopy(self.source_plan)}
        if path == "/initial_deployment":
            return {"plan": copy.deepcopy(self.initial_plan)}
        if path == "/redeployment":
            return {"plan": copy.deepcopy(self.redeployment_plan)}
        if path == "/runtime-directory" and method == "PUT":
            self.directory = copy.deepcopy(payload["directory"])
            return {"hash": self.directory["hash"] if self.initial_put_ack else "ambiguous"}
        if path == "/runtime-directory" and method == "GET":
            return copy.deepcopy(self.directory) if self.directory is not None else {
                "install_id": "",
                "revision": 0,
                "directory_revision": 0,
                "routes": [],
            }
        if path == "/runtime-directory" and method == "DELETE":
            if self.clear_failures:
                self.clear_failures -= 1
                raise RuntimeError("scheduler directory clear unavailable")
            previous_revision = int((self.directory or {}).get("revision") or 0)
            if self.directory is not None:
                assert payload["install_id"] == self.directory["install_id"]
            self.directory = None
            return {
                "cleared": self.clear_ack,
                "install_id": payload["install_id"],
                "previous_revision": previous_revision,
            }
        if path == "/runtime-directory/proposals" and method == "POST":
            current_revision = int((self.directory or {}).get("revision") or 0)
            if int(payload["base_revision"]) != current_revision:
                raise RuntimeError("proposal base revision conflict")
            self.proposals[payload["proposal_id"]] = copy.deepcopy(payload)
            return {"proposal_id": payload["proposal_id"]}
        if path.startswith("/runtime-directory/proposals/") and path.endswith("/commit"):
            proposal_id = path.split("/")[-2]
            proposal = self.proposals.pop(proposal_id)
            revision = int(payload["expected_revision"])
            assert revision == int(proposal["base_revision"])
            assert revision == int((self.directory or {}).get("revision") or 0)
            grace = float(payload["retirement_grace_seconds"])
            deadline = math.floor((self.wall_clock() + grace) * 1000) / 1000
            self.retirements.setdefault(revision, {
                "deadline": deadline,
                "retired": False,
                "revoked_count": 0,
            })
            self.directory = proposal["directory"]
            response = copy.deepcopy(self.directory)
            response["retirement"] = self._lease_status(revision)
            return response
        if path == "/runtime-directory/task-leases" and method == "GET":
            if self.lease_failures:
                self.lease_failures -= 1
                raise RuntimeError("scheduler lease API unavailable")
            return self._lease_status(kwargs["params"]["revision"])
        if path == "/runtime-directory/task-leases" and method == "PATCH":
            if self.lease_failures:
                self.lease_failures -= 1
                raise RuntimeError("scheduler lease API unavailable")
            revision = int(payload["revision"])
            deadline = float(payload["deadline"])
            retirement = self.retirements.setdefault(revision, {
                "deadline": deadline,
                "retired": False,
                "revoked_count": 0,
            })
            retirement["deadline"] = min(retirement["deadline"], deadline)
            return self._lease_status(revision)
        raise AssertionError(f"unexpected scheduler request: {method} {path}")


def make_orchestrator(
        initial_plan, *, agents_ok=True, initial_put_ack=True, clock=None,
        default_cloud_processor_backup=False, source_selection_scope="selected_edge_nodes",
        managed_nodes=None,
):
    events = []
    clock = clock or FakeClock()
    cluster = FakeCluster(
        agents_ok=agents_ok, managed_nodes=managed_nodes, events=events,
    )
    runtime = FakeRuntimeClient(events)
    sessions = FakeSessionStore()
    wall_clock = lambda: 1_000_000 + clock()
    scheduler = FakeScheduler(
        initial_plan,
        events,
        initial_put_ack=initial_put_ack,
        wall_clock=wall_clock,
    )
    orchestrator = RuntimeOrchestrator(
        FakeTemplateHelper(
            default_cloud_processor_backup=default_cloud_processor_backup,
            source_selection_scope=source_selection_scope,
        ),
        "dayu",
        cluster_client=cluster,
        runtime_client=runtime,
        session_store=sessions,
        request=scheduler,
        clock=clock,
        wall_clock=wall_clock,
    )
    return orchestrator, cluster, runtime, sessions, scheduler, events


def test_absent_session_snapshot_is_loaded_once_and_stays_absent_after_delete():
    orchestrator, _, _, sessions, _, _ = make_orchestrator({"detector": ["edge-a"]})

    assert orchestrator.current_session() is None
    assert orchestrator.current_session() is None
    assert sessions.load_calls == 1

    install(orchestrator, "detector")
    assert sessions.load_calls == 2  # one deliberate transaction-boundary reload
    assert orchestrator.current_session() is not None
    assert sessions.load_calls == 2

    orchestrator.uninstall()
    calls_after_delete = sessions.load_calls
    assert orchestrator.current_session() is None
    assert orchestrator.current_session() is None
    assert sessions.load_calls == calls_after_delete


def test_first_session_create_accepts_only_exact_response_lost_readback():
    orchestrator, _, _, _, _, _ = make_orchestrator({"detector": ["edge-a"]})
    sessions = UncertainSessionStore()
    orchestrator._sessions = sessions
    desired = RuntimeSession(
        install_id=INSTALL_ID,
        operation_id="create-operation",
        phase="activating-scheduler",
    )
    sessions.lose_next_response = True

    saved = orchestrator._save(desired)

    assert saved == desired
    assert orchestrator.current_session() == desired
    assert sessions.load_calls == 1


def test_begin_uninstall_accepts_exact_response_lost_readback():
    orchestrator, _, _, _, _, _ = make_orchestrator({"detector": ["edge-a"]})
    sessions = UncertainSessionStore()
    orchestrator._sessions = sessions
    active = RuntimeSession(
        install_id=INSTALL_ID,
        operation_id="install-operation",
        phase="active",
        active_directory_revision=1,
    )
    orchestrator._save(active)
    sessions.lose_next_response = True

    saved = orchestrator.begin_uninstall(INSTALL_ID)

    assert saved.phase == "uninstalling"
    assert saved.install_id == INSTALL_ID
    assert saved.operation_id != active.operation_id
    assert orchestrator.current_session() == saved
    assert sessions.load_calls == 2


def test_uncertain_session_write_never_accepts_a_different_readback():
    orchestrator, _, _, _, _, _ = make_orchestrator({"detector": ["edge-a"]})
    sessions = UncertainSessionStore()
    orchestrator._sessions = sessions
    active = RuntimeSession(
        install_id=INSTALL_ID,
        operation_id="install-operation",
        phase="active",
        active_directory_revision=1,
    )
    orchestrator._save(active)
    desired = replace(
        active,
        operation_id="stop-operation",
        phase="uninstalling",
    )
    replacement = replace(
        active,
        operation_id="other-operation",
        phase="failed",
    )
    sessions.lose_next_response = True
    sessions.replace_after_lost_response = replacement

    with pytest.raises(RuntimeError, match="write response lost"):
        orchestrator._save(desired)

    assert orchestrator.current_session() == replacement


def test_double_uncertain_session_io_forces_next_management_read_to_reload():
    orchestrator, _, _, _, _, _ = make_orchestrator({"detector": ["edge-a"]})
    sessions = UncertainSessionStore()
    orchestrator._sessions = sessions
    desired = RuntimeSession(
        install_id=INSTALL_ID,
        operation_id="create-operation",
        phase="activating-scheduler",
    )
    sessions.lose_next_response = True
    sessions.fail_next_load = True

    with pytest.raises(RuntimeError, match="write response lost"):
        orchestrator._save(desired)

    assert orchestrator._snapshot_loaded is False
    assert orchestrator.current_session() == desired


def test_lazy_snapshot_load_cannot_overwrite_a_transaction_reload():
    class BlockingStore(FakeSessionStore):
        def __init__(self):
            super().__init__()
            self.first_load_started = threading.Event()
            self.release_first_load = threading.Event()

        def load(self):
            self.load_calls += 1
            snapshot = self.stored
            if self.load_calls == 1:
                self.first_load_started.set()
                assert self.release_first_load.wait(1)
            return snapshot

    orchestrator, _, _, _, _, _ = make_orchestrator({"detector": ["edge-a"]})
    store = BlockingStore()
    orchestrator._sessions = store
    marker = object()
    results = {}

    reader = threading.Thread(
        target=lambda: results.setdefault("reader", orchestrator.current_session())
    )
    reader.start()
    assert store.first_load_started.wait(1)

    store.stored = StoredRuntimeSession(marker, "1", "session-uid")
    transaction_done = threading.Event()

    def reload_transaction():
        results["transaction"] = orchestrator._reload_for_transaction()
        transaction_done.set()

    transaction = threading.Thread(target=reload_transaction)
    transaction.start()
    assert transaction_done.wait(0.05) is False

    store.release_first_load.set()
    reader.join(timeout=1)
    transaction.join(timeout=1)

    assert results["reader"] is None
    assert results["transaction"] is store.stored
    assert orchestrator.current_session() is marker
    assert store.load_calls == 2


def test_runtime_config_requires_a_positive_retirement_grace():
    class InvalidTimeouts(FakeTemplateHelper):
        def load_base_info(self):
            value = super().load_base_info()
            value["runtime"]["retirement-grace-seconds"] = 0
            return value

    with pytest.raises(ValueError, match="timeouts must be positive"):
        RuntimeOrchestrator(InvalidTimeouts(), "dayu")


@pytest.mark.parametrize("value", ["true", 1, None, []])
def test_runtime_config_rejects_non_boolean_default_cloud_processor_backup(value):
    class InvalidCloudBackup(FakeTemplateHelper):
        def load_base_info(self):
            base_info = super().load_base_info()
            base_info["default-cloud-processor-backup"] = value
            return base_info

    with pytest.raises(ValueError, match="must be a boolean"):
        RuntimeOrchestrator(InvalidCloudBackup(), "dayu")


def install(orchestrator, *services):
    return orchestrator.install(
        {"id": "fixed"}, source_deploy(*services), source_label="camera-a",
        install_id=INSTALL_ID,
    )


def test_install_activates_scheduler_first_then_binds_observed_hash_and_endpoint_uids():
    orchestrator, cluster, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )

    directory = install(orchestrator, "detect")

    assert runtime.wait_batches[0] == ("scheduler",)
    assert "scheduler" not in runtime.wait_batches[1]
    assert set(runtime.wait_batches[1]) == {
        "distributor", "controller", "monitor", "generator", "processor",
    }
    assert sessions.saved_phases == [
        "activating-scheduler", "activating-scheduler", "activating-runtime",
        "publishing", "active",
    ]
    assert sessions.stored.session.install_id == INSTALL_ID
    assert sessions.stored.session.active_directory_revision == 1
    assert directory == sessions.stored.session.directory
    assert all(unit.spec_hash == HASH for unit in directory.routes)
    endpoint_units = [unit for unit in directory.routes if unit.endpoint is not None]
    assert endpoint_units
    assert all(unit.endpoint.runtime_service_uid.startswith("runtime-") for unit in endpoint_units)
    assert all(unit.endpoint.service_uid.startswith("service-") for unit in endpoint_units)
    assert all(unit.endpoint.pod_uid.startswith("pod-") for unit in endpoint_units)
    assert cluster.inventory_calls == 1
    assert cluster.preflight_calls == [("cloud-a", "edge-a", "edge-b")]
    assert scheduler.directory["hash"] == directory.content_hash
    assert {
        unit.slot.target_node for unit in directory.routes
        if unit.slot.component == "controller"
    } == {"cloud-a", "edge-a", "edge-b"}
    assert {
        unit.slot.target_node for unit in directory.routes
        if unit.slot.component == "monitor"
    } == {"cloud-a", "edge-a", "edge-b"}
    assert directory.deployment == {"detect": ["edge-a"]}
    edge_monitor = next(
        manifest for manifest in runtime.created.values()
        if manifest["spec"]["component"] == "monitor"
        and manifest["spec"]["targetNode"] == "edge-a"
    )
    env = {
        item["name"]: item["value"]
        for item in edge_monitor["spec"]["podTemplate"]["spec"]["containers"][0]["env"]
    }
    bootstrap = json.loads(env["DAYU_RUNTIME_BOOTSTRAP"])
    assert any(
        endpoint.get("component") == "monitor"
        and endpoint.get("target_node") == "cloud-a"
        and endpoint.get("port") == 9000
        for endpoint in bootstrap["endpoints"]
    )


def test_install_rejects_noncanonical_install_identity_before_cluster_io():
    orchestrator, cluster, _, _, _, _ = make_orchestrator({"detect": ["edge-a"]})

    with pytest.raises(ValueError, match="canonical UUID"):
        orchestrator.install(
            {"id": "fixed"},
            source_deploy("detect"),
            source_label="camera-a",
            install_id="not-a-uuid",
        )

    assert cluster.inventory_calls == 0


def test_install_specializes_edge_monitor_and_processor_images_for_node_jetpack():
    orchestrator, cluster, runtime, _, _, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    cluster.nodes["edge-a"]["labels"] = {"jetson.nvidia.com/jetpack.major": "6"}

    install(orchestrator, "detect")

    assert cluster.inventory_calls == 1
    containers = {
        (manifest["spec"]["component"], manifest["spec"]["targetNode"]):
            manifest["spec"]["podTemplate"]["spec"]["containers"][0]
        for manifest in runtime.created.values()
    }
    edge_monitor = containers[("monitor", "edge-a")]
    edge_processor = containers[("processor", "edge-a")]
    cloud_monitor = containers[("monitor", "cloud-a")]

    assert edge_monitor["image"] == "monitor:v1-jp6"
    assert edge_processor["image"] == "processor:test-jp6"
    assert {item["name"]: item["value"] for item in edge_monitor["env"]}["JETPACK"] == "6"
    assert {item["name"]: item["value"] for item in edge_processor["env"]}["JETPACK"] == "6"
    assert cloud_monitor["image"] == "monitor:v1"
    assert "JETPACK" not in {item["name"] for item in cloud_monitor["env"]}


@pytest.mark.parametrize(
    ("label", "message"),
    [
        ("invalid", "invalid JetPack major label"),
        ("7", "unsupported JetPack major 7"),
        ("0", "unsupported JetPack major 0"),
    ],
)
def test_node_image_specialization_rejects_invalid_or_unpublished_jetpack_variants(label, message):
    orchestrator, cluster, _, _, _, _ = make_orchestrator({"detect": ["edge-a"]})
    cluster.nodes["edge-a"]["labels"] = {"jetson.nvidia.com/jetpack.major": label}

    with pytest.raises(RuntimeOrchestrationError, match=message):
        orchestrator._specialize_template_for_node(
            {"pod-template": {"image": "monitor:v1"}},
            "edge-a",
            cluster.nodes,
        )


def test_install_propagates_one_cancellation_token_through_watch_and_scheduler_calls():
    orchestrator, _, runtime, _, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    cancel_event = threading.Event()
    wait_tokens = []
    scheduler_tokens = []
    scheduler_paths = []
    original_wait = runtime.wait_for_conditions

    def capture_wait(*args, **kwargs):
        wait_tokens.append(kwargs.get("cancel_event"))
        return original_wait(*args, **kwargs)

    def capture_scheduler(url, method, **kwargs):
        scheduler_tokens.append(kwargs.get("cancel_event"))
        scheduler_paths.append(url.split(":9000", 1)[1])
        return scheduler(url, method, **kwargs)

    runtime.wait_for_conditions = capture_wait
    orchestrator._request = capture_scheduler

    directory = orchestrator.install(
        {"id": "fixed"},
        source_deploy("detect"),
        source_label="camera-a",
        install_id=INSTALL_ID,
        cancel_event=cancel_event,
    )

    assert directory.revision == 1
    assert wait_tokens == [cancel_event, cancel_event]
    assert scheduler_paths == [
        "/source_nodes_selection",
        "/initial_deployment",
        "/runtime-directory",
    ]
    assert scheduler_tokens == [cancel_event] * len(scheduler_paths)


def test_cancelled_scheduler_activation_is_bounded_and_uninstalls_without_scheduler_io():
    orchestrator, _, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    activation_started = threading.Event()
    cancel_event = threading.Event()
    errors = []

    def cancellable_wait(expectations, **kwargs):
        assert tuple(expectations.values())[0].slot.component == "scheduler"
        assert kwargs.get("cancel_event") is cancel_event
        activation_started.set()
        assert cancel_event.wait(2)
        raise RuntimeServiceCancelled("RuntimeService wait was cancelled")

    runtime.wait_for_conditions = cancellable_wait

    def run_install():
        try:
            orchestrator.install(
                {"id": "fixed"},
                source_deploy("detect"),
                source_label="camera-a",
                install_id=INSTALL_ID,
                cancel_event=cancel_event,
            )
        except Exception as exc:  # asserted below from the worker thread
            errors.append(exc)

    worker = threading.Thread(target=run_install)
    worker.start()
    assert activation_started.wait(1)

    started_at = time.monotonic()
    cancel_event.set()
    worker.join(timeout=1)
    elapsed = time.monotonic() - started_at

    assert worker.is_alive() is False
    assert elapsed < 0.5
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeOperationCancelled)
    cancelled = sessions.stored.session
    assert cancelled.phase == "activating-scheduler"
    assert cancelled.last_error == ""
    assert len(cancelled.pending) == 1
    scheduler_runtime_id = cancelled.pending[0].runtime_id
    assert scheduler_runtime_id in runtime.created
    assert scheduler.calls == []

    orchestrator.uninstall("22222222-2222-4222-8222-222222222222")
    assert sessions.stored.session.install_id == INSTALL_ID

    orchestrator.uninstall(INSTALL_ID)

    assert sessions.stored is None
    assert sessions.deleted is True
    assert orchestrator.current_session() is None
    assert runtime.created == {}
    assert runtime.deleted == [
        (scheduler_runtime_id, f"runtime-{scheduler_runtime_id}", "scheduler"),
    ]
    assert scheduler.calls == []
    assert "failed" not in sessions.saved_phases


def test_activation_checks_cancellation_between_individual_creates():
    orchestrator, _, runtime, _, _, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    renderer = FakeRenderer("cancel-create")
    rendered = (
        renderer.render(
            {"component": "controller"},
            RuntimeSlot("controller", "edge-a", "edge"),
            1,
        ),
        renderer.render(
            {"component": "monitor"},
            RuntimeSlot("monitor", "edge-a", "edge"),
            1,
        ),
    )
    cancel_event = threading.Event()
    original_create = runtime.create
    create_calls = []

    def cancel_after_first_create(manifest, request_timeout_seconds=None):
        create_calls.append(manifest["metadata"]["name"])
        result = original_create(
            manifest,
            request_timeout_seconds=request_timeout_seconds,
        )
        cancel_event.set()
        return result

    runtime.create = cancel_after_first_create

    with pytest.raises(RuntimeOperationCancelled, match="lifecycle operation"):
        orchestrator._activate(rendered, cancel_event=cancel_event)

    assert create_calls == [rendered[0].unit.runtime_id]
    assert rendered[1].unit.runtime_id not in runtime.created


def test_unresolved_deletion_checks_cancellation_after_identity_list():
    orchestrator, _, runtime, _, _, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    unit = FakeRenderer("cancel-list").render(
        {"component": "processor"},
        RuntimeSlot("processor", "edge-a", "edge", logical_service="detect"),
        1,
    ).unit
    cancel_event = threading.Event()
    delete_many_called = []

    def list_then_cancel(label_selector=None, request_timeout_seconds=None):
        cancel_event.set()
        return {"items": []}

    runtime.list = list_then_cancel
    runtime.delete_many = lambda *args, **kwargs: delete_many_called.append(True)

    with pytest.raises(RuntimeOperationCancelled, match="lifecycle operation"):
        orchestrator._delete_units(
            (unit,),
            "cancel-list",
            cancel_event=cancel_event,
        )

    assert delete_many_called == []


def test_install_fails_before_create_when_managed_agents_do_not_cover_targets():
    orchestrator, _, runtime, sessions, _, _ = make_orchestrator(
        {"detect": ["edge-a"]}, agents_ok=False,
    )

    with pytest.raises(RuntimePreflightError, match="prerequisites"):
        install(orchestrator, "detect")

    assert runtime.created == {}
    assert sessions.stored is None


def test_install_rejects_invalid_source_scope_before_kubernetes_snapshot():
    orchestrator, cluster, runtime, sessions, _, _ = make_orchestrator(
        {"detect": ["edge-a"]}, source_selection_scope="cluster",
    )

    with pytest.raises(ValueError, match="source selection scope"):
        install(orchestrator, "detect")

    assert cluster.inventory_calls == 0
    assert cluster.preflight_calls == []
    assert runtime.created == {}
    assert sessions.stored is None


def test_install_rejects_non_edge_processor_candidates_before_runtime_creation():
    orchestrator, cluster, runtime, sessions, _, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    cluster.nodes["edge-a"]["role"] = "worker"

    with pytest.raises(RuntimePreflightError, match="must be edge nodes"):
        install(orchestrator, "detect")

    assert runtime.created == {}
    assert sessions.stored is None


def test_selected_edge_scope_rejects_scheduler_source_outside_processor_candidates():
    orchestrator, cluster, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    cluster.nodes["edge-c"] = {
        "name": "edge-c", "role": "edge", "address": "10.0.0.4",
        "ready": True, "labels": {},
    }
    scheduler.source_plan = {"1": "edge-c"}

    with pytest.raises(RuntimeOrchestrationError, match="selected unexpected node 'edge-c'"):
        install(orchestrator, "detect")

    assert sessions.stored.session.phase == "failed"
    assert cluster.preflight_calls == [("cloud-a", "edge-a", "edge-b")]
    assert not any(
        manifest["spec"]["component"] == "generator"
        for manifest in runtime.created.values()
    )


def test_all_edge_scope_allows_source_outside_processor_candidates_from_one_snapshot():
    orchestrator, cluster, _, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]}, source_selection_scope="all_edge_nodes",
    )
    cluster.nodes["edge-c"] = {
        "name": "edge-c", "role": "edge", "address": "10.0.0.4",
        "ready": True, "labels": {},
    }
    scheduler.source_plan = {"1": "edge-c"}

    directory = install(orchestrator, "detect")

    assert directory.deployment == {"detect": ["edge-a"]}
    assert next(
        unit.slot.target_node for unit in directory.routes
        if unit.slot.component == "generator"
    ) == "edge-c"
    assert {
        unit.slot.target_node for unit in directory.routes
        if unit.slot.component == "controller"
    } == {"cloud-a", "edge-a", "edge-b", "edge-c"}
    persisted_source = sessions.stored.session.source_deploy[0]
    assert persisted_source["node_set"] == ["edge-a", "edge-b"]
    assert persisted_source["source_candidate_nodes"] == ["edge-a", "edge-b", "edge-c"]
    assert persisted_source["source_selection_scope"] == "all_edge_nodes"
    selection_payload = next(
        payload for _, path, payload, _ in scheduler.calls
        if path == "/source_nodes_selection"
    )
    assert selection_payload[0]["source_candidate_nodes"] == [
        "edge-a", "edge-b", "edge-c",
    ]
    assert cluster.inventory_calls == 1
    assert cluster.preflight_calls == [("cloud-a", "edge-a", "edge-b", "edge-c")]


def test_all_edge_scope_excludes_optional_nodes_without_managed_agents():
    orchestrator, cluster, _, sessions, _, _ = make_orchestrator(
        {"detect": ["edge-a"]},
        source_selection_scope="all_edge_nodes",
        managed_nodes={"cloud-a", "edge-a", "edge-b"},
    )
    cluster.nodes["edge-c"] = {
        "name": "edge-c", "role": "edge", "address": "10.0.0.4",
        "ready": True, "labels": {},
    }

    install(orchestrator, "detect")

    assert sessions.stored.session.source_deploy[0]["source_candidate_nodes"] == [
        "edge-a", "edge-b",
    ]
    assert cluster.preflight_calls == [("cloud-a", "edge-a", "edge-b", "edge-c")]


@pytest.mark.parametrize(
    ("enabled", "plan", "expected"),
    [
        (False, {"detect": ["edge-a"]}, ["edge-a"]),
        (False, {"detect": ["cloud-a"]}, ["cloud-a"]),
        (True, {"detect": ["edge-a"]}, ["cloud-a", "edge-a"]),
        (True, {"detect": ["cloud-a"]}, ["cloud-a"]),
    ],
)
def test_default_cloud_processor_backup_composes_exact_desired_placement(
        enabled, plan, expected,
):
    orchestrator, cluster, _, _, _, _ = make_orchestrator(
        plan,
        default_cloud_processor_backup=enabled,
    )

    directory = install(orchestrator, "detect")

    assert directory.deployment == {"detect": expected}
    assert cluster.inventory_calls == 1
    assert cluster.preflight_calls == [("cloud-a", "edge-a", "edge-b")]


def test_default_cloud_processor_backup_applies_to_every_logical_service():
    orchestrator, _, _, _, _, _ = make_orchestrator(
        {
            "detect": ["edge-a"],
            "classify": ["edge-b"],
        },
        default_cloud_processor_backup=True,
    )

    directory = install(orchestrator, "detect", "classify")

    assert directory.deployment == {
        "classify": ["cloud-a", "edge-b"],
        "detect": ["cloud-a", "edge-a"],
    }


def test_cloud_backup_does_not_repair_a_plan_that_omits_a_logical_service():
    orchestrator, _, runtime, sessions, _, _ = make_orchestrator(
        {"detect": ["edge-a"]},
        default_cloud_processor_backup=True,
    )

    with pytest.raises(RuntimeOrchestrationError, match="omitted services.*classify"):
        install(orchestrator, "detect", "classify")

    assert runtime.wait_batches == [("scheduler",)]
    assert sessions.stored.session.phase == "failed"
    assert not any(
        manifest["spec"]["component"] == "processor"
        for manifest in runtime.created.values()
    )


def test_deployment_contract_rejects_node_oriented_and_scalar_policy_outputs():
    sources = source_deploy("detect")
    orchestrator, _, _, _, _, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )

    with pytest.raises(RuntimeOrchestrationError, match="unknown logical service 'edge-a'"):
        orchestrator._deployment(
            {"edge-a": ["detect"]}, sources, "cloud-a",
        )
    with pytest.raises(RuntimeOrchestrationError, match="JSON list"):
        orchestrator._deployment(
            {"detect": "edge-a"}, sources, "cloud-a",
        )


def test_initial_directory_uses_readback_when_put_ack_is_ambiguous():
    orchestrator, _, _, _, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]}, initial_put_ack=False,
    )

    directory = install(orchestrator, "detect")

    directory_calls = [
        (method, path) for method, path, _, _ in scheduler.calls
        if path == "/runtime-directory"
    ]
    assert directory_calls == [("PUT", "/runtime-directory"), ("GET", "/runtime-directory")]
    assert scheduler.directory["hash"] == directory.content_hash


def test_redeploy_commit_atomically_arms_retirement_then_reconcile_deletes_old_units():
    orchestrator, _, runtime, sessions, scheduler, events = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    initial = install(orchestrator, "detect")
    old_processor = next(
        unit for unit in initial.routes
        if unit.slot.component == "processor" and unit.slot.logical_service == "detect"
    )
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    start = len(events)

    assert orchestrator.redeploy({"id": "fixed"}) is True

    rollout_events = events[start:]
    candidate_activation = rollout_events.index("activate:processor")
    proposal = next(
        index for index, event in enumerate(rollout_events)
        if event == "scheduler:POST:/runtime-directory/proposals"
    )
    commit = next(
        index for index, event in enumerate(rollout_events)
        if event.startswith("scheduler:POST:/runtime-directory/proposals/")
        and event.endswith("/commit")
    )
    assert candidate_activation < proposal < commit
    assert not any("/runtime-directory/task-leases" in event for event in rollout_events)
    assert "delete:processor" not in rollout_events
    commit_payload = next(
        payload for method, path, payload, _ in scheduler.calls
        if method == "POST" and path.endswith("/commit")
    )
    assert commit_payload == {
        "expected_revision": 1,
        "retirement_grace_seconds": 10.0,
    }
    assert scheduler.retirements[1]["deadline"] == 1_000_010

    active = sessions.stored.session
    assert active.phase == "active"
    assert active.active_directory_revision == 2
    assert active.next_runtime_revision == 3
    assert active.retirement is not None
    assert active.retirement.revision == 1
    assert active.retirement.units == (old_processor,)
    assert active.retirement.deadline == 1_000_010
    new_processor = next(
        unit for unit in active.active
        if unit.slot.component == "processor" and unit.slot.logical_service == "detect"
    )
    assert new_processor.slot.target_node == "cloud-a"
    assert new_processor.runtime_revision == 2
    assert old_processor.runtime_id in runtime.created
    assert scheduler.directory["hash"] == active.directory.content_hash

    reconcile_start = len(events)
    assert orchestrator.reconcile_retirement() is True

    reconcile_events = events[reconcile_start:]
    status = reconcile_events.index("scheduler:GET:/runtime-directory/task-leases")
    deletion = reconcile_events.index("delete:processor")
    assert status < deletion
    assert sessions.stored.session.retirement is None
    assert old_processor.runtime_id not in runtime.created


def test_redeploy_and_committed_reads_do_not_wait_for_active_old_revision_leases():
    orchestrator, _, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    initial = install(orchestrator, "detect")
    old_processor = next(
        unit for unit in initial.routes if unit.slot.component == "processor"
    )
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    scheduler.lease_counts[1] = 1

    assert orchestrator.redeploy({"id": "fixed"}) is True
    load_calls = sessions.load_calls
    assert not any(
        method == "PATCH" and path == "/runtime-directory/task-leases"
        for method, path, _, _ in scheduler.calls
    )
    assert orchestrator.active_directory().revision == 2
    assert orchestrator.active_directory() == sessions.stored.session.directory
    assert orchestrator.node_inventory()["edge-a"]["ready"] is True
    assert old_processor.runtime_id in runtime.created

    assert orchestrator.reconcile_retirement() is False
    assert sessions.load_calls == load_calls
    assert sessions.stored.session.retirement is not None
    assert old_processor.runtime_id in runtime.created


def test_retirement_deadline_revokes_stuck_leases_and_completes_cleanup():
    clock = FakeClock()
    orchestrator, _, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]}, clock=clock,
    )
    initial = install(orchestrator, "detect")
    old_processor = next(
        unit for unit in initial.routes if unit.slot.component == "processor"
    )
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    scheduler.lease_counts[1] = 2
    assert orchestrator.redeploy({"id": "fixed"}) is True

    assert orchestrator.reconcile_retirement() is False
    clock.value = orchestrator.retirement_grace
    assert orchestrator.reconcile_retirement() is True

    assert scheduler.retirements[1] == {
        "deadline": 1_000_010,
        "retired": True,
        "revoked_count": 2,
    }
    assert sessions.stored.session.retirement is None
    assert old_processor.runtime_id not in runtime.created


def test_retirement_deadline_releases_rollout_gate_when_fence_and_cleanup_fail():
    clock = FakeClock()
    orchestrator, _, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]}, clock=clock,
    )
    initial = install(orchestrator, "detect")
    old_processor = next(
        unit for unit in initial.routes if unit.slot.component == "processor"
    )
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    scheduler.lease_counts[1] = 1
    assert orchestrator.redeploy({"id": "fixed"}) is True

    clock.value = orchestrator.retirement_grace
    scheduler.lease_failures = 1
    # Retirement and cleanup are separate lanes in one tick, so fail both the
    # immediate retirement deletion and the independent cleanup retry.
    runtime.delete_many_failures["processor"] = 2

    assert orchestrator.reconcile_retirement() is True

    deferred = sessions.stored.session
    assert deferred.phase == "active"
    assert deferred.retirement is None
    assert deferred.cleanup == (RuntimeCleanupRef.from_unit(old_processor),)
    assert deferred.last_error == "transient processor batch deletion failure"
    assert old_processor.runtime_id in runtime.created

    # The immutable deadline, rather than Scheduler/finalizer availability,
    # releases the single-rollout gate. The current plan is therefore a normal
    # no-op instead of another RuntimeRetirementPending failure.
    assert orchestrator.redeploy({"id": "fixed"}) is False
    assert sessions.stored.session.cleanup == (RuntimeCleanupRef.from_unit(old_processor),)

    assert orchestrator.reconcile_retirement() is True
    assert sessions.stored.session.cleanup == ()
    assert old_processor.runtime_id not in runtime.created


def test_pending_retirement_does_not_starve_existing_cleanup_backlog():
    orchestrator, _, runtime, sessions, scheduler, events = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    scheduler.lease_counts[1] = 1
    assert orchestrator.redeploy({"id": "fixed"}) is True

    orphan_id = "processor-orphan-r9"
    orphan_uid = f"runtime-{orphan_id}"
    runtime.created[orphan_id] = {
        "metadata": {"name": orphan_id, "uid": orphan_uid},
        "spec": {"component": "processor"},
    }
    current = sessions.stored.session
    orchestrator._save(replace(
        current,
        cleanup=(RuntimeCleanupRef(orphan_id, orphan_uid),),
    ))
    start = len(events)

    assert orchestrator.reconcile_retirement() is True

    reconcile_events = events[start:]
    status = reconcile_events.index(
        "scheduler:GET:/runtime-directory/task-leases"
    )
    cleanup = reconcile_events.index("delete:processor")
    assert status < cleanup
    assert sessions.stored.session.retirement is not None
    assert sessions.stored.session.cleanup == ()
    assert orphan_id not in runtime.created


def test_activating_rollout_failure_restores_old_active_and_defers_candidate_cleanup():
    orchestrator, _, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    initial = install(orchestrator, "detect")
    old_processor = next(
        unit for unit in initial.routes if unit.slot.component == "processor"
    )
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    wait_for_conditions = runtime.wait_for_conditions

    def fail_candidate_activation(expectations, **kwargs):
        units = tuple(expectations.values())
        if any(unit.slot.component == "processor" for unit in units):
            raise RuntimeError("candidate activation failed")
        return wait_for_conditions(expectations, **kwargs)

    runtime.wait_for_conditions = fail_candidate_activation

    with pytest.raises(RuntimeError, match="candidate activation failed"):
        orchestrator.redeploy({"id": "fixed"})

    restored = sessions.stored.session
    assert restored.phase == "active"
    assert restored.active_directory_revision == 1
    assert restored.active == initial.routes
    assert restored.pending == ()
    assert restored.retirement is None
    assert len(restored.cleanup) == 1
    candidate = restored.cleanup[0]
    expected_candidate_id = RuntimeSlot(
        "processor", "cloud-a", "cloud", logical_service="detect",
    ).runtime_name(2, initial.install_id)
    assert candidate.runtime_id == expected_candidate_id
    assert restored.next_runtime_revision == 3
    assert restored.last_error == "candidate activation failed"
    assert orchestrator.active_directory() == initial
    assert old_processor.runtime_id in runtime.created
    assert candidate.runtime_id in runtime.created

    assert orchestrator.reconcile_retirement() is True
    assert sessions.stored.session.cleanup == ()
    assert old_processor.runtime_id in runtime.created
    assert candidate.runtime_id not in runtime.created


def test_failed_candidate_cleanup_cannot_force_the_next_rollout_to_reuse_its_name():
    orchestrator, _, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    normal_wait = runtime.wait_for_conditions
    failed_once = {"value": False}

    def fail_first_candidate(expectations, **kwargs):
        units = tuple(expectations.values())
        if (
            not failed_once["value"]
            and any(unit.slot.component == "processor" for unit in units)
        ):
            failed_once["value"] = True
            raise RuntimeError("candidate activation failed")
        return normal_wait(expectations, **kwargs)

    runtime.wait_for_conditions = fail_first_candidate
    with pytest.raises(RuntimeError, match="candidate activation failed"):
        orchestrator.redeploy({"id": "fixed"})

    failed_candidate = sessions.stored.session.cleanup[0]
    runtime.delete_many_failures["processor"] = 1
    assert orchestrator.reconcile_retirement() is False
    assert sessions.stored.session.cleanup == (failed_candidate,)

    assert orchestrator.redeploy({"id": "fixed"}) is True
    active_candidate = next(
        unit for unit in sessions.stored.session.active
        if unit.slot.component == "processor"
    )
    assert failed_candidate.runtime_id.endswith("-r2")
    assert active_candidate.runtime_revision == 3
    assert failed_candidate.runtime_id != active_candidate.runtime_id
    assert failed_candidate in sessions.stored.session.cleanup


def test_cleanup_backlog_accepts_same_processor_slot_from_consecutive_revisions():
    clock = FakeClock()
    orchestrator, _, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]}, clock=clock,
    )
    first = install(orchestrator, "detect")
    revision_one = next(
        unit for unit in first.routes if unit.slot.component == "processor"
    )

    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    assert orchestrator.redeploy({"id": "fixed"}) is True
    revision_two = next(
        unit for unit in sessions.stored.session.active
        if unit.slot.component == "processor"
    )
    runtime.delete_many_failures["processor"] = 2
    assert orchestrator.reconcile_retirement() is True
    assert sessions.stored.session.cleanup == (
        RuntimeCleanupRef.from_unit(revision_one),
    )

    scheduler.redeployment_plan = {"detect": ["edge-a"]}
    assert orchestrator.redeploy({"id": "fixed"}) is True
    runtime.delete_many_failures["processor"] = 2
    assert orchestrator.reconcile_retirement() is True

    assert sessions.stored.session.cleanup == tuple(sorted(
        (
            RuntimeCleanupRef.from_unit(revision_one),
            RuntimeCleanupRef.from_unit(revision_two),
        ),
        key=lambda unit: unit.runtime_id,
    ))


def test_rollout_grace_is_armed_after_slow_directory_publication():
    clock = FakeClock()
    orchestrator, _, _, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]}, clock=clock,
    )
    install(orchestrator, "detect")
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    publish = orchestrator._publish_rollout

    def slow_publish(*args, **kwargs):
        clock.value += 5
        return publish(*args, **kwargs)

    orchestrator._publish_rollout = slow_publish
    assert orchestrator.redeploy({"id": "fixed"}) is True

    assert sessions.stored.session.retirement.deadline == (
        1_000_000 + 5 + orchestrator.retirement_grace
    )


def test_cleanup_refuses_to_delete_an_active_runtime_unit():
    orchestrator, _, runtime, sessions, _, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    active_processor = next(
        unit for unit in sessions.stored.session.active
        if unit.slot.component == "processor"
    )
    deleted_before = list(runtime.deleted)

    with pytest.raises(
        RuntimeOrchestrationError,
        match="refuse to garbage-collect active RuntimeServices",
    ):
        orchestrator._delete_units(
            (active_processor,),
            sessions.stored.session.install_id,
        )

    assert runtime.deleted == deleted_before
    assert active_processor.runtime_id in runtime.created


def test_redeploy_defers_immediately_while_previous_retirement_is_pending():
    orchestrator, _, _, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    assert orchestrator.redeploy({"id": "fixed"}) is True
    calls_before = len(scheduler.calls)

    with pytest.raises(RuntimeRetirementPending, match="still retiring"):
        orchestrator.redeploy({"id": "fixed"})

    preserved = sessions.stored.session
    assert preserved.phase == "active"
    assert preserved.retirement is not None
    assert len(scheduler.calls) == calls_before


def test_cancelled_retirement_reconcile_does_no_scheduler_io():
    orchestrator, _, _, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    assert orchestrator.redeploy({"id": "fixed"}) is True
    calls_before = len(scheduler.calls)
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(RuntimeOperationCancelled, match="lifecycle operation"):
        orchestrator.reconcile_retirement(cancel_event=cancel_event)

    assert len(scheduler.calls) == calls_before
    assert sessions.stored.session.retirement is not None


def test_redeploy_keeps_default_cloud_backup_while_replacing_only_changed_edge_slot():
    orchestrator, cluster, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
        default_cloud_processor_backup=True,
    )
    initial = install(orchestrator, "detect")
    initial_cloud = next(
        unit for unit in initial.routes
        if unit.slot.component == "processor" and unit.slot.target_node == "cloud-a"
    )
    initial_edge = next(
        unit for unit in initial.routes
        if unit.slot.component == "processor" and unit.slot.target_node == "edge-a"
    )
    inventory_calls = cluster.inventory_calls
    preflight_calls = list(cluster.preflight_calls)
    scheduler.redeployment_plan = {"detect": ["edge-b"]}

    assert orchestrator.redeploy({"id": "fixed"}) is True

    active = sessions.stored.session
    assert active.directory.deployment == {"detect": ["cloud-a", "edge-b"]}
    current_cloud = next(
        unit for unit in active.directory.routes
        if unit.slot.component == "processor" and unit.slot.target_node == "cloud-a"
    )
    current_edge = next(
        unit for unit in active.directory.routes
        if unit.slot.component == "processor" and unit.slot.target_node == "edge-b"
    )
    assert current_cloud.runtime_id == initial_cloud.runtime_id
    assert current_cloud.runtime_revision == initial_cloud.runtime_revision == 1
    assert current_edge.runtime_revision == 2
    assert initial_edge.runtime_id in runtime.created
    assert cluster.inventory_calls == inventory_calls
    assert cluster.preflight_calls == preflight_calls

    assert orchestrator.reconcile_retirement() is True
    assert initial_edge.runtime_id not in runtime.created


@pytest.mark.parametrize("default_cloud_processor_backup", [False, True])
def test_redeploy_with_identical_exact_placement_is_a_noop(
        default_cloud_processor_backup,
):
    orchestrator, cluster, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
        default_cloud_processor_backup=default_cloud_processor_backup,
    )
    install(orchestrator, "detect")
    created = tuple(runtime.created)
    saved_phases = list(sessions.saved_phases)
    call_count = len(scheduler.calls)
    inventory_calls = cluster.inventory_calls
    preflight_calls = list(cluster.preflight_calls)

    assert orchestrator.redeploy({"id": "fixed"}) is False

    assert tuple(runtime.created) == created
    assert sessions.saved_phases == saved_phases
    assert cluster.inventory_calls == inventory_calls
    assert cluster.preflight_calls == preflight_calls
    assert not any(
        path == "/runtime-directory/proposals"
        for _, path, _, _ in scheduler.calls[call_count:]
    )


def test_retirement_api_failure_keeps_committed_directory_active_and_retries_later():
    orchestrator, _, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    initial = install(orchestrator, "detect")
    old_processor = next(
        unit for unit in initial.routes if unit.slot.component == "processor"
    )
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    scheduler.lease_failures = 1

    assert orchestrator.redeploy({"id": "fixed"}) is True

    committed = sessions.stored.session
    assert committed.phase == "active"
    assert committed.active_directory_revision == 2
    assert committed.retirement.units == (old_processor,)
    assert old_processor.runtime_id in runtime.created

    assert orchestrator.reconcile_retirement() is False
    pending = sessions.stored.session
    assert pending.phase == "active"
    assert pending.last_error == "scheduler lease API unavailable"
    assert old_processor.runtime_id in runtime.created
    with pytest.raises(RuntimeRetirementPending):
        orchestrator.redeploy({"id": "fixed"})

    assert orchestrator.reconcile_retirement() is True

    recovered = sessions.stored.session
    assert recovered.phase == "active"
    assert recovered.retirement is None
    assert old_processor.runtime_id not in runtime.created
    assert orchestrator.redeploy({"id": "fixed"}) is False


def test_uninstall_stops_generators_then_fences_and_deletes_scheduler_before_workers():
    orchestrator, _, runtime, sessions, lease_scheduler, events = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    lease_scheduler.lease_counts[1] = 3
    start = len(events)

    orchestrator.uninstall()

    uninstall_events = events[start:]
    generator = uninstall_events.index("delete:generator")
    fence = uninstall_events.index("scheduler:PATCH:/runtime-directory/task-leases")
    clear = uninstall_events.index("scheduler:DELETE:/runtime-directory")
    scheduler = len(uninstall_events) - 1 - uninstall_events[::-1].index("delete:scheduler")
    all_deletes = [index for index, event in enumerate(uninstall_events) if event.startswith("delete:")]
    route_target_deletes = [
        index for index, event in enumerate(uninstall_events)
        if event.startswith("delete:") and event not in {"delete:generator", "delete:scheduler"}
    ]
    assert generator < fence < clear < scheduler < min(route_target_deletes)
    assert generator == min(all_deletes)
    assert lease_scheduler.retirements[1]["retired"] is True
    assert lease_scheduler.retirements[1]["revoked_count"] == 3
    assert sessions.deleted is True
    assert runtime.created == {}


def test_uninstall_deduplicates_failed_pending_transaction_and_still_stops_generator_first():
    orchestrator, _, runtime, sessions, _, events = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    renderer = FakeRenderer("failed-install")
    scheduler = renderer.render(
        {"component": "scheduler"}, RuntimeSlot("scheduler", "cloud-a", "cloud"), 1,
    )
    generator = renderer.render(
        {"component": "generator"},
        RuntimeSlot("generator", "edge-a", "edge", source_id="1"), 1,
    )
    for item in (scheduler, generator):
        runtime.create(item.manifest)
    failed = RuntimeSession(
        install_id="failed-install",
        operation_id="failed-operation",
        phase="failed",
        pending=(scheduler.unit, generator.unit),
    )
    sessions.revision = 1
    sessions.stored = StoredRuntimeSession(failed, "1", "session-uid")
    start = len(events)

    orchestrator.uninstall()

    uninstall_events = events[start:]
    assert uninstall_events.count("delete:generator") == 1
    assert uninstall_events.count("delete:scheduler") == 1
    assert uninstall_events.index("delete:generator") < uninstall_events.index("delete:scheduler")


def test_recover_promotes_initial_directory_committed_before_session_cas_without_kubernetes_discovery():
    orchestrator, cluster, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    committed = sessions.stored.session
    publishing = replace(
        committed,
        operation_id="recover-initial",
        phase="publishing",
        next_runtime_revision=1,
        active_directory_revision=0,
        active=(),
        pending=committed.active,
    )
    sessions.revision += 1
    sessions.stored = StoredRuntimeSession(
        publishing, str(sessions.revision), "session-uid",
    )
    orchestrator._stored = None
    inventory_calls = cluster.inventory_calls
    wait_batches = list(runtime.wait_batches)

    recovered = orchestrator.recover()

    assert recovered.phase == "active"
    assert recovered.active_directory_revision == 1
    assert recovered.directory.content_hash == committed.directory.content_hash
    assert recovered.pending == ()
    assert cluster.inventory_calls == inventory_calls
    assert runtime.wait_batches == wait_batches


@pytest.mark.parametrize("phase", ["activating-scheduler", "activating-runtime"])
def test_recover_marks_interrupted_initial_activation_failed_without_losing_ownership(
        phase,
):
    orchestrator, cluster, runtime, sessions, _, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    committed = sessions.stored.session
    interrupted = replace(
        committed,
        phase=phase,
        active=(),
        pending=committed.active,
        active_directory_revision=0,
    )
    sessions.revision += 1
    sessions.stored = StoredRuntimeSession(
        interrupted, str(sessions.revision), "session-uid",
    )
    orchestrator._stored = None
    inventory_calls = cluster.inventory_calls
    wait_batches = list(runtime.wait_batches)

    recovered = orchestrator.recover()

    assert recovered.phase == "failed"
    assert recovered.pending == interrupted.pending
    assert "backend restarted during" in recovered.last_error
    assert cluster.inventory_calls == inventory_calls
    assert runtime.wait_batches == wait_batches


def test_publication_recovery_persists_error_and_a_later_attempt_clears_it():
    orchestrator, _, _, sessions, _, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    committed = sessions.stored.session
    publishing = replace(
        committed,
        operation_id="recover-initial",
        phase="publishing",
        next_runtime_revision=1,
        active_directory_revision=0,
        active=(),
        pending=committed.active,
    )
    sessions.revision += 1
    sessions.stored = StoredRuntimeSession(
        publishing, str(sessions.revision), "session-uid",
    )
    orchestrator._stored = None
    recover_publication = orchestrator._recover_publication

    def fail_once(_session):
        raise RuntimeError("scheduler temporarily unavailable")

    orchestrator._recover_publication = fail_once
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        orchestrator.recover()

    failed_attempt = sessions.stored.session
    assert failed_attempt.phase == "publishing"
    assert failed_attempt.last_error == "scheduler temporarily unavailable"

    orchestrator._recover_publication = recover_publication
    recovered = orchestrator.recover()
    assert recovered.phase == "active"
    assert recovered.last_error == ""


def test_recover_promotes_committed_rollout_and_keeps_old_revision_retirement():
    orchestrator, cluster, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    old = sessions.stored.session
    old_processor = next(
        unit for unit in old.active if unit.slot.component == "processor"
    )
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    assert orchestrator.redeploy({"id": "fixed"}) is True
    committed = sessions.stored.session
    new_processor = next(
        unit for unit in committed.active if unit.slot.component == "processor"
    )
    publishing = replace(
        old,
        operation_id="recover-rollout",
        phase="publishing-rollout",
        pending=(new_processor,),
        retirement=committed.retirement,
    )
    sessions.revision += 1
    sessions.stored = StoredRuntimeSession(
        publishing, str(sessions.revision), "session-uid",
    )
    orchestrator._stored = None
    inventory_calls = cluster.inventory_calls
    wait_batches = list(runtime.wait_batches)

    recovered = orchestrator.recover()

    assert recovered.phase == "active"
    assert recovered.active_directory_revision == 2
    assert recovered.directory.content_hash == committed.directory.content_hash
    assert recovered.retirement == committed.retirement
    assert recovered.retirement.units == (old_processor,)
    assert cluster.inventory_calls == inventory_calls
    assert runtime.wait_batches == wait_batches


def test_redeploy_publication_recovery_propagates_cancellation_without_finalizing():
    orchestrator, _, _, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    old = sessions.stored.session
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    assert orchestrator.redeploy({"id": "fixed"}) is True
    committed = sessions.stored.session
    new_processor = next(
        unit for unit in committed.active
        if unit.slot.component == "processor"
    )
    publishing = replace(
        old,
        operation_id="cancel-recovery",
        phase="publishing-rollout",
        pending=(new_processor,),
        retirement=committed.retirement,
    )
    sessions.revision += 1
    sessions.stored = StoredRuntimeSession(
        publishing,
        str(sessions.revision),
        "session-uid",
    )
    cancel_event = threading.Event()
    request = orchestrator._request

    def cancel_after_scheduler_response(url, method, **kwargs):
        response = request(url, method, **kwargs)
        cancel_event.set()
        return response

    orchestrator._request = cancel_after_scheduler_response

    with pytest.raises(RuntimeOperationCancelled, match="lifecycle operation"):
        orchestrator.redeploy({"id": "fixed"}, cancel_event=cancel_event)

    assert sessions.stored.session == publishing
    assert sessions.stored.session.last_error == ""


def test_uninstall_immediately_fences_current_and_retiring_directory_revisions():
    orchestrator, _, _, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    old_processor = next(
        unit for unit in sessions.stored.session.active
        if unit.slot.component == "processor"
    )
    scheduler.redeployment_plan = {"detect": ["cloud-a"]}
    orchestrator.redeploy({"id": "fixed"})
    assert sessions.stored.session.retirement.units == (old_processor,)
    call_start = len(scheduler.calls)

    orchestrator.uninstall()

    revisions = [
        payload["revision"]
        for method, path, payload, _ in scheduler.calls[call_start:]
        if method == "PATCH" and path == "/runtime-directory/task-leases"
    ]
    assert revisions == [1, 2]


def test_uninstall_retry_after_scheduler_deleted_skips_fence_and_finishes_session_delete():
    orchestrator, _, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    sessions.delete_failures = 1

    with pytest.raises(RuntimeError, match="ConfigMap deletion"):
        orchestrator.uninstall()

    assert sessions.stored.session.phase == "finalizing-uninstall"
    fence_calls = len([
        call for call in scheduler.calls
        if call[0] == "PATCH" and call[1] == "/runtime-directory/task-leases"
    ])
    clear_calls = len([
        call for call in scheduler.calls
        if call[0] == "DELETE" and call[1] == "/runtime-directory"
    ])
    assert not any(
        manifest.get("spec", {}).get("component") == "scheduler"
        for manifest in runtime.created.values()
    )

    orchestrator.uninstall()

    assert sessions.deleted is True
    assert len([
        call for call in scheduler.calls
        if call[0] == "PATCH" and call[1] == "/runtime-directory/task-leases"
    ]) == fence_calls
    assert len([
        call for call in scheduler.calls
        if call[0] == "DELETE" and call[1] == "/runtime-directory"
    ]) == clear_calls


def test_uninstall_accepts_ambiguous_session_delete_when_readback_is_absent():
    orchestrator, _, _, sessions, _, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    delete = sessions.delete

    def delete_then_lose_response(expected_resource_version=None):
        delete(expected_resource_version=expected_resource_version)
        raise RuntimeError("ConfigMap DELETE response lost")

    sessions.delete = delete_then_lose_response

    orchestrator.uninstall()

    assert sessions.deleted is True
    assert sessions.stored is None
    assert orchestrator.current_session() is None


def test_uninstall_continues_exact_teardown_when_scheduler_fence_and_clear_are_unavailable():
    orchestrator, _, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    scheduler.lease_failures = 1
    scheduler.clear_failures = 1

    orchestrator.uninstall()

    assert sessions.deleted is True
    assert runtime.created == {}
    assert len([
        call for call in scheduler.calls
        if call[0] == "DELETE" and call[1] == "/runtime-directory"
    ]) == 1


def test_uninstall_accepts_empty_directory_readback_after_ambiguous_clear_ack():
    orchestrator, _, _, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    scheduler.clear_ack = False

    orchestrator.uninstall()

    assert sessions.deleted is True
    clear_calls = [
        (method, path) for method, path, _, _ in scheduler.calls
        if path == "/runtime-directory"
    ]
    assert ("DELETE", "/runtime-directory") in clear_calls
    assert clear_calls[-1] == ("GET", "/runtime-directory")


def test_uninstall_retries_route_target_deletion_from_finalizing_state():
    orchestrator, _, runtime, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    install(orchestrator, "detect")
    runtime.delete_many_failures["processor"] = 1

    with pytest.raises(RuntimeError, match="processor batch deletion"):
        orchestrator.uninstall()

    failed = sessions.stored.session
    assert failed.phase == "finalizing-uninstall"
    assert scheduler.directory is None
    assert any(unit.slot.component == "processor" for unit in failed.active)
    assert any(
        manifest.get("spec", {}).get("component") == "processor"
        for manifest in runtime.created.values()
    )

    orchestrator.uninstall()

    assert sessions.deleted is True
    assert runtime.created == {}


def test_install_checks_shared_operation_deadline_between_scheduler_decisions():
    clock = FakeClock()
    orchestrator, _, _, sessions, scheduler, _ = make_orchestrator(
        {"detect": ["edge-a"]}, clock=clock,
    )
    request = orchestrator._request

    def advance_after_source_selection(url, method, **kwargs):
        response = request(url, method, **kwargs)
        if url.endswith("/source_nodes_selection"):
            clock.value = orchestrator.operation_timeout
        return response

    orchestrator._request = advance_after_source_selection

    with pytest.raises(RuntimeOrchestrationError, match="deadline"):
        install(orchestrator, "detect")

    assert sessions.stored.session.phase == "failed"


def test_node_inventory_has_one_ttl_owned_by_backend_and_returns_isolated_snapshots():
    clock = FakeClock()
    orchestrator, cluster, _, _, _, _ = make_orchestrator(
        {"detect": ["edge-a"]}, clock=clock,
    )

    first = orchestrator.node_inventory()
    first["edge-a"]["ready"] = False
    assert orchestrator.node_inventory()["edge-a"]["ready"] is True
    assert cluster.inventory_calls == 1

    clock.value = 0.99
    orchestrator.node_inventory()
    assert cluster.inventory_calls == 1

    clock.value = 1.0
    orchestrator.node_inventory()
    assert cluster.inventory_calls == 2


def test_empty_node_inventory_is_negative_cached_until_the_same_ttl_expires():
    clock = FakeClock()
    orchestrator, cluster, _, _, _, _ = make_orchestrator(
        {"detect": ["edge-a"]}, clock=clock,
    )
    cluster.nodes = {}

    assert orchestrator.node_inventory() == {}
    assert orchestrator.node_inventory() == {}
    assert cluster.inventory_calls == 1

    clock.value = 1.0
    assert orchestrator.node_inventory() == {}
    assert cluster.inventory_calls == 2


def test_runtime_metrics_samples_bound_pod_uids_with_bounded_cached_inventory():
    orchestrator, cluster, _, _, _, _ = make_orchestrator(
        {"detect": ["edge-a"]},
    )
    directory = install(orchestrator, "detect")
    inventory_calls = cluster.inventory_calls

    processor = next(
        unit for unit in directory.routes
        if unit.slot.component == "processor" and unit.slot.logical_service == "detect"
    )
    refs = [{"name": processor.pod_name, "uid": processor.pod_uid}]

    assert orchestrator.sample_runtime_metrics(
        refs, request_timeout_seconds=4.0,
    ) == {}

    assert cluster.inventory_calls == inventory_calls
    sampled_refs, snapshot, timeout = cluster.metric_calls[-1]
    assert sampled_refs == refs
    assert refs == [{"name": processor.pod_name, "uid": processor.pod_uid}]
    assert snapshot == inventory()
    assert timeout == 4.0
