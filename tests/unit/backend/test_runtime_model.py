import json

import pytest

from runtime_model import (
    RuntimeDirectory,
    RuntimeEndpoint,
    RuntimeSession,
    RuntimeSlot,
    RuntimeUnit,
    canonical_hash,
    canonical_json,
)


def make_unit(service="face-detection", node="edge-x1", revision=3, endpoint=True):
    slot = RuntimeSlot(
        component="processor",
        logical_service=service,
        target_node=node,
        position="edge",
    )
    return RuntimeUnit(
        slot=slot,
        runtime_id=slot.runtime_name(revision),
        runtime_revision=revision,
        spec_hash="a" * 64,
        endpoint=RuntimeEndpoint(
            dns_name=f"{slot.runtime_name(revision)}.dayu.svc.cluster.local",
            port=9000,
            runtime_service_uid="runtime-uid",
            service_uid="service-uid",
            pod_uid="pod-uid",
        ) if endpoint else None,
    )


def test_runtime_slot_name_is_revision_scoped_dns_safe_and_collision_resistant():
    first = RuntimeSlot("processor", "edge_x1", "edge", logical_service="face_detection")
    alias = RuntimeSlot("processor", "edge-x1", "edge", logical_service="face-detection")
    long = RuntimeSlot("processor", "edge-x1", "edge", logical_service="a" * 240)

    assert first.runtime_name(7).endswith("-r7")
    assert first.runtime_name(7) != first.runtime_name(8)
    assert first.runtime_name(7) != alias.runtime_name(7)
    assert len(long.runtime_name(123456)) <= 63
    assert long.runtime_name(123456)[0].isalpha()


def test_slot_validates_required_component_specific_identity():
    with pytest.raises(ValueError, match="logical_service"):
        RuntimeSlot("processor", "edge-1", "edge")
    with pytest.raises(ValueError, match="source_id"):
        RuntimeSlot("generator", "edge-1", "edge")
    with pytest.raises(ValueError, match="position"):
        RuntimeSlot("scheduler", "cloud", "both")


def test_directory_has_flat_canonical_routes_hash_nodes_and_deployment():
    face = make_unit()
    vehicle = make_unit(service="vehicle-detection", node="edge-x2", revision=4)
    directory = RuntimeDirectory("install-1", revision=9, routes=(vehicle, face))

    value = directory.to_dict()
    assert value["revision"] == value["directory_revision"] == 9
    assert value["nodes"] == ["edge-x1", "edge-x2"]
    assert value["deployment"] == {
        "face-detection": ["edge-x1"],
        "vehicle-detection": ["edge-x2"],
    }
    assert [route["logical_service"] for route in value["routes"]] == [
        "face-detection", "vehicle-detection",
    ]
    assert value["routes"][0]["runtime_service_uid"] == "runtime-uid"
    assert value["hash"] == directory.content_hash
    assert value["hash"] == canonical_hash({
        key: value[key]
        for key in ("install_id", "directory_revision", "nodes", "deployment", "routes")
    })

    restored = RuntimeDirectory.from_dict(json.loads(canonical_json(value)))
    assert restored == directory
    assert restored.content_hash == directory.content_hash


def test_directory_rejects_duplicate_logical_slots_and_runtime_ids():
    unit = make_unit()
    with pytest.raises(ValueError, match="duplicate runtime slot"):
        RuntimeDirectory("install-1", 1, routes=(unit, unit))

    other_slot = RuntimeSlot("processor", "edge-x2", "edge", logical_service="face-detection")
    duplicate_id = RuntimeUnit(other_slot, unit.runtime_id, 3, "b" * 64)
    with pytest.raises(ValueError, match="duplicate runtime_id"):
        RuntimeDirectory("install-1", 1, routes=(unit, duplicate_id))


def test_runtime_session_round_trip_contains_only_transaction_state():
    active = make_unit()
    pending = make_unit(service="vehicle-detection", revision=4)
    session = RuntimeSession(
        install_id="install-1",
        operation_id="operation-42",
        phase="activating",
        next_runtime_revision=5,
        active_directory_revision=2,
        active=(active,),
        pending=(pending,),
        retired=(),
        source_label="source-a",
        policy_id="hedger",
        source_deploy=[{"source": {"id": 0}, "node_set": ["edge-x1"]}],
        last_error="",
        updated_at="2026-07-12T00:00:00Z",
    )

    value = session.to_dict()
    assert "manifest" not in canonical_json(value)
    assert value["active_directory_revision"] == 2
    assert session.directory.routes == (active,)
    assert RuntimeSession.from_dict(value) == session
    assert session.content_hash == canonical_hash(value)


def test_runtime_unit_replaces_provisional_hash_with_observed_hash_immutably():
    provisional = make_unit()
    observed = provisional.with_observed_spec_hash("b" * 64)

    assert provisional.spec_hash == "a" * 64
    assert observed.spec_hash == "b" * 64
    assert observed.slot == provisional.slot
    assert observed.endpoint == provisional.endpoint
    with pytest.raises(ValueError, match="spec_hash"):
        provisional.with_observed_spec_hash("")


def test_runtime_endpoint_uses_absolute_cluster_dns_only_at_connection_boundary():
    endpoint = RuntimeEndpoint("scheduler.dayu.svc.cluster.local", 9000)

    assert endpoint.connection_host == "scheduler.dayu.svc.cluster.local."
    assert endpoint.url_authority == "scheduler.dayu.svc.cluster.local.:9000"
    assert endpoint.to_dict() == {
        "dns_name": "scheduler.dayu.svc.cluster.local",
        "port": 9000,
    }

    already_absolute = RuntimeEndpoint("scheduler.dayu.svc.cluster.local.", 9000)
    assert already_absolute.connection_host == "scheduler.dayu.svc.cluster.local."


def test_endpointless_unit_persists_hidden_uid_identity_only_in_control_plane_state():
    slot = RuntimeSlot("monitor", "edge-x1", "edge")
    unit = RuntimeUnit(
        slot=slot,
        runtime_id=slot.runtime_name(3),
        runtime_revision=3,
        spec_hash="a" * 64,
        rollout_hash="b" * 64,
        runtime_service_uid="runtime-uid",
        pod_name="monitor-pod",
        pod_uid="pod-uid",
    )

    assert "resource_identity" not in unit.to_dict()
    state = unit.to_state_dict()
    assert state["rollout_hash"] == "b" * 64
    assert state["resource_identity"] == {
        "runtime_service_uid": "runtime-uid",
        "pod_name": "monitor-pod",
        "pod_uid": "pod-uid",
    }
    assert RuntimeUnit.from_dict(state) == unit
