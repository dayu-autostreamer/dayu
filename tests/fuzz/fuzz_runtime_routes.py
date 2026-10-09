"""Endpoint shape normalization and exact, unambiguous task route resolution."""

from copy import deepcopy

from _support import choice, decode, expect_error, integer, make_mutator, run

MODES = ("valid", "missing", "invalid_port", "conflict", "no_route", "wrong_type")
SHAPES = ("snake", "camel", "nested", "url")
IDENTITIES = (
    "runtime_id",
    "runtime_service_uid",
    "service_uid",
    "endpoint_pod_uid",
    "deployment_revision",
    "fqdn",
    "target_node",
    "component",
    "logical_service",
)
CONFLICTS = (
    "runtime_id",
    "runtime_service_uid",
    "service_uid",
    "endpoint_pod_uid",
    "deployment_revision",
    "fqdn",
    "port",
    "protocol",
    "base_path",
)


def check_input(data):
    from core.lib.runtime import RuntimeContext, RuntimeEndpoint, RuntimeResolver

    case = decode(data)
    component = choice(case.get("component"), ("processor", "controller"))
    host = choice(case.get("host"), ("worker.dayu.svc.cluster.local", "127.0.0.1", "localhost"))
    port = integer(case.get("port"), 9000, 1, 65535)
    revision = integer(case.get("revision"), 1, 1, 2**31 - 1)
    path = choice(case.get("path"), ("", "/api/v1", "/predict/", "///nested//path///"))
    protocol = choice(case.get("protocol"), ("http", "https"))
    route = {
        "component": component,
        "target_node": "edge",
        "logical_service": "detector",
        "runtime_id": "runtime-{}".format(revision),
        "deployment_revision": revision,
        "runtime_service_uid": "runtime-uid",
        "service_uid": "service-uid",
        "endpoint_pod_uid": "pod-uid",
        "fqdn": host,
        "port": port,
        "protocol": protocol,
        "base_path": path,
    }
    shape = choice(case.get("shape"), SHAPES)
    payload = deepcopy(route)
    if shape == "camel":
        aliases = {
            "target_node": "targetNode",
            "logical_service": "logicalService",
            "runtime_id": "runtimeID",
            "deployment_revision": "deploymentRevision",
            "runtime_service_uid": "runtimeServiceUID",
            "service_uid": "serviceUID",
            "endpoint_pod_uid": "endpointPodUID",
            "base_path": "basePath",
        }
        payload = {aliases.get(key, key): value for key, value in payload.items()}
    elif shape == "nested":
        payload = {
            "slot": {key: route[key] for key in ("component", "target_node", "logical_service")},
            "runtime_id": route["runtime_id"],
            "runtime_revision": revision,
            "endpoint": {
                key: route[key]
                for key in (
                    "fqdn",
                    "port",
                    "protocol",
                    "base_path",
                    "runtime_service_uid",
                    "service_uid",
                    "endpoint_pod_uid",
                )
            },
        }
    elif shape == "url":
        for key in ("fqdn", "port", "protocol", "base_path"):
            del payload[key]
        payload["url"] = "{}://{}:{}{}".format(protocol, host, port, path)

    before = deepcopy(payload)
    endpoint = RuntimeEndpoint.from_value(payload).validate_exact()
    assert payload == before
    assert endpoint == RuntimeEndpoint.from_value(route)
    assert RuntimeEndpoint.from_value(endpoint.to_dict()) == endpoint
    connection_host = host + "." if host.endswith(".svc.cluster.local") else host
    normalized_path = "/" + path.strip("/") if path.strip("/") else ""
    assert endpoint.base_url == "{}://{}:{}{}".format(protocol, connection_host, port, normalized_path)
    assert endpoint.url("/predict") == endpoint.base_url.rstrip("/") + "/predict"

    context = RuntimeContext(
        {
            "local_node": "edge",
            "cloud_node": "cloud",
            "namespace": "dayu",
            "lease_ttl_seconds": 60,
            "endpoints": [route],
        }
    )
    resolver = RuntimeResolver(context)
    query = {"component": component, "target_node": "edge", "logical_service": "detector", "exact": True}
    # Exercise accepted task envelopes and identical duplicate entries.
    for routes in ([payload], {"routes": [payload]}, {"runtime_routes": [payload, deepcopy(payload)]}):
        assert resolver.resolve(task=routes, **query) == endpoint

    mode = choice(case.get("mode"), MODES)
    broken = deepcopy(route)
    if mode == "missing":
        field = choice(case.get("field"), IDENTITIES)
        if field == "logical_service" and component != "processor":
            field = "runtime_id"
        del broken[field]
        expect_error(ValueError, RuntimeEndpoint.from_value(broken).validate_exact)
    elif mode == "invalid_port":
        broken["port"] = choice(case.get("bad_port"), ("0", "-1", "65536", "invalid"))
        expect_error(ValueError, RuntimeEndpoint.from_value(broken).validate_exact)
    elif mode == "conflict":
        field = choice(case.get("field"), CONFLICTS)
        value = broken[field]
        broken[field] = (
            (value % 65535 + 1 if field == "port" else value + 1) if type(value) is int else value + "-other"
        )
        expect_error(ValueError, resolver.resolve, task=[route, broken], **query)
    elif mode == "no_route":
        # Bootstrap contains the route, but task workers must never fall back to it.
        expect_error(LookupError, resolver.resolve, task=[], **query)
        assert resolver.resolve(task=[], required=False, **query) is None
        query["exact"] = False
        expect_error(LookupError, resolver.resolve, task=[], **query)
    elif mode == "wrong_type":
        expect_error(TypeError, RuntimeEndpoint.from_value, [route])


mutate = make_mutator(
    {
        "component": lambda rng: rng.choice(("processor", "controller")),
        "shape": lambda rng: rng.choice(SHAPES),
        "mode": lambda rng: rng.choice(MODES),
        "field": lambda rng: rng.choice(IDENTITIES + CONFLICTS),
        "port": lambda rng: rng.choice((1, 65535, rng.randint(1, 65535))),
        "bad_port": lambda rng: rng.choice(("0", "-1", "65536", "invalid")),
        "revision": lambda rng: rng.randint(1, 2**31 - 1),
        "host": lambda rng: rng.choice(("worker.dayu.svc.cluster.local", "127.0.0.1", "localhost")),
        "path": lambda rng: rng.choice(("", "/api/v1", "/predict/", "///nested//path///")),
        "protocol": lambda rng: rng.choice(("http", "https")),
    }
)


if __name__ == "__main__":
    run(check_input, mutate)
