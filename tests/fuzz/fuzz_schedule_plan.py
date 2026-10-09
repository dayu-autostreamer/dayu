"""Complete deployment/offloading coverage, candidate scoping and isolation."""

from copy import deepcopy

from _support import choice, decode, expect_error, integer, make_mutator, run

MODES = ("valid", "missing", "extra", "empty", "scalar", "outside", "blank", "wrong_type")


def check_input(data):
    from core.lib.content import Task
    from core.lib.scheduling.dag import START, END
    from core.lib.scheduling.deployment_plan import validate_plan, fixed_plan, cloud_replica_plan
    from core.lib.scheduling.offloading_plan import materialize_offloading_plan

    case = decode(data)
    services = ["service_{}".format(i) for i in range(integer(case.get("services"), 3, 1, 24))]
    nodes = ["edge_{}".format(i) for i in range(integer(case.get("nodes"), 2, 1, 12))]
    cloud = "cloud"
    dag = Task.extract_dag_from_dict(
        {
            name: {"service": {"service_name": name}, "next_nodes": services[i + 1 : i + 2]}
            for i, name in enumerate(services)
        }
    ).to_dict()
    info = {"dag": dag, "node_set": nodes}
    offsets = case.get("placements", [0, 1])
    offsets = offsets[:24] if isinstance(offsets, list) else [0]
    offsets = [value for value in offsets if type(value) is int] or [0]
    candidates = nodes + [cloud]
    plan = {
        name: [" {} ".format(candidates[(value + i) % len(candidates)]) for value in offsets]
        for i, name in enumerate(services)
    }
    before = deepcopy((info, plan))
    normalized = validate_plan(plan, info, cloud)
    assert (info, plan) == before
    assert set(normalized) == set(services)
    for name, targets in normalized.items():
        assert targets == sorted({node.strip() for node in plan[name]})
        assert set(targets) <= set(candidates)
    assert validate_plan(normalized, info, cloud) == normalized
    replicas = cloud_replica_plan(plan, info, cloud)
    assert replicas == {name: sorted(set(targets) | {cloud}) for name, targets in normalized.items()}
    fixed = {name: ["@cloud"] for name in services}
    fixed["unused_policy_service"] = ["unavailable_node"]
    assert fixed_plan(fixed, info, cloud) == {name: [cloud] for name in services}

    selected = {name: targets[0] for name, targets in normalized.items()}
    config = {"metadata": {"labels": ["original"]}, "dag": {"old": True}}
    original = deepcopy((config, dag, selected))
    result = materialize_offloading_plan(config, dag, selected, nodes[0], cloud)
    assert (config, dag, selected) == original
    assert result["dag"][START]["service"]["execute_device"] == nodes[0]
    assert result["dag"][END]["service"]["execute_device"] == cloud
    for name, target in selected.items():
        assert result["dag"][name]["service"]["execute_device"] == target
    result["metadata"]["labels"].append("changed")
    result["dag"][services[0]]["service"]["execute_device"] = "changed"
    result["dag"][services[0]]["next_nodes"].append("changed")
    assert (config, dag, selected) == original, "materialization must not alias its inputs"

    mode = choice(case.get("mode"), MODES)
    broken = deepcopy(plan)
    if mode == "valid":
        return
    if mode == "missing":
        del broken[services[0]]
        del selected[services[0]]
    elif mode == "extra":
        broken["unknown"] = [nodes[0]]
        selected["unknown"] = nodes[0]
    elif mode == "empty":
        broken[services[0]] = []
    elif mode == "scalar":
        broken[services[0]] = nodes[0]
    elif mode == "outside":
        broken[services[0]] = ["not_a_candidate"]
    elif mode == "blank":
        broken[services[0]] = [" \t"]
    elif mode == "wrong_type":
        broken = []
        expect_error(TypeError, materialize_offloading_plan, [], dag, selected, nodes[0], cloud)
        expect_error(TypeError, materialize_offloading_plan, config, dag, [], nodes[0], cloud)
    expect_error(ValueError, validate_plan, broken, info, cloud)
    if mode in ("missing", "extra"):
        expect_error(ValueError, materialize_offloading_plan, config, dag, selected, nodes[0], cloud)


mutate = make_mutator(
    {
        "services": lambda rng: rng.randint(1, 24),
        "nodes": lambda rng: rng.randint(1, 12),
        "placements": lambda rng: [rng.randint(-24, 24) for _ in range(rng.randrange(25))],
        "mode": lambda rng: rng.choice(MODES),
    }
)


if __name__ == "__main__":
    run(check_input, mutate)
