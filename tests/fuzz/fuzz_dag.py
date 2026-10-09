"""DAG repair, round trips, topological order and controlled invalid graphs."""

from copy import deepcopy

from _support import choice, decode, expect_error, integer, make_mutator, run

MODES = ("valid", "repair", "cycle", "duplicate", "dangling", "missing_start", "missing_end", "disconnected")


def check_input(data):
    from core.lib.content import DAG, Task
    from core.lib.scheduling.dag import START, END, topological_order

    case = decode(data)
    count = integer(case.get("nodes"), 3, 1, 24)
    suffix = case.get("suffix", "")
    suffix = suffix[:24] if isinstance(suffix, str) else ""
    names = ["service_{}_{}".format(index, suffix) for index in range(count)]
    raw = {name: {"service": {"service_name": name}, "next_nodes": []} for name in names}
    edges = case.get("edges", [[0, 1], [1, 2]])
    if isinstance(edges, list):
        for edge in edges[:96]:
            if not isinstance(edge, list) or len(edge) != 2:
                continue
            if not all(type(index) is int for index in edge):
                continue
            left, right = sorted(index % count for index in edge)
            if left != right and names[right] not in raw[names[left]]["next_nodes"]:
                raw[names[left]]["next_nodes"].append(names[right])

    dag = Task.extract_dag_from_dict(deepcopy(raw))
    canonical = deepcopy(dag.to_dict())
    dag.validate_dag()
    assert dag.to_dict() == canonical, "validation must be idempotent"
    restored = DAG.deserialize(dag.serialize())
    restored.validate_dag()
    assert restored.to_dict() == canonical
    order = topological_order(canonical)
    assert len(order) == len(set(order)) == len(canonical)
    assert set(order) == set(canonical)
    assert topological_order(dict(reversed(list(canonical.items())))) == order
    positions = {name: index for index, name in enumerate(order)}
    for name, node in canonical.items():
        for child in node["next_nodes"]:
            assert name in canonical[child]["prev_nodes"]
            assert positions[name] < positions[child]
        for parent in node["prev_nodes"]:
            assert name in canonical[parent]["next_nodes"]

    mode = choice(case.get("mode"), MODES)
    damaged = deepcopy(canonical)
    first = names[0]
    if mode == "valid":
        return
    if mode == "repair":
        for node in damaged.values():
            node["prev_nodes"] = []
        repaired = DAG.from_dict(damaged)
        repaired.validate_dag()
        for name, node in repaired.to_dict().items():
            assert set(node["prev_nodes"]) == set(canonical[name]["prev_nodes"])
            assert node["next_nodes"] == canonical[name]["next_nodes"]
        return
    if mode == "cycle":
        damaged[first]["next_nodes"].append(first)
    elif mode == "duplicate":
        damaged[START]["next_nodes"].append(damaged[START]["next_nodes"][0])
    elif mode == "dangling":
        damaged[first]["next_nodes"].append("absent_service")
    elif mode == "missing_start":
        del damaged[START]
    elif mode == "missing_end":
        del damaged[END]
    elif mode == "disconnected":
        damaged["isolated"] = {"service": {"service_name": "isolated"}}
    expect_error(ValueError, DAG.from_dict(damaged).validate_dag)


mutate = make_mutator(
    {
        "nodes": lambda rng: rng.randint(1, 24),
        "edges": lambda rng: [[rng.randrange(24), rng.randrange(24)] for _ in range(rng.randrange(97))],
        "suffix": lambda rng: "".join(rng.choices("abc_09 /\u4e91\u8fb9\x00", k=rng.randrange(25))),
        "mode": lambda rng: rng.choice(MODES),
    }
)


if __name__ == "__main__":
    run(check_input, mutate)
