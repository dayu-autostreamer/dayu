"""Pure helpers for the scheduler's serialized DAG contract."""

from core.lib.common import TaskConstant

__all__ = ("END", "START", "service_names", "topological_order")


START = TaskConstant.START.value
END = TaskConstant.END.value


def service_names(dag):
    """Return the logical services, excluding synthetic boundary nodes."""

    if not isinstance(dag, dict):
        raise TypeError("scheduling DAG must be a mapping")
    return [name for name in dag if name not in (START, END)]


def topological_order(dag):
    """Return a deterministic topological order for a serialized DAG."""

    if not isinstance(dag, dict):
        raise TypeError("scheduling DAG must be a mapping")
    nodes = list(dag)
    indegree = {
        name: len([
            predecessor
            for predecessor in dag[name].get("prev_nodes", [])
            if predecessor in dag
        ])
        for name in nodes
    }
    ready = sorted(name for name, degree in indegree.items() if degree == 0)
    result = []
    while ready:
        current = ready.pop(0)
        result.append(current)
        for successor in dag[current].get("next_nodes", []):
            if successor not in indegree:
                continue
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(result) != len(nodes):
        raise ValueError("scheduling requires an acyclic DAG")
    return result
