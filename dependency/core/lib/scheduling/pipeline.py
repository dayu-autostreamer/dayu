"""Partition-index tools for scheduling pipeline DAGs.

A pipeline partition indexes the ordered business services plus the terminal
``_end`` entry. Index zero places every business service on the cloud, while
the terminal index places every business service on the source edge. These
helpers reject branching, merging, disconnected, and inconsistent DAGs.
"""

from copy import deepcopy
from numbers import Integral

from .dag import END, START
from .offloading_plan import materialize_offloading_plan

__all__ = (
    "apply_pipeline_partition",
    "materialize_pipeline_policy",
    "pipeline_entries",
    "pipeline_partition_index",
    "rematerialize_pipeline_policy",
)


def _node_links(dag, service_name, field):
    node = dag.get(service_name)
    if not isinstance(node, dict):
        raise ValueError(
            f"pipeline node {service_name!r} must be an object"
        )
    links = node.get(field)
    if not isinstance(links, list):
        raise ValueError(
            f"pipeline node {service_name!r} must provide {field} as a list"
        )
    if any(not isinstance(linked, str) or not linked for linked in links):
        raise ValueError(
            f"pipeline node {service_name!r} has an invalid {field} entry"
        )
    if len(links) != len(set(links)):
        raise ValueError(
            f"pipeline node {service_name!r} has duplicate {field}"
        )
    unknown = [linked for linked in links if linked not in dag]
    if unknown:
        raise ValueError(
            f"pipeline node {service_name!r} references unknown nodes "
            f"in {field}: {unknown}"
        )
    return links


def _node_service(dag, service_name):
    node = dag.get(service_name)
    service = node.get("service") if isinstance(node, dict) else None
    if not isinstance(service, dict):
        raise ValueError(
            f"pipeline node {service_name!r} has no service object"
        )
    if service.get("service_name") != service_name:
        raise ValueError(
            f"pipeline node {service_name!r} has a mismatched service name"
        )
    return service


def _pipeline_order(dag):
    if not isinstance(dag, dict):
        raise TypeError("pipeline scheduling requires a DAG object")
    if START not in dag or END not in dag:
        raise ValueError(
            "pipeline scheduling requires explicit _start and _end nodes"
        )

    _node_service(dag, START)
    start_previous = _node_links(dag, START, "prev_nodes")
    if start_previous:
        raise ValueError("the _start node of a pipeline cannot have inputs")

    order = []
    visited = {START}
    current = START
    while current != END:
        following = _node_links(dag, current, "next_nodes")
        if len(following) != 1:
            raise ValueError(
                "this scheduling algorithm supports only a pipeline; "
                f"node {current!r} has {len(following)} outputs"
            )

        successor = following[0]
        if successor in visited:
            raise ValueError("pipeline contains a cycle")
        _node_service(dag, successor)
        predecessor_links = _node_links(dag, successor, "prev_nodes")
        if predecessor_links != [current]:
            raise ValueError(
                "this scheduling algorithm supports only a pipeline; "
                f"node {successor!r} does not have exactly one matching input"
            )

        visited.add(successor)
        order.append(successor)
        current = successor

    if _node_links(dag, END, "next_nodes"):
        raise ValueError("the _end node of a pipeline cannot have outputs")
    if len(visited) != len(dag):
        unvisited = sorted(str(name) for name in set(dag) - visited)
        raise ValueError(
            "this scheduling algorithm supports only a pipeline; "
            f"unreachable or parallel nodes were found: {unvisited}"
        )
    return order


def pipeline_entries(dag):
    """Return detached pipeline entries, including the terminal ``_end``."""

    entries = []
    for service_name in _pipeline_order(dag):
        service = _node_service(dag, service_name)
        entries.append({
            "service_name": service_name,
            "execute_device": service.get("execute_device"),
        })
    return entries


def pipeline_partition_index(dag, edge_device, cloud_device):
    """Return the split index of a monotonic edge-to-cloud placement."""

    edge_device = str(edge_device or "").strip()
    cloud_device = str(cloud_device or "").strip()
    if not edge_device or not cloud_device:
        raise ValueError("pipeline partitioning requires edge and cloud devices")

    entries = pipeline_entries(dag)
    terminal_index = len(entries) - 1
    partition = terminal_index
    seen_cloud = False
    for index, entry in enumerate(entries[:-1]):
        device = str(entry.get("execute_device") or "").strip()
        if device == cloud_device:
            if not seen_cloud:
                partition = index
            seen_cloud = True
        elif device == edge_device:
            if seen_cloud:
                raise ValueError(
                    "pipeline placement is non-monotonic: an edge stage "
                    "appears after a cloud stage"
                )
        else:
            raise ValueError(
                f"pipeline service {entry['service_name']!r} targets "
                f"{device!r}; expected edge {edge_device!r} or cloud "
                f"{cloud_device!r}"
            )
    return partition


def apply_pipeline_partition(dag, partition_index, edge_device, cloud_device):
    """Apply a split index to a pipeline DAG without rebuilding its nodes."""

    if isinstance(partition_index, bool) or not isinstance(
        partition_index,
        Integral,
    ):
        raise TypeError("pipeline partition index must be an integer")
    partition_index = int(partition_index)

    edge_device = str(edge_device or "").strip()
    cloud_device = str(cloud_device or "").strip()
    if not edge_device or not cloud_device:
        raise ValueError("pipeline partitioning requires edge and cloud devices")

    entries = pipeline_entries(dag)
    terminal_index = len(entries) - 1
    if partition_index < 0 or partition_index > terminal_index:
        raise ValueError(
            f"pipeline partition index {partition_index} is outside "
            f"[0, {terminal_index}]"
        )

    plan = {
        entry["service_name"]: (
            edge_device if index < partition_index else cloud_device
        )
        for index, entry in enumerate(entries[:-1])
    }
    return materialize_offloading_plan(
        {},
        dag,
        plan,
        source_device=edge_device,
        cloud_device=cloud_device,
    )["dag"]


def materialize_pipeline_policy(
    configuration,
    dag,
    partition_index,
    edge_device,
    cloud_device,
):
    """Return a complete schedule policy for one pipeline partition."""

    policy = deepcopy(configuration) if configuration is not None else {}
    if not isinstance(policy, dict):
        raise TypeError("pipeline configuration must be an object")
    policy.pop("pipeline", None)
    policy.pop("partition_index", None)
    policy["dag"] = apply_pipeline_partition(
        dag,
        partition_index,
        edge_device,
        cloud_device,
    )
    return policy


def rematerialize_pipeline_policy(policy, dag, edge_device, cloud_device):
    """Rebind a stored pipeline partition to the current DAG instance."""

    if not isinstance(policy, dict):
        raise TypeError("stored pipeline policy must be an object")
    stored_dag = policy.get("dag")
    if not isinstance(stored_dag, dict):
        raise ValueError("stored pipeline policy has no DAG decision")
    partition = pipeline_partition_index(
        stored_dag,
        edge_device=edge_device,
        cloud_device=cloud_device,
    )
    configuration = {
        key: deepcopy(value)
        for key, value in policy.items()
        if key != "dag"
    }
    return materialize_pipeline_policy(
        configuration,
        dag,
        partition,
        edge_device,
        cloud_device,
    )
