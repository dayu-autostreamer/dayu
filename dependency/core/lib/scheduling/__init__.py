"""Stable scheduling contracts shared by Backend, Scheduler and plugins."""

from .dag import END, START, service_names, topological_order
from .decision import build_schedule_decision, canonical_digest
from .offloading_plan import materialize_offloading_plan
from .queue_state import (
    queue_waiting_counts,
    service_waiting_count,
    snapshot_queue_states,
)
from .snapshot import (
    SchedulingSnapshotScope,
    deployment_from_snapshot,
    normalize_scheduling_snapshot_scope,
)

from .source_selection import (
    ALL_EDGE_NODES,
    SELECTED_EDGE_NODES,
    SOURCE_CANDIDATE_NODES_FIELD,
    SOURCE_SELECTION_SCOPE_FIELD,
    VALID_SOURCE_SELECTION_SCOPES,
    normalize_source_selection_scope,
    selection_scope_from_template,
    source_selection_candidates,
)

__all__ = (
    "END",
    "START",
    "service_names",
    "topological_order",
    "build_schedule_decision",
    "canonical_digest",
    "materialize_offloading_plan",
    "queue_waiting_counts",
    "service_waiting_count",
    "snapshot_queue_states",
    "SchedulingSnapshotScope",
    "deployment_from_snapshot",
    "normalize_scheduling_snapshot_scope",
    "ALL_EDGE_NODES",
    "SELECTED_EDGE_NODES",
    "SOURCE_CANDIDATE_NODES_FIELD",
    "SOURCE_SELECTION_SCOPE_FIELD",
    "VALID_SOURCE_SELECTION_SCOPES",
    "normalize_source_selection_scope",
    "selection_scope_from_template",
    "source_selection_candidates",
)
