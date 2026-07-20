"""Stable scheduling contracts shared by Backend, Scheduler and plugins."""

from .decision import build_schedule_decision, canonical_digest
from .queue_state import queue_waiting_counts, service_waiting_count

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
    "build_schedule_decision",
    "canonical_digest",
    "queue_waiting_counts",
    "service_waiting_count",
    "ALL_EDGE_NODES",
    "SELECTED_EDGE_NODES",
    "SOURCE_CANDIDATE_NODES_FIELD",
    "SOURCE_SELECTION_SCOPE_FIELD",
    "VALID_SOURCE_SELECTION_SCOPES",
    "normalize_source_selection_scope",
    "selection_scope_from_template",
    "source_selection_candidates",
)
