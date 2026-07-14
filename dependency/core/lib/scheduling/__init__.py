"""Stable scheduling contracts shared by Backend, Scheduler and plugins."""

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
    "ALL_EDGE_NODES",
    "SELECTED_EDGE_NODES",
    "SOURCE_CANDIDATE_NODES_FIELD",
    "SOURCE_SELECTION_SCOPE_FIELD",
    "VALID_SOURCE_SELECTION_SCOPES",
    "normalize_source_selection_scope",
    "selection_scope_from_template",
    "source_selection_candidates",
)
