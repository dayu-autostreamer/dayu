from pathlib import Path

import pytest

from core.lib.common import YamlOps
from core.lib.scheduling.source_selection import (
    selection_scope_from_template,
    source_selection_candidates,
)


def scheduler_template(parameters=None):
    env = [{"name": "SCH_SELECTION_POLICY_NAME", "value": "fixed"}]
    if parameters is not None:
        env.append({
            "name": "SCH_SELECTION_POLICY_PARAMETERS",
            "value": parameters,
        })
    return {"pod-template": {"env": env}}


def test_selection_scope_defaults_to_selected_edge_nodes_and_parses_all_edges():
    assert selection_scope_from_template(scheduler_template()) == "selected_edge_nodes"
    assert selection_scope_from_template(scheduler_template(
        "{'scope': 'all_edge_nodes', 'fixed_type': 'hostname'}"
    )) == "all_edge_nodes"


@pytest.mark.parametrize(
    "parameters",
    ["not-an-object", "['all_edge_nodes']", "{'scope': 'cluster'}"],
)
def test_selection_scope_rejects_ambiguous_or_legacy_permissions(parameters):
    with pytest.raises(ValueError):
        selection_scope_from_template(scheduler_template(parameters))


def test_source_candidates_are_distinct_from_processor_candidates():
    info = {
        "node_set": ["processor-a", "processor-b"],
        "source_candidate_nodes": ["source-a", "source-b", "source-a"],
    }

    assert source_selection_candidates(info, "selected_edge_nodes") == [
        "processor-a", "processor-b",
    ]
    assert source_selection_candidates(info, "all_edge_nodes") == [
        "source-a", "source-b",
    ]


def test_all_edge_scope_requires_backend_authorized_candidates():
    with pytest.raises(ValueError, match="source_candidate_nodes"):
        source_selection_candidates(
            {"node_set": ["edge-a"], "all_edge_nodes": ["edge-b"]},
            "all_edge_nodes",
        )


def test_scheduler_scope_must_match_backend_authorized_scope():
    with pytest.raises(ValueError, match="does not match"):
        source_selection_candidates({
            "node_set": ["edge-a"],
            "source_candidate_nodes": ["edge-a", "edge-b"],
            "source_selection_scope": "selected_edge_nodes",
        }, "all_edge_nodes")


@pytest.mark.parametrize(
    "template_path",
    sorted((Path(__file__).parents[4] / "template" / "scheduler").glob("*.yaml")),
    ids=lambda path: path.name,
)
def test_repository_scheduler_templates_expose_a_valid_source_scope(template_path):
    assert selection_scope_from_template(YamlOps.read_yaml(template_path)) in {
        "selected_edge_nodes", "all_edge_nodes",
    }
