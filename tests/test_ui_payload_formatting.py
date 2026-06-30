from __future__ import annotations

from app.ui_helpers import evidence_to_rows, facts_to_rows, graph_context_stats, no_exact_match_warning, subgraph_to_tables


def test_facts_convert_to_rows() -> None:
    rows = facts_to_rows({"facts": [{"experiment_id": "EXP-1", "material": "ВТ6"}]})
    assert rows == [{"experiment_id": "EXP-1", "material": "ВТ6"}]


def test_evidence_convert_to_table_rows() -> None:
    rows = evidence_to_rows(
        {
            "evidence": [
                {
                    "source_name": "demo.txt",
                    "document_id": "doc1",
                    "chunk_id": "chunk1",
                    "score": 0.9,
                    "retrieval_backend": "bm25",
                    "quote": "ВТ6 отжиг",
                }
            ]
        }
    )
    assert rows[0]["source_name"] == "demo.txt"
    assert rows[0]["doc_id"] == "doc1"


def test_subgraph_convert_to_node_edge_tables() -> None:
    nodes, edges = subgraph_to_tables(
        {
            "nodes": [{"id": "Material:ВТ6"}],
            "edges": [{"source": "Experiment:E1", "target": "Material:ВТ6"}],
        }
    )
    assert nodes[0]["id"] == "Material:ВТ6"
    assert edges[0]["target"] == "Material:ВТ6"


def test_missing_optional_fields_do_not_crash() -> None:
    assert facts_to_rows({}) == []
    assert evidence_to_rows({}) == []
    assert subgraph_to_tables(None) == ([], [])
    assert graph_context_stats({})["facts_count"] == 0


def test_no_exact_match_warning_is_generated() -> None:
    warning = no_exact_match_warning({"status": "no_exact_match"})
    assert warning
    assert "Точного факта" in warning
