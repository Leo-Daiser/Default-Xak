from __future__ import annotations

from app.graph.answer_graph import answer_graph_to_html, build_answer_graph


def test_answer_graph_renderer_uses_layered_interactive_layout() -> None:
    graph = build_answer_graph(
        {
            "status": "ok",
            "constraints": {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]},
            "facts": [{"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 1120.0, "unit": "MPa"}],
            "sources": [{"source_name": "synthetic_vt6_heat_treatment.csv"}],
        }
    )
    html = answer_graph_to_html(graph)
    assert "<svg" in html
    assert "hierarchical" in html
    assert "zoomView" in html
    assert "dragView" in html
    assert "dragNodes" in html
    assert "physics" in html
    assert "enabled: false" in html
    assert "wheel" in html
    assert "pointerdown" in html


def test_answer_graph_renderer_keeps_labels_clean_in_visible_html() -> None:
    graph = build_answer_graph(
        {
            "status": "ok",
            "constraints": {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]},
            "facts": [
                {
                    "experiment_id": "EXP-123",
                    "material": "ВТ6",
                    "regime": "отжиг",
                    "property": "прочность",
                    "value": 1120.0,
                    "unit": "MPa",
                    "effect": "unknown",
                }
            ],
            "sources": [{"document_id": "doc_abc", "chunk_id": "chunk_def"}],
        }
    )
    html = answer_graph_to_html(graph)
    visible = html.split("<style>", 1)[0]
    for token in ["doc_", "chunk_", "EXP-", "SCI-", "Experiment", "PropertyValue", "SourceChunk", "effect: unknown"]:
        assert token not in visible
    assert "эффект не указан" in visible
