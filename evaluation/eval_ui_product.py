from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.answer_graph import answer_graph_to_html, build_answer_graph  # noqa: E402
from app.ui_helpers import graph_to_interactive_html  # noqa: E402


FORBIDDEN_NAV_LABELS = [
    "Ask / GraphRAG",
    "Graph Explorer",
    "Entity Explorer",
    "Decision History",
    "Data Gaps",
    "Similar Experiments",
    "Evidence & Sources",
    "Demo Scenarios",
]


def main() -> int:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    answer_graph = build_answer_graph(
        {
            "status": "ok",
            "constraints": {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]},
            "primary_facts": [
                {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 1120.0, "unit": "MPa", "effect": "increase"},
                {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 980.0, "unit": "MPa", "effect": "decrease"},
                {"material": "ВТ6", "regime": "отжиг", "property": "прочность", "value": 862.0, "unit": "MPa", "effect": "unknown"},
            ],
            "sources": [
                {"source_name": "synthetic_vt6_heat_treatment.csv"},
                {"source_name": "article_vt6.txt"},
                {"source_name": "catalog.csv"},
            ],
        }
    )
    answer_graph_html = answer_graph_to_html(answer_graph)
    answer_graph_labels = " ".join(node.label for node in answer_graph.nodes)
    answer_graph_labels_lower = answer_graph_labels.lower()
    comparison_graph = build_answer_graph(
        {
            "status": "ok",
            "answer_mode": "comparison",
            "analytical_intent": "material_comparison",
            "constraints": {"materials": ["ВТ6", "7075-T6"], "properties": ["прочность"]},
            "facts": [
                {"material": "ВТ6", "property": "прочность", "value": 1120.0, "unit": "MPa", "effect": "increase"},
                {"material": "ВТ6", "property": "прочность", "value": 980.0, "unit": "MPa", "effect": "decrease"},
                {"material": "7075-T6", "property": "прочность", "value": 77.0, "unit": "ksi", "effect": "unknown"},
                {"material": "7075-T6", "property": "прочность", "value": 66.0, "unit": "ksi", "effect": "unknown"},
            ],
        }
    )
    comparison_labels = " ".join(node.label for node in comparison_graph.nodes)
    forbidden_graph_tokens = [
        "doc_",
        "chunk_",
        "EXP-",
        "SCI-",
        "Experiment",
        "PropertyValue",
        "SourceChunk",
        "FACT_SUPPORTED_BY_CHUNK",
        "OF_PROPERTY",
        "STUDIES",
        "MEASURES",
        "USES_REGIME",
        "effect: increase",
        "effect: decrease",
        "effect: unknown",
        "increase",
        "decrease",
        "unknown",
    ]
    graph_html = graph_to_interactive_html(
        {
            "nodes": [
                {"id": "doc_123", "label": "doc_29765440445babcdef", "type": "Document"},
                {"id": "m1", "label": "ВТ6", "type": "Material"},
                {"id": "p1", "label": "Прочность: 1120 MPa, effect: increase", "type": "Measurement"},
            ],
            "edges": [{"source": "m1", "target": "p1", "type": "MEASURED"}],
        }
    )
    visible_html = graph_html.split("<style>", 1)[0]
    checks = {
        "document_upload_available": "Загрузить в базу" in ui_text and "file_uploader" in ui_text,
        "document_management_available": "/documents" in ui_text and "/active" in ui_text,
        "document_management_editable_active_column": "st.data_editor" in ui_text and "CheckboxColumn" in ui_text and "Активен" in ui_text,
        "no_document_toggle_selectbox": "Документ для включения/выключения" not in ui_text and "Выключить документ" not in ui_text,
        "no_nested_expanders": "_render_document_controls" in ui_text and 'st.expander("Metadata выбранного документа")' not in ui_text,
        "interactive_graph_available": "graph_to_interactive_html" in ui_text and "components.html" in ui_text,
        "no_sidebar_page_navigation": '"Раздел"' not in ui_text and all(label not in ui_text for label in FORBIDDEN_NAV_LABELS),
        "graph_labels_clean": all(token not in visible_html for token in ["doc_", "chunk_", "EXP-", "SCI-", "effect: increase"]),
        "answer_graph_available": "build_answer_graph(payload)" in ui_text
        and "answer_graph_to_html(" in ui_text
        and "answerGraphCompact_" in ui_text
        and "answerGraphExpanded_" in ui_text,
        "answer_graph_large_mode_available": "Развернуть карту" in ui_text
        and "Открыть крупно" not in ui_text
        and "render_height=820" in ui_text
        and "render_width=1500" in ui_text
        and "min(85vw, 1500px)" in ui_text
        and 'data-testid="stModal"' in ui_text
        and 'button[aria-label="Close"]' in ui_text
        and "answer_graph_modal_open" in ui_text
        and 'getattr(st, "dialog", None)' in ui_text,
        "answer_graph_node_limit_ok": len(answer_graph.nodes) <= 10 and len(answer_graph.edges) <= 12,
        "answer_graph_labels_clean": all(token not in answer_graph_labels for token in forbidden_graph_tokens),
        "answer_graph_has_semantic_path": "ВТ6" in answer_graph_labels
        and "титановый сплав" in answer_graph_labels
        and all(token in answer_graph_labels_lower for token in ["отжиг", "прочность"])
        and "862–1120 MPa" in answer_graph_labels,
        "comparison_answer_graph_compact_clean": len(comparison_graph.nodes) <= 10
        and len(comparison_graph.edges) <= 12
        and all(token not in comparison_labels for token in forbidden_graph_tokens)
        and all(token in comparison_labels for token in ["ВТ6", "титановый сплав", "7075-T6", "алюминиевый сплав", "состояние T6", "980–1120 MPa", "455–531 MPa", "сравнение ограничено"]),
        "technical_graph_hidden_by_default": 'with st.expander("Технический подграф")' in ui_text,
        "main_graph_not_raw_subgraph": "graph_to_interactive_html(" not in ui_text.split('with st.expander("Технический подграф")', 1)[0],
        "answer_graph_hierarchical_renderer": all(
            token in answer_graph_html
            for token in ["hierarchical", "physics", "enabled: false", "zoomView", "dragView", "dragNodes: false", "Колесо — масштаб", "узлы зафиксированы"]
        ),
        "answer_graph_no_drag_node_hint": "Wheel: zoom" not in answer_graph_html
        and "drag node: move" not in answer_graph_html
        and "dragNodes: true" not in answer_graph_html,
        "llm_diagnostics_available": "/system/test-llm" in ui_text and "LLM:" in ui_text,
        "examples_not_as_long_top_buttons": "Подставить пример" in ui_text and "EXAMPLE_QUESTIONS):" not in ui_text,
    }
    print("UI product evaluation:")
    for key, value in checks.items():
        print(f"{key}: {1 if value else 0}")
    passed = all(checks.values())
    print("PASS" if passed else "FAIL")
    print("SUMMARY", json.dumps(checks, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
