from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_ui_uses_answer_graph_for_primary_graph_area() -> None:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    assert "build_answer_graph(payload)" in ui_text
    assert "answer_graph_to_html(answer_graph)" in ui_text
    assert "Технический подграф" in ui_text

    primary_section = ui_text.split('with st.expander("Технический подграф")', 1)[0]
    assert "graph_to_interactive_html(" not in primary_section


def test_technical_raw_subgraph_is_hidden_behind_expander() -> None:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    assert 'with st.expander("Технический подграф")' in ui_text
    technical_section = ui_text.split('with st.expander("Технический подграф")', 1)[1]
    assert "Raw subgraph" in technical_section
    assert "graph_to_interactive_html" in technical_section
