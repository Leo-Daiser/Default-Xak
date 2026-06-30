from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_ui_uses_answer_graph_for_primary_graph_area() -> None:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    assert "build_answer_graph(payload)" in ui_text
    assert "answer_graph_to_html(" in ui_text
    assert "answerGraphCompact_" in ui_text
    assert "answerGraphExpanded_" in ui_text
    assert "render_height=820" in ui_text
    assert "render_width=1500" in ui_text
    assert "Развернуть карту" in ui_text
    assert "Открыть крупно" not in ui_text
    assert "Технический подграф" in ui_text
    assert ui_text.index("Развернуть карту") < ui_text.index("def _render_interactive_graph")

    primary_section = ui_text.split('with st.expander("Технический подграф")', 1)[0]
    assert "graph_to_interactive_html(" not in primary_section


def test_technical_raw_subgraph_is_hidden_behind_expander() -> None:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    assert 'with st.expander("Технический подграф")' in ui_text
    technical_section = ui_text.split('with st.expander("Технический подграф")', 1)[1]
    assert "Raw subgraph" in technical_section
    assert "graph_to_interactive_html" in technical_section


def test_large_answer_graph_uses_dialog_or_inline_block_without_backend_request() -> None:
    ui_text = (ROOT / "app" / "ui.py").read_text(encoding="utf-8")
    assert "def _render_large_answer_graph" in ui_text
    assert 'getattr(st, "dialog", None)' in ui_text
    assert "answer_graph_modal_open" in ui_text
    assert "open_answer_graph_modal" in ui_text
    assert "close_answer_graph_modal" in ui_text
    assert "_close_answer_graph_modal(answer_key)" in ui_text
    assert "min(85vw, 1500px)" in ui_text
    assert 'data-testid="stModal"' in ui_text
    assert 'button[aria-label="Close"]' in ui_text
    assert "ask_api(" not in ui_text.split("def _render_large_answer_graph", 1)[1].split("def _render_answer", 1)[0]
