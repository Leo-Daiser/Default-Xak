from __future__ import annotations

from app.ui_helpers import graph_to_interactive_html


def test_interactive_graph_returns_html_with_zoom_pan_drag() -> None:
    html = graph_to_interactive_html(
        {
            "nodes": [
                {"id": "Material:ВТ6", "label": "ВТ6", "type": "Material"},
                {"id": "Regime:отжиг", "label": "отжиг", "type": "ProcessRegime"},
            ],
            "edges": [{"source": "Material:ВТ6", "target": "Regime:отжиг", "type": "HAS_REGIME"}],
        }
    )
    assert "<svg" in html
    assert "wheel" in html
    assert "pointerdown" in html
    assert "drag node" in html
    assert "ВТ6" in html
