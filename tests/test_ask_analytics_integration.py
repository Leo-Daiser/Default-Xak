from __future__ import annotations

from tests.strict_qa_helpers import seeded_client


def test_ask_material_overview_returns_analytical_fields(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post("/ask", params={"question": "Что уже делали по ВТ6?", "top_k": 6})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["analytical_intent"] == "material_overview"
    assert payload["answer_mode"] == "overview"
    assert payload["graph_context"]["facts_count"] > 0
    assert payload["sources"] or payload["evidence"]
    assert payload["diagnostics"]["answer_synthesis_mode"]


def test_ask_graph_neighborhood_returns_subgraph_stats(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post("/ask", params={"question": "Покажи связанные сущности по ВТ6", "top_k": 6})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["analytical_intent"] == "graph_neighborhood"
    assert payload["graph_context"]["subgraph_nodes"] > 0
    assert payload["subgraph"]["nodes"]


def test_ask_strict_no_exact_match_still_has_no_facts(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post(
        "/ask",
        params={"question": "Что делали по сплаву ВТ6 при криообработке и как изменилась вязкость?", "top_k": 6},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "no_exact_match"
    assert payload["facts"] == []
    assert payload["analytical_intent"] == "strict_material_regime_property"
    assert payload["retrieval"]["kg_backend_active"]
    assert "точных данных не найдено" in payload["answer"].lower()
