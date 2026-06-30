from __future__ import annotations

from app.graph.neo4j_repository import Neo4jGraphRepository


def _node(**properties):
    return properties


def _experiment_record(experiment_id: str = "EXP-VT6-AN"):
    chunk = _node(chunk_id="chunk-1", document_id="doc-1", source_name="source.txt", page=1, text="evidence quote")
    doc = _node(document_id="doc-1", source_name="source.txt")
    return {
        "e": _node(experiment_id=experiment_id),
        "materials": [_node(canonical_name="ВТ6")],
        "regimes": [_node(canonical_name="отжиг")],
        "measurements": [
            {
                "measurement": _node(
                    measurement_id="m1",
                    value=1120.0,
                    raw_value="1120",
                    unit="MPa",
                    effect="increase",
                    confidence=0.9,
                ),
                "property": _node(canonical_name="прочность"),
            }
        ],
        "equipment": [_node(canonical_name="Вакуумная печь")],
        "teams": [_node(canonical_name="Лаборатория легких сплавов")],
        "laboratories": [_node(canonical_name="Лаборатория легких сплавов")],
        "conclusions": [_node(text="отжиг повысил прочность")],
        "chunks": [chunk],
        "documents": [doc],
    }


class FakeGraphDB:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, **params):
        self.calls.append((query, params))
        if "MATCH (g:DataGap)" in query:
            return [
                {
                    "g": _node(gap_id="gap-1", material="7075-T6", regime="старение", property="коррозионная стойкость", reason="нет данных"),
                    "materials": [_node(canonical_name="7075-T6")],
                    "regimes": [_node(canonical_name="старение")],
                    "properties": [_node(canonical_name="коррозионная стойкость")],
                    "chunks": [],
                    "documents": [],
                }
            ]
        return [_experiment_record()]


def test_exact_query_uses_material_regime_property_params() -> None:
    graph_db = FakeGraphDB()
    repo = Neo4jGraphRepository(graph_db)  # type: ignore[arg-type]
    facts = repo.find_exact_material_regime_property("vt6", "annealing", "tensile strength")

    assert facts[0].experiment_id == "EXP-VT6-AN"
    assert facts[0].materials == ["ВТ6"]
    assert facts[0].regimes == ["отжиг"]
    assert facts[0].measurements[0].property_name == "прочность"
    _, params = graph_db.calls[0]
    assert params == {"material": "ВТ6", "regime": "отжиг", "property": "прочность"}


def test_partial_matches_are_separate_from_exact_facts() -> None:
    graph_db = FakeGraphDB()
    repo = Neo4jGraphRepository(graph_db)  # type: ignore[arg-type]
    exact = repo.find_exact_material_regime_property("ВТ6", "отжиг", "прочность")
    partial = repo.find_partial_matches(material="ВТ6", regime="криообработка", property_name="вязкость")

    assert exact
    assert partial.same_material
    assert len(exact) == 1
    assert graph_db.calls[0][1]["property"] == "прочность"
    assert any(call[1].get("property") == "вязкость" for call in graph_db.calls[1:])


def test_decision_history_filters_by_material_param() -> None:
    graph_db = FakeGraphDB()
    repo = Neo4jGraphRepository(graph_db)  # type: ignore[arg-type]
    history = repo.get_decision_history("ВТ6")
    assert history
    assert history[0].material == "ВТ6"
    assert graph_db.calls[0][1]["material"] == "ВТ6"


def test_gaps_query_filters_by_constraints() -> None:
    graph_db = FakeGraphDB()
    repo = Neo4jGraphRepository(graph_db)  # type: ignore[arg-type]
    gaps = repo.find_gaps(material="7075", regime="aging", property_name="corrosion resistance")
    assert gaps[0].material == "7075-T6"
    assert gaps[0].property == "коррозионная стойкость"
    assert graph_db.calls[0][1] == {"material": "7075-T6", "regime": "старение", "property": "коррозионная стойкость"}

