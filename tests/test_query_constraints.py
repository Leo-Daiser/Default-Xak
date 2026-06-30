from __future__ import annotations

from app.domain.query_constraints import QueryIntent
from app.retrieval.query_planner import QueryPlanner


def test_material_regime_property_constraints() -> None:
    constraints = QueryPlanner().parse("Что уже делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?")
    assert constraints.intent == QueryIntent.MATERIAL_REGIME_PROPERTY_EFFECT
    assert constraints.materials == ["ВТ6"]
    assert constraints.regimes == ["отжиг"]
    assert constraints.properties == ["прочность"]
    assert constraints.require_exact_match is True


def test_cryo_toughness_constraints() -> None:
    constraints = QueryPlanner().parse("Что делали по ВТ6 при криообработке и как изменилась вязкость?")
    assert constraints.intent == QueryIntent.MATERIAL_REGIME_PROPERTY_EFFECT
    assert constraints.materials == ["ВТ6"]
    assert constraints.regimes == ["криообработка"]
    assert constraints.properties == ["вязкость"]
    assert constraints.require_exact_match is True


def test_decision_history_constraints() -> None:
    constraints = QueryPlanner().parse("Покажи историю решений по ВТ6.")
    assert constraints.intent == QueryIntent.DECISION_HISTORY
    assert constraints.materials == ["ВТ6"]
    assert constraints.require_exact_match is False


def test_gap_constraints() -> None:
    constraints = QueryPlanner().parse("Какие пробелы по 7075-T6 и коррозионной стойкости?")
    assert constraints.intent == QueryIntent.GAP_ANALYSIS
    assert constraints.materials == ["7075-T6"]
    assert constraints.properties == ["коррозионная стойкость"]

