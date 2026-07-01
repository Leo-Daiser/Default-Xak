from __future__ import annotations

from app.answering.human_answer import enhance_answer_payload


def _exact_payload() -> dict:
    return {
        "answer": "legacy technical answer",
        "status": "ok",
        "answer_mode": "graph_exact",
        "analytical_intent": "strict_material_regime_property",
        "constraints": {"materials": ["ВТ6"], "regimes": ["отжиг"], "properties": ["прочность"]},
        "facts": [
            {
                "experiment_id": "SCI-VT6-AN-900",
                "material": "Titanium Alpha-Beta, ВТ6",
                "regime": "отжиг",
                "property": "прочность",
                "value": 1120.0,
                "unit": "MPa",
                "effect": "increase",
                "evidence": [
                    {
                        "document_id": "doc_demo",
                        "chunk_id": "chunk_demo",
                        "source_name": "demo.csv",
                        "quote": "ВТ6 отжиг прочность 1120 MPa",
                    }
                ],
            },
            {
                "experiment_id": "EXP-1cc92a75d794",
                "material": "ВТ6",
                "regime": "отжиг",
                "property": "прочность",
                "value": 980.0,
                "unit": "MPa",
                "effect": "unknown",
                "evidence": [
                    {
                        "document_id": "doc_demo2",
                        "chunk_id": "chunk_demo2",
                        "source_name": "demo.txt",
                        "quote": "ВТ6 отжиг прочность 980 MPa",
                    }
                ],
            },
        ],
        "sources": [],
        "subgraph": {"nodes": [{"id": "Material:ВТ6"}], "edges": []},
        "graph_context": {},
        "diagnostics": {},
        "retrieval": {},
    }


def test_strict_positive_human_answer_hides_internal_ids() -> None:
    payload = enhance_answer_payload(_exact_payload(), "expert_max")
    answer = payload["answer"]
    assert "ВТ6" in answer
    assert "отжиг" in answer
    assert "прочность" in answer
    assert "1120" in answer
    assert "Ограничения" in answer
    for forbidden in ["doc_", "chunk_", "EXP-", "SCI-", "effect:", "unknown"]:
        assert forbidden not in answer


def test_strict_negative_human_answer_is_clear() -> None:
    payload = enhance_answer_payload(
        {
            "answer": "Ближайшие данные: закалка при 1050 °C для другого материала.",
            "status": "no_exact_match",
            "constraints": {"materials": ["ВТ6"], "regimes": ["криообработка"], "properties": ["вязкость"]},
            "facts": [],
            "sources": [],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {},
            "diagnostics": {},
            "retrieval": {},
        },
        "expert_max",
    )
    answer = payload["answer"].lower()
    assert "точных данных" in answer
    assert "вт6" in answer
    assert "криообработка" in answer
    assert "вязкость" in answer
    assert "нельзя считать ответом" in answer


def test_comparison_answer_warns_about_comparability() -> None:
    payload = _exact_payload()
    payload["answer_mode"] = "comparison"
    payload["analytical_intent"] = "material_comparison"
    payload["constraints"] = {"materials": ["ВТ6", "7075-T6"], "regimes": [], "properties": ["прочность"]}
    answer = enhance_answer_payload(payload, "expert_max")["answer"].lower()
    assert "сравнение ограничено" in answer
    assert "не прямое экспериментальное сравнение" in answer


def test_overview_without_facts_does_not_claim_experiments_found() -> None:
    payload = enhance_answer_payload(
        {
            "answer": "legacy",
            "status": "partial",
            "answer_mode": "overview",
            "analytical_intent": "material_overview",
            "constraints": {"materials": ["X999"], "raw_question": "Что известно о сплаве X999 при лазерной обработке?"},
            "facts": [],
            "sources": [{"source_name": "nearest.txt", "quote": "unrelated evidence"}],
            "evidence": [{"source_name": "nearest.txt", "quote": "unrelated evidence"}],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {},
            "diagnostics": {"llm_answer_polished": True},
            "retrieval": {},
        },
        "expert_max",
    )
    answer = payload["answer"].lower()

    assert "структурированных фактов" in answer
    assert "нет подтверждённых" in answer or "не найдено" in answer
    assert "найдены связанные эксперименты" not in answer
    assert "1050" not in answer


def test_lab_activity_human_answer_lists_laboratories_cleanly() -> None:
    payload = enhance_answer_payload(
        {
            "answer": "legacy",
            "status": "ok",
            "answer_mode": "overview",
            "analytical_intent": "lab_activity",
            "constraints": {"raw_question": "Какие лаборатории или команды выполняли эксперименты?"},
            "facts": [],
            "laboratories": [
                {"canonical_name": "Лаборатория легких сплавов"},
                {"canonical_name": "Лаборатория термообработки"},
            ],
            "sources": [],
            "evidence": [],
            "subgraph": {"nodes": [], "edges": []},
            "graph_context": {},
            "diagnostics": {},
            "retrieval": {},
        },
        "expert_max",
    )
    answer = payload["answer"]

    assert "Лаборатории и команды" in answer
    assert "Лаборатория легких сплавов" in answer
    assert "Лаборатория термообработки" in answer
    for forbidden in ["doc_", "chunk_", "EXP-", "SCI-"]:
        assert forbidden not in answer
