"""Human-grade answer synthesis for the product UI.

This layer does not create new facts.  It rewrites already grounded API
payloads into a readable main answer, ranks facts for presentation and
normalizes evidence rows for the UI.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from ..domain.fact_normalization import build_conflict_summary, dedupe_fact_rows
from ..domain.unit_normalization import normalize_strength_to_mpa
from ..runtime.presets import RuntimePresetId, get_runtime_preset
from .grounding_guard import (
    build_repair_request,
    diagnostics_after_repair,
    guard_llm_polished_answer,
    skipped_guard_diagnostics,
)


class HumanAnswer(BaseModel):
    """User-facing answer block."""

    title: str
    summary: str
    key_findings: list[str]
    caveats: list[str] = Field(default_factory=list)
    recommendation: str | None = None
    confidence_label: Literal["высокая", "средняя", "низкая"]


INTERNAL_ID_RE = re.compile(
    r"\b(?:doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|SCI-[A-Za-z0-9_-]+|EXP-[A-Za-z0-9_-]+|VT6-AN-TXT|Experiment\s+doc_[A-Za-z0-9_:.-]+)\b"
)


LLMRepairer = Callable[[dict[str, Any]], str | None]


def enhance_answer_payload(
    payload: dict[str, Any],
    preset_id: RuntimePresetId | str | None = None,
    *,
    llm_repairer: LLMRepairer | None = None,
) -> dict[str, Any]:
    """Attach human answer, evidence and ranked facts to an ask payload."""

    preset = get_runtime_preset(preset_id)
    payload["facts"] = dedupe_fact_rows(payload.get("facts") or [])
    evidence = normalize_evidence(payload)
    ranked = rank_facts(payload.get("facts") or [])
    payload["technical_answer"] = payload.get("answer")
    payload["evidence"] = evidence
    payload["primary_facts"] = ranked["primary_facts"]
    payload["supporting_facts"] = ranked["supporting_facts"]
    payload["low_confidence_or_context_facts"] = ranked["low_confidence_or_context_facts"]
    payload["subgraph"] = _ensure_no_match_subgraph(payload)
    payload["diagnostics"] = {
        **(payload.get("diagnostics") or {}),
        "fact_conflicts": build_conflict_summary(payload.get("facts") or []),
    }
    human = build_human_answer(payload, preset.preset_id, llm_repairer=llm_repairer)
    payload["human_answer"] = human.model_dump()
    payload["answer"] = render_human_answer_markdown(human)
    payload["graph_context"] = _with_evidence_count(payload.get("graph_context") or {}, evidence, payload)
    return payload


def _ensure_no_match_subgraph(payload: dict[str, Any]) -> dict[str, Any]:
    subgraph = payload.get("subgraph")
    if payload.get("status") != "no_exact_match" or not isinstance(subgraph, dict):
        return subgraph if isinstance(subgraph, dict) else {"nodes": [], "edges": []}
    if subgraph.get("nodes"):
        return subgraph
    constraints = payload.get("constraints") or {}
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    gap_id = "DataGap:inferred"
    nodes.append({"id": gap_id, "label": "Точный факт не найден", "type": "DataGap", "properties": {"inferred": True}})
    for node_type, key in [("Material", "materials"), ("ProcessRegime", "regimes"), ("Property", "properties")]:
        for value in constraints.get(key) or []:
            node_id = f"{node_type}:{value}"
            nodes.append({"id": node_id, "label": value, "type": node_type, "properties": {}})
            edges.append(
                {
                    "id": f"{gap_id}:MISSING_FOR:{node_id}",
                    "source": gap_id,
                    "target": node_id,
                    "label": "MISSING_FOR",
                    "type": "MISSING_FOR",
                    "properties": {},
                }
            )
    return {"nodes": nodes, "edges": edges}


def build_human_answer(
    payload: dict[str, Any],
    preset_id: RuntimePresetId | str | None = None,
    *,
    llm_repairer: LLMRepairer | None = None,
) -> HumanAnswer:
    """Build a readable grounded answer from an already structured payload."""

    preset = get_runtime_preset(preset_id)
    _record_llm_grounding_guard(payload)
    if payload.get("answer_mode") == "needs_clarification" or payload.get("intent") == "clarification":
        answer = _clarification_answer(payload)
    elif preset.preset_id == RuntimePresetId.STRICT_AUDIT:
        answer = _strict_audit_answer(payload)
    elif _is_conflict_answer(payload):
        answer = _conflict_answer(payload)
    elif _is_gap_answer(payload):
        answer = _gap_answer(payload)
    elif _is_technical_object_payload(payload):
        answer = _technical_object_answer(payload)
    elif payload.get("status") == "no_exact_match":
        answer = _negative_answer(payload)
    elif _is_lab_activity_answer(payload):
        answer = _lab_activity_answer(payload)
    elif _is_overview_answer(payload) and not (payload.get("facts") or []):
        answer = _overview_answer(payload)
    elif _has_grounded_llm_answer(payload):
        answer = _grounded_llm_answer(payload, llm_repairer=llm_repairer)
    elif _is_comparison(payload):
        answer = _comparison_answer(payload)
    elif _is_history_answer(payload):
        answer = _history_answer(payload)
    elif _is_similar_answer(payload):
        answer = _similar_answer(payload)
    elif _is_overview_answer(payload):
        answer = _overview_answer(payload)
    else:
        answer = _strict_positive_or_generic_answer(payload)

    if preset.preset_id == RuntimePresetId.OFFLINE_RELIABLE:
        answer.caveats.insert(
            0,
            "Работа выполнена в офлайн-режиме: использован локальный validated graph без Neo4j/LLM. "
            "Выводы основаны только на локально извлечённых фактах.",
        )
        answer.title = "Офлайн-режим: " + answer.title
    answer = _with_conflict_caveats(answer, payload)
    return answer


def _has_grounded_llm_answer(payload: dict[str, Any]) -> bool:
    diagnostics = payload.get("diagnostics") or {}
    answer_mode = str(payload.get("answer_mode") or "")
    return bool(
        payload.get("technical_answer")
        and (
            diagnostics.get("llm_answer_polished")
            or answer_mode.startswith("llm_grounded")
        )
    )


def _grounded_llm_answer(payload: dict[str, Any], *, llm_repairer: LLMRepairer | None = None) -> HumanAnswer:
    if _is_comparison(payload):
        base = _comparison_answer(payload)
    elif _is_gap_answer(payload):
        base = _gap_answer(payload)
    elif _is_history_answer(payload):
        base = _history_answer(payload)
    elif _is_similar_answer(payload):
        base = _similar_answer(payload)
    elif _is_overview_answer(payload):
        base = _overview_answer(payload)
    else:
        base = _strict_positive_or_generic_answer(payload)
    guard = (payload.get("diagnostics") or {}).get("llm_grounding_guard") or {}
    if guard.get("status") == "pass":
        base.summary = _sanitize_main_answer(str(payload.get("technical_answer") or base.summary).strip())
        return base
    repaired = _try_repair_llm_answer(payload, base, llm_repairer)
    if repaired:
        base.summary = _sanitize_main_answer(repaired)
        return base
    if guard.get("status") != "pass":
        return base
    return base


def _record_llm_grounding_guard(payload: dict[str, Any]) -> None:
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
        payload["diagnostics"] = diagnostics
    if diagnostics.get("llm_grounding_guard"):
        return
    if not _has_grounded_llm_answer(payload):
        diagnostics["llm_grounding_guard"] = skipped_guard_diagnostics()
        return
    result = guard_llm_polished_answer(str(payload.get("technical_answer") or ""), payload)
    diagnostics["llm_grounding_guard"] = result.diagnostics()


def _try_repair_llm_answer(payload: dict[str, Any], base: HumanAnswer, llm_repairer: LLMRepairer | None) -> str | None:
    if llm_repairer is None:
        return None
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    guard = diagnostics.get("llm_grounding_guard") if isinstance(diagnostics.get("llm_grounding_guard"), dict) else {}
    if guard.get("status") != "fallback" or guard.get("repair_attempted"):
        return None
    if ((guard.get("grounding_context") or {}).get("no_facts_mode")):
        return None
    unsafe_answer = str(payload.get("technical_answer") or "")
    first_result = guard_llm_polished_answer(unsafe_answer, payload)
    if not first_result.violations:
        return None
    repair_request = build_repair_request(
        question=_question_text(payload),
        unsafe_answer=unsafe_answer,
        deterministic_answer=render_human_answer_markdown(base),
        first_result=first_result,
    )
    try:
        repaired = llm_repairer(repair_request)
    except Exception as exc:  # pragma: no cover - defensive; tested through fallback behavior.
        diagnostics["llm_grounding_guard"] = diagnostics_after_repair(
            first_result,
            None,
            fallback_reason=f"repair_exception:{type(exc).__name__}",
        )
        return None
    if not repaired:
        diagnostics["llm_grounding_guard"] = diagnostics_after_repair(first_result, None, fallback_reason="repair_empty")
        return None
    repair_payload = {**payload, "technical_answer": str(repaired)}
    repair_result = guard_llm_polished_answer(str(repaired), repair_payload)
    diagnostics["llm_grounding_guard"] = diagnostics_after_repair(first_result, repair_result)
    if repair_result.status == "pass":
        payload["technical_answer_repaired"] = str(repaired)
        return str(repaired)
    return None


def _question_text(payload: dict[str, Any]) -> str:
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    return str(constraints.get("raw_question") or payload.get("question") or "")


def render_human_answer_markdown(answer: HumanAnswer) -> str:
    """Render a HumanAnswer as compact Markdown suitable for API/UI display."""

    lines = [f"### {answer.title}", "", answer.summary]
    if answer.key_findings:
        lines.extend(["", "**Что найдено:**"])
        lines.extend(f"{index}. {item}" for index, item in enumerate(answer.key_findings, start=1))
    if answer.caveats:
        lines.extend(["", "**Ограничения:**"])
        lines.extend(f"- {item}" for item in answer.caveats)
    if answer.recommendation:
        lines.extend(["", f"**Вывод:** {answer.recommendation}"])
    lines.extend(["", f"**Уверенность:** {answer.confidence_label}"])
    return _sanitize_main_answer("\n".join(lines))


def normalize_evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Populate top-level evidence from evidence, fact evidence and sources."""

    rows: list[dict[str, Any]] = []

    def add(row: dict[str, Any], evidence_type: str) -> None:
        quote = str(row.get("quote") or "").strip()
        source_name = row.get("source_name") or row.get("title") or row.get("filename")
        chunk_id = row.get("chunk_id") or row.get("source_chunk_id")
        if not quote and not source_name:
            return
        rows.append(
            {
                "source_name": source_name,
                "document_id": row.get("document_id") or row.get("doc_id"),
                "chunk_id": chunk_id,
                "page": row.get("page") or row.get("page_start"),
                "quote": quote or "Цитата не сохранена; источник доступен в карточке документа.",
                "score": row.get("score"),
                "retrieval_backend": row.get("retrieval_backend"),
                "evidence_type": evidence_type,
            }
        )

    for row in payload.get("evidence") or []:
        if isinstance(row, dict):
            add(row, "retrieval")
    for fact in payload.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        for evidence in fact.get("evidence") or []:
            if isinstance(evidence, dict):
                add(evidence, "graph_fact")
    for row in payload.get("sources") or []:
        if isinstance(row, dict):
            add(row, "source")

    seen = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("document_id"), row.get("chunk_id"), row.get("quote"))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result[:12]


def rank_facts(facts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Rank facts for main-answer use without removing original facts."""

    deduped = dedupe_fact_rows([fact for fact in facts if isinstance(fact, dict)])
    scored = [(_fact_score(fact), index, fact) for index, fact in enumerate(deduped)]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    ordered = [fact for _, _, fact in scored]
    primary = ordered[:5]
    supporting = ordered[5:20]
    low_quality = [fact for score, _, fact in scored if score < 35]
    return {
        "primary_facts": primary,
        "supporting_facts": supporting,
        "low_confidence_or_context_facts": low_quality,
    }


def _fact_score(fact: dict[str, Any]) -> int:
    score = 0
    if fact.get("material"):
        score += 12
    if fact.get("regime"):
        score += 12
    if fact.get("property"):
        score += 12
    if fact.get("value") is not None or fact.get("raw_value"):
        score += 10
    if fact.get("unit"):
        score += 8
    effect = str(fact.get("effect") or "").lower()
    if effect and effect != "unknown":
        score += 8
    if fact.get("evidence"):
        score += 10
    source_text = _fact_source_text(fact).lower()
    if "table columns" in source_text or ".csv" in source_text or ".xlsx" in source_text:
        score += 6
    exp_id = str(fact.get("experiment_id") or "")
    if exp_id.startswith("Experiment doc_") or re.match(r"^EXP-[0-9a-f]{8,}", exp_id):
        score -= 16
    if effect == "unknown":
        score -= 6
    if not fact.get("regime"):
        score -= 5
    return score


def _strict_positive_or_generic_answer(payload: dict[str, Any]) -> HumanAnswer:
    constraints = payload.get("constraints") or {}
    materials = constraints.get("materials") or _unique(row.get("material") for row in payload.get("primary_facts") or [])
    regimes = constraints.get("regimes") or _unique(row.get("regime") for row in payload.get("primary_facts") or [])
    properties = constraints.get("properties") or _unique(row.get("property") for row in payload.get("primary_facts") or [])
    material = _join_or_default(materials, "заданному материалу")
    regime = _join_or_default(regimes, "заданному режиму")
    prop = _join_or_default(properties, "заданному свойству")
    primary = payload.get("primary_facts") or []
    material_for_findings = materials[0] if materials else None
    findings = [_fact_sentence(fact, material_override=material_for_findings) for fact in primary[:5]]
    if not findings:
        findings = ["Структурированные факты в графе есть, но для основного вывода недостаточно численных измерений."]
    values = _numeric_values(primary)
    caveats: list[str] = []
    effects = {str(f.get("effect") or "unknown") for f in primary}
    if len(values) > 1 or len(effects) > 1:
        caveats.append(
            "Найдено несколько значений или эффектов; результат зависит от конкретного режима, источника и исходного состояния материала."
        )
    if any(str(f.get("effect") or "").lower() == "unknown" for f in primary):
        caveats.append("В части фактов направление эффекта не указано явно.")
    summary = f"По {material} при режиме {regime} найдены подтверждённые данные по свойству {prop}."
    fact_properties = _unique(row.get("property") for row in primary)
    fact_units = _unique(row.get("unit") for row in primary)
    if values and len(fact_properties) <= 1 and len(fact_units) <= 1:
        summary += f" Основной численный диапазон в найденных фактах: {_format_value_range(values, primary)}."
    return HumanAnswer(
        title="Подтверждённые экспериментальные данные",
        summary=summary,
        key_findings=findings,
        caveats=caveats,
        recommendation=_strict_recommendation(material, regime, prop, primary),
        confidence_label=_confidence(payload, primary),
    )


def _clarification_answer(payload: dict[str, Any]) -> HumanAnswer:
    technical = _sanitize_main_answer(str(payload.get("technical_answer") or payload.get("answer") or "Уточните вопрос."))
    if "уточните вопрос" not in technical.lower():
        technical = "Уточните вопрос: " + technical
    return HumanAnswer(
        title="Нужно уточнить вопрос",
        summary=technical,
        key_findings=["Система не нашла достаточных предметных ограничений для надёжного поиска по графу."],
        caveats=["Случайные или бессмысленные запросы не используются для подстановки похожих фактов."],
        recommendation="Укажите материал, объект, режим, свойство, оборудование или лабораторию.",
        confidence_label="высокая",
    )


def _technical_object_answer(payload: dict[str, Any]) -> HumanAnswer:
    objects = _unique(item.get("name") for item in payload.get("technical_objects") or [] if isinstance(item, dict))
    params = _unique(item.get("name") for item in payload.get("parameters") or [] if isinstance(item, dict))
    materials = _unique(item.get("name") for item in payload.get("materials") or [] if isinstance(item, dict))
    standards = _unique(item.get("name") for item in payload.get("standards") or [] if isinstance(item, dict))
    articles = _unique(
        fact.get("object")
        for fact in payload.get("facts") or []
        if isinstance(fact, dict) and str(fact.get("predicate")) == "PART_HAS_ARTICLE_NUMBER"
    )
    subject = objects[0] if objects else "техническому объекту"
    findings = []
    if params:
        findings.append(f"Параметры: {', '.join(params[:6])}.")
    if materials:
        findings.append(f"Материалы: {', '.join(materials[:6])}.")
    if articles:
        findings.append(f"Артикулы: {', '.join(articles[:6])}.")
    if standards:
        findings.append(f"Стандарты: {', '.join(standards[:6])}.")
    if not findings:
        findings.append(_sanitize_main_answer(str(payload.get("technical_answer") or payload.get("answer") or "Найдены связанные фрагменты.")))
    return HumanAnswer(
        title=f"Сводка по объекту {subject}",
        summary=f"По запросу найден технический объект {subject} и связанные с ним параметры, материалы или документы.",
        key_findings=findings,
        caveats=["Это объектная сводка; для строгого экспериментального вывода задайте материал, режим и свойство."],
        recommendation="Для проверки источников откройте блок «Источники и evidence».",
        confidence_label=_confidence(payload, payload.get("facts") or []),
    )


def _negative_answer(payload: dict[str, Any]) -> HumanAnswer:
    constraints = payload.get("constraints") or {}
    material = _join_or_default(constraints.get("materials") or [], "материал не уточнён")
    regime = _join_or_default(constraints.get("regimes") or [], "режим не уточнён")
    prop = _join_or_default(constraints.get("properties") or [], "свойство не уточнено")
    findings = [
        f"Материал: {material}.",
        f"Режим: {regime}.",
        f"Свойство: {prop}.",
        "Найдены только частичные совпадения; их нельзя считать ответом на исходный вопрос.",
    ]
    return HumanAnswer(
        title="Точных данных не найдено",
        summary=(
            f"Точных данных по сочетанию {material} + {regime} + {prop} в загруженном корпусе не найдено. "
            "Система не нашла одного эксперимента, где одновременно связаны все указанные ограничения."
        ),
        key_findings=findings,
        caveats=["Источники частичных совпадений не подтверждают несуществующий exact-факт."],
        recommendation=f"Пробел в данных: нужны эксперименты по {material} после режима {regime} с измерением свойства {prop}.",
        confidence_label="высокая",
    )


def _overview_answer(payload: dict[str, Any]) -> HumanAnswer:
    constraints = payload.get("constraints") or {}
    subject = _join_or_default(
        constraints.get("materials") or constraints.get("regimes") or constraints.get("properties") or [],
        "запросу",
    )
    facts = payload.get("facts") or []
    evidence = payload.get("evidence") or payload.get("sources") or []
    regimes = _unique(row.get("regime") for row in facts)
    props = _unique(row.get("property") for row in facts)
    materials = _unique(row.get("material") for row in facts)
    if not facts:
        findings = [
            "Подтверждённых graph facts по этому запросу не найдено.",
        ]
        if evidence:
            findings.append("Найдены только релевантные фрагменты/evidence; они не считаются подтверждёнными экспериментальными фактами.")
        else:
            findings.append("Релевантные источники для ответа также не найдены.")
        return HumanAnswer(
            title=f"Структурированных фактов по {subject} не найдено",
            summary=(
                f"По {subject} в графе нет подтверждённых экспериментов, режимов или измеренных свойств. "
                "Система не должна делать положительный вывод без graph facts."
            ),
            key_findings=findings,
            caveats=["Partial evidence можно использовать только как навигацию к источникам, а не как доказанный факт."],
            recommendation=f"Для ответа нужны документы, где явно описан {subject} и связанные с ним режимы или свойства.",
            confidence_label="высокая" if not evidence else "средняя",
        )
    findings = []
    if regimes:
        findings.append(f"Режимы: {', '.join(regimes[:6])}.")
    if props:
        findings.append(f"Измерялись свойства: {', '.join(props[:6])}.")
    if materials and not constraints.get("materials"):
        findings.append(f"Материалы: {', '.join(materials[:6])}.")
    if not findings:
        findings.append("Структурированных фактов мало; см. источники и граф ниже.")
    gaps = payload.get("data_gaps") or payload.get("gaps") or []
    caveats = ["Это обзор по корпусу, а не доказательство конкретного эффекта для одного режима."] if facts else []
    if gaps:
        caveats.append(f"В графе отмечены пробелы в данных: {len(gaps)}.")
    return HumanAnswer(
        title=f"Обзор по {subject}",
        summary=f"По {subject} в корпусе найдены связанные эксперименты, режимы и измеренные свойства.",
        key_findings=findings,
        caveats=caveats,
        recommendation="Для строгого вывода задайте материал, режим и свойство в одном вопросе.",
        confidence_label=_confidence(payload, facts),
    )


def _lab_activity_answer(payload: dict[str, Any]) -> HumanAnswer:
    labs = _entity_names(payload.get("laboratories") or [])
    teams = _entity_names(payload.get("teams") or payload.get("research_teams") or [])
    facts = payload.get("facts") or []
    for fact in facts:
        if isinstance(fact, dict):
            labs.extend(_list_values(fact.get("laboratory") or fact.get("laboratories")))
            teams.extend(_list_values(fact.get("team") or fact.get("teams")))
    labs = _unique(labs)
    teams = _unique(teams)
    findings: list[str] = []
    if labs:
        findings.append(f"Лаборатории: {', '.join(labs[:8])}.")
    if teams:
        findings.append(f"Команды: {', '.join(teams[:8])}.")
    if not findings:
        findings.append("Явно выделенных лабораторий или команд в структурированных фактах не найдено.")
    return HumanAnswer(
        title="Лаборатории и команды",
        summary=(
            "Система проверила структурированные сущности Laboratory/ResearchTeam и связанные эксперименты. "
            "Ниже перечислены только явно извлечённые названия без внутренних идентификаторов."
        ),
        key_findings=findings,
        caveats=["Если лаборатория указана только в свободном тексте источника и не извлечена как сущность, она останется в evidence, а не в списке фактов."],
        recommendation="Для аудита откройте блок «Основание ответа» и raw diagnostics с источниками.",
        confidence_label="средняя" if labs or teams else "высокая",
    )


def _comparison_answer(payload: dict[str, Any]) -> HumanAnswer:
    facts = [fact for fact in payload.get("facts") or payload.get("primary_facts") or [] if isinstance(fact, dict)]
    constraints = payload.get("constraints") or {}
    requested_materials = [str(item) for item in constraints.get("materials") or [] if item]
    requested_property = _join_or_default(constraints.get("properties") or [], "прочность")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        material = str(fact.get("material") or "").strip()
        if not material:
            continue
        if requested_materials and not any(_same_text(material, requested) for requested in requested_materials):
            continue
        if constraints.get("properties") and not _same_text(str(fact.get("property") or ""), str(constraints["properties"][0])):
            continue
        grouped.setdefault(material, []).append(fact)

    ordered_materials = requested_materials or list(grouped)
    findings: list[str] = []
    converted_notes: list[str] = []
    maxima: dict[str, float] = {}
    for material in ordered_materials:
        rows = grouped.get(material) or []
        summary, max_value, notes = _comparison_material_summary(material, rows)
        findings.append(summary)
        if max_value is not None:
            maxima[material] = max_value
        converted_notes.extend(notes)
    if not findings:
        findings = ["Для сравнения не хватает структурированных численных фактов."]
    caveats = [
        "Сравнение ограничено: найденные значения относятся к разным режимам, состояниям материала или единицам измерения. "
        "Это обзор доступных фактов, а не прямое экспериментальное сравнение в одинаковых экспериментальных условиях."
    ]
    if converted_notes:
        caveats.append("Часть значений была задана в ksi и пересчитана в MPa: " + "; ".join(_unique(converted_notes[:4])) + ".")
    recommendation = "Для корректного сравнения нужны факты в одинаковом режиме, состоянии материала и единицах измерения."
    if len(maxima) >= 2:
        best_material, best_value = max(maxima.items(), key=lambda item: item[1])
        recommendation = (
            f"В найденном корпусе {best_material} имеет более высокий верхний уровень {requested_property} "
            f"(около {best_value:g} MPa), но это нельзя считать строгим преимуществом без одинаковых условий испытаний."
        )
    return HumanAnswer(
        title=f"Сравнение {requested_property} по найденным данным",
        summary=(
            f"В корпусе есть данные по {requested_property} для сравниваемых материалов, "
            "но прямое сравнение ограничено различием режимов, исходных состояний или единиц измерения."
        ),
        key_findings=findings,
        caveats=caveats,
        recommendation=recommendation,
        confidence_label="средняя" if facts else "низкая",
    )


def _gap_answer(payload: dict[str, Any]) -> HumanAnswer:
    gaps = payload.get("data_gaps") or payload.get("gaps") or []
    findings = []
    seen = set()
    for gap in gaps[:6]:
        if isinstance(gap, dict):
            subject = _gap_subject(gap, payload)
            reason = str(gap.get("reason") or gap.get("gap") or "данные отсутствуют").strip()
            line = f"{subject}: {reason}."
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            findings.append(line)
    if not findings:
        findings = ["По заданным ограничениям явных пробелов в графе не найдено."]
    return HumanAnswer(
        title="Пробелы в данных",
        summary="Система проверила граф на отсутствие фактов по указанным ограничениям.",
        key_findings=findings,
        caveats=[],
        recommendation="Приоритет для доразметки: добавить документы или таблицы, где явно связаны материал, режим и отсутствующее свойство.",
        confidence_label="средняя" if gaps else "высокая",
    )


def _gap_subject(gap: dict[str, Any], payload: dict[str, Any]) -> str:
    explicit = " + ".join(str(gap.get(key)) for key in ["material", "regime", "property"] if gap.get(key))
    if explicit:
        return explicit
    missing_for = str(gap.get("missing_for") or "").strip()
    text = " ".join(
        str(item)
        for item in [
            missing_for,
            gap.get("gap"),
            gap.get("reason"),
            *(str(source.get("quote") or "") for source in payload.get("sources") or [] if isinstance(source, dict)),
            *(str(source.get("quote") or "") for source in payload.get("evidence") or [] if isinstance(source, dict)),
        ]
    ).lower()
    if "корроз" in text or "corrosion" in text:
        return "коррозионная стойкость"
    if "прочн" in text or "strength" in text:
        return "прочность"
    if "тверд" in text or "твёрд" in text or "hardness" in text:
        return "твёрдость"
    return missing_for or "Неуточнённая область"


def _conflict_answer(payload: dict[str, Any]) -> HumanAnswer:
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    conflicts = [item for item in diagnostics.get("fact_conflicts") or [] if isinstance(item, dict)]
    facts = payload.get("primary_facts") or payload.get("facts") or []
    findings = []
    for conflict in conflicts[:6]:
        material = str(conflict.get("material") or "материал").strip()
        regime = str(conflict.get("regime") or "режим").strip()
        prop = str(conflict.get("property") or "свойство").strip()
        values = _format_conflict_values(conflict.get("values") or [])
        if values:
            findings.append(f"{material}: {regime}; {prop} расходится между источниками: {values}.")
        else:
            findings.append(f"{material}: {regime}; {prop} имеет разные качественные эффекты в источниках.")
    if not findings and facts:
        grouped = _conflict_candidates_from_facts(facts)
        findings = grouped[:6]
    if not findings:
        findings = ["В canonical fact layer не найдено групп с расходящимися значениями для одной связки material + regime + property."]
    return HumanAnswer(
        title="Неоднородность данных",
        summary=(
            "Система проверила canonical facts и сгруппировала случаи, где для одного материала, режима и свойства "
            "найдены разные значения или эффекты."
        ),
        key_findings=findings,
        caveats=[
            "Это не выбор единственно правильного значения: расхождения могут быть связаны с источниками, параметрами режима или исходным состоянием материала."
        ],
        recommendation="Для строгого вывода откройте evidence и сравните условия экспериментов в источниках.",
        confidence_label="средняя" if conflicts else "низкая",
    )


def _conflict_candidates_from_facts(facts: list[dict[str, Any]]) -> list[str]:
    groups: dict[tuple[str, str, str], set[str]] = {}
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        material = str(fact.get("material") or "").strip()
        regime = str(fact.get("regime") or "").strip()
        prop = str(fact.get("property") or "").strip()
        value = fact.get("value_normalized") if fact.get("value_normalized") is not None else fact.get("value")
        unit = fact.get("unit_normalized") or fact.get("unit") or ""
        if not (material and regime and prop and value is not None):
            continue
        groups.setdefault((material, regime, prop), set()).add(f"{_format_number(value)} {unit}".strip())
    result = []
    for (material, regime, prop), values in groups.items():
        if len(values) > 1:
            result.append(f"{material}: {regime}; {prop} расходится между источниками: {' и '.join(sorted(values))}.")
    return result


def _history_answer(payload: dict[str, Any]) -> HumanAnswer:
    rows = payload.get("decision_history") or payload.get("facts") or []
    constraints = payload.get("constraints") or {}
    material = _join_or_default(constraints.get("materials") or [], "материалу")
    regimes = _unique(row.get("regime") for row in rows if isinstance(row, dict))
    props = _unique(row.get("property") for row in rows if isinstance(row, dict))
    findings = []
    if regimes:
        findings.append(f"В истории встречаются режимы: {', '.join(regimes[:6])}.")
    if props:
        findings.append(f"Измерялись свойства: {', '.join(props[:6])}.")
    if not findings:
        findings.append(f"Найдено записей истории решений: {len(rows)}.")
    return HumanAnswer(
        title=f"История решений по {material}",
        summary=(
            f"По {material} показана цепочка найденных экспериментов и решений. "
            "Это навигационный обзор, а не один exact-effect ответ."
        ),
        key_findings=findings,
        caveats=["Для строгого вывода по эффекту задайте material + regime + property."],
        recommendation="Откройте блок «История решений» и «Источники и evidence» для проверки конкретных записей.",
        confidence_label=_confidence(payload, rows if isinstance(rows, list) else []),
    )


def _similar_answer(payload: dict[str, Any]) -> HumanAnswer:
    facts = payload.get("primary_facts") or payload.get("facts") or []
    findings = []
    for fact in facts[:5]:
        score = fact.get("similarity_score")
        score_text = f", score {score:.2f}" if isinstance(score, float) else ""
        findings.append(f"{fact.get('material') or 'материал не указан'}, {fact.get('regime') or 'режим не указан'}, {fact.get('property') or 'свойство не указано'}{score_text}.")
    if not findings:
        findings = ["Похожих экспериментов в графе не найдено."]
    return HumanAnswer(
        title="Похожие эксперименты",
        summary="Похожие эксперименты ранжированы по совпадению материала, режима, свойства, оборудования и лаборатории.",
        key_findings=findings,
        caveats=["Similarity score объясняет близость контекста, но не является доказательством одинакового эффекта."],
        recommendation="Используйте похожие эксперименты как навигацию по корпусу, а не как exact answer.",
        confidence_label=_confidence(payload, facts),
    )


def _strict_audit_answer(payload: dict[str, Any]) -> HumanAnswer:
    status = str(payload.get("status") or "unknown")
    constraints = payload.get("constraints") or {}
    materials = constraints.get("materials") or []
    regimes = constraints.get("regimes") or []
    properties = constraints.get("properties") or []
    facts = payload.get("primary_facts") or payload.get("facts") or []
    if status == "no_exact_match":
        summary = "Статус проверки: точное совпадение не найдено."
        recommendation = "Аудиторский вывод: положительный ответ запрещён, потому что exact graph path отсутствует."
    else:
        summary = "Статус проверки: точное совпадение найдено." if facts else f"Статус проверки: {status}."
        recommendation = "Аудиторский вывод: ответ основан только на структурированных фактах графа."
    path = (
        f"Проверенная цепочка: Material({_join_or_default(materials, '*')}) -> Experiment -> "
        f"Regime({_join_or_default(regimes, '*')}) -> Measurement({_join_or_default(properties, '*')})."
    )
    findings = [
        path,
        f"Количество exact-фактов: {len(payload.get('facts') or [])}.",
        f"Количество primary facts: {len(payload.get('primary_facts') or [])}.",
        f"Количество источников/evidence: {len(payload.get('evidence') or [])}.",
    ]
    if status == "no_exact_match":
        findings.append("Partial matches отделены от exact facts и не используются как ответ.")
    return HumanAnswer(
        title="Строгая проверка графа",
        summary=summary,
        key_findings=findings,
        caveats=["Интерпретация минимальна: показан только результат проверки структурной цепочки."],
        recommendation=recommendation,
        confidence_label="высокая" if payload.get("evidence") or status == "no_exact_match" else "средняя",
    )


def _is_comparison(payload: dict[str, Any]) -> bool:
    return str(payload.get("answer_mode")) == "comparison" or "comparison" in str(payload.get("analytical_intent") or "")


def _is_gap_answer(payload: dict[str, Any]) -> bool:
    return (
        str(payload.get("answer_mode")) in {"gaps", "graph_gap_analysis"}
        or str(payload.get("analytical_intent")) == "gap_analysis"
        or str(payload.get("intent")) == "gap_analysis"
    )


def _is_conflict_answer(payload: dict[str, Any]) -> bool:
    return (
        str(payload.get("answer_mode")) in {"conflict", "graph_conflict_analysis"}
        or str(payload.get("analytical_intent")) == "conflict_analysis"
        or str(payload.get("intent")) == "conflict_analysis"
    )


def _is_similar_answer(payload: dict[str, Any]) -> bool:
    return str(payload.get("analytical_intent")) == "similar_experiments"


def _is_history_answer(payload: dict[str, Any]) -> bool:
    return str(payload.get("answer_mode")) in {"history", "graph_decision_history"} or str(payload.get("analytical_intent")) == "decision_history"


def _is_lab_activity_answer(payload: dict[str, Any]) -> bool:
    intent = str(payload.get("analytical_intent") or payload.get("intent") or "")
    constraints = payload.get("constraints") if isinstance(payload.get("constraints"), dict) else {}
    raw_question = str(constraints.get("raw_question") or "").lower()
    return intent == "lab_activity" or any(term in raw_question for term in ["лаборатор", "команд", "laboratory", "team"])


def _is_overview_answer(payload: dict[str, Any]) -> bool:
    intent = str(payload.get("analytical_intent") or "")
    return payload.get("answer_mode") == "overview" or intent.endswith("_overview") or intent in {"topic_search", "graph_neighborhood", "equipment_usage", "lab_activity"}


def _is_technical_object_payload(payload: dict[str, Any]) -> bool:
    if _is_gap_answer(payload) or _is_conflict_answer(payload):
        return False
    if payload.get("intent") in {"object_overview", "parameter_lookup", "part_article_lookup", "material_lookup", "standard_lookup", "requirement_lookup"}:
        return True
    return any(isinstance(fact, dict) and fact.get("predicate") for fact in payload.get("facts") or [])


def _fact_sentence(fact: dict[str, Any], material_override: str | None = None) -> str:
    material = material_override or fact.get("material") or "материал не указан"
    regime = fact.get("regime") or "режим не указан"
    measurement = _measurement_phrase(fact)
    effect = _effect_label(fact.get("effect"))
    effect_text = f"эффект: {effect}" if effect != "эффект не указан явно" else "направление эффекта не указано явно"
    return f"{material}: {regime}; {measurement}; {effect_text}."


def _measurement_phrase(fact: dict[str, Any]) -> str:
    prop = fact.get("property") or "свойство"
    value = fact.get("value") if fact.get("value") is not None else fact.get("raw_value")
    unit = fact.get("unit") or ""
    if value is None or value == "":
        return f"{prop}: численное значение не указано"
    return f"{prop}: {value:g} {unit}".strip() if isinstance(value, float) else f"{prop}: {value} {unit}".strip()


def _comparison_material_summary(material: str, rows: list[dict[str, Any]]) -> tuple[str, float | None, list[str]]:
    if not rows:
        return f"{material}: структурированных численных значений для сравнения не найдено.", None, []
    converted_values: list[float] = []
    original_values: list[str] = []
    effects: list[str] = []
    notes: list[str] = []
    for fact in rows:
        value = fact.get("value") if fact.get("value") is not None else fact.get("raw_value")
        normalized_value = fact.get("value_normalized")
        normalized_unit = fact.get("unit_normalized")
        if normalized_value is not None and normalized_unit == "MPa":
            converted_values.append(float(normalized_value))
            if str(fact.get("unit_original") or fact.get("unit") or "").strip() == "ksi":
                notes.append(f"{float(fact.get('value_original') if fact.get('value_original') is not None else value):g} ksi ≈ {float(normalized_value):.0f} MPa")
        else:
            converted, note = normalize_strength_to_mpa(value, fact.get("unit"))
            if converted is not None:
                converted_values.append(converted)
                if note:
                    notes.append(note)
        if value is not None and value != "":
            original_unit = str(fact.get("unit") or "").strip()
            original_values.append(f"{float(value):g} {original_unit}".strip() if isinstance(value, int | float) else f"{value} {original_unit}".strip())
        effects.append(_effect_label(fact.get("effect")))
    effect_text = ", ".join(_unique(effects)) if effects else "эффект не указан явно"
    if converted_values:
        low, high = min(converted_values), max(converted_values)
        if low == high:
            range_text = f"примерно {_format_mpa(low)} MPa"
        else:
            range_text = f"примерно {_format_mpa(low)}-{_format_mpa(high)} MPa"
        strongest = sorted({round(value) for value in converted_values}, reverse=True)[:3]
        strongest_text = ", ".join(f"{value:g} MPa" for value in strongest)
        return (
            f"{material}: найденный диапазон прочности после пересчёта — {range_text}; "
            f"наиболее высокие значения: {strongest_text}; эффекты: {effect_text}.",
            high,
            notes,
        )
    values_text = ", ".join(_unique(original_values[:4])) if original_values else "численные значения не извлечены"
    return f"{material}: численные значения не удалось привести к MPa; найдено: {values_text}; эффекты: {effect_text}.", None, notes


def _with_conflict_caveats(answer: HumanAnswer, payload: dict[str, Any]) -> HumanAnswer:
    diagnostics = payload.get("diagnostics") or {}
    conflicts = [item for item in diagnostics.get("fact_conflicts") or [] if isinstance(item, dict)]
    if not conflicts:
        return answer
    existing = set(answer.caveats)
    for caveat in [_format_conflict_caveat(item) for item in conflicts[:2]]:
        if caveat and caveat not in existing:
            answer.caveats.append(caveat)
            existing.add(caveat)
    if answer.confidence_label == "высокая":
        answer.confidence_label = "средняя"
    return answer


def _format_conflict_caveat(conflict: dict[str, Any]) -> str:
    material = str(conflict.get("material") or "материала").strip()
    regime = str(conflict.get("regime") or "").strip()
    prop = _property_genitive(str(conflict.get("property") or "свойства"))
    values = _format_conflict_values(conflict.get("values") or [])
    regime_text = _regime_phrase(regime)
    if values:
        return (
            f"В корпусе найдены разные значения {prop} для {material}{regime_text}: {values}. "
            "Это может быть связано с различиями в параметрах режима, источниках или исходном состоянии материала."
        )
    return (
        f"В корпусе найдены разные качественные эффекты по {prop} для {material}{regime_text}. "
        "Это требует проверки источников и условий эксперимента."
    )


def _format_conflict_values(values: list[Any]) -> str:
    rendered: list[str] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        unit = item.get("unit")
        if value is None or not unit:
            continue
        rendered.append(f"{_format_number(value)} {unit}")
    return " и ".join(_unique(rendered[:4]))


def _format_number(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.0f}" if abs(numeric) >= 10 else f"{numeric:g}"


def _property_genitive(value: str) -> str:
    return {
        "прочность": "прочности",
        "твёрдость": "твёрдости",
        "твердость": "твёрдости",
        "пластичность": "пластичности",
        "вязкость": "вязкости",
        "коррозионная стойкость": "коррозионной стойкости",
    }.get(value, value)


def _regime_phrase(value: str) -> str:
    return {
        "отжиг": " после отжига",
        "старение": " после старения",
        "закалка": " после закалки",
        "криообработка": " после криообработки",
        "термообработка": " после термообработки",
    }.get(value, f" при режиме {value}" if value else "")


def _effect_label(effect: Any) -> str:
    return {
        "increase": "рост",
        "decrease": "снижение",
        "no_change": "без заметного изменения",
        "unchanged": "без заметного изменения",
        "mixed": "смешанный эффект",
        "unknown": "эффект не указан явно",
        None: "эффект не указан явно",
        "": "эффект не указан явно",
    }.get(str(effect), str(effect))


def _same_text(left: str, right: str) -> bool:
    left_norm = left.strip().lower().replace("ё", "е")
    right_norm = right.strip().lower().replace("ё", "е")
    return bool(left_norm and right_norm and (left_norm == right_norm or left_norm in right_norm or right_norm in left_norm))


def _format_mpa(value: float) -> str:
    return f"{value:.0f}" if abs(value) >= 10 else f"{value:g}"


def _strict_recommendation(material: str, regime: str, prop: str, facts: list[dict[str, Any]]) -> str:
    if any(str(f.get("effect") or "").lower() == "increase" for f in facts) and len(_numeric_values(facts)) > 1:
        return (
            f"По корпусу нельзя утверждать, что любой режим {regime} для {material} одинаково влияет на {prop}; "
            "подтверждённый эффект зависит от параметров режима и источника."
        )
    return "Вывод основан на exact graph facts; для расширения вывода проверьте источники и supporting facts."


def _confidence(payload: dict[str, Any], facts: list[dict[str, Any]]) -> Literal["высокая", "средняя", "низкая"]:
    if payload.get("status") == "no_exact_match":
        return "высокая"
    if len(facts) >= 2 and payload.get("evidence"):
        return "высокая"
    if facts or payload.get("sources"):
        return "средняя"
    return "низкая"


def _numeric_values(facts: list[dict[str, Any]]) -> list[float]:
    values = []
    for fact in facts:
        value = fact.get("value_normalized") if fact.get("value_normalized") is not None else fact.get("value")
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def _format_value_range(values: list[float], facts: list[dict[str, Any]]) -> str:
    unit = next((str(f.get("unit_normalized") or f.get("unit")) for f in facts if f.get("unit_normalized") or f.get("unit")), "")
    if not values:
        return "нет численных значений"
    if len(values) == 1 or min(values) == max(values):
        return f"{values[0]:g} {unit}".strip()
    return f"{min(values):g}-{max(values):g} {unit}".strip()


def _join_or_default(values: list[Any], default: str) -> str:
    cleaned = [str(value) for value in values if value]
    return ", ".join(dict.fromkeys(cleaned)) if cleaned else default


def _entity_names(rows: list[Any]) -> list[str]:
    result: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            value = row.get("canonical_name") or row.get("name") or row.get("label")
        else:
            value = row
        if value:
            result.append(str(value))
    return result


def _list_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []


def _unique(values) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _fact_source_text(fact: dict[str, Any]) -> str:
    pieces = []
    for evidence in fact.get("evidence") or []:
        if isinstance(evidence, dict):
            pieces.append(str(evidence.get("source_name") or ""))
            pieces.append(str(evidence.get("quote") or ""))
    return " ".join(pieces)


def _with_evidence_count(context: dict[str, Any], evidence: list[dict[str, Any]], payload: dict[str, Any] | None = None) -> dict[str, Any]:
    result = dict(context)
    result["evidence_count"] = len(evidence)
    if "sources_count" not in result:
        result["sources_count"] = len(evidence)
    if payload is not None:
        subgraph = payload.get("subgraph") or {}
        result["subgraph_nodes"] = len(subgraph.get("nodes") or [])
        result["subgraph_edges"] = len(subgraph.get("edges") or [])
    return result


def _sanitize_main_answer(text: str) -> str:
    text = INTERNAL_ID_RE.sub("", text)
    text = text.replace("effect:", "эффект:")
    text = re.sub(r"\bunknown\b", "эффект не указан явно", text)
    text = re.sub(r"\bincrease\b", "рост", text)
    text = re.sub(r"\bdecrease\b", "снижение", text)
    return "\n".join(re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.splitlines()).strip()
