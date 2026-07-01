from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.answer_graph import build_answer_graph  # noqa: E402
from app.ui_helpers import answer_evidence_summary_rows  # noqa: E402


ARTIFACT_PATH = ROOT / "artifacts" / "eval_demo_regression.json"
DEFAULT_API_BASE = os.getenv("API_BASE", "http://localhost:8000")
PRESET_ID = "expert_max"

RAW_LEAK_RE = re.compile(
    r"\b(?:technical_answer|doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|"
    r"EXP-[A-Za-z0-9_-]+|SCI-[A-Za-z0-9_-]+|PropertyValue|SourceChunk|"
    r"Experiment|MEASURES|OF_PROPERTY|STUDIES|increase|decrease|unknown)\b",
    re.IGNORECASE,
)
RAW_SOURCE_RE = re.compile(r"\b(?:doc_[A-Za-z0-9_:-]+|chunk_[A-Za-z0-9_:-]+|EXP-[A-Za-z0-9_-]+|SCI-[A-Za-z0-9_-]+)\b")
MEASUREMENT_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:MPa|МПа|ksi|HV|HRC|%|°C|C|ч|h)\b", re.IGNORECASE)


@dataclass(frozen=True)
class DemoCase:
    case_id: str
    question: str
    description: str
    check: Callable[[dict[str, Any]], list[str]]


def raw_leak_count(text: Any) -> int:
    return len(RAW_LEAK_RE.findall(str(text or "")))


def graph_contract(payload: dict[str, Any]) -> dict[str, Any]:
    graph = build_answer_graph(payload)
    labels = "\n".join(str(node.label) for node in graph.nodes)
    return {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "raw_label_leaks": raw_leak_count(labels),
        "labels": [str(node.label) for node in graph.nodes],
    }


def evidence_summary_has_raw_ids(payload: dict[str, Any]) -> bool:
    rendered = json.dumps(answer_evidence_summary_rows(payload), ensure_ascii=False)
    return bool(RAW_SOURCE_RE.search(rendered))


def friendly_source_warnings(payload: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for row in answer_evidence_summary_rows(payload):
        source = str(row.get("Источник") or "")
        lowered = source.lower()
        if RAW_SOURCE_RE.search(source):
            warnings.append(f"source contains raw id: {source}")
        if any(token in lowered for token in ["synthetic", "demo", "test", "doc_", "chunk_"]):
            warnings.append(f"source contains technical token: {source}")
        if re.search(r"\.(csv|txt|html|htm|xlsx|md)$", lowered):
            warnings.append(f"source looks like raw filename: {source}")
    return warnings


def negative_query_has_hallucinated_number(answer: str) -> bool:
    return bool(MEASUREMENT_NUMBER_RE.search(answer or ""))


def comparison_normalized_unit_errors(answer: str) -> list[str]:
    errors: list[str] = []
    if "MPa" not in answer and "МПа" not in answer:
        errors.append("comparison answer does not mention MPa")
    for material in ["ВТ6", "7075-T6"]:
        if material not in answer:
            errors.append(f"comparison answer does not mention {material}")
    if "ksi" in answer.lower() and not re.search(r"(≈|исход|пересч|нормализ|converted|conversion)", answer, re.IGNORECASE):
        errors.append("ksi appears without conversion/original-value explanation")
    if not re.search(r"(неоднород|разн\w* режим|разн\w* источник|источник|режим|услов)", answer, re.IGNORECASE):
        errors.append("comparison answer lacks comparability/conflict caveat")
    return errors


def _answer(payload: dict[str, Any]) -> str:
    return str(payload.get("answer") or "")


def _diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else {}


def _conflict_count(payload: dict[str, Any]) -> int:
    conflicts = _diagnostics(payload).get("fact_conflicts") or []
    return len(conflicts) if isinstance(conflicts, list) else 0


def _generic_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    answer = _answer(payload)
    diagnostics = _diagnostics(payload)
    if payload.get("status") not in {"ok", "no_exact_match", "partial"}:
        errors.append(f"unexpected response status: {payload.get('status')}")
    if diagnostics.get("preset_id") != PRESET_ID:
        errors.append(f"expected diagnostics.preset_id={PRESET_ID}, got {diagnostics.get('preset_id')}")
    if answer.strip().lower().startswith("офлайн-режим"):
        errors.append("answer starts with offline mode banner")
    leaks = raw_leak_count(answer)
    if leaks:
        errors.append(f"main answer raw leaks: {leaks}")
    graph = graph_contract(payload)
    if graph["nodes"] > 10:
        errors.append(f"answer graph nodes exceed contract: {graph['nodes']}")
    if graph["edges"] > 12:
        errors.append(f"answer graph edges exceed contract: {graph['edges']}")
    if graph["raw_label_leaks"]:
        errors.append(f"answer graph raw label leaks: {graph['raw_label_leaks']}")
    if evidence_summary_has_raw_ids(payload):
        errors.append("evidence summary contains raw doc/chunk ids")
    errors.extend(friendly_source_warnings(payload))
    return errors


def _check_exact(payload: dict[str, Any]) -> list[str]:
    errors = []
    answer = _answer(payload)
    if payload.get("status") != "ok":
        errors.append("exact graph query did not return ok")
    if not payload.get("facts"):
        errors.append("exact graph query returned no facts")
    if not (payload.get("evidence") or payload.get("sources")):
        errors.append("exact graph query returned no evidence/provenance")
    for token in ["ВТ6", "отжиг", "прочность"]:
        if token.lower() not in answer.lower():
            errors.append(f"exact answer misses {token}")
    return errors


def _check_comparison(payload: dict[str, Any]) -> list[str]:
    errors = comparison_normalized_unit_errors(_answer(payload))
    if payload.get("answer_mode") != "comparison":
        errors.append(f"expected comparison answer_mode, got {payload.get('answer_mode')}")
    if _conflict_count(payload) < 1 and not re.search(r"(неоднород|разные значения|источник|режим)", _answer(payload), re.IGNORECASE):
        errors.append("comparison lacks conflict summary/caveat")
    return errors


def _check_conflicts(payload: dict[str, Any]) -> list[str]:
    answer = _answer(payload)
    errors = []
    if _conflict_count(payload) < 1 and not re.search(r"(разные значения|неоднород|противореч)", answer, re.IGNORECASE):
        errors.append("conflict query did not expose conflict groups or conflict wording")
    if not re.search(r"(разные значения|неоднород|источник|режим)", answer, re.IGNORECASE):
        errors.append("conflict answer is not human-readable enough")
    if re.search(r"(единственно правильн|абсолютно истинн|точно правильн)", answer, re.IGNORECASE):
        errors.append("conflict answer claims one value is absolutely correct")
    return errors


def _check_gaps(payload: dict[str, Any]) -> list[str]:
    answer = _answer(payload)
    gaps = payload.get("data_gaps") or payload.get("gaps") or []
    if gaps:
        return [] if re.search(r"(пробел|не измер|нет данных|gap)", answer, re.IGNORECASE) else ["gaps exist but answer does not explain them"]
    return [] if re.search(r"(пробел|не найден|нет данных|не выяв)", answer, re.IGNORECASE) else ["no controlled no-gaps/no-data answer"]


def _check_evidence(payload: dict[str, Any]) -> list[str]:
    errors = []
    answer = _answer(payload)
    if not (payload.get("evidence") or payload.get("sources")):
        errors.append("evidence query returned no evidence")
    if not answer_evidence_summary_rows(payload):
        errors.append("evidence query has no user-facing evidence summary rows")
    for token in ["7075-T6", "прочность"]:
        if token.lower() not in answer.lower():
            errors.append(f"evidence answer misses {token}")
    return errors


def _check_negative(payload: dict[str, Any]) -> list[str]:
    answer = _answer(payload)
    errors = []
    if payload.get("status") not in {"no_exact_match", "ok", "partial"}:
        errors.append(f"unexpected negative status: {payload.get('status')}")
    if negative_query_has_hallucinated_number(answer):
        errors.append("negative answer contains measurement-like numeric value")
    if re.search(r"(X999.*(?:составил|достиг|показал|имеет)\s+\d+|\d+\s*(?:MPa|МПа|ksi).{0,40}X999)", answer, re.IGNORECASE):
        errors.append("negative answer appears to assert fake X999 measurement")
    if not re.search(r"(точных данных|не найден|нет данных|не удалось|отсутств)", answer, re.IGNORECASE):
        errors.append("negative answer lacks controlled no-data wording")
    return errors


def _check_lab_team(payload: dict[str, Any]) -> list[str]:
    answer = _answer(payload)
    if re.search(r"(лаборатор|команд|research team|team|данных нет|не найден|отсутств)", answer, re.IGNORECASE):
        return []
    return ["lab/team query neither lists clean labels nor gives honest no-data answer"]


DEMO_CASES = [
    DemoCase(
        "exact_vt6_anneal",
        "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?",
        "exact material/regime/property graph query",
        _check_exact,
    ),
    DemoCase(
        "comparison_strength",
        "Сравни ВТ6 и 7075-T6 по прочности.",
        "comparison, normalized units, caveat/conflict summary",
        _check_comparison,
    ),
    DemoCase(
        "conflicts_strength",
        "Какие есть противоречия или неоднородные данные по прочности?",
        "conflict detection and explanation",
        _check_conflicts,
    ),
    DemoCase(
        "data_gaps",
        "Какие пробелы в данных найдены?",
        "DataGap path",
        _check_gaps,
    ),
    DemoCase(
        "english_evidence_7075",
        "Find evidence for strength of 7075-T6 after aging treatment.",
        "English/Russian hybrid evidence retrieval",
        _check_evidence,
    ),
    DemoCase(
        "negative_x999_laser",
        "Что известно о сплаве X999 при лазерной обработке?",
        "negative/no exact match without hallucinated facts",
        _check_negative,
    ),
    DemoCase(
        "labs_or_teams",
        "Какие лаборатории или команды выполняли эксперименты?",
        "laboratory/team overview or honest no-data answer",
        _check_lab_team,
    ),
]


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _api_url(api_base: str, path: str) -> str:
    return f"{api_base.rstrip('/')}/{path.lstrip('/')}"


def _ask(api_base: str, question: str) -> dict[str, Any]:
    return _request_json(
        "POST",
        _api_url(api_base, "/ask"),
        {"question": question, "top_k": 12, "preset_id": PRESET_ID},
        timeout=180,
    )


def validate_case(case: DemoCase, payload: dict[str, Any]) -> dict[str, Any]:
    generic_errors = _generic_errors(payload)
    specific_errors = case.check(payload)
    graph = graph_contract(payload)
    evidence_rows = answer_evidence_summary_rows(payload)
    errors = generic_errors + specific_errors
    warnings = []
    if not evidence_rows and payload.get("status") == "ok" and payload.get("facts"):
        warnings.append("facts exist but no compact evidence summary rows")
    row = {
        "case_id": case.case_id,
        "question": case.question,
        "description": case.description,
        "passed": not errors,
        "reasons": errors or ["ok"],
        "raw_leaks_count": raw_leak_count(_answer(payload)) + int(graph["raw_label_leaks"]),
        "graph_nodes": graph["nodes"],
        "graph_edges": graph["edges"],
        "evidence_count": len(payload.get("evidence") or payload.get("sources") or []),
        "evidence_summary_count": len(evidence_rows),
        "conflict_count": _conflict_count(payload),
        "answer_mode": payload.get("answer_mode"),
        "status": payload.get("status"),
        "warnings": warnings,
    }
    return row


def run_eval(api_base: str = DEFAULT_API_BASE) -> tuple[dict[str, Any], int]:
    try:
        health = _request_json("GET", _api_url(api_base, "/health"), timeout=10)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        message = f"API is not available; run docker compose up first. Details: {type(exc).__name__}"
        result = {"summary": "FAIL", "error": message, "rows": []}
        return result, 1

    rows = []
    for case in DEMO_CASES:
        try:
            payload = _ask(api_base, case.question)
            row = validate_case(case, payload)
        except Exception as exc:
            row = {
                "case_id": case.case_id,
                "question": case.question,
                "description": case.description,
                "passed": False,
                "reasons": [f"request/validation failed: {type(exc).__name__}: {exc}"],
                "raw_leaks_count": 0,
                "graph_nodes": 0,
                "graph_edges": 0,
                "evidence_count": 0,
                "evidence_summary_count": 0,
                "conflict_count": 0,
                "answer_mode": "",
                "status": "",
                "warnings": [],
            }
        rows.append(row)

    failed = [row for row in rows if not row["passed"]]
    warned = [row for row in rows if row.get("warnings")]
    summary = "FAIL" if failed else ("WARN" if warned else "PASS")
    result = {
        "summary": summary,
        "api_base": api_base,
        "health": {
            "kg_backend_active": health.get("kg_backend_active"),
            "neo4j_available": health.get("neo4j_available"),
            "llm_provider": (health.get("llm") or {}).get("provider"),
            "llm_ready": (health.get("llm") or {}).get("ready"),
            "retrieval": {
                key: (health.get("retrieval") or {}).get(key)
                for key in [
                    "effective_retrieval_mode",
                    "hybrid_dense_enabled",
                    "local_embeddings_ready",
                    "local_embedding_vectors",
                    "hybrid_degraded_reason",
                ]
            },
        },
        "rows": rows,
        "failures_count": len(failed),
        "warnings_count": sum(len(row.get("warnings") or []) for row in rows),
    }
    return result, 1 if failed else 0


def main() -> int:
    result, exit_code = run_eval(DEFAULT_API_BASE)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"SUMMARY: {result['summary']}")
    if result.get("error"):
        print(result["error"])
    for row in result.get("rows", []):
        label = "PASS" if row["passed"] else "FAIL"
        reason = "; ".join(row["reasons"][:3])
        print(
            f"[{label}] {row['case_id']}: {reason} | "
            f"raw_leaks={row['raw_leaks_count']} graph={row['graph_nodes']}/{row['graph_edges']} "
            f"evidence={row['evidence_count']} conflicts={row['conflict_count']} "
            f"warnings={len(row.get('warnings') or [])}"
        )
    print(f"JSON report: {ARTIFACT_PATH}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
