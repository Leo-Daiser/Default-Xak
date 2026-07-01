from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "artifacts" / "eval_resource_ablation.json"

PROFILES = ["economy_core", "economy_guarded_llm", "balanced_hybrid", "quality_full"]


PROFILE_ENV: dict[str, dict[str, str]] = {
    "economy_core": {
        "RETRIEVAL_MODE": "bm25",
        "ENABLE_LOCAL_EMBEDDINGS": "false",
        "EAGER_LOCAL_EMBEDDINGS": "false",
        "DIRECT_QDRANT_PROJECTION": "false",
        "ENABLE_LLM": "false",
        "LLM_PROVIDER": "offline",
        "ANSWER_SYNTHESIS_MODE": "template",
    },
    "economy_guarded_llm": {
        "RETRIEVAL_MODE": "bm25",
        "ENABLE_LOCAL_EMBEDDINGS": "false",
        "EAGER_LOCAL_EMBEDDINGS": "false",
        "DIRECT_QDRANT_PROJECTION": "false",
        "ENABLE_LLM": "true",
        "LLM_PROVIDER": "auto",
        "ANSWER_SYNTHESIS_MODE": "hybrid",
    },
    "balanced_hybrid": {
        "RETRIEVAL_MODE": "hybrid",
        "ENABLE_LOCAL_EMBEDDINGS": "true",
        "EAGER_LOCAL_EMBEDDINGS": "false",
        "DIRECT_QDRANT_PROJECTION": "false",
        "ENABLE_LLM": "false",
        "LLM_PROVIDER": "auto",
        "ANSWER_SYNTHESIS_MODE": "hybrid",
        "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
    "quality_full": {
        "RETRIEVAL_MODE": "hybrid",
        "ENABLE_LOCAL_EMBEDDINGS": "true",
        "EAGER_LOCAL_EMBEDDINGS": "false",
        "DIRECT_QDRANT_PROJECTION": "false",
        "ENABLE_LLM": "true",
        "LLM_PROVIDER": "auto",
        "ANSWER_SYNTHESIS_MODE": "hybrid",
        "EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    },
}

COMMON_ENV = {
    "KG_BACKEND": "fallback",
    "EXTRACTION_MODE": "deterministic",
    "EXTRACTION_ENABLE_LLM": "false",
    "RETRIEVAL_QUERY_EXPANSION": "true",
}

_RUNNER_CODE = r"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(os.environ["RESOURCE_ABLATION_ROOT"])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from evaluation.eval_demo_regression import DEMO_CASES, PRESET_ID, validate_case

with tempfile.TemporaryDirectory() as tmp:
    os.environ["METADATA_DB_PATH"] = str(Path(tmp) / "outbox.sqlite3")
    os.environ["CATALOG_DB_PATH"] = str(Path(tmp) / "catalog.sqlite3")

    import app.api as api
    from app.retrieval.retrieval import RetrievalEngine
    from app.storage.catalog import SQLiteCatalog
    from app.storage.outbox import SQLiteOutbox

    api.graph_db = None
    api.graph_db_error = None
    api.catalog = SQLiteCatalog(Path(tmp) / "catalog.sqlite3")
    api.outbox = SQLiteOutbox(Path(tmp) / "outbox.sqlite3")
    api.retrieval_engine = RetrievalEngine()
    api.DOCUMENTS.clear()
    api.CHUNKS.clear()

    client = TestClient(api.app)
    files = [
        ("files", (path.name, path.read_bytes(), "application/octet-stream"))
        for path in sorted((ROOT / "demo_data").iterdir())
        if path.suffix.lower() in {".csv", ".xlsx", ".txt", ".html", ".htm", ".docx", ".md"}
    ]
    ingest = client.post("/ingest/documents", files=files)
    if ingest.status_code != 200:
        raise RuntimeError(ingest.text)

    health = client.get("/health").json()
    rows = []
    for case in DEMO_CASES:
        started = time.perf_counter()
        response = client.post("/ask", json={"question": case.question, "top_k": 12, "preset_id": PRESET_ID})
        latency_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code != 200:
            rows.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "passed": False,
                    "reasons": [f"HTTP {response.status_code}: {response.text[:160]}"],
                    "raw_leaks_count": 0,
                    "graph_nodes": 0,
                    "graph_edges": 0,
                    "evidence_count": 0,
                    "latency_ms": latency_ms,
                    "llm_grounding_guard_status": "skipped",
                    "guard_repair_attempted": False,
                    "guard_fallback_used": False,
                    "guard_violations_count": 0,
                    "llm_polished": False,
                    "warnings": [],
                }
            )
            continue
        payload = response.json()
        row = validate_case(case, payload)
        diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
        row["latency_ms"] = latency_ms
        row["llm_polished"] = bool(diagnostics.get("llm_answer_polished"))
        rows.append(row)

    print(json.dumps({"health": health, "rows": rows}, ensure_ascii=False))
"""


def profile_environment(profile: str, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    env.update(COMMON_ENV)
    env.update(PROFILE_ENV[profile])
    env["RUNTIME_PROFILE"] = profile
    env["RESOURCE_ABLATION_ROOT"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_profile(profile: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-c", _RUNNER_CODE],
        cwd=ROOT,
        env=profile_environment(profile),
        text=True,
        capture_output=True,
        timeout=360,
        check=False,
    )
    if result.returncode != 0:
        return {
            "profile": profile,
            "status": "FAIL",
            "error": (result.stderr or result.stdout).strip()[-1200:],
            "rows": [],
            "warnings": [],
        }
    payload = _parse_last_json_line(result.stdout)
    rows = payload.get("rows") or []
    health = payload.get("health") or {}
    summary = summarize_profile(profile, rows, health)
    return {**summary, "rows": rows, "health": _health_digest(health)}


def _parse_last_json_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError("profile runner did not emit JSON")


def summarize_profile(profile: str, rows: list[dict[str, Any]], health: dict[str, Any]) -> dict[str, Any]:
    failed = [row for row in rows if not row.get("passed")]
    warnings = _profile_warnings(profile, health)
    status = "FAIL" if failed else ("WARN" if warnings else "PASS")
    latencies = [int(row.get("latency_ms") or 0) for row in rows if row.get("latency_ms") is not None]
    guard_statuses = [str(row.get("llm_grounding_guard_status") or "skipped") for row in rows]
    unsupported = sum(
        int(row.get("guard_violations_count") or 0)
        + sum("unsupported" in str(reason).lower() or "hallucinated" in str(reason).lower() for reason in row.get("reasons") or [])
        for row in rows
    )
    llm_calls = sum(1 for row in rows if row.get("llm_polished")) + sum(1 for row in rows if row.get("guard_repair_attempted"))
    return {
        "profile": profile,
        "status": status,
        "queries_passed": len(rows) - len(failed),
        "queries_failed": len(failed),
        "raw_leaks_count": sum(int(row.get("raw_leaks_count") or 0) for row in rows),
        "unsupported_numeric_claims_count": unsupported,
        "average_latency_ms": int(mean(latencies)) if latencies else None,
        "llm_calls_count": llm_calls,
        "guard_fallback_count": guard_statuses.count("fallback"),
        "guard_repaired_count": guard_statuses.count("repaired"),
        "evidence_count": sum(int(row.get("evidence_count") or 0) for row in rows),
        "graph_contract_pass": all(int(row.get("graph_nodes") or 0) <= 10 and int(row.get("graph_edges") or 0) <= 12 for row in rows),
        "effective_retrieval_mode": ((health.get("retrieval") or {}).get("effective_retrieval_mode")),
        "resource_notes": _resource_notes(health),
        "warnings": warnings,
        "failed_cases": [row.get("case_id") for row in failed],
    }


def _profile_warnings(profile: str, health: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    retrieval = health.get("retrieval") or {}
    llm = health.get("llm") or {}
    if profile in {"economy_guarded_llm", "quality_full"} and not llm.get("ready"):
        warnings.append(f"LLM mode ran without ready LLM provider: {llm.get('last_error') or 'not ready'}")
    if profile in {"balanced_hybrid", "quality_full"} and retrieval.get("effective_retrieval_mode") != "hybrid":
        warnings.append(f"Hybrid retrieval degraded: {retrieval.get('hybrid_degraded_reason') or 'unknown reason'}")
    if profile == "economy_core" and (retrieval.get("local_embeddings_enabled") or llm.get("enabled")):
        warnings.append("economy_core has embeddings or LLM enabled")
    return warnings


def _resource_notes(health: dict[str, Any]) -> list[str]:
    retrieval = health.get("retrieval") or {}
    llm = health.get("llm") or {}
    return [
        f"retrieval={retrieval.get('effective_retrieval_mode') or retrieval.get('retrieval_mode')}",
        f"vectors={retrieval.get('local_embedding_vectors', 0)}",
        f"llm_provider={llm.get('provider')}",
        f"llm_ready={llm.get('ready')}",
    ]


def _health_digest(health: dict[str, Any]) -> dict[str, Any]:
    retrieval = health.get("retrieval") or {}
    llm = health.get("llm") or {}
    return {
        "runtime_profile": health.get("runtime_profile"),
        "retrieval": {
            key: retrieval.get(key)
            for key in [
                "retrieval_mode",
                "effective_retrieval_mode",
                "local_embeddings_enabled",
                "local_embeddings_ready",
                "local_embedding_vectors",
                "hybrid_dense_enabled",
                "hybrid_degraded_reason",
            ]
        },
        "llm": {key: llm.get(key) for key in ["enabled", "provider", "ready", "model", "last_error"]},
    }


def run_eval() -> tuple[dict[str, Any], int]:
    profiles = [run_profile(profile) for profile in PROFILES]
    failed = [row for row in profiles if row.get("status") == "FAIL"]
    result = {
        "summary": "FAIL" if failed else ("WARN" if any(row.get("status") == "WARN" for row in profiles) else "PASS"),
        "profiles": profiles,
    }
    return result, 1 if failed else 0


def _print_table(result: dict[str, Any]) -> None:
    headers = ["profile", "status", "passed", "failed", "latency_ms", "retrieval", "llm_calls", "guard_fb", "warnings"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in result.get("profiles", []):
        print(
            "| "
            + " | ".join(
                [
                    str(row.get("profile")),
                    str(row.get("status")),
                    str(row.get("queries_passed")),
                    str(row.get("queries_failed")),
                    str(row.get("average_latency_ms")),
                    str(row.get("effective_retrieval_mode")),
                    str(row.get("llm_calls_count")),
                    str(row.get("guard_fallback_count")),
                    str(len(row.get("warnings") or [])),
                ]
            )
            + " |"
        )


def main() -> int:
    result, exit_code = run_eval()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY: {result['summary']}")
    _print_table(result)
    for profile in result.get("profiles", []):
        for warning in profile.get("warnings") or []:
            print(f"[WARN] {profile['profile']}: {warning}")
        if profile.get("error"):
            print(f"[FAIL] {profile['profile']}: {profile['error']}")
    print(f"JSON report: {ARTIFACT_PATH}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
