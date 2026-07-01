from __future__ import annotations

from evaluation.eval_resource_ablation import profile_environment, summarize_profile


def test_resource_ablation_marks_missing_llm_as_warning_not_failure() -> None:
    summary = summarize_profile(
        "economy_guarded_llm",
        rows=[{"passed": True, "latency_ms": 10, "raw_leaks_count": 0, "evidence_count": 1}],
        health={
            "llm": {"ready": False, "last_error": "MISTRAL_API_KEY is missing."},
            "retrieval": {"effective_retrieval_mode": "bm25"},
        },
    )

    assert summary["status"] == "WARN"
    assert summary["queries_failed"] == 0
    assert any("LLM mode ran without ready LLM provider" in item for item in summary["warnings"])


def test_resource_ablation_profile_environment_sets_economy_core_without_external_dependencies() -> None:
    env = profile_environment("economy_core", base_env={})

    assert env["RUNTIME_PROFILE"] == "economy_core"
    assert env["RETRIEVAL_MODE"] == "bm25"
    assert env["ENABLE_LOCAL_EMBEDDINGS"] == "false"
    assert env["ENABLE_LLM"] == "false"
    assert env["LLM_PROVIDER"] == "offline"
