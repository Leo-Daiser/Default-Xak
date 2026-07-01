from __future__ import annotations

from app.runtime.profiles import (
    MINILM_MULTILINGUAL_MODEL,
    normalize_runtime_profile,
    profile_defaults,
)


def test_economy_core_disables_embeddings_and_llm_polish() -> None:
    defaults = profile_defaults("economy_core")

    assert defaults["RETRIEVAL_MODE"] == "bm25"
    assert defaults["ENABLE_LOCAL_EMBEDDINGS"] is False
    assert defaults["ENABLE_LLM"] is False
    assert defaults["LLM_PROVIDER"] == "offline"
    assert defaults["ANSWER_SYNTHESIS_MODE"] == "template"
    assert defaults["EXTRACTION_MODE"] == "deterministic"
    assert defaults["EXTRACTION_ENABLE_LLM"] is False


def test_balanced_hybrid_enables_lazy_minilm_embeddings_without_qdrant() -> None:
    defaults = profile_defaults("balanced_hybrid")

    assert defaults["RETRIEVAL_MODE"] == "hybrid"
    assert defaults["ENABLE_LOCAL_EMBEDDINGS"] is True
    assert defaults["EAGER_LOCAL_EMBEDDINGS"] is False
    assert defaults["DIRECT_QDRANT_PROJECTION"] is False
    assert defaults["EMBEDDING_MODEL"] == MINILM_MULTILINGUAL_MODEL


def test_unknown_runtime_profile_falls_back_to_economy_core() -> None:
    assert normalize_runtime_profile("unknown-heavy-profile") == "economy_core"
