from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from app.config import settings  # noqa: E402


_AUTO = object()


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def check_embeddings_runtime(sentence_transformer_cls: Any = _AUTO) -> dict[str, Any]:
    """Return a secret-free local embeddings runtime report.

    The function is intentionally small and import-injected for tests. By
    default it tries to load and encode with the configured model. Set
    EMBEDDINGS_SKIP_MODEL_LOAD=true to verify only that the dependency imports.
    """

    model_name = os.getenv("EMBEDDING_MODEL") or getattr(
        settings,
        "embedding_model",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    report: dict[str, Any] = {
        "retrieval_mode": os.getenv("RETRIEVAL_MODE") or getattr(settings, "retrieval_mode", "bm25"),
        "enable_local_embeddings": _bool_env(
            "ENABLE_LOCAL_EMBEDDINGS",
            bool(getattr(settings, "enable_local_embeddings", False)),
        ),
        "embedding_model": model_name,
        "sentence_transformers_import_ok": False,
        "model_load_ok": False,
        "short_embedding_ok": False,
        "vector_dimension": None,
        "model_load_skipped": False,
        "error": "",
    }

    if sentence_transformer_cls is _AUTO:
        try:
            with warnings.catch_warnings(), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                warnings.simplefilter("ignore")
                logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
                logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
                logging.getLogger("transformers").setLevel(logging.ERROR)
                from sentence_transformers import SentenceTransformer as sentence_transformer_cls  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            report["error"] = _safe_error(exc)
            return report
    elif sentence_transformer_cls is None:
        report["error"] = "sentence-transformers dependency missing"
        return report

    report["sentence_transformers_import_ok"] = True
    if _bool_env("EMBEDDINGS_SKIP_MODEL_LOAD", False):
        report["model_load_skipped"] = True
        return report

    try:
        with warnings.catch_warnings(), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            warnings.simplefilter("ignore")
            model = sentence_transformer_cls(model_name)
            report["model_load_ok"] = True
            dimension_getter = getattr(
                model,
                "get_sentence_embedding_dimension",
                getattr(model, "get_embedding_dimension", lambda: None),
            )
            dimension = dimension_getter()
            encoded = model.encode(["test"], normalize_embeddings=True, show_progress_bar=False)
        vector = encoded[0] if len(encoded) else []
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        report["short_embedding_ok"] = bool(vector)
        report["vector_dimension"] = int(dimension or len(vector or [])) if (dimension or vector) else None
    except Exception as exc:
        report["error"] = _safe_error(exc)
    return report


def main() -> int:
    report = check_embeddings_runtime()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["sentence_transformers_import_ok"]:
        return 2
    if report["model_load_skipped"]:
        return 0
    if not report["model_load_ok"] or not report["short_embedding_ok"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
