"""Resource-efficient file and corpus readiness profiler."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..domain.fact_normalization import build_conflict_summary, dedupe_fact_rows, fact_rows_from_experiments
from ..extraction.pipeline import ExtractionPipeline
from ..extraction.to_graph_models import bundle_to_data_gaps, bundle_to_experiment_facts
from ..ingestion.parser_router import ParserRouter
from .text_quality import SUPPORTED_FILE_EXTENSIONS, text_quality_metrics


def profile_file(
    path: str | Path,
    *,
    parser: ParserRouter | None = None,
    pipeline: ExtractionPipeline | None = None,
) -> dict[str, Any]:
    """Profile one document without LLM, embeddings, Neo4j or API state."""

    file_path = Path(path)
    ext = file_path.suffix.lower()
    base = _base_profile(file_path)
    if ext not in SUPPORTED_FILE_EXTENSIONS:
        return {
            **base,
            "parse_status": "unsupported",
            "warnings": ["unsupported_format"],
            "parser_error": f"Unsupported file extension: {ext or '<none>'}",
        }

    parser = parser or ParserRouter()
    pipeline = pipeline or ExtractionPipeline(mode="deterministic", enable_llm=False, audit_enabled=False)
    try:
        parsed = parser.parse_document_intelligence(
            str(file_path),
            doc_id=_stable_profile_doc_id(file_path),
            source_type="file",
        )
    except Exception as exc:
        return {
            **base,
            "parser_backend": "unknown",
            "parse_status": "failed",
            "warnings": ["parser_failed"],
            "parser_error": _safe_error(exc),
        }

    pages = _pages_estimated(parsed)
    quality = text_quality_metrics(parsed.text or "\n".join(chunk.text for chunk in parsed.chunks), pages_estimated=pages)
    raw_facts, data_gaps, rejected_count = _extract_rows(parsed.chunks, pipeline)
    canonical_facts = dedupe_fact_rows(raw_facts)
    conflicts = build_conflict_summary(canonical_facts)
    warnings = _profile_warnings(parsed, quality, canonical_facts, data_gaps)
    parse_status = _parse_status(parsed, quality, warnings)
    facts_without_evidence = sum(1 for row in canonical_facts if not row.get("evidence"))
    return {
        **base,
        "parser_backend": parsed.parser_name,
        "parser_backend_requested": parsed.diagnostics.get("parser_backend_requested"),
        "parse_status": parse_status,
        "text_chars": quality["text_chars"],
        "text_density": quality["text_density"],
        "pages_estimated": pages,
        "tables_detected": len(parsed.tables),
        "images_detected": len(parsed.images),
        "blocks_count": len(parsed.blocks),
        "chunks_count": len(parsed.chunks),
        "facts_extracted": len(raw_facts),
        "canonical_facts": len(canonical_facts),
        "facts_without_evidence": facts_without_evidence,
        "conflict_groups": len(conflicts),
        "data_gaps": len(data_gaps),
        "rejected_or_low_confidence_candidates": rejected_count,
        "warnings": warnings,
        "parser_diagnostics": _public_parser_diagnostics(parsed.diagnostics),
        "text_quality": quality,
        "canonical_fact_keys": [row.get("canonical_fact_key") for row in canonical_facts if row.get("canonical_fact_key")],
        "fact_preview": _fact_preview(canonical_facts),
        "data_gap_preview": _gap_preview(data_gaps),
    }


def profile_corpus(input_path: str | Path) -> dict[str, Any]:
    """Profile all files under a corpus path."""

    root = Path(input_path)
    files = _corpus_files(root)
    parser = ParserRouter()
    pipeline = ExtractionPipeline(mode="deterministic", enable_llm=False, audit_enabled=False)
    profiles = [profile_file(path, parser=parser, pipeline=pipeline) for path in files]
    all_keys = []
    for item in profiles:
        all_keys.extend(item.get("canonical_fact_keys") or [])
    summary = _corpus_summary(profiles, all_keys)
    return {
        "status": "ok",
        "input": str(root),
        "summary": summary,
        "files": profiles,
        "high_risk_documents": _high_risk_documents(profiles),
        "resource_profile": {
            "runtime_profile": "economy_core_compatible",
            "llm_required": False,
            "embeddings_required": False,
            "ocr_executed": False,
            "note": "OCR is detected as required but not executed by this profiler.",
        },
    }


def write_corpus_report(report: dict[str, Any], *, json_path: str | Path, markdown_path: str | Path | None = None) -> None:
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if markdown_path:
        md_path = Path(markdown_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    lines = [
        "# Corpus Readiness Report",
        "",
        "## Summary",
        f"- documents_total: {summary.get('documents_total', 0)}",
        f"- successfully_parsed: {summary.get('parse_status_counts', {}).get('ok', 0)}",
        f"- partial: {summary.get('parse_status_counts', {}).get('partial', 0)}",
        f"- failed: {summary.get('parse_status_counts', {}).get('failed', 0)}",
        f"- ocr_required: {summary.get('ocr_required_count', 0)}",
        f"- zero_fact_documents: {summary.get('zero_fact_documents_count', 0)}",
        f"- facts_without_evidence: {summary.get('facts_without_evidence', 0)}",
        "",
        "## File Type Coverage",
        "| extension | files |",
        "|---|---:|",
    ]
    for ext, count in sorted((summary.get("files_by_extension") or {}).items()):
        lines.append(f"| {ext or '<none>'} | {count} |")
    lines.extend(["", "## High-Risk Documents"])
    high_risk = report.get("high_risk_documents") or []
    if not high_risk:
        lines.append("- none")
    for item in high_risk:
        warnings = ", ".join(item.get("warnings") or [])
        lines.append(f"- {item.get('filename')}: status={item.get('parse_status')}; warnings={warnings}")
    lines.extend(
        [
            "",
            "## Extraction Coverage",
            f"- total_chunks: {summary.get('total_chunks', 0)}",
            f"- total_raw_facts: {summary.get('total_raw_facts', 0)}",
            f"- total_canonical_facts: {summary.get('total_canonical_facts', 0)}",
            f"- zero_fact_documents_count: {summary.get('zero_fact_documents_count', 0)}",
            "",
            "## Conflicts And Gaps",
            f"- conflict_groups: {summary.get('conflict_groups', 0)}",
            f"- data_gaps: {summary.get('data_gaps', 0)}",
            "",
            "## Resource Profile",
            "- economy_core: LLM disabled, embeddings disabled, deterministic extraction only.",
            "- OCR is not executed by this report. Image-only PDFs are marked as ocr_required.",
        ]
    )
    return "\n".join(lines) + "\n"


def _extract_rows(chunks: Iterable[Any], pipeline: ExtractionPipeline) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    raw_facts: list[dict[str, Any]] = []
    data_gaps: list[dict[str, Any]] = []
    rejected_count = 0
    for chunk in chunks:
        bundle = pipeline.extract_from_chunk(chunk)
        rejected_count += len(bundle.rejected_items)
        raw_facts.extend(fact_rows_from_experiments(bundle_to_experiment_facts(bundle)))
        for gap in bundle_to_data_gaps(bundle):
            data_gaps.append(
                {
                    "gap_id": gap.gap_id,
                    "material": gap.material,
                    "regime": gap.regime,
                    "property": gap.property,
                    "reason": gap.reason,
                    "evidence": [item.model_dump() for item in gap.evidence],
                }
            )
    return raw_facts, data_gaps, rejected_count


def _profile_warnings(parsed: Any, quality: dict[str, Any], canonical_facts: list[dict[str, Any]], data_gaps: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    diagnostics = parsed.diagnostics or {}
    ext = Path(str(getattr(parsed, "source_name", ""))).suffix.lower()
    if diagnostics.get("scanned_pdf_detected"):
        warnings.append("ocr_required")
    if quality.get("text_density") == "empty" or (ext == ".pdf" and quality.get("text_density") in {"very_low", "low"}):
        warnings.append("low_text_density")
    if quality.get("text_chars", 0) == 0:
        warnings.append("empty_text")
    if parsed.tables and (len(parsed.tables) >= 2 or len(parsed.tables) >= max(1, len(parsed.blocks) // 2)):
        warnings.append("table_heavy_document")
    if quality.get("dirty_ocr_score", 0) >= 2:
        warnings.append("dirty_ocr_text")
    if not canonical_facts and not data_gaps and parsed.chunks:
        warnings.append("zero_facts")
    if any("OCR" in str(item) or "scanned" in str(item).lower() for item in diagnostics.get("warnings") or []):
        warnings.append("ocr_required")
    return sorted(set(warnings))


def _parse_status(parsed: Any, quality: dict[str, Any], warnings: list[str]) -> str:
    if "ocr_required" in warnings and quality.get("text_density") in {"empty", "very_low", "low"}:
        return "ocr_required"
    if not parsed.chunks or quality.get("text_density") == "empty":
        return "partial"
    if any(item in warnings for item in ["low_text_density", "dirty_ocr_text"]):
        return "partial"
    return "ok"


def _corpus_summary(profiles: list[dict[str, Any]], all_keys: list[str]) -> dict[str, Any]:
    status_counts = Counter(item.get("parse_status") for item in profiles)
    extension_counts = Counter(item.get("extension") for item in profiles)
    warning_counts = Counter(warning for item in profiles for warning in item.get("warnings") or [])
    canonical_unique = len({key for key in all_keys if key})
    return {
        "documents_total": len(profiles),
        "files_by_extension": dict(sorted(extension_counts.items())),
        "parse_status_counts": dict(status_counts),
        "parser_failures_count": status_counts.get("failed", 0),
        "ocr_required_count": warning_counts.get("ocr_required", 0),
        "zero_fact_documents_count": warning_counts.get("zero_facts", 0),
        "empty_text_documents_count": warning_counts.get("empty_text", 0),
        "table_heavy_documents_count": warning_counts.get("table_heavy_document", 0),
        "dirty_ocr_documents_count": warning_counts.get("dirty_ocr_text", 0),
        "total_chunks": sum(int(item.get("chunks_count") or 0) for item in profiles),
        "total_raw_facts": sum(int(item.get("facts_extracted") or 0) for item in profiles),
        "total_canonical_facts": canonical_unique,
        "per_file_canonical_facts_sum": sum(int(item.get("canonical_facts") or 0) for item in profiles),
        "facts_without_evidence": sum(int(item.get("facts_without_evidence") or 0) for item in profiles),
        "conflict_groups": sum(int(item.get("conflict_groups") or 0) for item in profiles),
        "data_gaps": sum(int(item.get("data_gaps") or 0) for item in profiles),
        "economy_core_compatible": True,
        "warning_counts": dict(warning_counts),
    }


def _high_risk_documents(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    risk_warnings = {"parser_failed", "ocr_required", "zero_facts", "empty_text", "dirty_ocr_text", "low_text_density"}
    result = []
    for item in profiles:
        warnings = set(item.get("warnings") or [])
        if item.get("parse_status") in {"failed", "ocr_required", "unsupported"} or warnings.intersection(risk_warnings):
            result.append(
                {
                    "path": item.get("path"),
                    "filename": item.get("filename"),
                    "extension": item.get("extension"),
                    "parse_status": item.get("parse_status"),
                    "warnings": item.get("warnings") or [],
                    "text_chars": item.get("text_chars", 0),
                    "chunks_count": item.get("chunks_count", 0),
                    "canonical_facts": item.get("canonical_facts", 0),
                }
            )
    return result


def _base_profile(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "filename": path.name,
        "extension": path.suffix.lower(),
        "source_type": "file",
        "parser_backend": None,
        "parse_status": "unknown",
        "text_chars": 0,
        "text_density": "empty",
        "pages_estimated": 0,
        "tables_detected": 0,
        "images_detected": 0,
        "chunks_count": 0,
        "facts_extracted": 0,
        "canonical_facts": 0,
        "facts_without_evidence": 0,
        "conflict_groups": 0,
        "data_gaps": 0,
        "warnings": [],
    }


def _pages_estimated(parsed: Any) -> int:
    diagnostics = parsed.diagnostics or {}
    if diagnostics.get("scanned_pdf_page_count"):
        return max(1, int(diagnostics["scanned_pdf_page_count"]))
    pages = [int(chunk.page_end or chunk.page_start or 1) for chunk in parsed.chunks or [] if chunk.page_end or chunk.page_start]
    return max(pages or [1])


def _public_parser_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in (diagnostics or {}).items()
        if "password" not in key.lower() and "key" not in key.lower() and "authorization" not in key.lower()
    }


def _fact_preview(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "material": row.get("material"),
            "regime": row.get("regime"),
            "property": row.get("property"),
            "value_normalized": row.get("value_normalized"),
            "unit_normalized": row.get("unit_normalized"),
            "evidence_count": len(row.get("evidence") or []),
        }
        for row in rows[:limit]
    ]


def _gap_preview(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "material": row.get("material"),
            "regime": row.get("regime"),
            "property": row.get("property"),
            "reason": row.get("reason"),
            "evidence_count": len(row.get("evidence") or []),
        }
        for row in rows[:limit]
    ]


def _stable_profile_doc_id(path: Path) -> str:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    except Exception:
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]
    return f"profile_doc_{digest}"


def _corpus_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_file() and not path.name.startswith("."))


def _safe_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text[:500]
