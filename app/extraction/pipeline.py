"""Evidence-grounded structured extraction pipeline."""

from __future__ import annotations

from typing import Any

from ..config import settings
from ..models.schemas import Chunk, Document
from .audit import ExtractionAuditWriter
from .deterministic import DeterministicExtractor, _source_from_chunk
from .llm_structured import StructuredLLMExtractor
from .models import ExtractionBundle, RejectedExtraction
from .table_extractor import TableExtractor
from .validators import validate_items


class ExtractionPipeline:
    """Run deterministic/table/optional LLM extraction, validation and audit."""

    extractor_version = "pipeline_v1"

    def __init__(
        self,
        mode: str | None = None,
        min_confidence: float | None = None,
        enable_llm: bool | None = None,
        audit_dir: str | None = None,
        audit_enabled: bool = True,
        deterministic_extractor: DeterministicExtractor | None = None,
        table_extractor: TableExtractor | None = None,
        llm_extractor: StructuredLLMExtractor | None = None,
    ) -> None:
        self.mode = (mode or getattr(settings, "extraction_mode", "deterministic") or "deterministic").lower()
        self.min_confidence = float(min_confidence if min_confidence is not None else getattr(settings, "extraction_min_confidence", 0.55))
        self.enable_llm = bool(enable_llm if enable_llm is not None else getattr(settings, "extraction_enable_llm", False))
        self.deterministic_extractor = deterministic_extractor or DeterministicExtractor()
        self.table_extractor = table_extractor or TableExtractor()
        self.llm_extractor = llm_extractor or StructuredLLMExtractor()
        self.audit_writer = ExtractionAuditWriter(audit_dir or getattr(settings, "extraction_audit_dir", "data/extraction_audit")) if audit_enabled else None

    def extract_from_chunk(self, chunk: Chunk) -> ExtractionBundle:
        """Extract accepted/rejected facts from one chunk."""
        if self.mode not in {"deterministic", "hybrid", "llm"}:
            raise ValueError(f"Unsupported EXTRACTION_MODE={self.mode!r}")
        candidates: list[ExtractionBundle] = []
        diagnostics: dict[str, Any] = {"mode": self.mode, "extractor_version": self.extractor_version}

        if self.mode in {"deterministic", "hybrid"}:
            if chunk.metadata.get("chunk_kind") == "table_row":
                candidates.append(self.table_extractor.extract_from_chunk(chunk))
            else:
                candidates.append(self.deterministic_extractor.extract_from_chunk(chunk))

        if self.mode in {"hybrid", "llm"}:
            if not self.enable_llm or not self.llm_extractor.available:
                warning = "llm_extractor_unavailable"
                diagnostics.setdefault("warnings", []).append(warning)
                if self.mode == "llm":
                    raise RuntimeError("EXTRACTION_MODE=llm requested, but LLM extractor is unavailable")
            else:
                candidates.append(self.llm_extractor.extract(chunk.text or "", _source_from_chunk(chunk)))

        bundle = self._merge_and_validate(candidates, chunk, diagnostics)
        if self.audit_writer:
            self.audit_writer.write_bundle(bundle)
        return bundle

    def extract_from_table(self, table: Any) -> ExtractionBundle:
        """Extract from a table-like row chunk."""
        return self.extract_from_chunk(table)

    def extract_from_document(self, document: Document, chunks: list[Chunk] | None = None) -> ExtractionBundle:
        """Extract from all chunks of a document and return one merged bundle."""
        candidates = [self.extract_from_chunk(chunk) for chunk in (chunks or [])]
        return self._merge_raw(candidates, document.doc_id, document.title, {"document_id": document.doc_id})

    def _merge_and_validate(self, candidates: list[ExtractionBundle], chunk: Chunk, diagnostics: dict[str, Any]) -> ExtractionBundle:
        merged = self._merge_raw(candidates, chunk.doc_id, str(chunk.metadata.get("filename") or chunk.doc_id), diagnostics)
        validation = validate_items(merged.entities, merged.experiments, merged.data_gaps, min_confidence=self.min_confidence)
        accepted = ExtractionBundle(
            document_id=merged.document_id,
            source_name=merged.source_name,
            extractor_version=self.extractor_version,
            entities=validation.entities,
            experiments=validation.experiments,
            data_gaps=validation.data_gaps,
            rejected_items=[*merged.rejected_items, *validation.rejected],
            diagnostics={
                **merged.diagnostics,
                "accepted_entities": len(validation.entities),
                "accepted_experiments": len(validation.experiments),
                "accepted_gaps": len(validation.data_gaps),
                "rejected_items": len(merged.rejected_items) + len(validation.rejected),
                "min_confidence": self.min_confidence,
            },
        )
        return accepted

    def _merge_raw(self, candidates: list[ExtractionBundle], document_id: str | None, source_name: str | None, diagnostics: dict[str, Any]) -> ExtractionBundle:
        entities = []
        experiments = []
        data_gaps = []
        rejected: list[RejectedExtraction] = []
        for candidate in candidates:
            entities.extend(candidate.entities)
            experiments.extend(candidate.experiments)
            data_gaps.extend(candidate.data_gaps)
            rejected.extend(candidate.rejected_items)
            diagnostics.setdefault("candidate_diagnostics", []).append(candidate.diagnostics)
        return ExtractionBundle(
            document_id=document_id,
            source_name=source_name,
            extractor_version=self.extractor_version,
            entities=_dedupe_entities(entities),
            experiments=_dedupe_experiments(experiments),
            data_gaps=_dedupe_gaps(data_gaps),
            rejected_items=rejected,
            diagnostics=diagnostics,
        )


def _dedupe_entities(items):
    seen = set()
    result = []
    for item in items:
        key = (item.entity_type, item.canonical_name)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_experiments(items):
    seen = set()
    result = []
    for item in items:
        key = item.experiment_id
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_gaps(items):
    seen = set()
    result = []
    for item in items:
        key = item.gap_id
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result

