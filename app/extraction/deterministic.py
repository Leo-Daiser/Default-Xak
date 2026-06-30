"""Deterministic extraction adapter producing typed ExtractionBundle objects."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from ..models.schemas import Chunk
from .confidence import experiment_confidence
from .extraction import EntityRelationExtractor
from .models import (
    EvidenceSpan,
    ExtractedDataGap,
    ExtractedEntity,
    ExtractedExperiment,
    ExtractedMeasurement,
    ExtractedRegime,
    ExtractionBundle,
    ExtractionSource,
)
from .resolver import resolve_material, resolve_property, resolve_regime, resolve_unit


class DeterministicExtractor:
    """Wrap the existing rule-based extractor in the typed extraction contract."""

    extractor_version = "deterministic_v2"

    def __init__(self, legacy_extractor: EntityRelationExtractor | None = None) -> None:
        self.legacy_extractor = legacy_extractor or EntityRelationExtractor()

    def extract_from_chunk(self, chunk: Chunk) -> ExtractionBundle:
        chunks = self._split_experiment_segments(chunk)
        bundle = ExtractionBundle(
            document_id=chunk.doc_id,
            source_name=str(chunk.metadata.get("filename") or chunk.doc_id),
            extractor_version=self.extractor_version,
            diagnostics={"segments": len(chunks), "ambiguous_multiple_experiment_ids": len(chunks) > 1},
        )
        for segment in chunks:
            partial = self._extract_single_chunk(segment)
            bundle.entities.extend(partial.entities)
            bundle.experiments.extend(partial.experiments)
            bundle.data_gaps.extend(partial.data_gaps)
            bundle.rejected_items.extend(partial.rejected_items)
        bundle.entities = _dedupe_entities(bundle.entities)
        bundle.experiments = _dedupe_experiments(bundle.experiments)
        bundle.data_gaps = _dedupe_gaps(bundle.data_gaps)
        return bundle

    def _extract_single_chunk(self, chunk: Chunk) -> ExtractionBundle:
        extraction = self.legacy_extractor.extract_from_chunk(chunk)
        source = _source_from_chunk(chunk)
        default_evidence = EvidenceSpan(source=source, quote=(chunk.text or "")[:700], confidence=0.8)
        prop_by_value: dict[str, str] = {}
        change_by_value: dict[str, str] = {}
        for rel in extraction.relations:
            if rel.predicate == "OF_PROPERTY":
                prop_by_value[rel.subject] = resolve_property(rel.object)
            elif rel.predicate == "HAS_CHANGE":
                change_by_value[rel.subject] = _normalize_effect(str((rel.qualifiers or {}).get("direction") or rel.object))

        entities: list[ExtractedEntity] = []
        for entity in extraction.entities:
            converted = _entity_from_legacy(entity, default_evidence)
            if converted is not None:
                entities.append(converted)

        experiment_map: dict[str, dict[str, Any]] = {}
        data_gaps: list[ExtractedDataGap] = []
        for rel in extraction.relations:
            evidence = _evidence_from_relation(rel, default_evidence)
            if rel.predicate == "MISSING_FOR":
                data_gaps.append(_gap_from_text(rel.subject, rel.object, evidence))
                continue
            if rel.predicate not in {"STUDIES", "USES_REGIME", "MEASURES", "USES_EQUIPMENT", "PERFORMED_BY"}:
                continue
            exp = experiment_map.setdefault(
                rel.subject,
                {"materials": [], "regimes": [], "measurements": [], "equipment": [], "laboratories": [], "evidence": [], "conclusions": []},
            )
            exp["evidence"].append(evidence)
            if rel.predicate == "STUDIES":
                if _is_experiment_identifier(rel.object):
                    continue
                exp["materials"].append(_entity("Material", rel.object, resolve_material(rel.object), evidence, 0.82))
            elif rel.predicate == "USES_REGIME":
                exp["regimes"].append(ExtractedRegime(raw_name=rel.object, canonical_name=resolve_regime(rel.object), confidence=0.78, evidence=[evidence]))
            elif rel.predicate == "USES_EQUIPMENT":
                exp["equipment"].append(_entity("Equipment", rel.object, rel.object, evidence, 0.72))
            elif rel.predicate == "PERFORMED_BY":
                exp["laboratories"].append(_entity("Laboratory", rel.object, rel.object, evidence, 0.72))
            elif rel.predicate == "MEASURES":
                qualifiers = rel.qualifiers or {}
                property_name = prop_by_value.get(rel.object) or resolve_property(rel.object)
                if not property_name:
                    continue
                exp["measurements"].append(
                    ExtractedMeasurement(
                        property_raw=rel.object,
                        property_canonical=property_name,
                        value=_float_or_none(qualifiers.get("value")),
                        unit=qualifiers.get("unit"),
                        effect=_normalize_effect(change_by_value.get(rel.object) or qualifiers.get("direction")),
                        confidence=rel.confidence,
                        evidence=[evidence],
                    )
                )

        conclusion_entities = [entity.canonical_name for entity in extraction.entities if entity.entity_type == "Conclusion"]
        experiments: list[ExtractedExperiment] = []
        for experiment_id, data in experiment_map.items():
            experiment = ExtractedExperiment(
                experiment_id=experiment_id,
                materials=_dedupe_entities(data["materials"]),
                regimes=_dedupe_regimes(data["regimes"]),
                measurements=_dedupe_measurements(data["measurements"]),
                equipment=_dedupe_entities(data["equipment"]),
                laboratories=_dedupe_entities(data["laboratories"]),
                conclusions=list(dict.fromkeys(conclusion_entities)),
                evidence=_dedupe_evidence(data["evidence"] or [default_evidence]),
                confidence=0.0,
            )
            confidence = experiment_confidence(experiment, ambiguous=False)
            experiment = experiment.model_copy(update={"confidence": confidence}) if hasattr(experiment, "model_copy") else experiment.copy(update={"confidence": confidence})
            experiments.append(experiment)

        for entity in extraction.entities:
            if entity.entity_type == "DataGap":
                data_gaps.append(_gap_from_text(entity.canonical_name, None, default_evidence))

        pattern_bundle = _extract_direct_text_patterns(chunk, default_evidence)
        entities.extend(pattern_bundle.entities)
        experiments.extend(pattern_bundle.experiments)
        data_gaps.extend(pattern_bundle.data_gaps)

        return ExtractionBundle(
            document_id=chunk.doc_id,
            source_name=str(chunk.metadata.get("filename") or chunk.doc_id),
            extractor_version=self.extractor_version,
            entities=_dedupe_entities(entities),
            experiments=_dedupe_experiments(experiments),
            data_gaps=_dedupe_gaps(data_gaps),
            diagnostics={"chunk_id": chunk.chunk_id},
        )

    @staticmethod
    def _split_experiment_segments(chunk: Chunk) -> list[Chunk]:
        text = chunk.text or ""
        marker_pattern = (
            r"(?:эксперимент|experiment)\s+[A-ZА-Я0-9_.-]+\s*:"
            r"|(?:experiment_id|experiment\s+id|id\s+эксперимента)\s*[:=]"
        )
        markers = re.findall(marker_pattern, text, flags=re.IGNORECASE)
        if len(markers) <= 1:
            return [chunk]
        segments = [
            part.strip(" .;\n\t")
            for part in re.split(rf"(?={marker_pattern})", text, flags=re.IGNORECASE)
            if part.strip(" .;\n\t")
        ]
        if len(segments) <= 1:
            segments = [line.strip() for line in text.splitlines() if line.strip()]
        result: list[Chunk] = []
        for idx, segment in enumerate(segments):
            if not segment:
                continue
            update = {
                "chunk_id": f"{chunk.chunk_id}:seg{idx}",
                "text": segment,
                "ordinal": (chunk.ordinal or 0) * 1000 + idx,
                "metadata": {**(chunk.metadata or {}), "parent_chunk_id": chunk.chunk_id, "segment_id": idx},
            }
            result.append(chunk.model_copy(update=update) if hasattr(chunk, "model_copy") else chunk.copy(update=update))
        return result or [chunk]


_EXPERIMENT_ID_RE = re.compile(r"\b(?:E\d+|EXP-[A-ZА-Я0-9_.-]+)\b", re.IGNORECASE)
_PERSON_RE = re.compile(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*[А-ЯЁ]\.")
_LAB_RE = re.compile(r"\b(?:в\s+)?(лаборатори[ия]\s+[^;,.]+)", re.IGNORECASE)
_MATERIAL_CANDIDATES = ["ВТ6", "VT6", "Ti-6Al-4V", "7075-T6", "7075", "12Х18Н10Т", "09Г2С"]
_REGIME_CANDIDATES = ["отжиг", "старение", "закалка", "криообработка", "annealing", "aging", "quenching"]


def _extract_direct_text_patterns(chunk: Chunk, evidence: EvidenceSpan) -> ExtractionBundle:
    """Extract common Russian materials-science phrasings missed by legacy rules."""
    text = chunk.text or ""
    sanitized = _EXPERIMENT_ID_RE.sub(" ", text)
    materials = _extract_materials(sanitized, evidence)
    regimes = _extract_regimes(sanitized, evidence)
    measurements = _extract_measurements(text, evidence)
    laboratories = [_entity("Laboratory", match.group(1), match.group(1).strip(), evidence, 0.74) for match in _LAB_RE.finditer(text)]
    employees = [_entity("Employee", match.group(0), match.group(0).strip(), evidence, 0.72) for match in _PERSON_RE.finditer(text)]
    gaps = _extract_gap_patterns(text, evidence)
    entities: list[ExtractedEntity] = [
        *materials,
        *[_entity("ProcessRegime", regime.raw_name, regime.canonical_name, evidence, regime.confidence) for regime in regimes],
        *[_entity("Property", measurement.property_raw, measurement.property_canonical, evidence, measurement.confidence) for measurement in measurements],
        *laboratories,
        *employees,
    ]
    if not materials or not (regimes or measurements):
        return ExtractionBundle(
            document_id=evidence.source.document_id,
            source_name=evidence.source.source_name,
            extractor_version=DeterministicPatternVersion.VALUE,
            entities=_dedupe_entities(entities),
            data_gaps=gaps,
            diagnostics={"direct_patterns": True, "experiment_created": False},
        )

    experiment_id = _extract_experiment_id(text, chunk)
    experiment = ExtractedExperiment(
        experiment_id=experiment_id,
        materials=_dedupe_entities(materials),
        regimes=_dedupe_regimes(regimes),
        measurements=_dedupe_measurements(measurements),
        laboratories=_dedupe_entities(laboratories),
        employees=_dedupe_entities(employees),
        evidence=[evidence],
        confidence=0.86,
    )
    return ExtractionBundle(
        document_id=evidence.source.document_id,
        source_name=evidence.source.source_name,
        extractor_version=DeterministicPatternVersion.VALUE,
        entities=_dedupe_entities(entities),
        experiments=[experiment],
        data_gaps=gaps,
        diagnostics={"direct_patterns": True, "experiment_created": True},
    )


class DeterministicPatternVersion:
    VALUE = "deterministic_text_patterns_v1"


def _extract_materials(text: str, evidence: EvidenceSpan) -> list[ExtractedEntity]:
    result: list[ExtractedEntity] = []
    lowered = text.lower()
    for raw in _MATERIAL_CANDIDATES:
        if raw.lower() not in lowered:
            continue
        canonical = resolve_material(raw)
        if canonical and not _is_experiment_identifier(raw):
            result.append(_entity("Material", raw, canonical, evidence, 0.84))
    return _dedupe_entities(result)


def _extract_regimes(text: str, evidence: EvidenceSpan) -> list[ExtractedRegime]:
    result: list[ExtractedRegime] = []
    lowered = text.lower().replace("ё", "е")
    temp_match = re.search(r"(\d+(?:[,.]\d+)?)\s*(?:°\s*)?[CСс]", text)
    duration_match = re.search(r"(\d+(?:[,.]\d+)?)\s*(?:ч|h|час)", text, flags=re.IGNORECASE)
    for raw in _REGIME_CANDIDATES:
        if raw.lower().replace("ё", "е") not in lowered:
            continue
        result.append(
            ExtractedRegime(
                raw_name=raw,
                canonical_name=resolve_regime(raw),
                temperature=_float_or_none(temp_match.group(1)) if temp_match else None,
                temperature_unit="C" if temp_match else None,
                duration=_float_or_none(duration_match.group(1)) if duration_match else None,
                duration_unit="h" if duration_match else None,
                confidence=0.82,
                evidence=[evidence],
            )
        )
    return _dedupe_regimes(result)


def _extract_measurements(text: str, evidence: EvidenceSpan) -> list[ExtractedMeasurement]:
    measurements: list[ExtractedMeasurement] = []
    strength_pattern = re.compile(
        r"(?:предел\s+прочности\s*(?:σв|sigma_b)?|прочност[ьи])"
        r"[^.;,\n]{0,80}?(увелич\w+\s+до\s+|составил\w*\s+|=|:)?"
        r"(\d+(?:[,.]\d+)?)\s*(МПа|MPa|мПа|ГПа|GPa)",
        re.IGNORECASE,
    )
    for match in strength_pattern.finditer(text):
        window = match.group(0).lower()
        measurements.append(
            ExtractedMeasurement(
                property_raw="предел прочности",
                property_canonical=resolve_property("прочность"),
                value=_float_or_none(match.group(2)),
                unit=resolve_unit(match.group(3)),
                effect="increase" if "увелич" in window else "unknown",
                confidence=0.88,
                evidence=[evidence],
            )
        )

    ductility_pattern = re.compile(r"(?:относительное\s+)?удлинение[^.;,\n]{0,40}?(\d+(?:[,.]\d+)?)\s*%", re.IGNORECASE)
    for match in ductility_pattern.finditer(text):
        measurements.append(
            ExtractedMeasurement(
                property_raw="удлинение",
                property_canonical=resolve_property("пластичность"),
                value=_float_or_none(match.group(1)),
                unit="%",
                confidence=0.86,
                evidence=[evidence],
            )
        )

    hardness_patterns = [
        re.compile(r"(?:тв[её]рдость[^.;,\n]{0,25}?)?(HV|HRC)\s*(\d+(?:[,.]\d+)?)", re.IGNORECASE),
        re.compile(r"тв[её]рдость[^.;,\n]{0,30}?(\d+(?:[,.]\d+)?)\s*(HV|HRC)", re.IGNORECASE),
    ]
    for pattern in hardness_patterns:
        for match in pattern.finditer(text):
            unit, value = (match.group(1), match.group(2)) if match.group(1).upper() in {"HV", "HRC"} else (match.group(2), match.group(1))
            measurements.append(
                ExtractedMeasurement(
                    property_raw="твёрдость",
                    property_canonical=resolve_property("твёрдость"),
                    value=_float_or_none(value),
                    unit=resolve_unit(unit),
                    confidence=0.86,
                    evidence=[evidence],
                )
            )
    return _dedupe_measurements(measurements)


def _extract_gap_patterns(text: str, evidence: EvidenceSpan) -> list[ExtractedDataGap]:
    gaps: list[ExtractedDataGap] = []
    if re.search(r"коррозионн\w+\s+стойк\w+[^.;\n]{0,40}не\s+измер", text, flags=re.IGNORECASE):
        gaps.append(_gap_from_text("коррозионная стойкость не измерялась", "коррозионная стойкость", evidence))
    return gaps


def _extract_experiment_id(text: str, chunk: Chunk) -> str:
    match = _EXPERIMENT_ID_RE.search(text)
    if match:
        return match.group(0)
    raw = f"{chunk.chunk_id}|{text[:120]}"
    return f"EXP-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def _is_experiment_identifier(value: str | None) -> bool:
    return bool(_EXPERIMENT_ID_RE.fullmatch(str(value or "").strip()))


def _source_from_chunk(chunk: Chunk, column_name: str | None = None) -> ExtractionSource:
    row_id = chunk.metadata.get("row_id")
    try:
        row_index = int(row_id) if row_id is not None else None
    except ValueError:
        row_index = None
    return ExtractionSource(
        document_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        source_name=str(chunk.metadata.get("filename") or chunk.doc_id),
        page=chunk.page_start,
        section_path=chunk.section_path,
        block_type=str(chunk.metadata.get("chunk_kind") or "text"),
        row_index=row_index,
        column_name=column_name,
    )


def _entity(entity_type: str, raw: str, canonical: str, evidence: EvidenceSpan, confidence: float) -> ExtractedEntity:
    return ExtractedEntity(entity_type=entity_type, raw_name=raw, canonical_name=canonical, confidence=confidence, evidence=[evidence])


def _entity_from_legacy(entity, evidence: EvidenceSpan) -> ExtractedEntity | None:
    if entity.entity_type == "Material":
        if _is_experiment_identifier(entity.canonical_name):
            return None
        return _entity("Material", entity.canonical_name, resolve_material(entity.canonical_name), evidence, 0.8)
    if entity.entity_type in {"ProcessRegime", "ProcessCondition"}:
        return _entity("ProcessRegime", entity.canonical_name, resolve_regime(entity.canonical_name), evidence, 0.72)
    if entity.entity_type == "Property":
        return _entity("Property", entity.canonical_name, resolve_property(entity.canonical_name), evidence, 0.72)
    if entity.entity_type in {"Equipment", "Laboratory", "ResearchTeam", "Employee", "TopicTag"}:
        return _entity(entity.entity_type, entity.canonical_name, entity.canonical_name, evidence, 0.7)
    return None


def _evidence_from_relation(rel, default: EvidenceSpan) -> EvidenceSpan:
    if not rel.evidence:
        return default
    src = rel.evidence[0]
    quote = src.quote or default.quote
    source = ExtractionSource(
        document_id=src.doc_id,
        chunk_id=src.chunk_id,
        source_name=default.source.source_name,
        page=src.page_start,
        section_path=default.source.section_path,
        block_type=default.source.block_type,
        row_index=default.source.row_index,
    )
    return EvidenceSpan(source=source, quote=quote, confidence=rel.confidence or default.confidence)


def _gap_from_text(text: str, missing_for: str | None, evidence: EvidenceSpan) -> ExtractedDataGap:
    joined = f"{text} {missing_for or ''} {evidence.quote or ''}"
    material = _first_canonical(joined, resolve_material, ["ВТ6", "7075-T6", "12Х18Н10Т", "09Г2С"])
    regime = _first_canonical(joined, resolve_regime, ["отжиг", "старение", "закалка", "криообработка"])
    property_name = None
    if re.search(r"коррозионн\w+\s+стойк", str(text or ""), flags=re.IGNORECASE):
        property_name = "коррозионная стойкость"
    property_name = property_name or _first_canonical(missing_for or "", resolve_property, ["коррозионная стойкость", "прочность", "твёрдость", "пластичность", "вязкость"])
    property_name = property_name or _first_canonical(joined, resolve_property, ["коррозионная стойкость", "прочность", "твёрдость", "пластичность", "вязкость"])
    reason = re.sub(r"\s+", " ", str(text or "").strip(" .;|"))
    raw = "|".join([material or "", regime or "", property_name or "", reason])
    return ExtractedDataGap(
        gap_id=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
        material=material,
        regime=regime,
        property=property_name,
        reason=reason,
        confidence=0.75,
        evidence=[evidence],
    )


def _first_canonical(text: str, resolver, candidates: list[str]) -> str | None:
    text_norm = str(text or "").lower().replace("ё", "е")
    for candidate in candidates:
        canonical = resolver(candidate)
        canonical_norm = str(canonical or "").lower().replace("ё", "е")
        if canonical and canonical_norm in text_norm:
            return canonical
        if resolver(text) == canonical:
            return canonical
    return None


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _normalize_effect(value: str | None) -> str:
    norm = str(value or "").lower()
    if norm in {"increase", "increased"} or "повыс" in norm or "увелич" in norm:
        return "increase"
    if norm in {"decrease", "decreased"} or "сниз" in norm or "уменьш" in norm:
        return "decrease"
    if norm in {"unchanged", "no_change"} or "без измен" in norm:
        return "no_change"
    return "unknown"


def _dedupe_entities(items: list[ExtractedEntity]) -> list[ExtractedEntity]:
    seen = set()
    result = []
    for item in items:
        key = (item.entity_type, item.canonical_name)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_regimes(items: list[ExtractedRegime]) -> list[ExtractedRegime]:
    seen = set()
    result = []
    for item in items:
        key = item.canonical_name
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_measurements(items: list[ExtractedMeasurement]) -> list[ExtractedMeasurement]:
    seen = set()
    result = []
    for item in items:
        key = (item.property_canonical, item.value, item.unit, item.effect)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_evidence(items: list[EvidenceSpan]) -> list[EvidenceSpan]:
    seen = set()
    result = []
    for item in items:
        key = (item.source.document_id, item.source.chunk_id, item.quote)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _dedupe_experiments(items: list[ExtractedExperiment]) -> list[ExtractedExperiment]:
    by_id: dict[str, ExtractedExperiment] = {}
    for item in items:
        existing = by_id.get(item.experiment_id)
        if existing is None:
            by_id[item.experiment_id] = item
            continue
        updates = {
            "materials": _dedupe_entities([*existing.materials, *item.materials]),
            "regimes": _dedupe_regimes([*existing.regimes, *item.regimes]),
            "measurements": _dedupe_measurements([*existing.measurements, *item.measurements]),
            "equipment": _dedupe_entities([*existing.equipment, *item.equipment]),
            "laboratories": _dedupe_entities([*existing.laboratories, *item.laboratories]),
            "teams": _dedupe_entities([*existing.teams, *item.teams]),
            "employees": _dedupe_entities([*existing.employees, *item.employees]),
            "conclusions": list(dict.fromkeys([*existing.conclusions, *item.conclusions])),
            "evidence": _dedupe_evidence([*existing.evidence, *item.evidence]),
            "confidence": max(existing.confidence, item.confidence),
        }
        by_id[item.experiment_id] = existing.model_copy(update=updates) if hasattr(existing, "model_copy") else existing.copy(update=updates)
    return list(by_id.values())


def _dedupe_gaps(items: list[ExtractedDataGap]) -> list[ExtractedDataGap]:
    seen = set()
    result = []
    for item in items:
        key = (item.material, item.regime, item.property, item.reason)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
