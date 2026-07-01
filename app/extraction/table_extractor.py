"""Column-aware extractor for CSV/XLSX/catalog table row chunks."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

from ..models.schemas import Chunk
from .confidence import experiment_confidence
from .deterministic import _source_from_chunk
from .models import (
    EvidenceSpan,
    ExtractedDataGap,
    ExtractedEntity,
    ExtractedExperiment,
    ExtractedMeasurement,
    ExtractedRegime,
    ExtractionBundle,
    RejectedExtraction,
)
from .resolver import clean_raw, resolve_material, resolve_property, resolve_regime, resolve_unit


MATERIAL_COLUMNS = ["material", "материал", "сплав", "alloy", "grade", "марка"]
REGIME_COLUMNS = ["regime", "режим", "process", "обработка", "термообработка", "process_regime"]
PROPERTY_COLUMNS = ["property", "свойство", "показатель", "metric"]
VALUE_COLUMNS = ["value", "значение", "result", "результат"]
UNIT_COLUMNS = ["unit", "единица", "ед. изм.", "единицы"]
EQUIPMENT_COLUMNS = ["equipment", "оборудование", "установка", "stand", "device"]
LAB_COLUMNS = ["laboratory", "лаборатория", "lab"]
TEAM_COLUMNS = ["team", "команда", "group", "группа"]
EMPLOYEE_COLUMNS = ["employee", "сотрудник", "researcher", "исполнитель", "author"]
CONCLUSION_COLUMNS = ["conclusion", "вывод", "effect", "эффект"]
TOPIC_TAG_COLUMNS = ["tag", "topic", "тематика", "тег"]
EXPERIMENT_COLUMNS = ["experiment_id", "experiment", "id эксперимента", "опыт", "exp_id"]
GAP_COLUMNS = ["data_gap", "gap", "пробел", "нет данных"]


class TableExtractor:
    """Extract experiments from serialized table row chunks."""

    extractor_version = "table_v1"

    def extract_from_chunk(self, chunk: Chunk) -> ExtractionBundle:
        row = parse_serialized_row(chunk.text or "")
        source_name = str(chunk.metadata.get("source_name") or chunk.metadata.get("filename") or chunk.doc_id)
        bundle = ExtractionBundle(document_id=chunk.doc_id, source_name=source_name, extractor_version=self.extractor_version)
        quote = chunk.text.strip()
        if not row:
            bundle.rejected_items.append(RejectedExtraction(item_type="table_row", reason="empty_row", raw_payload=chunk.text or ""))
            return bundle
        source = _source_from_chunk(chunk)
        evidence = EvidenceSpan(source=source, quote=quote, confidence=0.95)
        material_values = _multi_values(_pick(row, MATERIAL_COLUMNS))
        if not material_values and not any(clean_raw(value) for value in row.values()):
            bundle.rejected_items.append(RejectedExtraction(item_type="table_row", reason="empty_row", raw_payload=row, evidence=[evidence]))
            return bundle
        if not material_values:
            bundle.rejected_items.append(RejectedExtraction(item_type="table_row", reason="missing_material", raw_payload=row, evidence=[evidence]))
            return bundle

        regime_raw = _pick(row, REGIME_COLUMNS)
        property_raw = _pick(row, PROPERTY_COLUMNS)
        value_raw = _pick(row, VALUE_COLUMNS)
        unit = resolve_unit(_pick(row, UNIT_COLUMNS))
        effect_raw = _pick(row, CONCLUSION_COLUMNS)
        gap_raw = _pick(row, GAP_COLUMNS)

        materials = _dedupe_entities([
            ExtractedEntity(entity_type="Material", raw_name=value, canonical_name=resolve_material(value), confidence=0.9, evidence=[evidence])
            for value in material_values
            if clean_raw(value)
        ])
        regimes: list[ExtractedRegime] = []
        if regime_raw:
            regimes.append(ExtractedRegime(raw_name=regime_raw, canonical_name=resolve_regime(regime_raw), confidence=0.86, evidence=[evidence]))

        measurements: list[ExtractedMeasurement] = []
        if property_raw:
            measurements.append(
                ExtractedMeasurement(
                    property_raw=property_raw,
                    property_canonical=resolve_property(property_raw),
                    value=_float_or_none(value_raw),
                    unit=unit,
                    effect=_effect_from_text(effect_raw or value_raw),
                    confidence=0.86 if value_raw or effect_raw else 0.5,
                    evidence=[evidence],
                )
            )

        equipment = _dedupe_entities([_entity("Equipment", item, evidence) for item in _multi_values(_pick(row, EQUIPMENT_COLUMNS))])
        laboratories = _dedupe_entities([_entity("Laboratory", item, evidence) for item in _multi_values(_pick(row, LAB_COLUMNS))])
        teams = _dedupe_entities([_entity("ResearchTeam", item, evidence) for item in _multi_values(_pick(row, TEAM_COLUMNS))])
        employees = _dedupe_entities([_entity("Employee", item, evidence) for item in _multi_values(_pick(row, EMPLOYEE_COLUMNS))])
        topic_tags = _dedupe_entities([_entity("TopicTag", item, evidence) for item in _multi_values(_pick(row, TOPIC_TAG_COLUMNS))])
        conclusions = [text for text in [effect_raw] if text]

        if regimes or measurements:
            experiment_id = clean_raw(_pick(row, EXPERIMENT_COLUMNS)) or _stable_experiment_id(source_name, source.row_index, materials, regimes, measurements)
            experiment = ExtractedExperiment(
                experiment_id=experiment_id,
                materials=materials,
                regimes=regimes,
                measurements=measurements,
                equipment=equipment,
                laboratories=laboratories,
                teams=teams,
                employees=employees,
                conclusions=conclusions,
                topic_tags=topic_tags,
                evidence=[evidence],
                confidence=0.0,
            )
            confidence = experiment_confidence(experiment)
            experiment = experiment.model_copy(update={"confidence": confidence}) if hasattr(experiment, "model_copy") else experiment.copy(update={"confidence": confidence})
            bundle.experiments.append(experiment)
            bundle.entities.extend([*materials, *equipment, *laboratories, *teams, *employees, *topic_tags])
        else:
            bundle.entities.extend(materials)
            bundle.rejected_items.append(RejectedExtraction(item_type="table_row", reason="missing_regime_or_measurement", raw_payload=row, evidence=[evidence]))

        if gap_raw:
            material = materials[0].canonical_name if materials else None
            regime = regimes[0].canonical_name if regimes else None
            prop = resolve_property(property_raw) if property_raw else None
            raw = "|".join([material or "", regime or "", prop or "", gap_raw])
            bundle.data_gaps.append(
                ExtractedDataGap(
                    gap_id=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24],
                    material=material,
                    regime=regime,
                    property=prop or _property_from_gap(gap_raw),
                    reason=gap_raw,
                    confidence=0.85,
                    evidence=[evidence],
                )
            )

        bundle.diagnostics = {"row_index": source.row_index, "columns": list(row)}
        return bundle


def parse_serialized_row(text: str) -> dict[str, str]:
    row: dict[str, str] = {}
    for line in (text or "").splitlines():
        if line.startswith("Table:") or line.startswith("Table columns:"):
            continue
        for part in line.split(" | "):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            clean_key = _norm_col(key)
            clean_value = value.strip()
            if clean_key and clean_value:
                row[clean_key] = clean_value
    return row


def _pick(row: dict[str, str], aliases: Iterable[str]) -> str:
    alias_set = {_norm_col(alias) for alias in aliases}
    for key, value in row.items():
        if key in alias_set:
            return value
    for key, value in row.items():
        if any(alias in key for alias in alias_set):
            return value
    return ""


def _norm_col(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower().replace("ё", "е"))


def _multi_values(value: str) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[;\n]+", value)
    return [clean_raw(part) for part in parts if clean_raw(part)]


def _float_or_none(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[-+]?\d+(?:[\.,]\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0).replace(",", "."))


def _effect_from_text(value: str | None) -> str:
    norm = str(value or "").lower()
    if any(term in norm for term in ["increase", "increased", "повыс", "увелич"]):
        return "increase"
    if any(term in norm for term in ["decrease", "decreased", "сниз", "уменьш"]):
        return "decrease"
    if any(term in norm for term in ["unchanged", "без измен", "no change"]):
        return "no_change"
    return "unknown"


def _entity(entity_type: str, raw: str, evidence: EvidenceSpan) -> ExtractedEntity:
    return ExtractedEntity(entity_type=entity_type, raw_name=raw, canonical_name=clean_raw(raw), confidence=0.75, evidence=[evidence])


def _dedupe_entities(items: list[ExtractedEntity]) -> list[ExtractedEntity]:
    seen = set()
    result: list[ExtractedEntity] = []
    for item in items:
        key = (item.entity_type, item.canonical_name)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _stable_experiment_id(source_name: str, row_index: int | None, materials, regimes, measurements) -> str:
    raw = "|".join(
        [
            source_name,
            str(row_index or 0),
            ",".join(item.canonical_name for item in materials),
            ",".join(item.canonical_name for item in regimes),
            ",".join(item.property_canonical for item in measurements),
        ]
    )
    return "table_exp_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _property_from_gap(text: str) -> str | None:
    prop = resolve_property(text)
    return prop if prop != clean_raw(text) else None
