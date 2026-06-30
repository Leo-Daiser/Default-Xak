"""Validation and rejection rules for extraction bundles."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import (
    EvidenceSpan,
    ExtractedDataGap,
    ExtractedEntity,
    ExtractedExperiment,
    ExtractedMeasurement,
    RejectedExtraction,
)
from .resolver import resolve_unit


VALID_UNITS = {"MPa", "GPa", "HV", "HRC", "%", "C", "h", "min"}


@dataclass
class ValidationResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    experiments: list[ExtractedExperiment] = field(default_factory=list)
    data_gaps: list[ExtractedDataGap] = field(default_factory=list)
    rejected: list[RejectedExtraction] = field(default_factory=list)


def validate_entity(entity: ExtractedEntity) -> tuple[ExtractedEntity | None, RejectedExtraction | None]:
    if not entity.canonical_name:
        return None, _reject("entity", "missing_canonical_name", entity)
    if not 0 <= entity.confidence <= 1:
        return None, _reject("entity", "invalid_confidence", entity)
    if not entity.evidence:
        return None, _reject("entity", "missing_evidence", entity)
    return entity, None


def validate_measurement(measurement: ExtractedMeasurement) -> tuple[ExtractedMeasurement | None, RejectedExtraction | None]:
    if not measurement.property_canonical:
        return None, _reject("measurement", "missing_property", measurement)
    if not measurement.evidence:
        return None, _reject("measurement", "missing_evidence", measurement)
    if measurement.value is None and measurement.effect == "unknown" and measurement.delta_abs is None and measurement.delta_rel_percent is None:
        return None, _reject("measurement", "measurement_without_value_or_effect", measurement)
    evidence_text = " ".join(item.quote.lower().replace("ё", "е") for item in measurement.evidence if item.quote)
    if measurement.property_canonical == "коррозионная стойкость" and "не измер" in evidence_text:
        return None, _reject("measurement", "gap_phrase_not_measurement", measurement)
    unit = resolve_unit(measurement.unit)
    if measurement.property_canonical == "прочность" and unit == "%":
        return None, _reject("measurement", "property_unit_mismatch", measurement)
    if measurement.property_canonical == "пластичность" and unit in {"MPa", "GPa"}:
        return None, _reject("measurement", "property_unit_mismatch", measurement)
    if unit and unit in VALID_UNITS:
        measurement = measurement.model_copy(update={"unit": unit}) if hasattr(measurement, "model_copy") else measurement.copy(update={"unit": unit})
    return measurement, None


def validate_experiment(experiment: ExtractedExperiment, min_confidence: float) -> tuple[ExtractedExperiment | None, list[RejectedExtraction]]:
    rejected: list[RejectedExtraction] = []
    if not experiment.materials:
        return None, [_reject("experiment", "missing_material", experiment)]
    if not experiment.regimes and not experiment.measurements:
        return None, [_reject("experiment", "missing_regime_or_measurement", experiment)]
    if not experiment.evidence:
        return None, [_reject("experiment", "missing_evidence", experiment)]
    if experiment.regimes and not experiment.measurements and _gap_signal(experiment.evidence):
        return None, [_reject("experiment", "gap_only_not_experiment", experiment)]
    measurements: list[ExtractedMeasurement] = []
    for measurement in experiment.measurements:
        accepted, rejection = validate_measurement(measurement)
        if accepted is not None:
            measurements.append(accepted)
        if rejection is not None:
            rejected.append(rejection)
    updated = experiment.model_copy(update={"measurements": measurements}) if hasattr(experiment, "model_copy") else experiment.copy(update={"measurements": measurements})
    if updated.confidence < min_confidence:
        return None, [*rejected, _reject("experiment", "low_confidence", updated)]
    return updated, rejected


def validate_gap(gap: ExtractedDataGap) -> tuple[ExtractedDataGap | None, RejectedExtraction | None]:
    if not gap.reason:
        return None, _reject("data_gap", "missing_reason", gap)
    if not gap.evidence:
        return None, _reject("data_gap", "missing_evidence", gap)
    return gap, None


def validate_items(
    entities: list[ExtractedEntity],
    experiments: list[ExtractedExperiment],
    data_gaps: list[ExtractedDataGap],
    min_confidence: float,
) -> ValidationResult:
    result = ValidationResult()
    for entity in entities:
        accepted, rejection = validate_entity(entity)
        if accepted is not None:
            result.entities.append(accepted)
        if rejection is not None:
            result.rejected.append(rejection)
    for experiment in experiments:
        accepted, rejections = validate_experiment(experiment, min_confidence=min_confidence)
        if accepted is not None:
            result.experiments.append(accepted)
        result.rejected.extend(rejections)
    for gap in data_gaps:
        accepted, rejection = validate_gap(gap)
        if accepted is not None:
            result.data_gaps.append(accepted)
        if rejection is not None:
            result.rejected.append(rejection)
    return result


def has_evidence_quotes(evidence: list[EvidenceSpan]) -> bool:
    return all(bool(item.quote.strip()) for item in evidence)


def _reject(item_type: str, reason: str, item) -> RejectedExtraction:
    payload = item.model_dump() if hasattr(item, "model_dump") else item.dict()
    evidence = payload.get("evidence") if isinstance(payload, dict) else []
    return RejectedExtraction(item_type=item_type, reason=reason, raw_payload=payload, evidence=evidence or [])


def _gap_signal(evidence: list[EvidenceSpan]) -> bool:
    text = " ".join(item.quote for item in evidence).lower().replace("ё", "е")
    return any(marker in text for marker in ["нет данных", "не измер", "отсутств", "missing data", "not measured"])
