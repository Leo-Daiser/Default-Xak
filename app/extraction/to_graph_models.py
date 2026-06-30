"""Adapter from ExtractionBundle to graph ontology models used by GraphWriter."""

from __future__ import annotations

from ..domain.ontology import DataGap, Evidence, Measurement
from ..graph.graph_models import ExperimentFact
from .models import EvidenceSpan, ExtractionBundle


def bundle_to_experiment_facts(bundle: ExtractionBundle) -> list[ExperimentFact]:
    """Convert accepted extracted experiments to GraphWriter-compatible facts."""
    facts: list[ExperimentFact] = []
    for experiment in bundle.experiments:
        facts.append(
            ExperimentFact(
                experiment_id=experiment.experiment_id,
                materials=[item.canonical_name for item in experiment.materials],
                regimes=[item.canonical_name for item in experiment.regimes],
                measurements=[
                    Measurement(
                        property_name=measurement.property_canonical,
                        value=measurement.value,
                        raw_value=None if measurement.value is not None else "",
                        unit=measurement.unit,
                        effect=measurement.effect,
                        baseline_value=measurement.baseline_value,
                        delta_abs=measurement.delta_abs,
                        delta_rel_percent=measurement.delta_rel_percent,
                        confidence=measurement.confidence,
                        evidence=[_to_evidence(item) for item in measurement.evidence],
                    )
                    for measurement in experiment.measurements
                ],
                equipment=[item.canonical_name for item in experiment.equipment],
                laboratories=[item.canonical_name for item in experiment.laboratories],
                teams=[item.canonical_name for item in experiment.teams],
                employees=[item.canonical_name for item in experiment.employees],
                conclusions=experiment.conclusions,
                evidence=[_to_evidence(item) for item in experiment.evidence],
                source_chunk_ids=list(dict.fromkeys(item.source.chunk_id for item in experiment.evidence if item.source.chunk_id)),
            )
        )
    return facts


def bundle_to_data_gaps(bundle: ExtractionBundle) -> list[DataGap]:
    """Convert accepted extracted gaps to graph data gaps."""
    return [
        DataGap(
            gap_id=gap.gap_id,
            material=gap.material,
            regime=gap.regime,
            property=gap.property,
            reason=gap.reason,
            evidence=[_to_evidence(item) for item in gap.evidence],
        )
        for gap in bundle.data_gaps
    ]


def _to_evidence(span: EvidenceSpan) -> Evidence:
    return Evidence(
        document_id=span.source.document_id,
        chunk_id=span.source.chunk_id,
        source_name=span.source.source_name,
        page=span.source.page,
        quote=span.quote,
        confidence=span.confidence,
    )

