"""Idempotent materialization of strict ontology facts into Neo4j."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

from ..domain.aliases import MATERIAL_ALIASES, PROPERTY_ALIASES, REGIME_ALIASES
from ..domain.fact_normalization import measurement_normalization_fields, with_normalized_measurement_fields
from ..domain.ontology import DataGap, Evidence
from ..extraction.deterministic import DeterministicExtractor
from ..models.schemas import Chunk, Document
from ..storage.catalog import SQLiteCatalog
from ..extraction.extraction import EntityRelationExtractor
from ..extraction.pipeline import ExtractionPipeline
from ..extraction.to_graph_models import bundle_to_data_gaps, bundle_to_experiment_facts
from .graph_db import GraphDB
from .graph_models import ExperimentFact


def deterministic_measurement_id(
    experiment_id: str,
    material: str | None,
    regime: str | None,
    property_name: str,
    value: object,
    unit: str | None,
    source_chunk_id: str | None,
) -> str:
    """Return a stable measurement ID for idempotent Neo4j MERGE writes."""
    raw = "|".join(
        [
            experiment_id or "",
            material or "",
            regime or "",
            property_name or "",
            "" if value is None else str(value),
            unit or "",
            source_chunk_id or "",
        ]
    )
    return "measurement_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:32]


@dataclass
class GraphWriteStats:
    documents_processed: int = 0
    chunks_processed: int = 0
    tables_processed: int = 0
    accepted_entities: int = 0
    accepted_experiments: int = 0
    accepted_measurements: int = 0
    accepted_gaps: int = 0
    rejected_items: int = 0
    confidence_values: list[float] = field(default_factory=list)
    documents_written: set[str] = field(default_factory=set)
    chunks_written: set[str] = field(default_factory=set)
    experiments_written: set[str] = field(default_factory=set)
    materials_written: set[str] = field(default_factory=set)
    regimes_written: set[str] = field(default_factory=set)
    properties_written: set[str] = field(default_factory=set)
    measurements_written: set[str] = field(default_factory=set)
    equipment_written: set[str] = field(default_factory=set)
    laboratories_written: set[str] = field(default_factory=set)
    teams_written: set[str] = field(default_factory=set)
    employees_written: set[str] = field(default_factory=set)
    conclusions_written: set[str] = field(default_factory=set)
    gaps_written: set[str] = field(default_factory=set)
    relationships_written: int = 0
    normalized_measurements_backfilled: int = 0

    def to_dict(self) -> dict[str, int | float]:
        mean_confidence = 0.0
        if self.confidence_values:
            mean_confidence = round(sum(self.confidence_values) / len(self.confidence_values), 4)
        return {
            "documents_processed": self.documents_processed,
            "chunks_processed": self.chunks_processed,
            "tables_processed": self.tables_processed,
            "accepted_entities": self.accepted_entities,
            "accepted_experiments": self.accepted_experiments,
            "accepted_measurements": self.accepted_measurements,
            "accepted_gaps": self.accepted_gaps,
            "rejected_items": self.rejected_items,
            "mean_confidence": mean_confidence,
            "documents_written": len(self.documents_written),
            "chunks_written": len(self.chunks_written),
            "experiments_written": len(self.experiments_written),
            "materials_written": len(self.materials_written),
            "regimes_written": len(self.regimes_written),
            "properties_written": len(self.properties_written),
            "measurements_written": len(self.measurements_written),
            "equipment_written": len(self.equipment_written),
            "laboratories_written": len(self.laboratories_written),
            "teams_written": len(self.teams_written),
            "employees_written": len(self.employees_written),
            "conclusions_written": len(self.conclusions_written),
            "gaps_written": len(self.gaps_written),
            "relationships_written": self.relationships_written,
            "normalized_measurements_backfilled": self.normalized_measurements_backfilled,
        }


class GraphWriter:
    """Write strict ontology facts to Neo4j with idempotent MERGE statements."""

    def __init__(self, graph_db: GraphDB, pipeline: ExtractionPipeline | None = None) -> None:
        self.graph_db = graph_db
        self.pipeline = pipeline

    def sync_catalog(
        self,
        catalog: SQLiteCatalog,
        extractor: EntityRelationExtractor | None = None,
        document_getter: Callable[[str], Document | None] | None = None,
        pipeline: ExtractionPipeline | None = None,
    ) -> dict[str, int | float]:
        """Run structured extraction over the catalog and materialize accepted facts."""
        _ = document_getter  # kept for backward-compatible call sites
        active_pipeline = pipeline or self.pipeline
        if active_pipeline is None:
            deterministic = DeterministicExtractor(extractor) if extractor is not None else None
            active_pipeline = ExtractionPipeline(deterministic_extractor=deterministic)
        stats = GraphWriteStats()
        with self.graph_db.session() as session:
            for document in catalog.list_documents():
                active = catalog.is_document_active(document.doc_id) if hasattr(catalog, "is_document_active") else True
                self.write_document(session, document, stats, active=active)
                if not active:
                    self.mark_document_chunks_active(session, document.doc_id, active=False)
                    continue
                stats.documents_processed += 1
                self.mark_document_chunks_active(session, document.doc_id, active=False)
                for chunk in catalog.list_chunks(document.doc_id):
                    stats.chunks_processed += 1
                    if chunk.metadata.get("chunk_kind") == "table_row":
                        stats.tables_processed += 1
                    self.write_chunk(session, document, chunk, stats, active=True)
                    bundle = active_pipeline.extract_from_chunk(chunk)
                    self.write_bundle(session, bundle, stats)
            self.backfill_normalized_measurements(session, stats)
        return stats.to_dict()

    def backfill_normalized_measurements(self, session, stats: GraphWriteStats) -> None:
        """Populate normalized fields on legacy Measurement nodes without deleting data."""

        rows = list(
            session.run(
                """
                MATCH (meas:Measurement)-[:OF_PROPERTY]->(p:Property)
                WHERE meas.value IS NOT NULL
                  AND (
                    meas.value_original IS NULL OR
                    meas.unit_original IS NULL OR
                    meas.value_normalized IS NULL OR
                    meas.unit_normalized IS NULL OR
                    meas.normalization_family IS NULL
                  )
                RETURN meas.measurement_id AS measurement_id,
                       meas.value AS value,
                       meas.raw_value AS raw_value,
                       meas.unit AS unit,
                       p.canonical_name AS property
                """
            )
        )
        for row in rows:
            measurement_id = _record_get(row, "measurement_id")
            if not measurement_id:
                continue
            value = _record_get(row, "value")
            raw_value = _record_get(row, "raw_value")
            fields = measurement_normalization_fields(
                _record_get(row, "property"),
                value if value is not None else raw_value,
                _record_get(row, "unit"),
            )
            session.run(
                """
                MATCH (meas:Measurement {measurement_id: $measurement_id})
                SET meas.value_original = $value_original,
                    meas.unit_original = $unit_original,
                    meas.value_normalized = $value_normalized,
                    meas.unit_normalized = $unit_normalized,
                    meas.normalization_family = $normalization_family
                """,
                measurement_id=measurement_id,
                **fields,
            )
            stats.normalized_measurements_backfilled += 1

    def write_bundle(self, session, bundle, stats: GraphWriteStats) -> None:
        """Write accepted extraction bundle facts. Rejected items are intentionally skipped."""
        stats.accepted_entities += len(bundle.entities)
        stats.accepted_experiments += len(bundle.experiments)
        stats.accepted_gaps += len(bundle.data_gaps)
        stats.rejected_items += len(bundle.rejected_items)
        for experiment in bundle.experiments:
            stats.accepted_measurements += len(experiment.measurements)
            stats.confidence_values.append(float(experiment.confidence))
            stats.confidence_values.extend(float(item.confidence) for item in experiment.measurements)
        for gap in bundle.data_gaps:
            stats.confidence_values.append(float(gap.confidence))
        for experiment_fact in bundle_to_experiment_facts(bundle):
            self.write_experiment(session, experiment_fact, stats)
        for gap in bundle_to_data_gaps(bundle):
            self.write_gap(session, gap, stats)

    def write_document(self, session, document: Document, stats: GraphWriteStats, *, active: bool = True) -> None:
        query = """
        MERGE (d:Document {document_id: $document_id})
        SET d.source_name = $source_name,
            d.title = $title,
            d.parser = $parser,
            d.status = $status,
            d.version = $version,
            d.active = $active,
            d.updated_at = datetime()
        """
        session.run(
            query,
            document_id=document.doc_id,
            source_name=document.title,
            title=document.title,
            parser=document.parser,
            status=document.status,
            version=document.version,
            active=bool(active),
        )
        stats.documents_written.add(document.doc_id)

    def mark_document_chunks_active(self, session, document_id: str, *, active: bool) -> None:
        session.run(
            """
            MATCH (:Document {document_id: $document_id})-[:HAS_CHUNK]->(c:DocumentChunk)
            SET c.active = $active
            """,
            document_id=document_id,
            active=bool(active),
        )

    def write_chunk(self, session, document: Document, chunk: Chunk, stats: GraphWriteStats, *, active: bool = True) -> None:
        query = """
        MERGE (d:Document {document_id: $document_id})
        SET d.source_name = coalesce(d.source_name, $source_name),
            d.title = coalesce(d.title, $source_name),
            d.active = $active,
            d.updated_at = datetime()
        MERGE (c:DocumentChunk {chunk_id: $chunk_id})
        SET c.text = $text,
            c.page = $page,
            c.page_start = $page_start,
            c.page_end = $page_end,
            c.section_path = $section_path,
            c.source_name = $source_name,
            c.document_id = $document_id,
            c.active = $active,
            c.updated_at = datetime()
        MERGE (d)-[:HAS_CHUNK]->(c)
        """
        session.run(
            query,
            document_id=document.doc_id,
            source_name=document.title,
            active=bool(active),
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            page=chunk.page_start,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section_path=chunk.section_path,
        )
        stats.chunks_written.add(chunk.chunk_id)
        stats.relationships_written += 1

    def write_experiment(self, session, experiment: ExperimentFact, stats: GraphWriteStats) -> None:
        session.run(
            """
            MERGE (e:Experiment {experiment_id: $experiment_id})
            SET e.updated_at = datetime()
            """,
            experiment_id=experiment.experiment_id,
        )
        stats.experiments_written.add(experiment.experiment_id)

        for evidence in experiment.evidence:
            self._write_evidence_chunk(session, evidence, stats)
            if evidence.chunk_id:
                session.run(
                    """
                    MATCH (e:Experiment {experiment_id: $experiment_id})
                    MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                    MERGE (e)-[:SUPPORTED_BY]->(c)
                    """,
                    experiment_id=experiment.experiment_id,
                    chunk_id=evidence.chunk_id,
                )
                stats.relationships_written += 1

        for material in experiment.materials:
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (m:Material {canonical_name: $material})
                SET m.aliases = $aliases
                MERGE (e)-[:USES_MATERIAL]->(m)
                """,
                experiment_id=experiment.experiment_id,
                material=material,
                aliases=_aliases_for(material, MATERIAL_ALIASES),
            )
            stats.materials_written.add(material)
            stats.relationships_written += 1

        for regime in experiment.regimes:
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (r:ProcessRegime {canonical_name: $regime})
                SET r.temperature = $temperature,
                    r.temperature_unit = $temperature_unit,
                    r.duration = $duration,
                    r.duration_unit = $duration_unit,
                    r.medium = $medium
                MERGE (e)-[:HAS_REGIME]->(r)
                """,
                experiment_id=experiment.experiment_id,
                regime=regime,
                temperature=None,
                temperature_unit=None,
                duration=None,
                duration_unit=None,
                medium=None,
            )
            stats.regimes_written.add(regime)
            stats.relationships_written += 1

        for raw_measurement in experiment.measurements:
            measurement = with_normalized_measurement_fields(raw_measurement)
            material = experiment.materials[0] if experiment.materials else None
            regime = experiment.regimes[0] if experiment.regimes else None
            source_chunk_id = None
            if measurement.evidence:
                source_chunk_id = measurement.evidence[0].chunk_id
            elif experiment.evidence:
                source_chunk_id = experiment.evidence[0].chunk_id
            measurement_id = deterministic_measurement_id(
                experiment.experiment_id,
                material,
                regime,
                measurement.property_name,
                measurement.value_normalized if measurement.value_normalized is not None else measurement.value if measurement.value is not None else measurement.raw_value,
                measurement.unit_normalized or measurement.unit,
                source_chunk_id,
            )
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (p:Property {canonical_name: $property})
                SET p.aliases = $property_aliases,
                    p.unit_family = $unit_family
                MERGE (meas:Measurement {measurement_id: $measurement_id})
                SET meas.value = $value,
                    meas.raw_value = $raw_value,
                    meas.unit = $unit,
                    meas.value_original = $value_original,
                    meas.unit_original = $unit_original,
                    meas.value_normalized = $value_normalized,
                    meas.unit_normalized = $unit_normalized,
                    meas.normalization_family = $normalization_family,
                    meas.effect = $effect,
                    meas.baseline_value = $baseline_value,
                    meas.delta_abs = $delta_abs,
                    meas.delta_rel_percent = $delta_rel_percent,
                    meas.confidence = $confidence
                MERGE (e)-[:MEASURED]->(meas)
                MERGE (meas)-[:OF_PROPERTY]->(p)
                """,
                experiment_id=experiment.experiment_id,
                property=measurement.property_name,
                property_aliases=_aliases_for(measurement.property_name, PROPERTY_ALIASES),
                unit_family=None,
                measurement_id=measurement_id,
                value=measurement.value,
                raw_value=measurement.raw_value,
                unit=measurement.unit,
                value_original=measurement.value_original,
                unit_original=measurement.unit_original,
                value_normalized=measurement.value_normalized,
                unit_normalized=measurement.unit_normalized,
                normalization_family=measurement.normalization_family,
                effect=measurement.effect,
                baseline_value=measurement.baseline_value,
                delta_abs=measurement.delta_abs,
                delta_rel_percent=measurement.delta_rel_percent,
                confidence=measurement.confidence,
            )
            stats.properties_written.add(measurement.property_name)
            stats.measurements_written.add(measurement_id)
            stats.relationships_written += 2
            for evidence in measurement.evidence:
                self._write_evidence_chunk(session, evidence, stats)
                if evidence.chunk_id:
                    session.run(
                        """
                        MATCH (meas:Measurement {measurement_id: $measurement_id})
                        MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                        MERGE (meas)-[:SUPPORTED_BY]->(c)
                        """,
                        measurement_id=measurement_id,
                        chunk_id=evidence.chunk_id,
                    )
                    stats.relationships_written += 1

        for equipment in experiment.equipment:
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (eq:Equipment {canonical_name: $equipment})
                MERGE (e)-[:USED_EQUIPMENT]->(eq)
                """,
                experiment_id=experiment.experiment_id,
                equipment=equipment,
            )
            stats.equipment_written.add(equipment)
            stats.relationships_written += 1

        for laboratory in experiment.laboratories:
            team = laboratory
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (lab:Laboratory {canonical_name: $laboratory})
                MERGE (team:ResearchTeam {canonical_name: $team})
                MERGE (team)-[:BELONGS_TO]->(lab)
                MERGE (e)-[:PERFORMED_BY]->(team)
                MERGE (e)-[:PERFORMED_AT]->(lab)
                """,
                experiment_id=experiment.experiment_id,
                laboratory=laboratory,
                team=team,
            )
            stats.laboratories_written.add(laboratory)
            stats.teams_written.add(team)
            stats.relationships_written += 3

        for conclusion in experiment.conclusions:
            conclusion_id = _stable_id("conclusion", experiment.experiment_id, conclusion)
            session.run(
                """
                MATCH (e:Experiment {experiment_id: $experiment_id})
                MERGE (concl:Conclusion {conclusion_id: $conclusion_id})
                SET concl.text = $text
                MERGE (e)-[:LED_TO]->(concl)
                """,
                experiment_id=experiment.experiment_id,
                conclusion_id=conclusion_id,
                text=conclusion,
            )
            stats.conclusions_written.add(conclusion_id)
            stats.relationships_written += 1

    def write_gap(self, session, gap: DataGap, stats: GraphWriteStats) -> None:
        for evidence in gap.evidence:
            self._write_evidence_chunk(session, evidence, stats)
        session.run(
            """
            MERGE (g:DataGap {gap_id: $gap_id})
            SET g.material = $material,
                g.regime = $regime,
                g.property = $property,
                g.reason = $reason,
                g.updated_at = datetime()
            """,
            gap_id=gap.gap_id,
            material=gap.material,
            regime=gap.regime,
            property=gap.property,
            reason=gap.reason,
        )
        stats.gaps_written.add(gap.gap_id)
        if gap.material:
            session.run(
                """
                MATCH (g:DataGap {gap_id: $gap_id})
                MERGE (m:Material {canonical_name: $material})
                SET m.aliases = $aliases
                MERGE (g)-[:GAP_FOR_ENTITY]->(m)
                """,
                gap_id=gap.gap_id,
                material=gap.material,
                aliases=_aliases_for(gap.material, MATERIAL_ALIASES),
            )
            stats.materials_written.add(gap.material)
            stats.relationships_written += 1
        if gap.regime:
            session.run(
                """
                MATCH (g:DataGap {gap_id: $gap_id})
                MERGE (r:ProcessRegime {canonical_name: $regime})
                SET r.aliases = $aliases
                MERGE (g)-[:GAP_FOR_REGIME]->(r)
                """,
                gap_id=gap.gap_id,
                regime=gap.regime,
                aliases=_aliases_for(gap.regime, REGIME_ALIASES),
            )
            stats.regimes_written.add(gap.regime)
            stats.relationships_written += 1
        if gap.property:
            session.run(
                """
                MATCH (g:DataGap {gap_id: $gap_id})
                MERGE (p:Property {canonical_name: $property})
                SET p.aliases = $aliases
                MERGE (g)-[:GAP_FOR_PROPERTY]->(p)
                """,
                gap_id=gap.gap_id,
                property=gap.property,
                aliases=_aliases_for(gap.property, PROPERTY_ALIASES),
            )
            stats.properties_written.add(gap.property)
            stats.relationships_written += 1
        for evidence in gap.evidence:
            if evidence.chunk_id:
                session.run(
                    """
                    MATCH (g:DataGap {gap_id: $gap_id})
                    MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                    MERGE (g)-[:SUPPORTED_BY]->(c)
                    """,
                    gap_id=gap.gap_id,
                    chunk_id=evidence.chunk_id,
                )
                stats.relationships_written += 1

    def _write_evidence_chunk(self, session, evidence: Evidence, stats: GraphWriteStats) -> None:
        if not evidence.chunk_id:
            return
        session.run(
            """
            MERGE (d:Document {document_id: $document_id})
            SET d.source_name = coalesce(d.source_name, $source_name),
                d.title = coalesce(d.title, $source_name),
                d.updated_at = datetime()
            MERGE (c:DocumentChunk {chunk_id: $chunk_id})
            SET c.text = coalesce($quote, c.text),
                c.page = $page,
                c.source_name = $source_name,
                c.document_id = $document_id,
                c.active = coalesce(c.active, true),
                c.updated_at = datetime()
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            document_id=evidence.document_id,
            source_name=evidence.source_name,
            chunk_id=evidence.chunk_id,
            quote=evidence.quote,
            page=evidence.page,
        )
        if evidence.document_id:
            stats.documents_written.add(evidence.document_id)
        stats.chunks_written.add(evidence.chunk_id)
        stats.relationships_written += 1


def sync_catalog_to_neo4j(
    graph_db: GraphDB,
    catalog: SQLiteCatalog,
    extractor: EntityRelationExtractor | None = None,
    document_getter: Callable[[str], Document | None] | None = None,
    pipeline: ExtractionPipeline | None = None,
) -> dict[str, int | float]:
    """Convenience wrapper used by API and CLI scripts."""
    return GraphWriter(graph_db, pipeline=pipeline).sync_catalog(
        catalog=catalog,
        extractor=extractor,
        document_getter=document_getter,
        pipeline=pipeline,
    )


def _aliases_for(canonical: str, aliases: dict[str, str]) -> list[str]:
    values = [alias for alias, value in aliases.items() if value == canonical]
    return list(dict.fromkeys([canonical, *values]))


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:32]}"


def _record_get(record, key: str):
    try:
        return record[key]
    except Exception:
        if isinstance(record, dict):
            return record.get(key)
        return getattr(record, key, None)
