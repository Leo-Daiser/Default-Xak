"""
Data models and schemas for ingestion, extraction and API.

This module defines the core entities used in the hackathon project.
The classes are intentionally simple but cover the most important
fields, including identifiers, provenance information and
confidence scores.  Pydantic models allow validation and JSON
serialisation for API responses.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Metadata about a source document.

    A document belongs to a workspace and originates from a particular
    source.  It may have multiple versions if the same logical
    document is updated over time.  Use `external_id` to record
    upstream identifiers (e.g. DOI, URL hash) for deduplication.
    """
    doc_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    workspace_uid: str
    title: str
    source_uid: Optional[str] = None
    external_id: Optional[str] = None
    parser: str
    language: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    version: Optional[int] = None


class Chunk(BaseModel):
    """A continuous segment of text extracted from a document.

    In addition to the document ID, a chunk records its ordinal
    position within the document, character offsets and optional
    hashing information for deduplication.  Embedding version is the
    model used to produce its dense representation.  The
    `workspace_uid` is stored redundantly for fast filtering.
    """
    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    doc_id: str
    workspace_uid: Optional[str] = None
    text: str
    page_start: int
    page_end: int
    section_path: str
    ordinal: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    token_count: Optional[int] = None
    text_hash: Optional[str] = None
    preview: Optional[str] = None
    embedding_version: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceRef(BaseModel):
    """Reference to a source chunk used as evidence for a fact."""
    doc_id: str
    page_start: int
    page_end: int
    chunk_id: str
    quote: str


class MaterialEntity(BaseModel):
    """A canonicalised entity representing a material, property, equipment, or concept.

    The `canonical_name` is the standardised form used in the graph; it
    should be unique across the workspace.  `entity_type` describes
    the category (e.g. material, property, process), and `aliases`
    include any observed variants or synonyms.  Normalised name
    (`norm_name`) can be used for case-folded comparisons.
    """
    canonical_name: str
    entity_type: str
    aliases: List[str] = Field(default_factory=list)
    norm_name: Optional[str] = None


class RelationAssertion(BaseModel):
    """A triple linking two entities via a predicate with qualifiers and evidence."""
    subject: str
    predicate: str
    object: str
    qualifiers: Dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.0
    evidence: List[SourceRef] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    """Result of entity and relation extraction from a chunk."""
    entities: List[MaterialEntity]
    relations: List[RelationAssertion]
    unresolved_terms: List[str] = Field(default_factory=list)


class Section(BaseModel):
    section_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    doc_id: str
    section_path: str
    title: Optional[str] = None


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    filename: Optional[str] = None
    source_type: str = "file"
    source_url: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_path: Optional[str] = None
    quote: str


class TableArtifact(BaseModel):
    table_id: str
    doc_id: str
    section_path: Optional[str] = None
    columns: List[str] = Field(default_factory=list)


class ImageArtifact(BaseModel):
    image_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    doc_id: Optional[str] = None
    url: Optional[str] = None
    alt: Optional[str] = None
    caption: Optional[str] = None
    section_path: Optional[str] = None


class TechnicalObject(BaseModel):
    name: str
    object_type: str = "TechnicalObject"
    source_chunk_id: Optional[str] = None


class Part(BaseModel):
    name: str
    source_chunk_id: Optional[str] = None


class ArticleNumber(BaseModel):
    value: str
    source_chunk_id: Optional[str] = None


class Standard(BaseModel):
    value: str
    source_chunk_id: Optional[str] = None


class Parameter(BaseModel):
    name: str
    value: Optional[str] = None
    unit: Optional[str] = None
    source_chunk_id: Optional[str] = None


class Measurement(BaseModel):
    value: str
    unit: Optional[str] = None
    parameter: Optional[str] = None
    source_chunk_id: Optional[str] = None


class Requirement(BaseModel):
    text: str
    applies_to: Optional[str] = None
    source_chunk_id: Optional[str] = None


class Process(BaseModel):
    name: str
    source_chunk_id: Optional[str] = None


class Property(BaseModel):
    name: str
    source_chunk_id: Optional[str] = None


class PropertyValue(BaseModel):
    property: str
    value: Optional[str] = None
    unit: Optional[str] = None
    change: Optional[str] = None
    source_chunk_id: Optional[str] = None


class Experiment(BaseModel):
    name: str
    material: Optional[str] = None
    process: Optional[str] = None
    property: Optional[str] = None
    source_chunk_id: Optional[str] = None


class DataGap(BaseModel):
    text: str
    missing_for: Optional[str] = None
    source_chunk_id: Optional[str] = None


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    answer: str
    status: Optional[str] = None
    answer_mode: Optional[str] = None
    analytical_intent: Optional[str] = None
    intent: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    experiments: List[Dict[str, Any]] = Field(default_factory=list)
    technical_objects: List[Dict[str, Any]] = Field(default_factory=list)
    parts: List[Dict[str, Any]] = Field(default_factory=list)
    parameters: List[Dict[str, Any]] = Field(default_factory=list)
    standards: List[Dict[str, Any]] = Field(default_factory=list)
    materials: List[Dict[str, Any]] = Field(default_factory=list)
    requirements: List[Dict[str, Any]] = Field(default_factory=list)
    equipment: List[Dict[str, Any]] = Field(default_factory=list)
    laboratories: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    gaps: List[Dict[str, Any]] = Field(default_factory=list)
    data_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    partial_matches: Dict[str, Any] = Field(default_factory=dict)
    decision_history: List[Dict[str, Any]] = Field(default_factory=list)
    subgraph: Dict[str, List[Dict[str, Any]]] = Field(default_factory=lambda: {"nodes": [], "edges": []})
    graph_context: Dict[str, Any] = Field(default_factory=dict)
    retrieval: Dict[str, Any] = Field(default_factory=dict)
    llm: Dict[str, Any] = Field(default_factory=dict)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


# Additional domain models

class Workspace(BaseModel):
    """A logical container for documents, users and sources."""
    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    slug: str
    name: str
    created_at: Optional[str] = None


class User(BaseModel):
    """System user associated with a workspace."""
    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    email: str
    display_name: Optional[str] = None


class Source(BaseModel):
    """Represents the origin of a document (file, web, API, etc.)."""
    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: str
    uri: Optional[str] = None
    checksum: Optional[str] = None
    imported_at: Optional[str] = None


class Tag(BaseModel):
    """Simple tag for documents."""
    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    name: str


class IngestJob(BaseModel):
    """Metadata for an ingestion operation."""
    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    pipeline_version: str
    status: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class SyncEvent(BaseModel):
    """Event representing a change that needs to be projected to retrieval stores."""
    uid: str = Field(default_factory=lambda: uuid.uuid4().hex)
    aggregate_type: str
    aggregate_uid: str
    op: str
    version: int
    status: str = "pending"
    created_at: Optional[str] = None
