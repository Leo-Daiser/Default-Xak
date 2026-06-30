"""Cypher snippets for the Neo4j-backed strict graph repository."""

EXACT_MATERIAL_REGIME_PROPERTY = """
MATCH (m:Material)<-[:USES_MATERIAL]-(e:Experiment)
MATCH (e)-[:HAS_REGIME]->(r:ProcessRegime)
MATCH (e)-[:MEASURED]->(meas:Measurement)-[:OF_PROPERTY]->(p:Property)
WHERE m.canonical_name = $material
  AND r.canonical_name = $regime
  AND p.canonical_name = $property
OPTIONAL MATCH (e)-[:USED_EQUIPMENT]->(eq:Equipment)
OPTIONAL MATCH (e)-[:PERFORMED_BY]->(team:ResearchTeam)
OPTIONAL MATCH (team)-[:BELONGS_TO]->(lab_from_team:Laboratory)
OPTIONAL MATCH (e)-[:PERFORMED_AT]->(lab_direct:Laboratory)
OPTIONAL MATCH (e)-[:LED_TO]->(concl:Conclusion)
OPTIONAL MATCH (e)-[:SUPPORTED_BY]->(chunk:DocumentChunk)<-[:HAS_CHUNK]-(doc:Document)
RETURN e,
       collect(DISTINCT m) AS materials,
       collect(DISTINCT r) AS regimes,
       collect(DISTINCT {measurement: meas, property: p}) AS measurements,
       collect(DISTINCT eq) AS equipment,
       collect(DISTINCT team) AS teams,
       collect(DISTINCT lab_from_team) + collect(DISTINCT lab_direct) AS laboratories,
       collect(DISTINCT concl) AS conclusions,
       collect(DISTINCT chunk) AS chunks,
       collect(DISTINCT doc) AS documents
ORDER BY e.experiment_id
"""

FIND_EXPERIMENTS_BY_CONSTRAINTS = """
MATCH (e:Experiment)
MATCH (e)-[:USES_MATERIAL]->(m:Material)
WHERE $material IS NULL OR m.canonical_name = $material
WITH e, collect(DISTINCT m) AS materials
MATCH (e)-[:HAS_REGIME]->(r:ProcessRegime)
WHERE $regime IS NULL OR r.canonical_name = $regime
WITH e, materials, collect(DISTINCT r) AS regimes
MATCH (e)-[:MEASURED]->(meas:Measurement)-[:OF_PROPERTY]->(p:Property)
WHERE $property IS NULL OR p.canonical_name = $property
OPTIONAL MATCH (e)-[:USED_EQUIPMENT]->(eq:Equipment)
OPTIONAL MATCH (e)-[:PERFORMED_BY]->(team:ResearchTeam)
OPTIONAL MATCH (team)-[:BELONGS_TO]->(lab_from_team:Laboratory)
OPTIONAL MATCH (e)-[:PERFORMED_AT]->(lab_direct:Laboratory)
OPTIONAL MATCH (e)-[:LED_TO]->(concl:Conclusion)
OPTIONAL MATCH (e)-[:SUPPORTED_BY]->(chunk:DocumentChunk)<-[:HAS_CHUNK]-(doc:Document)
RETURN e,
       materials,
       regimes,
       collect(DISTINCT {measurement: meas, property: p}) AS measurements,
       collect(DISTINCT eq) AS equipment,
       collect(DISTINCT team) AS teams,
       collect(DISTINCT lab_from_team) + collect(DISTINCT lab_direct) AS laboratories,
       collect(DISTINCT concl) AS conclusions,
       collect(DISTINCT chunk) AS chunks,
       collect(DISTINCT doc) AS documents
ORDER BY e.experiment_id
LIMIT $limit
"""

DECISION_HISTORY_BY_MATERIAL = """
MATCH (m:Material)<-[:USES_MATERIAL]-(e:Experiment)
WHERE m.canonical_name = $material
OPTIONAL MATCH (e)-[:HAS_REGIME]->(r:ProcessRegime)
OPTIONAL MATCH (e)-[:MEASURED]->(meas:Measurement)-[:OF_PROPERTY]->(p:Property)
OPTIONAL MATCH (e)-[:USED_EQUIPMENT]->(eq:Equipment)
OPTIONAL MATCH (e)-[:PERFORMED_BY]->(team:ResearchTeam)
OPTIONAL MATCH (team)-[:BELONGS_TO]->(lab_from_team:Laboratory)
OPTIONAL MATCH (e)-[:PERFORMED_AT]->(lab_direct:Laboratory)
OPTIONAL MATCH (e)-[:LED_TO]->(concl:Conclusion)
OPTIONAL MATCH (e)-[:SUPPORTED_BY]->(chunk:DocumentChunk)<-[:HAS_CHUNK]-(doc:Document)
RETURN e,
       collect(DISTINCT m) AS materials,
       collect(DISTINCT r) AS regimes,
       collect(DISTINCT {measurement: meas, property: p}) AS measurements,
       collect(DISTINCT eq) AS equipment,
       collect(DISTINCT team) AS teams,
       collect(DISTINCT lab_from_team) + collect(DISTINCT lab_direct) AS laboratories,
       collect(DISTINCT concl) AS conclusions,
       collect(DISTINCT chunk) AS chunks,
       collect(DISTINCT doc) AS documents
ORDER BY e.experiment_id
"""

FIND_GAPS = """
MATCH (g:DataGap)
OPTIONAL MATCH (g)-[:GAP_FOR_ENTITY]->(m:Material)
OPTIONAL MATCH (g)-[:GAP_FOR_REGIME]->(r:ProcessRegime)
OPTIONAL MATCH (g)-[:GAP_FOR_PROPERTY]->(p:Property)
OPTIONAL MATCH (g)-[:SUPPORTED_BY]->(chunk:DocumentChunk)<-[:HAS_CHUNK]-(doc:Document)
WITH g,
     collect(DISTINCT m) AS materials,
     collect(DISTINCT r) AS regimes,
     collect(DISTINCT p) AS properties,
     collect(DISTINCT chunk) AS chunks,
     collect(DISTINCT doc) AS documents
WHERE ($material IS NULL OR any(item IN materials WHERE item.canonical_name = $material) OR g.material = $material)
  AND ($regime IS NULL OR any(item IN regimes WHERE item.canonical_name = $regime) OR g.regime = $regime)
  AND ($property IS NULL OR any(item IN properties WHERE item.canonical_name = $property) OR g.property = $property)
RETURN g,
       materials,
       regimes,
       properties,
       chunks,
       documents
ORDER BY g.gap_id
"""
