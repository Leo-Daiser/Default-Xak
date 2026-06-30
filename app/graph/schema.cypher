CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (n:Document) REQUIRE n.document_id IS UNIQUE;

CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (n:DocumentChunk) REQUIRE n.chunk_id IS UNIQUE;

CREATE CONSTRAINT experiment_id_unique IF NOT EXISTS
FOR (n:Experiment) REQUIRE n.experiment_id IS UNIQUE;

CREATE CONSTRAINT material_name_unique IF NOT EXISTS
FOR (n:Material) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT regime_name_unique IF NOT EXISTS
FOR (n:ProcessRegime) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT property_name_unique IF NOT EXISTS
FOR (n:Property) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT measurement_id_unique IF NOT EXISTS
FOR (n:Measurement) REQUIRE n.measurement_id IS UNIQUE;

CREATE CONSTRAINT equipment_name_unique IF NOT EXISTS
FOR (n:Equipment) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT laboratory_name_unique IF NOT EXISTS
FOR (n:Laboratory) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT team_name_unique IF NOT EXISTS
FOR (n:ResearchTeam) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT employee_name_unique IF NOT EXISTS
FOR (n:Employee) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT topic_tag_name_unique IF NOT EXISTS
FOR (n:TopicTag) REQUIRE n.canonical_name IS UNIQUE;

CREATE CONSTRAINT conclusion_id_unique IF NOT EXISTS
FOR (n:Conclusion) REQUIRE n.conclusion_id IS UNIQUE;

CREATE CONSTRAINT gap_id_unique IF NOT EXISTS
FOR (n:DataGap) REQUIRE n.gap_id IS UNIQUE;

CREATE INDEX material_aliases_index IF NOT EXISTS
FOR (n:Material) ON (n.aliases);

CREATE INDEX chunk_text_index IF NOT EXISTS
FOR (n:DocumentChunk) ON (n.text);

CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
FOR (n:DocumentChunk) ON EACH [n.text];
