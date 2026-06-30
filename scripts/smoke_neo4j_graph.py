from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.extraction.extraction import EntityRelationExtractor  # noqa: E402
from app.graph.graph_db import GraphDB  # noqa: E402
from app.graph.graph_writer import sync_catalog_to_neo4j  # noqa: E402
from app.graph.neo4j_client import apply_schema  # noqa: E402
from app.graph.neo4j_repository import Neo4jGraphRepository  # noqa: E402
from app.storage.catalog import SQLiteCatalog  # noqa: E402
from app.storage.outbox import SQLiteOutbox  # noqa: E402
from app.retrieval.retrieval import RetrievalEngine  # noqa: E402


def _load_demo_into_temp_catalog(tmp: Path) -> SQLiteCatalog:
    import app.api as api

    api.graph_db = None
    api.catalog = SQLiteCatalog(tmp / "catalog.sqlite3")
    api.outbox = SQLiteOutbox(tmp / "outbox.sqlite3")
    api.retrieval_engine = RetrievalEngine()
    api.retrieval_engine.dense_retrieve = lambda question, top_k=20: []
    api.DOCUMENTS.clear()
    api.CHUNKS.clear()
    client = TestClient(api.app)
    allowed = {".csv", ".xlsx", ".txt", ".html", ".htm", ".docx", ".md"}
    files = [
        ("files", (path.name, path.read_bytes(), "application/octet-stream"))
        for path in sorted((ROOT / "demo_data").iterdir())
        if path.suffix.lower() in allowed
    ]
    response = client.post("/ingest/documents", files=files)
    if response.status_code != 200:
        raise RuntimeError(f"demo ingestion failed: {response.status_code} {response.text}")
    return api.catalog


def main() -> int:
    try:
        graph_db = GraphDB()
    except Exception as exc:
        print(f"Neo4j unavailable: {exc}")
        return 2
    try:
        apply_schema(graph_db)
        with tempfile.TemporaryDirectory() as tmp:
            catalog = _load_demo_into_temp_catalog(Path(tmp))
            sync_catalog_to_neo4j(graph_db=graph_db, catalog=catalog, extractor=EntityRelationExtractor(), document_getter=catalog.get_document)

        repository = Neo4jGraphRepository(graph_db)
        exact = repository.find_exact_material_regime_property("ВТ6", "отжиг", "прочность")
        if not exact:
            print("Neo4j graph smoke failed: exact ВТ6 + отжиг + прочность not found")
            return 1
        missing = repository.find_exact_material_regime_property("ВТ6", "криообработка", "вязкость")
        if missing:
            print("Neo4j graph smoke failed: missing ВТ6 + криообработка + вязкость returned exact facts")
            return 1
        print("NEO4J GRAPH SMOKE TEST PASSED")
        return 0
    except Exception as exc:
        print(f"Neo4j graph smoke failed: {exc}")
        return 1
    finally:
        graph_db.close()


if __name__ == "__main__":
    raise SystemExit(main())

