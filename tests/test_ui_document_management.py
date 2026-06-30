from __future__ import annotations

from app.ui_helpers import active_document_changes, documents_to_rows


def test_document_rows_format_active_metadata() -> None:
    rows = documents_to_rows(
        {
            "items": [
                {
                    "doc_id": "doc1",
                    "filename": "demo.csv",
                    "source_type": "file",
                    "chunks": 3,
                    "active": False,
                    "updated_at": "2026-01-01",
                    "parser": "pandas",
                    "document_intelligence": {"blocks_count": 1, "tables_count": 1},
                }
            ]
        }
    )
    assert rows == [
        {
            "Документ": "demo.csv",
            "Тип": "file",
            "Chunks": 3,
            "Активен": False,
            "Дата загрузки": "2026-01-01",
            "Parser": "pandas",
            "Blocks": 1,
            "Tables": 1,
            "doc_id": "doc1",
        }
    ]


def test_active_document_changes_detect_checkbox_diff() -> None:
    original = [
        {"doc_id": "doc1", "Активен": True},
        {"doc_id": "doc2", "Активен": False},
    ]
    edited = [
        {"doc_id": "doc1", "Активен": False},
        {"doc_id": "doc2", "Активен": False},
    ]
    assert active_document_changes(original, edited) == [("doc1", False)]
