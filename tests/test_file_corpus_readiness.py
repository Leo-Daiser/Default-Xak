from __future__ import annotations

from pathlib import Path

from app.ingestion.document_models import DocumentIntelligenceResult
from app.parsing.file_profile import profile_corpus, profile_file, render_markdown_report
from app.parsing.text_quality import normalize_dirty_scientific_text


def test_dirty_scientific_text_normalization_is_generic() -> None:
    text = "ВТ 6 после отжига: предел прочности 980 M Pa; твердость НV 240. 7075 Т6: 77 ksi."

    normalized = normalize_dirty_scientific_text(text)

    assert "ВТ6" in normalized
    assert "980 MPa" in normalized
    assert "HV 240" in normalized
    assert "7075-T6" in normalized


def test_profile_file_extracts_facts_with_evidence(tmp_path: Path) -> None:
    path = tmp_path / "vt6.txt"
    path.write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")

    profile = profile_file(path)

    assert profile["parse_status"] == "ok"
    assert profile["chunks_count"] > 0
    assert profile["canonical_facts"] >= 1
    assert profile["facts_without_evidence"] == 0
    assert profile["fact_preview"][0]["material"] == "ВТ6"


def test_profile_file_marks_scanned_pdf_as_ocr_required(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    path.write_bytes(b"%PDF-1.4\n% fake test file")

    class FakeParser:
        def parse_document_intelligence(self, *_args, **_kwargs):
            return DocumentIntelligenceResult(
                doc_id="doc_scan",
                source_name="scan.pdf",
                parser_name="fake_pdf",
                text="",
                chunks=[],
                diagnostics={
                    "scanned_pdf_detected": True,
                    "scanned_pdf_page_count": 3,
                    "warnings": ["PDF appears scanned; OCR is disabled"],
                },
            )

    profile = profile_file(path, parser=FakeParser())  # type: ignore[arg-type]

    assert profile["parse_status"] == "ocr_required"
    assert "ocr_required" in profile["warnings"]
    assert profile["canonical_facts"] == 0


def test_profile_corpus_and_markdown_report(tmp_path: Path) -> None:
    (tmp_path / "vt6.txt").write_text("После отжига сплава ВТ6 предел прочности составил 980 MPa.", encoding="utf-8")
    (tmp_path / "unsupported.zip").write_bytes(b"PK")

    report = profile_corpus(tmp_path)
    markdown = render_markdown_report(report)

    assert report["summary"]["documents_total"] == 2
    assert report["summary"]["facts_without_evidence"] == 0
    assert report["summary"]["parse_status_counts"]["unsupported"] == 1
    assert "Corpus Readiness Report" in markdown
    assert "economy_core" in markdown
