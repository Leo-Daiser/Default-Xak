from __future__ import annotations

import json

from scripts.resource_efficiency_report import build_resource_report


def test_resource_efficiency_report_produces_json_safe_summary() -> None:
    report = build_resource_report(
        api_base=None,
        include_docker=False,
        extraction_report={
            "summary": {
                "total_documents": 2,
                "total_chunks": 5,
                "canonical_facts_count": 3,
                "facts_without_evidence": 0,
                "normalized_measurements_count": 2,
            }
        },
    )

    rendered = json.dumps(report, ensure_ascii=False)

    assert report["summary"]["runtime_profile"]
    assert report["summary"]["api_docker_image_size_bytes"] == "unknown"
    assert report["summary"]["facts_without_evidence"] == 0
    assert "MISTRAL_API_KEY" not in rendered
    assert "OPENROUTER_API_KEY" not in rendered
