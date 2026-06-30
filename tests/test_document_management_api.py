from __future__ import annotations

from tests.strict_qa_helpers import seeded_client


def test_documents_endpoint_returns_items_with_active_flag(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.get("/documents")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["active"] is True
    assert payload[0]["chunks"] > 0


def test_document_active_flag_can_be_toggled(tmp_path) -> None:
    client = seeded_client(tmp_path)
    doc_id = client.get("/documents").json()[0]["doc_id"]
    off = client.patch(f"/documents/{doc_id}/active", json={"active": False})
    assert off.status_code == 200
    assert off.json()["active"] is False
    docs = client.get("/documents").json()
    assert next(item for item in docs if item["doc_id"] == doc_id)["active"] is False
    on = client.patch(f"/documents/{doc_id}/active", json={"active": True})
    assert on.status_code == 200
    assert on.json()["active"] is True


def test_inactive_document_is_excluded_from_strict_answer(tmp_path) -> None:
    client = seeded_client(tmp_path)
    question = "Что делали по сплаву ВТ6 при отжиге и какой был эффект на прочность?"
    positive = client.post("/ask", json={"question": question, "preset_id": "strict_audit"})
    assert positive.status_code == 200
    assert positive.json()["status"] == "ok"

    doc_id = client.get("/documents").json()[0]["doc_id"]
    toggle = client.patch(f"/documents/{doc_id}/active", json={"active": False})
    assert toggle.status_code == 200
    negative = client.post("/ask", json={"question": question, "preset_id": "strict_audit"})
    assert negative.status_code == 200
    assert negative.json()["status"] == "no_exact_match"


def test_graph_refresh_reindexes_active_documents(tmp_path) -> None:
    client = seeded_client(tmp_path)
    response = client.post("/graph/refresh")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "refreshed"
    assert payload["active_documents"] >= 1
    assert payload["active_chunks"] >= 1
