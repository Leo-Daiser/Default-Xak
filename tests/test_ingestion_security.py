from __future__ import annotations

from fastapi.testclient import TestClient


def test_url_safety_blocks_localhost_private_and_non_http() -> None:
    from app.security.url_safety import UnsafeUrlError, validate_ingest_url

    for url in ["http://localhost/page.html", "http://127.0.0.1/page.html", "file:///etc/passwd"]:
        try:
            validate_ingest_url(url)
        except UnsafeUrlError:
            continue
        raise AssertionError(f"URL should be blocked: {url}")


def test_upload_rejects_unsupported_extension() -> None:
    import app.api as api

    client = TestClient(api.app)
    response = client.post("/ingest/documents", files=[("files", ("payload.exe", b"bad", "application/octet-stream"))])
    assert response.status_code == 400
    assert "Unsupported file extension" in response.text


def test_upload_max_size_enforced(monkeypatch) -> None:
    import app.api as api

    monkeypatch.setattr(api.settings, "max_upload_mb", 0)
    client = TestClient(api.app)
    response = client.post("/ingest/documents", files=[("files", ("small.txt", b"x", "text/plain"))])
    assert response.status_code == 413
