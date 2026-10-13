import json
import shutil
import uuid
from pathlib import Path

import pytest

from tests.conftest import make_pdf, make_scanned_pdf

pytestmark = pytest.mark.db

API = "/api/v1"
POLICY = make_pdf(
    [
        "Eligibility. Employees become eligible for remote work after completing a 90-day onboarding period.",
        "Equipment. Each remote employee receives a one-time home office stipend of 600 euros.",
    ]
)


def upload(client, data, name, content_type="application/octet-stream"):
    return client.post(f"{API}/documents", files={"file": (name, data, content_type)})


def test_health_and_ready(client):
    assert client.get(f"{API}/health").json() == {"status": "ok"}
    ready = client.get(f"{API}/ready")
    assert ready.status_code == 200
    body = ready.json()
    assert body["pgvector"] is True and body["migration"] == "0001_initial" and body["demo_mode"] is True


def test_openapi_is_available(client):
    paths = client.get(f"{API}/openapi.json").json()["paths"]
    for path in ("/api/v1/documents", "/api/v1/ask", "/api/v1/conversations", "/api/v1/evaluation/runs"):
        assert path in paths


def test_upload_lifecycle(client):
    resp = upload(client, POLICY, "Remote Policy.pdf", "application/pdf")
    assert resp.status_code == 201, resp.text
    doc = resp.json()["document"]
    assert doc["title"] == "Remote Policy" and doc["content_type"] == "pdf"
    # Ingestion runs inline in tests (DOCUMIND_INGEST_IN_BACKGROUND=false)
    status = client.get(f"{API}/documents/{doc['id']}/status").json()
    assert status["status"] == "completed" and status["chunk_count"] >= 2

    dup = upload(client, POLICY, "copy.pdf")
    assert dup.status_code == 200 and dup.json()["duplicate"] is True and dup.json()["document"]["id"] == doc["id"]

    listing = client.get(f"{API}/documents", params={"limit": 1}).json()
    assert listing["total"] == 1 and len(listing["items"]) == 1

    chunks = client.get(f"{API}/documents/{doc['id']}/chunks").json()
    assert chunks["total"] == status["chunk_count"]
    chunk = client.get(f"{API}/chunks/{chunks['items'][0]['id']}").json()
    assert chunk["document_id"] == doc["id"]

    assert client.delete(f"{API}/documents/{doc['id']}").status_code == 204
    assert client.get(f"{API}/documents/{doc['id']}").status_code == 404
    assert client.get(f"{API}/chunks/{chunk['id']}").status_code == 404


def test_pagination_and_status_filter(client):
    for i in range(3):
        upload(client, f"Document number {i} about distinct topic {uuid.uuid4().hex}".encode(), f"d{i}.txt")
    page = client.get(f"{API}/documents", params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 3 and len(page["items"]) == 1
    assert client.get(f"{API}/documents", params={"status": "failed"}).json()["total"] == 0
    assert client.get(f"{API}/documents", params={"status": "bogus"}).status_code == 422
    assert client.get(f"{API}/documents", params={"limit": 1000}).status_code == 422


@pytest.mark.parametrize(
    ("data", "name", "status", "code"),
    [
        (b"", "empty.txt", 422, "empty_file"),
        (b"hello", "image.png", 415, "unsupported_type"),
        (b"not a pdf", "fake.pdf", 415, "content_mismatch"),
        (b"x" * (2 * 1024 * 1024 + 10), "big.txt", 413, "file_too_large"),
    ],
)
def test_upload_validation_errors(client, data, name, status, code):
    resp = upload(client, data, name)
    assert resp.status_code == status
    assert resp.json()["error"]["code"] == code


def test_scanned_pdf_reports_ocr_required_and_retry_rules(client):
    doc = upload(client, make_scanned_pdf(), "scan.pdf").json()["document"]
    status = client.get(f"{API}/documents/{doc['id']}/status").json()
    assert status["status"] == "failed" and status["error_code"] == "ocr_required"
    # Retrying is allowed for failed documents (it fails again for the same reason).
    assert client.post(f"{API}/documents/{doc['id']}/retry").status_code == 202
    ok = upload(client, POLICY, "policy.pdf").json()["document"]
    assert client.post(f"{API}/documents/{ok['id']}/retry").status_code == 409
    assert client.post(f"{API}/documents/{ok['id']}/retry", params={"force": "true"}).status_code == 202


def test_ask_and_search(client):
    upload(client, POLICY, "policy.pdf")
    answer = client.post(f"{API}/ask", json={"question": "  How much is the home   office stipend? "}).json()
    assert answer["evidence_status"] == "supported" and answer["citations"][0]["page_number"] == 2
    search = client.post(f"{API}/search", json={"query": "stipend", "mode": "keyword", "top_k": 2}).json()
    assert search["passages"][0]["keyword_rank"] == 1
    none = client.post(f"{API}/ask", json={"question": "What is the capital of Australia?"}).json()
    assert none["evidence_status"] == "insufficient_evidence"


@pytest.mark.parametrize(
    "body",
    [{"question": ""}, {"question": "   "}, {"question": "x" * 2001}, {"question": "q", "document_ids": ["nope"]}, {}],
)
def test_ask_validation(client, body):
    resp = client.post(f"{API}/ask", json=body)
    assert resp.status_code == 422 and resp.json()["error"]["code"] == "validation_error"


def test_ask_unknown_document_is_404(client):
    resp = client.post(f"{API}/ask", json={"question": "stipend", "document_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404


def test_conversation_endpoints(client):
    doc = upload(client, POLICY, "policy.pdf").json()["document"]
    conv = client.post(f"{API}/conversations", json={"document_ids": [doc["id"]]}).json()
    assert conv["document_ids"] == [doc["id"]]
    turn = client.post(
        f"{API}/conversations/{conv['id']}/messages", json={"question": "How much is the home office stipend?"}
    )
    assert turn.status_code == 200
    assistant = turn.json()["assistant_message"]
    assert assistant["evidence_status"] == "supported" and assistant["citations"]
    detail = client.get(f"{API}/conversations/{conv['id']}").json()
    assert len(detail["messages"]) == 2 and detail["title"] == "How much is the home office stipend?"
    assert client.get(f"{API}/conversations").json()["items"][0]["message_count"] == 2
    assert client.delete(f"{API}/conversations/{conv['id']}").status_code == 204
    assert client.get(f"{API}/conversations/{conv['id']}").status_code == 404
    bad = client.post(f"{API}/conversations", json={"document_ids": [str(uuid.uuid4())]})
    assert bad.status_code == 404


def test_study_endpoint(client):
    doc = upload(client, POLICY, "policy.pdf").json()["document"]
    out = client.post(f"{API}/documents/{doc['id']}/study", json={"task": "key_points"}).json()
    assert out["items"] and out["items"][0]["citations"]
    assert client.post(f"{API}/documents/{doc['id']}/study", json={"task": "poem"}).status_code == 422


def test_evaluation_endpoints(client):
    sample_dir = Path(__file__).resolve().parents[2] / "data" / "sample_documents"
    for path in sorted(sample_dir.iterdir()):
        upload(client, path.read_bytes(), path.name)
    run = client.post(f"{API}/evaluation/runs", json={"dataset": "default", "top_k": 5})
    assert run.status_code == 201, run.text
    body = run.json()
    assert body["status"] == "completed"
    assert 0 <= body["metrics"]["retrieval"]["recall_at_5"] <= 1
    assert body["metrics"]["generation"]["correct_abstention_rate"] is not None
    listing = client.get(f"{API}/evaluation/runs").json()
    assert listing["total"] == 1
    assert client.get(f"{API}/evaluation/runs/{body['id']}").json()["id"] == body["id"]
    bad = client.post(f"{API}/evaluation/runs", json={"dataset": "../../etc/passwd"})
    assert bad.status_code == 422


def test_evaluation_requires_documents(client):
    resp = client.post(f"{API}/evaluation/runs", json={"dataset": "default"})
    assert resp.status_code == 422 and resp.json()["error"]["code"] == "evaluation_documents_missing"


def test_error_responses_do_not_leak_internals(client, monkeypatch):
    from app.services import documents

    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail /etc/hosts")

    monkeypatch.setattr(documents.DocumentService, "list", boom)
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        resp = c.get(f"{API}/documents")
    assert resp.status_code == 500
    assert "secret" not in resp.text and json.loads(resp.text)["error"]["code"] == "internal_error"


def test_uploaded_files_are_stored_under_generated_names(client, storage):
    doc = upload(client, b"Some text content for storage test.", "../../evil name.txt").json()["document"]
    assert doc["original_filename"] == "evil name.txt"
    files = list(Path(storage.root).iterdir())
    assert all(".." not in f.name and " " not in f.name for f in files)
    shutil.rmtree(storage.root, ignore_errors=True)
