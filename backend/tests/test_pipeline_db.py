import pytest
from sqlalchemy import func, select

from app.ingestion.pipeline import IngestionPipeline
from app.models import Chunk, Document, DocumentStatus
from app.services.documents import DocumentService
from tests.conftest import make_pdf, make_scanned_pdf

pytestmark = pytest.mark.db

POLICY = [
    "Remote work policy. Employees are eligible after a 90-day onboarding period. " * 3,
    "Equipment. Each remote employee receives a home office stipend of 600 euros. " * 3,
]


def chunk_count(session, doc_id):
    return session.execute(select(func.count(Chunk.id)).where(Chunk.document_id == doc_id)).scalar_one()


def test_ingestion_persists_chunks_with_metadata(db_session, ingest):
    doc_id, result = ingest(make_pdf(POLICY), "policy.pdf")
    assert result.status == "completed" and result.chunk_count >= 2
    doc = db_session.get(Document, doc_id)
    assert doc.status == DocumentStatus.completed
    assert doc.chunk_count == chunk_count(db_session, doc_id) == result.chunk_count
    assert doc.page_count == 2 and doc.processed_at and doc.attempts == 1
    assert doc.embedding_model == "hashing-384"
    chunks = db_session.execute(select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.chunk_index)).scalars()
    chunks = list(chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page_number for c in chunks} == {1, 2}
    assert all(len(c.embedding) == 384 for c in chunks)
    # The generated tsvector column is populated by PostgreSQL.
    assert all(c.content_tsv for c in chunks)


def test_reingestion_is_idempotent(db_session, ingest, pipeline):
    doc_id, first = ingest(make_pdf(POLICY), "policy.pdf")
    skipped = pipeline.run(doc_id)
    assert skipped.skipped and chunk_count(db_session, doc_id) == first.chunk_count
    forced = pipeline.run(doc_id, force=True)
    assert forced.status == "completed" and not forced.skipped
    assert chunk_count(db_session, doc_id) == first.chunk_count, "re-indexing must replace, not duplicate"


def test_duplicate_upload_detected_by_hash(db_session):
    service = DocumentService(db_session)
    data = make_pdf(POLICY)
    first, dup1 = service.upload(data, "a.pdf")
    second, dup2 = service.upload(data, "renamed copy.pdf")
    assert not dup1 and dup2 and first.id == second.id
    assert db_session.execute(select(func.count(Document.id))).scalar_one() == 1


def test_scanned_pdf_fails_with_ocr_required(db_session, ingest):
    doc_id, result = ingest(make_scanned_pdf(), "scan.pdf")
    assert result.status == "failed" and result.error_code == "ocr_required"
    doc = db_session.get(Document, doc_id)
    assert doc.status == DocumentStatus.failed and "OCR" in doc.error_message
    assert chunk_count(db_session, doc_id) == 0


def test_partially_scanned_pdf_records_pages_without_text(db_session, ingest):
    doc_id, result = ingest(make_pdf([POLICY[0], "", POLICY[1]]), "mixed.pdf")
    assert result.status == "completed"
    assert db_session.get(Document, doc_id).pages_without_text == [2]


class FlakyEmbedder:
    name = "hashing-384"
    dim = 384

    def __init__(self):
        from app.ingestion.embeddings import HashingEmbedder

        self.inner = HashingEmbedder(384)
        self.fail = True

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("model unavailable")
        return self.inner.embed(texts)

    def embed_query(self, text):
        return self.inner.embed_query(text)


def test_failed_ingestion_can_be_retried(db_session, storage):
    from app.db.session import get_session_factory

    embedder = FlakyEmbedder()
    pipe = IngestionPipeline(get_session_factory(), storage, embedder=embedder)
    doc, _ = DocumentService(db_session).upload(make_pdf(POLICY), "flaky.pdf")

    failed = pipe.run(doc.id)
    assert failed.status == "failed" and failed.error_code == "embedding_failed"
    db_session.expire_all()
    assert db_session.get(Document, doc.id).status == DocumentStatus.failed
    assert chunk_count(db_session, doc.id) == 0

    embedder.fail = False
    DocumentService(db_session).prepare_retry(doc.id)
    retried = pipe.run(doc.id, force=True)
    db_session.expire_all()
    stored = db_session.get(Document, doc.id)
    assert retried.status == "completed" and stored.attempts == 2 and stored.error_code is None


def test_missing_file_fails_cleanly(db_session, storage, pipeline):
    doc, _ = DocumentService(db_session).upload(make_pdf(POLICY), "gone.pdf")
    storage.delete(doc.stored_filename)
    result = pipeline.run(doc.id)
    assert result.status == "failed" and result.error_code == "file_missing"


def test_run_pending_only_processes_new_documents(db_session, ingest, pipeline):
    done_id, _ = ingest(make_pdf(POLICY), "done.pdf")
    new_doc, _ = DocumentService(db_session).upload(
        b"A brand new text document about solar panels and inverters.", "new.txt"
    )
    results = pipeline.run_pending()
    assert [r.document_id for r in results] == [new_doc.id]
    assert results[0].status == "completed"
    assert db_session.get(Document, done_id).attempts == 1


def test_deleting_document_removes_chunks_and_file(db_session, ingest, storage):
    doc_id, _ = ingest(make_pdf(POLICY), "policy.pdf")
    stored = db_session.get(Document, doc_id).stored_filename
    assert storage.exists(stored)
    DocumentService(db_session).delete(doc_id)
    assert chunk_count(db_session, doc_id) == 0
    assert not storage.exists(stored)


def test_database_constraints_reject_invalid_rows(db_session):
    from sqlalchemy.exc import IntegrityError

    db_session.add(
        Document(
            title="x", original_filename="x.pdf", stored_filename="y", content_type="exe", size_bytes=0, sha256="z"
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_stale_processing_document_can_be_retried(db_session, ingest):
    from datetime import UTC, datetime, timedelta

    from app.core.errors import ConflictError

    doc_id, _ = ingest(make_pdf(POLICY), "stuck.pdf")
    doc = db_session.get(Document, doc_id)
    doc.status = DocumentStatus.processing
    doc.processing_started_at = datetime.now(UTC)
    db_session.commit()
    with pytest.raises(ConflictError):
        DocumentService(db_session).prepare_retry(doc_id)
    doc.processing_started_at = datetime.now(UTC) - timedelta(hours=1)  # simulated crash
    db_session.commit()
    assert DocumentService(db_session).prepare_retry(doc_id).status == DocumentStatus.pending
