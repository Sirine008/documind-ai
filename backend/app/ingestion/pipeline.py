"""The document ingestion pipeline.

Stages: load → extract → detect missing text → chunk → embed (batched) → persist.
Each run is idempotent: chunks for a document are replaced inside one transaction, so
re-running ingestion never duplicates chunks, and a failure leaves the previous
state intact. Only the requested document is processed (incremental ingestion).
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.ingestion.chunking import TextChunk, chunk_pages
from app.ingestion.embeddings import Embedder, embed_in_batches, get_embedder
from app.ingestion.extraction import ExtractionError, PageText, extract
from app.ingestion.storage import FileStorage
from app.models import Chunk, Document, DocumentStatus

logger = logging.getLogger(__name__)

STALE_PROCESSING_AFTER = timedelta(minutes=30)


class IngestionFailure(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class IngestionResult:
    document_id: uuid.UUID
    status: str
    chunk_count: int = 0
    skipped: bool = False
    error_code: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


class IngestionPipeline:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        storage: FileStorage,
        embedder: Embedder | None = None,
        settings: Settings | None = None,
    ):
        self.session_factory = session_factory
        self.storage = storage
        self.embedder = embedder or get_embedder()
        self.settings = settings or get_settings()

    # -- stage helpers (pure, individually testable) ------------------------------
    def extract_pages(self, kind: str, data: bytes) -> list[PageText]:
        try:
            return extract(kind, data)
        except ExtractionError as exc:
            raise IngestionFailure(exc.code, exc.message) from exc

    def check_text_coverage(self, pages: list[PageText]) -> list[int]:
        """Return page numbers with no usable text; fail if the whole document is empty."""
        min_chars = self.settings.min_page_text_chars
        empty = [p.page_number for p in pages if p.page_number and len(p.text) < min_chars]
        if all(len(p.text) < min_chars for p in pages):
            if any(p.page_number for p in pages):
                raise IngestionFailure(
                    "ocr_required",
                    "No extractable text was found. The PDF appears to contain scanned images; "
                    "run OCR on it (e.g., with OCRmyPDF) and upload the searchable version.",
                )
            raise IngestionFailure("empty_document", "The document contains no readable text.")
        return empty

    def make_chunks(self, pages: list[PageText]) -> list[TextChunk]:
        chunks = chunk_pages(
            pages,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
            min_chars=self.settings.min_chunk_chars,
        )
        if not chunks:
            raise IngestionFailure("empty_document", "The document produced no indexable text chunks.")
        return chunks

    def embed_chunks(self, chunks: list[TextChunk]) -> list[list[float]]:
        try:
            vectors = embed_in_batches(self.embedder, [c.content for c in chunks], self.settings.embedding_batch_size)
        except Exception as exc:
            logger.exception("embedding_failed")
            raise IngestionFailure("embedding_failed", f"Embedding generation failed: {exc}") from exc
        if any(len(v) != self.embedder.dim for v in vectors):
            raise IngestionFailure("embedding_failed", "Embedding dimension mismatch.")
        return vectors

    # -- orchestration -------------------------------------------------------------
    def _claim(self, document_id: uuid.UUID, force: bool) -> Document | IngestionResult:
        with self.session_factory() as session:
            doc = session.execute(
                select(Document).where(Document.id == document_id).with_for_update()
            ).scalar_one_or_none()
            if doc is None:
                return IngestionResult(document_id, "missing", skipped=True)
            if (
                doc.status == DocumentStatus.completed
                and not force
                and doc.chunk_count > 0
                and doc.embedding_model == self.embedder.name
            ):
                return IngestionResult(document_id, doc.status.value, doc.chunk_count, skipped=True)
            now = datetime.now(UTC)
            if (
                doc.status == DocumentStatus.processing
                and doc.processing_started_at
                and now - doc.processing_started_at < STALE_PROCESSING_AFTER
            ):
                return IngestionResult(document_id, doc.status.value, skipped=True)
            doc.status = DocumentStatus.processing
            doc.processing_started_at = now
            doc.attempts += 1
            doc.error_code = None
            doc.error_message = None
            session.commit()
            session.expunge(doc)
            return doc

    def _fail(self, document_id: uuid.UUID, code: str, message: str) -> None:
        with self.session_factory() as session:
            doc = session.get(Document, document_id)
            if doc is not None:
                doc.status = DocumentStatus.failed
                doc.error_code = code
                doc.error_message = message[:2000]
                session.commit()

    def run(self, document_id: uuid.UUID, *, force: bool = False) -> IngestionResult:
        claimed = self._claim(document_id, force)
        if isinstance(claimed, IngestionResult):
            logger.info("ingestion_skipped", extra={"document_id": str(document_id), "status": claimed.status})
            return claimed
        doc = claimed
        timings: dict[str, float] = {}
        log_ctx = {"document_id": str(document_id), "attempt": doc.attempts}
        logger.info("ingestion_started", extra=log_ctx)
        try:
            t = time.perf_counter()
            try:
                data = self.storage.read(doc.stored_filename)
            except (OSError, ValueError) as exc:
                raise IngestionFailure("file_missing", "The stored file could not be read.") from exc
            pages = self.extract_pages(doc.content_type, data)
            timings["extract"] = (time.perf_counter() - t) * 1000

            empty_pages = self.check_text_coverage(pages)
            t = time.perf_counter()
            chunks = self.make_chunks(pages)
            timings["chunk"] = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            vectors = self.embed_chunks(chunks)
            timings["embed"] = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            with self.session_factory() as session, session.begin():
                session.execute(delete(Chunk).where(Chunk.document_id == document_id))
                session.add_all(
                    Chunk(
                        document_id=document_id,
                        chunk_index=c.chunk_index,
                        page_number=c.page_number,
                        char_start=c.char_start,
                        char_end=c.char_end,
                        content=c.content,
                        content_hash=c.content_hash,
                        token_estimate=c.token_estimate,
                        embedding=v,
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                )
                stored = session.get(Document, document_id)
                if stored is None:  # deleted while processing
                    raise IngestionFailure("document_deleted", "Document was deleted during ingestion.")
                stored.status = DocumentStatus.completed
                stored.chunk_count = len(chunks)
                stored.page_count = len(pages) if doc.content_type == "pdf" else None
                stored.pages_without_text = empty_pages
                stored.embedding_model = self.embedder.name
                stored.processed_at = datetime.now(UTC)
            timings["persist"] = (time.perf_counter() - t) * 1000
        except IngestionFailure as exc:
            logger.warning("ingestion_failed", extra={**log_ctx, "error_code": exc.code})
            if exc.code != "document_deleted":
                self._fail(document_id, exc.code, exc.message)
            return IngestionResult(document_id, "failed", error_code=exc.code, timings_ms=timings)
        except Exception:
            logger.exception("ingestion_crashed", extra=log_ctx)
            self._fail(document_id, "internal_error", "An unexpected error occurred during ingestion.")
            return IngestionResult(document_id, "failed", error_code="internal_error", timings_ms=timings)

        logger.info(
            "ingestion_completed",
            extra={**log_ctx, "chunks": len(chunks), **{f"ms_{k}": round(v, 1) for k, v in timings.items()}},
        )
        return IngestionResult(document_id, "completed", len(chunks), timings_ms=timings)

    def run_pending(self, *, include_failed: bool = False) -> list[IngestionResult]:
        """Process only documents that still need ingestion (incremental batch mode)."""
        statuses = [DocumentStatus.pending] + ([DocumentStatus.failed] if include_failed else [])
        with self.session_factory() as session:
            ids = (
                session.execute(select(Document.id).where(Document.status.in_(statuses)).order_by(Document.created_at))
                .scalars()
                .all()
            )
        return [self.run(doc_id) for doc_id in ids]
