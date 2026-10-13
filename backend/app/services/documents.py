"""Document management service."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import ConflictError, NotFoundError
from app.ingestion.pipeline import STALE_PROCESSING_AFTER
from app.ingestion.storage import FileStorage
from app.ingestion.validation import validate_upload
from app.models import Chunk, Document, DocumentStatus

logger = logging.getLogger(__name__)


def _is_stale(doc: Document) -> bool:
    """A processing claim older than the pipeline's stale window (e.g. after a crash)."""
    started = doc.processing_started_at
    return started is None or datetime.now(UTC) - started >= STALE_PROCESSING_AFTER


def get_storage(settings: Settings | None = None) -> FileStorage:
    return FileStorage((settings or get_settings()).upload_dir)


class DocumentService:
    def __init__(self, session: Session, storage: FileStorage | None = None, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.storage = storage or get_storage(self.settings)

    def upload(self, data: bytes, filename: str | None) -> tuple[Document, bool]:
        """Validate and store a file. Returns ``(document, is_duplicate)``."""
        validated = validate_upload(
            data, filename, max_bytes=self.settings.max_upload_bytes, max_pages=self.settings.max_pdf_pages
        )
        existing = self.session.execute(
            select(Document).where(Document.sha256 == validated.sha256)
        ).scalar_one_or_none()
        if existing is not None:
            logger.info("duplicate_upload", extra={"document_id": str(existing.id)})
            return existing, True

        stored_name = self.storage.save(data, validated.kind)
        doc = Document(
            title=validated.title,
            original_filename=validated.safe_filename,
            stored_filename=stored_name,
            content_type=validated.kind,
            size_bytes=validated.size_bytes,
            sha256=validated.sha256,
            page_count=validated.page_count,
            status=DocumentStatus.pending,
        )
        self.session.add(doc)
        try:
            self.session.commit()
        except IntegrityError:  # concurrent upload of the same file
            self.session.rollback()
            self.storage.delete(stored_name)
            existing = self.session.execute(select(Document).where(Document.sha256 == validated.sha256)).scalar_one()
            return existing, True
        logger.info(
            "document_uploaded",
            extra={"document_id": str(doc.id), "kind": doc.content_type, "bytes": doc.size_bytes},
        )
        return doc, False

    def list(self, *, limit: int = 50, offset: int = 0, status: str | None = None) -> tuple[list[Document], int]:
        stmt = select(Document)
        count = select(func.count(Document.id))
        if status:
            stmt = stmt.where(Document.status == DocumentStatus(status))
            count = count.where(Document.status == DocumentStatus(status))
        items = (
            self.session.execute(stmt.order_by(Document.created_at.desc(), Document.id).limit(limit).offset(offset))
            .scalars()
            .all()
        )
        return list(items), int(self.session.execute(count).scalar_one())

    def get(self, document_id: uuid.UUID) -> Document:
        doc = self.session.get(Document, document_id)
        if doc is None:
            raise NotFoundError("Document not found.")
        return doc

    def get_chunk(self, chunk_id: uuid.UUID) -> Chunk:
        chunk = self.session.get(Chunk, chunk_id)
        if chunk is None:
            raise NotFoundError("Passage not found.")
        return chunk

    def list_chunks(self, document_id: uuid.UUID, *, limit: int, offset: int) -> tuple[list[Chunk], int]:
        self.get(document_id)
        items = (
            self.session.execute(
                select(Chunk)
                .where(Chunk.document_id == document_id)
                .order_by(Chunk.chunk_index)
                .limit(limit)
                .offset(offset)
            )
            .scalars()
            .all()
        )
        total = self.session.execute(select(func.count(Chunk.id)).where(Chunk.document_id == document_id)).scalar_one()
        return list(items), int(total)

    def delete(self, document_id: uuid.UUID) -> None:
        doc = self.get(document_id)
        stored = doc.stored_filename
        self.session.delete(doc)  # chunks and conversation links cascade in the database
        self.session.commit()
        self.storage.delete(stored)
        logger.info("document_deleted", extra={"document_id": str(document_id)})

    def prepare_retry(self, document_id: uuid.UUID, *, force: bool = False) -> Document:
        doc = self.get(document_id)
        if doc.status == DocumentStatus.processing and not _is_stale(doc):
            raise ConflictError("The document is currently being processed.", code="already_processing")
        if doc.status == DocumentStatus.completed and not force:
            raise ConflictError(
                "The document is already indexed. Use force=true to re-index it.", code="already_completed"
            )
        if not self.storage.exists(doc.stored_filename):
            raise ConflictError("The original file is no longer available; upload it again.", code="file_missing")
        doc.status = DocumentStatus.pending
        doc.error_code = None
        doc.error_message = None
        self.session.commit()
        return doc
