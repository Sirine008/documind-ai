"""REST API routes. Route handlers stay thin: they validate, call a service, and
serialize; business logic lives in ``app.services``."""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, Request, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import schedule_ingestion
from app.core.config import get_settings
from app.core.errors import PayloadTooLarge, ServiceUnavailable
from app.db.session import check_database, get_db
from app.evaluation.runner import run_and_store
from app.ingestion.embeddings import get_embedder
from app.llm import get_llm_provider
from app.models import Conversation, EvaluationRun
from app.retrieval.engine import Retriever
from app.schemas import (
    AnswerOut,
    AskRequest,
    ChunkOut,
    ConversationAsk,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    ConversationTurn,
    DocumentOut,
    DocumentStatusOut,
    ErrorResponse,
    EvaluationRunOut,
    EvaluationRunRequest,
    EvaluationRunSummary,
    MessageOut,
    Page,
    SearchRequest,
    SearchResponse,
    StudyRequest,
    StudyResponse,
    UploadResponse,
)
from app.services.conversations import ConversationService
from app.services.documents import DocumentService
from app.services.qa import QAService
from app.services.study import StudyService

DB = Annotated[Session, Depends(get_db)]
ERRORS = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}

router = APIRouter()


def _limit() -> int:
    return get_settings().max_page_size


# --- Health -------------------------------------------------------------------------
@router.get("/health", tags=["health"], summary="Liveness probe")
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready", tags=["health"], summary="Readiness probe (database, pgvector, migrations)")
def ready() -> dict:
    settings = get_settings()
    try:
        db = check_database()
    except Exception as exc:
        raise ServiceUnavailable("Database is not reachable.", code="database_unavailable") from exc
    if not db["pgvector"] or not db["migration"]:
        raise ServiceUnavailable(
            "Database is reachable but not migrated. Run `alembic upgrade head`.",
            code="not_migrated",
            details=db,
        )
    return {
        "status": "ready",
        **db,
        "llm_provider": settings.llm_provider,
        "demo_mode": settings.llm_provider == "mock",
        "embedding_provider": settings.embedding_provider,
        "embedding_model": get_embedder().name,
        "max_upload_mb": settings.max_upload_mb,
    }


# --- Documents ---------------------------------------------------------------------
@router.post(
    "/documents",
    tags=["documents"],
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={200: {"model": UploadResponse, "description": "Duplicate of an existing document"}, **ERRORS},
    summary="Upload a PDF or plain-text document and queue it for ingestion",
)
async def upload_document(
    request: Request,
    response: Response,
    background: BackgroundTasks,
    db: DB,
    file: Annotated[UploadFile, File(description="PDF (.pdf) or UTF-8 text (.txt, .md)")],
) -> UploadResponse:
    settings = get_settings()
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + 64 * 1024:
        raise PayloadTooLarge(f"The file exceeds the {settings.max_upload_mb} MB upload limit.", code="file_too_large")
    data = await file.read(settings.max_upload_bytes + 1)
    await file.close()
    service = DocumentService(db)
    doc, duplicate = service.upload(data, file.filename)
    if duplicate:
        response.status_code = status.HTTP_200_OK
        message = "This file was already uploaded; returning the existing document."
    else:
        schedule_ingestion(background, doc.id)
        message = "Upload accepted. Ingestion has been queued."
    return UploadResponse(document=DocumentOut.model_validate(doc), duplicate=duplicate, message=message)


@router.get("/documents", tags=["documents"], response_model=Page[DocumentOut])
def list_documents(
    db: DB,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[
        Literal["pending", "processing", "completed", "failed"] | None, Query(alias="status")
    ] = None,
) -> Page[DocumentOut]:
    items, total = DocumentService(db).list(limit=min(limit, _limit()), offset=offset, status=status_filter)
    return Page(items=[DocumentOut.model_validate(d) for d in items], total=total, limit=limit, offset=offset)


@router.get("/documents/{document_id}", tags=["documents"], response_model=DocumentOut, responses=ERRORS)
def get_document(document_id: uuid.UUID, db: DB) -> DocumentOut:
    return DocumentOut.model_validate(DocumentService(db).get(document_id))


@router.get("/documents/{document_id}/status", tags=["documents"], response_model=DocumentStatusOut, responses=ERRORS)
def document_status(document_id: uuid.UUID, db: DB) -> DocumentStatusOut:
    d = DocumentService(db).get(document_id)
    return DocumentStatusOut(
        id=d.id,
        status=d.status.value,
        chunk_count=d.chunk_count,
        attempts=d.attempts,
        error_code=d.error_code,
        error_message=d.error_message,
        pages_without_text=d.pages_without_text,
        processed_at=d.processed_at,
    )


@router.post(
    "/documents/{document_id}/retry",
    tags=["documents"],
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
    responses=ERRORS,
    summary="Retry a failed ingestion (or re-index a completed document with force=true)",
)
def retry_document(document_id: uuid.UUID, background: BackgroundTasks, db: DB, force: bool = False) -> DocumentOut:
    doc = DocumentService(db).prepare_retry(document_id, force=force)
    schedule_ingestion(background, doc.id, force=True)
    return DocumentOut.model_validate(doc)


@router.delete("/documents/{document_id}", tags=["documents"], status_code=status.HTTP_204_NO_CONTENT, responses=ERRORS)
def delete_document(document_id: uuid.UUID, db: DB) -> Response:
    DocumentService(db).delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/documents/{document_id}/chunks", tags=["documents"], response_model=Page[ChunkOut], responses=ERRORS)
def list_chunks(
    document_id: uuid.UUID,
    db: DB,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Page[ChunkOut]:
    items, total = DocumentService(db).list_chunks(document_id, limit=limit, offset=offset)
    return Page(items=[ChunkOut.model_validate(c) for c in items], total=total, limit=limit, offset=offset)


@router.get("/chunks/{chunk_id}", tags=["documents"], response_model=ChunkOut, responses=ERRORS)
def get_chunk(chunk_id: uuid.UUID, db: DB) -> ChunkOut:
    return ChunkOut.model_validate(DocumentService(db).get_chunk(chunk_id))


@router.post(
    "/documents/{document_id}/study",
    tags=["study"],
    response_model=StudyResponse,
    responses={502: {"model": ErrorResponse}, **ERRORS},
    summary="Generate a summary, key points, revision notes or a quiz grounded in the document",
)
def study(document_id: uuid.UUID, body: StudyRequest, db: DB) -> StudyResponse:
    return StudyResponse(**StudyService(db).generate(document_id, body.task))


# --- Retrieval & QA ------------------------------------------------------------------
@router.post("/search", tags=["retrieval"], response_model=SearchResponse, responses=ERRORS)
def search(body: SearchRequest, db: DB) -> SearchResponse:
    result = Retriever(db).retrieve(body.query, top_k=body.top_k, mode=body.mode, document_ids=body.document_ids)
    return SearchResponse(
        query=result.query,
        mode=result.mode,
        passages=[p.__dict__ for p in result.passages],
        candidates_considered=result.candidates_considered,
        removed_duplicates=result.removed_duplicates,
    )


@router.post(
    "/ask",
    tags=["qa"],
    response_model=AnswerOut,
    responses={502: {"model": ErrorResponse}, **ERRORS},
    summary="Stateless grounded question answering",
)
def ask(body: AskRequest, db: DB) -> AnswerOut:
    r = QAService(db).answer(body.question, document_ids=body.document_ids, top_k=body.top_k)
    return AnswerOut(
        answer=r.answer,
        inference=r.inference,
        evidence_status=r.evidence_status,
        citations=r.citations,
        retrieved=r.retrieved,
        provider=r.provider,
        model=r.model,
        usage=r.usage,
        latency_ms=r.latency_ms,
    )


# --- Conversations -------------------------------------------------------------------
def _conv_out(c: Conversation, count: int) -> ConversationOut:
    return ConversationOut(
        id=c.id,
        title=c.title,
        document_ids=[d.id for d in c.documents],
        created_at=c.created_at,
        updated_at=c.updated_at,
        message_count=count,
    )


@router.post(
    "/conversations",
    tags=["conversations"],
    response_model=ConversationOut,
    status_code=status.HTTP_201_CREATED,
    responses=ERRORS,
)
def create_conversation(body: ConversationCreate, db: DB) -> ConversationOut:
    return _conv_out(ConversationService(db).create(body.title, body.document_ids), 0)


@router.get("/conversations", tags=["conversations"], response_model=Page[ConversationOut])
def list_conversations(
    db: DB, limit: Annotated[int, Query(ge=1, le=100)] = 30, offset: Annotated[int, Query(ge=0)] = 0
) -> Page[ConversationOut]:
    rows, total = ConversationService(db).list(limit=limit, offset=offset)
    return Page(items=[_conv_out(c, n) for c, n in rows], total=total, limit=limit, offset=offset)


@router.get(
    "/conversations/{conversation_id}", tags=["conversations"], response_model=ConversationDetail, responses=ERRORS
)
def get_conversation(conversation_id: uuid.UUID, db: DB) -> ConversationDetail:
    c = ConversationService(db).get(conversation_id)
    return ConversationDetail(
        **_conv_out(c, len(c.messages)).model_dump(),
        messages=[MessageOut.model_validate(m) for m in c.messages],
    )


@router.delete(
    "/conversations/{conversation_id}",
    tags=["conversations"],
    status_code=status.HTTP_204_NO_CONTENT,
    responses=ERRORS,
)
def delete_conversation(conversation_id: uuid.UUID, db: DB) -> Response:
    ConversationService(db).delete(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/conversations/{conversation_id}/messages",
    tags=["conversations"],
    response_model=ConversationTurn,
    responses={502: {"model": ErrorResponse}, **ERRORS},
    summary="Ask a (follow-up) question within a conversation",
)
def conversation_ask(conversation_id: uuid.UUID, body: ConversationAsk, db: DB) -> ConversationTurn:
    user, assistant = ConversationService(db).ask(conversation_id, body.question)
    return ConversationTurn(
        user_message=MessageOut.model_validate(user), assistant_message=MessageOut.model_validate(assistant)
    )


# --- Evaluation ----------------------------------------------------------------------
@router.post(
    "/evaluation/runs",
    tags=["evaluation"],
    response_model=EvaluationRunOut,
    status_code=status.HTTP_201_CREATED,
    responses=ERRORS,
    summary="Run the evaluation dataset synchronously and store the results",
)
def run_evaluation(body: EvaluationRunRequest, db: DB) -> EvaluationRunOut:
    run = run_and_store(db, body.dataset, body.top_k, body.include_generation)
    return EvaluationRunOut.model_validate(run)


@router.get("/evaluation/runs", tags=["evaluation"], response_model=Page[EvaluationRunSummary])
def list_evaluation_runs(
    db: DB, limit: Annotated[int, Query(ge=1, le=100)] = 20, offset: Annotated[int, Query(ge=0)] = 0
) -> Page[EvaluationRunSummary]:
    from sqlalchemy import func

    rows = (
        db.execute(select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(limit).offset(offset))
        .scalars()
        .all()
    )
    total = db.execute(select(func.count(EvaluationRun.id))).scalar_one()
    return Page(items=[EvaluationRunSummary.model_validate(r) for r in rows], total=total, limit=limit, offset=offset)


@router.get("/evaluation/runs/{run_id}", tags=["evaluation"], response_model=EvaluationRunOut, responses=ERRORS)
def get_evaluation_run(run_id: uuid.UUID, db: DB) -> EvaluationRunOut:
    from app.core.errors import NotFoundError

    run = db.get(EvaluationRun, run_id)
    if run is None:
        raise NotFoundError("Evaluation run not found.")
    return EvaluationRunOut.model_validate(run)


@router.get("/config", tags=["health"], summary="Public, non-secret runtime configuration for the UI")
def public_config() -> dict:
    s = get_settings()
    provider = get_llm_provider()
    return {
        "demo_mode": s.llm_provider == "mock",
        "llm_provider": provider.name,
        "llm_model": provider.model,
        "embedding_provider": s.embedding_provider,
        "retrieval_mode": s.retrieval_mode,
        "max_upload_mb": s.max_upload_mb,
        "allowed_extensions": [".pdf", ".txt", ".md"],
    }
