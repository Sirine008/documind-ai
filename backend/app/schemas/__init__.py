"""Pydantic request and response schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T")

EvidenceStatus = Literal["supported", "insufficient_evidence", "unverified", "refused"]


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


# --- Documents -------------------------------------------------------------------
class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    original_filename: str
    content_type: str
    size_bytes: int
    sha256: str
    status: str
    error_code: str | None
    error_message: str | None
    page_count: int | None
    chunk_count: int
    pages_without_text: list[int]
    attempts: int
    embedding_model: str | None
    created_at: datetime
    updated_at: datetime
    processing_started_at: datetime | None
    processed_at: datetime | None

    @field_validator("status", mode="before")
    @classmethod
    def _enum_value(cls, v: object) -> object:
        return getattr(v, "value", v)


class UploadResponse(BaseModel):
    document: DocumentOut
    duplicate: bool = False
    message: str


class DocumentStatusOut(BaseModel):
    id: uuid.UUID
    status: str
    chunk_count: int
    attempts: int
    error_code: str | None
    error_message: str | None
    pages_without_text: list[int]
    processed_at: datetime | None


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    page_number: int | None
    content: str


# --- Retrieval -------------------------------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[uuid.UUID] | None = None
    top_k: int = Field(default=6, ge=1, le=50)
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("query must not be blank")
        return v


class PassageOut(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    page_number: int | None
    chunk_index: int
    content: str
    score: float
    semantic_score: float | None
    keyword_score: float | None
    semantic_rank: int | None
    keyword_rank: int | None


class SearchResponse(BaseModel):
    query: str
    mode: str
    passages: list[PassageOut]
    candidates_considered: int
    removed_duplicates: int


# --- Question answering ---------------------------------------------------------
class Citation(BaseModel):
    label: str
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    page_number: int | None
    chunk_index: int
    excerpt: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[uuid.UUID] | None = Field(
        default=None, description="Restrict retrieval to these documents. Omit to search all documents."
    )
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _normalize(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("question must not be blank")
        return v


class AnswerOut(BaseModel):
    answer: str
    inference: str | None
    evidence_status: EvidenceStatus
    citations: list[Citation]
    retrieved: int
    provider: str
    model: str
    usage: dict
    latency_ms: float


# --- Conversations --------------------------------------------------------------
class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    document_ids: list[uuid.UUID] = Field(
        default_factory=list, description="Documents this conversation is scoped to. Empty = all documents."
    )


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    inference: str | None
    evidence_status: str | None
    citations: list[dict]
    usage: dict
    latency_ms: float | None
    created_at: datetime


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str
    document_ids: list[uuid.UUID]
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class ConversationAsk(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def _normalize(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("question must not be blank")
        return v


class ConversationTurn(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut


# --- Study tools -------------------------------------------------------------------
StudyTask = Literal["summary", "key_points", "revision_notes", "quiz"]


class StudyRequest(BaseModel):
    task: StudyTask


class StudyItem(BaseModel):
    text: str
    citations: list[Citation]


class QuizQuestion(BaseModel):
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    citations: list[Citation]


class StudyResponse(BaseModel):
    document_id: uuid.UUID
    task: StudyTask
    items: list[StudyItem] = Field(default_factory=list)
    questions: list[QuizQuestion] = Field(default_factory=list)
    passages_used: int
    dropped_items: int = Field(description="Generated items discarded because they cited no valid source.")
    provider: str
    model: str
    usage: dict


# --- Evaluation ------------------------------------------------------------------
class EvaluationRunRequest(BaseModel):
    dataset: str = Field(default="default", pattern=r"^[a-zA-Z0-9_\-]{1,64}$")
    top_k: int = Field(default=5, ge=1, le=20)
    include_generation: bool = True


class EvaluationRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset: str
    status: str
    config: dict
    metrics: dict
    results: list[dict]
    error_message: str | None
    created_at: datetime
    finished_at: datetime | None


class EvaluationRunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset: str
    status: str
    metrics: dict
    created_at: datetime
    finished_at: datetime | None
