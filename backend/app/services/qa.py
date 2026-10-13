"""Grounded question answering (the RAG workflow)."""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError
from app.core.text import shorten, split_sentences
from app.ingestion.embeddings import Embedder, HashingEmbedder, get_embedder
from app.llm import get_llm_provider
from app.llm.base import LLMProvider, LLMRequest, SourceRef
from app.llm.prompts import (
    ANSWER_SYSTEM_PROMPT,
    build_answer_messages,
    extract_citation_labels,
    is_secret_request,
    parse_answer,
    redact_secrets,
    strip_unknown_citations,
)
from app.models import Document, DocumentStatus
from app.retrieval.engine import RetrievalResult, RetrievedPassage, Retriever
from app.retrieval.rerank import term_coverage

logger = logging.getLogger(__name__)

INSUFFICIENT_MESSAGE = "I couldn't find enough evidence in the selected documents to answer this question reliably."
REFUSAL_MESSAGE = (
    "I can't share system instructions, configuration or credentials. "
    "Ask me about the content of your documents instead."
)
NO_DOCUMENTS_MESSAGE = (
    "There are no indexed documents to search yet. Upload a document and wait for indexing to finish."
)

_FOLLOW_UP = re.compile(r"\b(it|its|that|this|they|them|those|these|he|she|there|more|why|how so)\b", re.I)


@dataclass
class AnswerResult:
    answer: str
    inference: str | None
    evidence_status: str
    citations: list[dict]
    retrieved: int
    provider: str
    model: str
    usage: dict
    latency_ms: float
    passages: list[RetrievedPassage] = field(default_factory=list)


def best_excerpt(chunk_text: str, answer_text: str, limit: int = 300) -> str:
    """The sentence(s) of a cited chunk that best overlap the answer."""
    sentences = split_sentences(chunk_text) or [chunk_text]
    ranked = sorted(enumerate(sentences), key=lambda x: (-term_coverage(x[1], answer_text), x[0]))
    best_idx = ranked[0][0]
    excerpt = sentences[best_idx]
    if len(excerpt) < 120 and best_idx + 1 < len(sentences):
        excerpt = f"{excerpt} {sentences[best_idx + 1]}"
    return shorten(excerpt, limit)


def build_retrieval_query(question: str, history: Sequence[dict[str, str]]) -> str:
    """Expand short follow-up questions with the previous user question."""
    previous = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    if previous and (len(question.split()) <= 8 or _FOLLOW_UP.search(question)):
        return f"{previous} {question}"
    return question


class QAService:
    def __init__(
        self,
        session: Session,
        provider: LLMProvider | None = None,
        embedder: Embedder | None = None,
        settings: Settings | None = None,
    ):
        self.session = session
        self.settings = settings or get_settings()
        self.provider = provider or get_llm_provider()
        self.embedder = embedder or get_embedder()
        self.retriever = Retriever(session, embedder=self.embedder, settings=self.settings)

    def _resolve_scope(self, document_ids: Sequence[uuid.UUID] | None) -> list[uuid.UUID] | None:
        if not document_ids:
            return None
        ids = list(dict.fromkeys(document_ids))
        found = set(self.session.execute(select(Document.id).where(Document.id.in_(ids))).scalars())
        missing = [str(i) for i in ids if i not in found]
        if missing:
            raise NotFoundError("One or more selected documents do not exist.", details={"missing": missing})
        return ids

    def _has_indexed(self, scope: list[uuid.UUID] | None) -> bool:
        stmt = select(Document.id).where(Document.status == DocumentStatus.completed).limit(1)
        if scope is not None:
            stmt = stmt.where(Document.id.in_(scope))
        return self.session.execute(stmt).first() is not None

    def evidence_is_sufficient(self, retrieval: RetrievalResult, *questions: str) -> bool:
        """Cheap pre-generation gate that avoids calling the LLM with irrelevant passages."""
        if not retrieval.passages:
            return False
        coverage = max(term_coverage(q, p.content) for q in questions for p in retrieval.passages)
        if coverage >= 0.34:
            return True
        semantic_model = not isinstance(self.embedder, HashingEmbedder)
        return semantic_model and retrieval.max_semantic_score >= self.settings.min_semantic_score

    def _result(self, start: float, status: str, answer: str, **kw) -> AnswerResult:
        return AnswerResult(
            answer=answer,
            inference=kw.get("inference"),
            evidence_status=status,
            citations=kw.get("citations", []),
            retrieved=kw.get("retrieved", 0),
            provider=kw.get("provider", self.provider.name),
            model=kw.get("model", self.provider.model),
            usage=kw.get("usage", {}),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            passages=kw.get("passages", []),
        )

    def answer(
        self,
        question: str,
        *,
        document_ids: Sequence[uuid.UUID] | None = None,
        history: Sequence[dict[str, str]] = (),
        top_k: int | None = None,
    ) -> AnswerResult:
        start = time.perf_counter()
        question = " ".join(question.split())[: self.settings.max_question_chars]
        if is_secret_request(question):
            logger.info("qa_refused_secret_request")
            return self._result(start, "refused", REFUSAL_MESSAGE)

        scope = self._resolve_scope(document_ids)
        if not self._has_indexed(scope):
            return self._result(start, "insufficient_evidence", NO_DOCUMENTS_MESSAGE)

        query = build_retrieval_query(question, history)
        retrieval = self.retriever.retrieve(query, top_k=top_k, document_ids=scope)
        if not self.evidence_is_sufficient(retrieval, question, query):
            return self._result(
                start,
                "insufficient_evidence",
                INSUFFICIENT_MESSAGE,
                retrieved=len(retrieval.passages),
                passages=retrieval.passages,
            )

        sources = [
            SourceRef(label=f"S{i}", title=p.document_title, page_number=p.page_number, content=p.content)
            for i, p in enumerate(retrieval.passages, start=1)
        ]
        by_label = {s.label: p for s, p in zip(sources, retrieval.passages, strict=True)}
        request = LLMRequest(
            system=ANSWER_SYSTEM_PROMPT,
            messages=build_answer_messages(question, sources, list(history)),
            task="answer",
            sources=sources,
            question=query,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
        )
        response = self.provider.generate(request)
        usage = response.usage.as_dict()
        if self.settings.llm_cost_per_1k_input is not None and self.settings.llm_cost_per_1k_output is not None:
            usage["estimated_cost_usd"] = round(
                usage["input_tokens"] / 1000 * self.settings.llm_cost_per_1k_input
                + usage["output_tokens"] / 1000 * self.settings.llm_cost_per_1k_output,
                6,
            )

        secrets = [self.settings.llm_api_key.get_secret_value()] if self.settings.llm_api_key else []
        answer, inference, insufficient = parse_answer(redact_secrets(response.text, secrets))
        common = {
            "retrieved": len(retrieval.passages),
            "provider": response.provider,
            "model": response.model,
            "usage": usage,
            "passages": retrieval.passages,
        }
        if insufficient:
            return self._result(start, "insufficient_evidence", INSUFFICIENT_MESSAGE, **common)

        valid_labels = [lbl for lbl in extract_citation_labels(answer) if lbl in by_label]
        answer = strip_unknown_citations(answer, set(valid_labels)).strip()
        if inference:
            inference = strip_unknown_citations(inference, set(valid_labels)).strip() or None
        citations = []
        for label in valid_labels:
            p = by_label[label]
            citations.append(
                {
                    "label": label,
                    "chunk_id": str(p.chunk_id),
                    "document_id": str(p.document_id),
                    "document_title": p.document_title,
                    "page_number": p.page_number,
                    "chunk_index": p.chunk_index,
                    "excerpt": best_excerpt(p.content, answer),
                }
            )
        status = "supported" if citations else "unverified"
        logger.info(
            "qa_answered",
            extra={"status": status, "citations": len(citations), "retrieved": len(retrieval.passages)},
        )
        return self._result(start, status, answer, inference=inference, citations=citations, **common)
