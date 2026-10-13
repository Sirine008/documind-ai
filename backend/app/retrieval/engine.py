"""Retrieval engine: pgvector semantic search, PostgreSQL full-text search, and a
hybrid strategy combining both with Reciprocal Rank Fusion (RRF).

All queries are built with SQLAlchemy expressions and bound parameters; user text is
never interpolated into SQL.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

from sqlalchemy import Text, cast, func, literal, select
from sqlalchemy.dialects.postgresql import REGCONFIG
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.ingestion.embeddings import Embedder, get_embedder, tokenize
from app.models import Chunk, Document, DocumentStatus
from app.retrieval.rerank import Reranker, get_reranker

logger = logging.getLogger(__name__)

Mode = Literal["hybrid", "semantic", "keyword"]
_SAFE_TERM = re.compile(r"^[a-z0-9]{2,40}$")


@dataclass(frozen=True)
class RetrievedPassage:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    page_number: int | None
    chunk_index: int
    content: str
    score: float = 0.0
    semantic_score: float | None = None
    keyword_score: float | None = None
    semantic_rank: int | None = None
    keyword_rank: int | None = None


@dataclass
class RetrievalResult:
    query: str
    mode: Mode
    passages: list[RetrievedPassage]
    max_semantic_score: float = 0.0
    keyword_hits: int = 0
    candidates_considered: int = 0
    removed_duplicates: int = 0
    debug: dict = field(default_factory=dict)


def build_tsquery_terms(query: str, max_terms: int = 24) -> list[str]:
    """Alphanumeric query terms, safe to combine into a ``to_tsquery`` OR expression."""
    terms: list[str] = []
    for tok in tokenize(query):
        if _SAFE_TERM.match(tok) and tok not in terms:
            terms.append(tok)
    return terms[:max_terms]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def deduplicate(passages: Sequence[RetrievedPassage], threshold: float) -> tuple[list[RetrievedPassage], int]:
    """Drop passages that are near-duplicates of a higher-ranked passage."""
    kept: list[RetrievedPassage] = []
    removed = 0
    for p in passages:
        if any(jaccard(p.content, k.content) >= threshold for k in kept):
            removed += 1
            continue
        kept.append(p)
    return kept, removed


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievedPassage]], k: int = 60, weights: Sequence[float] | None = None
) -> list[RetrievedPassage]:
    """Fuse ranked lists; each list contributes ``weight / (k + rank)`` per item."""
    weights = weights or [1.0] * len(ranked_lists)
    scores: dict[uuid.UUID, float] = {}
    merged: dict[uuid.UUID, RetrievedPassage] = {}
    for ranked, weight in zip(ranked_lists, weights, strict=True):
        for rank, passage in enumerate(ranked, start=1):
            scores[passage.chunk_id] = scores.get(passage.chunk_id, 0.0) + weight / (k + rank)
            existing = merged.get(passage.chunk_id)
            if existing is None:
                merged[passage.chunk_id] = passage
            else:  # combine per-method diagnostics from both lists
                merged[passage.chunk_id] = replace(
                    existing,
                    semantic_score=existing.semantic_score
                    if existing.semantic_score is not None
                    else passage.semantic_score,
                    semantic_rank=existing.semantic_rank or passage.semantic_rank,
                    keyword_score=existing.keyword_score
                    if existing.keyword_score is not None
                    else passage.keyword_score,
                    keyword_rank=existing.keyword_rank or passage.keyword_rank,
                )
    fused = [replace(p, score=scores[cid]) for cid, p in merged.items()]
    fused.sort(key=lambda p: (-p.score, p.document_title, p.chunk_index))
    return fused


class Retriever:
    def __init__(
        self,
        session: Session,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ):
        self.session = session
        self.embedder = embedder or get_embedder()
        self.settings = settings or get_settings()
        self.reranker = reranker or get_reranker(self.settings.reranker)

    def _base_select(self, *score_cols):
        return (
            select(
                Chunk.id,
                Chunk.document_id,
                Document.title,
                Chunk.page_number,
                Chunk.chunk_index,
                Chunk.content,
                *score_cols,
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.status == DocumentStatus.completed)
        )

    @staticmethod
    def _apply_scope(stmt, document_ids: Sequence[uuid.UUID] | None):
        if document_ids is not None:
            stmt = stmt.where(Chunk.document_id.in_(list(document_ids)))
        return stmt

    def semantic_search(
        self, query: str, limit: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[RetrievedPassage]:
        vector = self.embedder.embed_query(query)
        distance = Chunk.embedding.cosine_distance(vector)
        stmt = self._apply_scope(self._base_select((1 - distance).label("score")), document_ids)
        rows = self.session.execute(stmt.order_by(distance).limit(limit)).all()
        return [
            RetrievedPassage(
                chunk_id=r.id,
                document_id=r.document_id,
                document_title=r.title,
                page_number=r.page_number,
                chunk_index=r.chunk_index,
                content=r.content,
                score=float(r.score),
                semantic_score=float(r.score),
                semantic_rank=i,
            )
            for i, r in enumerate(rows, start=1)
        ]

    def keyword_search(
        self, query: str, limit: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[RetrievedPassage]:
        terms = build_tsquery_terms(query)
        if not terms:
            return []
        tsquery = func.to_tsquery(cast(literal("english"), REGCONFIG), cast(literal(" | ".join(terms)), Text))
        rank = func.ts_rank_cd(Chunk.content_tsv, tsquery, 32)
        stmt = self._apply_scope(self._base_select(rank.label("score")), document_ids).where(
            Chunk.content_tsv.op("@@")(tsquery)
        )
        rows = self.session.execute(stmt.order_by(rank.desc(), Chunk.id).limit(limit)).all()
        return [
            RetrievedPassage(
                chunk_id=r.id,
                document_id=r.document_id,
                document_title=r.title,
                page_number=r.page_number,
                chunk_index=r.chunk_index,
                content=r.content,
                score=float(r.score),
                keyword_score=float(r.score),
                keyword_rank=i,
            )
            for i, r in enumerate(rows, start=1)
        ]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: Mode | None = None,
        document_ids: Sequence[uuid.UUID] | None = None,
        rerank: bool = True,
    ) -> RetrievalResult:
        top_k = max(1, min(top_k or self.settings.retrieval_top_k, 50))
        mode = mode or self.settings.retrieval_mode
        candidates = max(top_k, self.settings.retrieval_candidates)
        if document_ids is not None and len(document_ids) == 0:
            return RetrievalResult(query=query, mode=mode, passages=[])

        semantic: list[RetrievedPassage] = []
        keyword: list[RetrievedPassage] = []
        if mode in ("hybrid", "semantic"):
            semantic = self.semantic_search(query, candidates, document_ids)
        if mode in ("hybrid", "keyword"):
            keyword = self.keyword_search(query, candidates, document_ids)

        if mode == "hybrid":
            ranked = reciprocal_rank_fusion([semantic, keyword], k=self.settings.rrf_k)
        else:
            ranked = semantic or keyword

        considered = len(ranked)
        ranked, removed = deduplicate(ranked, self.settings.dedup_jaccard_threshold)
        if rerank:
            ranked = self.reranker.rerank(query, ranked)

        return RetrievalResult(
            query=query,
            mode=mode,
            passages=ranked[:top_k],
            max_semantic_score=max((p.semantic_score or 0.0 for p in semantic), default=0.0),
            keyword_hits=len(keyword),
            candidates_considered=considered,
            removed_duplicates=removed,
        )
