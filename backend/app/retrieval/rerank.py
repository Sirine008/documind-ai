"""Optional reranking behind a small interface.

``LexicalReranker`` blends the first-stage score with query-term coverage. It is a
lightweight heuristic, not a learned cross-encoder; a cross-encoder can be added by
implementing the same ``rerank`` method.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from app.ingestion.embeddings import _stem, tokenize

if TYPE_CHECKING:
    from app.retrieval.engine import RetrievedPassage


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, passages: Sequence[RetrievedPassage]) -> list[RetrievedPassage]: ...


def term_coverage(query: str, text: str) -> float:
    """Fraction of distinct (stemmed) query terms that appear in ``text``."""
    q = {_stem(t) for t in tokenize(query)}
    if not q:
        return 0.0
    t = {_stem(x) for x in tokenize(text)}
    return len(q & t) / len(q)


class NoopReranker:
    name = "none"

    def rerank(self, query: str, passages: Sequence[RetrievedPassage]) -> list[RetrievedPassage]:
        return list(passages)


class LexicalReranker:
    name = "lexical"

    def __init__(self, coverage_weight: float = 0.35):
        self.coverage_weight = coverage_weight

    def rerank(self, query: str, passages: Sequence[RetrievedPassage]) -> list[RetrievedPassage]:
        if not passages:
            return []
        top = max(p.score for p in passages) or 1.0
        scored = []
        for p in passages:
            blended = (1 - self.coverage_weight) * (p.score / top) + self.coverage_weight * term_coverage(
                query, p.content
            )
            scored.append(replace(p, score=round(blended, 6)))
        scored.sort(key=lambda p: (-p.score, p.document_title, p.chunk_index))
        return scored


def get_reranker(name: str) -> Reranker:
    return LexicalReranker() if name == "lexical" else NoopReranker()
