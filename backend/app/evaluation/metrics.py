"""Evaluation metrics. Pure functions so they can be unit-tested in isolation.

The answer-quality metrics here are transparent lexical heuristics, not an LLM judge.
They are useful for regression tracking but are not ground truth; see
``docs/evaluation.md`` for the manual review procedure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.ingestion.embeddings import _stem, tokenize


def _terms(text: str) -> set[str]:
    return {_stem(t) for t in tokenize(text) if len(t) > 1}


def normalize_ws(text: str) -> str:
    return " ".join(text.lower().split())


def is_relevant(passage_text: str, evidence: Sequence[str]) -> bool:
    """A passage is relevant when it contains every evidence phrase (case/space-insensitive)."""
    body = normalize_ws(passage_text)
    return bool(evidence) and all(normalize_ws(e) in body for e in evidence)


def recall_at_k(relevance: Sequence[bool], k: int) -> float:
    """1.0 if any of the top-k results is relevant (hit rate / binary recall@k)."""
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    for i, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def groundedness(answer: str, cited_texts: Sequence[str]) -> float:
    """Share of the answer's content terms that also appear in the cited passages."""
    answer_terms = _terms(answer)
    if not answer_terms:
        return 0.0
    support = set().union(*(_terms(t) for t in cited_texts)) if cited_texts else set()
    return len(answer_terms & support) / len(answer_terms)


def reference_term_recall(answer: str, reference: str) -> float:
    """Share of the reference answer's content terms present in the generated answer."""
    ref = _terms(reference)
    if not ref:
        return 0.0
    return len(ref & _terms(answer)) / len(ref)


def mean(values: Sequence[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def percentile(values: Sequence[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = max(0, math.ceil(p / 100 * len(vals)) - 1)
    return round(vals[idx], 1)
