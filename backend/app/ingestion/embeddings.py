"""Embedding providers.

* ``SentenceTransformerEmbedder`` — the default; runs a local Sentence Transformers
  model (``all-MiniLM-L6-v2``, 384 dimensions). Requires ``requirements-ml.txt`` and a
  one-time model download from Hugging Face.
* ``HashingEmbedder`` — a deterministic, dependency-free feature-hashing embedder.
  It captures lexical overlap only (no real semantics). It exists so tests, CI and
  offline environments can run the whole pipeline without downloading a model.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import re
import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "as",
        "at",
        "by",
        "from",
        "what",
        "which",
        "who",
        "how",
        "when",
        "where",
        "why",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "i",
        "you",
        "we",
        "they",
    }
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _stem(token: str) -> str:
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


class HashingEmbedder:
    """Signed feature hashing over stemmed unigrams and bigrams, L2-normalized."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def _vector(self, text: str) -> list[float]:
        tokens = [_stem(t) for t in tokenize(text)]
        vec = np.zeros(self.dim, dtype=np.float64)
        features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        for feature in features:
            idx, sign = self._bucket(feature)
            vec[idx] += sign * (0.5 if "_" in feature else 1.0)
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec[0] = 1e-6  # pgvector cosine distance is undefined for the zero vector
            norm = 1e-6
        return (vec / norm).astype(np.float32).tolist()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dim: int, batch_size: int = 32):
        self.name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:  # pragma: no cover - depends on optional deps
                        raise RuntimeError(
                            "sentence-transformers is not installed. Install backend/requirements-ml.txt "
                            "or set DOCUMIND_EMBEDDING_PROVIDER=hashing."
                        ) from exc
                    logger.info("loading_embedding_model", extra={"model": self.name})
                    model = SentenceTransformer(self.name, device="cpu")
                    actual = model.get_sentence_embedding_dimension()
                    if actual != self.dim:
                        raise RuntimeError(
                            f"Model dimension {actual} does not match DOCUMIND_EMBEDDING_DIM={self.dim}."
                        )
                    self._model = model
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return [v.astype(np.float32).tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    if settings.embedding_provider == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim, settings.embedding_batch_size)


def embed_in_batches(embedder: Embedder, texts: Sequence[str], batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out = embedder.embed(batch)
        if len(out) != len(batch):
            raise RuntimeError("Embedding provider returned a different number of vectors than inputs.")
        vectors.extend(out)
    return vectors
