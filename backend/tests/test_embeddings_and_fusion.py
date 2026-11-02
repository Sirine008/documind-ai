import sys
import types
import uuid

import numpy as np
import pytest

from app.ingestion.embeddings import HashingEmbedder, SentenceTransformerEmbedder, embed_in_batches
from app.retrieval.engine import RetrievedPassage, build_tsquery_terms, deduplicate, reciprocal_rank_fusion
from app.retrieval.rerank import LexicalReranker, NoopReranker, term_coverage


def test_hashing_embedder_is_deterministic_normalized_and_sized():
    emb = HashingEmbedder(384)
    a, b = emb.embed(["solar panels convert light", "solar panels convert light"])
    assert a == b and len(a) == 384
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)


def test_hashing_embedder_similarity_tracks_overlap():
    emb = HashingEmbedder(384)
    q = np.array(emb.embed_query("home office stipend"))
    near = np.array(emb.embed_query("The home office stipend is 600 euros"))
    far = np.array(emb.embed_query("Coffee grind size for espresso"))
    assert q @ near > q @ far


def test_empty_text_does_not_produce_zero_vector():
    vec = HashingEmbedder(16).embed_query("")
    assert any(v != 0 for v in vec)


@pytest.mark.parametrize("method", ["get_embedding_dimension", "get_sentence_embedding_dimension"])
def test_sentence_transformer_dimension_check_supports_old_and_new_api(monkeypatch, method):
    """sentence-transformers 5.x renamed the dimension getter; both names must work."""

    class FakeModel:
        def __init__(self, name, device):
            pass

        def encode(self, texts, **kw):
            return np.ones((len(texts), 4), dtype=np.float32) / 2

    setattr(FakeModel, method, lambda self: 4)
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))
    assert SentenceTransformerEmbedder("fake", dim=4).embed_query("x") == [0.5] * 4
    with pytest.raises(RuntimeError, match="does not match"):
        SentenceTransformerEmbedder("fake", dim=8).embed_query("x")


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__(8)
        self.calls = []

    def embed(self, texts):
        self.calls.append(len(texts))
        return super().embed(texts)


def test_embed_in_batches_respects_batch_size():
    emb = CountingEmbedder()
    out = embed_in_batches(emb, [f"t{i}" for i in range(10)], batch_size=4)
    assert len(out) == 10 and emb.calls == [4, 4, 2]


def test_embed_in_batches_detects_provider_mismatch():
    class Broken(HashingEmbedder):
        def embed(self, texts):
            return []

    with pytest.raises(RuntimeError):
        embed_in_batches(Broken(8), ["a"], batch_size=4)


def _p(content, title="Doc", idx=0, **kw):
    return RetrievedPassage(uuid.uuid4(), uuid.uuid4(), title, 1, idx, content, **kw)


def test_rrf_rewards_items_ranked_by_both_methods():
    a, b, c = _p("a", idx=0), _p("b", idx=1), _p("c", idx=2)
    fused = reciprocal_rank_fusion([[a, b], [b, c]], k=60)
    assert fused[0].chunk_id == b.chunk_id
    assert {p.chunk_id for p in fused} == {a.chunk_id, b.chunk_id, c.chunk_id}


def test_rrf_merges_per_method_diagnostics():
    shared = _p("x")
    sem = [RetrievedPassage(**{**shared.__dict__, "semantic_score": 0.8, "semantic_rank": 1})]
    kw = [RetrievedPassage(**{**shared.__dict__, "keyword_score": 0.2, "keyword_rank": 3})]
    (fused,) = reciprocal_rank_fusion([sem, kw])
    assert fused.semantic_rank == 1 and fused.keyword_rank == 3
    assert fused.score == pytest.approx(2 / 61)  # rank 1 in both lists


def test_deduplicate_removes_near_duplicates_keeping_best():
    first = _p("the stipend is six hundred euros for equipment")
    dup = _p("the stipend is six hundred euros for equipment purchases")
    other = _p("lithium batteries store energy")
    kept, removed = deduplicate([first, dup, other], threshold=0.8)
    assert [p.chunk_id for p in kept] == [first.chunk_id, other.chunk_id] and removed == 1


def test_tsquery_terms_are_sanitized():
    terms = build_tsquery_terms("What's the policy? '); DROP TABLE chunks; -- & | !")
    assert all(t.isalnum() for t in terms)
    assert "drop" in terms and "policy" in terms


def test_term_coverage_and_lexical_reranker():
    assert term_coverage("office stipend", "The office stipend") == 1.0
    weak = _p("unrelated words", score=0.9)
    strong = _p("office stipend details", score=0.8)
    assert LexicalReranker().rerank("office stipend", [weak, strong])[0].chunk_id == strong.chunk_id
    assert NoopReranker().rerank("q", [weak, strong]) == [weak, strong]
