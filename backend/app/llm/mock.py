"""Deterministic, offline "demo" provider.

It does not generate free text. It answers extractively by selecting the retrieved
sentences that best cover the question and citing them, and abstains when coverage is
too low. Study tools are produced extractively from the supplied passages. This keeps
demo mode honest: every statement it returns is copied from a retrieved passage.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterator

from app.core.text import split_sentences
from app.ingestion.embeddings import _stem, tokenize
from app.llm.base import LLMRequest, LLMResponse, LLMUsage, SourceRef

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def _usage(request: LLMRequest, text: str) -> LLMUsage:
    prompt_chars = len(request.system) + sum(len(m["content"]) for m in request.messages)
    return LLMUsage(input_tokens=max(1, prompt_chars // 4), output_tokens=max(1, len(text) // 4), estimated=True)


def _is_statement(sentence: str) -> bool:
    """Skip headings, titles and footers: keep text that ends like a sentence or is long."""
    return len(sentence) >= 25 and (sentence.rstrip()[-1:] in ".!?" or len(sentence.split()) >= 8)


def _sentences(sources: list[SourceRef]) -> list[tuple[SourceRef, str]]:
    return [(s, sent) for s in sources for sent in split_sentences(s.content) if _is_statement(sent)]


class MockProvider:
    name = "mock"
    model = "extractive-demo"

    def __init__(self, min_coverage: float = 0.5):
        self.min_coverage = min_coverage

    def generate(self, request: LLMRequest) -> LLMResponse:
        handler = {
            "answer": self._answer,
            "summary": self._summary,
            "key_points": self._key_points,
            "revision_notes": self._revision_notes,
            "quiz": self._quiz,
        }[request.task]
        text = handler(request)
        return LLMResponse(text=text, provider=self.name, model=self.model, usage=_usage(request, text))

    def stream(self, request: LLMRequest) -> Iterator[str | LLMResponse]:
        """Stream the deterministic output word by word, so demo mode and tests exercise streaming."""
        response = self.generate(request)
        yield from re.findall(r"\S+\s*|\s+", response.text)
        yield response

    # -- question answering ----------------------------------------------------------
    @staticmethod
    def weighted_coverage(question: str, sentences: list[str]) -> list[float]:
        """IDF-weighted share of question terms found in each sentence.

        Rare question terms (e.g. "parental" in a policy that never mentions it) weigh
        more than terms that appear everywhere (e.g. the company name), so a sentence
        that only matches generic terms does not count as an answer.
        """
        q_terms = {_stem(t) for t in tokenize(question)}
        if not q_terms or not sentences:
            return [0.0] * len(sentences)
        sent_terms = [{_stem(t) for t in tokenize(s)} for s in sentences]
        n = len(sentences)
        df = {t: sum(1 for st in sent_terms if t in st) for t in q_terms}
        idf = {t: math.log((n + 1) / (df[t] + 1)) + 1 for t in q_terms}
        total = sum(idf.values())
        return [sum(idf[t] for t in q_terms & st) / total for st in sent_terms]

    def _answer(self, request: LLMRequest) -> str:
        pairs = _sentences(request.sources)
        coverage = self.weighted_coverage(request.question, [s for _, s in pairs])
        scored = [(cov, -i, src, sent) for i, (cov, (src, sent)) in enumerate(zip(coverage, pairs, strict=True))]
        if not scored:
            return INSUFFICIENT
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = scored[0][0]
        if best < self.min_coverage:
            return INSUFFICIENT
        chosen: list[tuple[SourceRef, str]] = []
        for cov, _, src, sent in scored:
            if cov < max(0.34, best * 0.75) or len(chosen) == 2:
                break
            if all(sent != s for _, s in chosen):
                chosen.append((src, sent))
        body = " ".join(f"{sent} [{src.label}]" for src, sent in chosen)
        return f"ANSWER: {body}\nINFERENCE: None."

    # -- study tools -----------------------------------------------------------------
    @staticmethod
    def _doc_freq(sents: list[tuple[SourceRef, str]]) -> Counter:
        counts: Counter = Counter()
        for _, sent in sents:
            counts.update({_stem(t) for t in tokenize(sent) if len(t) > 3})
        return counts

    def _central(self, request: LLMRequest, n: int) -> list[tuple[SourceRef, str]]:
        sents = _sentences(request.sources)
        freq = self._doc_freq(sents)

        def score(sent: str) -> float:
            terms = {_stem(t) for t in tokenize(sent) if len(t) > 3}
            return sum(freq[t] for t in terms) / (len(terms) + 3) if terms else 0.0

        ranked = sorted(enumerate(sents), key=lambda x: (-score(x[1][1]), x[0]))
        picked = sorted(ranked[:n], key=lambda x: x[0])  # keep document order
        return [s for _, s in picked]

    def _summary(self, request: LLMRequest) -> str:
        items = []
        for src in request.sources[:6]:
            sents = [s for s in split_sentences(src.content) if len(s) >= 40 and _is_statement(s)]
            if sents:
                items.append({"text": sents[0], "sources": [src.label]})
        return json.dumps({"items": items})

    def _key_points(self, request: LLMRequest) -> str:
        return json.dumps({"items": [{"text": s, "sources": [src.label]} for src, s in self._central(request, 6)]})

    def _revision_notes(self, request: LLMRequest) -> str:
        items = []
        for src, sent in self._central(request, 8):
            terms = [t for t in tokenize(sent) if len(t) > 4]
            heading = terms[0].capitalize() if terms else "Note"
            items.append({"text": f"{heading}: {sent}", "sources": [src.label]})
        return json.dumps({"items": items})

    def _quiz(self, request: LLMRequest) -> str:
        central = self._central(request, 12)
        all_terms: list[str] = []
        per_sentence: list[list[str]] = []
        for _, sent in central:
            words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{4,}", sent) if w.lower() not in all_terms]
            words = [w for w in words if tokenize(w)]
            per_sentence.append(words)
            all_terms.extend(w.lower() for w in words)
        questions = []
        used: set[str] = set()
        for (src, sent), words in zip(central, per_sentence, strict=True):
            candidates = sorted(words, key=lambda w: (-len(w), w))
            answer = next((w for w in candidates if w.lower() not in used), None)
            if not answer:
                continue
            distractors = sorted(
                {t for t in all_terms if t != answer.lower() and t not in sent.lower()},
                key=lambda t: (abs(len(t) - len(answer)), t),
            )[:3]
            if len(distractors) < 3:
                continue
            used.add(answer.lower())
            options = [answer.lower(), *distractors]
            seed = int(hashlib.sha256(sent.encode()).hexdigest(), 16)
            rotation = seed % 4
            options = options[rotation:] + options[:rotation]
            blanked = re.sub(rf"\b{re.escape(answer)}\b", "_____", sent, count=1)
            questions.append(
                {
                    "question": f"Fill in the blank: {blanked}",
                    "options": options,
                    "correct_index": options.index(answer.lower()),
                    "explanation": f"The source states: “{sent}”",
                    "sources": [src.label],
                }
            )
            if len(questions) == 5:
                break
        return json.dumps({"questions": questions})
