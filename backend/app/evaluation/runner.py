"""Evaluation runner: retrieval metrics and answer-generation metrics, kept separate."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import PROJECT_DIR, Settings, get_settings
from app.core.errors import NotFoundError, ValidationFailed
from app.evaluation import metrics as m
from app.models import Chunk, Document, DocumentStatus, EvaluationRun
from app.retrieval.engine import Retriever
from app.services.qa import QAService

logger = logging.getLogger(__name__)

DATASET_DIR = PROJECT_DIR / "data" / "evaluation"


def load_dataset(name: str) -> dict:
    path = (DATASET_DIR / f"{name}.json").resolve()
    if path.parent != DATASET_DIR.resolve() or not path.exists():
        raise NotFoundError(f"Evaluation dataset '{name}' was not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("questions"), list) or not data["questions"]:
        raise ValidationFailed("The evaluation dataset has no questions.")
    return data


class EvaluationRunner:
    def __init__(self, session: Session, qa: QAService | None = None, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.qa = qa or QAService(session, settings=self.settings)
        self.retriever = Retriever(session, embedder=self.qa.embedder, settings=self.settings)

    def _document_map(self, dataset: dict) -> dict[str, Document]:
        names = sorted({q["document"] for q in dataset["questions"] if q.get("document")})
        docs = self.session.execute(
            select(Document).where(Document.original_filename.in_(names), Document.status == DocumentStatus.completed)
        ).scalars()
        found = {d.original_filename: d for d in docs}
        missing = [n for n in names if n not in found]
        if missing:
            raise ValidationFailed(
                "Some evaluation documents are not indexed. Run `python scripts/seed_demo.py` first.",
                code="evaluation_documents_missing",
                details={"missing": missing},
            )
        return found

    def run(self, dataset_name: str = "default", *, top_k: int = 5, include_generation: bool = True) -> dict:
        dataset = load_dataset(dataset_name)
        docs = self._document_map(dataset)
        results: list[dict] = []
        totals = {"input_tokens": 0, "output_tokens": 0, "estimated": False, "cost_usd": 0.0, "has_cost": False}
        # Load a lazily loaded embedding model before timing, so its start-up time is not
        # reported as the first question's retrieval latency.
        self.qa.embedder.embed_query("warm-up")

        for q in dataset["questions"]:
            answerable = bool(q.get("answerable", True))
            evidence = q.get("evidence", [])
            target = docs.get(q.get("document", ""))
            row: dict = {"id": q["id"], "question": q["question"], "answerable": answerable}

            t0 = time.perf_counter()
            retrieval = self.retriever.retrieve(q["question"], top_k=top_k)
            row["retrieval_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            relevance = [
                bool(target) and p.document_id == target.id and m.is_relevant(p.content, evidence)
                for p in retrieval.passages
            ]
            row["retrieved"] = [
                {
                    "document": p.document_title,
                    "page": p.page_number,
                    "chunk_index": p.chunk_index,
                    "score": round(p.score, 4),
                    "relevant": rel,
                }
                for p, rel in zip(retrieval.passages, relevance, strict=True)
            ]
            if answerable:
                row["recall_at_k"] = m.recall_at_k(relevance, top_k)
                row["reciprocal_rank"] = m.reciprocal_rank(relevance)

            if include_generation:
                result = self.qa.answer(q["question"], top_k=top_k)
                row.update(
                    answer=result.answer,
                    inference=result.inference,
                    evidence_status=result.evidence_status,
                    latency_ms=result.latency_ms,
                    usage=result.usage,
                    citations=result.citations,
                )
                totals["input_tokens"] += int(result.usage.get("input_tokens", 0))
                totals["output_tokens"] += int(result.usage.get("output_tokens", 0))
                totals["estimated"] |= bool(result.usage.get("estimated", False))
                if "estimated_cost_usd" in result.usage:
                    totals["cost_usd"] += float(result.usage["estimated_cost_usd"])
                    totals["has_cost"] = True
                abstained = result.evidence_status in ("insufficient_evidence", "refused")
                row["abstained"] = abstained
                if answerable and not abstained:
                    cited_chunks = self._load_chunks([c["chunk_id"] for c in result.citations])
                    valid = [
                        c
                        for c in result.citations
                        if (ch := cited_chunks.get(c["chunk_id"])) is not None
                        and m.normalize_ws(c["excerpt"].rstrip("…")) in m.normalize_ws(ch.content)
                    ]
                    correct = [
                        c
                        for c in valid
                        if target
                        and cited_chunks[c["chunk_id"]].document_id == target.id
                        and m.is_relevant(cited_chunks[c["chunk_id"]].content, evidence)
                    ]
                    n = len(result.citations)
                    row["citation_validity"] = len(valid) / n if n else 0.0
                    row["citation_correctness"] = len(correct) / n if n else 0.0
                    row["groundedness"] = m.groundedness(
                        result.answer, [cited_chunks[c["chunk_id"]].content for c in valid]
                    )
                    row["answer_relevance"] = m.reference_term_recall(result.answer, q.get("reference_answer", ""))
                    row["page_correct"] = (
                        any(c["page_number"] in q.get("pages", []) for c in valid) if q.get("pages") else None
                    )
            results.append(row)

        answerable_rows = [r for r in results if r["answerable"]]
        unanswerable_rows = [r for r in results if not r["answerable"]]
        retrieval_metrics = {
            "questions": len(answerable_rows),
            f"recall_at_{top_k}": m.mean([r["recall_at_k"] for r in answerable_rows]),
            "mrr": m.mean([r["reciprocal_rank"] for r in answerable_rows]),
            "latency_ms_p50": m.percentile([r["retrieval_latency_ms"] for r in results], 50),
            "latency_ms_p95": m.percentile([r["retrieval_latency_ms"] for r in results], 95),
        }
        metrics: dict = {"retrieval": retrieval_metrics}
        if include_generation:
            answered = [r for r in answerable_rows if not r.get("abstained")]
            metrics["generation"] = {
                "answerable_questions": len(answerable_rows),
                "answered": len(answered),
                "false_abstention_rate": m.mean([1.0 if r["abstained"] else 0.0 for r in answerable_rows]),
                "unanswerable_questions": len(unanswerable_rows),
                "correct_abstention_rate": m.mean([1.0 if r["abstained"] else 0.0 for r in unanswerable_rows]),
                "citation_validity": m.mean([r["citation_validity"] for r in answered]),
                "citation_correctness": m.mean([r["citation_correctness"] for r in answered]),
                "page_accuracy": m.mean(
                    [1.0 if r["page_correct"] else 0.0 for r in answered if r.get("page_correct") is not None]
                ),
                "groundedness": m.mean([r["groundedness"] for r in answered]),
                "answer_relevance": m.mean([r["answer_relevance"] for r in answered]),
                "latency_ms_p50": m.percentile([r["latency_ms"] for r in results], 50),
                "latency_ms_p95": m.percentile([r["latency_ms"] for r in results], 95),
                "tokens": {
                    "input": totals["input_tokens"],
                    "output": totals["output_tokens"],
                    "estimated": totals["estimated"],
                },
                "estimated_cost_usd": round(totals["cost_usd"], 6) if totals["has_cost"] else None,
            }
        config = {
            "top_k": top_k,
            "retrieval_mode": self.settings.retrieval_mode,
            "reranker": self.settings.reranker,
            "embedding": self.qa.embedder.name,
            "llm_provider": self.qa.provider.name,
            "llm_model": self.qa.provider.model,
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "include_generation": include_generation,
        }
        return {"dataset": dataset_name, "config": config, "metrics": metrics, "results": results}

    def _load_chunks(self, ids: list[str]) -> dict[str, Chunk]:
        if not ids:
            return {}
        rows = self.session.execute(select(Chunk).where(Chunk.id.in_(ids))).scalars()
        return {str(c.id): c for c in rows}


def run_and_store(
    session: Session, dataset: str, top_k: int, include_generation: bool, settings: Settings | None = None
) -> EvaluationRun:
    run = EvaluationRun(dataset=dataset, status="running", config={"top_k": top_k})
    session.add(run)
    session.commit()
    try:
        output = EvaluationRunner(session, settings=settings).run(
            dataset, top_k=top_k, include_generation=include_generation
        )
    except Exception as exc:
        session.rollback()
        run.status = "failed"
        run.error_message = getattr(exc, "message", None) or "Evaluation failed."
        run.finished_at = datetime.now(UTC)
        session.commit()
        raise
    run.status = "completed"
    run.config = output["config"]
    run.metrics = output["metrics"]
    run.results = output["results"]
    run.finished_at = datetime.now(UTC)
    session.commit()
    return run


def write_results_file(output: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    payload = {"generated_at": datetime.now(UTC).isoformat(), **output}
    path = directory / f"{output['dataset']}_{stamp}.json"
    text = json.dumps(payload, indent=2, default=str)
    path.write_text(text, encoding="utf-8")
    (directory / f"{output['dataset']}_latest.json").write_text(text, encoding="utf-8")
    return path
