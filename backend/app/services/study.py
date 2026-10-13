"""Evidence-grounded study tools: summaries, key points, revision notes, quizzes."""

from __future__ import annotations

import json
import logging
import re
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import ConflictError, NotFoundError, ProviderError
from app.core.text import shorten
from app.llm import get_llm_provider
from app.llm.base import LLMProvider, LLMRequest, SourceRef
from app.llm.prompts import STUDY_SYSTEM_PROMPT, build_study_messages
from app.models import Chunk, Document, DocumentStatus
from app.services.qa import best_excerpt

logger = logging.getLogger(__name__)

MAX_STUDY_PASSAGES = 12


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences or surrounding prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ProviderError("The language model did not return valid JSON.") from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProviderError("The language model did not return valid JSON.") from exc
    if not isinstance(value, dict):
        raise ProviderError("The language model did not return a JSON object.")
    return value


def sample_evenly(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


class StudyService:
    def __init__(self, session: Session, provider: LLMProvider | None = None, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self.provider = provider or get_llm_provider()

    def generate(self, document_id: uuid.UUID, task: str) -> dict:
        doc = self.session.get(Document, document_id)
        if doc is None:
            raise NotFoundError("Document not found.")
        if doc.status != DocumentStatus.completed:
            raise ConflictError("The document has not finished indexing yet.", code="not_indexed")

        chunks = list(
            self.session.execute(
                select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
            ).scalars()
        )
        selected = sample_evenly(chunks, MAX_STUDY_PASSAGES)
        sources = [
            SourceRef(label=f"S{i}", title=doc.title, page_number=c.page_number, content=c.content)
            for i, c in enumerate(selected, start=1)
        ]
        by_label = dict(zip((s.label for s in sources), selected, strict=True))
        request = LLMRequest(
            system=STUDY_SYSTEM_PROMPT,
            messages=build_study_messages(task, doc.title, sources),
            task=task,  # type: ignore[arg-type]
            sources=sources,
            max_tokens=max(self.settings.llm_max_tokens, 1200),
            temperature=self.settings.llm_temperature,
            json_output=True,
        )
        response = self.provider.generate(request)
        data = parse_json_object(response.text)

        def cite(labels: object, text: str) -> list[dict]:
            out = []
            for label in labels if isinstance(labels, list) else []:
                chunk = by_label.get(str(label))
                if chunk is not None and all(c["label"] != label for c in out):
                    out.append(
                        {
                            "label": str(label),
                            "chunk_id": str(chunk.id),
                            "document_id": str(doc.id),
                            "document_title": doc.title,
                            "page_number": chunk.page_number,
                            "chunk_index": chunk.chunk_index,
                            "excerpt": best_excerpt(chunk.content, text),
                        }
                    )
            return out

        items, questions, dropped = [], [], 0
        if task == "quiz":
            for q in data.get("questions", []) if isinstance(data.get("questions"), list) else []:
                try:
                    options = [shorten(str(o), 200) for o in q["options"]]
                    correct = int(q["correct_index"])
                    question = str(q["question"]).strip()
                    explanation = str(q.get("explanation", "")).strip()
                except (KeyError, TypeError, ValueError):
                    dropped += 1
                    continue
                answer_text = options[correct] if 0 <= correct < len(options) else ""
                citations = cite(q.get("sources"), f"{question} {answer_text}")
                valid = question and len(options) == 4 and len(set(options)) == 4 and 0 <= correct < 4 and citations
                if not valid:
                    dropped += 1
                    continue
                questions.append(
                    {
                        "question": question,
                        "options": options,
                        "correct_index": correct,
                        "explanation": explanation,
                        "citations": citations,
                    }
                )
        else:
            for item in data.get("items", []) if isinstance(data.get("items"), list) else []:
                text = str(item.get("text", "")).strip() if isinstance(item, dict) else ""
                citations = cite(item.get("sources"), text) if text else []
                if not text or not citations:
                    dropped += 1
                    continue
                items.append({"text": text, "citations": citations})

        logger.info(
            "study_generated",
            extra={
                "document_id": str(document_id),
                "task": task,
                "items": len(items) + len(questions),
                "dropped": dropped,
            },
        )
        return {
            "document_id": document_id,
            "task": task,
            "items": items,
            "questions": questions,
            "passages_used": len(sources),
            "dropped_items": dropped,
            "provider": response.provider,
            "model": response.model,
            "usage": response.usage.as_dict(),
        }
