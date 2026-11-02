"""Persisted, optionally document-scoped conversations."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError
from app.models import Conversation, Document, Message, utcnow
from app.services.qa import AnswerResult, QAService

DEFAULT_TITLE = "New conversation"
SCOPE_EMPTY_MESSAGE = (
    "All documents this conversation was scoped to have been deleted, so there is nothing to search. "
    "Start a new conversation to ask about other documents."
)


class ConversationService:
    def __init__(self, session: Session, qa: QAService | None = None, settings: Settings | None = None):
        self.session = session
        self.settings = settings or get_settings()
        self._qa = qa

    @property
    def qa(self) -> QAService:
        if self._qa is None:
            self._qa = QAService(self.session, settings=self.settings)
        return self._qa

    def create(self, title: str | None, document_ids: list[uuid.UUID]) -> Conversation:
        docs: list[Document] = []
        if document_ids:
            ids = list(dict.fromkeys(document_ids))
            docs = list(self.session.execute(select(Document).where(Document.id.in_(ids))).scalars())
            if len(docs) != len(ids):
                found = {d.id for d in docs}
                raise NotFoundError(
                    "One or more selected documents do not exist.",
                    details={"missing": [str(i) for i in ids if i not in found]},
                )
        conv = Conversation(title=(title or "").strip()[:255] or DEFAULT_TITLE, documents=docs, is_scoped=bool(docs))
        self.session.add(conv)
        self.session.commit()
        return conv

    def list(self, *, limit: int, offset: int) -> tuple[list[tuple[Conversation, int]], int]:
        counts = (
            select(Message.conversation_id, func.count(Message.id).label("n"))
            .group_by(Message.conversation_id)
            .subquery()
        )
        rows = self.session.execute(
            select(Conversation, func.coalesce(counts.c.n, 0))
            .outerjoin(counts, counts.c.conversation_id == Conversation.id)
            .order_by(Conversation.updated_at.desc(), Conversation.id)
            .limit(limit)
            .offset(offset)
        ).all()
        total = self.session.execute(select(func.count(Conversation.id))).scalar_one()
        return [(c, int(n)) for c, n in rows], int(total)

    def get(self, conversation_id: uuid.UUID) -> Conversation:
        conv = self.session.get(Conversation, conversation_id)
        if conv is None:
            raise NotFoundError("Conversation not found.")
        return conv

    def delete(self, conversation_id: uuid.UUID) -> None:
        self.session.delete(self.get(conversation_id))
        self.session.commit()

    def ask(self, conversation_id: uuid.UUID, question: str) -> tuple[Message, Message]:
        conv, history, scope = self._load_for_question(conversation_id)
        if conv.is_scoped and not scope:
            return self._persist(conv, question, None)
        result = self.qa.answer(question, document_ids=scope or None, history=history)
        return self._persist(conv, question, result)

    def ask_stream(self, conversation_id: uuid.UUID, question: str) -> Iterator[str | tuple[Message, Message]]:
        """Like :meth:`ask`, but yields answer preview deltas before the persisted turn.

        The turn is stored only once the answer is complete, so a stream that is
        cancelled or fails part-way leaves the conversation unchanged.
        """
        conv, history, scope = self._load_for_question(conversation_id)
        if conv.is_scoped and not scope:
            yield self._persist(conv, question, None)
            return
        for item in self.qa.answer_stream(question, document_ids=scope or None, history=history):
            if isinstance(item, AnswerResult):
                yield self._persist(conv, question, item)
            else:
                yield item

    def _load_for_question(self, conversation_id: uuid.UUID) -> tuple[Conversation, list[dict], list[uuid.UUID]]:
        conv = self.get(conversation_id)
        self.session.refresh(conv)  # pick up documents deleted since the conversation was loaded
        turns = self.settings.conversation_history_turns * 2
        history = [{"role": m.role, "content": m.content} for m in conv.messages[-turns:]] if turns else []
        return conv, history, [d.id for d in conv.documents]

    def _persist(self, conv: Conversation, question: str, result: AnswerResult | None) -> tuple[Message, Message]:
        """Store the question and its answer; ``result=None`` means the scope is empty."""
        user_msg = Message(conversation_id=conv.id, role="user", content=question)
        if result is None:
            assistant_msg = Message(
                conversation_id=conv.id,
                role="assistant",
                content=SCOPE_EMPTY_MESSAGE,
                evidence_status="insufficient_evidence",
                citations=[],
                usage={},
            )
        else:
            assistant_msg = Message(
                conversation_id=conv.id,
                role="assistant",
                content=result.answer,
                inference=result.inference,
                evidence_status=result.evidence_status,
                citations=result.citations,
                usage={
                    **result.usage,
                    "provider": result.provider,
                    "model": result.model,
                    "retrieved": result.retrieved,
                },
                latency_ms=result.latency_ms,
            )
        self.session.add(user_msg)
        self.session.flush()
        assistant_msg.created_at = utcnow()  # strictly after the user message
        self.session.add(assistant_msg)
        if conv.title == DEFAULT_TITLE:
            conv.title = question[:80] + ("…" if len(question) > 80 else "")
        conv.updated_at = utcnow()
        self.session.commit()
        return user_msg, assistant_msg
