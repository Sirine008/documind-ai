import { useEffect, useRef, useState } from "react";

import { CitedText } from "../components/CitedText";
import type { Citation, ConversationDetail, DocumentInfo, Message } from "../types/api";

interface Props {
  conversation: ConversationDetail | null;
  documentsById: Map<string, DocumentInfo>;
  selectedIds: Set<string>;
  readyCount: number;
  asking: string | null;
  demoMode: boolean;
  activeChunkId: string | null;
  onAsk: (question: string) => void;
  onNewConversation: () => void;
  onOpenCitation: (citation: Citation) => void;
}

const STATUS_NOTE: Record<string, string> = {
  insufficient_evidence: "Not enough evidence in the documents",
  unverified: "No valid citation. Treat this answer with caution",
  refused: "Request declined",
};

function scopeLabel(ids: string[], docs: Map<string, DocumentInfo>) {
  if (!ids.length) return "Searching all documents";
  const names = ids.map((id) => docs.get(id)?.title ?? "a deleted document");
  return `Searching ${names.length === 1 ? names[0] : `${names.length} documents: ${names.join(", ")}`}`;
}

function AssistantMessage({
  message,
  activeChunkId,
  onOpenCitation,
}: {
  message: Message;
  activeChunkId: string | null;
  onOpenCitation: (c: Citation) => void;
}) {
  const status = message.evidence_status ?? "supported";
  const sources = [...new Map(message.citations.map((c) => [c.document_id + c.page_number, c])).values()];
  return (
    <article className={`answer answer-${status}`}>
      {status !== "supported" && <p className="answer-status">{STATUS_NOTE[status]}</p>}
      <p className="answer-text">
        <CitedText
          text={message.content}
          citations={message.citations}
          activeChunkId={activeChunkId}
          onOpen={onOpenCitation}
        />
      </p>
      {message.inference && (
        <div className="inference">
          <h3>Model interpretation</h3>
          <p>{message.inference}</p>
          <p className="muted small">This part is the model's reasoning, not a statement found in the documents.</p>
        </div>
      )}
      {sources.length > 0 && (
        <footer className="answer-sources">
          <span className="muted small">Sources</span>
          {sources.map((c) => (
            <button key={c.chunk_id} type="button" className="source-pill" onClick={() => onOpenCitation(c)}>
              {c.document_title}
              {c.page_number ? `, p. ${c.page_number}` : ""}
            </button>
          ))}
        </footer>
      )}
    </article>
  );
}

export function ChatPage(props: Props) {
  const {
    conversation, documentsById, selectedIds, readyCount, asking, demoMode, activeChunkId,
    onAsk, onNewConversation, onOpenCitation,
  } = props;
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const messages = conversation?.messages ?? [];

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: "end", behavior: "smooth" });
  }, [messages.length, asking]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const question = draft.trim();
    if (!question || asking) return;
    onAsk(question);
    setDraft("");
  };

  const scopeIds = conversation ? conversation.document_ids : [...selectedIds];

  return (
    <section className="chat" aria-label="Conversation">
      <header className="chat-head">
        <div>
          <h1>{conversation?.title ?? "New conversation"}</h1>
          <p className="muted small">{scopeLabel(scopeIds, documentsById)}</p>
        </div>
        {conversation && (
          <button type="button" className="button button-quiet" onClick={onNewConversation}>
            New conversation
          </button>
        )}
      </header>

      <div className="chat-scroll">
        {messages.length === 0 && !asking && (
          <div className="chat-empty">
            <h2>Ask about your documents</h2>
            {readyCount === 0 ? (
              <p>Upload a document on the left. Once it shows Ready, you can ask questions about it here.</p>
            ) : (
              <p>
                Answers quote only what the documents say. Each highlighted mark links to the passage and page it
                came from. If the documents don't cover a question, DocuMind says so instead of guessing.
              </p>
            )}
            {demoMode && (
              <p className="demo-note">
                Demo mode is on: answers are sentences copied from the best-matching passages, without a language
                model. Set an LLM provider in <code>.env</code> for written answers.
              </p>
            )}
          </div>
        )}

        <ol className="messages">
          {messages.map((m) =>
            m.role === "user" ? (
              <li key={m.id} className="message message-user">
                <p>{m.content}</p>
              </li>
            ) : (
              <li key={m.id} className="message message-assistant">
                <AssistantMessage message={m} activeChunkId={activeChunkId} onOpenCitation={onOpenCitation} />
              </li>
            ),
          )}
          {asking && (
            <>
              <li className="message message-user">
                <p>{asking}</p>
              </li>
              <li className="message message-assistant" aria-live="polite">
                <p className="thinking">
                  <span className="spinner" aria-hidden="true" /> Searching the documents…
                </p>
              </li>
            </>
          )}
        </ol>
        <div ref={endRef} />
      </div>

      <form className="composer" onSubmit={submit}>
        <label htmlFor="question" className="visually-hidden">
          Your question
        </label>
        <textarea
          id="question"
          value={draft}
          rows={2}
          maxLength={2000}
          placeholder={readyCount ? "Ask a question about your documents" : "Upload a document to start asking"}
          disabled={!readyCount}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) submit(e);
          }}
        />
        <button type="submit" className="button button-primary" disabled={!draft.trim() || !!asking || !readyCount}>
          Ask
        </button>
      </form>
    </section>
  );
}
