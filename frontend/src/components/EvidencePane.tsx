import { useState } from "react";

import { api } from "../services/api";
import type { Chunk, Citation } from "../types/api";

interface Props {
  citation: Citation | null;
  onClose: () => void;
}

function highlight(content: string, excerpt: string) {
  const needle = excerpt.replace(/…$/, "").trim();
  const at = needle ? content.indexOf(needle) : -1;
  if (at === -1) return <>{content}</>;
  return (
    <>
      {content.slice(0, at)}
      <mark className="evidence-mark">{content.slice(at, at + needle.length)}</mark>
      {content.slice(at + needle.length)}
    </>
  );
}

export function EvidencePane({ citation, onClose }: Props) {
  const [chunk, setChunk] = useState<Chunk | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "error">("idle");

  if (!citation) {
    return (
      <aside className="evidence" aria-label="Evidence">
        <div className="evidence-empty">
          <h2>Evidence</h2>
          <p>Select a highlighted citation in an answer to see the passage it came from, with its page number.</p>
        </div>
      </aside>
    );
  }

  const loadPassage = async () => {
    setState("loading");
    try {
      setChunk(await api.getChunk(citation.chunk_id));
      setState("idle");
    } catch {
      setState("error");
    }
  };

  return (
    <aside className="evidence" aria-label="Evidence">
      <header className="evidence-head">
        <div>
          <h2>{citation.document_title}</h2>
          <p className="muted">
            {citation.page_number ? `Page ${citation.page_number}` : "Plain-text document"}, passage{" "}
            {citation.chunk_index + 1}
          </p>
        </div>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Close evidence">
          ×
        </button>
      </header>

      <blockquote className="evidence-quote">
        <mark className="evidence-mark">{citation.excerpt}</mark>
      </blockquote>

      {chunk ? (
        <section className="evidence-full">
          <h3>Full passage</h3>
          <p>{highlight(chunk.content, citation.excerpt)}</p>
        </section>
      ) : (
        <button type="button" className="button button-quiet" onClick={loadPassage} disabled={state === "loading"}>
          {state === "loading" ? "Loading passage…" : "Show full passage"}
        </button>
      )}
      {state === "error" && (
        <p className="form-error" role="alert">
          The passage could not be loaded. Its document may have been deleted.
        </p>
      )}
    </aside>
  );
}
