import type { Citation } from "../types/api";

interface Props {
  text: string;
  citations: Citation[];
  activeChunkId?: string | null;
  onOpen: (citation: Citation) => void;
}

const MARKER = /(\[S\d{1,3}\])/g;

/**
 * Renders answer text, turning citation markers like [S1] into highlighter-style
 * buttons that open the supporting passage. Markers with no matching citation are
 * dropped (the backend already strips them; this is a second guard).
 */
export function CitedText({ text, citations, activeChunkId, onOpen }: Props) {
  const byLabel = new Map(citations.map((c) => [c.label, c]));
  const parts = text.split(MARKER);
  return (
    <>
      {parts.map((part, i) => {
        const match = /^\[(S\d{1,3})\]$/.exec(part);
        if (!match) return <span key={i}>{part}</span>;
        const citation = byLabel.get(match[1] ?? "");
        if (!citation) return null;
        const page = citation.page_number ? `, page ${citation.page_number}` : "";
        return (
          <button
            key={i}
            type="button"
            className={`cite-mark${citation.chunk_id === activeChunkId ? " is-active" : ""}`}
            onClick={() => onOpen(citation)}
            aria-label={`Source: ${citation.document_title}${page}`}
            title={`${citation.document_title}${page}`}
          >
            {citation.page_number ? `p. ${citation.page_number}` : citation.label}
          </button>
        );
      })}
    </>
  );
}
