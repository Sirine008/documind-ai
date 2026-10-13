import { useRef, useState } from "react";

import type { Conversation, DocumentInfo } from "../types/api";

interface Props {
  documents: DocumentInfo[];
  conversations: Conversation[];
  loading: boolean;
  selectedIds: Set<string>;
  activeConversationId: string | null;
  maxUploadMb: number;
  uploads: string[];
  onToggleSelect: (id: string) => void;
  onUpload: (files: File[]) => void;
  onDelete: (doc: DocumentInfo) => void;
  onRetry: (doc: DocumentInfo) => void;
  onOpenConversation: (id: string) => void;
  onDeleteConversation: (conv: Conversation) => void;
}

const STATUS_TEXT: Record<DocumentInfo["status"], string> = {
  pending: "Queued",
  processing: "Indexing…",
  completed: "Ready",
  failed: "Failed",
};

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function describe(doc: DocumentInfo) {
  const parts = [doc.content_type === "pdf" ? `PDF, ${doc.page_count ?? "?"} pages` : "Text", formatSize(doc.size_bytes)];
  if (doc.status === "completed") parts.push(`${doc.chunk_count} passages`);
  return parts.join(", ");
}

export function Library(props: Props) {
  const {
    documents, conversations, loading, selectedIds, activeConversationId, maxUploadMb, uploads,
    onToggleSelect, onUpload, onDelete, onRetry, onOpenConversation, onDeleteConversation,
  } = props;
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFiles = (list: FileList | null) => {
    if (list && list.length) onUpload(Array.from(list));
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <nav className="library" aria-label="Library">
      <div
        className={`dropzone${dragging ? " is-dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          id="file-input"
          type="file"
          accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown"
          multiple
          className="visually-hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <label htmlFor="file-input" className="button button-primary">
          Upload documents
        </label>
        <p className="muted small">PDF, TXT or Markdown, up to {maxUploadMb} MB. Or drop files here.</p>
        {uploads.length > 0 && (
          <ul className="upload-progress" aria-live="polite">
            {uploads.map((name) => (
              <li key={name}>
                <span className="spinner" aria-hidden="true" /> Uploading {name}
              </li>
            ))}
          </ul>
        )}
      </div>

      <section className="library-section">
        <h2>Documents</h2>
        {loading && documents.length === 0 ? (
          <p className="muted small">Loading documents…</p>
        ) : documents.length === 0 ? (
          <p className="empty-hint">
            Your library is empty. Upload a document, or run <code>python scripts/seed_demo.py</code> to load the
            sample set.
          </p>
        ) : (
          <>
            <p className="muted small">Tick documents to limit a new conversation to them.</p>
            <ul className="doc-list">
              {documents.map((doc) => (
                <li key={doc.id} className={`doc-row status-${doc.status}`}>
                  <input
                    type="checkbox"
                    id={`sel-${doc.id}`}
                    checked={selectedIds.has(doc.id)}
                    disabled={doc.status !== "completed"}
                    onChange={() => onToggleSelect(doc.id)}
                  />
                  <label htmlFor={`sel-${doc.id}`} className="doc-main">
                    <span className="doc-title">{doc.title}</span>
                    <span className="doc-meta">{describe(doc)}</span>
                    <span className={`doc-status status-text-${doc.status}`}>
                      {(doc.status === "pending" || doc.status === "processing") && (
                        <span className="spinner" aria-hidden="true" />
                      )}
                      {STATUS_TEXT[doc.status]}
                      {doc.status === "completed" && doc.pages_without_text.length > 0 &&
                        `. Pages without text: ${doc.pages_without_text.join(", ")}`}
                    </span>
                    {doc.status === "failed" && doc.error_message && (
                      <span className="doc-error">{doc.error_message}</span>
                    )}
                  </label>
                  <div className="doc-actions">
                    {doc.status === "failed" && (
                      <button type="button" className="text-button" onClick={() => onRetry(doc)}>
                        Retry
                      </button>
                    )}
                    <button
                      type="button"
                      className="text-button text-danger"
                      onClick={() => onDelete(doc)}
                      aria-label={`Delete ${doc.title}`}
                    >
                      Delete
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          </>
        )}
      </section>

      <section className="library-section">
        <h2>Conversations</h2>
        {conversations.length === 0 ? (
          <p className="empty-hint">No conversations yet. Ask a question to start one.</p>
        ) : (
          <ul className="conv-list">
            {conversations.map((c) => (
              <li key={c.id} className={c.id === activeConversationId ? "is-active" : undefined}>
                <button type="button" className="conv-open" onClick={() => onOpenConversation(c.id)}>
                  <span className="conv-title">{c.title}</span>
                  <span className="doc-meta">
                    {c.document_ids.length ? `${c.document_ids.length} selected documents` : "All documents"}
                  </span>
                </button>
                <button
                  type="button"
                  className="icon-button"
                  onClick={() => onDeleteConversation(c)}
                  aria-label={`Delete conversation ${c.title}`}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </nav>
  );
}
