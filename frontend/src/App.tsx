import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { EvidencePane } from "./components/EvidencePane";
import { Library } from "./components/Library";
import { Toasts } from "./components/Toasts";
import { useToasts } from "./hooks/useToasts";
import { ChatPage } from "./pages/ChatPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { StudyPage } from "./pages/StudyPage";
import { ApiError, StreamUnavailableError, api } from "./services/api";
import type {
  Citation,
  Conversation,
  ConversationDetail,
  ConversationTurn,
  DocumentInfo,
  PublicConfig,
} from "./types/api";

type View = "chat" | "study" | "evaluation";
type Pane = "library" | "main" | "evidence";

const VIEWS: { id: View; label: string }[] = [
  { id: "chat", label: "Chat" },
  { id: "study", label: "Study" },
  { id: "evaluation", label: "Evaluation" },
];

const message = (err: unknown, fallback: string) => (err instanceof ApiError ? err.message : fallback);

export default function App() {
  const { toasts, notify, dismiss } = useToasts();
  const [config, setConfig] = useState<PublicConfig | null>(null);
  const [apiDown, setApiDown] = useState(false);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [docsLoading, setDocsLoading] = useState(true);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversation, setConversation] = useState<ConversationDetail | null>(null);
  const [selection, setSelection] = useState<Set<string>>(new Set());
  const [asking, setAsking] = useState<string | null>(null);
  const [streamingText, setStreamingText] = useState<string | null>(null);
  const askAbort = useRef<AbortController | null>(null);
  const [uploads, setUploads] = useState<string[]>([]);
  const [evidence, setEvidence] = useState<Citation | null>(null);
  const [view, setView] = useState<View>("chat");
  const [pane, setPane] = useState<Pane>("main");

  const refreshDocuments = useCallback(
    () =>
      api
        .listDocuments()
        .then((page) => {
          setDocuments(page.items);
          setApiDown(false);
        })
        .catch((err: unknown) => {
          if (err instanceof ApiError && err.status === 0) setApiDown(true);
        })
        .finally(() => setDocsLoading(false)),
    [],
  );

  const refreshConversations = useCallback(
    () =>
      api
        .listConversations()
        .then((page) => setConversations(page.items))
        .catch(() => {
          /* connectivity problems are surfaced through the documents request */
        }),
    [],
  );

  useEffect(() => {
    api.config().then(setConfig).catch(() => setApiDown(true));
    void refreshDocuments();
    void refreshConversations();
  }, [refreshDocuments, refreshConversations]);

  // Poll while any document is still being indexed.
  const indexing = documents.some((d) => d.status === "pending" || d.status === "processing");
  useEffect(() => {
    if (!indexing) return;
    const timer = window.setInterval(() => void refreshDocuments(), 2000);
    return () => window.clearInterval(timer);
  }, [indexing, refreshDocuments]);

  // Only documents that still exist and are indexed count as selected.
  const selectedIds = useMemo(() => {
    const ready = new Set(documents.filter((d) => d.status === "completed").map((d) => d.id));
    return new Set([...selection].filter((id) => ready.has(id)));
  }, [documents, selection]);

  const documentsById = useMemo(() => new Map(documents.map((d) => [d.id, d])), [documents]);
  const readyCount = documents.filter((d) => d.status === "completed").length;

  const upload = async (files: File[]) => {
    for (const file of files) {
      setUploads((u) => [...u, file.name]);
      try {
        const res = await api.uploadDocument(file);
        notify(res.duplicate ? "info" : "success", res.duplicate ? `${file.name} is already in your library.` : `${file.name} uploaded. Indexing has started.`);
      } catch (err) {
        notify("error", `${file.name}: ${message(err, "upload failed.")}`);
      } finally {
        setUploads((u) => u.filter((n) => n !== file.name));
      }
    }
    await refreshDocuments();
  };

  const deleteDocument = async (doc: DocumentInfo) => {
    if (!window.confirm(`Delete “${doc.title}”? Its passages are removed from search, and citations to it stop working.`)) return;
    try {
      await api.deleteDocument(doc.id);
      notify("success", `Deleted ${doc.title}.`);
      if (evidence?.document_id === doc.id) setEvidence(null);
      await refreshDocuments();
    } catch (err) {
      notify("error", message(err, "The document could not be deleted."));
    }
  };

  const retryDocument = async (doc: DocumentInfo) => {
    try {
      await api.retryDocument(doc.id);
      notify("info", `Retrying ${doc.title}.`);
      await refreshDocuments();
    } catch (err) {
      notify("error", message(err, "The document could not be retried."));
    }
  };

  const openConversation = async (id: string) => {
    try {
      setConversation(await api.getConversation(id));
      setEvidence(null);
      setView("chat");
      setPane("main");
    } catch (err) {
      notify("error", message(err, "The conversation could not be opened."));
    }
  };

  const deleteConversation = async (conv: Conversation) => {
    if (!window.confirm(`Delete the conversation “${conv.title}”?`)) return;
    try {
      await api.deleteConversation(conv.id);
      if (conversation?.id === conv.id) setConversation(null);
      await refreshConversations();
    } catch (err) {
      notify("error", message(err, "The conversation could not be deleted."));
    }
  };

  const ask = async (question: string) => {
    setAsking(question);
    setStreamingText("");
    const controller = new AbortController();
    askAbort.current = controller;
    try {
      let current = conversation;
      if (!current) {
        const created = await api.createConversation([...selectedIds]);
        current = { ...created, messages: [] };
        setConversation(current);
      }
      let turn: ConversationTurn;
      try {
        turn = await api.askStream(current.id, question, {
          signal: controller.signal,
          onToken: (text) => setStreamingText((shown) => (shown ?? "") + text),
        });
      } catch (err) {
        if (!(err instanceof StreamUnavailableError)) throw err;
        // Nothing was stored for an incomplete stream, so asking again without streaming is safe.
        setStreamingText("");
        turn = await api.ask(current.id, question);
      }
      setConversation((c) =>
        c && c.id === current.id ? { ...c, messages: [...c.messages, turn.user_message, turn.assistant_message] } : c,
      );
      const fresh = await api.getConversation(current.id);
      setConversation(fresh);
      void refreshConversations();
    } catch (err) {
      if (controller.signal.aborted) notify("info", "Stopped. The unfinished answer was not saved.");
      else notify("error", message(err, "The question could not be answered."));
    } finally {
      askAbort.current = null;
      setAsking(null);
      setStreamingText(null);
    }
  };

  const cancelAsk = () => askAbort.current?.abort();

  const openCitation = (c: Citation) => {
    setEvidence(c);
    setPane("evidence");
  };

  return (
    <div className="app" data-pane={pane}>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">DocuMind</span>
        </div>
        <nav className="view-tabs" aria-label="Views">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              type="button"
              className={`view-tab${view === v.id ? " is-active" : ""}`}
              aria-current={view === v.id ? "page" : undefined}
              onClick={() => {
                setView(v.id);
                setPane("main");
              }}
            >
              {v.label}
            </button>
          ))}
        </nav>
        {config?.demo_mode && <span className="demo-badge">Demo mode</span>}
      </header>

      {apiDown && (
        <div className="banner" role="alert">
          Can't reach the DocuMind API. Start the backend (see the README), then reload this page.
        </div>
      )}

      <div className="panes">
        <div className="pane pane-library">
          <Library
            documents={documents}
            conversations={conversations}
            loading={docsLoading}
            selectedIds={selectedIds}
            activeConversationId={conversation?.id ?? null}
            maxUploadMb={config?.max_upload_mb ?? 20}
            uploads={uploads}
            onToggleSelect={(id) =>
              setSelection((s) => {
                const next = new Set(s);
                if (next.has(id)) next.delete(id);
                else next.add(id);
                return next;
              })
            }
            onUpload={upload}
            onDelete={deleteDocument}
            onRetry={retryDocument}
            onOpenConversation={openConversation}
            onDeleteConversation={deleteConversation}
          />
        </div>
        <main className="pane pane-main">
          {view === "chat" && (
            <ChatPage
              conversation={conversation}
              documentsById={documentsById}
              selectedIds={selectedIds}
              readyCount={readyCount}
              asking={asking}
              streamingText={streamingText}
              demoMode={!!config?.demo_mode}
              activeChunkId={evidence?.chunk_id ?? null}
              onAsk={ask}
              onCancelAsk={cancelAsk}
              onNewConversation={() => {
                setConversation(null);
                setEvidence(null);
              }}
              onOpenCitation={openCitation}
            />
          )}
          {view === "study" && <StudyPage documents={documents} onOpenCitation={openCitation} />}
          {view === "evaluation" && <EvaluationPage notify={notify} />}
        </main>
        <div className="pane pane-evidence">
          <EvidencePane
            key={evidence?.chunk_id ?? "none"}
            citation={evidence}
            onClose={() => {
              setEvidence(null);
              setPane("main");
            }}
          />
        </div>
      </div>

      <nav className="pane-switch" aria-label="Panels">
        {(["library", "main", "evidence"] as Pane[]).map((p) => (
          <button
            key={p}
            type="button"
            className={pane === p ? "is-active" : undefined}
            aria-pressed={pane === p}
            onClick={() => setPane(p)}
          >
            {p === "library" ? "Library" : p === "main" ? VIEWS.find((v) => v.id === view)?.label : "Evidence"}
          </button>
        ))}
      </nav>

      <Toasts toasts={toasts} onDismiss={dismiss} />
    </div>
  );
}
