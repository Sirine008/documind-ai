import type {
  ApiErrorBody,
  Chunk,
  Conversation,
  ConversationDetail,
  ConversationTurn,
  DocumentInfo,
  EvaluationRun,
  Page,
  PublicConfig,
  StudyResponse,
  StudyTask,
  UploadResponse,
} from "../types/api";

const BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "/api/v1";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 120_000): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, { ...init, signal: controller.signal });
  } catch (err) {
    const aborted = err instanceof DOMException && err.name === "AbortError";
    throw new ApiError(
      0,
      aborted ? "timeout" : "network_error",
      aborted
        ? "The server took too long to respond."
        : "Can't reach the DocuMind API. Check that the backend is running.",
    );
  } finally {
    window.clearTimeout(timer);
  }
  if (response.status === 204) return undefined as T;
  const text = await response.text();
  let body: unknown;
  try {
    body = text ? JSON.parse(text) : undefined;
  } catch {
    body = undefined;
  }
  if (!response.ok) {
    const err = (body as ApiErrorBody | undefined)?.error;
    throw new ApiError(
      response.status,
      err?.code ?? "http_error",
      err?.message ?? `Request failed with status ${response.status}.`,
    );
  }
  return body as T;
}

/** Streaming did not work at the transport level; the caller may retry without streaming. */
export class StreamUnavailableError extends Error {}

const isAbort = (err: unknown) => err instanceof DOMException && err.name === "AbortError";

function parseEvent(block: string): { event: string; data: unknown } | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) };
  } catch {
    return null;
  }
}

/**
 * Ask within a conversation over server-sent events. Calls `onToken` with preview text as the
 * model writes it and resolves with the stored turn. Server-reported errors are thrown as
 * `ApiError`; transport problems as `StreamUnavailableError`; cancellation as an AbortError.
 */
async function askStream(
  conversationId: string,
  question: string,
  { onToken, signal }: { onToken: (text: string) => void; signal?: AbortSignal },
): Promise<ConversationTurn> {
  let response: Response;
  try {
    response = await fetch(`${BASE}/conversations/${conversationId}/messages/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ question }),
      signal,
    });
  } catch (err) {
    if (isAbort(err)) throw err;
    throw new StreamUnavailableError("The answer stream could not be opened.");
  }
  if (response.status === 422) {
    const body = (await response.json().catch(() => undefined)) as ApiErrorBody | undefined;
    throw new ApiError(422, body?.error?.code ?? "validation_error", body?.error?.message ?? "Invalid question.");
  }
  if (!response.ok || !response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
    throw new StreamUnavailableError(`Streaming is not available (HTTP ${response.status}).`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split(/\r?\n\r?\n/);
      buffer = blocks.pop() ?? "";
      for (const block of blocks) {
        const parsed = parseEvent(block);
        if (!parsed) continue;
        if (parsed.event === "token") onToken((parsed.data as { text: string }).text);
        else if (parsed.event === "done") return parsed.data as ConversationTurn;
        else if (parsed.event === "error") {
          const err = (parsed.data as ApiErrorBody).error;
          throw new ApiError(502, err.code, err.message);
        }
      }
    }
  } catch (err) {
    if (isAbort(err) || err instanceof ApiError) throw err;
    throw new StreamUnavailableError("The answer stream was interrupted.");
  } finally {
    reader.cancel().catch(() => undefined); // closes the connection after `done`, errors or cancel
  }
  throw new StreamUnavailableError("The answer stream ended before the answer was complete.");
}

const json = (data: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data),
});

export const api = {
  config: () => request<PublicConfig>("/config"),

  listDocuments: () => request<Page<DocumentInfo>>("/documents?limit=100"),
  getDocument: (id: string) => request<DocumentInfo>(`/documents/${id}`),
  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<UploadResponse>("/documents", { method: "POST", body: form });
  },
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  retryDocument: (id: string, force = false) =>
    request<DocumentInfo>(`/documents/${id}/retry?force=${force}`, { method: "POST" }),
  getChunk: (id: string) => request<Chunk>(`/chunks/${id}`),

  listConversations: () => request<Page<Conversation>>("/conversations?limit=50"),
  createConversation: (documentIds: string[]) =>
    request<Conversation>("/conversations", json({ document_ids: documentIds })),
  getConversation: (id: string) => request<ConversationDetail>(`/conversations/${id}`),
  deleteConversation: (id: string) => request<void>(`/conversations/${id}`, { method: "DELETE" }),
  // Local models on CPU can take minutes; keep this in line with DOCUMIND_LLM_TIMEOUT_S.
  ask: (conversationId: string, question: string) =>
    request<ConversationTurn>(`/conversations/${conversationId}/messages`, json({ question }), 300_000),
  askStream,

  study: (documentId: string, task: StudyTask) =>
    request<StudyResponse>(`/documents/${documentId}/study`, json({ task })),

  listEvaluationRuns: () => request<Page<EvaluationRun>>("/evaluation/runs?limit=10"),
  getEvaluationRun: (id: string) => request<EvaluationRun>(`/evaluation/runs/${id}`),
  runEvaluation: (topK = 5) =>
    request<EvaluationRun>("/evaluation/runs", json({ dataset: "default", top_k: topK }), 600_000),
};
