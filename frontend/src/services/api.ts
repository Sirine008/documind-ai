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
  ask: (conversationId: string, question: string) =>
    request<ConversationTurn>(`/conversations/${conversationId}/messages`, json({ question })),

  study: (documentId: string, task: StudyTask) =>
    request<StudyResponse>(`/documents/${documentId}/study`, json({ task })),

  listEvaluationRuns: () => request<Page<EvaluationRun>>("/evaluation/runs?limit=10"),
  getEvaluationRun: (id: string) => request<EvaluationRun>(`/evaluation/runs/${id}`),
  runEvaluation: (topK = 5) =>
    request<EvaluationRun>("/evaluation/runs", json({ dataset: "default", top_k: topK }), 600_000),
};
