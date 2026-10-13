// Types mirroring the backend's Pydantic schemas (backend/app/schemas).

export type DocumentStatus = "pending" | "processing" | "completed" | "failed";
export type EvidenceStatus = "supported" | "insufficient_evidence" | "unverified" | "refused";
export type StudyTask = "summary" | "key_points" | "revision_notes" | "quiz";

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface DocumentInfo {
  id: string;
  title: string;
  original_filename: string;
  content_type: "pdf" | "text";
  size_bytes: number;
  sha256: string;
  status: DocumentStatus;
  error_code: string | null;
  error_message: string | null;
  page_count: number | null;
  chunk_count: number;
  pages_without_text: number[];
  attempts: number;
  embedding_model: string | null;
  created_at: string;
  updated_at: string;
  processing_started_at: string | null;
  processed_at: string | null;
}

export interface UploadResponse {
  document: DocumentInfo;
  duplicate: boolean;
  message: string;
}

export interface Citation {
  label: string;
  chunk_id: string;
  document_id: string;
  document_title: string;
  page_number: number | null;
  chunk_index: number;
  excerpt: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  inference: string | null;
  evidence_status: EvidenceStatus | null;
  citations: Citation[];
  usage: Record<string, unknown>;
  latency_ms: number | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  document_ids: string[];
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface ConversationTurn {
  user_message: Message;
  assistant_message: Message;
}

export interface Chunk {
  id: string;
  document_id: string;
  chunk_index: number;
  page_number: number | null;
  content: string;
}

export interface StudyItem {
  text: string;
  citations: Citation[];
}

export interface QuizQuestion {
  question: string;
  options: string[];
  correct_index: number;
  explanation: string;
  citations: Citation[];
}

export interface StudyResponse {
  document_id: string;
  task: StudyTask;
  items: StudyItem[];
  questions: QuizQuestion[];
  passages_used: number;
  dropped_items: number;
  provider: string;
  model: string;
  usage: Record<string, unknown>;
}

export interface EvaluationMetrics {
  retrieval?: Record<string, number | null>;
  generation?: Record<string, number | null | Record<string, number | boolean>>;
}

export interface EvaluationResultRow {
  id: string;
  question: string;
  answerable: boolean;
  recall_at_k?: number;
  reciprocal_rank?: number;
  evidence_status?: EvidenceStatus;
  answer?: string;
  abstained?: boolean;
  citation_correctness?: number;
  groundedness?: number;
  latency_ms?: number;
}

export interface EvaluationRun {
  id: string;
  dataset: string;
  status: "running" | "completed" | "failed";
  config: Record<string, unknown>;
  metrics: EvaluationMetrics;
  results: EvaluationResultRow[];
  error_message: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface PublicConfig {
  demo_mode: boolean;
  llm_provider: string;
  llm_model: string;
  embedding_provider: string;
  retrieval_mode: string;
  max_upload_mb: number;
  allowed_extensions: string[];
}

export interface ApiErrorBody {
  error: { code: string; message: string; details: Record<string, unknown> };
}
