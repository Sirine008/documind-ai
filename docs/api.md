# API guide

Base path: `/api/v1`. Interactive documentation: `/api/v1/docs` (Swagger UI), `/api/v1/redoc`, and the raw schema at `/api/v1/openapi.json`.

The API has **no authentication**; it is intended for a single local user.

## Conventions

- JSON request and response bodies, except uploads (`multipart/form-data`).
- IDs are UUIDs. Timestamps are ISO 8601 with time zone.
- List endpoints accept `limit` (1–100) and `offset` and return `{"items": [...], "total": n, "limit": l, "offset": o}`.
- Every response carries an `X-Request-ID` header (echoing the request's header if present).
- Errors always have this shape:

```json
{ "error": { "code": "unsupported_type", "message": "Unsupported file type. Upload a PDF (.pdf) or plain-text (.txt, .md) file.", "details": {} } }
```

| Status | Typical codes |
|---|---|
| 404 | `not_found` |
| 409 | `already_processing`, `already_completed`, `not_indexed`, `file_missing` |
| 413 | `file_too_large`, `too_many_pages` |
| 415 | `unsupported_type`, `content_mismatch`, `invalid_text_encoding` |
| 422 | `validation_error` (with `details.errors`), `empty_file`, `corrupt_pdf`, `encrypted_pdf`, `evaluation_documents_missing` |
| 502 | `llm_provider_error` |
| 503 | `database_unavailable`, `not_migrated` |
| 500 | `internal_error` (generic message; details only in server logs) |

## Health

```bash
curl http://localhost:8000/api/v1/health      # {"status": "ok"}
curl http://localhost:8000/api/v1/ready       # database, pgvector, migration revision, provider info
curl http://localhost:8000/api/v1/config      # non-secret settings used by the UI
```

## Documents

Upload (201 for a new document, 200 when the same file already exists):

```bash
curl -F "file=@data/sample_documents/northwind_remote_work_policy.pdf" http://localhost:8000/api/v1/documents
```

```json
{
  "document": { "id": "9ab0…", "title": "northwind remote work policy", "status": "pending", "content_type": "pdf", "page_count": 4, "chunk_count": 0, "…": "…" },
  "duplicate": false,
  "message": "Upload accepted. Ingestion has been queued."
}
```

PowerShell:

```powershell
curl.exe -F "file=@data/sample_documents/northwind_remote_work_policy.pdf" http://localhost:8000/api/v1/documents
```

Other document endpoints:

| Request | Result |
|---|---|
| `GET /documents?status=failed&limit=20` | Paginated list, newest first |
| `GET /documents/{id}` | Full metadata |
| `GET /documents/{id}/status` | `status`, `chunk_count`, `attempts`, `error_code`, `error_message`, `pages_without_text` |
| `POST /documents/{id}/retry` | 202; re-queues a failed (or stale processing) document. `?force=true` re-indexes a completed one |
| `DELETE /documents/{id}` | 204; removes the document, its chunks and vectors, conversation links and the stored file |
| `GET /documents/{id}/chunks` | Indexed passages with page numbers |
| `GET /chunks/{id}` | One passage (used by the evidence panel) |

## Retrieval

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "home office stipend", "mode": "hybrid", "top_k": 3}'
```

Each passage includes `score` plus per-method diagnostics: `semantic_score`, `semantic_rank`, `keyword_score`, `keyword_rank`. Add `"document_ids": ["…"]` to restrict the search.

## Question answering

Stateless:

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How much is the home office stipend?"}'
```

```json
{
  "answer": "Each remote employee receives a one-time home office stipend of 600 euros to buy a desk, a chair or other ergonomic equipment. [S1]",
  "inference": null,
  "evidence_status": "supported",
  "citations": [
    {
      "label": "S1",
      "chunk_id": "…",
      "document_id": "…",
      "document_title": "northwind remote work policy",
      "page_number": 3,
      "chunk_index": 2,
      "excerpt": "Each remote employee receives a one-time home office stipend of 600 euros to buy a desk, a chair or other ergonomic equipment."
    }
  ],
  "retrieved": 6,
  "provider": "mock",
  "model": "extractive-demo",
  "usage": { "input_tokens": 1588, "output_tokens": 39, "estimated": true },
  "latency_ms": 21.2
}
```

(Response captured from demo mode with the sample documents; IDs shortened.)

`evidence_status` is one of:

| Value | Meaning |
|---|---|
| `supported` | At least one citation refers to a retrieved passage. |
| `insufficient_evidence` | Retrieval or the model found no adequate support; `answer` explains this and `citations` is empty. |
| `unverified` | The model answered without any valid citation. Show with a warning. |
| `refused` | The question asked for system prompts, configuration or credentials. |

`inference` holds the model's interpretation, kept apart from what the documents state. `usage.estimated` is `true` when token counts are approximated from text length.

## Conversations

```bash
# Create, optionally scoped to documents (empty list = all documents)
curl -X POST http://localhost:8000/api/v1/conversations -H "Content-Type: application/json" \
  -d '{"document_ids": ["<document-id>"]}'

# Ask (follow-ups use the conversation's recent history)
curl -X POST http://localhost:8000/api/v1/conversations/<id>/messages -H "Content-Type: application/json" \
  -d '{"question": "What must be reported within 4 hours?"}'

curl http://localhost:8000/api/v1/conversations/<id>       # messages with citations
curl -X DELETE http://localhost:8000/api/v1/conversations/<id>
```

## Study tools

```bash
curl -X POST http://localhost:8000/api/v1/documents/<id>/study -H "Content-Type: application/json" \
  -d '{"task": "quiz"}'
```

`task` is `summary`, `key_points`, `revision_notes` or `quiz`. Summaries and notes return `items[]` (`text`, `citations`); quizzes return `questions[]` (`question`, four `options`, `correct_index`, `explanation`, `citations`). `dropped_items` counts generated entries discarded for lacking a valid source. The document must be fully indexed (409 `not_indexed` otherwise).

## Evaluation

```bash
curl -X POST http://localhost:8000/api/v1/evaluation/runs -H "Content-Type: application/json" \
  -d '{"dataset": "default", "top_k": 5, "include_generation": true}'
curl http://localhost:8000/api/v1/evaluation/runs
curl http://localhost:8000/api/v1/evaluation/runs/<run-id>
```

Runs execute synchronously; with a remote LLM they can take minutes. The dataset name must match a file in `data/evaluation/`. The sample documents must be indexed first (`scripts/seed_demo.py`).
