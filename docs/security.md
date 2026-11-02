# Security and limitations

DocuMind AI is an educational, **single-user local** application. This document lists the controls that exist, how they are implemented, and what they don't cover. Nothing here has been independently audited or penetration-tested.

## Threat model

In scope: mistakes and hostile **content** — malformed or malicious uploads, documents containing prompt-injection text, questions trying to extract secrets, and accidental leakage through logs or errors.

Out of scope for this version: hostile **users** of the API. There is no authentication, authorization or rate limiting. Anyone who can reach the backend can list, read and delete all documents and conversations.

## Controls

### Configuration and secrets
- All configuration comes from environment variables or a git-ignored `.env`; `.env.example` contains no real secrets.
- `DOCUMIND_LLM_API_KEY` is a Pydantic `SecretStr`: it doesn't appear in `repr()` or logs.
- The log formatter redacts `sk-…` style keys and `api_key=`/`Authorization:`-style values as a second line of defense.
- The configured API key is redacted from model output before it is returned.
- Docker Compose refuses to start without `POSTGRES_PASSWORD` and binds host ports to `127.0.0.1`.

### Uploads
- Size limit (`DOCUMIND_MAX_UPLOAD_MB`, default 20 MB) checked on the declared `Content-Length` and on the bytes actually read; PDF page limit (`DOCUMIND_MAX_PDF_PAGES`, default 500).
- Type is decided by content, not by the client: PDFs must start with `%PDF-` and open in PyMuPDF; text must be valid UTF-8 with almost no control characters; extension and content must agree.
- Encrypted and corrupt PDFs are rejected with specific errors.
- Client filenames are sanitized for display only. Files are stored as `<uuid4>.pdf|txt`; the storage layer validates names against that pattern and checks that resolved paths stay inside the upload directory.
- Uploaded files and the database are excluded from git and from Docker build contexts.

### Database
- All queries go through SQLAlchemy with bound parameters. Full-text queries are built from alphanumeric terms only and passed as parameters to `to_tsquery`.
- No endpoint accepts SQL, filesystem paths or URLs to fetch. Evaluation dataset names are restricted to `[A-Za-z0-9_-]{1,64}` and resolved inside `data/evaluation/`.
- A `statement_timeout` (default 15 s) applies to every connection.
- Deleting a document deletes its chunks and vectors through `ON DELETE CASCADE` in the same transaction, then removes the file.

### Errors and logging
- Domain errors return a code and a safe message. Unhandled exceptions return a generic `internal_error`; the traceback goes only to server logs.
- Logs are structured JSON with request IDs, paths, status codes, document IDs, counts and timings. Document text, questions and answers are not logged.

### HTTP
- CORS origins come from `DOCUMIND_CORS_ORIGINS`; credentials are not allowed; only `GET`, `POST`, `DELETE` and the `Content-Type` header are permitted.
- Responses carry `X-Content-Type-Options: nosniff`. The nginx config adds a restrictive Content-Security-Policy (same-origin scripts, styles, fonts and API calls) and `Referrer-Policy: same-origin`.
- LLM HTTP calls have a timeout (`DOCUMIND_LLM_TIMEOUT_S`, default 60 s); provider error bodies are not passed to clients.
- Streamed answer previews redact the configured API key before sending: the preview holds back a tail at least as long as the key, so a key split across model deltas is still recognized. Citations are validated only in the final `done` event.
- The frontend loads no third-party scripts or fonts.

### Prompt injection
Layered mitigations:
1. Retrieved passages are wrapped in `<source>` delimiters, and delimiter-like text inside documents (`</source>`, `<|…|>` tokens, code fences) is neutralized so a document can't close its own block or forge another source.
2. The system prompt states that source content is untrusted data and that security rules take priority.
3. Passages matching instruction-like patterns ("ignore previous instructions", "you are now", role tokens, requests to reveal keys) are flagged to the model by label.
4. Questions asking for system prompts, instructions, credentials, environment variables or connection strings are refused before retrieval.
5. Output validation: citation labels not supplied to the model are removed; answers without a valid citation are marked `unverified`; secrets are redacted.
6. Study material items without a valid source are discarded.

**Residual risk:** these measures lower the risk but cannot prevent it. A crafted document can still bias or mislead a model's answer within the rules (for example by stating false "facts" that then get cited faithfully). Pattern lists produce false positives (a policy legitimately about "API key rotation" is fine, but "show the API key" is refused) and false negatives (paraphrased or non-English attacks). Treat answers about untrusted documents with the same caution as the documents themselves.

## Known limitations

- **No authentication or multi-tenancy.** Required before any shared or networked deployment.
- **No rate limiting.** Large or many uploads, or repeated evaluation runs, can exhaust CPU and memory.
- **In-process background ingestion.** A crash leaves a document in `processing`; it becomes retryable after 30 minutes.
- **Evaluation runs synchronously** inside an HTTP request.
- **No OCR.** Scanned pages are detected and reported, not read.
- **No malware scanning** of uploads. Files are parsed by PyMuPDF only; keep it updated.
- **Data at rest is not encrypted** by the application; rely on disk or volume encryption.
- **The hashing embedder is lexical only** and should not be used to judge retrieval quality.

## Before deploying beyond localhost

Add authentication and per-user ownership checks on every query; put the API behind TLS; add rate and size limits at the proxy; move ingestion to a durable queue with resource limits; set up backups for the database volume; and run dependency vulnerability scanning (for example `pip-audit` and `npm audit`) in CI.
