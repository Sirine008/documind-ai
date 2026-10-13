# Troubleshooting

Start with the readiness probe; it reports the database, pgvector and migration state in one call:

```bash
curl http://localhost:8000/api/v1/ready
```

## Startup and database

**`connection refused` / `database_unavailable`**
PostgreSQL isn't reachable at `DOCUMIND_DATABASE_URL`. Check the host, port and credentials. With Docker: `docker compose ps` and `docker compose logs db`.

**`/ready` returns 503 `not_migrated`**
Run migrations: `cd backend && alembic upgrade head`. In Docker this runs automatically at container start; check `docker compose logs backend` for migration errors.

**`type "vector" does not exist` or `extension "vector" is not available`**
pgvector isn't installed in your PostgreSQL. Use the `pgvector/pgvector:pg16` image, or install the `postgresql-16-pgvector` package (Debian/Ubuntu) and run `CREATE EXTENSION vector;` as a superuser in the database.

**`permission denied to create extension "vector"`**
The application role isn't a superuser. Create the extension once as a superuser (`psql -d documind -c "CREATE EXTENSION vector;"`), then rerun migrations.

**Docker Compose: `required variable POSTGRES_PASSWORD is missing a value`**
Copy `.env.example` to `.env` and set `POSTGRES_PASSWORD`. Use URL-safe characters (letters, digits, `-`, `_`), because it is embedded in the database URL.

**Changed the password after the first start and now authentication fails**
PostgreSQL only reads `POSTGRES_PASSWORD` when the volume is first created. Either change the password inside the database or reset the data with `docker compose down -v` (this deletes all documents).

**`error parsing value for field "cors_origins"`**
Give a comma-separated list (`http://localhost:5173,http://localhost:8080`) or a JSON list (`["http://localhost:5173"]`).

## Embeddings

**`sentence-transformers is not installed`**
Install it: `pip install torch --index-url https://download.pytorch.org/whl/cpu` then `pip install -r backend/requirements-ml.txt`. Or set `DOCUMIND_EMBEDDING_PROVIDER=hashing` for offline use.

**Model download hangs or fails**
The first indexing downloads `sentence-transformers/all-MiniLM-L6-v2` from huggingface.co. Behind a proxy, set `HTTPS_PROXY`. Offline, download the model elsewhere, copy it into the cache (`HF_HOME`, `/srv/models` in Docker) and set `HF_HUB_OFFLINE=1`.

**`Model dimension … does not match DOCUMIND_EMBEDDING_DIM`**
The database column has a fixed dimension (384). Using a model with another dimension needs a new migration that changes `chunks.embedding`, followed by re-indexing.

**Switched embedding provider and results got worse or errored**
Vectors from different models aren't comparable. Re-index: `python scripts/seed_demo.py --reindex` for samples, or `POST /api/v1/documents/{id}/retry?force=true` per document.

## Documents

| Error code | Meaning and fix |
|---|---|
| `ocr_required` | The PDF has no extractable text (scanned images). Run OCR, e.g. `ocrmypdf input.pdf output.pdf`, and upload the output. |
| `encrypted_pdf` | Remove the password (e.g. `qpdf --decrypt in.pdf out.pdf`) and upload again. |
| `corrupt_pdf` | The file can't be opened. Re-export or repair it (`qpdf in.pdf out.pdf`). |
| `content_mismatch` | Extension and content disagree (e.g. a PDF renamed to `.txt`). Use the correct extension. |
| `invalid_text_encoding` | Text files must be UTF-8. Re-save as UTF-8. |
| `file_too_large`, `too_many_pages` | Raise `DOCUMIND_MAX_UPLOAD_MB` / `DOCUMIND_MAX_PDF_PAGES`, or split the file. Behind nginx, also raise `client_max_body_size` in `frontend/nginx.conf`. |
| `embedding_failed` | See the backend logs; usually the model failed to load. Fix, then use **Retry**. |
| `file_missing` | The stored file was removed from the upload volume. Delete the document and upload again. |

**A document stays in "Indexing…"**
The backend may have restarted mid-ingestion. After 30 minutes the claim is considered stale and **Retry** works (`POST /documents/{id}/retry`). Check `docker compose logs backend` for the cause.

**Uploading the same file again does nothing**
Duplicates are detected by content hash; the existing document is returned. Delete it first to re-upload.

## Answers

**Everything returns "insufficient evidence"**
- Check that documents show **Ready** and that the conversation's scope includes them (the chat header lists what is searched).
- In demo mode, the extractive provider matches words, so phrase questions with terms the document uses, or configure an LLM.
- With Sentence Transformers, lower `DOCUMIND_MIN_SEMANTIC_SCORE` (default 0.25) if relevant passages are being gated out.

**`llm_provider_error` (502)**
The provider rejected or failed the call. Check `DOCUMIND_LLM_API_KEY`, `DOCUMIND_LLM_MODEL` and `DOCUMIND_LLM_BASE_URL`, and the backend log line `llm_http_error`, which records the HTTP status. For Ollama, the base URL is `http://localhost:11434/v1` (from inside a container with Docker Desktop: `http://host.docker.internal:11434/v1`; on Linux also add `extra_hosts: ["host.docker.internal:host-gateway"]` to the backend service).

**Answers are marked "unverified"**
The model answered without citing a supplied passage. Smaller local models often ignore citation instructions; try a larger model or a lower temperature.

## Frontend

**Banner: "Can't reach the DocuMind API"**
In development the Vite server proxies `/api` to `http://localhost:8000`; start the backend there or set `VITE_API_PROXY_TARGET`. In Docker, check that the backend container is healthy (`docker compose ps`).

**Browser shows CORS errors**
Only when the frontend calls the API on another origin (e.g. with `VITE_API_BASE_URL`). Add that origin to `DOCUMIND_CORS_ORIGINS`.

## Tests

**Database tests are skipped**
They need PostgreSQL with pgvector at `DOCUMIND_TEST_DATABASE_URL` (default `postgresql+psycopg://documind@localhost:5432/documind_test`, which only works with passwordless local auth; otherwise set the variable, e.g. `postgresql+psycopg://documind:<password>@localhost:5432/documind_test`). Point it at a disposable database: the tests drop and recreate the schema.

**Windows: `Activate.ps1 cannot be loaded because running scripts is disabled`**
Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or activate with `.\.venv\Scripts\activate.bat` from `cmd`.
