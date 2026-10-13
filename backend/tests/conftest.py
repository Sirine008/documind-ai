"""Shared test fixtures.

Unit tests need nothing external. Tests marked ``db`` need PostgreSQL with pgvector;
the URL comes from ``DOCUMIND_TEST_DATABASE_URL`` (default: a local ``documind_test``
database). They are skipped automatically when the database is unreachable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

TEST_DB_URL = os.environ.get(
    "DOCUMIND_TEST_DATABASE_URL",
    "postgresql+psycopg://documind@localhost:5432/documind_test",
)
_UPLOAD_DIR = tempfile.mkdtemp(prefix="documind-test-uploads-")

# Configure the application *before* it is imported anywhere.
os.environ.update(
    {
        "DOCUMIND_ENVIRONMENT": "test",
        "DOCUMIND_DATABASE_URL": TEST_DB_URL,
        "DOCUMIND_EMBEDDING_PROVIDER": "hashing",
        "DOCUMIND_LLM_PROVIDER": "mock",
        "DOCUMIND_UPLOAD_DIR": _UPLOAD_DIR,
        "DOCUMIND_INGEST_IN_BACKGROUND": "false",
        "DOCUMIND_LOG_LEVEL": "WARNING",
        "DOCUMIND_LOG_FORMAT": "text",
        "DOCUMIND_MAX_UPLOAD_MB": "2",
    }
)
os.environ.pop("DOCUMIND_LLM_API_KEY", None)

import pymupdf  # noqa: E402
import pytest  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parents[1]


# --- document builders ----------------------------------------------------------------
def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        if body:
            page.insert_textbox(pymupdf.Rect(50, 50, 545, 800), body, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def make_scanned_pdf(pages: int = 2) -> bytes:
    """A PDF whose pages contain only an image (no text layer), like a scan."""
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
    pix.clear_with(200)
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(pymupdf.Rect(50, 50, 300, 300), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


def make_encrypted_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Secret text")
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="owner")
    doc.close()
    return data


@pytest.fixture
def pdf_factory():
    return make_pdf


# --- database ---------------------------------------------------------------------------
def _database_available() -> bool:
    try:
        engine = create_engine(TEST_DB_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


_DB_OK: bool | None = None


def pytest_collection_modifyitems(config, items):
    global _DB_OK
    if not any("db" in item.keywords for item in items):
        return
    _DB_OK = _database_available()
    if not _DB_OK:
        skip = pytest.mark.skip(reason=f"PostgreSQL not reachable at DOCUMIND_TEST_DATABASE_URL ({TEST_DB_URL})")
        for item in items:
            if "db" in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def migrated_db():
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    cfg.set_main_option("sqlalchemy.url", TEST_DB_URL)
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield
    # Leave the schema in place for inspection; data is truncated per test.


@pytest.fixture
def db_session(migrated_db):
    from app.db.session import get_engine, get_session_factory

    with get_engine().begin() as conn:
        conn.execute(
            text(
                "TRUNCATE messages, conversation_documents, conversations, chunks, documents, "
                "evaluation_runs RESTART IDENTITY CASCADE"
            )
        )
    session = get_session_factory()()
    yield session
    session.close()


@pytest.fixture
def storage():
    from app.services.documents import get_storage

    return get_storage()


@pytest.fixture
def pipeline(db_session, storage):
    from app.db.session import get_session_factory
    from app.ingestion.pipeline import IngestionPipeline

    return IngestionPipeline(get_session_factory(), storage)


@pytest.fixture
def ingest(db_session, pipeline):
    """Upload bytes via the service layer and ingest synchronously; returns the document id."""
    from app.services.documents import DocumentService

    def _ingest(data: bytes, filename: str):
        doc, _ = DocumentService(db_session).upload(data, filename)
        result = pipeline.run(doc.id)
        db_session.expire_all()
        return doc.id, result

    return _ingest


@pytest.fixture
def client(db_session):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
