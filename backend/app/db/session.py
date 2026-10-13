"""Database engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    settings = get_settings()
    engine = create_engine(
        url or settings.database_url,
        pool_size=settings.db_pool_size,
        pool_pre_ping=True,
        future=True,
    )
    timeout = int(settings.db_statement_timeout_ms)

    @event.listens_for(engine, "connect")
    def _set_timeout(dbapi_conn, _record):
        with dbapi_conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {timeout}")
        dbapi_conn.commit()

    return engine


@lru_cache
def get_session_factory(url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False, autoflush=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database() -> dict[str, object]:
    """Return database readiness information (used by the readiness probe)."""
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
        has_vector = conn.execute(text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")).scalar()
        revision = None
        try:
            revision = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
        except Exception:
            conn.rollback()
    return {"database": "ok", "pgvector": bool(has_vector), "migration": revision}
