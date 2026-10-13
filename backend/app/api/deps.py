from __future__ import annotations

from fastapi import BackgroundTasks

from app.core.config import get_settings
from app.db.session import get_session_factory
from app.ingestion.pipeline import IngestionPipeline
from app.services.documents import get_storage


def get_pipeline() -> IngestionPipeline:
    return IngestionPipeline(get_session_factory(), get_storage())


def schedule_ingestion(background: BackgroundTasks, document_id, force: bool = False) -> None:
    """Run ingestion after the response is sent (or inline when background mode is off)."""
    pipeline = get_pipeline()
    if get_settings().ingest_in_background:
        background.add_task(pipeline.run, document_id, force=force)
    else:
        pipeline.run(document_id, force=force)
