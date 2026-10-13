"""Load the bundled sample documents into the database and index them.

Idempotent: files already uploaded (same SHA-256) are detected as duplicates and are
only re-indexed if their ingestion previously failed.

Usage:
    python scripts/seed_demo.py              # from the project root, with the backend venv
    python scripts/seed_demo.py --reindex    # rebuild chunks/embeddings (e.g. after changing chunk settings)
    docker compose exec backend python scripts/seed_demo.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))  # inside the Docker image the backend lives at the root

from app.core.config import get_settings  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.db.session import get_session_factory  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.models import DocumentStatus  # noqa: E402
from app.services.documents import DocumentService, get_storage  # noqa: E402

SAMPLE_DIR = ROOT / "data" / "sample_documents"


def main() -> int:
    parser = argparse.ArgumentParser(description="Load and index the sample documents.")
    parser.add_argument("--reindex", action="store_true", help="Re-index documents that are already indexed")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging("WARNING", "text")
    files = sorted(p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in {".pdf", ".txt", ".md"})
    if not files:
        print(f"No sample documents found in {SAMPLE_DIR}. Run scripts/generate_sample_docs.py.")
        return 1
    factory = get_session_factory()
    pipeline = IngestionPipeline(factory, get_storage(settings))
    failures = 0
    for path in files:
        with factory() as session:
            doc, duplicate = DocumentService(session).upload(path.read_bytes(), path.name)
            doc_id, status = doc.id, doc.status
        if duplicate and status == DocumentStatus.completed and not args.reindex:
            print(f"= {path.name}: already indexed ({doc_id})")
            continue
        result = pipeline.run(doc_id, force=args.reindex or status == DocumentStatus.failed)
        mark = "+" if result.status == "completed" else "!"
        print(f"{mark} {path.name}: {result.status}, {result.chunk_count} chunks ({doc_id})")
        failures += result.status != "completed"
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
