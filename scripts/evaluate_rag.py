"""Run the RAG evaluation dataset and write machine-readable results.

Usage:
    python scripts/evaluate_rag.py [--dataset default] [--top-k 5] [--retrieval-only] [--no-store]

Results are written to data/evaluation/results/<dataset>_<timestamp>.json and
<dataset>_latest.json, and (unless --no-store) saved as an evaluation run in the
database so the UI can display them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from app.core.logging import configure_logging  # noqa: E402
from app.db.session import get_session_factory  # noqa: E402
from app.evaluation.runner import EvaluationRunner, run_and_store, write_results_file  # noqa: E402

RESULTS_DIR = ROOT / "data" / "evaluation" / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="default")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true", help="Skip answer generation")
    parser.add_argument("--no-store", action="store_true", help="Do not save the run in the database")
    args = parser.parse_args()
    configure_logging("WARNING", "text")

    with get_session_factory()() as session:
        if args.no_store:
            output = EvaluationRunner(session).run(
                args.dataset, top_k=args.top_k, include_generation=not args.retrieval_only
            )
        else:
            run = run_and_store(session, args.dataset, args.top_k, not args.retrieval_only)
            output = {
                "run_id": str(run.id),
                "dataset": run.dataset,
                "config": run.config,
                "metrics": run.metrics,
                "results": run.results,
            }
    path = write_results_file(output, RESULTS_DIR)
    print(json.dumps({"config": output["config"], "metrics": output["metrics"]}, indent=2))
    print(f"\nFull results: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
