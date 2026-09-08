"""Run the D1 pipeline: raw extracts to validated analysis-ready datasets.

Writes Parquet outputs to ``data/processed/`` and a machine-readable audit trail
of every cleaning decision.

Usage::

    python scripts/01_run_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import sys

from foresight.config import get_settings
from foresight.exceptions import ForesightError
from foresight.logging_setup import get_logger
from foresight.pipeline import run_pipeline

log = get_logger("scripts.run_pipeline")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run the pipeline without persisting outputs (useful for validation).",
    )
    arguments = parser.parse_args(argv)

    settings = get_settings()

    try:
        result = run_pipeline(settings, write=not arguments.no_write)
    except ForesightError as exc:
        log.error("pipeline failed", extra={"context": {"error": str(exc)}})
        return 1

    report = result.cleaning_report
    severity_counts = (
        report.to_frame()["severity"].value_counts().to_dict() if report.actions else {}
    )

    summary = {
        "daily_rows": len(result.analysis_ready),
        "weekly_rows": len(result.weekly_panel),
        "skus": result.n_skus,
        "weeks": result.n_weeks,
        "first_week": str(result.weekly_panel["week_start"].min().date()),
        "last_week": str(result.weekly_panel["week_start"].max().date()),
        "cleaning_actions": len(report.actions),
        "cleaning_severity": severity_counts,
        "rows_touched_by_cleaning": report.total_rows_touched(),
        "duration_seconds": result.duration_seconds,
    }

    if not arguments.no_write:
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        (settings.artifacts_dir / "pipeline_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    log.info("pipeline summary", extra={"context": summary})
    return 0


if __name__ == "__main__":
    sys.exit(main())
