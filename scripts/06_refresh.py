"""Refresh the plan against new data.

This is the routine the client runs each week once fresh extracts land. It
re-runs the pipeline and re-scores every product **reusing the trained model**,
so a refresh takes seconds rather than the ~9 minutes a full retrain costs.

Typical use::

    # 1. Drop new extracts into data/raw/ (same four Appendix A filenames).
    # 2. Check they are readable before committing to a full run.
    python scripts/06_refresh.py --validate-only

    # 3. Refresh the plan.
    python scripts/06_refresh.py

    # 4. Occasionally, retrain on the longer history.
    python scripts/06_refresh.py --retrain

WHEN TO RETRAIN
---------------
Not every week. The model learns demand *patterns*, which move slowly; the plan
depends on recent demand and current stock, which move weekly and are picked up
by a plain refresh. Retrain when:

* the forecast has been drifting - average error above ~35% over several weeks;
* the assortment has changed materially (a wave of new products, a category
  discontinued);
* roughly quarterly regardless, so the model keeps seeing recent seasons.

This script warns when the data has moved far enough past the model's training
cutoff that a retrain is worth considering.

FOR INTRA-WEEK STOCK CHANGES, DO NOT USE THIS SCRIPT.
Stock moves daily while the forecast does not. Post the current position to
``POST /api/score`` instead and the risk layer is recomputed immediately against
it, using the same model and the same arithmetic as this batch.
"""

from __future__ import annotations

import argparse
import subprocess  # noqa: S404 - fixed argv, no shell, no user input
import sys
from pathlib import Path

import pandas as pd

from foresight.config import get_settings
from foresight.exceptions import ForesightError
from foresight.forecast import MODEL_FILENAME, load_forecaster
from foresight.ingest import load_raw_extracts
from foresight.logging_setup import get_logger
from foresight.pipeline import run_pipeline

log = get_logger("scripts.refresh")

#: Weeks of new data past the training cutoff before a retrain is suggested.
STALENESS_WARNING_WEEKS = 13

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _validate_only() -> int:
    """Read and profile the extracts without writing anything."""
    settings = get_settings()
    try:
        raw = load_raw_extracts(settings)
    except ForesightError as exc:
        log.error("extracts are not usable", extra={"context": {"error": str(exc)}})
        return 1

    for name, profile in raw.profiles.items():
        worst = sorted(profile.columns, key=lambda column: column.null_pct, reverse=True)[:3]
        log.info(
            "extract readable",
            extra={
                "context": {
                    "table": name,
                    "rows": profile.row_count,
                    "columns": profile.column_count,
                    "exact_duplicate_rows": profile.exact_duplicate_rows,
                    "highest_null_columns": {
                        column.name: f"{column.null_pct}%" for column in worst
                    },
                }
            },
        )

    log.info("validation passed; run without --validate-only to refresh the plan")
    return 0


def _check_staleness(panel_last_week: pd.Timestamp) -> None:
    """Warn when the data has run well past what the model was trained on."""
    settings = get_settings()
    if not (settings.artifacts_dir / MODEL_FILENAME).exists():
        return

    try:
        forecaster = load_forecaster(settings=settings)
    except ForesightError:
        return

    trained_to = forecaster.metadata.get("train_target_week_max")
    if not trained_to:
        return

    gap_weeks = int((panel_last_week - pd.Timestamp(trained_to)).days // 7)
    if gap_weeks >= STALENESS_WARNING_WEEKS:
        log.warning(
            "model is trained on materially older data; consider --retrain",
            extra={
                "context": {
                    "model_trained_to": trained_to,
                    "data_now_reaches": str(panel_last_week.date()),
                    "weeks_behind": gap_weeks,
                }
            },
        )
    else:
        log.info(
            "model currency",
            extra={"context": {"model_trained_to": trained_to, "weeks_behind": gap_weeks}},
        )


def _run(script: str) -> int:
    """Run one pipeline script in a subprocess.

    A subprocess rather than an import so each stage starts from clean module
    state, exactly as it does when run by hand. The argv is fixed in code - no
    shell, and nothing from user input reaches it.
    """
    log.info("running stage", extra={"context": {"script": script}})
    completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
        [sys.executable, str(PROJECT_ROOT / "scripts" / script)],
        check=False,
        cwd=PROJECT_ROOT,
    )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the extracts are readable and report their shape. Writes nothing.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Also retrain and re-backtest the model (~9 minutes).",
    )
    parser.add_argument(
        "--skip-readout",
        action="store_true",
        help="Skip regenerating the executive readout PDF.",
    )
    arguments = parser.parse_args(argv)

    if arguments.validate_only:
        return _validate_only()

    settings = get_settings()

    # --- 1. Pipeline ------------------------------------------------------- #
    try:
        result = run_pipeline(settings, write=True)
    except ForesightError as exc:
        log.error("refresh failed in the pipeline", extra={"context": {"error": str(exc)}})
        return 1

    panel_last_week = pd.Timestamp(result.weekly_panel["week_start"].max())
    log.info(
        "data refreshed",
        extra={
            "context": {
                "skus": result.n_skus,
                "weeks": result.n_weeks,
                "latest_week": str(panel_last_week.date()),
                "cleaning_actions": len(result.cleaning_report.actions),
            }
        },
    )

    # --- 2. Retrain, or check whether one is due --------------------------- #
    if arguments.retrain:
        if _run("03_train_backtest.py") != 0:
            log.error("retraining failed; the previous model is unchanged")
            return 1
    else:
        _check_staleness(panel_last_week)
        if not (settings.artifacts_dir / MODEL_FILENAME).exists():
            log.error(
                "no trained model found",
                extra={"context": {"hint": "run with --retrain, or scripts/03_train_backtest.py"}},
            )
            return 1

    # --- 3. Re-score ------------------------------------------------------- #
    if _run("04_score_risk.py") != 0:
        log.error("scoring failed")
        return 1

    # --- 4. Readout -------------------------------------------------------- #
    if not arguments.skip_readout and _run("05_build_readout.py") != 0:
        log.warning("readout generation failed; the plan itself is still refreshed")

    log.info(
        "refresh complete",
        extra={
            "context": {
                "retrained": arguments.retrain,
                "dashboard": "restart the service to pick up new artifacts",
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
