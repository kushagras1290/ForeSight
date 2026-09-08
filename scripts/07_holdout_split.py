"""Produce a chronological train / holdout split for independent testing.

Trains on the earliest 70% of weeks and holds the most recent 30% back
completely, then forecasts and scores that holdout.

The split is by **time**, never at random. A random split would put later weeks
in training and earlier weeks in test, so the model would be asked to "predict"
a past it had already been shown, and the resulting accuracy would be fiction.

Outputs::

    data/holdout/holdout_actuals.csv    upload this on the dashboard
    data/holdout/holdout_forecast.csv   what the model predicted, for reference
    artifacts/holdout_forecast.parquet  same, consumed by the scoring service
    artifacts/holdout_metrics.json      the honest score against the baseline

Usage::

    python scripts/07_holdout_split.py
    python scripts/07_holdout_split.py --holdout-fraction 0.2
    python scripts/07_holdout_split.py --skip-ensemble     # faster, GBM only
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from foresight.config import get_settings
from foresight.exceptions import ForesightError
from foresight.features import build_weekly_calendar
from foresight.holdout import run_holdout
from foresight.logging_setup import get_logger

log = get_logger("scripts.holdout")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.3,
        help="Share of the timeline held back for testing. Default 0.3 (a 70/30 split).",
    )
    parser.add_argument(
        "--skip-ensemble",
        action="store_true",
        help="Fit the LightGBM model only. Roughly halves the run time.",
    )
    parser.add_argument(
        "--no-drivers",
        action="store_true",
        help="Hold out using the four Appendix A extracts alone.",
    )
    arguments = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_directories()

    holdout_dir = settings.data_dir / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    try:
        panel = pd.read_parquet(settings.processed_dir / "weekly_panel.parquet")
        calendar = pd.read_parquet(settings.processed_dir / "calendar.parquet")
    except (FileNotFoundError, OSError) as exc:
        log.error(
            "processed data missing",
            extra={"context": {"error": str(exc), "hint": "run scripts/01_run_pipeline.py"}},
        )
        return 1

    # Match whatever the shipped model was trained on, so the holdout measures
    # that model rather than a different one that happens to share its name.
    plan_path = settings.processed_dir / "marketing_clean.parquet"
    marketing_plan = None
    if plan_path.exists():
        plan = pd.read_parquet(plan_path)
        marketing_plan = plan if not plan.empty else None
    use_drivers = settings.use_drivers and not arguments.no_drivers and marketing_plan is not None

    try:
        result = run_holdout(
            panel,
            build_weekly_calendar(calendar),
            settings,
            holdout_fraction=arguments.holdout_fraction,
            use_ensemble=not arguments.skip_ensemble,
            use_drivers=use_drivers,
            marketing_plan=marketing_plan,
        )
    except ForesightError as exc:
        log.error("holdout run failed", extra={"context": {"error": str(exc)}})
        return 1

    # --- The file the client uploads ---------------------------------------- #
    actuals_path = holdout_dir / "holdout_actuals.csv"
    result.actuals.assign(
        week_starting=result.actuals["week_starting"].dt.strftime("%Y-%m-%d")
    ).to_csv(actuals_path, index=False)

    forecast_csv = holdout_dir / "holdout_forecast.csv"
    result.forecast.assign(
        target_week=result.forecast["target_week"].dt.strftime("%Y-%m-%d"),
        origin_week=result.forecast["origin_week"].dt.strftime("%Y-%m-%d"),
    ).to_csv(forecast_csv, index=False)

    # --- Consumed by the scoring service so uploads match against it -------- #
    result.forecast.to_parquet(settings.artifacts_dir / "holdout_forecast.parquet", index=False)

    # Forecast joined to outcome. This is what the dashboard's test-data view
    # reads, so what a reviewer sees on screen is the same table the metrics
    # above were computed from rather than a second join that could drift.
    result.scored.to_parquet(settings.artifacts_dir / "holdout_scored.parquet", index=False)

    summary = result.summary()
    (settings.artifacts_dir / "holdout_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    log.info("holdout artifacts written", extra={"context": summary})

    verdict = (
        f"holdout WAPE {result.model.wape:.4f} vs baseline {result.baseline.wape:.4f} "
        f"({result.improvement_vs_baseline:+.1%})"
    )
    if result.model.wape >= result.baseline.wape:
        # Brief section 7.1: report it, do not bury it.
        log.warning(
            "the model did NOT beat the baseline on the holdout",
            extra={"context": {"verdict": verdict}},
        )
    else:
        log.info("holdout verdict", extra={"context": {"verdict": verdict}})

    log.info(
        "ready to test",
        extra={
            "context": {
                "upload_this": str(actuals_path),
                "rows": len(result.actuals),
                "weeks": f"{result.holdout_start.date()} to {result.holdout_end.date()}",
                "where": "dashboard > Forecast accuracy > Calibrate on your own data",
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
