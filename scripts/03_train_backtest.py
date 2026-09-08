"""Train, backtest and select the demand forecast (deliverable D3).

Sequence:

1. Build features from the weekly panel.
2. **Run the leakage guard.** If any feature reads the future the run aborts
   here - no model is trained and nothing is written. Brief section 7.1 makes
   this the non-negotiable rule of the engagement, so it is enforced as a build
   gate rather than a code-review convention.
3. Rolling-origin backtest of seasonal-naive, LightGBM and the adaptive
   ensemble on identical folds.
4. Select whichever posts the lowest WAPE - including the baseline.
5. Persist models, metrics and backtest predictions.

Usage::

    python scripts/03_train_backtest.py
    python scripts/03_train_backtest.py --skip-ensemble    # GBM only, faster
    python scripts/03_train_backtest.py --no-drivers       # four Appendix A tables only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from foresight.backtest import run_backtest
from foresight.config import get_settings
from foresight.custom_model import ENSEMBLE_FILENAME
from foresight.exceptions import ForesightError
from foresight.features import (
    assert_no_leakage,
    build_base_features,
    build_supervised_frame,
    build_weekly_calendar,
    resolve_feature_spec,
)
from foresight.forecast import MODEL_FILENAME
from foresight.logging_setup import get_logger

log = get_logger("scripts.train_backtest")


def _load_marketing_plan(path: Path) -> pd.DataFrame | None:
    """Read the forward media plan, or ``None`` when the client did not send one.

    The plan extends past the end of the sales history on purpose: committed
    spend for a future target week is a legitimate forecast input, and the weekly
    panel has no rows for those weeks.
    """
    if not path.exists():
        return None
    plan = pd.read_parquet(path)
    return plan if not plan.empty else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-ensemble",
        action="store_true",
        help="Backtest the GBM only. Much faster; use while iterating.",
    )
    parser.add_argument(
        "--skip-leakage-check",
        action="store_true",
        help="Not recommended. The guard is the project's core correctness control.",
    )
    parser.add_argument(
        "--no-drivers",
        action="store_true",
        help=(
            "Forecast from the four Appendix A extracts alone, ignoring any "
            "commercial driver data. This is the brief's own scope and the "
            "honest baseline for measuring what the driver fields are worth."
        ),
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Suffix for the artifact filenames, so two runs can be compared side by side.",
    )
    arguments = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_directories()
    suffix = f"_{arguments.tag}" if arguments.tag else ""

    panel_path = settings.processed_dir / "weekly_panel.parquet"
    calendar_path = settings.processed_dir / "calendar.parquet"
    if not panel_path.exists() or not calendar_path.exists():
        log.error(
            "processed data missing",
            extra={"context": {"hint": "run scripts/01_run_pipeline.py first"}},
        )
        return 1

    panel = pd.read_parquet(panel_path)
    calendar = pd.read_parquet(calendar_path)
    weekly_calendar = build_weekly_calendar(calendar)
    horizons = tuple(range(1, settings.horizon_weeks + 1))

    marketing_plan = _load_marketing_plan(settings.processed_dir / "marketing_clean.parquet")
    # Drivers are used only when the client actually supplied them AND the run
    # was not explicitly restricted to the brief's four tables.
    use_drivers = settings.use_drivers and not arguments.no_drivers and marketing_plan is not None
    feature_spec = resolve_feature_spec(use_drivers=use_drivers)

    log.info(
        "feature contract resolved",
        extra={
            "context": {
                "drivers": use_drivers,
                "features": len(feature_spec.all_features),
                "reason": "explicitly disabled"
                if arguments.no_drivers
                else ("no driver extracts present" if marketing_plan is None else "drivers active"),
            }
        },
    )

    try:
        # --- Gate: prove the features do not read the future ---------------- #
        if not arguments.skip_leakage_check:
            assert_no_leakage(
                panel,
                weekly_calendar,
                horizons,
                use_drivers=use_drivers,
                marketing_plan=marketing_plan,
            )
        else:
            log.warning("leakage check skipped by flag; results are not trustworthy")

        base = build_base_features(panel, use_drivers=use_drivers)
        supervised = build_supervised_frame(
            base,
            weekly_calendar,
            horizons,
            use_drivers=use_drivers,
            marketing_plan=marketing_plan,
        )

        result, forecaster, ensemble = run_backtest(
            supervised,
            panel,
            settings,
            include_ensemble=not arguments.skip_ensemble,
            feature_spec=feature_spec,
        )
    except ForesightError as exc:
        log.error("training failed", extra={"context": {"error": str(exc)}})
        return 1

    # --- Persist ------------------------------------------------------------ #
    def artifact(name: str) -> Path:
        """Insert the run tag before the extension, so runs never overwrite."""
        stem, _, extension = name.rpartition(".")
        return settings.artifacts_dir / f"{stem}{suffix}.{extension}"

    forecaster.save(artifact(MODEL_FILENAME))
    if ensemble is not None:
        ensemble.save(artifact(ENSEMBLE_FILENAME))

    summary = result.summary()
    summary["uses_drivers"] = use_drivers
    if ensemble is not None:
        summary["ensemble_weights"] = {
            regime: [round(weight, 3) for weight in vector]
            for regime, vector in ensemble.weights.items()
        }
        summary["ensemble_regime_shares"] = {
            regime: round(share, 4) for regime, share in ensemble.regime_shares.items()
        }
        summary["ensemble_interval_scale"] = round(ensemble.interval_scale, 4)

    artifact("metrics.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    result.predictions.to_parquet(artifact("backtest_predictions.parquet"), index=False)
    result.by_horizon.to_parquet(artifact("metrics_by_horizon.parquet"), index=False)
    result.by_category.to_parquet(artifact("metrics_by_category.parquet"), index=False)
    if not result.by_regime.empty:
        result.by_regime.to_parquet(artifact("metrics_by_regime.parquet"), index=False)
    forecaster.feature_importance.to_parquet(artifact("feature_importance.parquet"), index=False)

    verdict = (
        f"{result.selected_model} selected: WAPE {result.best_wape:.4f} vs "
        f"seasonal-naive {result.baseline.wape:.4f} "
        f"({result.wape_improvement:+.1%})"
    )
    if not result.model_beats_baseline:
        # Brief section 7.1: this is a finding to report, not a failure to hide.
        log.warning(
            "no learned model beat the seasonal-naive baseline; the baseline ships",
            extra={"context": {"verdict": verdict}},
        )
    else:
        log.info("model selected", extra={"context": {"verdict": verdict}})

    return 0


if __name__ == "__main__":
    sys.exit(main())
