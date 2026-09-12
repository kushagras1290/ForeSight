"""Fit Conformal Interval Recalibration (custom model 7) and measure it.

Uses every backtest fold except the last as the calibration set and the last
fold as a genuinely held-out test - the calibration is never allowed to see
the fold it is then evaluated against, or the reported coverage improvement
would not be honest.

Usage::

    python scripts/14_fit_conformal_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from foresight.config import get_settings
from foresight.conformal import apply_conformal_calibration, fit_conformal_calibration
from foresight.exceptions import ForesightError
from foresight.logging_setup import get_logger
from foresight.metrics import interval_coverage

log = get_logger("scripts.fit_conformal_calibration")


def _coverage_by_regime(frame: pd.DataFrame, lower_col: str, upper_col: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for regime, group in frame.groupby("regime", sort=True):
        result[str(regime)] = round(
            interval_coverage(group["y"], group[lower_col], group[upper_col]), 4
        )
    return result


def main(argv: list[str] | None = None) -> int:
    del argv  # No arguments today; kept for parity with the other scripts.

    settings = get_settings()
    predictions_path = settings.artifacts_dir / "backtest_predictions.parquet"
    if not predictions_path.exists():
        log.error(
            "backtest predictions missing",
            extra={
                "context": {
                    "path": str(predictions_path),
                    "hint": "run scripts/03_train_backtest.py",
                }
            },
        )
        return 1

    predictions = pd.read_parquet(predictions_path)
    required = {"fold", "y", "ensemble", "ensemble_lower", "ensemble_upper", "regime"}
    missing = required - set(predictions.columns)
    if missing:
        log.error(
            "backtest predictions missing required columns for calibration",
            extra={"context": {"missing": sorted(missing)}},
        )
        return 1

    last_fold = int(predictions["fold"].max())
    calibration_rows = predictions[predictions["fold"] < last_fold]
    test_rows = predictions[predictions["fold"] == last_fold]

    try:
        calibration = fit_conformal_calibration(
            calibration_rows, target_coverage=settings.interval_coverage
        )
    except ForesightError as exc:
        log.error("fit failed", extra={"context": {"error": str(exc)}})
        return 1

    recalibrated = apply_conformal_calibration(calibration, test_rows)

    before_overall = interval_coverage(
        test_rows["y"], test_rows["ensemble_lower"], test_rows["ensemble_upper"]
    )
    after_overall = interval_coverage(
        recalibrated["y"], recalibrated["recalibrated_lower"], recalibrated["recalibrated_upper"]
    )

    payload: dict[str, Any] = {
        "target_coverage": calibration.target_coverage,
        "scale_by_regime": calibration.scale_by_regime,
        "diagnostics": calibration.diagnostics,
        "held_out_fold": last_fold,
        "held_out_rows": int(len(test_rows)),
        "coverage_before": round(before_overall, 4),
        "coverage_after": round(after_overall, 4),
        "coverage_before_by_regime": _coverage_by_regime(
            test_rows, "ensemble_lower", "ensemble_upper"
        ),
        "coverage_after_by_regime": _coverage_by_regime(
            recalibrated, "recalibrated_lower", "recalibrated_upper"
        ),
    }

    out_path: Path = settings.artifacts_dir / "conformal_calibration.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        "conformal calibration artifact written",
        extra={
            "context": {
                "path": str(out_path),
                "target_coverage": payload["target_coverage"],
                "coverage_before": payload["coverage_before"],
                "coverage_after": payload["coverage_after"],
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
