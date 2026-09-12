"""Stacked blend of LightGBM, CatBoost and the Adaptive Ensemble.

Follows up on `reports/model_benchmark.md`, where CatBoost narrowly beat
every other model including the ensemble. A single winner is not
necessarily the best available number: LightGBM, CatBoost and the ensemble
each make somewhat different errors (different algorithms, and the ensemble
additionally routes by regime), so a weighted blend of the three can beat
all of them individually if their errors are not fully correlated.

Weights are fit on calibration folds and evaluated on a genuinely held-out
fold - the same split-sample discipline `foresight.conformal` uses - so the
reported improvement (if any) is not just the blend overfitting to the data
it was tuned on. The weight search reuses
`foresight.custom_model._simplex_grid`, the same convex-combination search
the Adaptive Ensemble itself uses to route between its own components,
rather than a second copy of that logic.

Usage::

    python scripts/16_stack_blend.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from foresight.backtest import run_backtest
from foresight.config import get_settings
from foresight.custom_model import _simplex_grid
from foresight.exceptions import ForesightError
from foresight.features import (
    build_base_features,
    build_supervised_frame,
    build_weekly_calendar,
    resolve_feature_spec,
)
from foresight.logging_setup import get_logger
from foresight.metrics import evaluate_forecast

log = get_logger("scripts.stack_blend")

#: The three components blended - LightGBM's own column is named `prediction`
#: in the backtest output, not `lightgbm`.
_COMPONENTS: Final[tuple[str, ...]] = ("prediction", "catboost", "ensemble")
_COMPONENT_LABELS: Final[dict[str, str]] = {
    "prediction": "lightgbm",
    "catboost": "catboost",
    "ensemble": "adaptive_ensemble",
}
_WEIGHT_STEP: Final[float] = 0.1


def _wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(predicted - actual).sum() / max(np.abs(actual).sum(), 1e-9))


def _best_weights(calibration: pd.DataFrame) -> tuple[tuple[float, ...], float]:
    actual = calibration["y"].to_numpy(dtype="float64")
    columns = [calibration[name].to_numpy(dtype="float64") for name in _COMPONENTS]

    best_weights: tuple[float, ...] = (1.0, 0.0, 0.0)
    best_wape = float("inf")
    for weights in _simplex_grid(len(_COMPONENTS), _WEIGHT_STEP):
        blended = sum(weight * column for weight, column in zip(weights, columns, strict=True))
        wape = _wape(actual, blended)
        if wape < best_wape:
            best_wape = wape
            best_weights = weights
    return best_weights, best_wape


def main(argv: list[str] | None = None) -> int:
    del argv  # No arguments today; kept for parity with the other scripts.

    settings = get_settings()
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

    # Mirrors 10_run_all_models.py: same driver resolution the reported
    # CatBoost/LightGBM/ensemble numbers were produced with.
    marketing_plan_path = settings.processed_dir / "marketing_clean.parquet"
    marketing_plan = (
        pd.read_parquet(marketing_plan_path) if marketing_plan_path.exists() else None
    )
    if marketing_plan is not None and marketing_plan.empty:
        marketing_plan = None
    use_drivers = settings.use_drivers and marketing_plan is not None
    feature_spec = resolve_feature_spec(use_drivers=use_drivers)
    base = build_base_features(panel, use_drivers=use_drivers)
    supervised = build_supervised_frame(
        base, weekly_calendar, horizons, use_drivers=use_drivers, marketing_plan=marketing_plan
    )

    try:
        result, _forecaster, _ensemble = run_backtest(
            supervised,
            panel,
            settings,
            include_ensemble=True,
            include_benchmarks=True,
            feature_spec=feature_spec,
        )
    except ForesightError as exc:
        log.error("backtest failed", extra={"context": {"error": str(exc)}})
        return 1

    predictions = result.predictions
    missing = set(_COMPONENTS) - set(predictions.columns)
    if missing:
        log.error(
            "backtest predictions missing required columns for blending",
            extra={"context": {"missing": sorted(missing)}},
        )
        return 1

    last_fold = int(predictions["fold"].max())
    calibration = predictions[predictions["fold"] < last_fold]
    test = predictions[predictions["fold"] == last_fold]

    best_weights, calibration_wape = _best_weights(calibration)

    test_actual = test["y"].to_numpy(dtype="float64")
    single_model_wape = {
        _COMPONENT_LABELS[name]: round(_wape(test_actual, test[name].to_numpy(dtype="float64")), 6)
        for name in _COMPONENTS
    }
    blended_test = sum(
        weight * test[name].to_numpy(dtype="float64")
        for weight, name in zip(best_weights, _COMPONENTS, strict=True)
    )
    blend_metrics = evaluate_forecast(test_actual, blended_test)

    payload: dict[str, Any] = {
        "components": [_COMPONENT_LABELS[name] for name in _COMPONENTS],
        "weights": dict(zip((_COMPONENT_LABELS[c] for c in _COMPONENTS), best_weights, strict=True)),
        "calibration_folds": sorted(calibration["fold"].unique().tolist()),
        "held_out_fold": last_fold,
        "calibration_wape": round(calibration_wape, 6),
        "held_out_single_model_wape": single_model_wape,
        "held_out_blend_wape": round(blend_metrics.wape, 6),
        "held_out_blend_bias_relative": round(blend_metrics.bias_relative, 6),
        "best_single_model_on_held_out": min(single_model_wape, key=lambda name: single_model_wape[name]),
    }

    out_path: Path = settings.artifacts_dir / "stack_blend.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        "stack blend complete",
        extra={
            "context": {
                "path": str(out_path),
                "weights": payload["weights"],
                "held_out_blend_wape": payload["held_out_blend_wape"],
                "held_out_single_model_wape": single_model_wape,
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
