"""Hyperparameter tuning pass on LightGBM and CatBoost.

Not part of the standard model suite - every model in `foresight.benchmarks`
is deliberately a generic, untuned configuration, because the point of that
module is "how would a well-known method do out of the box", not "how good
can it be made to be". This script answers the narrower follow-up question:
starting from CatBoost's win over every other model in
`reports/model_benchmark.md`, does a small grid search over LightGBM's and
CatBoost's own hyperparameters find something better, on the identical
rolling-origin folds everything else in this project is scored on?

LightGBM is tuned via `foresight.backtest.run_backtest`'s existing `params`
override (`foresight.forecast.DEFAULT_PARAMS`, merged) - the same production
path, just with different hyperparameters, so there is no risk of silently
scoring it differently from how it is actually served.

CatBoost has no such hook in `run_backtest` (its hyperparameters are a fixed
module constant in `foresight.benchmarks`, by design - a benchmark for
comparison, not a tuning exercise), so this script reuses the exact same
fold-splitting and preprocessing (`select_origins`, `_split_fold`,
`_build_preprocessor` from :mod:`foresight.backtest`/:mod:`foresight.benchmarks`)
to fit each candidate directly, rather than duplicating that logic.

Usage::

    python scripts/15_tune_gbm_hyperparameters.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from foresight.backtest import _split_fold, run_backtest, select_origins
from foresight.benchmarks import _build_preprocessor
from foresight.config import get_settings
from foresight.exceptions import ForesightError
from foresight.features import (
    build_base_features,
    build_supervised_frame,
    build_weekly_calendar,
    resolve_feature_spec,
)
from foresight.logging_setup import get_logger
from foresight.metrics import evaluate_forecast

log = get_logger("scripts.tune_gbm_hyperparameters")

#: LightGBM candidates, each merged over `foresight.forecast.DEFAULT_PARAMS`.
#: Deliberately small - a real hyperparameter search would use far more
#: combinations, but each one here is a full 6-fold backtest.
_LIGHTGBM_GRID: Final[dict[str, list[Any]]] = {
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.03, 0.05, 0.08],
    "n_estimators": [500, 900],
}

#: CatBoost candidates, replacing `foresight.benchmarks._CATBOOST_PARAMS`.
_CATBOOST_GRID: Final[dict[str, list[Any]]] = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.08],
    "iterations": [300, 600],
}

_CATBOOST_FIXED: Final[dict[str, Any]] = {
    "loss_function": "MAE",
    "allow_writing_files": False,
    "verbose": False,
}


def _grid_combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]


def _tune_lightgbm(
    supervised: pd.DataFrame, panel: pd.DataFrame, settings: Any, feature_spec: Any
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate in _grid_combinations(_LIGHTGBM_GRID):
        started = time.perf_counter()
        try:
            result, _forecaster, _ensemble = run_backtest(
                supervised,
                panel,
                settings,
                params=candidate,
                include_ensemble=False,
                include_benchmarks=False,
                feature_spec=feature_spec,
            )
        except ForesightError as exc:
            log.warning(
                "lightgbm candidate failed", extra={"context": {**candidate, "error": str(exc)}}
            )
            continue
        elapsed = time.perf_counter() - started
        results.append(
            {
                "params": candidate,
                "wape": round(result.model.wape, 6),
                "bias_relative": round(result.model.bias_relative, 6),
                "seconds": round(elapsed, 1),
            }
        )
        log.info(
            "lightgbm candidate scored",
            extra={"context": {**candidate, "wape": round(result.model.wape, 4)}},
        )
    return results


def _tune_catboost(
    supervised: pd.DataFrame, settings: Any, feature_spec: Any, origins: list[pd.Timestamp]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate in _grid_combinations(_CATBOOST_GRID):
        started = time.perf_counter()
        predictions: list[np.ndarray] = []
        actuals: list[np.ndarray] = []
        for origin in origins:
            train, test = _split_fold(supervised, origin)
            pipeline = Pipeline(
                [
                    ("preprocess", _build_preprocessor(feature_spec)),
                    (
                        "model",
                        CatBoostRegressor(
                            random_state=settings.random_seed, **candidate, **_CATBOOST_FIXED
                        ),
                    ),
                ]
            )
            pipeline.fit(
                train.loc[:, feature_spec.all_features], train["y"].to_numpy(dtype="float64")
            )
            prediction = np.clip(
                pipeline.predict(test.loc[:, feature_spec.all_features]), 0.0, None
            )
            predictions.append(prediction)
            actuals.append(test["y"].to_numpy(dtype="float64"))

        elapsed = time.perf_counter() - started
        metrics = evaluate_forecast(np.concatenate(actuals), np.concatenate(predictions))
        results.append(
            {
                "params": candidate,
                "wape": round(metrics.wape, 6),
                "bias_relative": round(metrics.bias_relative, 6),
                "seconds": round(elapsed, 1),
            }
        )
        log.info(
            "catboost candidate scored",
            extra={"context": {**candidate, "wape": round(metrics.wape, 4)}},
        )
    return results


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

    # Mirrors 10_run_all_models.py exactly: the CatBoost/LightGBM numbers this
    # script is trying to beat were produced with drivers on (if configured and
    # available), so tuning must start from the same inputs or the comparison
    # is not apples to apples.
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
    log.info(
        "resolved feature configuration",
        extra={"context": {"use_drivers": use_drivers, "rows": len(supervised)}},
    )

    origins = select_origins(
        supervised["week_start"].unique(),
        settings.horizon_weeks,
        settings.backtest_folds,
        settings.backtest_step_weeks,
        settings.min_train_weeks,
    )

    log.info(
        "tuning lightgbm",
        extra={"context": {"candidates": len(_grid_combinations(_LIGHTGBM_GRID))}},
    )
    lightgbm_results = _tune_lightgbm(supervised, panel, settings, feature_spec)

    log.info(
        "tuning catboost",
        extra={"context": {"candidates": len(_grid_combinations(_CATBOOST_GRID))}},
    )
    catboost_results = _tune_catboost(supervised, settings, feature_spec, origins)

    lightgbm_best = min(lightgbm_results, key=lambda row: row["wape"]) if lightgbm_results else None
    catboost_best = min(catboost_results, key=lambda row: row["wape"]) if catboost_results else None

    payload = {
        "lightgbm": {"grid": _LIGHTGBM_GRID, "results": lightgbm_results, "best": lightgbm_best},
        "catboost": {"grid": _CATBOOST_GRID, "results": catboost_results, "best": catboost_best},
    }
    out_path: Path = settings.artifacts_dir / "gbm_tuning.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    log.info(
        "tuning complete",
        extra={
            "context": {
                "path": str(out_path),
                "lightgbm_best_wape": lightgbm_best["wape"] if lightgbm_best else None,
                "catboost_best_wape": catboost_best["wape"] if catboost_best else None,
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
