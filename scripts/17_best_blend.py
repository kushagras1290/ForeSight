"""The best blend this project can build: tuned LightGBM + tuned CatBoost +
the Adaptive Ensemble, blended with per-regime weights.

Builds on two prior experiments rather than starting over:

* `scripts/15_tune_gbm_hyperparameters.py` found LightGBM's and CatBoost's
  best hyperparameters on this dataset (read back from
  `artifacts/gbm_tuning.json`) - this script uses those, not the untuned
  defaults `scripts/16_stack_blend.py` used.
* `scripts/16_stack_blend.py` showed a single global blend weight already
  beats every individual model on a held-out fold. This goes one step
  further: per-regime weights, the same idea the Adaptive Ensemble itself is
  built on (`foresight.custom_model` routes GBM/TSB/seasonal-profile by
  regime because steady and volatile demand fail differently) - applied here
  to the *blend*, not just inside the ensemble.

Honesty check built in: this dataset's backtest folds are heavily
steady-regime (per `reports/model_benchmark.md`'s `by_regime` breakdown), so
a per-regime blend is not guaranteed to beat a single global weight by much -
if a regime lacks enough calibration rows, it falls back to the pooled
weight rather than fitting an unstable one, the same pattern
`foresight.conformal`/`foresight.promotions`/`foresight.substitution` use.

Tuned LightGBM's params also feed the ensemble automatically -
`foresight.backtest.run_backtest`'s `params` override reaches the GBM that
`foresight.custom_model.train_ensemble` itself blends from
(`gbm=forecaster`), so this is not a partially-tuned comparison.

Usage::

    python scripts/17_best_blend.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline

from foresight.backtest import _split_fold, run_backtest, select_origins
from foresight.benchmarks import _build_preprocessor
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

log = get_logger("scripts.best_blend")

_COMPONENTS: Final[tuple[str, ...]] = ("prediction", "catboost_tuned", "ensemble")
_COMPONENT_LABELS: Final[dict[str, str]] = {
    "prediction": "lightgbm_tuned",
    "catboost_tuned": "catboost_tuned",
    "ensemble": "adaptive_ensemble",
}
_POOLED: Final[str] = "__POOLED__"
_WEIGHT_STEP: Final[float] = 0.1
#: A regime needs this many calibration rows before its own weights are
#: fitted; below it, the pooled weights are used - same threshold
#: `foresight.conformal`/`foresight.substitution` use.
_MIN_REGIME_OBSERVATIONS: Final[int] = 30
_CATBOOST_FIXED: Final[dict[str, Any]] = {
    "loss_function": "MAE",
    "allow_writing_files": False,
    "verbose": False,
}


def _wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(predicted - actual).sum() / max(np.abs(actual).sum(), 1e-9))


def _tuned_catboost_predictions(
    supervised: pd.DataFrame, feature_spec: Any, origins: list[pd.Timestamp], params: dict[str, Any], seed: int
) -> pd.DataFrame:
    """Tuned CatBoost, fit per fold - same discipline as every other candidate.

    Returns one row per (sku_id, fold, horizon) so it can be safely joined
    onto `run_backtest`'s output rather than trusted to align positionally.
    """
    frames: list[pd.DataFrame] = []
    for fold_index, origin in enumerate(origins, start=1):
        train, test = _split_fold(supervised, origin)
        pipeline = Pipeline(
            [
                ("preprocess", _build_preprocessor(feature_spec)),
                ("model", CatBoostRegressor(random_state=seed, **params, **_CATBOOST_FIXED)),
            ]
        )
        pipeline.fit(train.loc[:, feature_spec.all_features], train["y"].to_numpy(dtype="float64"))
        prediction = np.clip(pipeline.predict(test.loc[:, feature_spec.all_features]), 0.0, None)
        frames.append(
            pd.DataFrame(
                {
                    "sku_id": test["sku_id"].to_numpy(),
                    "fold": fold_index,
                    "horizon": test["horizon"].to_numpy(),
                    "catboost_tuned": prediction,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _fit_regime_weights(calibration: pd.DataFrame) -> dict[str, tuple[float, ...]]:
    """Per-regime convex blend weights, pooled fallback below the row threshold."""
    columns = {name: calibration[name].to_numpy(dtype="float64") for name in _COMPONENTS}
    actual = calibration["y"].to_numpy(dtype="float64")

    def best_weights(mask: np.ndarray) -> tuple[float, ...]:
        best: tuple[float, ...] = (0.0, 0.0, 1.0)
        best_wape = float("inf")
        for weights in _simplex_grid(len(_COMPONENTS), _WEIGHT_STEP):
            blended = sum(
                weight * columns[name][mask] for weight, name in zip(weights, _COMPONENTS, strict=True)
            )
            wape = _wape(actual[mask], blended)
            if wape < best_wape:
                best_wape = wape
                best = weights
        return best

    pooled_mask = np.ones(len(calibration), dtype=bool)
    weights_by_regime: dict[str, tuple[float, ...]] = {_POOLED: best_weights(pooled_mask)}
    for regime, group in calibration.groupby("regime"):
        if len(group) < _MIN_REGIME_OBSERVATIONS:
            continue
        mask = (calibration["regime"] == regime).to_numpy()
        weights_by_regime[str(regime)] = best_weights(mask)
    return weights_by_regime


def _apply_regime_weights(
    weights_by_regime: dict[str, tuple[float, ...]], frame: pd.DataFrame
) -> np.ndarray:
    blended = np.zeros(len(frame), dtype="float64")
    regimes = frame["regime"].astype(str).to_numpy()
    for regime in np.unique(regimes):
        weights = weights_by_regime.get(regime, weights_by_regime[_POOLED])
        mask = regimes == regime
        blended[mask] = sum(
            weight * frame.loc[mask, name].to_numpy(dtype="float64")
            for weight, name in zip(weights, _COMPONENTS, strict=True)
        )
    return blended


def main(argv: list[str] | None = None) -> int:
    del argv  # No arguments today; kept for parity with the other scripts.

    settings = get_settings()
    tuning_path = settings.artifacts_dir / "gbm_tuning.json"
    panel_path = settings.processed_dir / "weekly_panel.parquet"
    calendar_path = settings.processed_dir / "calendar.parquet"
    for path, hint in (
        (tuning_path, "run scripts/15_tune_gbm_hyperparameters.py"),
        (panel_path, "run scripts/01_run_pipeline.py"),
        (calendar_path, "run scripts/01_run_pipeline.py"),
    ):
        if not path.exists():
            log.error("input missing", extra={"context": {"path": str(path), "hint": hint}})
            return 1

    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    lightgbm_params = tuning["lightgbm"]["best"]["params"]
    catboost_params = tuning["catboost"]["best"]["params"]
    log.info(
        "using tuned hyperparameters",
        extra={"context": {"lightgbm": lightgbm_params, "catboost": catboost_params}},
    )

    panel = pd.read_parquet(panel_path)
    calendar = pd.read_parquet(calendar_path)
    weekly_calendar = build_weekly_calendar(calendar)
    horizons = tuple(range(1, settings.horizon_weeks + 1))

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
            params=lightgbm_params,
            include_ensemble=True,
            include_benchmarks=False,
            feature_spec=feature_spec,
        )
    except ForesightError as exc:
        log.error("backtest failed", extra={"context": {"error": str(exc)}})
        return 1

    origins = select_origins(
        supervised["week_start"].unique(),
        settings.horizon_weeks,
        settings.backtest_folds,
        settings.backtest_step_weeks,
        settings.min_train_weeks,
    )
    catboost_predictions = _tuned_catboost_predictions(
        supervised, feature_spec, origins, catboost_params, settings.random_seed
    )

    combined = result.predictions.merge(
        catboost_predictions, on=["sku_id", "fold", "horizon"], how="inner", validate="one_to_one"
    )
    if len(combined) != len(result.predictions):
        log.error(
            "catboost predictions did not align with the backtest frame",
            extra={
                "context": {
                    "backtest_rows": len(result.predictions),
                    "joined_rows": len(combined),
                }
            },
        )
        return 1

    last_fold = int(combined["fold"].max())
    calibration = combined[combined["fold"] < last_fold]
    test = combined[combined["fold"] == last_fold]

    weights_by_regime = _fit_regime_weights(calibration)
    blended_test = _apply_regime_weights(weights_by_regime, test)
    blend_metrics = evaluate_forecast(test["y"].to_numpy(dtype="float64"), blended_test)

    single_model_wape = {
        _COMPONENT_LABELS[name]: round(
            _wape(test["y"].to_numpy(dtype="float64"), test[name].to_numpy(dtype="float64")), 6
        )
        for name in _COMPONENTS
    }

    payload: dict[str, Any] = {
        "lightgbm_params": lightgbm_params,
        "catboost_params": catboost_params,
        "components": [_COMPONENT_LABELS[name] for name in _COMPONENTS],
        "weights_by_regime": {
            regime: dict(zip((_COMPONENT_LABELS[c] for c in _COMPONENTS), weights, strict=True))
            for regime, weights in weights_by_regime.items()
        },
        "calibration_folds": sorted(calibration["fold"].unique().tolist()),
        "held_out_fold": last_fold,
        "held_out_single_model_wape": single_model_wape,
        "held_out_blend_wape": round(blend_metrics.wape, 6),
        "held_out_blend_bias_relative": round(blend_metrics.bias_relative, 6),
        "best_single_model_on_held_out": min(single_model_wape, key=lambda name: single_model_wape[name]),
    }

    out_path: Path = settings.artifacts_dir / "best_blend.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        "best blend complete",
        extra={
            "context": {
                "path": str(out_path),
                "held_out_blend_wape": payload["held_out_blend_wape"],
                "held_out_single_model_wape": single_model_wape,
                "regimes_with_own_weights": [r for r in weights_by_regime if r != _POOLED],
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
