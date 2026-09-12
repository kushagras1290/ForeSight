"""Rolling-origin backtesting (deliverable D3, acceptance criterion 3).

A single random train/test split is meaningless for a time series: it trains on
next month to predict last month. This module instead walks an origin forward
through history, and at each stop trains only on what was knowable then and
forecasts the following ``H`` weeks.

THE SPLIT RULE THAT MATTERS
---------------------------
At origin ``O``, a training row is admissible only if its **target week** is at
or before ``O``::

    train:  target_week <= O
    test:   week_start == O          (targets O+1 .. O+H)

Filtering on the *origin* week instead - the obvious-looking
``week_start <= O`` is the subtle bug that inflates backtest scores across the
whole industry. A row with origin ``O - 2`` and horizon 8 has its target at
``O + 6``, six weeks past the split. Training on it teaches the model outcomes
from beyond the forecast date, and the resulting WAPE is fiction.

Because that rule is easy to state and easy to get wrong, it lives in exactly
one place - :func:`_split_fold` - and is asserted in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from foresight.baseline import naive_last_value_forecast, seasonal_naive_forecast
from foresight.benchmarks import (
    fit_predict_arima,
    fit_predict_catboost,
    fit_predict_ets,
    fit_predict_linear_regression,
    fit_predict_prophet,
    fit_predict_random_forest,
    fit_predict_sarima,
    fit_predict_xgboost,
)
from foresight.config import Settings, get_settings
from foresight.custom_model import AdaptiveDemandEnsemble, train_ensemble
from foresight.exceptions import InsufficientHistoryError
from foresight.features import FEATURE_SPEC, FeatureSpec
from foresight.forecast import TrainedForecaster, train_forecaster
from foresight.logging_setup import get_logger
from foresight.metrics import (
    ForecastMetrics,
    evaluate_forecast,
    interval_coverage,
    metrics_by_group,
    pinball_loss,
)

#: Benchmark candidates' names as they appear in reports, in a fixed order so
#: every output (fold logs, the summary payload, the comparison report) lists
#: them the same way.
BENCHMARK_MODEL_NAMES: tuple[str, ...] = (
    "arima",
    "sarima",
    "ets",
    "prophet",
    "linear_regression",
    "random_forest",
    "xgboost",
    "catboost",
)

__all__ = ["BacktestResult", "FoldResult", "run_backtest", "select_origins"]

log = get_logger(__name__)


@dataclass(slots=True)
class FoldResult:
    """One origin's worth of out-of-sample results."""

    fold: int
    origin_week: pd.Timestamp
    train_rows: int
    test_rows: int
    model: ForecastMetrics
    baseline: ForecastMetrics
    naive: ForecastMetrics
    ensemble: ForecastMetrics | None = None
    benchmarks: dict[str, ForecastMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "fold": self.fold,
            "origin_week": str(self.origin_week.date()),
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "gbm_wape": round(self.model.wape, 6),
            "baseline_wape": round(self.baseline.wape, 6),
            "naive_wape": round(self.naive.wape, 6),
            "gbm_bias_relative": round(self.model.bias_relative, 6),
            "baseline_bias_relative": round(self.baseline.bias_relative, 6),
        }
        for name, metrics in self.benchmarks.items():
            payload[f"{name}_wape"] = round(metrics.wape, 6)
        if self.ensemble is not None:
            payload["ensemble_wape"] = round(self.ensemble.wape, 6)
            payload["ensemble_bias_relative"] = round(self.ensemble.bias_relative, 6)
        return payload


@dataclass(slots=True)
class BacktestResult:
    """Aggregated backtest across every fold."""

    folds: list[FoldResult]
    predictions: pd.DataFrame
    model: ForecastMetrics
    baseline: ForecastMetrics
    naive: ForecastMetrics
    by_horizon: pd.DataFrame
    by_category: pd.DataFrame
    interval_coverage: float
    baseline_fallback_rate: float
    ensemble: ForecastMetrics | None = None
    ensemble_interval_coverage: float = float("nan")
    by_regime: pd.DataFrame = field(default_factory=pd.DataFrame)
    settings_snapshot: dict[str, Any] = field(default_factory=dict)
    #: The 4 standard-library benchmarks (ARIMA, ETS, linear regression, random
    #: forest), keyed by name. Deliberately kept out of `candidates` below:
    #: they exist to answer "how would a well-known method have done here",
    #: not to compete for which model ships - `04_score_risk.py` only knows
    #: how to serve a `TrainedForecaster` or an `AdaptiveDemandEnsemble`, so a
    #: benchmark winning on WAPE must never make it `selected_model`.
    benchmarks: dict[str, ForecastMetrics] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def candidates(self) -> dict[str, ForecastMetrics]:
        """Forecasts eligible to ship, keyed by the name used in reports.

        See the `benchmarks` field docstring for why the 4 standard-library
        comparison models are excluded from this property specifically.
        """
        entries = {
            "seasonal_naive": self.baseline,
            "lightgbm": self.model,
        }
        if self.ensemble is not None:
            entries["adaptive_ensemble"] = self.ensemble
        return entries

    @property
    def all_models(self) -> dict[str, ForecastMetrics]:
        """Every model this backtest scored, including the non-shippable
        benchmarks - for reporting/comparison, never for model selection."""
        return {**self.candidates, "naive_last_value": self.naive, **self.benchmarks}

    @property
    def best_wape(self) -> float:
        return min(
            (metrics.wape for metrics in self.candidates.values() if np.isfinite(metrics.wape)),
            default=float("nan"),
        )

    @property
    def selected_model(self) -> str:
        """Which forecast actually ships.

        Whichever candidate posts the lowest backtest WAPE - including the
        baseline. Brief section 7.1: if no learned model beats seasonal-naive,
        the baseline ships and that is reported as a finding rather than hidden.
        """
        finite = {
            name: metrics.wape
            for name, metrics in self.candidates.items()
            if np.isfinite(metrics.wape)
        }
        if not finite:
            return "seasonal_naive"
        return min(finite, key=lambda name: finite[name])

    def improvement_over_baseline(self, name: str) -> float:
        """Relative WAPE reduction of one candidate against the baseline."""
        metrics = self.candidates.get(name)
        if metrics is None or not np.isfinite(self.baseline.wape) or self.baseline.wape <= 0:
            return float("nan")
        return (self.baseline.wape - metrics.wape) / self.baseline.wape

    @property
    def wape_improvement(self) -> float:
        """Headline improvement: the selected model against the baseline."""
        return self.improvement_over_baseline(self.selected_model)

    @property
    def model_beats_baseline(self) -> bool:
        return self.selected_model != "seasonal_naive"

    @property
    def selected_interval_coverage(self) -> float:
        if self.selected_model == "adaptive_ensemble":
            return self.ensemble_interval_coverage
        return self.interval_coverage

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "folds": len(self.folds),
            "test_observations": self.model.n_observations,
            "baseline_wape": round(self.baseline.wape, 6),
            "naive_wape": round(self.naive.wape, 6),
            "gbm_wape": round(self.model.wape, 6),
            "gbm_improvement_vs_baseline": round(self.improvement_over_baseline("lightgbm"), 6),
            "gbm_bias_relative": round(self.model.bias_relative, 6),
            "gbm_interval_coverage": round(self.interval_coverage, 6),
            "baseline_bias_relative": round(self.baseline.bias_relative, 6),
            "baseline_fallback_rate": round(self.baseline_fallback_rate, 6),
            "selected_model": self.selected_model,
            "selected_model_beats_baseline": self.model_beats_baseline,
            "selected_wape": round(self.best_wape, 6),
            "selected_improvement_vs_baseline": round(self.wape_improvement, 6),
            "selected_interval_coverage": round(self.selected_interval_coverage, 6),
            "per_fold": [fold.to_dict() for fold in self.folds],
        }
        if self.ensemble is not None:
            payload.update(
                {
                    "ensemble_wape": round(self.ensemble.wape, 6),
                    "ensemble_improvement_vs_baseline": round(
                        self.improvement_over_baseline("adaptive_ensemble"), 6
                    ),
                    "ensemble_improvement_vs_gbm": round(
                        (self.model.wape - self.ensemble.wape) / self.model.wape, 6
                    )
                    if np.isfinite(self.model.wape) and self.model.wape > 0
                    else float("nan"),
                    "ensemble_bias_relative": round(self.ensemble.bias_relative, 6),
                    "ensemble_mape": round(self.ensemble.mape, 6),
                    "ensemble_mape_coverage": round(self.ensemble.mape_coverage, 6),
                    "ensemble_interval_coverage": round(self.ensemble_interval_coverage, 6),
                }
            )
        payload.update(self.settings_snapshot)
        return payload


def select_origins(
    weeks: np.ndarray,
    horizon: int,
    folds: int,
    step: int,
    min_train_weeks: int,
) -> list[pd.Timestamp]:
    """Choose evenly-spaced backtest origins, latest first then reversed.

    The last usable origin is ``horizon`` weeks before the end of history, so
    that every horizon in the fold has a real observation to score against.

    Raises:
        InsufficientHistoryError: if history cannot support the configuration.
    """
    ordered = np.sort(np.asarray(weeks))
    last_index = len(ordered) - 1 - horizon
    if last_index < 0:
        raise InsufficientHistoryError(
            f"History has {len(ordered)} weeks, which cannot cover a {horizon}-week horizon."
        )

    indices = [last_index - fold * step for fold in range(folds)]
    earliest = min(indices)
    if earliest < min_train_weeks:
        raise InsufficientHistoryError(
            f"Backtest needs an origin at week index {earliest}, but the first "
            f"{min_train_weeks} weeks are reserved for training. Reduce folds "
            f"({folds}), step ({step}) or min_train_weeks ({min_train_weeks})."
        )

    return [pd.Timestamp(ordered[index]) for index in sorted(indices)]


def _split_fold(
    supervised: pd.DataFrame,
    origin: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a supervised frame at ``origin``.

    Train rows are those whose *target* has already been observed by the origin.
    Test rows are the forecasts actually made at the origin. See the module
    docstring for why the target week, not the origin week, is the right filter.
    """
    train = supervised[supervised["target_week"] <= origin]
    test = supervised[supervised["week_start"] == origin]
    return train, test


def run_backtest(
    supervised: pd.DataFrame,
    panel: pd.DataFrame,
    settings: Settings | None = None,
    *,
    params: dict[str, Any] | None = None,
    include_ensemble: bool = True,
    include_benchmarks: bool = False,
    feature_spec: FeatureSpec = FEATURE_SPEC,
) -> tuple[BacktestResult, TrainedForecaster, AdaptiveDemandEnsemble | None]:
    """Run rolling-origin cross-validation over every candidate forecast.

    All candidates - seasonal-naive, the GBM and the adaptive ensemble - are
    scored on identical folds and identical rows, so the comparison between them
    is like for like.

    Args:
        supervised: The full supervised frame.
        panel: The weekly panel, needed for the ensemble's TSB component.
        settings: Configuration.
        params: LightGBM overrides.
        include_ensemble: Whether to fit and score the adaptive ensemble.
        include_benchmarks: Whether to also fit and score the 8 standard
            comparison models (:mod:`foresight.benchmarks`) on these same
            folds. Off by default - fitting a per-SKU ARIMA, SARIMA, ETS and
            Prophet model for every SKU on every fold multiplies the runtime
            of an already expensive backtest, and the normal training/serving
            path
            (`scripts/03_train_backtest.py`, `04_score_risk.py`) has no use
            for them. Turned on explicitly by
            `scripts/10_run_all_models.py`.
        feature_spec: The feature contract to train against. Must match the
            contract the supervised frame was built under - passing the
            driver-extended spec against a history-only frame fails loudly in
            :func:`foresight.forecast.train_forecaster` rather than silently
            training on fewer columns than it reports.

    Returns:
        The aggregated result, a final GBM refitted on all labelled rows, and a
        final ensemble (or ``None`` when it was not requested).
    """
    settings = settings or get_settings()
    horizon = settings.horizon_weeks

    weeks = np.sort(supervised["week_start"].unique())
    origins = select_origins(
        weeks,
        horizon=horizon,
        folds=settings.backtest_folds,
        step=settings.backtest_step_weeks,
        min_train_weeks=settings.min_train_weeks,
    )

    log.info(
        "starting rolling-origin backtest",
        extra={
            "context": {
                "folds": len(origins),
                "horizon": horizon,
                "first_origin": str(origins[0].date()),
                "last_origin": str(origins[-1].date()),
            }
        },
    )

    fold_results: list[FoldResult] = []
    prediction_frames: list[pd.DataFrame] = []
    fallback_rates: list[float] = []

    for fold_index, origin in enumerate(origins, start=1):
        train, test = _split_fold(supervised, origin)

        if train.empty or test.empty:
            log.warning(
                "skipping empty fold",
                extra={
                    "context": {
                        "fold": fold_index,
                        "origin": str(origin.date()),
                        "train_rows": len(train),
                        "test_rows": len(test),
                    }
                },
            )
            continue

        forecaster = train_forecaster(train, settings, params=params, feature_spec=feature_spec)
        predictions = forecaster.predict(test)

        baseline = seasonal_naive_forecast(test)
        fallback_rates.append(baseline.fallback_rate)
        naive_predictions = naive_last_value_forecast(test)

        actual = test["y"].to_numpy(dtype="float64")
        fold_frame = test[
            ["sku_id", "week_start", "target_week", "horizon", "category", "subcategory", "y"]
        ].copy()
        fold_frame["fold"] = fold_index
        fold_frame["origin_week"] = origin
        fold_frame["prediction"] = predictions["prediction"].to_numpy()
        fold_frame["prediction_lower"] = predictions["prediction_lower"].to_numpy()
        fold_frame["prediction_upper"] = predictions["prediction_upper"].to_numpy()
        fold_frame["baseline"] = baseline.predictions
        fold_frame["naive"] = naive_predictions

        ensemble_metrics: ForecastMetrics | None = None
        if include_ensemble:
            # The fold's GBM is handed over rather than refitted, so the
            # ensemble is judged on exactly the same GBM the GBM row reports.
            fold_ensemble = train_ensemble(train, panel, settings, gbm=forecaster)
            ensemble_predictions = fold_ensemble.predict(test)
            fold_frame["ensemble"] = ensemble_predictions["prediction"].to_numpy()
            fold_frame["ensemble_lower"] = ensemble_predictions["prediction_lower"].to_numpy()
            fold_frame["ensemble_upper"] = ensemble_predictions["prediction_upper"].to_numpy()
            fold_frame["regime"] = ensemble_predictions["regime"].to_numpy()
            ensemble_metrics = evaluate_forecast(actual, ensemble_predictions["prediction"])

        benchmark_metrics: dict[str, ForecastMetrics] = {}
        if include_benchmarks:
            benchmark_predictions = {
                "arima": fit_predict_arima(panel, test, origin),
                "sarima": fit_predict_sarima(panel, test, origin),
                "ets": fit_predict_ets(panel, test, origin),
                "prophet": fit_predict_prophet(panel, test, origin),
                "linear_regression": fit_predict_linear_regression(train, test, feature_spec),
                "random_forest": fit_predict_random_forest(
                    train, test, feature_spec, settings.random_seed
                ),
                "xgboost": fit_predict_xgboost(train, test, feature_spec, settings.random_seed),
                "catboost": fit_predict_catboost(train, test, feature_spec, settings.random_seed),
            }
            for name, values in benchmark_predictions.items():
                fold_frame[name] = values
                benchmark_metrics[name] = evaluate_forecast(actual, values)

        prediction_frames.append(fold_frame)

        fold_result = FoldResult(
            fold=fold_index,
            origin_week=origin,
            train_rows=len(train),
            test_rows=len(test),
            model=evaluate_forecast(actual, predictions["prediction"]),
            baseline=evaluate_forecast(actual, baseline.predictions),
            naive=evaluate_forecast(actual, naive_predictions),
            ensemble=ensemble_metrics,
            benchmarks=benchmark_metrics,
        )
        fold_results.append(fold_result)

        log.info(
            "fold complete",
            extra={
                "context": {
                    "fold": fold_index,
                    "origin": str(origin.date()),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "gbm_wape": round(fold_result.model.wape, 4),
                    "baseline_wape": round(fold_result.baseline.wape, 4),
                    "ensemble_wape": round(ensemble_metrics.wape, 4)
                    if ensemble_metrics is not None
                    else None,
                }
            },
        )

    if not fold_results:
        raise InsufficientHistoryError("Every backtest fold was empty; check the configuration.")

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    actual = all_predictions["y"].to_numpy(dtype="float64")
    lower_quantile, upper_quantile = settings.quantile_levels

    model_metrics = evaluate_forecast(
        actual,
        all_predictions["prediction"],
        extras={
            "pinball_lower": pinball_loss(
                actual, all_predictions["prediction_lower"], lower_quantile
            ),
            "pinball_upper": pinball_loss(
                actual, all_predictions["prediction_upper"], upper_quantile
            ),
        },
    )

    coverage = interval_coverage(
        actual, all_predictions["prediction_lower"], all_predictions["prediction_upper"]
    )

    by_horizon = metrics_by_group(all_predictions, "horizon")
    baseline_by_horizon = metrics_by_group(
        all_predictions.assign(prediction=all_predictions["baseline"]), "horizon"
    )[["horizon", "wape"]].rename(columns={"wape": "baseline_wape"})
    by_horizon = by_horizon.merge(baseline_by_horizon, on="horizon", how="left")

    by_category = metrics_by_group(all_predictions, "category")
    baseline_by_category = metrics_by_group(
        all_predictions.assign(prediction=all_predictions["baseline"]), "category"
    )[["category", "wape"]].rename(columns={"wape": "baseline_wape"})
    by_category = by_category.merge(baseline_by_category, on="category", how="left")

    # --- Ensemble aggregates ------------------------------------------------ #
    ensemble_metrics: ForecastMetrics | None = None
    ensemble_coverage = float("nan")
    by_regime = pd.DataFrame()

    if include_ensemble and "ensemble" in all_predictions.columns:
        ensemble_metrics = evaluate_forecast(actual, all_predictions["ensemble"])
        ensemble_coverage = interval_coverage(
            actual, all_predictions["ensemble_lower"], all_predictions["ensemble_upper"]
        )

        # Per-regime comparison. This is the table that shows whether the
        # routing earned its complexity or whether the GBM was fine alone.
        by_regime = metrics_by_group(
            all_predictions.assign(prediction=all_predictions["ensemble"]), "regime"
        ).rename(columns={"wape": "ensemble_wape", "bias_relative": "ensemble_bias_relative"})
        gbm_by_regime = metrics_by_group(all_predictions, "regime")[["regime", "wape"]].rename(
            columns={"wape": "gbm_wape"}
        )
        baseline_by_regime = metrics_by_group(
            all_predictions.assign(prediction=all_predictions["baseline"]), "regime"
        )[["regime", "wape"]].rename(columns={"wape": "baseline_wape"})
        by_regime = by_regime.merge(gbm_by_regime, on="regime", how="left").merge(
            baseline_by_regime, on="regime", how="left"
        )

        ensemble_by_horizon = metrics_by_group(
            all_predictions.assign(prediction=all_predictions["ensemble"]), "horizon"
        )[["horizon", "wape"]].rename(columns={"wape": "ensemble_wape"})
        by_horizon = by_horizon.merge(ensemble_by_horizon, on="horizon", how="left")

        ensemble_by_category = metrics_by_group(
            all_predictions.assign(prediction=all_predictions["ensemble"]), "category"
        )[["category", "wape"]].rename(columns={"wape": "ensemble_wape"})
        by_category = by_category.merge(ensemble_by_category, on="category", how="left")

    # --- Benchmark aggregates ------------------------------------------------ #
    benchmark_metrics: dict[str, ForecastMetrics] = {}
    if include_benchmarks:
        for name in BENCHMARK_MODEL_NAMES:
            if name in all_predictions.columns:
                benchmark_metrics[name] = evaluate_forecast(actual, all_predictions[name])

    result = BacktestResult(
        folds=fold_results,
        predictions=all_predictions,
        model=model_metrics,
        baseline=evaluate_forecast(actual, all_predictions["baseline"]),
        naive=evaluate_forecast(actual, all_predictions["naive"]),
        by_horizon=by_horizon,
        by_category=by_category,
        interval_coverage=coverage,
        baseline_fallback_rate=float(np.mean(fallback_rates)) if fallback_rates else 0.0,
        ensemble=ensemble_metrics,
        ensemble_interval_coverage=ensemble_coverage,
        by_regime=by_regime,
        benchmarks=benchmark_metrics,
        settings_snapshot={
            "horizon_weeks": settings.horizon_weeks,
            "backtest_folds": settings.backtest_folds,
            "backtest_step_weeks": settings.backtest_step_weeks,
            "min_train_weeks": settings.min_train_weeks,
            "interval_coverage_target": settings.interval_coverage,
            "random_seed": settings.random_seed,
            "n_features": len(feature_spec.all_features),
            "feature_set": "history_plus_drivers"
            if len(feature_spec.all_features) > len(FEATURE_SPEC.all_features)
            else "history_only",
        },
    )

    log.info(
        "backtest complete",
        extra={
            "context": {
                "gbm_wape": round(result.model.wape, 4),
                "ensemble_wape": round(ensemble_metrics.wape, 4)
                if ensemble_metrics is not None
                else None,
                "baseline_wape": round(result.baseline.wape, 4),
                "selected": result.selected_model,
                "improvement": round(result.wape_improvement, 4),
            }
        },
    )

    # Final production models: refit on every labelled row. The backtest has
    # already established what accuracy to expect; this just gives the shipped
    # models the most history possible.
    final_forecaster = train_forecaster(
        supervised, settings, params=params, feature_spec=feature_spec
    )
    final_ensemble = (
        train_ensemble(supervised, panel, settings, gbm=final_forecaster)
        if include_ensemble
        else None
    )

    return result, final_forecaster, final_ensemble
