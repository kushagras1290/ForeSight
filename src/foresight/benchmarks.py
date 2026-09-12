"""Standard, off-the-shelf forecasting methods, scored on the same folds.

Everywhere else in this project the forecast comes from a single global
LightGBM model (:mod:`foresight.forecast`) or the custom models built on top
of it (:mod:`foresight.custom_model` and friends). This module answers a
different, narrower question: how would well-known, textbook methods have
done on the exact same rolling-origin folds and the exact same rows?

Eight methods, chosen to cover the three families a forecasting comparison is
expected to include:

* **ARIMA**, **SARIMA**, **Exponential Smoothing (ETS/Holt)** and **Prophet** -
  classical per-series statistical forecasting, fit independently for each
  SKU. ``reports/model_comparison.md`` already explains why these were not
  chosen for production (200 series of ~100-190 weeks each is too little
  history to fit a per-series model well, and none of the four pools across
  SKUs the way the GBM does) - this module is what makes that argument
  checkable against real numbers instead of taken on trust. SARIMA is ARIMA
  plus an explicit seasonal term, included specifically to test whether
  seasonality alone closes plain ARIMA's gap to the GBM on this data.
* **Linear Regression**, **Random Forest**, **XGBoost** and **CatBoost** -
  standard scikit-learn/XGBoost/CatBoost regressors, trained on the
  *identical* feature matrix the GBM uses
  (:data:`foresight.features.FEATURE_SPEC`), so any accuracy difference is
  attributable to the estimator, not to different inputs. The three tree
  ensembles sit in this group rather than with LightGBM's own family because
  the point of this module is a fair, identical-input comparison - each is
  scored the same way as Linear Regression, not given any advantage LightGBM
  itself does not also get.

All four per-series models degrade to a last-value forecast when a series has
too little history to fit - a SKU with three weeks of history cannot support
an ARIMA(1,1,1) or a Prophet trend/seasonality decomposition, and refusing to
forecast at all would silently drop it from the comparison rather than report
an honest (bad) number for it.

PARALLELISM
-----------
Fitting one of the four per-series models is independent SKU to SKU, so the
per-SKU loop for each is parallelised with :mod:`joblib` - the dominant cost
at real catalogue sizes (490 SKUs x 4 classical models x 6 folds is ~12,000
fits) is otherwise a plain sequential Python loop. This is deliberately
*not* extended to the fold loop in :mod:`foresight.backtest`: each fold
retrains the full LightGBM model and ensemble on the whole panel, which is
already memory-heavy, and parallelising across folds would multiply that
footprint by the fold count - a real risk on a modest machine, not just a
theoretical one.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from joblib import Parallel, delayed
from prophet import Prophet
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from foresight.features import FeatureSpec
from foresight.logging_setup import get_logger

__all__ = [
    "fit_predict_arima",
    "fit_predict_catboost",
    "fit_predict_ets",
    "fit_predict_linear_regression",
    "fit_predict_prophet",
    "fit_predict_random_forest",
    "fit_predict_sarima",
    "fit_predict_xgboost",
]

log = get_logger(__name__)

# Prophet's cmdstanpy backend logs an "optimization terminated normally" INFO
# line per fit by default. Fitting one model per SKU per fold means thousands
# of these in a full backtest - harmless, but they would drown out this
# project's own structured logs. Silenced once, at import time.
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

#: Below this many weekly observations, a per-SKU classical model has too
#: little to fit against and falls back to a last-value forecast instead of
#: an unstable one. Chosen as roughly a quarter-year of weekly history.
_MIN_CLASSICAL_OBSERVATIONS: Final[int] = 10

#: A full annual cycle, needed before ETS attempts a seasonal component at
#: all - almost no SKU in this dataset's ~2 year history reaches two cycles
#: (the minimum a seasonal fit needs), so this mostly documents why the
#: seasonal term is skipped rather than switching it on in practice.
_SEASONAL_PERIOD_WEEKS: Final[int] = 52
_MIN_SEASONAL_OBSERVATIONS: Final[int] = 2 * _SEASONAL_PERIOD_WEEKS

#: Fitting one tiny ARIMA/ETS per SKU per fold occasionally fails to converge
#: on a short or oddly-shaped series - a singular matrix, a non-invertible MA
#: term. Caught narrowly (never bare Exception) so a fitting failure on one
#: SKU degrades that SKU's forecast rather than aborting the whole backtest.
_FIT_FAILURE_ERRORS: Final[tuple[type[Exception], ...]] = (ValueError, np.linalg.LinAlgError)

#: Worker count for the per-SKU classical model loops (ARIMA/SARIMA/ETS/
#: Prophet). Kept modest and fixed rather than "all cores" (`-1`): each fold
#: already runs after a memory-heavy LightGBM/ensemble fit in the same
#: process, and this project has already hit real memory pressure once from
#: uncoordinated concurrency - a small, predictable worker count is safer
#: than maximum parallelism on a machine of unknown size.
_CLASSICAL_MODEL_WORKERS: Final[int] = 4

#: RandomForest kept shallow and modestly sized: this is a benchmark for
#: comparison, not a tuning exercise, and the point is a fair standard
#: configuration, not the best possible one.
_RANDOM_FOREST_PARAMS: Final[dict[str, object]] = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "n_jobs": -1,
}

#: XGBoost kept to the same rough capacity as LightGBM's default so the
#: comparison is between algorithms, not between a tuned and an untuned tree
#: ensemble.
_XGBOOST_PARAMS: Final[dict[str, object]] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "objective": "reg:absoluteerror",
    "n_jobs": -1,
}

#: Prophet's Stan optimizer occasionally fails to converge on a short or
#: degenerate series (all-zero history, a single non-zero spike). Caught
#: narrowly so one SKU's fit failure degrades that SKU rather than the fold.
_PROPHET_FIT_FAILURE_ERRORS: Final[tuple[type[Exception], ...]] = (ValueError, RuntimeError)

#: CatBoost kept to the same rough capacity as the other two tree ensembles
#: here - a benchmark for comparison, not a tuning exercise.
_CATBOOST_PARAMS: Final[dict[str, object]] = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "MAE",
    "allow_writing_files": False,
    "verbose": False,
}

#: A full annual cycle at weekly granularity, the seasonal order SARIMA
#: attempts on top of plain ARIMA's (1,1,1) - same period ETS uses, gated by
#: the same `_MIN_SEASONAL_OBSERVATIONS` threshold for the same reason
#: (fitting a 52-period seasonal state with under two cycles of history is
#: fitting noise, not seasonality).
_SARIMA_SEASONAL_ORDER: Final[tuple[int, int, int, int]] = (1, 0, 1, _SEASONAL_PERIOD_WEEKS)
_SARIMA_NO_SEASONAL_ORDER: Final[tuple[int, int, int, int]] = (0, 0, 0, 0)


def _last_value_fallback(history: np.ndarray, horizon: int) -> np.ndarray:
    """What every classical model here falls back to when it cannot fit."""
    level = float(history[-1]) if history.size else 0.0
    return np.full(horizon, max(level, 0.0))


def _fit_forecast_one_arima(history: np.ndarray, horizon: int) -> np.ndarray:
    if history.size < _MIN_CLASSICAL_OBSERVATIONS:
        return _last_value_fallback(history, horizon)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(history, order=(1, 1, 1)).fit()
            forecast = np.asarray(fitted.forecast(steps=horizon), dtype="float64")
    except _FIT_FAILURE_ERRORS as exc:
        log.debug(
            "arima fit failed; falling back to last value", extra={"context": {"error": str(exc)}}
        )
        return _last_value_fallback(history, horizon)
    return np.clip(forecast, 0.0, None)


def _fit_forecast_one_ets(history: np.ndarray, horizon: int) -> np.ndarray:
    if history.size < _MIN_CLASSICAL_OBSERVATIONS:
        return _last_value_fallback(history, horizon)
    use_seasonal = history.size >= _MIN_SEASONAL_OBSERVATIONS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ExponentialSmoothing(
                history,
                trend="add",
                damped_trend=True,
                seasonal="add" if use_seasonal else None,
                seasonal_periods=_SEASONAL_PERIOD_WEEKS if use_seasonal else None,
            ).fit()
            forecast = np.asarray(fitted.forecast(steps=horizon), dtype="float64")
    except _FIT_FAILURE_ERRORS as exc:
        log.debug(
            "ets fit failed; falling back to last value", extra={"context": {"error": str(exc)}}
        )
        return _last_value_fallback(history, horizon)
    return np.clip(forecast, 0.0, None)


def _per_sku_classical_forecast(
    panel: pd.DataFrame,
    test: pd.DataFrame,
    origin: pd.Timestamp,
    fit_one: Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    """Shared driver for ARIMA/SARIMA/ETS: fit one model per SKU, slot in by horizon.

    ``fit_one`` is ``_fit_forecast_one_arima``, ``_fit_forecast_one_sarima`` or
    ``_fit_forecast_one_ets`` - all three take a history array and a horizon
    count and return that many forecast steps. Independent SKU to SKU, so the
    loop runs across `_CLASSICAL_MODEL_WORKERS` worker processes rather than
    one at a time - see the module docstring's "Parallelism" section.
    """
    predictions = np.empty(len(test), dtype="float64")
    history_by_sku = {
        sku_id: group.sort_values("week_start")["units"].to_numpy(dtype="float64")
        for sku_id, group in panel[panel["week_start"] <= origin].groupby("sku_id", observed=True)
    }

    groups = list(test.groupby("sku_id", observed=True))
    forecasts = Parallel(n_jobs=_CLASSICAL_MODEL_WORKERS)(
        delayed(fit_one)(
            history_by_sku.get(sku_id, np.empty(0, dtype="float64")), int(rows["horizon"].max())
        )
        for sku_id, rows in groups
    )

    for (_sku_id, rows), forecast in zip(groups, forecasts, strict=True):
        # horizon is 1-indexed and matches position in the forecast array.
        positions = rows["horizon"].to_numpy(dtype="int64") - 1
        predictions[test.index.get_indexer(rows.index)] = forecast[positions]

    return predictions


def fit_predict_arima(panel: pd.DataFrame, test: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
    """One ARIMA(1,1,1) per SKU, fit on history up to ``origin``.

    Args:
        panel: The weekly panel (one row per SKU-week, raw ``units``).
        test: This fold's test rows (one row per SKU-horizon pair).
        origin: The fold's origin week - history strictly at or before this
            is all the model may see, same rule as every other candidate.
    """
    return _per_sku_classical_forecast(panel, test, origin, _fit_forecast_one_arima)


def fit_predict_ets(panel: pd.DataFrame, test: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
    """One damped-trend Exponential Smoothing model per SKU."""
    return _per_sku_classical_forecast(panel, test, origin, _fit_forecast_one_ets)


def _fit_forecast_one_sarima(history: np.ndarray, horizon: int) -> np.ndarray:
    if history.size < _MIN_CLASSICAL_OBSERVATIONS:
        return _last_value_fallback(history, horizon)
    seasonal_order = (
        _SARIMA_SEASONAL_ORDER
        if history.size >= _MIN_SEASONAL_OBSERVATIONS
        else _SARIMA_NO_SEASONAL_ORDER
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = SARIMAX(
                history,
                order=(1, 1, 1),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
            forecast = np.asarray(fitted.forecast(steps=horizon), dtype="float64")
    except _FIT_FAILURE_ERRORS as exc:
        log.debug(
            "sarima fit failed; falling back to last value",
            extra={"context": {"error": str(exc)}},
        )
        return _last_value_fallback(history, horizon)
    return np.clip(forecast, 0.0, None)


def fit_predict_sarima(panel: pd.DataFrame, test: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
    """ARIMA(1,1,1) per SKU, plus a 52-week seasonal term where there is enough
    history to fit one (:data:`_MIN_SEASONAL_OBSERVATIONS`, the same gate ETS
    uses for its own seasonal component) - otherwise identical to
    :func:`fit_predict_arima`.
    """
    return _per_sku_classical_forecast(panel, test, origin, _fit_forecast_one_sarima)


def _fit_forecast_one_prophet(history: pd.DataFrame, future_dates: pd.DatetimeIndex) -> np.ndarray:
    """Fit one Prophet model on a single SKU's ``ds``/``y`` history."""
    if len(history) < _MIN_CLASSICAL_OBSERVATIONS:
        return _last_value_fallback(history["y"].to_numpy(dtype="float64"), len(future_dates))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(
                yearly_seasonality=len(history) >= _MIN_SEASONAL_OBSERVATIONS,
                weekly_seasonality=False,
                daily_seasonality=False,
            )
            model.fit(history)
            forecast = model.predict(pd.DataFrame({"ds": future_dates}))["yhat"].to_numpy(
                dtype="float64"
            )
    except _PROPHET_FIT_FAILURE_ERRORS as exc:
        log.debug(
            "prophet fit failed; falling back to last value",
            extra={"context": {"error": str(exc)}},
        )
        return _last_value_fallback(history["y"].to_numpy(dtype="float64"), len(future_dates))
    return np.clip(forecast, 0.0, None)


def _prophet_one_sku(history: pd.DataFrame, origin: pd.Timestamp, max_horizon: int) -> np.ndarray:
    """One SKU's worth of :func:`_fit_forecast_one_prophet` - a plain top-level
    function (not a closure) so it stays picklable for the worker pool below.
    """
    future_dates = pd.date_range(start=origin + pd.Timedelta(days=7), periods=max_horizon, freq="7D")
    return _fit_forecast_one_prophet(history, future_dates)


def fit_predict_prophet(panel: pd.DataFrame, test: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
    """One Prophet trend/seasonality model per SKU, fit on history up to ``origin``.

    Unlike ARIMA/ETS, Prophet needs real calendar dates rather than a bare
    array of values - its yearly-seasonality term is a function of actual day
    of year - so this keeps ``week_start`` alongside ``units`` instead of
    reusing :func:`_per_sku_classical_forecast`'s shared driver. Independent
    SKU to SKU like the other three, so it is parallelised the same way (see
    the module docstring's "Parallelism" section).
    """
    predictions = np.empty(len(test), dtype="float64")
    empty_history = pd.DataFrame(
        {"ds": pd.Series(dtype="datetime64[ns]"), "y": pd.Series(dtype="float64")}
    )
    history_by_sku = {
        sku_id: group.sort_values("week_start")
        .loc[:, ["week_start", "units"]]
        .rename(columns={"week_start": "ds", "units": "y"})
        for sku_id, group in panel[panel["week_start"] <= origin].groupby("sku_id", observed=True)
    }

    groups = list(test.groupby("sku_id", observed=True))
    forecasts = Parallel(n_jobs=_CLASSICAL_MODEL_WORKERS)(
        delayed(_prophet_one_sku)(
            history_by_sku.get(sku_id, empty_history), origin, int(rows["horizon"].max())
        )
        for sku_id, rows in groups
    )

    for (_sku_id, rows), forecast in zip(groups, forecasts, strict=True):
        positions = rows["horizon"].to_numpy(dtype="int64") - 1
        predictions[test.index.get_indexer(rows.index)] = forecast[positions]

    return predictions


def _build_preprocessor(feature_spec: FeatureSpec) -> ColumnTransformer:
    """Impute numeric features, one-hot encode categoricals.

    Unlike LightGBM, neither scikit-learn estimator below accepts NaN or a
    pandas categorical dtype directly, so this is the one place those get
    turned into plain numbers - fit on train, applied unchanged to test, so
    an unseen category at serving time is ignored (all-zero row) rather than
    raising.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), list(feature_spec.numeric)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(feature_spec.categorical),
            ),
        ]
    )


def fit_predict_linear_regression(
    train: pd.DataFrame, test: pd.DataFrame, feature_spec: FeatureSpec
) -> np.ndarray:
    """Ordinary least squares on the GBM's own feature matrix."""
    pipeline = Pipeline(
        [("preprocess", _build_preprocessor(feature_spec)), ("model", LinearRegression())]
    )
    pipeline.fit(train.loc[:, feature_spec.all_features], train["y"].to_numpy(dtype="float64"))
    prediction = pipeline.predict(test.loc[:, feature_spec.all_features])
    return np.clip(prediction, 0.0, None)


def fit_predict_random_forest(
    train: pd.DataFrame, test: pd.DataFrame, feature_spec: FeatureSpec, seed: int
) -> np.ndarray:
    """A standard Random Forest on the GBM's own feature matrix."""
    pipeline = Pipeline(
        [
            ("preprocess", _build_preprocessor(feature_spec)),
            ("model", RandomForestRegressor(random_state=seed, **_RANDOM_FOREST_PARAMS)),
        ]
    )
    pipeline.fit(train.loc[:, feature_spec.all_features], train["y"].to_numpy(dtype="float64"))
    prediction = pipeline.predict(test.loc[:, feature_spec.all_features])
    return np.clip(prediction, 0.0, None)


def fit_predict_catboost(
    train: pd.DataFrame, test: pd.DataFrame, feature_spec: FeatureSpec, seed: int
) -> np.ndarray:
    """A standard CatBoost regressor on the GBM's own feature matrix.

    Scored the same way as the other two tree ensembles above (same
    preprocessing, same feature matrix) - a third major boosting library,
    included for breadth rather than because it was expected to behave very
    differently from XGBoost here.
    """
    pipeline = Pipeline(
        [
            ("preprocess", _build_preprocessor(feature_spec)),
            ("model", CatBoostRegressor(random_state=seed, **_CATBOOST_PARAMS)),
        ]
    )
    pipeline.fit(train.loc[:, feature_spec.all_features], train["y"].to_numpy(dtype="float64"))
    prediction = pipeline.predict(test.loc[:, feature_spec.all_features])
    return np.clip(prediction, 0.0, None)


def fit_predict_xgboost(
    train: pd.DataFrame, test: pd.DataFrame, feature_spec: FeatureSpec, seed: int
) -> np.ndarray:
    """A standard XGBoost regressor on the GBM's own feature matrix.

    Scored the same way as Linear Regression and Random Forest above (same
    preprocessing, same feature matrix), not the way LightGBM itself is
    trained - the point of this comparison is a fair fight between
    off-the-shelf estimators, not XGBoost's best possible configuration.
    """
    pipeline = Pipeline(
        [
            ("preprocess", _build_preprocessor(feature_spec)),
            (
                "model",
                xgb.XGBRegressor(random_state=seed, **_XGBOOST_PARAMS),
            ),
        ]
    )
    pipeline.fit(train.loc[:, feature_spec.all_features], train["y"].to_numpy(dtype="float64"))
    prediction = pipeline.predict(test.loc[:, feature_spec.all_features])
    return np.clip(prediction, 0.0, None)
