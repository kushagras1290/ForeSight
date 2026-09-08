"""Chronological train / holdout split for independent testing.

WHY NOT A RANDOM SPLIT
----------------------
A random 70/30 split is the wrong tool for a time series and would invalidate
every number it produced. Shuffling rows puts January 2026 in the training set
and December 2025 in the test set, so the model is asked to "predict" a past it
has already been shown. The reported accuracy would be fiction.

The split here is by **time**: the earliest 70% of weeks train the model, the
most recent 30% are held back entirely. That is the only split that answers the
question actually being asked - *given what we knew then, how well would this
have done since?*

HOW THE HOLDOUT IS FORECAST
---------------------------
The holdout period is far longer than the forecast horizon, so it cannot be
covered from a single origin. The origin walks forward in horizon-sized steps,
exactly as it would in production: forecast the next 8 weeks, wait, observe what
happened, roll forward, forecast again.

That means **features** at a later origin do use observed holdout demand - which
is correct, because in production you genuinely do see last week before
forecasting next week. What never happens is the model **training** on any
holdout week. The split is on model fitting, not on observation, and that is the
distinction that makes the result honest.

WHAT YOU GET
------------
``data/holdout/holdout_actuals.csv``   the true demand for the holdout weeks,
                                       formatted for upload to the dashboard
``artifacts/holdout_forecast.parquet`` what the model predicted for them
``artifacts/holdout_metrics.json``     the honest score, against the baseline

Upload the CSV on the dashboard's *Forecast accuracy* tab and the service scores
it against the forecast, independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from foresight.baseline import seasonal_naive_forecast
from foresight.config import Settings, get_settings
from foresight.custom_model import train_ensemble
from foresight.exceptions import InsufficientHistoryError
from foresight.features import (
    build_base_features,
    build_inference_frame,
    build_supervised_frame,
    resolve_feature_spec,
)
from foresight.forecast import train_forecaster
from foresight.logging_setup import get_logger
from foresight.metrics import ForecastMetrics, evaluate_forecast, interval_coverage

__all__ = ["HoldoutResult", "run_holdout"]

log = get_logger(__name__)


@dataclass(slots=True)
class HoldoutResult:
    """Everything the holdout run produced."""

    split_week: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    train_weeks: int
    holdout_weeks: int
    train_rows: int
    n_origins: int
    model_name: str

    forecast: pd.DataFrame
    actuals: pd.DataFrame
    #: Forecast joined to outcome, one row per scored (SKU, target week).
    scored: pd.DataFrame

    model: ForecastMetrics
    baseline: ForecastMetrics
    interval_coverage: float
    baseline_fallback_rate: float

    @property
    def improvement_vs_baseline(self) -> float:
        if not np.isfinite(self.baseline.wape) or self.baseline.wape <= 0:
            return float("nan")
        return (self.baseline.wape - self.model.wape) / self.baseline.wape

    def summary(self) -> dict[str, Any]:
        return {
            "split_week": str(self.split_week.date()),
            "holdout_start": str(self.holdout_start.date()),
            "holdout_end": str(self.holdout_end.date()),
            "train_weeks": self.train_weeks,
            "holdout_weeks": self.holdout_weeks,
            "train_fraction": round(self.train_weeks / (self.train_weeks + self.holdout_weeks), 4),
            "train_rows": self.train_rows,
            "forecast_origins": self.n_origins,
            "model": self.model_name,
            "holdout_observations": self.model.n_observations,
            "model_wape": round(self.model.wape, 6),
            "baseline_wape": round(self.baseline.wape, 6),
            "improvement_vs_baseline": round(self.improvement_vs_baseline, 6),
            "model_bias_relative": round(self.model.bias_relative, 6),
            "baseline_bias_relative": round(self.baseline.bias_relative, 6),
            "model_mape": round(self.model.mape, 6),
            "model_mape_coverage": round(self.model.mape_coverage, 6),
            "interval_coverage": round(self.interval_coverage, 6),
            "baseline_fallback_rate": round(self.baseline_fallback_rate, 6),
        }


def run_holdout(
    panel: pd.DataFrame,
    weekly_calendar: pd.DataFrame,
    settings: Settings | None = None,
    *,
    holdout_fraction: float = 0.3,
    use_ensemble: bool = True,
    use_drivers: bool = False,
    marketing_plan: pd.DataFrame | None = None,
) -> HoldoutResult:
    """Train on the earliest weeks, forecast and score the most recent ones.

    Args:
        panel: The weekly panel.
        weekly_calendar: Weekly calendar dimension.
        settings: Configuration.
        holdout_fraction: Share of the timeline held back. 0.3 => a 70/30 split.
        use_ensemble: Fit the adaptive ensemble as well as the GBM.
        use_drivers: Include the optional commercial driver features. Applied to
            training and to inference together - splitting them would train one
            contract and serve another.
        marketing_plan: Forward media plan, needed for target-week driver
            features when ``use_drivers`` is set.

    Raises:
        InsufficientHistoryError: if either side of the split is too small to be
            meaningful.
    """
    settings = settings or get_settings()
    if not 0.05 <= holdout_fraction <= 0.6:
        raise ValueError(f"holdout_fraction must be between 0.05 and 0.6; got {holdout_fraction}")

    weeks = np.sort(panel["week_start"].unique())
    split_index = int(round(len(weeks) * (1.0 - holdout_fraction)))
    if split_index < settings.min_train_weeks:
        raise InsufficientHistoryError(
            f"A {1 - holdout_fraction:.0%} training split leaves {split_index} weeks, "
            f"below the {settings.min_train_weeks} required. Shorten the holdout."
        )
    if len(weeks) - split_index < settings.horizon_weeks:
        raise InsufficientHistoryError(
            "The holdout is shorter than one forecast horizon; nothing could be scored."
        )

    split_week = pd.Timestamp(weeks[split_index - 1])
    holdout_weeks = [pd.Timestamp(week) for week in weeks[split_index:]]

    log.info(
        "holdout split",
        extra={
            "context": {
                "split_week": str(split_week.date()),
                "train_weeks": split_index,
                "holdout_weeks": len(holdout_weeks),
                "fraction": holdout_fraction,
            }
        },
    )

    # --- Features over the whole panel; the split is applied to training ----- #
    feature_spec = resolve_feature_spec(use_drivers=use_drivers)
    base = build_base_features(panel, use_drivers=use_drivers)
    horizons = tuple(range(1, settings.horizon_weeks + 1))
    supervised = build_supervised_frame(
        base,
        weekly_calendar,
        horizons,
        use_drivers=use_drivers,
        marketing_plan=marketing_plan,
    )

    train = supervised[supervised["target_week"] <= split_week]
    if train.empty:
        raise InsufficientHistoryError("No training rows fall before the split week.")

    forecaster = train_forecaster(train, settings, feature_spec=feature_spec)
    model: Any = forecaster
    model_name = "lightgbm"
    if use_ensemble:
        model = train_ensemble(
            train,
            panel[panel["week_start"] <= split_week],
            settings,
            gbm=forecaster,
            feature_spec=feature_spec,
        )
        model_name = "adaptive_ensemble"

    # --- Walk the origin forward through the holdout ------------------------ #
    # Step by the horizon so every holdout week is covered exactly once, by the
    # freshest origin that can reach it.
    origins: list[pd.Timestamp] = []
    cursor = split_index - 1
    while cursor < len(weeks) - 1:
        origins.append(pd.Timestamp(weeks[cursor]))
        cursor += settings.horizon_weeks

    parts: list[pd.DataFrame] = []
    for origin in origins:
        inference = build_inference_frame(
            base,
            weekly_calendar,
            origin,
            horizons,
            use_drivers=use_drivers,
            marketing_plan=marketing_plan,
        )
        predictions = model.predict(inference)

        block = inference[
            ["sku_id", "week_start", "target_week", "horizon", "category", "subcategory"]
        ].copy()
        block["origin_week"] = origin
        block["prediction"] = predictions["prediction"].to_numpy()
        block["prediction_lower"] = predictions["prediction_lower"].to_numpy()
        block["prediction_upper"] = predictions["prediction_upper"].to_numpy()
        block["units_seasonal_lag"] = inference["units_seasonal_lag"].to_numpy()
        block["units_roll_mean_4"] = inference["units_roll_mean_4"].to_numpy()
        parts.append(block)

    forecast = pd.concat(parts, ignore_index=True)
    # Keep only holdout weeks, and only one forecast per week - the earliest
    # origin that reaches it, which is the freshest information available.
    forecast = forecast[forecast["target_week"] > split_week]
    forecast = forecast.sort_values(["sku_id", "target_week", "origin_week"])
    forecast = forecast.drop_duplicates(subset=["sku_id", "target_week"], keep="first")

    # --- Actuals, and the score --------------------------------------------- #
    actuals = panel.loc[panel["week_start"] > split_week, ["sku_id", "week_start", "units"]].rename(
        columns={"week_start": "target_week"}
    )

    scored = forecast.merge(actuals, on=["sku_id", "target_week"], how="inner")
    if scored.empty:
        raise InsufficientHistoryError("No holdout week could be matched to a forecast.")

    baseline = seasonal_naive_forecast(scored)
    actual_values = scored["units"].to_numpy(dtype="float64")

    # Prediction and outcome side by side, which is what any review of the
    # holdout actually needs. Assembled once here rather than re-joined by every
    # consumer, so the dashboard, the reports and the metrics below can never
    # disagree about which rows were scored.
    scored_out = scored.drop(columns=["units_seasonal_lag", "units_roll_mean_4"]).rename(
        columns={"units": "actual"}
    )
    scored_out["baseline"] = baseline.predictions
    scored_out["error"] = scored_out["prediction"] - scored_out["actual"]
    scored_out["abs_error"] = scored_out["error"].abs()
    scored_out["within_interval"] = (
        (scored_out["actual"] >= scored_out["prediction_lower"])
        & (scored_out["actual"] <= scored_out["prediction_upper"])
    ).astype("int64")

    result = HoldoutResult(
        split_week=split_week,
        holdout_start=holdout_weeks[0],
        holdout_end=holdout_weeks[-1],
        train_weeks=split_index,
        holdout_weeks=len(holdout_weeks),
        train_rows=len(train),
        n_origins=len(origins),
        model_name=model_name,
        forecast=forecast.drop(columns=["units_seasonal_lag", "units_roll_mean_4"]),
        actuals=actuals.rename(columns={"target_week": "week_starting"}),
        scored=scored_out,
        model=evaluate_forecast(actual_values, scored["prediction"]),
        baseline=evaluate_forecast(actual_values, baseline.predictions),
        interval_coverage=interval_coverage(
            actual_values, scored["prediction_lower"], scored["prediction_upper"]
        ),
        baseline_fallback_rate=baseline.fallback_rate,
    )

    log.info(
        "holdout scored",
        extra={
            "context": {
                "observations": result.model.n_observations,
                "model_wape": round(result.model.wape, 4),
                "baseline_wape": round(result.baseline.wape, 4),
                "improvement": round(result.improvement_vs_baseline, 4),
            }
        },
    )
    return result
