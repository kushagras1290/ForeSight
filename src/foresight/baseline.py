"""Seasonal-naive baseline (deliverable D3, acceptance criterion 2).

The bar every learned model must clear. Demand in the target week is predicted
as demand in the same week one year earlier::

    forecast(t + h) = units(t + h - 52)

Because the horizon is far shorter than a year, ``t + h - 52`` always falls
strictly before the origin ``t``, so this is computable at forecast time without
touching the future. :mod:`foresight.features` already materialises that value as
``units_seasonal_lag``, so the baseline is that column read directly - the same
number the model sees as an input, which makes the comparison exact.

FALLBACK
--------
A SKU with fewer than 52 weeks of history has no same-week-last-year value. Those
are not skipped, because skipping them would quietly measure the baseline on an
easier subset than the model. Instead they fall back to the trailing four-week
mean, and the fallback rate is reported so the comparison stays honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from foresight.logging_setup import get_logger

__all__ = ["BaselineResult", "seasonal_naive_forecast", "naive_last_value_forecast"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Baseline predictions plus how often the fallback was needed."""

    predictions: np.ndarray
    fallback_count: int
    total_count: int

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.total_count if self.total_count else 0.0


def seasonal_naive_forecast(frame: pd.DataFrame) -> BaselineResult:
    """Seasonal-naive prediction for every row of a supervised frame.

    Args:
        frame: Supervised frame containing ``units_seasonal_lag`` and
            ``units_roll_mean_4``.

    Returns:
        Predictions, clipped at zero because negative demand is not a thing.
    """
    required = {"units_seasonal_lag", "units_roll_mean_4"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Supervised frame is missing {sorted(missing)} for the baseline.")

    seasonal = frame["units_seasonal_lag"].to_numpy(dtype="float64", na_value=np.nan)
    fallback = frame["units_roll_mean_4"].to_numpy(dtype="float64", na_value=np.nan)

    needs_fallback = ~np.isfinite(seasonal)
    predictions = np.where(needs_fallback, fallback, seasonal)
    # A brand-new SKU may have neither; zero is the only defensible guess.
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    predictions = np.clip(predictions, 0.0, None)

    result = BaselineResult(
        predictions=predictions,
        fallback_count=int(needs_fallback.sum()),
        total_count=int(predictions.size),
    )
    log.info(
        "seasonal-naive baseline computed",
        extra={
            "context": {
                "rows": result.total_count,
                "fallback_rows": result.fallback_count,
                "fallback_rate": round(result.fallback_rate, 4),
            }
        },
    )
    return result


def naive_last_value_forecast(frame: pd.DataFrame) -> np.ndarray:
    """Secondary reference: carry the origin week's demand forward unchanged.

    Reported for context only. On a strongly seasonal assortment it is usually
    worse than seasonal-naive, and showing both makes clear that the chosen
    baseline is a genuine bar rather than a straw man.
    """
    if "units_lag_0" not in frame.columns:
        raise KeyError("Supervised frame is missing 'units_lag_0' for the naive baseline.")
    values = frame["units_lag_0"].to_numpy(dtype="float64", na_value=np.nan)
    return np.clip(np.nan_to_num(values, nan=0.0), 0.0, None)
