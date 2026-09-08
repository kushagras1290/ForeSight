"""Forecast accuracy metrics (brief Appendix B).

WAPE is the primary metric for this engagement. MAPE is reported alongside it
because stakeholders recognise it, but it is *not* used to select a model: on
low-volume SKUs a single unit of error on a week that sold one unit is a 100%
error, so MAPE is dominated by the smallest SKUs and effectively undefined on
weeks that sold nothing. WAPE weights by volume and stays finite, which is why
it is the number the model is judged on.

Bias is the honesty check. A forecast can post an excellent WAPE while running
systematically 15% light, which quietly guarantees stockouts. Accuracy and bias
have to be read together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "ForecastMetrics",
    "bias",
    "evaluate_forecast",
    "interval_coverage",
    "mape",
    "pinball_loss",
    "wape",
]

_EPSILON = 1e-12


def _to_arrays(actual: object, predicted: object) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to aligned float arrays, dropping non-finite pairs."""
    actual_array = np.asarray(actual, dtype="float64").ravel()
    predicted_array = np.asarray(predicted, dtype="float64").ravel()
    if actual_array.shape != predicted_array.shape:
        raise ValueError(
            f"actual and predicted must have the same shape; got "
            f"{actual_array.shape} and {predicted_array.shape}"
        )
    finite = np.isfinite(actual_array) & np.isfinite(predicted_array)
    return actual_array[finite], predicted_array[finite]


def wape(actual: object, predicted: object) -> float:
    """Weighted Absolute Percentage Error: ``sum|a - p| / sum(a)``.

    Returns NaN when total actual demand is zero, since the ratio is undefined -
    reporting 0.0 there would falsely read as a perfect forecast.
    """
    actual_array, predicted_array = _to_arrays(actual, predicted)
    if actual_array.size == 0:
        return float("nan")
    denominator = np.abs(actual_array).sum()
    if denominator <= _EPSILON:
        return float("nan")
    return float(np.abs(actual_array - predicted_array).sum() / denominator)


def mape(actual: object, predicted: object) -> tuple[float, float]:
    """Mean Absolute Percentage Error over non-zero actuals only.

    Returns:
        The MAPE, and the fraction of observations it could be computed on.
        The coverage figure matters: a MAPE computed on 40% of the data is not
        comparable to one computed on 95%, and quoting it without that context
        is misleading.
    """
    actual_array, predicted_array = _to_arrays(actual, predicted)
    if actual_array.size == 0:
        return float("nan"), 0.0
    usable = np.abs(actual_array) > _EPSILON
    coverage = float(usable.mean())
    if not usable.any():
        return float("nan"), coverage
    errors = np.abs((actual_array[usable] - predicted_array[usable]) / actual_array[usable])
    return float(errors.mean()), coverage


def bias(actual: object, predicted: object) -> tuple[float, float]:
    """Signed forecast error.

    Returns:
        Mean error in units (positive = over-forecasting), and the same as a
        proportion of total actual demand.
    """
    actual_array, predicted_array = _to_arrays(actual, predicted)
    if actual_array.size == 0:
        return float("nan"), float("nan")
    error = predicted_array - actual_array
    total = np.abs(actual_array).sum()
    relative = float(error.sum() / total) if total > _EPSILON else float("nan")
    return float(error.mean()), relative


def pinball_loss(actual: object, predicted: object, quantile: float) -> float:
    """Pinball (quantile) loss - how well a quantile forecast is calibrated."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1); got {quantile}")
    actual_array, predicted_array = _to_arrays(actual, predicted)
    if actual_array.size == 0:
        return float("nan")
    difference = actual_array - predicted_array
    loss = np.maximum(quantile * difference, (quantile - 1.0) * difference)
    return float(loss.mean())


def interval_coverage(actual: object, lower: object, upper: object) -> float:
    """Fraction of actuals falling inside the interval.

    An 80% interval should contain roughly 80% of outcomes. Materially below
    means the model is overconfident; materially above means it is uselessly
    wide.
    """
    actual_array = np.asarray(actual, dtype="float64").ravel()
    lower_array = np.asarray(lower, dtype="float64").ravel()
    upper_array = np.asarray(upper, dtype="float64").ravel()
    finite = np.isfinite(actual_array) & np.isfinite(lower_array) & np.isfinite(upper_array)
    if not finite.any():
        return float("nan")
    inside = (actual_array[finite] >= lower_array[finite]) & (
        actual_array[finite] <= upper_array[finite]
    )
    return float(inside.mean())


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    """A complete accuracy readout for one set of forecasts."""

    n_observations: int
    total_actual: float
    total_predicted: float
    wape: float
    mape: float
    mape_coverage: float
    bias_units: float
    bias_relative: float
    mae: float
    rmse: float
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {
            "n_observations": self.n_observations,
            "total_actual": round(self.total_actual, 2),
            "total_predicted": round(self.total_predicted, 2),
            "wape": round(self.wape, 6),
            "mape": round(self.mape, 6),
            "mape_coverage": round(self.mape_coverage, 6),
            "bias_units": round(self.bias_units, 6),
            "bias_relative": round(self.bias_relative, 6),
            "mae": round(self.mae, 6),
            "rmse": round(self.rmse, 6),
        }
        payload.update({key: round(value, 6) for key, value in self.extras.items()})
        return payload


def evaluate_forecast(
    actual: object,
    predicted: object,
    extras: dict[str, float] | None = None,
) -> ForecastMetrics:
    """Compute the full metric set for one forecast/actual pair."""
    actual_array, predicted_array = _to_arrays(actual, predicted)
    mape_value, mape_cov = mape(actual_array, predicted_array)
    bias_units, bias_relative = bias(actual_array, predicted_array)
    error = predicted_array - actual_array

    return ForecastMetrics(
        n_observations=int(actual_array.size),
        total_actual=float(actual_array.sum()),
        total_predicted=float(predicted_array.sum()),
        wape=wape(actual_array, predicted_array),
        mape=mape_value,
        mape_coverage=mape_cov,
        bias_units=bias_units,
        bias_relative=bias_relative,
        mae=float(np.abs(error).mean()) if error.size else float("nan"),
        rmse=float(np.sqrt((error**2).mean())) if error.size else float("nan"),
        extras=extras or {},
    )


def metrics_by_group(
    frame: pd.DataFrame,
    group_column: str,
    actual_column: str = "y",
    predicted_column: str = "prediction",
) -> pd.DataFrame:
    """Compute WAPE, bias and volume for each level of ``group_column``."""
    records: list[dict[str, object]] = []
    for key, group in frame.groupby(group_column, observed=True):
        bias_units, bias_relative = bias(group[actual_column], group[predicted_column])
        records.append(
            {
                group_column: key,
                "n_observations": len(group),
                "total_actual": float(group[actual_column].sum()),
                "wape": wape(group[actual_column], group[predicted_column]),
                "bias_relative": bias_relative,
                "bias_units": bias_units,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(group_column).reset_index(drop=True)
