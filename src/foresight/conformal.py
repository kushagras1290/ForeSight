"""FORESIGHT Conformal Interval Recalibration - the seventh custom model.

THE FAILURE THIS EXISTS FOR
----------------------------
The Adaptive Ensemble's prediction interval is widened by a single
`interval_scale` factor (:mod:`foresight.custom_model`), fitted once and
applied uniformly to every row regardless of regime. But the regime-level
backtest (``artifacts/metrics_by_regime.parquet``, surfaced as
`AccuracySummary.by_regime`) already shows WAPE is not uniform across
regimes - a steady-demand SKU and a volatile one do not fail the same way,
so there is no reason their intervals should need the same correction to hit
the same stated coverage. A single global scale that looks fine on average
can still be quietly overconfident for volatile/intermittent SKUs (the ones
where a wrong interval matters most for a reorder decision) while being
needlessly wide for steady ones.

WHAT THIS DOES
--------------
Standard split-conformal calibration, done per regime instead of once
globally: using backtest folds already produced by
`foresight.backtest.run_backtest` as the calibration set, this computes each
regime's empirical (1 - target_coverage) quantile of the *nonconformity
score* - the ratio of absolute error to the ensemble's already-fitted
half-width - and uses it to rescale that regime's interval on a held-out
fold. This is deliberately **not** a change to the point forecast; it only
recalibrates how wide the stated interval should be, per regime, so that an
80% interval actually covers close to 80% of outcomes in every regime, not
just on average across all of them.

Scored the same way `foresight.backtest` scores the ensemble itself: on the
held-out fold this calibration was never fitted on, comparing coverage before
(the ensemble's own uniform scale) against after (this recalibration),
overall and per regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from foresight.exceptions import InsufficientHistoryError
from foresight.logging_setup import get_logger

__all__ = ["ConformalCalibration", "apply_conformal_calibration", "fit_conformal_calibration"]

log = get_logger(__name__)

_POOLED: Final[str] = "__POOLED__"

#: A regime needs this many calibration rows before its own scale is fitted;
#: below it, the pooled all-regime scale is used instead.
_MIN_REGIME_OBSERVATIONS: Final[int] = 30

#: A half-width below this is too close to zero to divide by safely.
_MIN_HALF_WIDTH: Final[float] = 1e-6

#: Bounds on the fitted scale. A scale far outside this range would mean the
#: base interval was so mis-sized that recalibration is patching over a
#: different problem, not correcting for regime variation.
_SCALE_BOUNDS: Final[tuple[float, float]] = (0.25, 6.0)


@dataclass(slots=True)
class ConformalCalibration:
    """Per-regime interval-width scale, fitted via split-conformal calibration."""

    scale_by_regime: dict[str, float]
    target_coverage: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    def scale_for(self, regimes: np.ndarray) -> np.ndarray:
        return np.array(
            [self.scale_by_regime.get(str(r), self.scale_by_regime[_POOLED]) for r in regimes],
            dtype="float64",
        )


def _conformal_quantile(scores: np.ndarray, target_coverage: float) -> float:
    """The finite-sample split-conformal quantile: ceil((n+1) * coverage) / n, capped at 1."""
    n = scores.size
    if n == 0:
        return 1.0
    level = min(1.0, float(np.ceil((n + 1) * target_coverage) / n))
    return float(np.clip(np.quantile(scores, level), *_SCALE_BOUNDS))


def fit_conformal_calibration(
    calibration: pd.DataFrame, *, target_coverage: float = 0.8
) -> ConformalCalibration:
    """Fit a per-regime interval scale via split-conformal calibration.

    Args:
        calibration: Backtest predictions restricted to calibration folds -
            columns `y`, `ensemble`, `ensemble_lower`, `ensemble_upper`,
            `regime`. Must be disjoint from whatever is later evaluated with
            this calibration, or the reported coverage is not honest.
        target_coverage: The interval's stated probability of covering the
            actual (this project's default is 0.8).

    Raises:
        InsufficientHistoryError: if there are no calibration rows at all.
    """
    if calibration.empty:
        raise InsufficientHistoryError("No calibration rows; cannot fit interval recalibration.")

    half_width = (
        calibration["ensemble_upper"].to_numpy(dtype="float64")
        - calibration["ensemble_lower"].to_numpy(dtype="float64")
    ) / 2.0
    half_width = np.clip(half_width, _MIN_HALF_WIDTH, None)
    nonconformity = (
        np.abs(
            calibration["y"].to_numpy(dtype="float64")
            - calibration["ensemble"].to_numpy(dtype="float64")
        )
        / half_width
    )

    pooled_scale = _conformal_quantile(nonconformity, target_coverage)
    scale_by_regime: dict[str, float] = {_POOLED: pooled_scale}
    scored = calibration.assign(_nonconformity=nonconformity)
    for regime, group in scored.groupby("regime", sort=True):
        if len(group) < _MIN_REGIME_OBSERVATIONS:
            continue
        scale_by_regime[str(regime)] = _conformal_quantile(
            group["_nonconformity"].to_numpy(dtype="float64"), target_coverage
        )

    model = ConformalCalibration(
        scale_by_regime=scale_by_regime,
        target_coverage=target_coverage,
        diagnostics={
            "calibration_rows": float(len(calibration)),
            "regimes_with_own_scale": float(len(scale_by_regime) - 1),
            "pooled_scale": round(pooled_scale, 4),
        },
    )
    log.info("fitted conformal interval calibration", extra={"context": model.diagnostics})
    return model


def apply_conformal_calibration(
    calibration: ConformalCalibration, frame: pd.DataFrame
) -> pd.DataFrame:
    """Recalibrated interval bounds for `frame`.

    Args:
        calibration: A calibration fitted on a *disjoint* set of folds.
        frame: Rows to recalibrate - needs `ensemble`, `ensemble_lower`,
            `ensemble_upper`, `regime`.

    Returns:
        `frame` with `recalibrated_lower`/`recalibrated_upper` columns added.
        The point forecast (`ensemble`) is unchanged.
    """
    half_width = (
        frame["ensemble_upper"].to_numpy(dtype="float64")
        - frame["ensemble_lower"].to_numpy(dtype="float64")
    ) / 2.0
    half_width = np.clip(half_width, _MIN_HALF_WIDTH, None)
    scale = calibration.scale_for(frame["regime"].to_numpy())
    center = frame["ensemble"].to_numpy(dtype="float64")
    recalibrated_half_width = half_width * scale
    return frame.assign(
        recalibrated_lower=np.clip(center - recalibrated_half_width, 0.0, None),
        recalibrated_upper=center + recalibrated_half_width,
    )
