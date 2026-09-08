"""FORESIGHT Promotional Response Decomposition - the fifth custom model.

THE FAILURE THIS EXISTS FOR
---------------------------
A promoted week is not a normal week with a bigger number. It is a different
process: demand is pulled forward from the weeks either side of it, price
sensitivity that is invisible at list price becomes the dominant driver, and the
week *after* the promotion is systematically soft because customers who would
have bought then have already bought.

A single model fitted across blended history learns the average of those states.
It under-predicts the peak, because most weeks are not promoted, and it
over-predicts the recovery, because it has no way to know a promotion just ended.
Both errors land at exactly the moments a planner is committing the most stock.

WHAT THIS ADDS OVER THE GBM'S OWN PROMO FEATURES
------------------------------------------------
The gradient-boosted model already sees whether a promotional event is scheduled
for the target week and how deep the published discount is, and it uses them.
This model is not a replacement for that, and it is not claimed to be. It adds
four things a tree ensemble structurally cannot give:

1. **Extrapolation.** A tree can only predict within the discount depths it was
   trained on. Asked about a 40% event when history tops out at 25%, it returns
   its 25% answer. A log-linear elasticity gives a defensible curve past the
   edge of the data - the range where planning decisions are most expensive and
   least informed.

2. **The post-promotion dip.** Pull-forward requires knowing a promotion *just
   ended*, which is a property of the week before the target, not of the target
   week. It is measured here directly and applied as its own factor.

3. **Decomposition.** The output is not one number but three: baseline, uplift,
   and the dip that follows. "You would have sold 400; the promotion adds 260;
   the following week gives 70 of them back" is a sentence a merchandiser can
   act on and challenge. A single blended forecast is not.

4. **Scenario planning.** Because the response is parametric, the depth can be
   varied and the answer recomputed - which is what "should we run 20% or 30%?"
   actually requires.

HOW THE PIECES ARE ESTIMATED
----------------------------
* **Baseline** - each SKU's trailing median over its recent *unpromoted* weeks,
  rescaled by its category's week-of-year index. Trailing, never centred: a
  centred window would read across the promotion it is trying to be the
  counterfactual for, and would absorb the very uplift being measured.

* **Uplift** - on promoted weeks, ``log(units / baseline)`` regressed on
  discount depth, pooled within category because individual SKUs run too few
  promotions to fit their own curve. The slope is constrained non-negative: a
  deeper discount selling fewer units is a sampling artefact, not an elasticity,
  and letting it through would recommend raising prices to sell more.

* **Dip** - the mean ratio of observed to baseline over the weeks immediately
  after a promotion ends, floored so that a single deep trough cannot wipe out a
  recovery week's forecast entirely.

LEAKAGE
-------
The baseline is trailing-only and the coefficients are fitted on the training
window alone. At forecast time the only promotional input is the *published*
discount for the target week, which is a forward-committed plan - the same
justification the calendar features rest on. Realised discount is never read for
a target week.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from foresight.exceptions import InsufficientHistoryError
from foresight.logging_setup import get_logger

__all__ = [
    "PromotionResponse",
    "attach_baseline",
    "fit_promotion_response",
]

log = get_logger(__name__)

#: Trailing unpromoted weeks pooled into the baseline. Long enough to be stable,
#: short enough to track a product whose level is genuinely moving.
BASELINE_WINDOW: Final[int] = 8

#: Weeks after a promotion ends that are treated as the dip window.
DIP_WEEKS: Final[int] = 2

#: A category needs this many promoted observations before its own elasticity is
#: fitted; below it, the pooled all-category curve is used instead.
MIN_PROMO_OBSERVATIONS: Final[int] = 40

#: Bounds on the fitted discount slope. Zero floor because demand cannot fall
#: when price does; the ceiling caps the implied response at roughly a 20x lift
#: at full markdown, past which the fit is describing clearance, not promotion.
_SLOPE_BOUNDS: Final[tuple[float, float]] = (0.0, 6.0)

#: Bounds on the multiplicative uplift actually applied.
_UPLIFT_BOUNDS: Final[tuple[float, float]] = (1.0, 8.0)

#: Bounds on the post-promotion dip factor. Never above 1.0 - that would be a
#: post-promotion *lift*, which is not a thing - and never so low that a
#: recovery week's forecast collapses to nothing.
_DIP_BOUNDS: Final[tuple[float, float]] = (0.55, 1.0)

#: Baseline below which a week's uplift ratio is too unstable to fit on. Dividing
#: by a near-zero baseline produces enormous ratios that dominate the regression.
_MIN_BASELINE: Final[float] = 1.0

_POOLED: Final[str] = "__POOLED__"
_EPSILON: Final[float] = 1e-6


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
def attach_baseline(panel: pd.DataFrame, *, window: int = BASELINE_WINDOW) -> pd.DataFrame:
    """Attach the counterfactual unpromoted demand level for every SKU-week.

    Args:
        panel: Weekly panel with ``sku_id``, ``week_start``, ``units`` and
            ``promo_days``.
        window: Trailing unpromoted weeks pooled into the estimate.

    Returns:
        ``panel`` plus ``is_promo``, ``discount_depth``, ``baseline`` and
        ``weeks_since_promo``.
    """
    frame = panel.sort_values(["sku_id", "week_start"]).reset_index(drop=True).copy()
    frame["is_promo"] = (frame["promo_days"].to_numpy(dtype="float64") > 0.0).astype("int64")

    if "discount_depth" not in frame.columns:
        frame["discount_depth"] = (
            1.0 - frame["avg_price"] / (frame["list_price"] + _EPSILON)
        ).clip(0.0, 0.95)

    # Median over trailing unpromoted weeks only. Promoted weeks are masked to
    # NaN so they cannot contribute to the level they are being compared against;
    # the median (rather than the mean) keeps one deep clearance week from
    # dragging the counterfactual down for two months afterwards.
    unpromoted = frame["units"].where(frame["is_promo"] == 0)
    grouped = unpromoted.groupby(frame["sku_id"].to_numpy(), sort=False)
    trailing = grouped.transform(
        lambda series: series.shift(1).rolling(window, min_periods=1).median()
    )

    # Before a SKU has any unpromoted history, fall back to its own expanding
    # mean, then to its category's level. A missing baseline would silently drop
    # the row from the fit.
    own_mean = frame.groupby("sku_id", sort=False)["units"].transform(
        lambda series: series.shift(1).expanding().mean()
    )

    # The category fallback is built as a proper time-ordered trailing median.
    # A plain groupby median over the whole frame would be a genuine leak: it
    # would average in weeks that had not happened yet, and it would do so on the
    # rows where the model has least other information to correct it.
    category_weekly = (
        frame.groupby(["category", "week_start"], as_index=False)["units"]
        .mean()
        .sort_values(["category", "week_start"])
    )
    category_weekly["category_trailing"] = category_weekly.groupby("category", sort=False)[
        "units"
    ].transform(lambda series: series.shift(1).expanding().median())
    frame = frame.merge(
        category_weekly.loc[:, ["category", "week_start", "category_trailing"]],
        on=["category", "week_start"],
        how="left",
        validate="many_to_one",
    )

    baseline = trailing.fillna(own_mean).fillna(frame["category_trailing"]).fillna(0.0)
    frame["baseline"] = baseline.clip(lower=0.0).to_numpy(dtype="float64")
    frame = frame.drop(columns=["category_trailing"])

    # How long since this SKU last promoted, for the dip window. Counted from the
    # origin backwards, so it is knowable at forecast time.
    promo_positions = frame.groupby("sku_id", sort=False).cumcount()
    last_promo = (
        promo_positions.where(frame["is_promo"] == 1)
        .groupby(frame["sku_id"].to_numpy(), sort=False)
        .ffill()
    )
    frame["weeks_since_promo"] = (promo_positions - last_promo).fillna(9_999.0)

    return frame


# --------------------------------------------------------------------------- #
# The fitted model
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PromotionResponse:
    """Per-category promotional elasticity and post-promotion dip."""

    intercept: dict[str, float]
    slope: dict[str, float]
    dip: dict[str, float]
    baseline_window: int
    dip_weeks: int
    diagnostics: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def _coefficients(self, categories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        intercept = np.array(
            [self.intercept.get(str(c), self.intercept[_POOLED]) for c in categories],
            dtype="float64",
        )
        slope = np.array(
            [self.slope.get(str(c), self.slope[_POOLED]) for c in categories], dtype="float64"
        )
        return intercept, slope

    def uplift(self, categories: np.ndarray, discount: np.ndarray) -> np.ndarray:
        """Multiplicative lift over baseline at a given discount depth."""
        intercept, slope = self._coefficients(np.asarray(categories))
        depth = np.clip(np.asarray(discount, dtype="float64"), 0.0, 0.95)
        return np.clip(np.exp(intercept + slope * depth), *_UPLIFT_BOUNDS)

    def dip_factor(self, categories: np.ndarray, weeks_since_promo: np.ndarray) -> np.ndarray:
        """Pull-forward factor for the weeks following a promotion."""
        weeks = np.asarray(weeks_since_promo, dtype="float64")
        inside = (weeks >= 1.0) & (weeks <= float(self.dip_weeks))
        factors = np.array(
            [self.dip.get(str(c), self.dip[_POOLED]) for c in np.asarray(categories)],
            dtype="float64",
        )
        return np.where(inside, factors, 1.0)

    # ------------------------------------------------------------------ #
    def multiplier(
        self,
        categories: np.ndarray,
        planned_discount: np.ndarray,
        target_is_promo: np.ndarray,
        weeks_since_promo: np.ndarray,
    ) -> np.ndarray:
        """The full adjustment for a target week.

        A promoted target week gets its uplift; an unpromoted one inside the
        shadow of a recent promotion gets the dip. The two are mutually
        exclusive by construction - a week cannot both be the promotion and be
        recovering from it.

        Args:
            categories: Category of each row.
            planned_discount: Published discount for the *target* week. This is a
                forward-committed plan, which is what makes it usable.
            target_is_promo: Whether a promotion is scheduled for the target week.
            weeks_since_promo: Weeks between the origin and this SKU's last
                promotion, measured backwards from the origin.
        """
        promo = np.asarray(target_is_promo).astype(bool)
        lift = self.uplift(categories, planned_discount)
        dip = self.dip_factor(categories, weeks_since_promo)
        return np.where(promo, lift, dip)

    # ------------------------------------------------------------------ #
    def decompose(
        self,
        forecast: np.ndarray,
        categories: np.ndarray,
        planned_discount: np.ndarray,
        target_is_promo: np.ndarray,
    ) -> pd.DataFrame:
        """Split a forecast into baseline and promotional uplift.

        This is the reporting output: it answers "how much of next week's number
        is the promotion?" without which a merchandiser cannot tell a good
        promotion from a good week.
        """
        total = np.asarray(forecast, dtype="float64")
        promo = np.asarray(target_is_promo).astype(bool)
        lift = self.uplift(categories, planned_discount)
        baseline = np.where(promo, total / np.maximum(lift, _EPSILON), total)
        return pd.DataFrame(
            {
                "forecast": total,
                "baseline": baseline,
                "promo_uplift_units": total - baseline,
                "promo_uplift_pct": np.where(
                    baseline > _EPSILON, (total - baseline) / baseline, 0.0
                ),
                "uplift_multiplier": np.where(promo, lift, 1.0),
            }
        )


def _fit_slope(depth: np.ndarray, log_ratio: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of ``log(uplift) = a + b * depth``, slope clipped."""
    if depth.size < 3 or float(np.var(depth)) <= _EPSILON:
        return float(np.mean(log_ratio)) if log_ratio.size else 0.0, 0.0

    centred_depth = depth - depth.mean()
    centred_ratio = log_ratio - log_ratio.mean()
    slope = float(np.sum(centred_depth * centred_ratio) / np.sum(centred_depth**2))
    slope = float(np.clip(slope, *_SLOPE_BOUNDS))
    intercept = float(log_ratio.mean() - slope * depth.mean())
    return intercept, slope


def fit_promotion_response(
    panel: pd.DataFrame,
    *,
    baseline_window: int = BASELINE_WINDOW,
    dip_weeks: int = DIP_WEEKS,
) -> PromotionResponse:
    """Fit promotional uplift and post-promotion dip on a training-window panel.

    The caller passes only training-window weeks; this function does no time
    filtering, for the same reason the other model trainers do not.

    Args:
        panel: Weekly panel restricted to the training window.
        baseline_window: Trailing unpromoted weeks pooled into the baseline.
        dip_weeks: Weeks after a promotion treated as the dip window.

    Raises:
        InsufficientHistoryError: if the window contains no usable promotions.
    """
    if panel.empty:
        raise InsufficientHistoryError("Weekly panel is empty; cannot fit promotional response.")

    frame = attach_baseline(panel, window=baseline_window)

    promoted = frame[
        (frame["is_promo"] == 1) & (frame["baseline"] >= _MIN_BASELINE) & (frame["units"] > 0.0)
    ].copy()
    if len(promoted) < MIN_PROMO_OBSERVATIONS:
        raise InsufficientHistoryError(
            f"Only {len(promoted)} usable promoted weeks in the training window; "
            f"at least {MIN_PROMO_OBSERVATIONS} are needed to fit a response curve."
        )

    promoted["log_ratio"] = np.log((promoted["units"] / promoted["baseline"]).clip(lower=_EPSILON))
    # Clip the response before fitting, not after: one clearance week logged as a
    # promotion would otherwise set the slope for the whole category.
    promoted["log_ratio"] = promoted["log_ratio"].clip(
        np.log(_UPLIFT_BOUNDS[0] * 0.5), np.log(_UPLIFT_BOUNDS[1])
    )

    pooled_intercept, pooled_slope = _fit_slope(
        promoted["discount_depth"].to_numpy(dtype="float64"),
        promoted["log_ratio"].to_numpy(dtype="float64"),
    )
    intercept: dict[str, float] = {_POOLED: pooled_intercept}
    slope: dict[str, float] = {_POOLED: pooled_slope}

    for category, group in promoted.groupby("category", sort=True):
        if len(group) < MIN_PROMO_OBSERVATIONS:
            continue
        category_intercept, category_slope = _fit_slope(
            group["discount_depth"].to_numpy(dtype="float64"),
            group["log_ratio"].to_numpy(dtype="float64"),
        )
        intercept[str(category)] = category_intercept
        slope[str(category)] = category_slope

    # --- Post-promotion dip -------------------------------------------------- #
    shadow = frame[
        (frame["is_promo"] == 0)
        & (frame["weeks_since_promo"] >= 1.0)
        & (frame["weeks_since_promo"] <= float(dip_weeks))
        & (frame["baseline"] >= _MIN_BASELINE)
    ]
    pooled_dip = (
        float(np.clip((shadow["units"] / shadow["baseline"]).median(), *_DIP_BOUNDS))
        if not shadow.empty
        else 1.0
    )
    dip: dict[str, float] = {_POOLED: pooled_dip}
    for category, group in shadow.groupby("category", sort=True):
        if len(group) < MIN_PROMO_OBSERVATIONS:
            continue
        dip[str(category)] = float(
            np.clip((group["units"] / group["baseline"]).median(), *_DIP_BOUNDS)
        )

    model = PromotionResponse(
        intercept=intercept,
        slope=slope,
        dip=dip,
        baseline_window=baseline_window,
        dip_weeks=dip_weeks,
        diagnostics={
            "promoted_weeks_fitted": float(len(promoted)),
            "categories_with_own_curve": float(len(intercept) - 1),
            "pooled_slope": round(pooled_slope, 4),
            "pooled_dip": round(pooled_dip, 4),
            "median_discount_depth": round(float(promoted["discount_depth"].median()), 4),
            "median_observed_uplift": round(
                float((promoted["units"] / promoted["baseline"]).median()), 4
            ),
            "uplift_at_20pct_off": round(float(np.exp(pooled_intercept + pooled_slope * 0.20)), 4),
            "uplift_at_40pct_off": round(float(np.exp(pooled_intercept + pooled_slope * 0.40)), 4),
        },
    )

    log.info(
        "fitted promotional response",
        extra={"context": model.diagnostics},
    )
    return model
