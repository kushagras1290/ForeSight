"""FORESIGHT Cross-SKU Substitution - the sixth custom model.

THE FAILURE THIS EXISTS FOR
----------------------------
Every forecast in this project, including the other five custom models,
treats each SKU as an island: its own history, its own regime, its own
promotional calendar. But NorthBay's catalogue is built from close
substitutes within a subcategory - when one product runs a promotion, some of
the demand that would have gone to its subcategory siblings is pulled away
rather than created from nothing.

Left unmeasured, this shows up in two places a planner would otherwise
misread as noise or model error:

1. A promoted SKU's *net* portfolio lift is over-stated - some of what it
   sells is cannibalised from siblings, not new demand.
2. An unpromoted sibling's demand dips in exactly the same week, for a reason
   that has nothing to do with that sibling's own forecast quality.

WHAT THIS MEASURES
------------------
For every subcategory with at least two SKUs, this compares each *non*-
promoted SKU's actual demand against its own trailing, unpromoted baseline
(the same baseline machinery :mod:`foresight.promotions` already builds) on
weeks where **at least one of its subcategory siblings was running a
promotion**. The ratio of observed-to-baseline on those weeks is the
substitution dip - the fraction of a sibling's usual demand a promotion
elsewhere in the subcategory pulls away.

Like Censored Demand Recovery and Promotional Response Decomposition
(``reports/model_suite.md`` #7), this is a measured business effect, not a
forecasting model with its own WAPE - its output is a decomposition insight
a merchandiser can act on, not a point forecast to be scored against held-out
actuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from foresight.exceptions import InsufficientHistoryError
from foresight.logging_setup import get_logger
from foresight.promotions import attach_baseline

__all__ = ["SubstitutionEffect", "fit_substitution_effect"]

log = get_logger(__name__)

_POOLED: Final[str] = "__POOLED__"

#: A subcategory needs this many candidate observations (non-promoted weeks
#: with an actively-promoted sibling) before its own dip factor is fitted;
#: below it, the pooled all-subcategory figure is used instead. Lower than
#: `foresight.promotions.MIN_PROMO_OBSERVATIONS` because a subcategory's
#: sibling-promotion weeks are rarer than a category's own promotion weeks.
MIN_SUBSTITUTION_OBSERVATIONS: Final[int] = 30

#: A baseline below this is too unstable to divide by - same threshold and
#: reasoning as `foresight.promotions._MIN_BASELINE`.
_MIN_BASELINE: Final[float] = 1.0

#: Bounds on the fitted dip factor. Never above ~1.05 (allowing a little
#: noise around "no effect") and never so low that a single confounded
#: subcategory reports an implausible near-total demand transfer.
_DIP_BOUNDS: Final[tuple[float, float]] = (0.5, 1.05)


@dataclass(slots=True)
class SubstitutionEffect:
    """Per-subcategory cross-SKU substitution dip, fitted on a training window."""

    dip_by_subcategory: dict[str, float]
    diagnostics: dict[str, float] = field(default_factory=dict)

    def dip_factor(self, subcategories: np.ndarray) -> np.ndarray:
        """Fraction of baseline demand retained when a sibling is promoted."""
        return np.array(
            [self.dip_by_subcategory.get(str(s), self.dip_by_subcategory[_POOLED]) for s in subcategories],
            dtype="float64",
        )


def fit_substitution_effect(panel: pd.DataFrame) -> SubstitutionEffect:
    """Fit the cross-SKU substitution dip on a training-window weekly panel.

    Args:
        panel: Weekly panel restricted to the training window - `sku_id`,
            `week_start`, `units`, `promo_days`, `category`, `subcategory`,
            `avg_price`, `list_price`.

    Raises:
        InsufficientHistoryError: if the window contains no usable
            sibling-promotion weeks at all.
    """
    if panel.empty:
        raise InsufficientHistoryError("Weekly panel is empty; cannot fit substitution effect.")

    frame = attach_baseline(panel)

    # At least one *other* SKU in the same subcategory-week was promoted.
    subcat_promo_count = frame.groupby(["subcategory", "week_start"])["is_promo"].transform("sum")
    sibling_promo_active = (subcat_promo_count - frame["is_promo"]) > 0

    candidate = frame[
        (frame["is_promo"] == 0) & sibling_promo_active & (frame["baseline"] >= _MIN_BASELINE)
    ].copy()
    if candidate.empty:
        raise InsufficientHistoryError(
            "No non-promoted SKU-weeks with an actively-promoted subcategory sibling; "
            "cannot fit a substitution effect."
        )

    candidate["ratio"] = (candidate["units"] / candidate["baseline"]).clip(*_DIP_BOUNDS)

    pooled_dip = float(candidate["ratio"].median())
    dip_by_subcategory: dict[str, float] = {_POOLED: pooled_dip}
    for subcategory, group in candidate.groupby("subcategory", sort=True):
        if len(group) < MIN_SUBSTITUTION_OBSERVATIONS:
            continue
        dip_by_subcategory[str(subcategory)] = float(group["ratio"].median())

    model = SubstitutionEffect(
        dip_by_subcategory=dip_by_subcategory,
        diagnostics={
            "candidate_weeks": float(len(candidate)),
            "subcategories_with_own_estimate": float(len(dip_by_subcategory) - 1),
            "pooled_dip": round(pooled_dip, 4),
            "pooled_demand_pulled_away_pct": round((1.0 - pooled_dip) * 100.0, 2),
        },
    )
    log.info("fitted cross-sku substitution effect", extra={"context": model.diagnostics})
    return model
