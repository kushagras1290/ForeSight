"""Driver features, split by what is knowable at forecast time.

THE SPLIT THIS MODULE EXISTS TO ENFORCE
---------------------------------------
A driver may be used for week ``t + h`` only if it is genuinely knowable at week
``t``. Some are, and some are not, and the difference is not obvious from the
column names:

**Forward-known** - attached at the TARGET week.
  ``media_spend``       a media plan is committed weeks ahead; that is what a
                        plan is.
  ``planned_discount``  published with the promo calendar before the event.

**Observed only** - attached at the ORIGIN, then lagged.
  ``sessions``, ``add_to_cart``   traffic is measured after it happens.
  ``discount_pct``               the realised discount is an outcome.
  ``competitor_price_index``     next quarter's scrape does not exist yet.
  ``weather_anomaly``            not forecastable eight weeks out.

The last two are still useful because both series are highly persistent: today's
level is informative about the target week even though the target week's value is
unknown. That is a lag, not a peek.

The two families are built by two different functions and never merged, so
using an observed-only driver as if it were forward-known requires deliberately
calling the wrong function rather than mistyping a column name.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from foresight.logging_setup import get_logger

__all__ = [
    "DRIVER_DEFAULTS",
    "DRIVER_ORIGIN_FEATURES",
    "DRIVER_TARGET_FEATURES",
    "attach_target_drivers",
    "build_driver_features",
    "driver_columns_with_defaults",
]

log = get_logger(__name__)

_EPSILON: Final[float] = 1e-6

#: Raw driver column -> the value to use when the extract is absent. Chosen so
#: that a missing driver is neutral: no spend, no traffic, price parity.
DRIVER_DEFAULTS: Final[dict[str, float]] = {
    "media_spend": 0.0,
    "impressions": 0.0,
    "email_sends": 0.0,
    "sessions": 0.0,
    "add_to_cart": 0.0,
    "discount_pct": 0.0,
    "competitor_price_index": 1.0,
    "weather_anomaly": 0.0,
}

#: Features built at the origin from observed driver history.
DRIVER_ORIGIN_FEATURES: Final[tuple[str, ...]] = (
    "media_spend_roll_4",
    "media_spend_roll_13",
    "sessions_roll_4",
    "sessions_roll_13",
    "add_to_cart_roll_4",
    "discount_roll_4",
    "sessions_trend_4_13",
    "cart_conversion_4",
    "units_per_session_4",
    "media_trend_4_13",
    "log_media_spend",
    "competitor_price_index",
    "weather_anomaly",
)

#: Features describing the target week, from forward-committed plans only.
DRIVER_TARGET_FEATURES: Final[tuple[str, ...]] = (
    "target_media_spend_log",
    "target_media_ratio",
    "target_planned_discount",
)


def driver_columns_with_defaults(frame: pd.DataFrame) -> pd.DataFrame:
    """Guarantee every raw driver column exists, neutral-filled if absent."""
    out = frame.copy()
    for column, default in DRIVER_DEFAULTS.items():
        if column not in out.columns:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default).astype("float64")
    return out


def build_driver_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build origin-side driver features on a SKU-week panel.

    Every rolling window ends at and includes the origin week, matching the
    convention in :mod:`foresight.features`. Nothing here reads past it.
    """
    out = driver_columns_with_defaults(frame)
    grouped = out.groupby("sku_id", sort=False)

    for window in (4, 13):
        out[f"media_spend_roll_{window}"] = grouped["media_spend"].transform(
            lambda series, w=window: series.rolling(w, min_periods=1).mean()
        )
        out[f"sessions_roll_{window}"] = grouped["sessions"].transform(
            lambda series, w=window: series.rolling(w, min_periods=1).mean()
        )

    out["add_to_cart_roll_4"] = grouped["add_to_cart"].transform(
        lambda series: series.rolling(4, min_periods=1).mean()
    )
    out["discount_roll_4"] = grouped["discount_pct"].transform(
        lambda series: series.rolling(4, min_periods=1).mean()
    )

    # Traffic leads demand: intent shows up on the site before the ledger.
    out["sessions_trend_4_13"] = out["sessions_roll_4"] / (out["sessions_roll_13"] + _EPSILON)
    out["cart_conversion_4"] = out["add_to_cart_roll_4"] / (out["sessions_roll_4"] + _EPSILON)
    out["units_per_session_4"] = out["units_roll_mean_4"] / (out["sessions_roll_4"] + _EPSILON)
    out["media_trend_4_13"] = out["media_spend_roll_4"] / (out["media_spend_roll_13"] + _EPSILON)
    # Diminishing returns, matching how media actually behaves.
    out["log_media_spend"] = np.log1p(out["media_spend"].clip(lower=0.0))

    return out


def attach_target_drivers(
    frame: pd.DataFrame,
    marketing_plan: pd.DataFrame | None,
    weekly_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Attach forward-committed driver values for the target week.

    Args:
        frame: Rows carrying ``sku_id`` and ``target_week``.
        marketing_plan: The cleaned media plan, which extends past the end of
            the sales history. ``None`` leaves the media features neutral.
        weekly_calendar: Weekly calendar, carrying ``planned_discount``.

    Only committed plans are read here. A driver that is merely *observed* would
    not exist for a future target week, and reading one for a historical target
    week would train the model on information it will not have at inference -
    which is the failure this split prevents.
    """
    out = frame.copy()

    # --- Committed media spend for the target week -------------------------- #
    if marketing_plan is not None and not marketing_plan.empty:
        plan = marketing_plan.loc[:, ["sku_id", "week_start", "media_spend"]].rename(
            columns={"week_start": "target_week", "media_spend": "target_media_spend"}
        )
        plan["target_week"] = pd.to_datetime(plan["target_week"])
        out = out.merge(plan, on=["sku_id", "target_week"], how="left", validate="many_to_one")
    if "target_media_spend" not in out.columns:
        out["target_media_spend"] = 0.0
    out["target_media_spend"] = out["target_media_spend"].fillna(0.0).astype("float64")

    out["target_media_spend_log"] = np.log1p(out["target_media_spend"].clip(lower=0.0))
    # Planned pressure relative to this product's recent norm. A ratio travels
    # across SKUs of wildly different budget in a way an absolute figure cannot.
    baseline = out.get("media_spend_roll_13")
    if baseline is None:
        out["target_media_ratio"] = 1.0
    else:
        out["target_media_ratio"] = (out["target_media_spend"] + _EPSILON) / (
            baseline.astype("float64") + _EPSILON
        )
    out["target_media_ratio"] = out["target_media_ratio"].clip(0.0, 20.0)

    # --- Published promotional depth for the target week -------------------- #
    if "planned_discount" in weekly_calendar.columns:
        published = weekly_calendar.loc[:, ["week_start", "planned_discount"]].rename(
            columns={"week_start": "target_week", "planned_discount": "target_planned_discount"}
        )
        out = out.merge(published, on="target_week", how="left", validate="many_to_one")
    if "target_planned_discount" not in out.columns:
        out["target_planned_discount"] = 0.0
    out["target_planned_discount"] = out["target_planned_discount"].fillna(0.0).astype("float64")

    return out
