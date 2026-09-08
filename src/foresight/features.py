"""Feature engineering for the weekly demand forecast (deliverable D3).

THE LEAKAGE CONTRACT
--------------------
Every feature is built at a fixed **origin week** ``t`` and used to predict
``t + h``. Features fall into exactly two families, and nothing else is allowed:

1. **History-derived** - functions of the SKU's own past, evaluated at ``t`` and
   inclusive of ``t``. The week ``t`` has finished by the time we forecast, so
   its own value is known. Nothing after ``t`` is ever touched.

2. **Calendar-derived** - attributes of the *target* week ``t + h`` taken from
   the calendar dimension: week of year, month, holiday count, and whether a
   named promotional event is scheduled. These are legitimate because the
   calendar is a forward-published planning artifact - NorthBay set next
   quarter's promotion dates months ago, so they are genuinely known at forecast
   time. The *realised* ``promo_flag`` from the sales ledger is deliberately not
   used for the target week: that is an outcome, not a plan.

One feature deserves specific mention. ``units_seasonal_lag`` is demand in week
``t + h - 52``. Because the horizon never exceeds 26 weeks, ``t + h - 52`` is
always strictly before ``t``, so it is known at forecast time. It hands the model
the seasonal-naive baseline's own prediction as an input, which is exactly what
lets a learned model improve on it rather than rediscover it.

Section 7.1 of the brief makes leak-free features the non-negotiable rule of this
engagement, so :func:`assert_no_leakage` does not take that on trust. It rebuilds
the features against a copy of the panel whose future has been overwritten with
noise, and fails the build if a single feature value at or before the cutoff
moves. A model that cannot pass it never gets trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from foresight.exceptions import InsufficientHistoryError, LeakageError
from foresight.logging_setup import get_logger

__all__ = [
    "CATEGORICAL_FEATURES",
    "FeatureSpec",
    "assert_no_leakage",
    "build_base_features",
    "build_inference_frame",
    "build_supervised_frame",
    "build_weekly_calendar",
    "resolve_feature_spec",
]

log = get_logger(__name__)

#: Lags of weekly demand, measured back from the origin week (0 = origin).
DEMAND_LAGS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 8, 13, 26, 52)

#: Rolling windows ending at (and including) the origin week.
ROLLING_WINDOWS: Final[tuple[int, ...]] = (4, 8, 13, 26, 52)

#: Weeks in a seasonal cycle.
SEASONAL_PERIOD: Final[int] = 52

CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("category", "subcategory")

_EPSILON: Final[float] = 1e-6


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The feature contract shared by training, backtesting and serving."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def all_features(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
def build_weekly_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Collapse the daily calendar to one row per week.

    Covers the forecast horizon as well as history, because target-week features
    are needed for weeks that have not happened yet.
    """
    aggregations: dict[str, tuple[str, object]] = {
        "iso_week": ("iso_week", "first"),
        "month": ("month", "first"),
        "holiday_days": ("is_holiday", "sum"),
        "promo_event_days": (
            "promo_event",
            lambda values: int((values.astype(str) != "").sum()),
        ),
        "days_in_week": ("date", "count"),
    }
    # Planned promotional depth, published with the calendar ahead of the event.
    if "planned_discount" in calendar.columns:
        aggregations["planned_discount"] = ("planned_discount", "max")

    weekly = calendar.groupby("week_start", as_index=False).agg(**aggregations)
    if "planned_discount" not in weekly.columns:
        weekly["planned_discount"] = 0.0
    weekly["planned_discount"] = weekly["planned_discount"].fillna(0.0).astype("float64")
    weekly["has_promo_event"] = (weekly["promo_event_days"] > 0).astype("int64")

    # Cyclical encoding: week 52 and week 1 are one week apart, but the raw
    # integers are 51 apart. Sine/cosine keeps that adjacency intact.
    radians = 2.0 * np.pi * weekly["iso_week"].astype("float64") / 52.0
    weekly["week_sin"] = np.sin(radians)
    weekly["week_cos"] = np.cos(radians)

    weekly["holiday_days"] = weekly["holiday_days"].astype("float64")
    return weekly.sort_values("week_start").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Origin-level features
# --------------------------------------------------------------------------- #
def build_base_features(panel: pd.DataFrame, *, use_drivers: bool = False) -> pd.DataFrame:
    """Build history-derived features for every (SKU, origin week).

    Args:
        panel: The weekly panel, one row per SKU per week, sorted or not.
        use_drivers: Also build origin-side commercial driver features from the
            optional ``foresight_drivers`` extension.

    Returns:
        The panel plus one column per feature. Rows keep their origin week; the
        horizon expansion happens in :func:`build_supervised_frame`.
    """
    if panel.empty:
        raise InsufficientHistoryError("Weekly panel is empty; cannot build features.")

    frame = panel.sort_values(["sku_id", "week_start"]).reset_index(drop=True)
    grouped = frame.groupby("sku_id", sort=False)

    # --- Demand lags -------------------------------------------------------- #
    for lag in DEMAND_LAGS:
        frame[f"units_lag_{lag}"] = grouped["units"].shift(lag)

    # --- Rolling statistics, ending at and including the origin ------------- #
    for window in ROLLING_WINDOWS:
        minimum = max(2, window // 4)
        frame[f"units_roll_mean_{window}"] = grouped["units"].transform(
            lambda series, w=window, m=minimum: series.rolling(w, min_periods=m).mean()
        )
        frame[f"units_roll_std_{window}"] = grouped["units"].transform(
            lambda series, w=window, m=minimum: series.rolling(w, min_periods=m).std()
        )

    frame["units_roll_max_13"] = grouped["units"].transform(
        lambda series: series.rolling(13, min_periods=3).max()
    )

    # --- Intermittency: how often does this SKU sell nothing at all? -------- #
    is_zero = (frame["units"] <= 0).astype("float64")
    frame["_is_zero"] = is_zero
    zero_grouped = frame.groupby("sku_id", sort=False)["_is_zero"]
    for window in (8, 26):
        frame[f"zero_frac_{window}"] = zero_grouped.transform(
            lambda series, w=window: series.rolling(w, min_periods=max(2, w // 4)).mean()
        )
    frame = frame.drop(columns=["_is_zero"])

    # --- Shape of demand ---------------------------------------------------- #
    # Short window over long window: > 1 means the SKU is accelerating.
    frame["trend_ratio_4_13"] = frame["units_roll_mean_4"] / (
        frame["units_roll_mean_13"] + _EPSILON
    )
    frame["trend_ratio_13_52"] = frame["units_roll_mean_13"] / (
        frame["units_roll_mean_52"] + _EPSILON
    )
    # Coefficient of variation: the volatility signal the risk layer needs.
    frame["cv_13"] = frame["units_roll_std_13"] / (frame["units_roll_mean_13"] + _EPSILON)

    # --- Price ------------------------------------------------------------- #
    frame["price_roll_mean_4"] = grouped["avg_price"].transform(
        lambda series: series.rolling(4, min_periods=1).mean()
    )
    frame["price_vs_list"] = frame["avg_price"] / (frame["list_price"] + _EPSILON)
    frame["discount_depth"] = (1.0 - frame["price_vs_list"]).clip(lower=0.0)
    frame["gross_margin_per_unit"] = frame["list_price"] - frame["unit_cost"]

    # --- Promotion history (origin side only) ------------------------------- #
    frame["promo_days_roll_4"] = grouped["promo_days"].transform(
        lambda series: series.rolling(4, min_periods=1).sum()
    )
    frame["promo_days_roll_13"] = grouped["promo_days"].transform(
        lambda series: series.rolling(13, min_periods=1).sum()
    )

    # --- Commercial drivers, origin side ------------------------------------ #
    # Delegated to the optional extension package. Absent, or switched off, the
    # model forecasts from history alone and every contract below still holds.
    if use_drivers:
        from foresight_drivers import build_driver_features

        frame = build_driver_features(frame)

    log.info(
        "built base features",
        extra={
            "context": {
                "rows": len(frame),
                "skus": int(frame["sku_id"].nunique()),
                "drivers": use_drivers,
            }
        },
    )
    return frame


#: Numeric features produced at the origin, before horizon expansion.
_ORIGIN_NUMERIC: Final[tuple[str, ...]] = (
    *(f"units_lag_{lag}" for lag in DEMAND_LAGS),
    *(f"units_roll_mean_{window}" for window in ROLLING_WINDOWS),
    *(f"units_roll_std_{window}" for window in ROLLING_WINDOWS),
    "units_roll_max_13",
    "zero_frac_8",
    "zero_frac_26",
    "trend_ratio_4_13",
    "trend_ratio_13_52",
    "cv_13",
    "avg_price",
    "price_roll_mean_4",
    "price_vs_list",
    "discount_depth",
    "gross_margin_per_unit",
    "unit_cost",
    "list_price",
    "promo_days_roll_4",
    "promo_days_roll_13",
    "weeks_since_launch",
)

#: Features that depend on the horizon or the target week.
_TARGET_NUMERIC: Final[tuple[str, ...]] = (
    "horizon",
    "units_seasonal_lag",
    "target_week_sin",
    "target_week_cos",
    "target_month",
    "target_holiday_days",
    "target_has_promo_event",
)

#: The history-only contract - exactly the four Appendix A extracts.
FEATURE_SPEC: Final[FeatureSpec] = FeatureSpec(
    numeric=_ORIGIN_NUMERIC + _TARGET_NUMERIC,
    categorical=CATEGORICAL_FEATURES,
)


def resolve_feature_spec(*, use_drivers: bool = False) -> FeatureSpec:
    """Return the feature contract for a given driver setting.

    Kept a function rather than a second constant so a model can only ever be
    trained and served against one consistent contract - the spec travels inside
    the fitted model, so train and serve cannot disagree about which columns
    exist.
    """
    if not use_drivers:
        return FEATURE_SPEC

    from foresight_drivers import DRIVER_ORIGIN_FEATURES, DRIVER_TARGET_FEATURES

    return FeatureSpec(
        numeric=FEATURE_SPEC.numeric + DRIVER_ORIGIN_FEATURES + DRIVER_TARGET_FEATURES,
        categorical=CATEGORICAL_FEATURES,
    )


def _attach_target_week_features(
    frame: pd.DataFrame,
    weekly_calendar: pd.DataFrame,
    horizon: int,
    *,
    use_drivers: bool = False,
    marketing_plan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach attributes of the target week ``t + horizon``.

    Only forward-knowable information is attached here: the published calendar,
    and - when the extension is active - committed media plans and promotional
    depth. Observed-only drivers stay on the origin side.
    """
    frame = frame.copy()
    frame["horizon"] = horizon
    frame["target_week"] = frame["week_start"] + pd.to_timedelta(horizon * 7, unit="D")

    calendar_columns = [
        "week_start",
        "week_sin",
        "week_cos",
        "month",
        "holiday_days",
        "has_promo_event",
    ]
    target_calendar = weekly_calendar[calendar_columns].rename(
        columns={
            "week_start": "target_week",
            "week_sin": "target_week_sin",
            "week_cos": "target_week_cos",
            "month": "target_month",
            "holiday_days": "target_holiday_days",
            "has_promo_event": "target_has_promo_event",
        }
    )
    frame = frame.merge(target_calendar, on="target_week", how="left", validate="many_to_one")

    if use_drivers:
        from foresight_drivers import attach_target_drivers

        frame = attach_target_drivers(frame, marketing_plan, weekly_calendar)

    return frame


def build_supervised_frame(
    base: pd.DataFrame,
    weekly_calendar: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    use_drivers: bool = False,
    marketing_plan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expand origin-level features into one training row per (origin, horizon).

    The target ``y`` is demand in week ``t + h``, taken by shifting each SKU's
    series backwards. Rows whose target lies beyond the end of history have no
    label and are dropped.
    """
    if not horizons:
        raise ValueError("horizons must not be empty")

    base = base.sort_values(["sku_id", "week_start"]).reset_index(drop=True)
    grouped = base.groupby("sku_id", sort=False)["units"]

    parts: list[pd.DataFrame] = []
    for horizon in horizons:
        part = _attach_target_week_features(
            base,
            weekly_calendar,
            horizon,
            use_drivers=use_drivers,
            marketing_plan=marketing_plan,
        )

        # Target: demand h weeks after the origin.
        part["y"] = grouped.shift(-horizon).to_numpy()

        # Seasonal lag of the TARGET week: units at t + h - 52. Since
        # h < 52, this is always strictly before the origin, so it is known.
        part["units_seasonal_lag"] = grouped.shift(SEASONAL_PERIOD - horizon).to_numpy()

        parts.append(part)

    supervised = pd.concat(parts, ignore_index=True)
    labelled = supervised[supervised["y"].notna()].reset_index(drop=True)

    log.info(
        "built supervised frame",
        extra={
            "context": {
                "rows": len(labelled),
                "horizons": list(horizons),
                "dropped_unlabelled": len(supervised) - len(labelled),
            }
        },
    )
    return labelled


def build_inference_frame(
    base: pd.DataFrame,
    weekly_calendar: pd.DataFrame,
    origin_week: pd.Timestamp,
    horizons: tuple[int, ...],
    *,
    use_drivers: bool = False,
    marketing_plan: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build unlabelled rows for forecasting forward from ``origin_week``.

    Used by the scoring service and the final forecast. Identical feature
    construction to training - the same code path, so train and serve cannot
    drift apart.
    """
    base = base.sort_values(["sku_id", "week_start"]).reset_index(drop=True)
    grouped = base.groupby("sku_id", sort=False)["units"]

    seasonal_by_horizon = {
        horizon: grouped.shift(SEASONAL_PERIOD - horizon).to_numpy() for horizon in horizons
    }
    origin_mask = (base["week_start"] == origin_week).to_numpy()
    if not origin_mask.any():
        raise InsufficientHistoryError(
            f"No panel rows at origin week {origin_week.date()}; cannot forecast forward."
        )

    parts: list[pd.DataFrame] = []
    for horizon in horizons:
        part = _attach_target_week_features(
            base,
            weekly_calendar,
            horizon,
            use_drivers=use_drivers,
            marketing_plan=marketing_plan,
        )
        part["units_seasonal_lag"] = seasonal_by_horizon[horizon]
        parts.append(part.loc[origin_mask])

    inference = pd.concat(parts, ignore_index=True)
    log.info(
        "built inference frame",
        extra={
            "context": {
                "origin_week": str(origin_week.date()),
                "rows": len(inference),
                "skus": int(inference["sku_id"].nunique()),
            }
        },
    )
    return inference


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #
#: Panel columns the leakage guard overwrites after the cutoff. Every one is an
#: *observation* - something recorded once the week has happened. Whatever the
#: feature layer derives from these must be invariant at origins at or before the
#: cutoff, or it read the future. Forward-committed plans (the media plan, the
#: promo calendar) are deliberately absent: those are knowable in advance, which
#: is the entire reason they are allowed as target-week features.
_OBSERVED_PANEL_COLUMNS: Final[tuple[str, ...]] = (
    "units",
    "avg_price",
    "revenue",
    "promo_days",
    "sessions",
    "add_to_cart",
    "discount_pct",
    "media_spend",
    "impressions",
    "email_sends",
    "competitor_price_index",
    "weather_anomaly",
)


def assert_no_leakage(
    panel: pd.DataFrame,
    weekly_calendar: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    use_drivers: bool = False,
    marketing_plan: pd.DataFrame | None = None,
    cutoff_quantile: float = 0.7,
    seed: int = 0,
) -> None:
    """Prove empirically that no feature reads the future. Raises if one does.

    The test is deliberately behavioural rather than a code review: build the
    features once on the real panel, then again on a copy whose demand *after* a
    cutoff week has been replaced with noise. Any feature value at an origin at
    or before the cutoff that changes between the two runs must have read data
    from after that origin. There is nowhere else the difference could come from.

    This catches the mistakes a visual inspection misses - a centred rolling
    window, a ``shift`` with the wrong sign, a global mean computed over the full
    series, a merge that quietly pulls a future row in.

    Args:
        panel: The weekly panel to test against.
        weekly_calendar: Weekly calendar dimension.
        horizons: Horizons to expand to.
        use_drivers: Check the commercial driver features as well. When true,
            every observed driver column is corrupted alongside demand, so a
            driver feature that peeks is caught by the same mechanism.
        marketing_plan: Forward media plan, needed to build the target-week
            driver features when ``use_drivers`` is set.
        cutoff_quantile: Where in the timeline to place the cutoff.
        seed: Seed for the corrupting noise.

    Raises:
        LeakageError: naming the offending feature columns.
    """
    weeks = np.sort(panel["week_start"].unique())
    if len(weeks) < 10:
        raise InsufficientHistoryError("Too few weeks to run a meaningful leakage check.")
    cutoff = pd.Timestamp(weeks[int(len(weeks) * cutoff_quantile)])

    spec = resolve_feature_spec(use_drivers=use_drivers)

    def _build(source: pd.DataFrame) -> pd.DataFrame:
        return build_supervised_frame(
            build_base_features(source, use_drivers=use_drivers),
            weekly_calendar,
            horizons,
            use_drivers=use_drivers,
            marketing_plan=marketing_plan,
        )

    honest = _build(panel)

    # Overwrite everything strictly after the cutoff with noise on a different
    # scale, so any dependency on it shows up as a large, unmistakable change.
    rng = np.random.default_rng(seed)
    corrupted_panel = panel.copy()
    future_mask = (corrupted_panel["week_start"] > cutoff).to_numpy()
    future_rows = int(future_mask.sum())
    corrupted_columns = [
        column for column in _OBSERVED_PANEL_COLUMNS if column in corrupted_panel.columns
    ]
    for column in corrupted_columns:
        corrupted_panel[column] = corrupted_panel[column].astype("float64")
        corrupted_panel.loc[future_mask, column] = rng.uniform(5_000.0, 10_000.0, size=future_rows)
    corrupted = _build(corrupted_panel)

    key = ["sku_id", "week_start", "horizon"]
    honest_past = honest[honest["week_start"] <= cutoff].set_index(key).sort_index()
    corrupted_past = corrupted[corrupted["week_start"] <= cutoff].set_index(key).sort_index()

    shared = honest_past.index.intersection(corrupted_past.index)
    if shared.empty:
        raise InsufficientHistoryError("Leakage check produced no comparable rows.")

    offenders: list[str] = []
    for column in spec.numeric:
        if column not in honest_past.columns:
            continue
        left = honest_past.loc[shared, column].to_numpy(dtype="float64", na_value=np.nan)
        right = corrupted_past.loc[shared, column].to_numpy(dtype="float64", na_value=np.nan)
        if not np.allclose(left, right, rtol=1e-9, atol=1e-9, equal_nan=True):
            offenders.append(column)

    if offenders:
        raise LeakageError(
            "Feature(s) changed when only post-cutoff data was altered, so they read "
            f"the future: {', '.join(sorted(offenders))}. "
            f"Cutoff week {cutoff.date()}, {len(shared)} rows compared."
        )

    log.info(
        "leakage check passed",
        extra={
            "context": {
                "cutoff_week": str(cutoff.date()),
                "rows_compared": len(shared),
                "features_checked": len(spec.numeric),
                "columns_corrupted": len(corrupted_columns),
                "drivers": use_drivers,
            }
        },
    )
