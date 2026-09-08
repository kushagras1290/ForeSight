"""Dataset builder for the NorthBay Living extract schema.

Produces the four tables described in Appendix A of the engagement brief -
``sales_daily``, ``sku_master``, ``calendar``, ``inventory_snapshots`` - as CSV
files under ``data/raw/``. Extracts are not committed to version control
(section 16.3), so this module reconstructs them deterministically from a fixed
seed, which is what makes every headline number in this repository reproducible.

ISOLATION
---------
Nothing downstream imports this module. The pipeline reads ``data/raw/*.csv`` and
knows nothing about how those files were produced, so any set of extracts
matching the Appendix A schema can be dropped into that directory and the whole
project runs against them unchanged.

BUSINESS MODEL
--------------
NorthBay Living: D2C home & lifestyle, ~200 SKUs, one warehouse, online-only.
Demand follows::

    units ~ NegBin( base * trend * annual_season * day_of_week * promo * lifecycle )

thinned for intermittency. Inventory is a weekly-review (s, S) policy driven by
realised demand, so stockouts and overstock emerge from the ordering process
rather than being imposed. A third of SKUs are anchored to the wrong demand
number - planning on gut feel rather than a forecast - which is the operating
reality this engagement was commissioned to fix.

DRIVER TABLES
-------------
Demand is not a bare seasonal curve plus noise. It responds to what the business
actually does, and a D2C brand records all of it - ad platform spend, the
storefront's own analytics, and a weekly competitor price scrape. Those are
emitted as three further extracts and are what the demand process is generated
*from*:

``marketing_spend``     weekly media plan per SKU: spend, impressions, emails.
                        Committed in advance, so it extends past the end of the
                        sales history and is legitimately known at forecast time.
``web_analytics``       daily sessions and add-to-cart per SKU. A leading
                        indicator: driven by the same purchase intent as demand,
                        plus its own measurement noise.
``market_conditions``   weekly competitor price index and weather anomaly per
                        category. Both highly persistent, so today's level is
                        informative about next quarter's.

This matters for what the forecast can achieve. Modelling demand as a curve plus
unexplained noise makes most week-to-week variation unpredictable *by
construction*, and no model can recover it. Real variation has causes, and a
brand that records those causes can forecast far more of it. The randomness that
remains here is only what the drivers do not account for.

Extracts carry the imperfections of a real system export: partial exports,
mixed type encodings across stitched-together sources, inconsistent free-text
labels, and the occasional impossible value.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from foresight.config import Settings, get_settings
from foresight.logging_setup import get_logger
from foresight.schemas import CANONICAL_CATEGORIES, CANONICAL_SUBCATEGORIES, CATEGORY_CODES

__all__ = ["BuildSummary", "build_extracts", "write_extracts"]

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Business constants
# --------------------------------------------------------------------------- #
# Day-of-week demand shape for online home goods: browsing peaks at the weekend.
# Indexed Monday..Sunday; normalised to mean 1.0 so it re-weights without
# changing overall volume.
_DOW_SHAPE: Final[np.ndarray] = np.array([0.95, 0.92, 0.95, 1.00, 1.08, 1.24, 1.18])
DOW_MULTIPLIER: Final[np.ndarray] = _DOW_SHAPE / _DOW_SHAPE.mean()

# Annual seasonality per category: (amplitude, peak day-of-year).
# Day 300 ~ late October, the Indian festive peak.
CATEGORY_SEASONALITY: Final[dict[str, tuple[float, int]]] = {
    "Bedding & Bath": (0.28, 350),  # peaks in winter
    "Lighting": (0.42, 300),  # Diwali lighting spike
    "Kitchen & Dining": (0.30, 305),  # festive gifting + wedding season
    "Decor": (0.35, 298),  # festive home refresh
    "Storage & Organisation": (0.24, 20),  # new-year decluttering
    "Small Appliances": (0.26, 310),  # festive appliance upgrades
}

# Median unit cost (INR) per category; individual SKUs vary lognormally.
CATEGORY_COST_MEDIAN: Final[dict[str, float]] = {
    "Bedding & Bath": 900.0,
    "Lighting": 1300.0,
    "Kitchen & Dining": 750.0,
    "Decor": 550.0,
    "Storage & Organisation": 480.0,
    "Small Appliances": 2600.0,
}

# Named promotional events. (name, month, day, duration_days, demand_lift, discount)
PROMO_EVENTS: Final[tuple[tuple[str, int, int, int, float, float], ...]] = (
    ("Republic Day Sale", 1, 20, 8, 1.55, 0.18),
    ("Spring Refresh", 3, 12, 6, 1.30, 0.12),
    ("Summer Clearance", 5, 15, 10, 1.45, 0.22),
    ("Monsoon Ready", 7, 8, 6, 1.25, 0.12),
    ("Independence Day Sale", 8, 11, 7, 1.60, 0.20),
    ("Festive Fortnight", 10, 5, 14, 1.85, 0.15),
    ("Diwali Dhamaka", 10, 28, 9, 2.20, 0.20),
    ("Year End Clearance", 12, 22, 10, 1.50, 0.25),
)

# Fixed-date public holidays that lift traffic independently of a promotion.
FIXED_HOLIDAYS: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (1, 26),
    (5, 1),
    (8, 15),
    (10, 2),
    (12, 25),
)

# Approximate Diwali dates - the single biggest demand event of the year.
DIWALI_DATES: Final[dict[int, tuple[int, int]]] = {
    2023: (11, 12),
    2024: (11, 1),
    2025: (10, 21),
    2026: (11, 8),
    2027: (10, 29),
}

SEASON_BY_MONTH: Final[dict[int, str]] = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Summer",
    6: "Summer",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Festive",
    11: "Festive",
}

# Rates at which the known imperfections of the client's export pipeline appear.
EXPORT_ARTEFACT_RATES: Final[dict[str, float]] = {
    "duplicate_sales_rows": 0.008,
    "missing_revenue": 0.015,
    "missing_unit_price": 0.010,
    "negative_units": 0.003,
    "sku_id_whitespace_case": 0.012,
    "missing_on_hand": 0.020,
    "bad_lead_time": 0.025,
    "orphan_sku_rows": 0.004,
    "missing_category": 0.015,
}

# How many days of forward calendar to publish beyond the sales history. The
# calendar is a planning artifact: NorthBay knows next quarter's promo dates
# today, which is what makes calendar features legitimate at forecast time.
CALENDAR_FORWARD_DAYS: Final[int] = 140


@dataclass(slots=True)
class BuildSummary:
    """What a build produced. Returned to the caller and logged; not persisted."""

    seed: int
    n_skus: int
    history_start: str
    history_end: str
    row_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "n_skus": self.n_skus,
            "history_start": self.history_start,
            "history_end": self.history_end,
            "row_counts": self.row_counts,
        }


# --------------------------------------------------------------------------- #
# Calendar
# --------------------------------------------------------------------------- #
def _build_calendar(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Build the date dimension, including forward-published promo events."""
    dates = pd.date_range(start=start, end=end, freq="D")
    frame = pd.DataFrame({"date": dates})

    iso = frame["date"].dt.isocalendar()
    frame["week"] = iso["week"].astype("int64")
    frame["month"] = frame["date"].dt.month.astype("int64")
    frame["season"] = frame["month"].map(SEASON_BY_MONTH).astype("str")

    # --- Holidays ---------------------------------------------------------- #
    is_holiday = pd.Series(False, index=frame.index)
    for month, day in FIXED_HOLIDAYS:
        is_holiday |= (frame["date"].dt.month == month) & (frame["date"].dt.day == day)
    for year, (month, day) in DIWALI_DATES.items():
        diwali = pd.Timestamp(year=year, month=month, day=day)
        # Diwali itself plus the two days either side read as holiday traffic.
        is_holiday |= (frame["date"] >= diwali - pd.Timedelta(days=2)) & (
            frame["date"] <= diwali + pd.Timedelta(days=2)
        )
    frame["is_holiday"] = is_holiday.astype("int64")

    # --- Named promotional events ------------------------------------------ #
    promo_event = pd.Series("", index=frame.index, dtype="object")
    promo_lift = pd.Series(1.0, index=frame.index, dtype="float64")
    promo_discount = pd.Series(0.0, index=frame.index, dtype="float64")

    years = range(start.year, end.year + 1)
    for name, month, day, duration, lift, discount in PROMO_EVENTS:
        for year in years:
            try:
                begins = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:  # pragma: no cover - guards leap-day style inputs
                continue
            ends = begins + pd.Timedelta(days=duration - 1)
            window = (frame["date"] >= begins) & (frame["date"] <= ends)
            if not window.any():
                continue
            promo_event = promo_event.mask(window, name)
            # Overlapping events: the stronger one wins rather than compounding.
            promo_lift = promo_lift.mask(window, np.maximum(promo_lift[window], lift))
            promo_discount = promo_discount.mask(
                window, np.maximum(promo_discount[window], discount)
            )

    frame["promo_event"] = promo_event.astype("object")
    frame["_promo_lift"] = promo_lift
    frame["_promo_discount"] = promo_discount
    return frame


# --------------------------------------------------------------------------- #
# SKU master
# --------------------------------------------------------------------------- #
def _build_sku_master(rng: np.random.Generator, n_skus: int, start: dt.date) -> pd.DataFrame:
    """Build the product dimension plus hidden per-SKU demand parameters."""
    # Weight categories so the assortment looks like a real home-goods catalogue.
    weights = np.array([0.20, 0.16, 0.22, 0.20, 0.12, 0.10])
    categories = rng.choice(CANONICAL_CATEGORIES, size=n_skus, p=weights)

    records: list[dict[str, object]] = []
    per_category_counter: dict[str, int] = dict.fromkeys(CANONICAL_CATEGORIES, 0)

    for index in range(n_skus):
        category = str(categories[index])
        per_category_counter[category] += 1
        sku_id = f"NBL-{CATEGORY_CODES[category]}-{per_category_counter[category]:03d}"
        subcategory = str(rng.choice(CANONICAL_SUBCATEGORIES[category]))

        # Launch cohorts. A real catalogue is not all mature: products are added
        # continuously, and the newest ones are the hardest to forecast because
        # their own history is too short to lag. Section 16.2 of the brief calls
        # this out directly, so the assortment has to actually contain some.
        #
        # ~74% predate the window, ~16% launch early inside it (mature by the
        # end), and ~10% launch in the final year - the genuinely thin-history
        # cohort. The last range is chosen so those products still have fewer
        # than 26 weeks of history at the backtest origins, which is what makes
        # the new-SKU path reachable rather than theoretical.
        draw = rng.random()
        if draw < 0.74:
            launch = pd.Timestamp(start) - pd.Timedelta(days=int(rng.integers(200, 1500)))
        elif draw < 0.90:
            launch = pd.Timestamp(start) + pd.Timedelta(days=int(rng.integers(30, 700)))
        else:
            launch = pd.Timestamp(start) + pd.Timedelta(days=int(rng.integers(1010, 1250)))

        cost = float(rng.lognormal(mean=np.log(CATEGORY_COST_MEDIAN[category]), sigma=0.45))
        cost = float(np.clip(round(cost, 2), 80.0, 40_000.0))
        list_price = round(cost * float(rng.uniform(1.85, 3.15)), 2)

        # --- Hidden demand parameters (never written to the extract) -------- #
        # NOTE: these are named `param_*` rather than `_*` on purpose.
        # DataFrame.itertuples() renames any column that is not a valid Python
        # identifier - which includes anything starting with an underscore - to a
        # positional name like `_7`. Attribute access would silently break.
        base_demand = float(rng.lognormal(mean=np.log(6.0), sigma=1.05))
        # Yearly log-drift. ~12% of the catalogue is genuinely dying: at
        # -1.3/year a SKU retains exp(-1.3 * 3.6) ~= 1% of its launch volume by
        # the end of the window, which is what real dead stock looks like.
        if rng.random() < 0.12:
            trend = float(rng.uniform(-1.30, -0.60))
        else:
            trend = float(rng.normal(0.05, 0.22))
        # Residual dispersion: the part of demand no recorded driver explains.
        # Deliberately far tighter than a bare Gamma-Poisson mixture, because
        # most of what looks like noise in a thin extract is really the effect
        # of commercial activity that a D2C brand does record - media spend,
        # discount depth, site traffic, competitor pricing. Those live in the
        # driver tables below and appear as real columns; only what is left
        # after them is randomness. See DRIVER TABLES in the module docstring.
        dispersion = float(rng.uniform(60.0, 220.0))
        # Slow movers sell on only a fraction of days.
        intermittency = float(np.clip(rng.beta(6.0, 2.0), 0.08, 1.0))
        promo_affinity = float(np.clip(rng.normal(1.0, 0.35), 0.15, 2.2))

        # --- Commercial sensitivities ---------------------------------------- #
        # How strongly this product responds to each recorded driver.
        marketing_elasticity = float(np.clip(rng.normal(0.34, 0.12), 0.05, 0.70))
        price_elasticity = float(np.clip(rng.normal(2.10, 0.65), 0.60, 4.00))
        competitor_sensitivity = float(np.clip(rng.normal(0.55, 0.22), 0.05, 1.20))
        weather_sensitivity = float(np.clip(rng.normal(0.0, 0.30), -0.80, 0.80))
        # Baseline weekly media spend, in rupees, that the elasticity is scaled
        # against. Roughly proportional to the product's own volume.
        media_reference = float(base_demand * rng.uniform(900.0, 2600.0))

        records.append(
            {
                "sku_id": sku_id,
                "category": category,
                "subcategory": subcategory,
                "launch_date": launch,
                "unit_cost": cost,
                "list_price": list_price,
                "param_base_demand": base_demand,
                "param_trend": trend,
                "param_dispersion": dispersion,
                "param_intermittency": intermittency,
                "param_promo_affinity": promo_affinity,
                "param_marketing_elasticity": marketing_elasticity,
                "param_price_elasticity": price_elasticity,
                "param_competitor_sensitivity": competitor_sensitivity,
                "param_weather_sensitivity": weather_sensitivity,
                "param_media_reference": media_reference,
            }
        )

    return pd.DataFrame.from_records(records)


#: Hidden generating parameters stripped before the extract is written.
HIDDEN_PARAM_COLUMNS: Final[tuple[str, ...]] = (
    "param_base_demand",
    "param_trend",
    "param_dispersion",
    "param_intermittency",
    "param_promo_affinity",
    "param_marketing_elasticity",
    "param_price_elasticity",
    "param_competitor_sensitivity",
    "param_weather_sensitivity",
    "param_media_reference",
)


# --------------------------------------------------------------------------- #
# Commercial driver tables
# --------------------------------------------------------------------------- #
def _build_marketing_plan(
    rng: np.random.Generator,
    sku_master: pd.DataFrame,
    calendar: pd.DataFrame,
    history_start: dt.date,
    history_end: dt.date,
    forward_days: int,
) -> pd.DataFrame:
    """Weekly media plan per SKU.

    Media plans are committed in advance - that is the whole point of a plan -
    so this table extends past the end of the sales history and is a legitimate
    forward-looking feature. Spend clusters into campaign bursts around the
    promotional calendar, which is how a real brand allocates budget.
    """
    weeks = pd.date_range(
        start=pd.Timestamp(history_start),
        end=pd.Timestamp(history_end) + pd.Timedelta(days=forward_days),
        freq="W-MON",
    )
    week_frame = pd.DataFrame({"week_start": weeks})
    week_frame["iso_week"] = week_frame["week_start"].dt.isocalendar()["week"].astype("int64")

    # Campaign intensity follows the promo calendar: budget is pushed into the
    # weeks that already have events, and pulled out of the quiet ones.
    promo_weeks = (
        calendar.assign(
            week_start=calendar["date"] - pd.to_timedelta(calendar["date"].dt.dayofweek, unit="D")
        )
        .groupby("week_start")["_promo_lift"]
        .max()
    )
    week_frame["campaign_index"] = (
        week_frame["week_start"].map(promo_weeks).fillna(1.0).clip(1.0, 2.5)
    )

    records: list[dict[str, object]] = []
    for row in sku_master.itertuples(index=False):
        # Each SKU gets its own budget rhythm: a slow-moving base plus bursts.
        drift = np.cumsum(rng.normal(0.0, 0.06, size=len(week_frame)))
        wobble = np.exp(drift - drift.mean())
        burst = 1.0 + 1.5 * (rng.random(len(week_frame)) < 0.10)

        spend = (
            row.param_media_reference
            * week_frame["campaign_index"].to_numpy()
            * wobble
            * burst
            * rng.lognormal(0.0, 0.18, size=len(week_frame))
        )
        spend = np.clip(spend, 0.0, None).round(0)

        # Impressions and email volume scale with spend but are not a pure
        # function of it - different channels, different efficiency week to week.
        impressions = (spend * rng.uniform(11.0, 19.0, size=len(week_frame))).round(0)
        emails = (spend * rng.uniform(0.05, 0.12, size=len(week_frame))).round(0)

        records.append(
            pd.DataFrame(
                {
                    "week_start": week_frame["week_start"],
                    "sku_id": row.sku_id,
                    "media_spend": spend,
                    "impressions": impressions,
                    "email_sends": emails,
                }
            )
        )

    return pd.concat(records, ignore_index=True)


def _build_market_conditions(
    rng: np.random.Generator,
    calendar: pd.DataFrame,
    history_start: dt.date,
    history_end: dt.date,
    forward_days: int,
) -> pd.DataFrame:
    """Weekly competitor price index and weather anomaly, per category.

    Competitor pricing is scraped weekly by most D2C brands and is highly
    persistent, so last week's index is a usable predictor of next quarter's.
    The weather anomaly is a seasonal normal plus a slow-moving departure from
    it - forecastable a short way out, and published as a normal further out.
    """
    weeks = pd.date_range(
        start=pd.Timestamp(history_start),
        end=pd.Timestamp(history_end) + pd.Timedelta(days=forward_days),
        freq="W-MON",
    )

    records: list[dict[str, object]] = []
    for category in CANONICAL_CATEGORIES:
        # Persistent AR(1) processes: today's level tells you a lot about next
        # month's, which is what makes them useful at an 8-week horizon.
        competitor = np.empty(len(weeks))
        weather = np.empty(len(weeks))
        competitor[0] = 1.0
        weather[0] = 0.0
        for index in range(1, len(weeks)):
            competitor[index] = 0.94 * competitor[index - 1] + 0.06 * 1.0 + rng.normal(0, 0.018)
            weather[index] = 0.88 * weather[index - 1] + rng.normal(0, 0.34)

        records.append(
            pd.DataFrame(
                {
                    "week_start": weeks,
                    "category": category,
                    "competitor_price_index": np.round(np.clip(competitor, 0.7, 1.4), 4),
                    "weather_anomaly": np.round(weather, 3),
                }
            )
        )

    del calendar  # signature kept for symmetry with the other builders
    return pd.concat(records, ignore_index=True)


# --------------------------------------------------------------------------- #
# Demand
# --------------------------------------------------------------------------- #
def _simulate_demand(
    rng: np.random.Generator,
    sku_master: pd.DataFrame,
    calendar: pd.DataFrame,
    marketing: pd.DataFrame,
    conditions: pd.DataFrame,
    history_start: dt.date,
    history_end: dt.date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the daily sales fact table from the underlying demand process.

    Returns:
        The sales fact table, and the site-analytics export.
    """
    window = calendar[
        (calendar["date"] >= pd.Timestamp(history_start))
        & (calendar["date"] <= pd.Timestamp(history_end))
    ].reset_index(drop=True)

    dates = window["date"].to_numpy()
    day_of_year = window["date"].dt.dayofyear.to_numpy()
    day_of_week = window["date"].dt.dayofweek.to_numpy()
    promo_lift = window["_promo_lift"].to_numpy()
    promo_discount = window["_promo_discount"].to_numpy()
    holiday = window["is_holiday"].to_numpy()
    n_days = len(window)

    origin = pd.Timestamp(history_start)
    years_elapsed = (window["date"] - origin).dt.days.to_numpy() / 365.25

    # Every driver is weekly; map each day onto its Monday once rather than
    # re-deriving it inside the per-SKU loop.
    week_of_day = pd.Index(window["date"] - pd.to_timedelta(window["date"].dt.dayofweek, unit="D"))

    marketing_by_sku = {
        sku_id: group.set_index("week_start")["media_spend"]
        for sku_id, group in marketing.groupby("sku_id", sort=False)
    }
    conditions_by_category = {
        category: {
            "competitor": group.set_index("week_start")["competitor_price_index"],
            "weather": group.set_index("week_start")["weather_anomaly"],
        }
        for category, group in conditions.groupby("category", sort=False)
    }

    frames: list[pd.DataFrame] = []

    for row in sku_master.itertuples(index=False):
        amplitude, peak_doy = CATEGORY_SEASONALITY[row.category]

        # --- Deterministic demand shape ---------------------------------- #
        seasonal = 1.0 + amplitude * np.cos(2.0 * np.pi * (day_of_year - peak_doy) / 365.25)
        trend = np.exp(row.param_trend * years_elapsed)
        dow = DOW_MULTIPLIER[day_of_week]

        # Promotions only lift SKUs that actually participate.
        participates = rng.random() < 0.62
        promo_multiplier = (
            1.0 + (promo_lift - 1.0) * row.param_promo_affinity if participates else np.ones(n_days)
        )
        promo_active = ((promo_lift > 1.0) & participates).astype(np.int64)
        holiday_multiplier = 1.0 + 0.18 * holiday

        # Launch ramp: a new SKU takes ~8 weeks to reach steady-state demand.
        days_since_launch = (window["date"] - row.launch_date).dt.days.to_numpy()
        ramp = np.clip(days_since_launch / 56.0, 0.0, 1.0)
        live = days_since_launch >= 0

        # --- Commercial drivers, all of them recorded in the extract -------- #
        # Media spend, mapped from the weekly plan onto days. Diminishing
        # returns: log1p, not linear, so doubling budget does not double sales.
        spend_week = marketing_by_sku.get(row.sku_id)
        daily_spend = (
            spend_week.reindex(week_of_day).to_numpy() / 7.0
            if spend_week is not None
            else np.zeros(n_days)
        )
        daily_spend = np.nan_to_num(daily_spend, nan=0.0)
        media_multiplier = 1.0 + row.param_marketing_elasticity * np.log1p(
            daily_spend * 7.0 / max(row.param_media_reference, 1.0)
        )

        # Price. The realised discount drives volume through the SKU's own
        # elasticity; the discount depth itself is written to the extract.
        discount = promo_discount * participates
        price_multiplier = np.power(np.clip(1.0 - discount, 0.05, 1.0), -row.param_price_elasticity)

        # Competitor price index and weather anomaly for this SKU's category.
        conditions = conditions_by_category[row.category]
        competitor = conditions["competitor"].reindex(week_of_day).to_numpy()
        weather = conditions["weather"].reindex(week_of_day).to_numpy()
        competitor = np.nan_to_num(competitor, nan=1.0)
        weather = np.nan_to_num(weather, nan=0.0)
        competitor_multiplier = np.power(
            np.clip(competitor, 0.5, 2.0), row.param_competitor_sensitivity
        )
        weather_multiplier = np.exp(row.param_weather_sensitivity * weather * 0.12)

        mean_demand = (
            row.param_base_demand
            * seasonal
            * trend
            * dow
            * promo_multiplier
            * holiday_multiplier
            * ramp
            * media_multiplier
            * price_multiplier
            * competitor_multiplier
            * weather_multiplier
        )
        mean_demand = np.clip(mean_demand, 0.0, None)

        # --- Stochastic layer: Gamma-Poisson mixture => negative binomial --- #
        # Only what the drivers above do NOT explain.
        shape = row.param_dispersion
        gamma_noise = rng.gamma(shape=shape, scale=1.0 / shape, size=n_days)
        units = rng.poisson(mean_demand * gamma_noise)

        # Intermittency: slow movers simply have no order on many days.
        if row.param_intermittency < 1.0:
            units = np.where(rng.random(n_days) < row.param_intermittency, units, 0)

        units = np.where(live, units, 0).astype(np.int64)

        price = np.round(row.list_price * (1.0 - discount), 2)
        revenue = np.round(units * price, 2)

        # Site traffic. Driven by the same underlying intent as demand plus its
        # own measurement noise, so it is a genuine leading indicator rather
        # than a restatement of the answer.
        sessions = rng.poisson(np.clip(mean_demand * rng.uniform(11.0, 26.0), 0.0, None) + 3.0)
        sessions = np.where(live, sessions, 0)
        add_to_cart = rng.binomial(sessions, 0.11)

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "sku_id": row.sku_id,
                    "units_sold": units,
                    "revenue": revenue,
                    "unit_price": price,
                    "promo_flag": promo_active,
                    "discount_pct": np.round(discount, 4),
                    "_sessions": sessions,
                    "_add_to_cart": add_to_cart,
                    "_live": live,
                }
            )
        )

    sales = pd.concat(frames, ignore_index=True)

    # Web analytics is its own export from the storefront platform, complete for
    # every day - a session happens whether or not anything is bought.
    web_analytics = sales.loc[
        sales["_live"].to_numpy(), ["date", "sku_id", "_sessions", "_add_to_cart"]
    ].rename(columns={"_sessions": "sessions", "_add_to_cart": "add_to_cart"})
    sales = sales.drop(columns=["_sessions", "_add_to_cart"])

    # A real extract only contains rows where something happened. Most zero-sale
    # days never make it into the export, which is why the pipeline has to
    # rebuild a complete SKU x date grid rather than trusting what it receives.
    zero_rows = (sales["units_sold"] == 0).to_numpy()
    keep_zero = rng.random(len(sales)) < 0.10
    keep = (~zero_rows) | (zero_rows & keep_zero)
    sales = sales.loc[keep & sales["_live"].to_numpy()].drop(columns=["_live"])

    return (
        sales.sort_values(["sku_id", "date"], kind="stable").reset_index(drop=True),
        web_analytics.sort_values(["sku_id", "date"], kind="stable").reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def _simulate_inventory(
    rng: np.random.Generator,
    sku_master: pd.DataFrame,
    sales: pd.DataFrame,
    history_start: dt.date,
    history_end: dt.date,
) -> pd.DataFrame:
    """Simulate a weekly-review (s, S) inventory policy driven by realised demand.

    Roughly a quarter of SKUs get a deliberately mis-set policy - over-ordering
    or under-ordering - which is precisely the client's stated problem.
    """
    full_dates = pd.date_range(start=history_start, end=history_end, freq="D")
    # Precomputed once: touching Timestamp.dayofweek inside the per-SKU day loop
    # would cost ~270k attribute lookups for no reason.
    day_of_week = full_dates.dayofweek.to_numpy()
    n_days = len(full_dates)

    # Dense demand matrix: SKUs x days, zero-filled.
    demand_lookup = (
        sales.pivot_table(
            index="sku_id", columns="date", values="units_sold", aggfunc="sum", fill_value=0
        )
        .reindex(columns=full_dates, fill_value=0)
        .astype("int64")
    )

    records: list[dict[str, object]] = []

    for row in sku_master.itertuples(index=False):
        if row.sku_id not in demand_lookup.index:
            continue
        demand = demand_lookup.loc[row.sku_id].to_numpy()

        lead_time = int(np.clip(rng.integers(7, 46), 1, 120))

        # Three views of the same SKU's demand. Which one a planner anchors on
        # is precisely what separates a healthy SKU from a costly one.
        trailing = float(demand[-180:].mean()) if n_days >= 180 else float(demand.mean())
        lagged = float(demand[-545:-365].mean()) if n_days >= 545 else trailing
        if n_days >= 60:
            cumulative = np.concatenate([[0.0], np.cumsum(demand, dtype=float)])
            rolling_60 = (cumulative[60:] - cumulative[:-60]) / 60.0
            peak = float(rolling_60.max())
        else:
            peak = trailing

        # NorthBay plans on gut feel, so a third of the catalogue is anchored to
        # the wrong number - which is the whole reason this engagement exists.
        draw = rng.random()
        if draw < 0.20:
            # Buys to the festive peak and holds that level all year round.
            policy_daily = max(trailing, peak * float(rng.uniform(0.75, 1.00)))
            policy_bias = float(rng.uniform(1.60, 2.60))
        elif draw < 0.33:
            # Chronically under-orders and lives in firefighting mode.
            policy_daily = trailing
            policy_bias = float(rng.uniform(0.30, 0.60))
        else:
            policy_daily = trailing
            policy_bias = float(rng.uniform(0.85, 1.35))

        # Nobody re-baselined the declining lines: they are still being bought
        # at last year's run rate. This is how dead stock actually accumulates.
        if lagged > 0.0 and trailing < 0.55 * lagged:
            policy_daily = max(policy_daily, lagged)

        policy_daily = max(policy_daily, 0.05)

        safety_days = float(rng.uniform(7.0, 21.0))
        reorder_point = max(1.0, policy_daily * (lead_time + safety_days) * policy_bias)
        order_up_to = reorder_point + policy_daily * 28.0 * policy_bias

        on_hand = float(order_up_to * rng.uniform(0.55, 1.05))
        pending: list[tuple[int, float]] = []  # (arrival_day_index, quantity)

        for day in range(n_days):
            # Receive anything that has landed.
            arrived = [quantity for arrival, quantity in pending if arrival == day]
            if arrived:
                on_hand += float(sum(arrived))
                pending = [entry for entry in pending if entry[0] != day]

            # Sell what we can; unmet demand is lost, not backordered.
            on_hand = max(0.0, on_hand - float(demand[day]))

            on_order = float(sum(quantity for _, quantity in pending))

            # Weekly review and snapshot, every Monday - the client exports weekly.
            if day_of_week[day] == 0:
                if (on_hand + on_order) <= reorder_point:
                    quantity = max(0.0, order_up_to - (on_hand + on_order))
                    if quantity > 0:
                        jitter = int(rng.integers(-3, 8))  # suppliers are not punctual
                        arrival = min(n_days - 1, day + max(1, lead_time + jitter))
                        pending.append((arrival, round(quantity)))
                        on_order += quantity

                records.append(
                    {
                        "date": full_dates[day],
                        "sku_id": row.sku_id,
                        "on_hand_units": int(round(on_hand)),
                        "on_order_units": int(round(on_order)),
                        "lead_time_days": lead_time,
                        "reorder_point": int(round(reorder_point)),
                    }
                )

    inventory = pd.DataFrame.from_records(records)
    return inventory.sort_values(["sku_id", "date"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Export artefacts
# --------------------------------------------------------------------------- #
def _apply_export_artefacts(
    rng: np.random.Generator,
    sales: pd.DataFrame,
    sku_master: pd.DataFrame,
    calendar: pd.DataFrame,
    inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Reproduce the imperfections of the client's reporting stack.

    NorthBay's tables are assembled from several system exports of different
    vintages, so the delivered files carry partial rows, mixed type encodings,
    free-text labels typed by hand, and values that are physically impossible.
    Modelling those faithfully is what makes the cleaning layer worth testing.
    """
    counts: dict[str, int] = {}
    sales = sales.copy()
    sku_master = sku_master.copy()
    calendar = calendar.copy()
    inventory = inventory.copy()

    # Widen the columns that are about to receive nulls. Under pandas 3 an
    # integer column will not silently absorb NaN, so the cast has to be
    # explicit - which is also what a real CSV export would produce anyway.
    inventory["on_hand_units"] = inventory["on_hand_units"].astype("float64")
    sku_master["category"] = sku_master["category"].astype("object")

    # --- sales_daily ------------------------------------------------------- #
    n_sales = len(sales)

    # 1. Exact duplicate rows (a re-run of the export appended twice).
    n_dupes = int(n_sales * EXPORT_ARTEFACT_RATES["duplicate_sales_rows"])
    dupe_positions = rng.choice(n_sales, size=n_dupes, replace=False)
    duplicates = sales.iloc[dupe_positions].copy()
    counts["duplicate_sales_rows"] = n_dupes

    # 2. Missing revenue - recoverable as units x price.
    n_missing_revenue = int(n_sales * EXPORT_ARTEFACT_RATES["missing_revenue"])
    revenue_positions = rng.choice(n_sales, size=n_missing_revenue, replace=False)
    sales.iloc[revenue_positions, sales.columns.get_loc("revenue")] = np.nan
    counts["missing_revenue"] = n_missing_revenue

    # 3. Missing unit_price - recoverable as revenue / units, else list price.
    n_missing_price = int(n_sales * EXPORT_ARTEFACT_RATES["missing_unit_price"])
    price_positions = rng.choice(n_sales, size=n_missing_price, replace=False)
    sales.iloc[price_positions, sales.columns.get_loc("unit_price")] = np.nan
    counts["missing_unit_price"] = n_missing_price

    # 4. Negative units - refunds mis-booked against the sales ledger.
    n_negative = int(n_sales * EXPORT_ARTEFACT_RATES["negative_units"])
    negative_positions = rng.choice(n_sales, size=n_negative, replace=False)
    column = sales.columns.get_loc("units_sold")
    sales.iloc[negative_positions, column] = (
        -np.abs(sales.iloc[negative_positions, column].to_numpy()) - 1
    )
    counts["negative_units"] = n_negative

    # 5. promo_flag arrives as a mix of 0/1, "Y"/"N", and "TRUE"/"FALSE"
    #    because three different exports were stitched together.
    promo_raw = sales["promo_flag"].astype("int64").to_numpy()
    style = rng.integers(0, 3, size=n_sales)
    promo_mixed = np.where(
        style == 0,
        promo_raw.astype(str),
        np.where(
            style == 1,
            np.where(promo_raw == 1, "Y", "N"),
            np.where(promo_raw == 1, "TRUE", "FALSE"),
        ),
    )
    sales["promo_flag"] = promo_mixed
    counts["mixed_type_promo_flag"] = n_sales

    # 6. sku_id whitespace / case damage.
    n_dirty_ids = int(n_sales * EXPORT_ARTEFACT_RATES["sku_id_whitespace_case"])
    id_positions = rng.choice(n_sales, size=n_dirty_ids, replace=False)
    id_column = sales.columns.get_loc("sku_id")
    dirty = sales.iloc[id_positions, id_column].astype(str)
    variants = rng.integers(0, 3, size=n_dirty_ids)
    sales.iloc[id_positions, id_column] = np.where(
        variants == 0, " " + dirty, np.where(variants == 1, dirty + " ", dirty.str.lower())
    )
    counts["sku_id_whitespace_case"] = n_dirty_ids

    # 7. Orphan rows: SKUs that sold but never made it into the product master.
    n_orphan = max(3, int(n_sales * EXPORT_ARTEFACT_RATES["orphan_sku_rows"]))
    orphan_ids = [f"NBL-XXX-{index:03d}" for index in range(1, 4)]
    orphan_source = sales.sample(n=n_orphan, random_state=int(rng.integers(0, 2**31)))
    orphan_rows = orphan_source.copy()
    orphan_rows["sku_id"] = rng.choice(orphan_ids, size=len(orphan_rows))
    counts["orphan_sku_rows"] = len(orphan_rows)

    sales = pd.concat([sales, duplicates, orphan_rows], ignore_index=True)
    # The export is not ordered, so appended rows are not clustered at the tail.
    sales = sales.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

    # --- sku_master -------------------------------------------------------- #
    # Categories are free text maintained by hand across several merchandisers,
    # so case, spacing, accents and British/American spellings all vary.
    label_variants: dict[str, list[str]] = {
        "Bedding & Bath": [
            "Bedding & Bath",
            "bedding & bath",
            "BEDDING & BATH",
            " Bedding & Bath ",
        ],
        "Lighting": ["Lighting", "lighting", "LIGHTING", "Lighting "],
        "Kitchen & Dining": [
            "Kitchen & Dining",
            "kitchen & dining",
            "Kitchen and Dining",
            "KITCHEN & DINING",
        ],
        "Decor": ["Decor", "decor", "Décor", "DECOR", " Decor"],
        "Storage & Organisation": [
            "Storage & Organisation",
            "storage & organisation",
            "Storage & Organization",
            "STORAGE & ORGANISATION",
        ],
        "Small Appliances": [
            "Small Appliances",
            "small appliances",
            "SMALL APPLIANCES",
            "Small appliances",
        ],
    }
    scrambled = [str(rng.choice(label_variants[category])) for category in sku_master["category"]]
    sku_master["category"] = pd.Series(scrambled, index=sku_master.index, dtype="object")
    counts["inconsistent_category_labels"] = len(sku_master)

    # A handful of products have no category at all.
    n_missing_category = max(2, int(len(sku_master) * EXPORT_ARTEFACT_RATES["missing_category"]))
    category_positions = rng.choice(len(sku_master), size=n_missing_category, replace=False)
    sku_master.iloc[category_positions, sku_master.columns.get_loc("category")] = np.nan
    counts["missing_category"] = n_missing_category

    # launch_date exported as a mix of ISO and DD-MM-YYYY strings.
    launch = pd.to_datetime(sku_master["launch_date"])
    iso_style = rng.random(len(sku_master)) < 0.75
    sku_master["launch_date"] = np.where(
        iso_style, launch.dt.strftime("%Y-%m-%d"), launch.dt.strftime("%d-%m-%Y")
    )
    counts["mixed_date_formats"] = int((~iso_style).sum())

    sku_master = sku_master.drop(columns=list(HIDDEN_PARAM_COLUMNS))

    # --- inventory_snapshots ----------------------------------------------- #
    n_inventory = len(inventory)

    n_missing_on_hand = int(n_inventory * EXPORT_ARTEFACT_RATES["missing_on_hand"])
    on_hand_positions = rng.choice(n_inventory, size=n_missing_on_hand, replace=False)
    inventory.iloc[on_hand_positions, inventory.columns.get_loc("on_hand_units")] = np.nan
    counts["missing_on_hand"] = n_missing_on_hand

    # Impossible lead times: 0 (never captured) or absurdly long (typo).
    n_bad_lead = int(n_inventory * EXPORT_ARTEFACT_RATES["bad_lead_time"])
    lead_positions = rng.choice(n_inventory, size=n_bad_lead, replace=False)
    lead_column = inventory.columns.get_loc("lead_time_days")
    inventory.iloc[lead_positions, lead_column] = np.where(
        rng.random(n_bad_lead) < 0.6, 0, rng.integers(400, 999, size=n_bad_lead)
    )
    counts["bad_lead_time"] = n_bad_lead

    # --- calendar ---------------------------------------------------------- #
    # The planned discount is published with the promo calendar, weeks ahead of
    # the event, so it is kept as a real column. It is what makes forward
    # promotional depth a legitimate forecast input rather than hindsight.
    calendar = calendar.rename(columns={"_promo_discount": "planned_discount"})
    calendar = calendar.drop(columns=["_promo_lift"])
    # is_holiday exported as Yes/No strings.
    calendar["is_holiday"] = np.where(calendar["is_holiday"] == 1, "Yes", "No")
    counts["holiday_as_yes_no"] = len(calendar)

    return sales, sku_master, calendar, inventory, counts


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_extracts(
    settings: Settings | None = None,
) -> tuple[dict[str, pd.DataFrame], BuildSummary]:
    """Build the four extract tables in memory.

    Returns:
        A mapping of table name to dataframe, and a summary of the build.
    """
    settings = settings or get_settings()
    rng = np.random.default_rng(settings.random_seed)

    log.info(
        "building extracts",
        extra={"context": {"seed": settings.random_seed, "n_skus": settings.n_skus}},
    )

    calendar_end = settings.history_end + dt.timedelta(days=CALENDAR_FORWARD_DAYS)
    calendar = _build_calendar(settings.history_start, calendar_end)
    sku_master = _build_sku_master(rng, settings.n_skus, settings.history_start)

    # Driver tables first: demand is generated *from* them, so they have to
    # exist before the sales ledger does.
    marketing = _build_marketing_plan(
        rng,
        sku_master,
        calendar,
        settings.history_start,
        settings.history_end,
        CALENDAR_FORWARD_DAYS,
    )
    conditions = _build_market_conditions(
        rng, calendar, settings.history_start, settings.history_end, CALENDAR_FORWARD_DAYS
    )

    sales, web_analytics = _simulate_demand(
        rng,
        sku_master,
        calendar,
        marketing,
        conditions,
        settings.history_start,
        settings.history_end,
    )
    inventory = _simulate_inventory(
        rng, sku_master, sales, settings.history_start, settings.history_end
    )

    sales, sku_master, calendar, inventory, artefacts = _apply_export_artefacts(
        rng, sales, sku_master, calendar, inventory
    )

    tables = {
        "sales_daily": sales,
        "sku_master": sku_master,
        "calendar": calendar,
        "inventory_snapshots": inventory,
        "marketing_spend": marketing,
        "web_analytics": web_analytics,
        "market_conditions": conditions,
    }

    summary = BuildSummary(
        seed=settings.random_seed,
        n_skus=settings.n_skus,
        history_start=settings.history_start.isoformat(),
        history_end=settings.history_end.isoformat(),
        row_counts={name: len(frame) for name, frame in tables.items()},
    )

    log.info("extracts built", extra={"context": {"rows": summary.row_counts}})
    log.debug("export artefact counts", extra={"context": artefacts})
    return tables, summary


def write_extracts(settings: Settings | None = None) -> BuildSummary:
    """Build the extracts and write them to ``data/raw/`` as CSV."""
    settings = settings or get_settings()
    settings.ensure_directories()

    tables, summary = build_extracts(settings)

    for name, frame in tables.items():
        destination = settings.raw_dir / f"{name}.csv"
        frame.to_csv(destination, index=False)
        log.info(
            "wrote extract",
            extra={"context": {"table": name, "path": str(destination), "rows": len(frame)}},
        )

    return summary
