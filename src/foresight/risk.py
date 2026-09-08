"""Stockout / overstock risk scoring and decisioning (deliverable D4).

A forecast does not tell the ops team what to do. This layer turns the forecast
plus the current stock position into one recommended action per SKU, with the
rupee value at stake attached so the team can work the list in priority order.

Everything here is deliberately transparent arithmetic rather than a second
model. Acceptance criterion D4.3 requires the logic be explainable, and an ops
manager has to be able to challenge a reorder recommendation and get a straight
answer. Every intermediate quantity is kept on the output row for exactly that
reason.

HOW STOCKOUT RISK IS SCORED
---------------------------
Over the replenishment lead time, demand is treated as Normal with mean
``mu_LT`` (the summed forecast) and standard deviation ``sigma_LT`` recovered
from the forecast's own prediction interval. Against available stock ``A``::

    stockout_score = P(demand over lead time > A)

That is a genuine probability, not an index, which is what makes it comparable
across SKUs of wildly different volume and what lets it be read against the
configured service level.

HOW OVERSTOCK RISK IS SCORED
----------------------------
Symmetrically, and as a probability on the opposite tail. Over the cover window
the client considers excessive (12 weeks by default), demand is Normal with mean
``mu_C`` and standard deviation ``sigma_C``::

    overstock_score = P(demand over the cover window < A)

In plain language: *the chance that twelve weeks of trading will not clear the
stock currently held*. A SKU with no forward demand scores 1.0 - dead stock, and
it belongs at the top of the markdown list.

WHY BOTH AXES ARE PROBABILITIES ON OPPOSITE TAILS
-------------------------------------------------
This is the design decision that makes the decisioning grid work.

Score overstock on weeks of cover from the *point* forecast, and the two risks
become arithmetically mutually exclusive: plenty of stock forces the stockout
probability to zero, so "high on both" can never occur and the brief's fourth
quadrant - Watch / volatile - is unreachable by construction. An empty quadrant
is not a finding about the assortment; it is a bug in the scoring.

Framing both as probabilities fixes it, because they then respond to the
forecast's *spread* as well as its level:

* **Stockout**  ``P(D_leadtime > A)``  - do we run out before the next delivery?
* **Overstock** ``P(D_cover    < A)``  - will three months of sales fail to clear it?

For a SKU with a tight forecast these are near-complementary: high stock makes
the first ~0 and the second ~1. But when the forecast is genuinely uncertain,
both tails carry real mass, and the same SKU can be materially exposed in both
directions at once. That is exactly the erratic line a human should review, and
it is now reachable rather than definitionally impossible.

Note that ``sigma_C`` is extrapolated: the model forecasts 8 weeks and the cover
window is 12, so per-week uncertainty is scaled by ``sqrt(12)``. That assumes
weekly errors are independent, which slightly understates the spread. It is
stated in the readout rather than buried.

THE RUPEE FIGURES USE THE CENTRAL FORECAST
------------------------------------------
Throughout. The scores decide *what to look at*; the money has to be a best
estimate rather than a worst case, or every total reported to Finance would be
systematically alarmist.

THE INDEPENDENCE ASSUMPTION
---------------------------
``sigma_LT`` combines weekly forecast errors in quadrature, which assumes those
errors are independent week to week. They are not perfectly: a SKU the model is
misreading tends to be misread for several weeks running. This understates the
true variance somewhat, making stockout probabilities mildly conservative. It is
stated in the executive readout as a known limitation rather than buried.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd
from scipy import stats

from foresight.config import Settings, get_settings
from foresight.exceptions import RiskScoringError
from foresight.logging_setup import get_logger
from foresight.schemas import (
    RISK_ACTION_LABELS,
    RISK_ACTION_RATIONALE,
    RiskAction,
    RiskLevel,
)

__all__ = [
    "aggregate_impact",
    "latest_inventory_position",
    "score_risk",
]

log = get_logger(__name__)

DAYS_PER_WEEK: Final[float] = 7.0
_EPSILON: Final[float] = 1e-9

#: A SKU younger than this has too little history for its forecast to be trusted.
LOW_CONFIDENCE_WEEKS: Final[int] = 26
#: Coefficient of variation above which a forecast is called low confidence.
LOW_CONFIDENCE_CV: Final[float] = 1.0

#: Forecast coefficient of variation above which demand is treated as erratic.
#: Set from the observed distribution: the median SKU sits near 0.43 and the 95th
#: percentile near 0.70, so 0.75 isolates the genuinely unpredictable tail rather
#: than flagging ordinary noise.
VOLATILE_FORECAST_CV: Final[float] = 0.75

#: Below this weekly run rate a SKU is a dead line, not a volatile one. Without
#: this guard the coefficient of variation explodes on near-zero demand and every
#: dead SKU would be misrouted to manual review instead of to the markdown list.
MIN_WEEKLY_DEMAND_FOR_VOLATILITY: Final[float] = 0.5


def latest_inventory_position(inventory: pd.DataFrame) -> pd.DataFrame:
    """Take the most recent snapshot per SKU - the position we act from."""
    if inventory.empty:
        raise RiskScoringError("Inventory table is empty; cannot score risk.")

    latest = (
        inventory.sort_values(["sku_id", "date"])
        .groupby("sku_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return latest.rename(columns={"date": "inventory_as_of"})


def _normal_loss(mean: np.ndarray, sigma: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Expected unmet demand ``E[max(0, D - A)]`` for ``D ~ Normal(mean, sigma)``.

    The standard first-order loss function::

        E[(D - A)+] = sigma * (k * Phi(k) + phi(k)),  k = (mean - A) / sigma

    This is what converts a stockout *probability* into an expected number of
    lost units, which is what the rupee figure needs.
    """
    safe_sigma = np.where(sigma > _EPSILON, sigma, _EPSILON)
    k = (mean - available) / safe_sigma
    loss = safe_sigma * (k * stats.norm.cdf(k) + stats.norm.pdf(k))
    # With no uncertainty the loss collapses to the deterministic shortfall.
    deterministic = np.clip(mean - available, 0.0, None)
    return np.where(sigma > _EPSILON, np.clip(loss, 0.0, None), deterministic)


def _risk_level(score: np.ndarray) -> np.ndarray:
    """Bucket a 0-1 score into a severity band for display."""
    return np.select(
        [score >= 0.80, score >= 0.50, score >= 0.25, score >= 0.05],
        [
            RiskLevel.CRITICAL.value,
            RiskLevel.HIGH.value,
            RiskLevel.MEDIUM.value,
            RiskLevel.LOW.value,
        ],
        default=RiskLevel.NONE.value,
    )


def score_risk(
    forecast: pd.DataFrame,
    inventory: pd.DataFrame,
    sku_master: pd.DataFrame,
    settings: Settings | None = None,
    *,
    panel_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score stockout and overstock risk for every SKU and recommend an action.

    Args:
        forecast: One row per (sku_id, horizon) with ``prediction``,
            ``prediction_lower``, ``prediction_upper`` and ``target_week``.
        inventory: Cleaned inventory snapshots (all history; the latest per SKU
            is selected here).
        sku_master: Cleaned product master, for cost and price.
        settings: Configuration.
        panel_features: Optional origin-side features (``weeks_since_launch``,
            ``cv_13``) used to flag low-confidence forecasts.

    Returns:
        One row per SKU carrying both scores, the quadrant, the recommended
        action, the rupee value at stake and every intermediate quantity.
    """
    settings = settings or get_settings()
    if forecast.empty:
        raise RiskScoringError("Forecast table is empty; cannot score risk.")

    required = {"sku_id", "horizon", "prediction", "prediction_lower", "prediction_upper"}
    missing = required - set(forecast.columns)
    if missing:
        raise RiskScoringError(f"Forecast is missing required column(s): {sorted(missing)}")

    position = latest_inventory_position(inventory)
    horizon_weeks = int(forecast["horizon"].max())

    # --- Per-week forecast sigma from the prediction interval --------------- #
    # An interval of coverage c spans z = Phi^-1((1+c)/2) standard deviations
    # either side of centre, so dividing the width by 2z recovers sigma.
    coverage_z = float(stats.norm.ppf(0.5 * (1.0 + settings.interval_coverage)))
    work = forecast.sort_values(["sku_id", "horizon"]).copy()
    work["week_sigma"] = (work["prediction_upper"] - work["prediction_lower"]).clip(lower=0.0) / (
        2.0 * coverage_z
    )

    # --- Lead-time window --------------------------------------------------- #
    lead_time_weeks = (position["lead_time_days"] / DAYS_PER_WEEK).clip(lower=0.1)
    lead_weeks_by_sku = dict(zip(position["sku_id"], lead_time_weeks, strict=True))

    # Weight each forecast week by how much of it falls inside the lead time,
    # so a 10-day lead time counts week 1 fully and 3/7 of week 2 - rather than
    # rounding to a whole week and misstating the exposure by up to 40%.
    work["_lead_weeks"] = work["sku_id"].map(lead_weeks_by_sku).astype("float64")
    work["_weight"] = (work["_lead_weeks"] - (work["horizon"] - 1)).clip(0.0, 1.0)

    lead_time_demand = work.groupby("sku_id").apply(
        lambda group: float((group["prediction"] * group["_weight"]).sum()),
        include_groups=False,
    )
    # Variances add; standard deviations do not. Weighted by the same fraction.
    lead_time_variance = work.groupby("sku_id").apply(
        lambda group: float(((group["week_sigma"] * group["_weight"]) ** 2).sum()),
        include_groups=False,
    )
    horizon_demand = work.groupby("sku_id")["prediction"].sum()
    # Average per-week forecast uncertainty, used to extrapolate the spread of
    # demand over the cover window beyond the modelled horizon.
    weekly_sigma = work.groupby("sku_id")["week_sigma"].mean()

    summary = pd.DataFrame(
        {
            "lead_time_demand": lead_time_demand,
            "lead_time_sigma": np.sqrt(lead_time_variance),
            "horizon_demand": horizon_demand,
            "weekly_sigma": weekly_sigma,
        }
    ).reset_index()

    # --- Assemble --------------------------------------------------------- #
    frame = (
        summary.merge(position, on="sku_id", how="inner", validate="one_to_one")
        .merge(
            sku_master[["sku_id", "category", "subcategory", "unit_cost", "list_price"]],
            on="sku_id",
            how="left",
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    if frame.empty:
        raise RiskScoringError(
            "No SKU has both a forecast and an inventory position; cannot score risk."
        )

    frame["lead_time_weeks"] = frame["lead_time_days"] / DAYS_PER_WEEK
    frame["available_units"] = frame["on_hand_units"] + frame["on_order_units"]
    frame["mean_weekly_forecast"] = frame["horizon_demand"] / float(horizon_weeks)

    available = frame["available_units"].to_numpy(dtype="float64")
    mu_lead = frame["lead_time_demand"].to_numpy(dtype="float64")
    sigma_lead = frame["lead_time_sigma"].to_numpy(dtype="float64")

    # --- Stockout ----------------------------------------------------------- #
    safe_sigma = np.where(sigma_lead > _EPSILON, sigma_lead, _EPSILON)
    stockout_score = np.where(
        sigma_lead > _EPSILON,
        1.0 - stats.norm.cdf((available - mu_lead) / safe_sigma),
        (mu_lead > available).astype("float64"),
    )
    frame["stockout_score"] = np.clip(stockout_score, 0.0, 1.0)

    service_z = float(stats.norm.ppf(settings.service_level))
    frame["safety_stock_units"] = np.clip(service_z * sigma_lead, 0.0, None)
    frame["target_stock_units"] = mu_lead + frame["safety_stock_units"]
    frame["recommended_order_units"] = np.ceil(
        np.clip(frame["target_stock_units"] - available, 0.0, None)
    )
    frame["expected_lost_units"] = _normal_loss(mu_lead, sigma_lead, available)

    # --- Overstock ---------------------------------------------------------- #
    threshold = float(settings.overstock_cover_weeks)
    weekly = frame["mean_weekly_forecast"].to_numpy(dtype="float64")

    # Headline weeks of cover, on the central forecast. This is the number the
    # ops team reads; the score below is what drives the decision.
    has_demand = weekly > _EPSILON
    cover_weeks = np.where(has_demand, available / np.where(has_demand, weekly, 1.0), np.inf)
    cover_weeks = np.where(available <= _EPSILON, 0.0, cover_weeks)
    frame["cover_weeks"] = np.where(np.isfinite(cover_weeks), cover_weeks, 999.0)

    # Demand over the cover window. Extrapolated from the modelled horizon:
    # the mean scales linearly with weeks, the standard deviation with sqrt(weeks)
    # under the same independence assumption used for the lead time.
    cover_demand = weekly * threshold
    cover_sigma = frame["weekly_sigma"].to_numpy(dtype="float64") * math.sqrt(threshold)
    frame["cover_window_demand"] = cover_demand

    safe_cover_sigma = np.where(cover_sigma > _EPSILON, cover_sigma, _EPSILON)
    overstock_score = np.where(
        cover_sigma > _EPSILON,
        stats.norm.cdf((available - cover_demand) / safe_cover_sigma),
        (available > cover_demand).astype("float64"),
    )
    # Holding nothing cannot be overstocked, whatever the arithmetic says.
    frame["overstock_score"] = np.clip(
        np.where(available <= _EPSILON, 0.0, overstock_score), 0.0, 1.0
    )

    # Capital locked stays on the CENTRAL forecast: this figure is reported to
    # Finance, and pricing it off a worst case would overstate the exposure.
    frame["excess_units"] = np.ceil(np.clip(available - cover_demand, 0.0, None))

    # --- Rupee exposure ----------------------------------------------------- #
    frame["unit_margin"] = (frame["list_price"] - frame["unit_cost"]).clip(lower=0.0)
    frame["revenue_at_risk"] = (frame["expected_lost_units"] * frame["list_price"]).round(2)
    frame["margin_at_risk"] = (frame["expected_lost_units"] * frame["unit_margin"]).round(2)
    frame["locked_capital"] = (frame["excess_units"] * frame["unit_cost"]).round(2)
    frame["value_at_stake"] = frame[["revenue_at_risk", "locked_capital"]].max(axis=1).round(2)

    # --- Forecast volatility ------------------------------------------------ #
    # See the module docstring: a stock position cannot be simultaneously short
    # and long, so "high on both scores" is unreachable. What the brief's fourth
    # quadrant actually describes is a SKU whose demand is too erratic to act on,
    # and that is measured here from the forecast's own dispersion.
    weekly_sigma_values = frame["weekly_sigma"].to_numpy(dtype="float64")
    tradeable = weekly > MIN_WEEKLY_DEMAND_FOR_VOLATILITY
    frame["forecast_cv"] = np.where(
        tradeable, weekly_sigma_values / np.where(tradeable, weekly, 1.0), 0.0
    )
    frame["volatility_score"] = np.clip(frame["forecast_cv"] / VOLATILE_FORECAST_CV, 0.0, 1.0)

    # --- Quadrant ----------------------------------------------------------- #
    high = settings.risk_high_threshold
    elevated = high / 2.0
    stockout_high = frame["stockout_score"] >= high
    overstock_high = frame["overstock_score"] >= high

    # Erratic *and* exposed. Volatility alone is not actionable - a SKU nobody is
    # over- or under-stocked on needs no review however noisy it is.
    volatile = (frame["forecast_cv"] >= VOLATILE_FORECAST_CV) & (
        (frame["stockout_score"] >= elevated) | (frame["overstock_score"] >= elevated)
    )

    # Volatility is evaluated FIRST and overrides the other two. Issuing "order
    # exactly 340 units" against a forecast this uncertain is false precision;
    # the honest recommendation is that a human looks at it.
    frame["action"] = np.select(
        [
            volatile,
            stockout_high & ~overstock_high,
            overstock_high & ~stockout_high,
            stockout_high & overstock_high,
        ],
        [
            RiskAction.WATCH_VOLATILE.value,
            RiskAction.REORDER_NOW.value,
            RiskAction.MARKDOWN_CLEAR.value,
            RiskAction.WATCH_VOLATILE.value,
        ],
        default=RiskAction.HEALTHY.value,
    )
    frame["action_label"] = frame["action"].map(
        {action.value: label for action, label in RISK_ACTION_LABELS.items()}
    )
    frame["action_rationale"] = frame["action"].map(
        {action.value: text for action, text in RISK_ACTION_RATIONALE.items()}
    )
    frame["stockout_level"] = _risk_level(frame["stockout_score"].to_numpy())
    frame["overstock_level"] = _risk_level(frame["overstock_score"].to_numpy())

    # --- Forecast confidence (brief section 16.2) --------------------------- #
    frame["forecast_confidence"] = "high"
    if panel_features is not None and not panel_features.empty:
        confidence_source = panel_features[
            ["sku_id", "weeks_since_launch", "cv_13"]
        ].drop_duplicates(subset=["sku_id"], keep="last")
        frame = frame.merge(confidence_source, on="sku_id", how="left")
        thin_history = frame["weeks_since_launch"].fillna(999) < LOW_CONFIDENCE_WEEKS
        erratic = frame["cv_13"].fillna(0.0) >= LOW_CONFIDENCE_CV
        frame["forecast_confidence"] = np.select(
            [thin_history, erratic], ["low", "medium"], default="high"
        )

    frame["horizon_weeks"] = horizon_weeks
    frame["service_level"] = settings.service_level
    frame["overstock_threshold_weeks"] = threshold

    frame = frame.sort_values("value_at_stake", ascending=False).reset_index(drop=True)
    frame["priority_rank"] = np.arange(1, len(frame) + 1)

    log.info(
        "risk scored",
        extra={
            "context": {
                "skus": len(frame),
                "reorder_now": int((frame["action"] == RiskAction.REORDER_NOW.value).sum()),
                "markdown": int((frame["action"] == RiskAction.MARKDOWN_CLEAR.value).sum()),
                "watch": int((frame["action"] == RiskAction.WATCH_VOLATILE.value).sum()),
                "healthy": int((frame["action"] == RiskAction.HEALTHY.value).sum()),
            }
        },
    )
    return frame


def aggregate_impact(risk_table: pd.DataFrame) -> dict[str, object]:
    """Roll the SKU-level risk table up into the numbers Finance cares about.

    Objective 3 of the brief asks for the business impact in rupees. Revenue at
    risk and locked capital are reported separately and never added together:
    one is revenue that may not happen, the other is cash already spent. Summing
    them would produce a headline number that means nothing.
    """
    if risk_table.empty:
        raise RiskScoringError("Risk table is empty; nothing to aggregate.")

    by_action = (
        risk_table.groupby("action")
        .agg(
            skus=("sku_id", "count"),
            revenue_at_risk=("revenue_at_risk", "sum"),
            margin_at_risk=("margin_at_risk", "sum"),
            locked_capital=("locked_capital", "sum"),
            expected_lost_units=("expected_lost_units", "sum"),
            excess_units=("excess_units", "sum"),
        )
        .reset_index()
    )

    reorder = risk_table[risk_table["action"] == RiskAction.REORDER_NOW.value]
    markdown = risk_table[risk_table["action"] == RiskAction.MARKDOWN_CLEAR.value]
    watch = risk_table[risk_table["action"] == RiskAction.WATCH_VOLATILE.value]

    return {
        "total_skus": int(len(risk_table)),
        "horizon_weeks": int(risk_table["horizon_weeks"].iloc[0]),
        "revenue_at_risk_total": float(risk_table["revenue_at_risk"].sum()),
        "margin_at_risk_total": float(risk_table["margin_at_risk"].sum()),
        "locked_capital_total": float(risk_table["locked_capital"].sum()),
        "expected_lost_units_total": float(risk_table["expected_lost_units"].sum()),
        "excess_units_total": float(risk_table["excess_units"].sum()),
        "reorder_now_skus": int(len(reorder)),
        "reorder_now_revenue_at_risk": float(reorder["revenue_at_risk"].sum()),
        "reorder_now_order_units": float(reorder["recommended_order_units"].sum()),
        "markdown_skus": int(len(markdown)),
        "markdown_locked_capital": float(markdown["locked_capital"].sum()),
        "watch_skus": int(len(watch)),
        "healthy_skus": int((risk_table["action"] == RiskAction.HEALTHY.value).sum()),
        "low_confidence_skus": int((risk_table["forecast_confidence"] == "low").sum()),
        "top_10_share_of_revenue_at_risk": (
            float(
                risk_table.nlargest(10, "revenue_at_risk")["revenue_at_risk"].sum()
                / max(risk_table["revenue_at_risk"].sum(), _EPSILON)
            )
        ),
        "by_action": by_action.to_dict(orient="records"),
    }


def format_inr(amount: float) -> str:
    """Format a rupee amount the way an Indian finance team reads it.

    Lakh and crore, not thousands and millions - the readout goes to NorthBay's
    Finance lead, and 1.2 crore communicates instantly where 12,000,000 does not.
    """
    if not math.isfinite(amount):
        return "n/a"
    sign = "-" if amount < 0 else ""
    value = abs(amount)
    if value >= 1e7:
        return f"{sign}Rs {value / 1e7:,.2f} Cr"
    if value >= 1e5:
        return f"{sign}Rs {value / 1e5:,.2f} L"
    return f"{sign}Rs {value:,.0f}"
