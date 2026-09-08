"""Cleaning for the commercial driver extracts.

Every function returns an empty frame with the correct columns when handed an
empty input, rather than raising. That is what lets the core pipeline treat the
whole extension as optional: a client who cannot supply media or analytics data
still gets a working forecast built from history alone.

Cleaning decisions are recorded in the same :class:`~foresight.clean.CleaningReport`
the core uses, so the data-quality memo covers driver data too without knowing
anything about it.
"""

from __future__ import annotations

import pandas as pd

from foresight.clean import (
    CleaningAction,
    CleaningReport,
    _to_numeric,
    normalise_category,
    normalise_sku_id,
    parse_mixed_dates,
)
from foresight.logging_setup import get_logger

__all__ = ["clean_market_conditions", "clean_marketing", "clean_web_analytics"]

log = get_logger(__name__)

MARKETING_COLUMNS = ["week_start", "sku_id", "media_spend", "impressions", "email_sends"]
WEB_COLUMNS = ["date", "sku_id", "sessions", "add_to_cart"]
CONDITION_COLUMNS = ["week_start", "category", "competitor_price_index", "weather_anomaly"]

#: A competitor index is a ratio against our own price; anything outside this
#: band is a scrape failure rather than a real market move.
COMPETITOR_INDEX_BOUNDS = (0.3, 3.0)


def _snap_to_monday(dates: pd.Series) -> pd.Series:
    """Snap any date to the Monday of its week."""
    parsed = pd.to_datetime(dates)
    return parsed - pd.to_timedelta(parsed.dt.dayofweek, unit="D")


def clean_marketing(
    raw: pd.DataFrame,
    known_skus: set[str],
    report: CleaningReport,
) -> pd.DataFrame:
    """Clean the weekly media plan."""
    if raw.empty:
        return pd.DataFrame(columns=MARKETING_COLUMNS)

    table = "marketing_spend"
    frame = raw.copy()

    frame["sku_id"] = normalise_sku_id(frame["sku_id"])
    frame["week_start"] = _snap_to_monday(parse_mixed_dates(frame["week_start"]))
    frame = frame[frame["week_start"].notna()]

    unknown = int((~frame["sku_id"].isin(known_skus)).sum())
    if unknown:
        frame = frame[frame["sku_id"].isin(known_skus)]
        report.add(
            CleaningAction(
                table=table,
                issue=f"{unknown} media rows for products absent from the master",
                detection="anti-join against sku_master.sku_id",
                rows_affected=unknown,
                resolution="dropped",
                rationale="spend against an unknown product cannot be attributed to a "
                "forecast, and would otherwise inflate the plan totals",
                severity="warning",
            )
        )

    for column in ("media_spend", "impressions", "email_sends"):
        frame[column] = _to_numeric(frame[column])

    money = frame[["media_spend", "impressions", "email_sends"]]
    negative = int((money < 0).any(axis=1).sum())
    missing = int(money.isna().any(axis=1).sum())
    if negative or missing:
        for column in ("media_spend", "impressions", "email_sends"):
            frame[column] = frame[column].fillna(0.0).clip(lower=0.0)
        report.add(
            CleaningAction(
                table=table,
                issue=f"{negative + missing} media rows with missing or negative values",
                detection="null or < 0 check on spend, impressions and sends",
                rows_affected=negative + missing,
                resolution="set to zero",
                rationale="an absent plan line means no spend was committed, which is "
                "zero rather than unknown; a negative spend is a credit note "
                "and is not media pressure",
                severity="info",
            )
        )

    duplicates = int(frame.duplicated(subset=["week_start", "sku_id"]).sum())
    if duplicates:
        frame = frame.groupby(["week_start", "sku_id"], as_index=False)[
            ["media_spend", "impressions", "email_sends"]
        ].sum()
        report.add(
            CleaningAction(
                table=table,
                issue=f"{duplicates} duplicate (week, product) media rows",
                detection="duplicate check on the declared primary key",
                rows_affected=duplicates,
                resolution="summed",
                rationale="one product can be funded from several campaigns in the same "
                "week; total pressure is the sum, not the last row loaded",
                severity="warning",
            )
        )

    for column in ("media_spend", "impressions", "email_sends"):
        frame[column] = frame[column].astype("float64")

    return frame[MARKETING_COLUMNS].sort_values(["sku_id", "week_start"]).reset_index(drop=True)


def clean_web_analytics(
    raw: pd.DataFrame,
    known_skus: set[str],
    report: CleaningReport,
) -> pd.DataFrame:
    """Clean the daily storefront analytics export."""
    if raw.empty:
        return pd.DataFrame(columns=WEB_COLUMNS)

    table = "web_analytics"
    frame = raw.copy()

    frame["sku_id"] = normalise_sku_id(frame["sku_id"])
    frame["date"] = parse_mixed_dates(frame["date"])
    frame = frame[frame["date"].notna() & frame["sku_id"].isin(known_skus)]

    for column in ("sessions", "add_to_cart"):
        frame[column] = _to_numeric(frame[column]).fillna(0.0).clip(lower=0.0)

    # A basket cannot be added more often than the page was viewed.
    impossible = int((frame["add_to_cart"] > frame["sessions"]).sum())
    if impossible:
        frame["add_to_cart"] = frame[["add_to_cart", "sessions"]].min(axis=1)
        report.add(
            CleaningAction(
                table=table,
                issue=f"{impossible} rows where add-to-cart exceeded sessions",
                detection="add_to_cart > sessions",
                rows_affected=impossible,
                resolution="capped at the session count",
                rationale="the two counters come from different systems with different "
                "sessionisation; a conversion rate above 100% would make the "
                "derived feature meaningless",
                severity="warning",
            )
        )

    duplicates = int(frame.duplicated(subset=["date", "sku_id"]).sum())
    if duplicates:
        frame = frame.groupby(["date", "sku_id"], as_index=False)[["sessions", "add_to_cart"]].sum()
        report.add(
            CleaningAction(
                table=table,
                issue=f"{duplicates} duplicate (date, product) analytics rows",
                detection="duplicate check on the declared primary key",
                rows_affected=duplicates,
                resolution="summed",
                rationale="analytics exports are often split by device or channel; the "
                "day's traffic is the total across them",
                severity="info",
            )
        )

    for column in ("sessions", "add_to_cart"):
        frame[column] = frame[column].astype("float64")

    return frame[WEB_COLUMNS].sort_values(["sku_id", "date"]).reset_index(drop=True)


def clean_market_conditions(raw: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Clean the weekly competitor price index and weather anomaly."""
    if raw.empty:
        return pd.DataFrame(columns=CONDITION_COLUMNS)

    table = "market_conditions"
    frame = raw.copy()

    frame["week_start"] = _snap_to_monday(parse_mixed_dates(frame["week_start"]))
    frame = frame[frame["week_start"].notna()]

    resolved, _ = normalise_category(frame["category"], pd.Series("", index=frame.index))
    frame["category"] = resolved

    frame["competitor_price_index"] = _to_numeric(frame["competitor_price_index"])
    frame["weather_anomaly"] = _to_numeric(frame["weather_anomaly"]).fillna(0.0)

    low, high = COMPETITOR_INDEX_BOUNDS
    in_band = frame["competitor_price_index"].between(low, high)
    implausible = int((~in_band).sum())
    if implausible:
        frame["competitor_price_index"] = frame["competitor_price_index"].where(in_band).fillna(1.0)
        report.add(
            CleaningAction(
                table=table,
                issue=f"{implausible} implausible or missing competitor price index values",
                detection=f"range check outside {low}-{high}, or null",
                rows_affected=implausible,
                resolution="set to 1.0, meaning price parity",
                rationale="these are scrape failures rather than market moves; parity is "
                "the neutral assumption and does not push the forecast either way",
                severity="warning",
            )
        )

    duplicates = int(frame.duplicated(subset=["week_start", "category"]).sum())
    if duplicates:
        frame = frame.drop_duplicates(subset=["week_start", "category"], keep="last")
        report.add(
            CleaningAction(
                table=table,
                issue=f"{duplicates} duplicate (week, category) condition rows",
                detection="duplicate check on the declared primary key",
                rows_affected=duplicates,
                resolution="kept the last occurrence",
                rationale="a repeated scrape for the same week is a restatement; the "
                "latest reading is the most accurate",
                severity="info",
            )
        )

    for column in ("competitor_price_index", "weather_anomaly"):
        frame[column] = frame[column].astype("float64")

    return frame[CONDITION_COLUMNS].sort_values(["category", "week_start"]).reset_index(drop=True)
