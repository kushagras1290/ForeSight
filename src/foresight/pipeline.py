"""End-to-end data pipeline: raw extracts to analysis-ready datasets (D1).

One command, no manual steps::

    python scripts/01_run_pipeline.py

Stages
------
1. **Ingest**   - read the four extracts as text and profile them on arrival.
2. **Clean**    - normalise, repair and reconcile, recording every decision.
3. **Densify**  - rebuild the complete SKU x date grid. The extract only
   contains rows where something sold, so absent days are silently missing
   zeros. A time-series model must see those zeros or every lag is wrong.
4. **Join**     - attach product and calendar attributes.
5. **Aggregate**- roll daily records up to the weekly modelling grain.
6. **Validate** - enforce the full column contract before anything is written.

Outputs land in ``data/processed/`` as Parquet:

``analysis_ready.parquet``
    Daily grain, one row per SKU per day, fully joined. The canonical
    analysis-ready dataset.
``weekly_panel.parquet``
    Weekly grain, the modelling table.
``calendar.parquet``
    The full date dimension including forward-published events, needed for
    known-in-advance features over the forecast horizon.
``inventory_clean.parquet``
    Cleaned stock positions, consumed by the risk layer.
``cleaning_report.parquet``
    The audit trail of every cleaning decision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from foresight.clean import (
    CleaningAction,
    CleaningReport,
    clean_calendar,
    clean_inventory,
    clean_sales,
    clean_sku_master,
)
from foresight.config import Settings, get_settings
from foresight.exceptions import DataQualityError
from foresight.ingest import TableProfile, load_raw_extracts
from foresight.logging_setup import get_logger
from foresight.schemas import (
    ANALYSIS_READY_SCHEMA,
    CLEAN_CALENDAR_SCHEMA,
    CLEAN_INVENTORY_SCHEMA,
    CLEAN_SALES_SCHEMA,
    CLEAN_SKU_MASTER_SCHEMA,
    WEEKLY_PANEL_SCHEMA,
    validate_dataframe,
)

__all__ = ["PipelineResult", "run_pipeline"]

log = get_logger(__name__)

#: A week is only usable if all seven days are present. Partial weeks at a SKU's
#: launch or at the edge of the extract would understate demand and poison lags.
DAYS_PER_WEEK = 7


@dataclass(slots=True)
class PipelineResult:
    """Everything the pipeline produced, in memory."""

    analysis_ready: pd.DataFrame
    weekly_panel: pd.DataFrame
    calendar: pd.DataFrame
    inventory: pd.DataFrame
    sku_master: pd.DataFrame
    cleaning_report: CleaningReport
    raw_profiles: dict[str, TableProfile] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def n_skus(self) -> int:
        return int(self.weekly_panel["sku_id"].nunique())

    @property
    def n_weeks(self) -> int:
        return int(self.weekly_panel["week_start"].nunique())


def _build_daily_grid(
    sales: pd.DataFrame,
    sku_master: pd.DataFrame,
    report: CleaningReport,
) -> pd.DataFrame:
    """Expand sales to a complete SKU x date grid, zero-filling absent days.

    A SKU's grid starts at the later of the extract's first date and the SKU's
    launch date, so a product is never credited with zero demand before it
    existed.
    """
    first_date = sales["date"].min()
    last_date = sales["date"].max()
    all_dates = pd.date_range(first_date, last_date, freq="D")

    launch = sku_master.set_index("sku_id")["launch_date"]
    # Cross-join every SKU against every date, then trim to each SKU's lifetime.
    grid = pd.MultiIndex.from_product(
        [sku_master["sku_id"].to_numpy(), all_dates], names=["sku_id", "date"]
    ).to_frame(index=False)

    grid["_launch"] = grid["sku_id"].map(launch)
    grid = grid[grid["date"] >= grid["_launch"]].drop(columns=["_launch"])

    merged = grid.merge(sales, on=["sku_id", "date"], how="left")

    filled_rows = int(merged["units_sold"].isna().sum())
    if filled_rows:
        merged["units_sold"] = merged["units_sold"].fillna(0.0)
        merged["revenue"] = merged["revenue"].fillna(0.0)
        merged["promo_flag"] = merged["promo_flag"].fillna(0).astype("int64")
        report.add(
            CleaningAction(
                table="sales_daily",
                issue=f"{filled_rows} SKU-day combination(s) absent from the extract",
                detection="anti-join of the extract against a complete SKU x date grid "
                "bounded by each SKU's launch date",
                rows_affected=filled_rows,
                resolution="materialised as explicit zero-demand rows",
                rationale="the export only carries rows where a sale occurred, so a "
                "zero-demand day is simply missing. Lags, rolling means and "
                "the seasonal-naive baseline all index by position in time - "
                "without the zeros every one of them silently shifts and the "
                "model learns a demand level that never existed",
                severity="critical",
            )
        )

    # Price is an offered price, not a transaction outcome: it is known on days
    # with no sale. Carry it within the SKU, then fall back to list price.
    merged = merged.sort_values(["sku_id", "date"])
    merged["unit_price"] = merged.groupby("sku_id")["unit_price"].ffill()
    merged["unit_price"] = merged.groupby("sku_id")["unit_price"].bfill()
    list_price = sku_master.set_index("sku_id")["list_price"]
    merged["unit_price"] = merged["unit_price"].fillna(merged["sku_id"].map(list_price))

    return merged.reset_index(drop=True)


def _build_weekly_panel(
    daily: pd.DataFrame,
    report: CleaningReport,
) -> pd.DataFrame:
    """Aggregate the daily grid to the weekly modelling grain."""
    aggregations: dict[str, tuple[str, str]] = {
        "units": ("units_sold", "sum"),
        "revenue": ("revenue", "sum"),
        "avg_price": ("unit_price", "mean"),
        "promo_days": ("promo_flag", "sum"),
        "holiday_days": ("is_holiday", "sum"),
        "days_observed": ("date", "count"),
        "category": ("category", "first"),
        "subcategory": ("subcategory", "first"),
        "unit_cost": ("unit_cost", "first"),
        "list_price": ("list_price", "first"),
        "launch_date": ("launch_date", "first"),
    }
    # Discount depth is a property of the offer, so the week's representative
    # value is its mean across the days it applied - not a sum.
    if "discount_pct" in daily.columns:
        aggregations["discount_pct"] = ("discount_pct", "mean")
    if "sessions" in daily.columns:
        aggregations["sessions"] = ("sessions", "sum")
        aggregations["add_to_cart"] = ("add_to_cart", "sum")

    grouped = daily.groupby(["sku_id", "week_start"], as_index=False, observed=True).agg(
        **aggregations
    )

    partial = grouped["days_observed"] < DAYS_PER_WEEK
    partial_weeks = int(partial.sum())
    if partial_weeks:
        grouped = grouped[~partial]
        report.add(
            CleaningAction(
                table="weekly_panel",
                issue=f"{partial_weeks} SKU-week(s) covering fewer than seven days",
                detection="day count per (sku_id, week_start) after densification",
                rows_affected=partial_weeks,
                resolution="dropped from the modelling panel",
                rationale="a part-week at a SKU's launch or at the edge of the extract "
                "holds less demand purely because it is shorter. Kept, it would "
                "read as a demand collapse and drag both the lag features and "
                "the seasonal-naive baseline down",
                severity="warning",
            )
        )

    grouped = grouped.drop(columns=["days_observed"])

    weeks_since_launch = (grouped["week_start"] - grouped["launch_date"]).dt.days // DAYS_PER_WEEK
    grouped["weeks_since_launch"] = weeks_since_launch.clip(lower=0).astype("int64")
    grouped["is_post_launch"] = (weeks_since_launch >= 0).astype("int64")

    for column in ("units", "revenue", "avg_price", "promo_days", "holiday_days"):
        grouped[column] = grouped[column].astype("float64")

    return grouped.sort_values(["sku_id", "week_start"]).reset_index(drop=True)


def _attach_drivers(
    weekly: pd.DataFrame,
    marketing: pd.DataFrame,
    conditions: pd.DataFrame,
    report: CleaningReport,
) -> pd.DataFrame:
    """Join the commercial driver tables onto the weekly panel.

    Every driver column is guaranteed to exist afterwards, zero-filled where the
    extract was absent or a week had no matching row. That keeps the weekly
    panel's contract stable whether or not the client supplies driver data, so
    the model degrades to history-only forecasting rather than failing.
    """
    frame = weekly.copy()

    # --- Media plan ---------------------------------------------------------- #
    if not marketing.empty:
        before = len(frame)
        frame = frame.merge(
            marketing, on=["sku_id", "week_start"], how="left", validate="one_to_one"
        )
        unmatched = int(frame["media_spend"].isna().sum())
        if unmatched:
            report.add(
                CleaningAction(
                    table="marketing_spend",
                    issue=f"{unmatched} SKU-weeks with no line in the media plan",
                    detection="left join of the weekly panel against the media plan",
                    rows_affected=unmatched,
                    resolution="set to zero spend",
                    rationale="a product absent from the plan for a week received no "
                    "budget that week; zero is the fact, not a gap",
                    severity="info",
                )
            )
        assert len(frame) == before, "media join changed the panel grain"
    for column in ("media_spend", "impressions", "email_sends"):
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = frame[column].fillna(0.0).astype("float64")

    # --- Market conditions --------------------------------------------------- #
    if not conditions.empty:
        frame = frame.merge(
            conditions, on=["category", "week_start"], how="left", validate="many_to_one"
        )
    for column, default in (("competitor_price_index", 1.0), ("weather_anomaly", 0.0)):
        if column not in frame.columns:
            frame[column] = default
        frame[column] = frame[column].fillna(default).astype("float64")

    # --- Storefront analytics and discount depth ----------------------------- #
    # Both arrive through the daily grid rather than a separate join.
    for column in ("sessions", "add_to_cart", "discount_pct"):
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = frame[column].fillna(0.0).astype("float64")

    return frame


def run_pipeline(settings: Settings | None = None, *, write: bool = True) -> PipelineResult:
    """Run the full pipeline from raw extracts to validated analysis-ready data.

    Args:
        settings: Configuration; the process singleton by default.
        write: Whether to persist outputs to ``data/processed/``.

    Raises:
        MissingDataFileError: an extract is absent.
        SchemaValidationError: an output violates its contract.
        DataQualityError: the data is unusable after cleaning.
    """
    settings = settings or get_settings()
    settings.ensure_directories()
    started = time.perf_counter()

    # --- 1. Ingest ---------------------------------------------------------- #
    raw = load_raw_extracts(settings)
    report = CleaningReport(actions=[])

    # --- 2. Clean ----------------------------------------------------------- #
    # Order matters: sales and inventory are validated against the cleaned
    # product master, so that has to come first.
    sku_master = clean_sku_master(raw.sku_master, report)
    validate_dataframe(sku_master, CLEAN_SKU_MASTER_SCHEMA)

    calendar = clean_calendar(raw.calendar, report)
    validate_dataframe(calendar, CLEAN_CALENDAR_SCHEMA)

    sales = clean_sales(raw.sales_daily, sku_master, report)
    validate_dataframe(sales, CLEAN_SALES_SCHEMA)

    inventory = clean_inventory(raw.inventory_snapshots, sku_master, report, settings)
    validate_dataframe(inventory, CLEAN_INVENTORY_SCHEMA)

    # --- Optional commercial drivers ---------------------------------------- #
    # Delegated wholesale to the extension package, and skipped entirely when it
    # is not installed. This is the only seam between core and the extension on
    # the cleaning side.
    known_skus = set(sku_master["sku_id"])
    marketing = pd.DataFrame()
    web_analytics = pd.DataFrame()
    conditions = pd.DataFrame()

    if raw.has_drivers:
        from foresight_drivers import (
            CLEAN_MARKET_CONDITIONS_SCHEMA,
            CLEAN_MARKETING_SCHEMA,
            CLEAN_WEB_ANALYTICS_SCHEMA,
            clean_market_conditions,
            clean_marketing,
            clean_web_analytics,
        )

        marketing = clean_marketing(raw.marketing_spend, known_skus, report)
        web_analytics = clean_web_analytics(raw.web_analytics, known_skus, report)
        conditions = clean_market_conditions(raw.market_conditions, report)

        if not marketing.empty:
            validate_dataframe(marketing, CLEAN_MARKETING_SCHEMA)
        if not web_analytics.empty:
            validate_dataframe(web_analytics, CLEAN_WEB_ANALYTICS_SCHEMA)
        if not conditions.empty:
            validate_dataframe(conditions, CLEAN_MARKET_CONDITIONS_SCHEMA)

    log.info(
        "commercial drivers",
        extra={
            "context": {
                "marketing_rows": len(marketing),
                "web_analytics_rows": len(web_analytics),
                "market_conditions_rows": len(conditions),
                "history_only": raw.marketing_spend.empty and raw.web_analytics.empty,
            }
        },
    )

    # --- 3. Densify --------------------------------------------------------- #
    daily = _build_daily_grid(sales, sku_master, report)

    # --- 4. Join ------------------------------------------------------------ #
    calendar_columns = ["date", "week_start", "is_holiday", "promo_event", "season"]
    daily = daily.merge(sku_master, on="sku_id", how="left", validate="many_to_one")
    daily = daily.merge(calendar[calendar_columns], on="date", how="left", validate="many_to_one")

    # Daily analytics ride along on the grid so they aggregate with everything
    # else; absent days are genuine zero-traffic days once densified.
    if not web_analytics.empty:
        daily = daily.merge(web_analytics, on=["sku_id", "date"], how="left", validate="one_to_one")
        daily["sessions"] = daily["sessions"].fillna(0.0)
        daily["add_to_cart"] = daily["add_to_cart"].fillna(0.0)
    if "discount_pct" not in daily.columns:
        daily["discount_pct"] = 0.0
    daily["discount_pct"] = daily["discount_pct"].fillna(0.0)

    orphan_dates = int(daily["week_start"].isna().sum())
    if orphan_dates:
        raise DataQualityError(
            f"{orphan_dates} sales row(s) fall on dates absent from the calendar extract. "
            "The calendar must cover the full sales history; extend it and re-run."
        )

    daily = daily.sort_values(["sku_id", "date"]).reset_index(drop=True)
    validate_dataframe(daily, ANALYSIS_READY_SCHEMA)

    # --- 5. Aggregate ------------------------------------------------------- #
    weekly = _build_weekly_panel(daily, report)
    weekly = _attach_drivers(weekly, marketing, conditions, report)
    validate_dataframe(weekly, WEEKLY_PANEL_SCHEMA)

    if weekly.empty:
        raise DataQualityError("Weekly panel is empty after aggregation; nothing to model.")

    duration = time.perf_counter() - started
    result = PipelineResult(
        analysis_ready=daily,
        weekly_panel=weekly,
        calendar=calendar,
        inventory=inventory,
        sku_master=sku_master,
        cleaning_report=report,
        raw_profiles=raw.profiles,
        duration_seconds=round(duration, 3),
    )

    log.info(
        "pipeline complete",
        extra={
            "context": {
                "daily_rows": len(daily),
                "weekly_rows": len(weekly),
                "skus": result.n_skus,
                "weeks": result.n_weeks,
                "cleaning_actions": len(report.actions),
                "seconds": result.duration_seconds,
            }
        },
    )

    # --- 6. Persist --------------------------------------------------------- #
    if write:
        outputs = {
            "analysis_ready": daily,
            "weekly_panel": weekly,
            "calendar": calendar,
            "inventory_clean": inventory,
            "sku_master_clean": sku_master,
            "cleaning_report": report.to_frame(),
        }
        # The media plan is persisted whole, including weeks beyond the sales
        # history: forward-committed spend is a target-week feature, so the
        # forecaster needs rows the weekly panel does not have.
        if not marketing.empty:
            outputs["marketing_clean"] = marketing
        if not conditions.empty:
            outputs["market_conditions_clean"] = conditions
        for name, frame in outputs.items():
            path = settings.processed_dir / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            log.info(
                "wrote processed dataset",
                extra={"context": {"dataset": name, "path": str(path), "rows": len(frame)}},
            )

    return result
