"""Business Insights and Product Performance - pure aggregations, no new artifact.

Both back new dashboard pages (Business Insights, Product Performance) and are
deliberately **not** persisted: they are cheap groupbys over the weekly panel
and risk table the service already holds in memory (:mod:`service.store`), so
recomputing them per request is simpler than a third artifact file and can
never disagree with whichever `risk_table`/`weekly_panel` the store currently
has loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from foresight.logging_setup import get_logger

__all__ = [
    "BusinessInsights",
    "DeadStockSku",
    "MoverSku",
    "RevenueConcentrationPoint",
    "compute_business_insights",
    "compute_product_performance",
]

log = get_logger(__name__)

#: A SKU with at least this many consecutive zero-sale weeks, trailing the
#: most recent week in the panel, is flagged as dead stock. A full quarter -
#: long enough that a genuinely slow month doesn't trip it.
DEAD_STOCK_MIN_ZERO_WEEKS: Final[int] = 8

#: Trailing windows compared to rank top movers: the most recent N weeks
#: against the N weeks immediately before them.
MOVER_WINDOW_WEEKS: Final[int] = 8

#: How many gainers/decliners to report.
TOP_MOVERS_COUNT: Final[int] = 10

#: Revenue-concentration curve checkpoints, as a share of the total SKU count.
REVENUE_CONCENTRATION_FRACTIONS: Final[tuple[float, ...]] = (0.1, 0.2, 0.5)


@dataclass(frozen=True, slots=True)
class RevenueConcentrationPoint:
    """What share of total revenue the top X% of SKUs by revenue account for."""

    sku_fraction: float
    sku_count: int
    revenue_share: float


@dataclass(frozen=True, slots=True)
class DeadStockSku:
    sku_id: str
    category: str
    subcategory: str
    consecutive_zero_weeks: int
    last_sale_week: str | None


@dataclass(frozen=True, slots=True)
class MoverSku:
    sku_id: str
    category: str
    recent_revenue: float
    prior_revenue: float
    change_pct: float


@dataclass(frozen=True, slots=True)
class BusinessInsights:
    total_revenue: float
    total_skus: int
    revenue_concentration: tuple[RevenueConcentrationPoint, ...]
    dead_stock: tuple[DeadStockSku, ...]
    top_gainers: tuple[MoverSku, ...]
    top_decliners: tuple[MoverSku, ...]


def compute_business_insights(weekly_panel: pd.DataFrame) -> BusinessInsights:
    """Revenue concentration, dead stock, and top movers from the weekly panel."""
    if weekly_panel.empty:
        return BusinessInsights(
            total_revenue=0.0,
            total_skus=0,
            revenue_concentration=(),
            dead_stock=(),
            top_gainers=(),
            top_decliners=(),
        )

    panel = weekly_panel.sort_values(["sku_id", "week_start"])
    by_sku_revenue = (
        panel.groupby("sku_id", observed=True)["revenue"].sum().sort_values(ascending=False)
    )
    total_revenue = float(by_sku_revenue.sum())
    total_skus = int(by_sku_revenue.size)

    cumulative = by_sku_revenue.cumsum()
    concentration = [
        RevenueConcentrationPoint(
            sku_fraction=fraction,
            sku_count=(count := max(1, int(round(total_skus * fraction)))),
            revenue_share=(
                float(cumulative.iloc[count - 1] / total_revenue) if total_revenue > 0.0 else 0.0
            ),
        )
        for fraction in REVENUE_CONCENTRATION_FRACTIONS
    ]

    gainers, decliners = _top_movers(panel)
    return BusinessInsights(
        total_revenue=total_revenue,
        total_skus=total_skus,
        revenue_concentration=tuple(concentration),
        dead_stock=_dead_stock(panel),
        top_gainers=gainers,
        top_decliners=decliners,
    )


def _dead_stock(panel: pd.DataFrame) -> tuple[DeadStockSku, ...]:
    results: list[DeadStockSku] = []
    for sku_id, group in panel.groupby("sku_id", observed=True, sort=False):
        ordered = group.sort_values("week_start")
        is_zero = (ordered["units"] <= 0.0).to_numpy()
        trailing_zero = 0
        for value in is_zero[::-1]:
            if not value:
                break
            trailing_zero += 1
        if trailing_zero < DEAD_STOCK_MIN_ZERO_WEEKS:
            continue

        nonzero_weeks = ordered.loc[ordered["units"] > 0.0, "week_start"]
        last_sale = nonzero_weeks.max() if not nonzero_weeks.empty else None
        first_row = ordered.iloc[0]
        results.append(
            DeadStockSku(
                sku_id=str(sku_id),
                category=str(first_row["category"]),
                subcategory=str(first_row["subcategory"]),
                consecutive_zero_weeks=trailing_zero,
                last_sale_week=(
                    str(pd.Timestamp(last_sale).date()) if last_sale is not None else None
                ),
            )
        )
    results.sort(key=lambda row: row.consecutive_zero_weeks, reverse=True)
    return tuple(results)


def _trend_pct(revenue: pd.Series) -> float:
    """Recent N weeks vs. the N weeks immediately before them, as a fraction."""
    if len(revenue) < MOVER_WINDOW_WEEKS * 2:
        return 0.0
    recent = float(revenue.tail(MOVER_WINDOW_WEEKS).sum())
    prior = float(revenue.iloc[-(MOVER_WINDOW_WEEKS * 2) : -MOVER_WINDOW_WEEKS].sum())
    if prior <= 0.0:
        return 0.0
    return (recent - prior) / prior


def _top_movers(panel: pd.DataFrame) -> tuple[tuple[MoverSku, ...], tuple[MoverSku, ...]]:
    movers: list[MoverSku] = []
    for sku_id, group in panel.groupby("sku_id", observed=True, sort=False):
        ordered = group.sort_values("week_start")
        if len(ordered) < MOVER_WINDOW_WEEKS * 2:
            continue
        recent = float(ordered["revenue"].tail(MOVER_WINDOW_WEEKS).sum())
        prior = float(
            ordered["revenue"].iloc[-(MOVER_WINDOW_WEEKS * 2) : -MOVER_WINDOW_WEEKS].sum()
        )
        if prior <= 0.0:
            continue
        movers.append(
            MoverSku(
                sku_id=str(sku_id),
                category=str(ordered.iloc[-1]["category"]),
                recent_revenue=recent,
                prior_revenue=prior,
                change_pct=(recent - prior) / prior,
            )
        )
    gainers = tuple(sorted(movers, key=lambda row: row.change_pct, reverse=True)[:TOP_MOVERS_COUNT])
    decliners = tuple(sorted(movers, key=lambda row: row.change_pct)[:TOP_MOVERS_COUNT])
    return gainers, decliners


def compute_product_performance(
    weekly_panel: pd.DataFrame, risk_table: pd.DataFrame
) -> pd.DataFrame:
    """Per-SKU revenue/units rollup with recent trend and current risk action.

    Returns an empty frame if the panel itself is empty; a missing risk table
    (not yet scored) degrades to ``action="unknown"`` rows rather than failing,
    matching this project's "degrade honestly, never crash" convention.
    """
    if weekly_panel.empty:
        return pd.DataFrame()

    panel = weekly_panel.sort_values(["sku_id", "week_start"])
    totals = panel.groupby("sku_id", observed=True).agg(
        category=("category", "last"),
        subcategory=("subcategory", "last"),
        total_revenue=("revenue", "sum"),
        total_units=("units", "sum"),
    )
    total_revenue_sum = float(totals["total_revenue"].sum())
    totals["revenue_share"] = (
        totals["total_revenue"] / total_revenue_sum if total_revenue_sum > 0.0 else 0.0
    )
    totals["recent_trend_pct"] = panel.groupby("sku_id", observed=True)["revenue"].apply(
        _trend_pct
    )

    if not risk_table.empty:
        risk_columns = risk_table.set_index("sku_id")[["action", "action_label", "value_at_stake"]]
        totals = totals.join(risk_columns, how="left")
        totals["action"] = totals["action"].fillna("unknown")
        totals["action_label"] = totals["action_label"].fillna("Unknown")
        totals["value_at_stake"] = totals["value_at_stake"].fillna(0.0)
    else:
        totals["action"] = "unknown"
        totals["action_label"] = "Unknown"
        totals["value_at_stake"] = 0.0

    return totals.reset_index().sort_values("total_revenue", ascending=False)
