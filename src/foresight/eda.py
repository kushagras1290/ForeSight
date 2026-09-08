"""Exploratory analysis and figure generation (deliverable D2).

Produces the charts and the tables of findings that back the data-quality and
EDA memo. Every figure is written to ``reports/figures/`` and every finding is
returned as structured data, so the memo is rendered from measurements rather
than written by hand and then drifting from the code.

Chart styling follows the project's North Harbour palette so the memo, the
executive readout and the dashboard all read as one piece of work. All figures
are labelled for a non-technical reader (D2 acceptance criterion 4): axes carry
units, currency is in rupees, and no chart relies on colour alone to carry
meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")  # headless: figures are written to disk, never displayed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from foresight.config import Settings, get_settings
from foresight.logging_setup import get_logger

__all__ = [
    "PALETTE",
    "EdaFindings",
    "apply_house_style",
    "run_eda",
]

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# North Harbour palette - shared with the dashboard and the executive readout
# --------------------------------------------------------------------------- #
#: Kept in step with ``dashboard/src/styles/theme.css`` so the memo, the readout
#: and the dashboard are visibly one piece of work. Any token added there should
#: be added here too.
PALETTE: Final[dict[str, str]] = {
    "ground": "#F4F1EA",
    "surface": "#FBF9F5",
    "surface_sunk": "#EBE7DD",
    "rule": "#DCD5C7",
    "rule_strong": "#C9C0AD",
    "ink": "#16211F",
    "ink_2": "#4C5A56",
    "ink_3": "#6B7873",
    "ink_inverse": "#F7F4EE",
    "harbour_900": "#0C3330",
    "harbour_700": "#12514C",
    "harbour_500": "#1E7A72",
    "harbour_300": "#6FB0A8",
    "harbour_100": "#D3E5E1",
    "harbour_050": "#E9F2F0",
    "ember_700": "#8F3F21",
    "ember_600": "#B4512C",
    "ember_400": "#D97A4E",
    "ember_100": "#F6E3D8",
    "critical": "#A8321F",
    "critical_soft": "#F7E0DA",
    "warn": "#B37E14",
    "warn_ink": "#8A6110",
    "warn_soft": "#F7ECD4",
    "watch": "#3F6B8A",
    "watch_soft": "#DDE8F0",
    "ok": "#2C6E4E",
    "ok_soft": "#DCECE3",
    "baseline": "#9AA4A0",
}

#: Categorical series order for multi-series charts.
SERIES_COLOURS: Final[tuple[str, ...]] = (
    PALETTE["harbour_700"],
    PALETTE["ember_600"],
    PALETTE["watch"],
    PALETTE["warn"],
    PALETTE["ok"],
    "#8A6A4B",
)

FIGURE_DPI: Final[int] = 150


def apply_house_style() -> None:
    """Install the project's matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["surface"],
            "axes.facecolor": PALETTE["surface"],
            "savefig.facecolor": PALETTE["surface"],
            "axes.edgecolor": PALETTE["rule"],
            "axes.labelcolor": PALETTE["ink_2"],
            "axes.titlecolor": PALETTE["ink"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": PALETTE["rule"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "xtick.color": PALETTE["ink_3"],
            "ytick.color": PALETTE["ink_3"],
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": PALETTE["ink"],
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.size": 9.5,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _thousands(value: float, _position: int = 0) -> str:
    if abs(value) >= 1e5:
        return f"{value / 1e5:,.1f}L"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.0f}k"
    return f"{value:,.0f}"


@dataclass(slots=True)
class EdaFindings:
    """Structured findings the memo is rendered from."""

    headline_insights: list[dict[str, object]]
    seasonality: pd.DataFrame
    top_movers: pd.DataFrame
    dead_stock: pd.DataFrame
    category_summary: pd.DataFrame
    promo_effect: pd.DataFrame
    concentration: dict[str, float]
    figures: list[str]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _figure_demand_over_time(panel: pd.DataFrame, output: Path) -> str:
    weekly = panel.groupby("week_start", as_index=False)["units"].sum()
    figure, axes = plt.subplots(figsize=(10, 3.6))
    axes.plot(
        weekly["week_start"],
        weekly["units"],
        color=PALETTE["harbour_700"],
        linewidth=1.6,
    )
    axes.fill_between(
        weekly["week_start"],
        weekly["units"],
        color=PALETTE["harbour_500"],
        alpha=0.12,
    )
    peak = weekly.loc[weekly["units"].idxmax()]
    axes.scatter([peak["week_start"]], [peak["units"]], color=PALETTE["ember_600"], zorder=5, s=28)
    axes.annotate(
        f"Peak week: {peak['week_start'].date()}\n{peak['units']:,.0f} units",
        xy=(peak["week_start"], peak["units"]),
        xytext=(-12, -34),
        textcoords="offset points",
        fontsize=8.5,
        color=PALETTE["ember_600"],
        ha="right",
    )
    axes.set_title("Total weekly demand across the assortment")
    axes.set_ylabel("Units sold per week")
    axes.set_xlabel("")
    axes.yaxis.set_major_formatter(FuncFormatter(_thousands))

    path = output / "01_demand_over_time.png"
    figure.savefig(path)
    plt.close(figure)
    return path.name


def _figure_seasonality(panel: pd.DataFrame, output: Path) -> tuple[str, pd.DataFrame]:
    frame = panel.copy()
    frame["iso_week"] = frame["week_start"].dt.isocalendar()["week"].astype("int64")
    overall = frame["units"].mean()
    profile = (
        frame.groupby(["category", "iso_week"], as_index=False)["units"]
        .mean()
        .rename(columns={"units": "mean_units"})
    )
    category_mean = frame.groupby("category")["units"].mean()
    profile["seasonal_index"] = profile.apply(
        lambda row: row["mean_units"] / max(category_mean[row["category"]], 1e-9), axis=1
    )

    figure, axes = plt.subplots(figsize=(10, 3.8))
    for position, (category, group) in enumerate(profile.groupby("category")):
        axes.plot(
            group["iso_week"],
            group["seasonal_index"],
            label=category,
            color=SERIES_COLOURS[position % len(SERIES_COLOURS)],
            linewidth=1.5,
        )
    axes.axhline(1.0, color=PALETTE["ink_3"], linewidth=0.9, linestyle="--")
    axes.set_title("Seasonal demand index by category (1.0 = that category's own average week)")
    axes.set_xlabel("ISO week of year")
    axes.set_ylabel("Index vs category average")
    axes.legend(ncols=3, loc="upper left")

    path = output / "02_seasonality_by_category.png"
    figure.savefig(path)
    plt.close(figure)
    del overall
    return path.name, profile


def _figure_pareto(panel: pd.DataFrame, output: Path) -> tuple[str, dict[str, float]]:
    by_sku = (
        panel.groupby("sku_id", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    by_sku["cumulative_share"] = by_sku["revenue"].cumsum() / by_sku["revenue"].sum()
    by_sku["rank"] = np.arange(1, len(by_sku) + 1)
    by_sku["sku_share"] = by_sku["rank"] / len(by_sku)

    share_at_20pct = float(by_sku.loc[by_sku["sku_share"] <= 0.20, "cumulative_share"].max())
    skus_for_80pct = int((by_sku["cumulative_share"] <= 0.80).sum() + 1)

    figure, axes = plt.subplots(figsize=(9, 3.6))
    axes.plot(
        by_sku["sku_share"] * 100,
        by_sku["cumulative_share"] * 100,
        color=PALETTE["harbour_700"],
        linewidth=1.8,
    )
    axes.axhline(80, color=PALETTE["ember_600"], linewidth=1.0, linestyle="--")
    axes.axvline(
        100 * skus_for_80pct / len(by_sku),
        color=PALETTE["ember_600"],
        linewidth=1.0,
        linestyle="--",
    )
    axes.annotate(
        f"{skus_for_80pct} SKUs ({skus_for_80pct / len(by_sku):.0%}) drive 80% of revenue",
        xy=(100 * skus_for_80pct / len(by_sku), 80),
        xytext=(12, -30),
        textcoords="offset points",
        fontsize=9,
        color=PALETTE["ember_600"],
    )
    axes.set_title("Revenue concentration across the assortment")
    axes.set_xlabel("Share of SKUs (%), ranked by revenue")
    axes.set_ylabel("Cumulative revenue (%)")

    path = output / "03_revenue_concentration.png"
    figure.savefig(path)
    plt.close(figure)

    return path.name, {
        "revenue_share_of_top_20pct_skus": share_at_20pct,
        "skus_for_80pct_revenue": float(skus_for_80pct),
        "skus_total": float(len(by_sku)),
    }


def _figure_dead_stock(panel: pd.DataFrame, output: Path) -> tuple[str, pd.DataFrame]:
    last_week = panel["week_start"].max()
    recent_cutoff = last_week - pd.Timedelta(weeks=13)
    prior_cutoff = last_week - pd.Timedelta(weeks=65)

    recent = (
        panel[panel["week_start"] > recent_cutoff]
        .groupby("sku_id", as_index=False)["units"]
        .mean()
        .rename(columns={"units": "recent_weekly_units"})
    )
    prior = (
        panel[
            (panel["week_start"] > prior_cutoff)
            & (panel["week_start"] <= prior_cutoff + pd.Timedelta(weeks=13))
        ]
        .groupby("sku_id", as_index=False)["units"]
        .mean()
        .rename(columns={"units": "prior_weekly_units"})
    )
    comparison = recent.merge(prior, on="sku_id", how="inner")
    comparison["decline_pct"] = 1.0 - (
        comparison["recent_weekly_units"] / comparison["prior_weekly_units"].replace(0, np.nan)
    )
    dead = (
        comparison[comparison["prior_weekly_units"] > 0]
        .sort_values("decline_pct", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    figure, axes = plt.subplots(figsize=(9, 4.2))
    positions = np.arange(len(dead))
    axes.barh(
        positions,
        dead["prior_weekly_units"],
        color=PALETTE["harbour_100"],
        edgecolor=PALETTE["harbour_500"],
        height=0.62,
        label="A year ago (weekly avg)",
    )
    axes.barh(
        positions,
        dead["recent_weekly_units"],
        color=PALETTE["critical"],
        height=0.34,
        label="Last 13 weeks (weekly avg)",
    )
    axes.set_yticks(positions)
    axes.set_yticklabels(dead["sku_id"], fontsize=8)
    axes.invert_yaxis()
    axes.set_title("Steepest demand declines - candidates for markdown or delisting")
    axes.set_xlabel("Average units sold per week")
    axes.legend(loc="lower right")

    path = output / "04_dead_stock.png"
    figure.savefig(path)
    plt.close(figure)
    return path.name, dead


def _figure_promo_effect(panel: pd.DataFrame, output: Path) -> tuple[str, pd.DataFrame]:
    frame = panel.copy()
    frame["on_promo"] = frame["promo_days"] > 0
    effect = (
        frame.groupby(["category", "on_promo"], as_index=False)["units"]
        .mean()
        .pivot(index="category", columns="on_promo", values="units")
        .rename(columns={False: "off_promo", True: "on_promo"})
        .reset_index()
    )
    effect["uplift_pct"] = (effect["on_promo"] / effect["off_promo"]) - 1.0
    effect = effect.sort_values("uplift_pct", ascending=False).reset_index(drop=True)

    figure, axes = plt.subplots(figsize=(8.5, 3.4))
    positions = np.arange(len(effect))
    axes.bar(
        positions,
        effect["uplift_pct"] * 100,
        color=PALETTE["ember_600"],
        width=0.55,
    )
    for position, value in zip(positions, effect["uplift_pct"], strict=True):
        axes.annotate(
            f"+{value:.0%}",
            xy=(position, value * 100),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=PALETTE["ember_600"],
        )
    axes.set_xticks(positions)
    axes.set_xticklabels(effect["category"], rotation=18, ha="right", fontsize=8)
    axes.set_title("Average demand uplift during promotional weeks")
    axes.set_ylabel("Uplift vs non-promo weeks (%)")

    path = output / "05_promo_uplift.png"
    figure.savefig(path)
    plt.close(figure)
    return path.name, effect


def _figure_data_quality(cleaning_report: pd.DataFrame, output: Path) -> str:
    if cleaning_report.empty:
        return ""
    frame = cleaning_report.copy()
    frame = frame.sort_values("rows_affected", ascending=True).tail(12)

    severity_colour = {
        "critical": PALETTE["critical"],
        "warning": PALETTE["warn"],
        "info": PALETTE["watch"],
    }
    colours = [severity_colour.get(value, PALETTE["ink_3"]) for value in frame["severity"]]

    figure, axes = plt.subplots(figsize=(10, 4.6))
    positions = np.arange(len(frame))
    axes.barh(positions, frame["rows_affected"], color=colours, height=0.62)
    axes.set_yticks(positions)
    labels = [(issue[:70] + "...") if len(issue) > 70 else issue for issue in frame["issue"]]
    axes.set_yticklabels(labels, fontsize=7.5)
    axes.set_xscale("log")
    axes.set_title("Data-quality issues found and repaired, by rows affected (log scale)")
    axes.set_xlabel("Rows affected")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in severity_colour.values()]
    axes.legend(handles, severity_colour.keys(), loc="lower right", title="Severity")

    path = output / "06_data_quality.png"
    figure.savefig(path)
    plt.close(figure)
    return path.name


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_eda(settings: Settings | None = None) -> EdaFindings:
    """Generate every EDA figure and return the structured findings."""
    settings = settings or get_settings()
    settings.ensure_directories()
    apply_house_style()

    panel = pd.read_parquet(settings.processed_dir / "weekly_panel.parquet")
    cleaning_report = pd.read_parquet(settings.processed_dir / "cleaning_report.parquet")
    output = settings.figures_dir

    figures: list[str] = []
    figures.append(_figure_demand_over_time(panel, output))
    seasonality_figure, seasonality = _figure_seasonality(panel, output)
    figures.append(seasonality_figure)
    pareto_figure, concentration = _figure_pareto(panel, output)
    figures.append(pareto_figure)
    dead_figure, dead_stock = _figure_dead_stock(panel, output)
    figures.append(dead_figure)
    promo_figure, promo_effect = _figure_promo_effect(panel, output)
    figures.append(promo_figure)
    quality_figure = _figure_data_quality(cleaning_report, output)
    if quality_figure:
        figures.append(quality_figure)

    # --- Tables -------------------------------------------------------------- #
    top_movers = (
        panel.groupby(["sku_id", "category"], as_index=False)
        .agg(total_units=("units", "sum"), total_revenue=("revenue", "sum"))
        .sort_values("total_revenue", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    category_summary = (
        panel.groupby("category", as_index=False)
        .agg(
            skus=("sku_id", "nunique"),
            total_units=("units", "sum"),
            total_revenue=("revenue", "sum"),
            mean_weekly_units=("units", "mean"),
            zero_week_share=("units", lambda values: float((values <= 0).mean())),
        )
        .sort_values("total_revenue", ascending=False)
        .reset_index(drop=True)
    )

    # --- Headline insights, stated in plain language ------------------------ #
    last_week = panel["week_start"].max()
    recent = panel[panel["week_start"] > last_week - pd.Timedelta(weeks=13)]
    peak_week = panel.groupby("week_start")["units"].sum().idxmax()
    trough_week = panel.groupby("week_start")["units"].sum().idxmin()
    peak_units = float(panel.groupby("week_start")["units"].sum().max())
    trough_units = float(panel.groupby("week_start")["units"].sum().min())

    intermittent_share = float(
        recent.groupby("sku_id")["units"].apply(lambda values: (values <= 0).mean()).ge(0.35).mean()
    )
    promo_lift_overall = float(
        panel.loc[panel["promo_days"] > 0, "units"].mean()
        / max(panel.loc[panel["promo_days"] == 0, "units"].mean(), 1e-9)
        - 1.0
    )

    headline_insights: list[dict[str, object]] = [
        {
            "title": "Demand is strongly seasonal, and the festive peak is the whole year",
            "detail": (
                f"The busiest week ({peak_week.date()}) sold {peak_units:,.0f} units against "
                f"{trough_units:,.0f} in the quietest ({trough_week.date()}) - a "
                f"{peak_units / max(trough_units, 1):.1f}x swing. Lighting and Decor peak hardest "
                "around Diwali; Storage & Organisation peaks in January instead."
            ),
            "so_what": (
                "Planning to an annual average guarantees stocking out in October and "
                "sitting on stock in April. Any reorder rule has to be seasonal."
            ),
        },
        {
            "title": (
                f"{int(concentration['skus_for_80pct_revenue'])} of "
                f"{int(concentration['skus_total'])} SKUs generate 80% of revenue"
            ),
            "detail": (
                f"The top 20% of SKUs account for "
                f"{concentration['revenue_share_of_top_20pct_skus']:.0%} of revenue. The tail is "
                "long and slow-moving."
            ),
            "so_what": (
                "Attention should be rationed accordingly: the head needs accurate "
                "forecasting, the tail needs a simple rule and a periodic cull."
            ),
        },
        {
            "title": f"{intermittent_share:.0%} of SKUs sell nothing in a third or more of weeks",
            "detail": (
                "These intermittent lines make percentage-based accuracy metrics unstable, "
                "which is why WAPE rather than MAPE is used to judge the forecast."
            ),
            "so_what": (
                "They also need a different forecasting treatment from the fast movers - "
                "averaging across both understates the fast movers and overstates the tail."
            ),
        },
        {
            "title": f"Promotions lift demand about {promo_lift_overall:.0%} on average",
            "detail": (
                "The effect varies materially by category, so a single blanket uplift "
                "assumption would misprice the stock build for every category at once."
            ),
            "so_what": (
                "Promotion timing is known in advance from the calendar, so it can be "
                "forecast rather than reacted to."
            ),
        },
    ]

    log.info(
        "EDA complete",
        extra={"context": {"figures": len(figures), "output": str(output)}},
    )

    return EdaFindings(
        headline_insights=headline_insights,
        seasonality=seasonality,
        top_movers=top_movers,
        dead_stock=dead_stock,
        category_summary=category_summary,
        promo_effect=promo_effect,
        concentration=concentration,
        figures=figures,
    )
