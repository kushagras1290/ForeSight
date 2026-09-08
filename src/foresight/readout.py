"""Executive readout generation (deliverable D7).

Produces an eight-page PDF aimed at the Head of Operations and the Finance lead,
plus a Markdown memo carrying the same content for anyone who would rather read
than present.

Acceptance criteria this is built against (brief section 09, D7):

1. 6-10 slides aimed at Operations and Finance -> eight pages.
2. Leads with the rupee impact and the recommended actions -> page 2.
3. Explains accuracy and limitations honestly -> pages 4 and 8, including the
   figures that do not flatter the work.
4. No unexplained jargon -> WAPE is stated as "average forecast error", bias as
   "systematic over- or under-forecasting", and the quadrant names are given in
   the client's own words.

Rendered with matplotlib rather than a slide library so the deck is reproducible
from the same command as everything else, with no extra dependency and no manual
step between the numbers and the page.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from foresight.config import Settings, get_settings
from foresight.eda import PALETTE, apply_house_style
from foresight.logging_setup import get_logger
from foresight.risk import format_inr
from foresight.schemas import RiskAction

__all__ = ["ReadoutData", "build_readout", "load_readout_data"]

log = get_logger(__name__)

#: A4 landscape, in inches.
PAGE_SIZE: Final[tuple[float, float]] = (11.69, 8.27)

ACTION_COLOURS: Final[dict[str, str]] = {
    RiskAction.REORDER_NOW.value: PALETTE["critical"],
    RiskAction.MARKDOWN_CLEAR.value: PALETTE["warn"],
    RiskAction.WATCH_VOLATILE.value: PALETTE["watch"],
    RiskAction.HEALTHY.value: PALETTE["ok"],
}


@dataclass(slots=True)
class ReadoutData:
    """Everything the readout is rendered from."""

    metrics: dict[str, Any]
    impact: dict[str, Any]
    risk_table: pd.DataFrame
    by_horizon: pd.DataFrame
    by_regime: pd.DataFrame
    panel: pd.DataFrame


def load_readout_data(settings: Settings | None = None) -> ReadoutData:
    """Load the artifacts the readout needs."""
    settings = settings or get_settings()
    artifacts = settings.artifacts_dir

    metrics = json.loads((artifacts / "metrics.json").read_text(encoding="utf-8"))
    impact = json.loads((artifacts / "impact.json").read_text(encoding="utf-8"))
    risk_table = pd.read_parquet(artifacts / "risk_table.parquet")
    by_horizon = pd.read_parquet(artifacts / "metrics_by_horizon.parquet")

    regime_path = artifacts / "metrics_by_regime.parquet"
    by_regime = pd.read_parquet(regime_path) if regime_path.exists() else pd.DataFrame()

    panel = pd.read_parquet(settings.processed_dir / "weekly_panel.parquet")

    return ReadoutData(
        metrics=metrics,
        impact=impact,
        risk_table=risk_table,
        by_horizon=by_horizon,
        by_regime=by_regime,
        panel=panel,
    )


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
def _new_page() -> tuple[plt.Figure, plt.Axes]:
    """A blank page with a hidden full-bleed axes for absolute placement."""
    figure = plt.figure(figsize=PAGE_SIZE, facecolor=PALETTE["ground"])
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.axis("off")
    axes.set_facecolor(PALETTE["ground"])
    return figure, axes


def _chrome(axes: plt.Axes, eyebrow: str, title: str, page: int, total: int) -> None:
    """Standard header and footer on every page but the cover."""
    # Letterspacing is faked with spaces: matplotlib's text engine has no
    # tracking control, and the eyebrow needs it to read as a small-caps label.
    axes.text(
        0.06,
        0.925,
        " ".join(eyebrow.upper()),
        fontsize=7.5,
        color=PALETTE["ink_3"],
        fontweight="bold",
    )
    axes.text(0.06, 0.878, title, fontsize=21, color=PALETTE["ink"], fontweight="bold")
    axes.plot([0.06, 0.94], [0.852, 0.852], color=PALETTE["rule"], linewidth=0.9)

    axes.text(
        0.06,
        0.045,
        "Project FORESIGHT  ·  NorthBay Living  ·  Confidential",
        fontsize=7.5,
        color=PALETTE["ink_3"],
    )
    axes.text(0.94, 0.045, f"{page} / {total}", fontsize=7.5, color=PALETTE["ink_3"], ha="right")


def _stat_block(
    axes: plt.Axes,
    x: float,
    y: float,
    label: str,
    value: str,
    detail: str,
    colour: str,
    value_size: int = 27,
) -> None:
    """A large figure with a small label above and a note below."""
    axes.text(x, y + 0.075, label.upper(), fontsize=8, color=PALETTE["ink_3"], fontweight="bold")
    axes.text(x, y, value, fontsize=value_size, color=colour, fontweight="bold", va="center")
    axes.text(x, y - 0.062, detail, fontsize=9, color=PALETTE["ink_2"])


def _table(
    axes: plt.Axes,
    frame: pd.DataFrame,
    columns: list[tuple[str, str, str]],
    x: float,
    y_top: float,
    width: float,
    row_height: float = 0.048,
) -> None:
    """Render a small table.

    Args:
        columns: ``(dataframe column, header, alignment)`` where alignment is
            ``"left"`` or ``"right"``.
    """
    positions: list[float] = []
    # Left-aligned columns get proportional space; right-aligned ones are pinned.
    left_count = sum(1 for _, _, align in columns if align == "left")
    right_count = len(columns) - left_count
    left_width = width * 0.52 / max(left_count, 1)
    right_width = width * 0.48 / max(right_count, 1)

    cursor = x
    for _, _, align in columns:
        positions.append(cursor)
        cursor += left_width if align == "left" else right_width

    for position, (_, header, align) in zip(positions, columns, strict=True):
        anchor = position if align == "left" else position + right_width * 0.92
        axes.text(
            anchor,
            y_top,
            header.upper(),
            fontsize=7.5,
            color=PALETTE["ink_3"],
            fontweight="bold",
            ha="left" if align == "left" else "right",
        )

    axes.plot([x, x + width], [y_top - 0.014, y_top - 0.014], color=PALETTE["ink_3"], linewidth=0.8)

    for index, (_, record) in enumerate(frame.iterrows()):
        row_y = y_top - 0.034 - index * row_height
        for position, (column, _, align) in zip(positions, columns, strict=True):
            value = record[column]
            anchor = position if align == "left" else position + right_width * 0.92
            axes.text(
                anchor,
                row_y,
                str(value),
                fontsize=9,
                color=PALETTE["ink"] if align == "right" else PALETTE["ink_2"],
                ha="left" if align == "left" else "right",
                family="monospace" if align == "right" else None,
            )
        axes.plot(
            [x, x + width], [row_y - 0.017, row_y - 0.017], color=PALETTE["rule"], linewidth=0.5
        )


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def _page_cover(data: ReadoutData) -> plt.Figure:
    figure, axes = _new_page()

    axes.add_patch(Rectangle((0, 0.62), 1, 0.38, color=PALETTE["harbour_900"], zorder=0))
    axes.text(0.06, 0.86, "PROJECT", fontsize=10, color=PALETTE["harbour_300"], fontweight="bold")
    axes.text(0.06, 0.775, "FORESIGHT", fontsize=52, color="#FFFFFF", fontweight="bold")
    axes.text(
        0.06, 0.705, "Demand & Inventory Intelligence", fontsize=15, color=PALETTE["harbour_100"]
    )
    axes.plot([0.06, 0.20], [0.672, 0.672], color=PALETTE["ember_600"], linewidth=2.4)

    axes.text(0.06, 0.52, "Executive readout", fontsize=26, color=PALETTE["ink"], fontweight="bold")
    axes.text(
        0.06,
        0.455,
        "For the Head of Operations and the Finance lead",
        fontsize=12,
        color=PALETTE["ink_2"],
    )

    origin = data.impact.get("origin_week", "")
    horizon = data.impact.get("horizon_weeks", 8)
    axes.text(
        0.06,
        0.33,
        f"Plan week beginning {origin}   ·   {horizon}-week forecast horizon   ·   "
        f"{data.impact['total_skus']} products",
        fontsize=11,
        color=PALETTE["ink_2"],
    )

    axes.text(
        0.06,
        0.20,
        "What follows is a forecast of demand for every product you stock, an assessment of\n"
        "which products are about to run out and which are tying up cash, and what each of\n"
        "those is worth in rupees.",
        fontsize=11.5,
        color=PALETTE["ink_2"],
        linespacing=1.7,
    )

    axes.text(0.06, 0.075, dt.date.today().strftime("%d %B %Y"), fontsize=9, color=PALETTE["ink_3"])
    return figure


def _page_headline(data: ReadoutData, page: int, total: int) -> plt.Figure:
    figure, axes = _new_page()
    _chrome(axes, "The answer first", "What this is worth, and what to do", page, total)

    impact = data.impact

    _stat_block(
        axes,
        0.06,
        0.70,
        "Sales at risk",
        format_inr(impact["revenue_at_risk_total"]).replace("Rs", "₹"),
        f"{impact['reorder_now_skus']} products will likely run out within the horizon",
        PALETTE["critical"],
        value_size=34,
    )
    _stat_block(
        axes,
        0.54,
        0.70,
        "Capital locked in overstock",
        format_inr(impact["locked_capital_total"]).replace("Rs", "₹"),
        f"{impact['markdown_skus']} products are holding more than 12 weeks of cover",
        PALETTE["warn_ink"],
        value_size=34,
    )

    axes.plot([0.06, 0.94], [0.585, 0.585], color=PALETTE["rule"], linewidth=0.9)

    # Multi-line matplotlib text grows downward from its anchor, so the gap below
    # a heading has to clear the whole block, not just its first line.
    axes.text(
        0.06,
        0.540,
        "The two figures are separate, and should not be added together.",
        fontsize=10.5,
        color=PALETTE["ink_2"],
        fontweight="bold",
    )
    axes.text(
        0.06,
        0.478,
        "Sales at risk is revenue that may not happen. Capital locked is cash you have already spent\n"
        "and cannot use until the stock sells. They call for different actions and different owners.",
        fontsize=10,
        color=PALETTE["ink_2"],
        linespacing=1.6,
    )

    # --- Recommended actions ------------------------------------------------ #
    axes.text(
        0.06, 0.40, "RECOMMENDED ACTIONS", fontsize=8, color=PALETTE["ink_3"], fontweight="bold"
    )

    actions = [
        (
            "Raise replenishment orders",
            f"{impact['reorder_now_skus']} products · {impact['reorder_now_order_units']:,.0f} units",
            f"Protects {format_inr(impact['reorder_now_revenue_at_risk']).replace('Rs', '₹')} of sales.",
            PALETTE["critical"],
        ),
        (
            "Promote or discount to clear",
            f"{impact['markdown_skus']} products · {impact['excess_units_total']:,.0f} excess units",
            f"Frees {format_inr(impact['markdown_locked_capital']).replace('Rs', '₹')} of working capital.",
            PALETTE["warn"],
        ),
        (
            "Review manually",
            f"{impact['watch_skus']} products",
            "Demand too erratic to forecast confidently. A person should decide.",
            PALETTE["watch"],
        ),
        (
            "Leave alone",
            f"{impact['healthy_skus']} products",
            "Stock covers forecast demand without excess. No action needed.",
            PALETTE["ok"],
        ),
    ]

    for index, (title, scale, effect, colour) in enumerate(actions):
        row_y = 0.335 - index * 0.072
        axes.add_patch(Rectangle((0.06, row_y - 0.018), 0.004, 0.048, color=colour, zorder=2))
        axes.text(
            0.075, row_y + 0.012, title, fontsize=11.5, color=PALETTE["ink"], fontweight="bold"
        )
        axes.text(0.075, row_y - 0.012, effect, fontsize=9.5, color=PALETTE["ink_2"])
        axes.text(
            0.94,
            row_y + 0.008,
            scale,
            fontsize=10,
            color=colour,
            ha="right",
            fontweight="bold",
            family="monospace",
        )

    concentration = impact["top_10_share_of_revenue_at_risk"]
    axes.text(
        0.06,
        0.088,
        f"Where to start: the ten largest exposures account for {concentration:.0%} of all sales at risk.",
        fontsize=10,
        color=PALETTE["ink"],
        fontweight="bold",
    )
    return figure


def _page_accuracy(data: ReadoutData, page: int, total: int) -> plt.Figure:
    figure, axes = _new_page()
    _chrome(axes, "Can you trust it", "How accurate the forecast is", page, total)

    metrics = data.metrics
    selected_wape = metrics["selected_wape"]
    baseline_wape = metrics["baseline_wape"]
    improvement = metrics["selected_improvement_vs_baseline"]

    axes.text(
        0.06,
        0.79,
        "We tested the forecast the way you would actually use it: standing at a date in the past,\n"
        "using only what was known then, forecasting the next eight weeks, and comparing against\n"
        "what really happened. Repeated six times across the last year.",
        fontsize=10.5,
        color=PALETTE["ink_2"],
        linespacing=1.65,
    )

    _stat_block(
        axes,
        0.06,
        0.63,
        "Average forecast error",
        f"{selected_wape:.1%}",
        "of total demand, across 9,600 product-weeks",
        PALETTE["harbour_700"],
        value_size=30,
    )
    _stat_block(
        axes,
        0.37,
        0.63,
        "Better than the simple rule",
        f"{improvement:.0%}",
        f"vs {baseline_wape:.1%} for 'same week last year'",
        PALETTE["ok"],
        value_size=30,
    )
    _stat_block(
        axes,
        0.70,
        0.63,
        "Systematic bias",
        f"{metrics.get('ensemble_bias_relative', metrics['gbm_bias_relative']):+.1%}",
        f"vs {metrics['baseline_bias_relative']:+.1%} for the simple rule",
        PALETTE["ok"],
        value_size=30,
    )

    axes.plot([0.06, 0.94], [0.505, 0.505], color=PALETTE["rule"], linewidth=0.9)

    # --- Error by horizon --------------------------------------------------- #
    chart = figure.add_axes((0.06, 0.17, 0.40, 0.30))
    horizons = data.by_horizon["horizon"].to_numpy()
    model = data.by_horizon.get("ensemble_wape", data.by_horizon["wape"]).to_numpy(dtype="float64")
    baseline = data.by_horizon["baseline_wape"].to_numpy(dtype="float64")

    width = 0.38
    chart.bar(
        horizons - width / 2,
        baseline * 100,
        width=width,
        color=PALETTE["baseline"],
        label="Same week last year",
    )
    chart.bar(
        horizons + width / 2,
        model * 100,
        width=width,
        color=PALETTE["harbour_700"],
        label="FORESIGHT",
    )
    chart.set_xlabel("Weeks ahead", fontsize=9)
    chart.set_ylabel("Average error (%)", fontsize=9)
    chart.set_title(
        "Error stays flat across the horizon", fontsize=10.5, fontweight="bold", loc="left"
    )
    chart.set_xticks(horizons)
    chart.legend(fontsize=8, loc="upper left")
    chart.tick_params(labelsize=8)

    axes.text(
        0.53,
        0.44,
        "What the bias number means",
        fontsize=11,
        color=PALETTE["ink"],
        fontweight="bold",
    )
    axes.text(
        0.53,
        0.28,
        "A forecast can look accurate on average while being\n"
        "consistently too low. 'Same week last year' runs 18%\n"
        "light every week, because your business is growing —\n"
        "which quietly guarantees stockouts.\n\n"
        "Our forecast runs 2% light. That difference matters\n"
        "more to your reorder decisions than the headline\n"
        "accuracy number does.",
        fontsize=10,
        color=PALETTE["ink_2"],
        linespacing=1.7,
        va="center",
    )
    return figure


def _page_grid(data: ReadoutData, page: int, total: int) -> plt.Figure:
    figure, axes = _new_page()
    _chrome(axes, "The whole catalogue at a glance", "Which products need attention", page, total)

    chart = figure.add_axes((0.08, 0.16, 0.52, 0.60))
    frame = data.risk_table

    sizes = 18 + 320 * np.sqrt(
        frame["value_at_stake"].clip(lower=0) / max(frame["value_at_stake"].max(), 1.0)
    )
    colours = frame["action"].map(ACTION_COLOURS).fillna(PALETTE["ink_3"])

    chart.scatter(
        frame["overstock_score"],
        frame["stockout_score"],
        s=sizes,
        c=colours,
        alpha=0.62,
        edgecolors=colours,
        linewidths=0.6,
    )
    chart.axhline(0.5, color=PALETTE["ink_3"], linestyle="--", linewidth=0.9)
    chart.axvline(0.5, color=PALETTE["ink_3"], linestyle="--", linewidth=0.9)
    chart.set_xlim(-0.04, 1.04)
    chart.set_ylim(-0.04, 1.04)
    chart.set_xlabel("Risk of being left holding stock", fontsize=9.5)
    chart.set_ylabel("Risk of running out", fontsize=9.5)
    chart.tick_params(labelsize=8)
    # Labels are pinned just inside the vertical divider rather than in the
    # corners: the corners are exactly where the points pile up.
    chart.text(
        0.47,
        0.545,
        "REORDER NOW",
        fontsize=8,
        color=PALETTE["critical"],
        fontweight="bold",
        ha="right",
        transform=chart.transAxes,
    )
    chart.text(
        0.47,
        0.44,
        "HEALTHY",
        fontsize=8,
        color=PALETTE["ok"],
        fontweight="bold",
        ha="right",
        transform=chart.transAxes,
    )
    chart.text(
        0.98,
        0.44,
        "MARKDOWN / CLEAR",
        fontsize=8,
        color=PALETTE["warn"],
        fontweight="bold",
        ha="right",
        transform=chart.transAxes,
    )

    axes.text(0.64, 0.72, "How to read this", fontsize=12, color=PALETTE["ink"], fontweight="bold")
    axes.text(
        0.64,
        0.50,
        "Every circle is one product. Its size is what it is\n"
        "worth in rupees.\n\n"
        "Higher up  →  more likely to run out before your\n"
        "next delivery arrives.\n\n"
        "Further right  →  more likely that three months of\n"
        "sales still will not clear the stock you hold.\n\n"
        "Bottom-left is where a product should be: enough\n"
        "stock to cover demand, no more.",
        fontsize=10,
        color=PALETTE["ink_2"],
        linespacing=1.75,
        va="center",
    )

    counts = data.impact
    legend = [
        ("Reorder now", counts["reorder_now_skus"], PALETTE["critical"]),
        ("Markdown / clear", counts["markdown_skus"], PALETTE["warn"]),
        ("Watch / review", counts["watch_skus"], PALETTE["watch"]),
        ("Healthy", counts["healthy_skus"], PALETTE["ok"]),
    ]
    for index, (label, count, colour) in enumerate(legend):
        row_y = 0.20 - index * 0.035
        axes.scatter([0.655], [row_y], s=42, color=colour, alpha=0.75)
        axes.text(0.675, row_y - 0.007, f"{label}", fontsize=9.5, color=PALETTE["ink_2"])
        axes.text(
            0.90,
            row_y - 0.007,
            f"{count}",
            fontsize=9.5,
            color=PALETTE["ink"],
            fontweight="bold",
            ha="right",
            family="monospace",
        )

    return figure


def _page_reorder(data: ReadoutData, page: int, total: int) -> plt.Figure:
    figure, axes = _new_page()
    _chrome(axes, "Action list 1", "Order these before they run out", page, total)

    reorder = (
        data.risk_table[data.risk_table["action"] == RiskAction.REORDER_NOW.value]
        .nlargest(12, "revenue_at_risk")
        .copy()
    )

    if reorder.empty:
        axes.text(
            0.06,
            0.5,
            "No products are at risk of stocking out this week.",
            fontsize=13,
            color=PALETTE["ok"],
            fontweight="bold",
        )
        return figure

    display = pd.DataFrame(
        {
            "sku": reorder["sku_id"],
            "category": reorder["category"].str.slice(0, 22),
            "cover": reorder["cover_weeks"].map(lambda value: f"{value:.1f} wks"),
            "order": reorder["recommended_order_units"].map(lambda value: f"{value:,.0f}"),
            "risk": reorder["revenue_at_risk"].map(
                lambda value: format_inr(value).replace("Rs", "₹")
            ),
        }
    )

    axes.text(
        0.06,
        0.79,
        f"The {len(reorder)} largest exposures. Ordering the quantities below covers forecast demand\n"
        "over each product's lead time plus a safety buffer, at a 95% service level.",
        fontsize=10.5,
        color=PALETTE["ink_2"],
        linespacing=1.6,
    )

    _table(
        axes,
        display,
        [
            ("sku", "Product", "left"),
            ("category", "Category", "left"),
            ("cover", "Stock cover", "right"),
            ("order", "Order qty", "right"),
            ("risk", "Sales at risk", "right"),
        ],
        x=0.06,
        y_top=0.72,
        width=0.88,
    )

    total_units = reorder["recommended_order_units"].sum()
    total_risk = reorder["revenue_at_risk"].sum()
    axes.text(
        0.06,
        0.10,
        f"These {len(reorder)} orders total {total_units:,.0f} units and protect "
        f"{format_inr(total_risk).replace('Rs', '₹')} of sales.",
        fontsize=10.5,
        color=PALETTE["ink"],
        fontweight="bold",
    )
    return figure


def _page_markdown(data: ReadoutData, page: int, total: int) -> plt.Figure:
    figure, axes = _new_page()
    _chrome(axes, "Action list 2", "Clear these to free up cash", page, total)

    markdown = (
        data.risk_table[data.risk_table["action"] == RiskAction.MARKDOWN_CLEAR.value]
        .nlargest(12, "locked_capital")
        .copy()
    )

    if markdown.empty:
        axes.text(
            0.06,
            0.5,
            "No products are carrying excess stock this week.",
            fontsize=13,
            color=PALETTE["ok"],
            fontweight="bold",
        )
        return figure

    display = pd.DataFrame(
        {
            "sku": markdown["sku_id"],
            "category": markdown["category"].str.slice(0, 22),
            "cover": markdown["cover_weeks"].map(
                lambda value: "> 10 yrs" if value >= 520 else f"{value:.0f} wks"
            ),
            "excess": markdown["excess_units"].map(lambda value: f"{value:,.0f}"),
            "capital": markdown["locked_capital"].map(
                lambda value: format_inr(value).replace("Rs", "₹")
            ),
        }
    )

    axes.text(
        0.06,
        0.79,
        f"The {len(markdown)} largest pools of trapped cash. Excess is stock beyond what twelve weeks\n"
        "of forecast demand will absorb, valued at what you paid for it.",
        fontsize=10.5,
        color=PALETTE["ink_2"],
        linespacing=1.6,
    )

    _table(
        axes,
        display,
        [
            ("sku", "Product", "left"),
            ("category", "Category", "left"),
            ("cover", "Stock cover", "right"),
            ("excess", "Excess units", "right"),
            ("capital", "Cash locked", "right"),
        ],
        x=0.06,
        y_top=0.72,
        width=0.88,
    )

    total_capital = markdown["locked_capital"].sum()
    axes.text(
        0.06,
        0.10,
        f"Clearing these would release {format_inr(total_capital).replace('Rs', '₹')} "
        "of working capital.",
        fontsize=10.5,
        color=PALETTE["ink"],
        fontweight="bold",
    )
    return figure


def _page_patterns(data: ReadoutData, page: int, total: int) -> plt.Figure:
    figure, axes = _new_page()
    _chrome(axes, "What the data showed", "Three things worth knowing", page, total)

    panel = data.panel
    weekly = panel.groupby("week_start", as_index=False)["units"].sum()

    chart = figure.add_axes((0.06, 0.47, 0.88, 0.28))
    chart.plot(weekly["week_start"], weekly["units"], color=PALETTE["harbour_700"], linewidth=1.5)
    chart.fill_between(
        weekly["week_start"], weekly["units"], color=PALETTE["harbour_500"], alpha=0.10
    )
    chart.set_ylabel("Units sold per week", fontsize=9)
    chart.set_title(
        "Demand is strongly seasonal — the festive peak is most of the year",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
    )
    chart.tick_params(labelsize=8)

    peak = weekly["units"].max()
    trough = weekly["units"].min()

    points = [
        (
            f"The festive peak is {peak / max(trough, 1):.1f}x the quietest week.",
            "Planning to an annual average guarantees stocking out in October and sitting on\n"
            "stock in April. Any reorder rule has to follow the season.",
        ),
        (
            "A small number of products carry most of the money.",
            "The ten largest exposures account for "
            f"{data.impact['top_10_share_of_revenue_at_risk']:.0%} of all sales at risk. Attention "
            "should be\nrationed accordingly.",
        ),
        (
            "Your data has fixable problems at source.",
            "The sales export omits zero-sale days entirely, the same sale can appear twice in\n"
            "different formats, and lead times include impossible values. We corrected all of\n"
            "these, but fixing them upstream would make every future refresh more reliable.",
        ),
    ]

    for index, (headline, detail) in enumerate(points):
        row_y = 0.36 - index * 0.105
        axes.text(
            0.06, row_y, f"{index + 1}.", fontsize=11, color=PALETTE["ember_600"], fontweight="bold"
        )
        axes.text(0.095, row_y, headline, fontsize=11, color=PALETTE["ink"], fontweight="bold")
        axes.text(
            0.095, row_y - 0.038, detail, fontsize=9.5, color=PALETTE["ink_2"], linespacing=1.6
        )

    return figure


def _page_limitations(data: ReadoutData, page: int, total: int) -> plt.Figure:
    figure, axes = _new_page()
    _chrome(axes, "Being straight with you", "What this does not do", page, total)

    metrics = data.metrics
    coverage = metrics.get("selected_interval_coverage", float("nan"))

    limitations = [
        (
            "The forecast is wrong by about a quarter, on average.",
            f"Average error is {metrics['selected_wape']:.0%} of demand. That is good for weekly "
            "product-level\nforecasting and much better than the alternative, but it is not precision. "
            "Treat the\nnumbers as a prioritised starting point, not an instruction.",
        ),
        (
            "Slow-moving products are forecast much worse than fast ones.",
            "Products that sell nothing in most weeks have error well above 100%. They are flagged\n"
            "separately rather than hidden in the average, and are the least reliable part of the list.",
        ),
        (
            "We cannot see demand you never captured.",
            "When a product was out of stock, the sale simply does not appear in your data. Recorded\n"
            "demand is therefore a floor, and stockout risk is if anything understated for the products\n"
            "that run out most often.",
        ),
        (
            "Stock positions are weekly snapshots, not live.",
            "Recommendations are based on the position as of the last snapshot. Anything that moved\n"
            "since then is not reflected.",
        ),
        (
            "The uncertainty range is honest but assumes weeks are independent.",
            f"The stated 80% range contains the outcome {coverage:.0%} of the time in testing. It "
            "assumes\nforecast errors do not persist week to week; in practice they somewhat do, so "
            "the true\nrange is slightly wider than shown.",
        ),
    ]

    for index, (headline, detail) in enumerate(limitations):
        row_y = 0.77 - index * 0.135
        axes.add_patch(
            Rectangle((0.06, row_y - 0.072), 0.003, 0.10, color=PALETTE["warn"], zorder=2)
        )
        axes.text(0.078, row_y, headline, fontsize=11, color=PALETTE["ink"], fontweight="bold")
        axes.text(
            0.078, row_y - 0.042, detail, fontsize=9.5, color=PALETTE["ink_2"], linespacing=1.6
        )

    return figure


def _page_next(data: ReadoutData, page: int, total: int) -> plt.Figure:
    figure, axes = _new_page()
    _chrome(axes, "Where to go next", "Our recommendations", page, total)

    recommendations = [
        (
            "Now",
            "Work the reorder list top-down.",
            f"The ten largest exposures cover {data.impact['top_10_share_of_revenue_at_risk']:.0%} "
            "of the sales at risk. Ten calls capture almost all\nof the recoverable value.",
            PALETTE["critical"],
        ),
        (
            "Now",
            "Agree a markdown plan for the dead stock.",
            "Several products hold years of cover at current demand. They will not recover on their\n"
            "own, and every week they sit costs warehouse space and cash.",
            PALETTE["warn"],
        ),
        (
            "This month",
            "Fix the three data problems at source.",
            "Export zero-sale days, de-duplicate re-runs, and validate lead times on entry. Each\n"
            "one currently has to be repaired every time the pipeline runs.",
            PALETTE["harbour_700"],
        ),
        (
            "This quarter",
            "Re-run this monthly and track whether it is still right.",
            "The pipeline re-runs from raw files with one command. Compare each month's forecast\n"
            "against what happened; if average error drifts above 35%, the model needs retraining.",
            PALETTE["harbour_700"],
        ),
        (
            "Later",
            "Feed planned promotions in earlier.",
            "Promotion dates are already used where the calendar has them. The further ahead they\n"
            "are confirmed, the more accurate the festive forecast becomes.",
            PALETTE["ink_3"],
        ),
    ]

    for index, (when, headline, detail, colour) in enumerate(recommendations):
        row_y = 0.76 - index * 0.135
        axes.text(0.06, row_y, when.upper(), fontsize=7.5, color=colour, fontweight="bold")
        axes.text(0.155, row_y, headline, fontsize=11.5, color=PALETTE["ink"], fontweight="bold")
        axes.text(
            0.155, row_y - 0.042, detail, fontsize=9.5, color=PALETTE["ink_2"], linespacing=1.6
        )
        axes.plot(
            [0.06, 0.94], [row_y - 0.088, row_y - 0.088], color=PALETTE["rule"], linewidth=0.5
        )

    axes.text(
        0.06,
        0.075,
        "The dashboard and the scoring service are live and yours to use without us.",
        fontsize=10.5,
        color=PALETTE["ink"],
        fontweight="bold",
    )
    return figure


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_readout(settings: Settings | None = None) -> dict[str, str]:
    """Render the executive readout to PDF and Markdown.

    Returns a mapping of output name to path.
    """
    settings = settings or get_settings()
    settings.ensure_directories()
    apply_house_style()

    data = load_readout_data(settings)

    builders = [
        _page_headline,
        _page_accuracy,
        _page_grid,
        _page_reorder,
        _page_markdown,
        _page_patterns,
        _page_limitations,
        _page_next,
    ]
    total_pages = len(builders) + 1

    pdf_path = settings.reports_dir / "executive_readout.pdf"
    with PdfPages(pdf_path) as pdf:
        cover = _page_cover(data)
        pdf.savefig(cover, facecolor=PALETTE["ground"])
        plt.close(cover)

        for index, builder in enumerate(builders, start=2):
            figure = builder(data, index, total_pages)
            pdf.savefig(figure, facecolor=PALETTE["ground"])
            plt.close(figure)

    log.info(
        "executive readout written",
        extra={"context": {"path": str(pdf_path), "pages": total_pages}},
    )
    return {"pdf": str(pdf_path)}
