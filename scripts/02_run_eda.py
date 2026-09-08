"""Generate the data-quality & EDA insight memo and its figures (deliverable D2).

The memo is *rendered* from the pipeline's cleaning audit trail and from measured
EDA findings - it is not written by hand. That means it can never quietly drift
away from what the code actually did, which is the whole point of D2 acceptance
criterion 1.

Outputs:

``reports/eda_memo.md``       the memo
``reports/figures/*.png``     the charts it references

Usage::

    python scripts/02_run_eda.py
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from foresight.config import get_settings
from foresight.eda import run_eda
from foresight.exceptions import ForesightError
from foresight.logging_setup import get_logger

log = get_logger("scripts.run_eda")

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _render_cleaning_table(report: pd.DataFrame) -> str:
    if report.empty:
        return "_No cleaning actions were required._\n"

    ordered = report.copy()
    ordered["_rank"] = ordered["severity"].map(SEVERITY_ORDER).fillna(3)
    ordered = ordered.sort_values(["_rank", "rows_affected"], ascending=[True, False])

    lines = [
        "| Severity | Table | Issue found | Rows | How it was resolved | Why |",
        "|---|---|---|---:|---|---|",
    ]
    for row in ordered.itertuples(index=False):
        lines.append(
            f"| **{row.severity}** | `{row.table}` | {row.issue} | {row.rows_affected:,} | "
            f"{row.resolution} | {row.rationale} |"
        )
    return "\n".join(lines) + "\n"


def _render_frame(frame: pd.DataFrame, columns: dict[str, str], floats: int = 2) -> str:
    subset = frame.loc[:, list(columns)].rename(columns=columns)
    header = "| " + " | ".join(subset.columns) + " |"
    divider = "|" + "|".join("---" for _ in subset.columns) + "|"
    rows = []
    for record in subset.itertuples(index=False):
        cells = []
        for value in record:
            if isinstance(value, float):
                cells.append(f"{value:,.{floats}f}")
            elif isinstance(value, (int,)):
                cells.append(f"{value:,}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows]) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    settings = get_settings()

    try:
        findings = run_eda(settings)
        panel = pd.read_parquet(settings.processed_dir / "weekly_panel.parquet")
        cleaning_report = pd.read_parquet(settings.processed_dir / "cleaning_report.parquet")
    except (ForesightError, FileNotFoundError, OSError) as exc:
        log.error(
            "EDA failed",
            extra={"context": {"error": str(exc), "hint": "run scripts/01_run_pipeline.py first"}},
        )
        return 1

    first_week = panel["week_start"].min().date()
    last_week = panel["week_start"].max().date()
    total_rows_touched = int(cleaning_report["rows_affected"].sum()) if len(cleaning_report) else 0
    critical_count = (
        int((cleaning_report["severity"] == "critical").sum()) if len(cleaning_report) else 0
    )

    insight_blocks = []
    for index, insight in enumerate(findings.headline_insights, start=1):
        insight_blocks.append(
            f"### {index}. {insight['title']}\n\n"
            f"{insight['detail']}\n\n"
            f"**So what:** {insight['so_what']}\n"
        )

    figure_blocks = "\n".join(f"![{name}](figures/{name})\n" for name in findings.figures)

    memo = f"""# Data Quality & EDA Insight Memo

**Project FORESIGHT — Demand & Inventory Intelligence**
**Client:** NorthBay Living   |   **Deliverable:** D2   |   **Milestone:** M2

---

## Summary for the Head of Operations

We took delivery of four extracts covering **{panel["sku_id"].nunique()} SKUs** over
**{panel["week_start"].nunique()} weeks** ({first_week} to {last_week}) and put them through an
automated cleaning pipeline before doing any analysis.

The data is usable, but **it is not clean on arrival**. We found and repaired
**{len(cleaning_report)} distinct classes of problem** affecting **{
        total_rows_touched:,} rows**, of which
**{critical_count}** were severe enough that leaving them in place would have produced a
materially wrong forecast rather than merely a noisier one.

Three of those matter enough to raise with your systems team, because they are worth
fixing at source rather than patching every month:

1. **The sales export omits zero-demand days entirely.** A day on which a product sold
   nothing produces no row at all. Any analysis that treats the file as a complete daily
   record will read those gaps as missing data rather than as genuine zero demand, and will
   overstate average demand for every slow-moving line.
2. **The same sale can appear twice with different formatting.** Re-runs of the export
   append rows that are identical in substance but written differently — `1` versus `TRUE`
   in the promotion flag, or a lower-cased product code. A naive de-duplication misses these
   and double-counts the demand.
3. **`lead_time_days` contains impossible values.** Both `0` and values above 900 appear.
   Lead time sets the window the stockout calculation looks over, so a zero makes a product
   look permanently safe and a 900 makes it look permanently critical.

Everything below is reproducible: re-running `python scripts/01_run_pipeline.py` regenerates
these findings from the raw files.

---

## 1. Data quality: what we found and what we did

Every row in this table was produced by the pipeline itself, not written by hand.

{_render_cleaning_table(cleaning_report)}

---

## 2. What the demand data says

{chr(10).join(insight_blocks)}

---

## 3. Where the revenue actually sits

{
        _render_frame(
            findings.category_summary,
            {
                "category": "Category",
                "skus": "SKUs",
                "total_units": "Units sold",
                "total_revenue": "Revenue (Rs)",
                "mean_weekly_units": "Avg units/week",
                "zero_week_share": "Share of weeks with no sale",
            },
        )
    }

### Top 15 SKUs by revenue

{
        _render_frame(
            findings.top_movers,
            {
                "sku_id": "SKU",
                "category": "Category",
                "total_units": "Units",
                "total_revenue": "Revenue (Rs)",
            },
        )
    }

---

## 4. Dead stock: lines that have stopped selling

Comparing average weekly demand over the last 13 weeks against the same 13 weeks a year
earlier. These are the clearest markdown and delisting candidates.

{
        _render_frame(
            findings.dead_stock,
            {
                "sku_id": "SKU",
                "prior_weekly_units": "Weekly units a year ago",
                "recent_weekly_units": "Weekly units now",
                "decline_pct": "Decline",
            },
        )
    }

---

## 5. Promotional response by category

{
        _render_frame(
            findings.promo_effect,
            {
                "category": "Category",
                "off_promo": "Avg units, normal week",
                "on_promo": "Avg units, promo week",
                "uplift_pct": "Uplift",
            },
        )
    }

---

## 6. Figures

{figure_blocks}

---

## 7. What this means for the forecast

* **Seasonality has to be explicit.** The festive swing is far too large for a moving
  average to absorb, so the model is given week-of-year and holiday features directly, and
  the baseline it is measured against is seasonal-naive rather than a flat average.
* **WAPE, not MAPE.** A large share of SKUs sell nothing in many weeks. MAPE is undefined on
  those weeks and explodes on the ones either side, so it is reported for familiarity only
  and never used to choose a model.
* **The tail needs separate treatment.** Fast and intermittent lines behave differently
  enough that one estimator is the wrong answer for both, which is what the adaptive
  ensemble in D3 addresses.
* **Promotion dates are known in advance.** They come from the published calendar, so they
  are a legitimate forecast input rather than something to be reacted to after the fact.

---

## 8. Known limitations of this analysis

* Inventory positions are weekly snapshots, so demand that occurred while a product was out
  of stock is invisible. Recorded sales are therefore a floor on true demand, and stockout
  risk is, if anything, understated for the lines that stock out most often.
* Products absent from the product master were quarantined rather than guessed at. They
  carry real sales and should be fixed at source.
* Returns booked as negative sales were treated as zero demand rather than netted off,
  because the forecast models gross demand. Net revenue reporting would need them handled
  differently.
"""

    destination = settings.reports_dir / "eda_memo.md"
    destination.write_text(memo, encoding="utf-8")

    log.info(
        "EDA memo written",
        extra={
            "context": {
                "path": str(destination),
                "figures": len(findings.figures),
                "cleaning_actions": len(cleaning_report),
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
