"""Build three independent test datasets for the forward forecast (D3 / D5).

WHY THREE, AND WHY THESE THREE
------------------------------
One test set answers one question. These answer three different ones, because
"is the forecast any good?" decomposes into questions with genuinely different
answers:

``steady``
    A year of real demand the model never trained on, exactly as it happened.
    The control - it should reproduce the holdout figure, and if it does not,
    the measurement is wrong rather than the world.

``shock``
    A structural break: one category surges, another collapses, both building
    from the midpoint of the year. No forecast built on history can see this
    coming, so the honest expectation is that accuracy falls. The point is to
    know *how far* it falls and how quickly the intervals stop covering - a model
    that degrades gracefully is usable in a crisis, one that does not is
    dangerous precisely when it matters most.

``promo``
    A heavy promotional calendar - deep discounts on a third of the assortment
    across four bursts, driving multiples of baseline demand. This is the case
    the promotional response model exists for, and the one where a forecast
    trained mostly on unpromoted weeks is most likely to under-call the peak.

Each set is also scored on **just the rows it touched**, not only in aggregate.
A shock confined to two of six categories moves the portfolio number by a few
points however severe it is inside them, so the aggregate figure alone would
understate it to the point of being misleading.

WHAT THE SETS ARE BUILT FROM
----------------------------
All three are built on the **real chronological holdout** - a full year of weeks
the model was never trained on, with the forecasts it actually made for them.
``steady`` is that period untouched, so it scores exactly what the holdout
scores. The other two take the same real base and bend it, so the drop from the
control is attributable entirely to the stress applied and nothing else.

This is deliberate. An earlier version of this script synthesised a future from
each product's seasonal shape plus fresh noise, and the control scored 53%
against a holdout figure of 75%. The gap was not the model failing - it was the
generator inventing demand that differed from what any model would predict *and*
adding a second helping of noise on top. A control that a good forecast cannot
score well on is not a control. Building on real outcomes removes the problem at
the source rather than tuning around it.

WHAT EACH SET CONTAINS
----------------------
``actuals.csv``    sku_id, week_starting, units - the file to upload
``forecast.csv``   the forecast being tested, so the pair can be scored end to
                   end without relying on anything this service stores
``README.md``      what the scenario is, and the score it actually produced

Upload both files together on the dashboard's *Forecast accuracy* tab.

Usage::

    python scripts/09_generate_test_sets.py
    python scripts/09_generate_test_sets.py --seed 7
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from foresight.config import get_settings
from foresight.evaluate import evaluate_actuals
from foresight.exceptions import ForesightError
from foresight.logging_setup import get_logger

log = get_logger("scripts.generate_test_sets")


@dataclass(frozen=True, slots=True)
class Scenario:
    """One test set: a name, what it is for, and how it bends demand."""

    name: str
    title: str
    question: str
    detail: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="steady",
        title="Business as usual",
        question="Does the reported accuracy actually transfer?",
        detail=(
            "The chronological holdout exactly as it happened - a full year of "
            "real demand the model never trained on, against the forecasts it "
            "genuinely made for those weeks. Nothing is altered.\n\n"
            "This is the control. It should reproduce the holdout figure, and if "
            "it does not, something is wrong with the measurement rather than "
            "with the world. It is also the reference the other two sets are "
            "read against: their drop from this number is caused entirely by the "
            "stress applied, because everything else is identical."
        ),
    ),
    Scenario(
        name="shock",
        title="Structural break",
        question="How badly does it fail when the world changes underneath it?",
        detail=(
            "The same real year, with one category made to surge as though a "
            "product went viral or a competitor left the market, and another to "
            "collapse as though supply was disrupted. Both build gradually from "
            "the midpoint of the period, and nothing in the history predicts "
            "either.\n\n"
            "Accuracy is expected to fall, and it should. The useful questions "
            "are by how much, how fast the prediction intervals stop covering, "
            "and whether the failure is visible in the numbers or silent. A model "
            "that degrades gracefully is usable in a crisis; one that does not is "
            "dangerous precisely when it matters most."
        ),
    ),
    Scenario(
        name="promo",
        title="Heavy promotional period",
        question="Does it call the peak when a third of the range goes on offer?",
        detail=(
            "The same real year, with four two-week promotional bursts applied "
            "across a third of the assortment, driving several times baseline "
            "demand.\n\n"
            "A model trained mostly on unpromoted weeks tends to under-call these "
            "peaks, and that shows up as a *negative bias* rather than as "
            "scattered error - the signature to look for is the bias figure "
            "moving sharply while accuracy falls less than you would expect. "
            "This is the case the promotional response model exists for."
        ),
    ),
)


def _base_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """The real holdout, shaped for the scenario modifiers.

    ``actual`` is genuine observed demand and ``prediction`` is the forecast the
    model genuinely made for that week without having seen it. Nothing is
    synthesised here at all.
    """
    frame = scored.loc[
        :,
        [
            "sku_id",
            "category",
            "target_week",
            "horizon",
            "prediction",
            "prediction_lower",
            "prediction_upper",
            "actual",
        ],
    ].copy()
    frame["target_week"] = pd.to_datetime(frame["target_week"])
    return frame.rename(columns={"target_week": "week_starting", "actual": "units"})


def _apply_shock(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One category surges, another collapses, over the second half of the year."""
    out = frame.copy()
    categories = sorted(out["category"].unique())
    if len(categories) < 2:
        return out

    surging, collapsing = rng.choice(categories, size=2, replace=False)
    weeks = sorted(out["week_starting"].unique())
    onset = len(weeks) // 2

    # Ramped rather than stepped, and starting midway. A real shock builds over
    # weeks, and a step change on week one would be far easier to spot than
    # anything that happens in practice.
    def ramp(index: int, rate: float, limit: float) -> float:
        return min(limit, rate * max(0, index - onset))

    surge = {week: 1.0 + ramp(index, 0.06, 0.80) for index, week in enumerate(weeks)}
    collapse = {week: 1.0 - ramp(index, 0.045, 0.60) for index, week in enumerate(weeks)}

    multiplier = np.ones(len(out), dtype="float64")
    is_surging = (out["category"] == surging).to_numpy()
    is_collapsing = (out["category"] == collapsing).to_numpy()
    week_values = out["week_starting"].to_numpy()

    for week in weeks:
        mask = week_values == week
        multiplier[mask & is_surging] = surge[week]
        multiplier[mask & is_collapsing] = collapse[week]

    out["units"] = np.round(out["units"].to_numpy(dtype="float64") * multiplier)
    out.attrs["shock_up"] = str(surging)
    out.attrs["shock_down"] = str(collapsing)
    out.attrs["shock_onset"] = str(pd.Timestamp(weeks[onset]).date())
    out.attrs["focus_mask"] = multiplier != 1.0
    out.attrs["focus_label"] = "the two shocked categories, after the onset"
    return out


def _apply_promotion(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Deep discounts on a third of the assortment, in four separate bursts."""
    out = frame.copy()
    skus = sorted(out["sku_id"].unique())
    promoted = set(rng.choice(skus, size=max(1, len(skus) // 3), replace=False))

    weeks = sorted(out["week_starting"].unique())
    # Four two-week events spread through the year rather than one long sale,
    # which is how a retail promotional calendar is actually shaped.
    burst_starts = np.linspace(4, len(weeks) - 6, num=4).astype(int)
    promo_weeks = {weeks[start + offset] for start in burst_starts for offset in (0, 1)}

    # Uplift drawn per SKU so the event has a spread of depths rather than one
    # discount applied uniformly, which no real promotion looks like.
    uplift = {sku: float(rng.uniform(2.0, 4.5)) for sku in promoted}

    multiplier = np.array(
        [
            uplift.get(str(sku), 1.0) if (week in promo_weeks and str(sku) in promoted) else 1.0
            for sku, week in zip(out["sku_id"], out["week_starting"], strict=True)
        ],
        dtype="float64",
    )
    out["units"] = np.round(out["units"].to_numpy(dtype="float64") * multiplier)
    out.attrs["promoted_skus"] = len(promoted)
    out.attrs["promo_weeks"] = len(promo_weeks)
    out.attrs["focus_mask"] = multiplier != 1.0
    out.attrs["focus_label"] = "the promoted products, during their promotion weeks"
    return out


def _score(actuals: pd.DataFrame, forecast: pd.DataFrame, panel: pd.DataFrame) -> dict[str, float]:
    """Score a generated set against the forecast it ships with."""
    result = evaluate_actuals(
        actuals.loc[:, ["sku_id", "week_starting", "units"]],
        forecast=forecast,
        backtest=pd.DataFrame(),
        panel=panel,
        holdout=None,
    )
    return {
        "rows_matched": float(result.rows_matched),
        "wape": float(result.model.wape),
        "accuracy": 1.0 - float(result.model.wape),
        "bias_relative": float(result.model.bias_relative),
        "interval_coverage": float(result.interval_coverage),
        "baseline_wape": float(result.baseline.wape)
        if result.baseline is not None
        else float("nan"),
    }


def _focus_score(frame: pd.DataFrame, mask: np.ndarray) -> dict[str, float] | None:
    """Score just the rows a scenario actually touched.

    An aggregate figure dilutes a localised failure: a shock confined to two of
    six categories moves the portfolio number by a few points however severe it
    is inside those categories. Reporting the affected rows separately is what
    makes the test set say something useful rather than something arithmetically
    inevitable.
    """
    subset = frame.loc[mask]
    if subset.empty:
        return None
    actual = subset["units"].to_numpy(dtype="float64")
    prediction = subset["prediction"].to_numpy(dtype="float64")
    total = float(np.abs(actual).sum())
    if total <= 0.0:
        return None

    covered = (actual >= subset["prediction_lower"].to_numpy()) & (
        actual <= subset["prediction_upper"].to_numpy()
    )
    error = float(np.abs(prediction - actual).sum())
    return {
        "rows": float(len(subset)),
        "wape": error / total,
        "accuracy": 1.0 - error / total,
        "bias_relative": float((prediction - actual).sum() / total),
        "interval_coverage": float(covered.mean()),
    }


def _write_readme(
    directory: Path,
    scenario: Scenario,
    score: dict[str, float],
    notes: str,
    control: dict[str, float] | None,
    focus: dict[str, float] | None,
    focus_label: str,
) -> None:
    """A README stating what the set is and the score it actually produced."""
    lines = [
        f"# Test set: {scenario.title}",
        "",
        f"**{scenario.question}**",
        "",
        scenario.detail,
        "",
        "## Files",
        "",
        "| File | What it is |",
        "|---|---|",
        "| `actuals.csv` | What demand turned out to be. |",
        "| `forecast.csv` | The forecast under test, so the pair scores end to end. |",
        "",
        "Upload **both files together** on the dashboard's *Forecast accuracy* tab.",
        "",
        "## What it scored",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Accuracy | **{score['accuracy']:.1%}** |",
        f"| WAPE | {score['wape']:.4f} |",
        f"| Seasonal-naive WAPE | {score['baseline_wape']:.4f} |",
        f"| Bias | {score['bias_relative']:+.1%} |",
        f"| Interval coverage | {score['interval_coverage']:.1%} (80% stated) |",
        f"| Product-weeks scored | {int(score['rows_matched']):,} |",
        "",
    ]

    if control is not None:
        delta_points = (score["accuracy"] - control["accuracy"]) * 100.0
        bias_delta = score["bias_relative"] - control["bias_relative"]
        lines += [
            "### Against the control",
            "",
            f"The `steady` set scores **{control['accuracy']:.1%}** on the same forecast and the",
            f"same weeks. This scenario costs **{delta_points:+.1f} points** of accuracy across",
            f"the whole portfolio and moves bias by **{bias_delta:+.1%}**.",
            "",
        ]

    if focus is not None and focus_label:
        lines += [
            "### Where the damage actually is",
            "",
            "An aggregate figure dilutes a localised failure. Scored only on",
            f"{focus_label}:",
            "",
            "| Measure | Value |",
            "|---|---:|",
            f"| Accuracy | **{focus['accuracy']:.1%}** |",
            f"| Bias | {focus['bias_relative']:+.1%} |",
            f"| Interval coverage | {focus['interval_coverage']:.1%} |",
            f"| Product-weeks affected | {int(focus['rows']):,} |",
            "",
            "That is the number to quote about this scenario. The portfolio figure",
            "above is real, but it mostly measures how much of the assortment was",
            "left alone.",
            "",
        ]

    if notes:
        lines += ["## How this set was bent", "", notes, ""]

    lines += [
        "## How it was built",
        "",
        "The base is the project's **chronological holdout** - a year of real demand",
        "the model never trained on, with the forecasts it genuinely made for those",
        "weeks. `steady` is that period untouched; the other sets take the same real",
        "base and bend it, so any difference from the control is caused by the stress",
        "and nothing else.",
        "",
        "Regenerate with `python scripts/09_generate_test_sets.py`.",
        "",
    ]
    (directory / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="Override the configured seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write the sets into. Defaults to data/testsets/.",
    )
    arguments = parser.parse_args(argv)

    settings = get_settings()
    seed = arguments.seed if arguments.seed is not None else settings.random_seed
    out_root = arguments.out or (settings.data_dir / "testsets")

    panel_path = settings.processed_dir / "weekly_panel.parquet"
    scored_path = settings.artifacts_dir / "holdout_scored.parquet"
    for path, hint in (
        (panel_path, "run scripts/01_run_pipeline.py"),
        (scored_path, "run scripts/07_holdout_split.py"),
    ):
        if not path.exists():
            log.error("input missing", extra={"context": {"path": str(path), "hint": hint}})
            return 1

    panel = pd.read_parquet(panel_path)
    scored = pd.read_parquet(scored_path)
    base = _base_frame(scored)

    target_weeks = sorted(base["week_starting"].unique())
    log.info(
        "generating test sets",
        extra={
            "context": {
                "scenarios": [scenario.name for scenario in SCENARIOS],
                "source": "chronological holdout (real outcomes)",
                "weeks": len(target_weeks),
                "first_week": str(pd.Timestamp(target_weeks[0]).date()),
                "last_week": str(pd.Timestamp(target_weeks[-1]).date()),
                "rows": len(base),
                "skus": int(base["sku_id"].nunique()),
                "seed": seed,
            }
        },
    )

    forecast_out = base.loc[
        :,
        [
            "sku_id",
            "week_starting",
            "horizon",
            "prediction",
            "prediction_lower",
            "prediction_upper",
        ],
    ].rename(columns={"week_starting": "target_week"})

    summaries: list[tuple[Scenario, dict[str, float]]] = []

    for index, scenario in enumerate(SCENARIOS):
        # A separate stream per scenario, so adding or reordering scenarios does
        # not change the sets that were already generated.
        rng = np.random.default_rng(seed + 1_000 * (index + 1))
        try:
            frame = base.copy()
            notes = ""

            if scenario.name == "shock":
                frame = _apply_shock(frame, rng)
                notes = (
                    f"`{frame.attrs['shock_up']}` ramps up to +80% and "
                    f"`{frame.attrs['shock_down']}` falls to -60%, both starting "
                    f"{frame.attrs['shock_onset']} and building week by week. "
                    "Every other category is left exactly as it happened."
                )
            elif scenario.name == "promo":
                frame = _apply_promotion(frame, rng)
                notes = (
                    f"{frame.attrs['promoted_skus']} of {frame['sku_id'].nunique()} products are "
                    f"promoted across {frame.attrs['promo_weeks']} weeks, in four two-week "
                    "bursts, each at an uplift drawn between 2.0x and 4.5x."
                )

            actuals = frame.loc[:, ["sku_id", "week_starting", "units"]].copy()
            actuals["units"] = actuals["units"].clip(lower=0.0).round().astype("int64")

            score = _score(actuals, forecast_out, panel)
            focus_mask = frame.attrs.get("focus_mask")
            focus = _focus_score(frame, focus_mask) if focus_mask is not None else None
            focus_label = frame.attrs.get("focus_label", "")
        except ForesightError as exc:
            log.error(
                "scenario failed",
                extra={"context": {"scenario": scenario.name, "error": str(exc)}},
            )
            return 1

        directory = out_root / scenario.name
        directory.mkdir(parents=True, exist_ok=True)

        actuals.assign(week_starting=actuals["week_starting"].dt.strftime("%Y-%m-%d")).to_csv(
            directory / "actuals.csv", index=False
        )
        forecast_out.assign(target_week=forecast_out["target_week"].dt.strftime("%Y-%m-%d")).to_csv(
            directory / "forecast.csv", index=False
        )
        # The control is generated first, so later scenarios can be reported
        # relative to it rather than in isolation.
        control = summaries[0][1] if summaries else None
        _write_readme(directory, scenario, score, notes, control, focus, focus_label)

        summaries.append((scenario, score))
        log.info(
            "test set written",
            extra={
                "context": {
                    "scenario": scenario.name,
                    "path": str(directory),
                    "rows": len(actuals),
                    "accuracy": round(score["accuracy"], 4),
                    "bias": round(score["bias_relative"], 4),
                    "coverage": round(score["interval_coverage"], 4),
                }
            },
        )

    # --- An index, so the three are read as a set rather than in isolation --- #
    index_lines = [
        "# Test datasets",
        "",
        "Three independent test sets for the forward forecast, each answering a",
        "different question. Upload `actuals.csv` and `forecast.csv` together on the",
        "dashboard's *Forecast accuracy* tab.",
        "",
        "| Set | Question | Accuracy | Bias | Coverage |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario, score in summaries:
        index_lines.append(
            f"| [`{scenario.name}`]({scenario.name}/README.md) | {scenario.question} | "
            f"**{score['accuracy']:.1%}** | {score['bias_relative']:+.1%} | "
            f"{score['interval_coverage']:.0%} |"
        )
    index_lines += [
        "",
        "The spread between them is the point. A forecast that scores the same on",
        "all three is not being tested hard enough; one that collapses on `shock`",
        "is behaving exactly as a history-based model should, and the size of the",
        "collapse is what tells you how much headroom to leave when planning.",
        "",
        "Regenerate with `python scripts/09_generate_test_sets.py`.",
        "",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.md").write_text("\n".join(index_lines), encoding="utf-8")

    log.info(
        "test sets complete",
        extra={
            "context": {
                "path": str(out_root),
                "sets": len(summaries),
                "accuracy": {
                    scenario.name: round(score["accuracy"], 4) for scenario, score in summaries
                },
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
