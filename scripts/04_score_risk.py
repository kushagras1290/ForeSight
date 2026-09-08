"""Produce the forward forecast and the risk / decisioning table (D4).

Forecasts every SKU forward over the configured horizon from the latest week in
the panel, scores stockout and overstock risk against the most recent inventory
position, and quantifies the rupee exposure.

Outputs to ``artifacts/``:

``forecast.parquet``   forward forecast per SKU per week, with intervals
``risk_table.parquet`` one row per SKU: scores, action, rupee value at stake
``impact.json``        the aggregate figures the executive readout leads with

Usage::

    python scripts/04_score_risk.py
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from foresight.config import get_settings
from foresight.custom_model import load_ensemble
from foresight.exceptions import ForesightError
from foresight.features import (
    FEATURE_SPEC,
    build_base_features,
    build_inference_frame,
    build_weekly_calendar,
)
from foresight.forecast import load_forecaster
from foresight.logging_setup import get_logger
from foresight.risk import aggregate_impact, format_inr, score_risk

log = get_logger("scripts.score_risk")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["auto", "ensemble", "gbm"],
        default="auto",
        help="Which forecast to score with. 'auto' uses the model the backtest selected.",
    )
    arguments = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_directories()

    try:
        panel = pd.read_parquet(settings.processed_dir / "weekly_panel.parquet")
        calendar = pd.read_parquet(settings.processed_dir / "calendar.parquet")
        inventory = pd.read_parquet(settings.processed_dir / "inventory_clean.parquet")
        sku_master = pd.read_parquet(settings.processed_dir / "sku_master_clean.parquet")
    except (FileNotFoundError, OSError) as exc:
        log.error(
            "processed data missing",
            extra={"context": {"error": str(exc), "hint": "run scripts/01_run_pipeline.py"}},
        )
        return 1

    # --- Which model does the backtest say to use? -------------------------- #
    metrics_path = settings.artifacts_dir / "metrics.json"
    selected = "adaptive_ensemble"
    if metrics_path.exists():
        selected = json.loads(metrics_path.read_text(encoding="utf-8")).get(
            "selected_model", "adaptive_ensemble"
        )
    if arguments.model == "ensemble":
        selected = "adaptive_ensemble"
    elif arguments.model == "gbm":
        selected = "lightgbm"

    try:
        # Load the model first and ask *it* which features it needs. Deriving
        # this from configuration instead would let a model trained with drivers
        # be served a frame built without them - a train/serve skew that shows up
        # as a missing-column error at best, and as a silently wrong forecast at
        # worst. The fitted model is the only authority on its own contract.
        if selected == "adaptive_ensemble":
            model = load_ensemble(settings=settings)
            feature_spec = model.gbm.feature_spec
        else:
            model = load_forecaster(settings=settings)
            feature_spec = model.feature_spec

        use_drivers = bool(set(feature_spec.numeric) - set(FEATURE_SPEC.numeric))
        marketing_plan = None
        if use_drivers:
            plan_path = settings.processed_dir / "marketing_clean.parquet"
            if plan_path.exists():
                plan = pd.read_parquet(plan_path)
                marketing_plan = plan if not plan.empty else None

        log.info(
            "serving contract resolved",
            extra={
                "context": {
                    "selected": selected,
                    "features": len(feature_spec.all_features),
                    "drivers": use_drivers,
                    "marketing_plan_rows": 0 if marketing_plan is None else len(marketing_plan),
                }
            },
        )

        weekly_calendar = build_weekly_calendar(calendar)
        base = build_base_features(panel, use_drivers=use_drivers)
        horizons = tuple(range(1, settings.horizon_weeks + 1))
        origin_week = pd.Timestamp(panel["week_start"].max())

        inference = build_inference_frame(
            base,
            weekly_calendar,
            origin_week,
            horizons,
            use_drivers=use_drivers,
            marketing_plan=marketing_plan,
        )
        predictions = model.predict(inference)
    except ForesightError as exc:
        log.error("forecasting failed", extra={"context": {"error": str(exc)}})
        return 1

    forecast = inference[
        ["sku_id", "week_start", "target_week", "horizon", "category", "subcategory"]
    ].copy()
    forecast["prediction"] = predictions["prediction"].to_numpy().round(2)
    forecast["prediction_lower"] = predictions["prediction_lower"].to_numpy().round(2)
    forecast["prediction_upper"] = predictions["prediction_upper"].to_numpy().round(2)
    forecast["model"] = selected
    if "regime" in predictions.columns:
        forecast["regime"] = predictions["regime"].to_numpy()

    forecast = forecast.sort_values(["sku_id", "horizon"]).reset_index(drop=True)

    # --- Risk ---------------------------------------------------------------- #
    try:
        risk_table = score_risk(forecast, inventory, sku_master, settings, panel_features=base)
        impact = aggregate_impact(risk_table)
    except ForesightError as exc:
        log.error("risk scoring failed", extra={"context": {"error": str(exc)}})
        return 1

    forecast.to_parquet(settings.artifacts_dir / "forecast.parquet", index=False)
    risk_table.to_parquet(settings.artifacts_dir / "risk_table.parquet", index=False)

    impact["model"] = selected
    impact["origin_week"] = str(origin_week.date())
    (settings.artifacts_dir / "impact.json").write_text(
        json.dumps(impact, indent=2, default=str), encoding="utf-8"
    )

    log.info(
        "risk and impact written",
        extra={
            "context": {
                "model": selected,
                "origin_week": str(origin_week.date()),
                "skus": impact["total_skus"],
                "reorder_now": impact["reorder_now_skus"],
                "markdown": impact["markdown_skus"],
                "watch": impact["watch_skus"],
                "healthy": impact["healthy_skus"],
                "revenue_at_risk": format_inr(impact["revenue_at_risk_total"]),
                "locked_capital": format_inr(impact["locked_capital_total"]),
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
