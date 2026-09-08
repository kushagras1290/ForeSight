"""Rolling-origin evaluation of all three custom models on identical folds.

Each model is trained fresh at every origin on data available at that origin, and
all of them are scored on exactly the same test rows, so the comparison between
them is like for like and the numbers can be quoted without a caveat.

What is measured::

    seasonal_naive        the contractual baseline (brief section 7.1)
    lightgbm              the global gradient-boosted model
    adaptive_ensemble     custom model 1 - regime-routed blend
    reconciled            custom model 2 - MinT applied on top of the ensemble
    cold_start            custom model 3 - scored on young SKUs only, where it
                          is the only model with anything to say

Outputs ``artifacts/hierarchy_metrics.json`` and the per-row predictions, plus a
fitted reconciler and cold-start model refitted on the full history.

Usage::

    python scripts/08_hierarchy_backtest.py
    python scripts/08_hierarchy_backtest.py --folds 3      # quicker while iterating
    python scripts/08_hierarchy_backtest.py --no-drivers   # four Appendix A tables only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from foresight.backtest import select_origins
from foresight.coldstart import ColdStartModel, fit_cold_start
from foresight.config import get_settings
from foresight.custom_model import train_ensemble
from foresight.exceptions import ForesightError
from foresight.features import (
    build_base_features,
    build_supervised_frame,
    build_weekly_calendar,
    resolve_feature_spec,
)
from foresight.forecast import train_forecaster
from foresight.hierarchy import (
    CATEGORY_PREFIX,
    SUBCATEGORY_PREFIX,
    TOTAL_NODE,
    aggregate_panel,
    build_hierarchy,
)
from foresight.logging_setup import get_logger
from foresight.metrics import evaluate_forecast, wape
from foresight.reconcile import RECONCILER_FILENAME, train_reconciler

log = get_logger("scripts.hierarchy_backtest")

COLD_START_FILENAME = "cold_start.joblib"

#: A SKU is "young" while its own history is too short for lag features to carry
#: real information. This is the population cold-start exists to serve, and the
#: only population it is scored on - claiming credit for mature SKUs it never
#: influences would misrepresent it.
YOUNG_SKU_MAX_AGE_WEEKS = 26


def _load_marketing_plan(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    plan = pd.read_parquet(path)
    return plan if not plan.empty else None


def _bottom_up_aggregates(frame: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """Roll SKU-level ensemble forecasts up to every aggregate node.

    This is the free alternative MinT has to beat: plain summation, with no
    aggregate model fitted at all.
    """
    labelled = frame.merge(
        sku_master.loc[:, ["sku_id", "category", "subcategory"]].rename(
            columns={"category": "master_category"}
        ),
        on="sku_id",
        how="left",
    )
    keys = ["week_start", "horizon"]
    parts: list[pd.DataFrame] = []

    total = labelled.groupby(keys, as_index=False)["ensemble"].sum()
    total["node"] = TOTAL_NODE
    parts.append(total)

    by_category = labelled.groupby([*keys, "master_category"], as_index=False)["ensemble"].sum()
    by_category["node"] = CATEGORY_PREFIX + by_category["master_category"].astype(str)
    parts.append(by_category.drop(columns="master_category"))

    by_sub = labelled.groupby([*keys, "master_category", "subcategory"], as_index=False)[
        "ensemble"
    ].sum()
    by_sub["node"] = (
        SUBCATEGORY_PREFIX
        + by_sub["master_category"].astype(str)
        + "|"
        + by_sub["subcategory"].astype(str)
    )
    parts.append(by_sub.drop(columns=["master_category", "subcategory"]))

    rolled = pd.concat(parts, ignore_index=True)
    return rolled.rename(columns={"ensemble": "bottom_up"}).loc[
        :, ["node", "week_start", "horizon", "bottom_up"]
    ]


def _score(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    metrics = evaluate_forecast(actual, prediction)
    return {
        "wape": round(metrics.wape, 6),
        "accuracy": round(1.0 - metrics.wape, 6),
        "bias_relative": round(metrics.bias_relative, 6),
        "mae": round(metrics.mae, 6),
        "n": int(metrics.n_observations),
    }


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 - a linear pipeline reads best flat
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=None, help="Origins to evaluate.")
    parser.add_argument(
        "--no-drivers",
        action="store_true",
        help="Forecast from the four Appendix A extracts alone.",
    )
    arguments = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_directories()

    panel_path = settings.processed_dir / "weekly_panel.parquet"
    calendar_path = settings.processed_dir / "calendar.parquet"
    sku_path = settings.processed_dir / "sku_master_clean.parquet"
    for path in (panel_path, calendar_path, sku_path):
        if not path.exists():
            log.error(
                "processed data missing",
                extra={"context": {"path": str(path), "hint": "run scripts/01_run_pipeline.py"}},
            )
            return 1

    panel = pd.read_parquet(panel_path)
    calendar = pd.read_parquet(calendar_path)
    sku_master = pd.read_parquet(sku_path)
    weekly_calendar = build_weekly_calendar(calendar)
    horizons = tuple(range(1, settings.horizon_weeks + 1))

    marketing_plan = _load_marketing_plan(settings.processed_dir / "marketing_clean.parquet")
    use_drivers = settings.use_drivers and not arguments.no_drivers and marketing_plan is not None
    feature_spec = resolve_feature_spec(use_drivers=use_drivers)

    try:
        # --- Structure and both levels of features -------------------------- #
        hierarchy = build_hierarchy(sku_master)
        aggregates = aggregate_panel(panel, hierarchy)

        bottom_supervised = build_supervised_frame(
            build_base_features(panel, use_drivers=use_drivers),
            weekly_calendar,
            horizons,
            use_drivers=use_drivers,
            marketing_plan=marketing_plan,
        )
        # The aggregate nodes have no media plan of their own, so their
        # target-week driver features stay neutral. Their advantage comes from
        # volume, not from extra columns.
        aggregate_supervised = build_supervised_frame(
            build_base_features(aggregates, use_drivers=use_drivers),
            weekly_calendar,
            horizons,
            use_drivers=use_drivers,
            marketing_plan=None,
        )

        origins = select_origins(
            np.sort(bottom_supervised["week_start"].unique()),
            horizon=settings.horizon_weeks,
            folds=arguments.folds or settings.backtest_folds,
            step=settings.backtest_step_weeks,
            min_train_weeks=settings.min_train_weeks,
        )
    except ForesightError as exc:
        log.error("setup failed", extra={"context": {"error": str(exc)}})
        return 1

    log.info(
        "hierarchy backtest starting",
        extra={
            "context": {
                "folds": len(origins),
                "nodes": hierarchy.n_nodes,
                "drivers": use_drivers,
                "features": len(feature_spec.all_features),
            }
        },
    )

    fold_frames: list[pd.DataFrame] = []
    shrinkages: list[float] = []
    coherence_errors: list[float] = []
    blends: list[float] = []
    level_checks: list[pd.DataFrame] = []

    for fold_index, origin in enumerate(origins, start=1):
        bottom_train = bottom_supervised[bottom_supervised["target_week"] <= origin]
        bottom_test = bottom_supervised[bottom_supervised["week_start"] == origin]
        aggregate_train = aggregate_supervised[aggregate_supervised["target_week"] <= origin]
        aggregate_test = aggregate_supervised[aggregate_supervised["week_start"] == origin]
        panel_to_date = panel[panel["week_start"] <= origin]

        if bottom_train.empty or bottom_test.empty or aggregate_test.empty:
            log.warning("skipping empty fold", extra={"context": {"origin": str(origin.date())}})
            continue

        try:
            # --- Baseline and custom model 1 ------------------------------- #
            gbm = train_forecaster(bottom_train, settings, feature_spec=feature_spec)
            ensemble = train_ensemble(
                bottom_train, panel, settings, gbm=gbm, feature_spec=feature_spec
            )
            gbm_predictions = gbm.predict(bottom_test)
            ensemble_predictions = ensemble.predict(bottom_test)

            frame = bottom_test.loc[
                :,
                [
                    "sku_id",
                    "week_start",
                    "target_week",
                    "horizon",
                    "category",
                    "weeks_since_launch",
                    "y",
                ],
            ].copy()
            frame["fold"] = fold_index
            frame["gbm"] = gbm_predictions["prediction"].to_numpy()
            frame["ensemble"] = ensemble_predictions["prediction"].to_numpy()
            frame["regime"] = ensemble_predictions["regime"].to_numpy()

            # --- Custom model 2: reconcile against the aggregate levels ----- #
            # The residual harvest must use the same kind of model whose
            # forecasts are being reconciled, or the error covariance describes
            # the wrong errors and the reconciliation optimises for them.
            def harvest_with_ensemble(train_rows: pd.DataFrame) -> object:
                harvest_gbm = train_forecaster(train_rows, settings, feature_spec=feature_spec)
                return train_ensemble(
                    train_rows, panel, settings, gbm=harvest_gbm, feature_spec=feature_spec
                )

            reconciler = train_reconciler(
                bottom_train,
                aggregate_train,
                hierarchy,
                settings,
                feature_spec=feature_spec,
                bottom_predictor=harvest_with_ensemble,
            )
            aggregate_predictions = reconciler.predict_aggregates(aggregate_test)
            reconciled = reconciler.reconcile(
                frame.assign(prediction=frame["ensemble"]), aggregate_predictions
            )
            frame["reconciled"] = reconciled["reconciled"].to_numpy()
            shrinkages.append(reconciler.shrinkage)
            coherence_errors.append(reconciler.coherence_check(reconciled))
            blends.append(float(np.mean(list(reconciler.blend.values()))))

            # The diagnostic that decides whether reconciliation has anything to
            # work with. A lower category-level WAPE is NOT by itself evidence:
            # summing k independent SKU forecasts shrinks relative error by
            # sqrt(k) for free. MinT only helps if the *independently forecast*
            # category beats that free bottom-up sum, so measure exactly that.
            aggregate_truth = aggregate_test.loc[
                :, ["sku_id", "week_start", "horizon", "y"]
            ].rename(columns={"sku_id": "node", "y": "actual"})
            direct = aggregate_predictions.merge(
                aggregate_truth, on=["node", "week_start", "horizon"], how="inner"
            )
            bottom_up_totals = _bottom_up_aggregates(frame, sku_master)
            comparison = direct.merge(
                bottom_up_totals, on=["node", "week_start", "horizon"], how="inner"
            )
            if not comparison.empty:
                truth = comparison["actual"].to_numpy(dtype="float64")
                level_checks.append(
                    pd.DataFrame(
                        {
                            "node": comparison["node"],
                            "level": comparison["node"].map(
                                dict(zip(hierarchy.nodes, hierarchy.levels, strict=True))
                            ),
                            "actual": truth,
                            "direct": comparison["prediction"].to_numpy(dtype="float64"),
                            "bottom_up": comparison["bottom_up"].to_numpy(dtype="float64"),
                        }
                    )
                )

            # --- Custom model 3: cold start -------------------------------- #
            cold_start = fit_cold_start(panel_to_date)
            frame["cold_start"] = cold_start.predict(frame, panel_to_date)

        except ForesightError as exc:
            log.error(
                "fold failed",
                extra={
                    "context": {"fold": fold_index, "origin": str(origin.date()), "error": str(exc)}
                },
            )
            return 1

        fold_frames.append(frame)
        actual = frame["y"].to_numpy(dtype="float64")
        log.info(
            "fold complete",
            extra={
                "context": {
                    "fold": fold_index,
                    "origin": str(origin.date()),
                    "ensemble_wape": round(wape(actual, frame["ensemble"].to_numpy()), 4),
                    "reconciled_wape": round(wape(actual, frame["reconciled"].to_numpy()), 4),
                }
            },
        )

    if not fold_frames:
        log.error("every fold was empty")
        return 1

    predictions = pd.concat(fold_frames, ignore_index=True)
    actual = predictions["y"].to_numpy(dtype="float64")

    # --- Headline comparison ------------------------------------------------ #
    summary: dict[str, Any] = {
        "folds": len(fold_frames),
        "test_observations": int(len(predictions)),
        "uses_drivers": use_drivers,
        "n_features": len(feature_spec.all_features),
        "hierarchy": {
            "nodes": hierarchy.n_nodes,
            "bottom": hierarchy.n_bottom,
            "levels": list(dict.fromkeys(hierarchy.levels)),
        },
        "overall": {
            name: _score(actual, predictions[name].to_numpy(dtype="float64"))
            for name in ("gbm", "ensemble", "reconciled")
        },
        "mint_shrinkage_mean": round(float(np.mean(shrinkages)), 6) if shrinkages else None,
        "mint_blend_mean": round(float(np.mean(blends)), 6) if blends else None,
        "coherence_max_error": round(float(np.max(coherence_errors)), 9)
        if coherence_errors
        else None,
    }

    # --- Does an independently-forecast aggregate beat plain summation? ----- #
    if level_checks:
        checks = pd.concat(level_checks, ignore_index=True)
        by_level: dict[str, dict[str, float]] = {}
        for level, group in checks.groupby("level", sort=True):
            truth = group["actual"].to_numpy(dtype="float64")
            direct_wape = wape(truth, group["direct"].to_numpy(dtype="float64"))
            bottom_up_wape = wape(truth, group["bottom_up"].to_numpy(dtype="float64"))
            by_level[str(level)] = {
                "direct_forecast_wape": round(direct_wape, 6),
                "bottom_up_sum_wape": round(bottom_up_wape, 6),
                "direct_advantage": round((bottom_up_wape - direct_wape) / bottom_up_wape, 6)
                if bottom_up_wape > 0
                else None,
                "rows": int(len(group)),
            }
        summary["aggregate_forecast_vs_bottom_up"] = by_level

    ensemble_wape = summary["overall"]["ensemble"]["wape"]
    reconciled_wape = summary["overall"]["reconciled"]["wape"]
    summary["reconciliation_gain_vs_ensemble"] = (
        round((ensemble_wape - reconciled_wape) / ensemble_wape, 6) if ensemble_wape > 0 else None
    )

    # --- Cold start, on the population it exists for ------------------------ #
    young = predictions[predictions["weeks_since_launch"] <= YOUNG_SKU_MAX_AGE_WEEKS]
    if not young.empty and young["cold_start"].notna().any():
        scored = young[young["cold_start"].notna()]
        young_actual = scored["y"].to_numpy(dtype="float64")
        summary["cold_start"] = {
            "population": f"weeks_since_launch <= {YOUNG_SKU_MAX_AGE_WEEKS}",
            "rows": int(len(scored)),
            "share_of_test_rows": round(len(scored) / len(predictions), 6),
            "units_share": round(float(scored["y"].sum() / predictions["y"].sum()), 6)
            if predictions["y"].sum() > 0
            else None,
            "cold_start": _score(young_actual, scored["cold_start"].to_numpy(dtype="float64")),
            "gbm": _score(young_actual, scored["gbm"].to_numpy(dtype="float64")),
            "ensemble": _score(young_actual, scored["ensemble"].to_numpy(dtype="float64")),
            "reconciled": _score(young_actual, scored["reconciled"].to_numpy(dtype="float64")),
        }
    else:
        summary["cold_start"] = {"rows": 0, "note": "no young SKUs in the test folds"}

    # Cold start earns its place only where the alternative has nothing to use.
    # Broken out by age so the claim is scoped to the population it holds for,
    # rather than averaged across SKUs whose own lags are already informative.
    bands = []
    for label, low, high in (
        ("0-4 weeks", 0, 4),
        ("5-8 weeks", 5, 8),
        ("9-13 weeks", 9, 13),
        ("14-26 weeks", 14, 26),
        ("27+ weeks", 27, 10_000),
    ):
        band = predictions[
            (predictions["weeks_since_launch"] >= low)
            & (predictions["weeks_since_launch"] <= high)
            & predictions["cold_start"].notna()
        ]
        if band.empty:
            continue
        band_actual = band["y"].to_numpy(dtype="float64")
        bands.append(
            {
                "age_band": label,
                "rows": int(len(band)),
                "units_share": round(float(band["y"].sum() / predictions["y"].sum()), 6),
                "cold_start_wape": round(wape(band_actual, band["cold_start"].to_numpy()), 6),
                "ensemble_wape": round(wape(band_actual, band["ensemble"].to_numpy()), 6),
                "gbm_wape": round(wape(band_actual, band["gbm"].to_numpy()), 6),
            }
        )
    summary["cold_start_by_age"] = bands

    # --- Where the reconciliation helps and where it does not --------------- #
    by_regime = []
    for regime, group in predictions.groupby("regime", sort=True):
        group_actual = group["y"].to_numpy(dtype="float64")
        by_regime.append(
            {
                "regime": str(regime),
                "rows": int(len(group)),
                "ensemble_wape": round(wape(group_actual, group["ensemble"].to_numpy()), 6),
                "reconciled_wape": round(wape(group_actual, group["reconciled"].to_numpy()), 6),
            }
        )
    summary["by_regime"] = by_regime

    by_horizon = []
    for horizon, group in predictions.groupby("horizon", sort=True):
        group_actual = group["y"].to_numpy(dtype="float64")
        by_horizon.append(
            {
                "horizon": int(horizon),
                "ensemble_wape": round(wape(group_actual, group["ensemble"].to_numpy()), 6),
                "reconciled_wape": round(wape(group_actual, group["reconciled"].to_numpy()), 6),
            }
        )
    summary["by_horizon"] = by_horizon

    # --- Aggregated views: what a planner actually commits to --------------- #
    aggregated_views: dict[str, dict[str, float]] = {}
    for label, keys in (
        ("sku_month", ["sku_id", pd.Grouper(key="target_week", freq="MS")]),
        ("sku_full_horizon", ["sku_id", "week_start"]),
        ("category_week", ["category", "target_week"]),
        ("portfolio_week", ["target_week"]),
    ):
        rolled = predictions.groupby(keys, sort=False)[["y", "ensemble", "reconciled"]].sum()
        aggregated_views[label] = {
            "ensemble_accuracy": round(
                1.0 - wape(rolled["y"].to_numpy(), rolled["ensemble"].to_numpy()), 6
            ),
            "reconciled_accuracy": round(
                1.0 - wape(rolled["y"].to_numpy(), rolled["reconciled"].to_numpy()), 6
            ),
        }
    summary["aggregated_accuracy"] = aggregated_views

    # --- Persist models refitted on the full history ------------------------ #
    try:
        final_reconciler = train_reconciler(
            bottom_supervised,
            aggregate_supervised,
            hierarchy,
            settings,
            feature_spec=feature_spec,
        )
        final_reconciler.save(settings.artifacts_dir / RECONCILER_FILENAME)
        summary["reconciler_diagnostics"] = final_reconciler.diagnostics

        final_cold_start: ColdStartModel = fit_cold_start(panel)
        import joblib

        joblib.dump(final_cold_start, settings.artifacts_dir / COLD_START_FILENAME)
        summary["cold_start_diagnostics"] = final_cold_start.diagnostics
    except ForesightError as exc:
        log.error("final fit failed", extra={"context": {"error": str(exc)}})
        return 1

    (settings.artifacts_dir / "hierarchy_metrics.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    predictions.to_parquet(settings.artifacts_dir / "hierarchy_predictions.parquet", index=False)

    log.info(
        "hierarchy backtest complete",
        extra={
            "context": {
                "ensemble_wape": ensemble_wape,
                "reconciled_wape": reconciled_wape,
                "gain": summary["reconciliation_gain_vs_ensemble"],
                "coherence_max_error": summary["coherence_max_error"],
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
