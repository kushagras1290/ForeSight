"""Score the shipped forecast against actuals supplied after the fact.

WHY THIS EXISTS
---------------
The backtest says how the model performed on history. It cannot say how it is
performing *now*. Once the eight weeks a plan covered have actually happened,
the client has the one thing no backtest can provide - ground truth on the
forecast they actually acted on.

This module takes those actuals and reports the same metrics the backtest does,
against the same seasonal-naive baseline, so accuracy can be re-checked on live
data rather than taken on trust from a number computed months earlier. It is the
monitoring loop the brief names as a stretch goal: *"how NorthBay would detect
the forecast degrading."*

MATCHING
--------
Actuals are matched to forecasts on ``(sku_id, week_starting)``. Dates are
snapped back to the Monday of their week, so a caller can submit any day inside
a week and still match - a spreadsheet exported on a Friday should not silently
score zero rows.

Forward forecasts are preferred over backtest rows when both exist for a week:
the forward forecast is what the client was actually given, so it is what should
be judged.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd

from foresight.exceptions import DataQualityError
from foresight.logging_setup import get_logger
from foresight.metrics import ForecastMetrics, evaluate_forecast, interval_coverage, wape

__all__ = [
    "EvaluationResult",
    "REQUIRED_COLUMNS",
    "classify_csv",
    "evaluate_actuals",
    "parse_actuals_csv",
    "parse_forecast_csv",
]

log = get_logger(__name__)

SEASONAL_PERIOD_WEEKS: Final[int] = 52

#: Canonical column -> the header spellings accepted for it. Client exports do
#: not use our names, and rejecting a file over a header is a poor trade.
COLUMN_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "sku_id": (
        "sku_id",
        "sku",
        "skuid",
        "sku_code",
        "product",
        "product_id",
        "product_code",
        "item",
        "item_id",
        "item_code",
        "code",
        "article",
    ),
    "week_starting": (
        "week_starting",
        "week_start",
        "week",
        "date",
        "period",
        "week_beginning",
        "week_commencing",
        "week_ending",
        "week_of",
        "wc",
    ),
    "units": (
        "units",
        "units_sold",
        "unit_sold",
        "actual",
        "actuals",
        "actual_units",
        "demand",
        "qty",
        "qty_sold",
        "quantity",
        "quantity_sold",
        "sales",
        "sales_units",
        "sold",
        "volume",
    ),
}

#: Substring fallbacks, tried only when no exact alias matched. Ordered by how
#: specific they are, since a header can contain more than one token.
COLUMN_TOKENS: Final[dict[str, tuple[str, ...]]] = {
    "sku_id": ("sku", "product", "item", "article", "code"),
    "week_starting": ("week", "date", "period", "month"),
    "units": ("unit", "qty", "quantity", "sold", "sales", "demand", "actual", "volume"),
}

REQUIRED_COLUMNS: Final[tuple[str, ...]] = tuple(COLUMN_ALIASES)

#: Header spellings that identify a *forecast* file rather than an actuals file.
#: Deliberately narrow: "actual" is an accepted spelling for a quantity column,
#: so only unambiguously predictive words appear here.
FORECAST_VALUE_ALIASES: Final[tuple[str, ...]] = (
    "prediction",
    "predicted",
    "predicted_units",
    "forecast",
    "forecast_units",
    "forecast_qty",
    "yhat",
    "fcst",
    "expected_units",
)

#: Optional interval and horizon columns on a forecast file.
FORECAST_LOWER_ALIASES: Final[tuple[str, ...]] = (
    "prediction_lower",
    "forecast_lower",
    "lower",
    "lower_bound",
    "lo",
    "p10",
)
FORECAST_UPPER_ALIASES: Final[tuple[str, ...]] = (
    "prediction_upper",
    "forecast_upper",
    "upper",
    "upper_bound",
    "hi",
    "p90",
)
FORECAST_HORIZON_ALIASES: Final[tuple[str, ...]] = (
    "horizon",
    "h",
    "weeks_ahead",
    "lead_weeks",
    "step",
)

#: The week a forecast is *for*. Listed ahead of the generic week aliases because
#: a forecast export usually carries both this and the origin week, and picking
#: the origin would score every row against the wrong week.
FORECAST_TARGET_ALIASES: Final[tuple[str, ...]] = (
    "target_week",
    "target_date",
    "forecast_week",
    "week_starting",
    "week_start",
    "week",
    "date",
    "period",
)

#: Upper bound on an uploaded file, before parsing. Guards memory.
MAX_UPLOAD_BYTES: Final[int] = 8 * 1024 * 1024
MAX_ROWS: Final[int] = 100_000

#: SKUs returned in ``worst_contributors``, worst first. High enough to cover
#: every SKU in a realistic catalogue (the brief's own extract tops out at
#: ~200) rather than truncating to a "top few" - the dashboard's test-results
#: view is a full exploration, not a highlight reel.
MAX_WORST_CONTRIBUTORS: Final[int] = 200


@dataclass(slots=True)
class EvaluationResult:
    """How the forecast performed against supplied actuals."""

    rows_submitted: int
    rows_deduplicated: int
    rows_matched: int
    unmatched_examples: list[str]
    weeks_covered: list[str]
    source_counts: dict[str, int]

    model: ForecastMetrics
    baseline: ForecastMetrics | None
    baseline_coverage: float

    interval_coverage: float
    by_horizon: pd.DataFrame
    by_category: pd.DataFrame
    worst_contributors: pd.DataFrame
    matched: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def improvement_vs_baseline(self) -> float:
        """Relative WAPE reduction against seasonal-naive. NaN if unavailable."""
        if self.baseline is None or not np.isfinite(self.baseline.wape):
            return float("nan")
        if self.baseline.wape <= 0:
            return float("nan")
        return (self.baseline.wape - self.model.wape) / self.baseline.wape


def _resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Map the file's actual headers onto our canonical names.

    Two passes. An exact alias match first, then a substring fallback, because a
    client export says ``Qty Sold`` or ``Week Commencing`` and no alias list is
    ever going to be complete. A column already claimed is not reused, so three
    distinct headers are always returned.
    """
    normalised = {
        str(column).strip().lower().replace(" ", "_").replace("-", "_"): column
        for column in frame.columns
    }

    resolved: dict[str, str] = {}
    claimed: set[str] = set()

    # Pass 1 - exact.
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalised and normalised[alias] not in claimed:
                resolved[canonical] = normalised[alias]
                claimed.add(normalised[alias])
                break

    # Pass 2 - substring, only for whatever is still unresolved.
    for canonical, tokens in COLUMN_TOKENS.items():
        if canonical in resolved:
            continue
        for token in tokens:
            match = next(
                (
                    original
                    for header, original in normalised.items()
                    if token in header and original not in claimed
                ),
                None,
            )
            if match is not None:
                resolved[canonical] = match
                claimed.add(match)
                break

    missing = [name for name in REQUIRED_COLUMNS if name not in resolved]
    if missing:
        raise DataQualityError(
            f"The file is missing a column for: {', '.join(missing)}. "
            f"Found headers: {', '.join(str(c) for c in frame.columns[:12])}. "
            f"Accepted names for '{missing[0]}' include: "
            f"{', '.join(COLUMN_ALIASES[missing[0]][:6])}."
        )
    return resolved


def _decode(content: bytes | str) -> str:
    """Decode an upload to text, with the size and encoding guards."""
    if isinstance(content, str):
        return content
    if len(content) > MAX_UPLOAD_BYTES:
        raise DataQualityError(
            f"File is {len(content) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB."
        )
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataQualityError(
            "The file is not valid UTF-8 text. Export it as CSV (UTF-8) and retry."
        ) from exc


def _normalise_headers(columns: Any) -> dict[str, Any]:
    return {
        str(column).strip().lower().replace(" ", "_").replace("-", "_"): column
        for column in columns
    }


def classify_csv(content: bytes | str) -> str:
    """Decide whether an upload is a forecast or a file of actuals.

    Reads the header row only. A file is a forecast when it carries an
    unambiguously predictive value column; anything else is treated as actuals,
    because that is the far more common upload and the one whose failure message
    is most useful.

    Returns:
        ``"forecast"`` or ``"actuals"``.
    """
    text = _decode(content)
    if not text.strip():
        raise DataQualityError("The file is empty.")

    try:
        header = pd.read_csv(io.StringIO(text), nrows=0)
    except (pd.errors.ParserError, ValueError) as exc:
        raise DataQualityError(f"The file could not be read as CSV: {exc}") from exc

    normalised = _normalise_headers(header.columns)
    return "forecast" if any(name in normalised for name in FORECAST_VALUE_ALIASES) else "actuals"


def parse_forecast_csv(content: bytes | str) -> pd.DataFrame:
    """Parse an uploaded forecast into the shape :func:`evaluate_actuals` expects.

    Accepting a forecast alongside actuals is what lets a complete external test
    set be scored end to end - the numbers on screen then come entirely from the
    uploaded pair, with nothing borrowed from whatever this service happens to
    have stored.

    Returns:
        ``sku_id``, ``target_week``, ``horizon``, ``prediction``,
        ``prediction_lower``, ``prediction_upper``. Intervals fall back to the
        point forecast when the file has none, so coverage reads as degenerate
        rather than as a spurious hit.

    Raises:
        DataQualityError: naming the header it could not find.
    """
    text = _decode(content)
    if not text.strip():
        raise DataQualityError("The forecast file is empty.")

    try:
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=True)
    except (pd.errors.ParserError, ValueError) as exc:
        raise DataQualityError(f"The forecast file could not be read as CSV: {exc}") from exc

    if frame.empty:
        raise DataQualityError("The forecast file has headers but no rows.")
    if len(frame) > MAX_ROWS:
        raise DataQualityError(
            f"The forecast file has {len(frame):,} rows; the limit is {MAX_ROWS:,}."
        )

    normalised = _normalise_headers(frame.columns)

    def pick(aliases: tuple[str, ...]) -> Any | None:
        return next((normalised[name] for name in aliases if name in normalised), None)

    sku_column = next(
        (normalised[name] for name in COLUMN_ALIASES["sku_id"] if name in normalised), None
    )
    week_column = pick(FORECAST_TARGET_ALIASES)
    value_column = pick(FORECAST_VALUE_ALIASES)

    missing = [
        label
        for label, column in (
            ("a product code", sku_column),
            ("a target week", week_column),
            ("a forecast quantity", value_column),
        )
        if column is None
    ]
    if missing:
        raise DataQualityError(
            f"The forecast file is missing {', '.join(missing)}. "
            f"Found headers: {', '.join(str(c) for c in frame.columns[:12])}. "
            f"A forecast quantity column may be named any of: "
            f"{', '.join(FORECAST_VALUE_ALIASES[:5])}."
        )

    def numeric(column: Any | None) -> pd.Series | None:
        if column is None:
            return None
        return pd.to_numeric(
            frame[column].astype("string").str.replace(",", "", regex=False), errors="coerce"
        )

    out = pd.DataFrame(
        {
            "sku_id": frame[sku_column].astype("string").str.strip().str.upper(),
            "target_week": pd.to_datetime(
                frame[week_column].astype("string").str.strip(),
                errors="coerce",
                format="mixed",
                dayfirst=False,
            ),
            "prediction": numeric(value_column),
        }
    )

    horizon = numeric(pick(FORECAST_HORIZON_ALIASES))
    out["horizon"] = (horizon if horizon is not None else pd.Series(1, index=out.index)).fillna(1)

    lower = numeric(pick(FORECAST_LOWER_ALIASES))
    upper = numeric(pick(FORECAST_UPPER_ALIASES))
    # Without an interval, collapse it onto the point forecast. Coverage then
    # reads as near zero, which is honest: the file supplied no interval to
    # cover anything with.
    out["prediction_lower"] = lower if lower is not None else out["prediction"]
    out["prediction_upper"] = upper if upper is not None else out["prediction"]

    out = out[out["target_week"].notna() & out["prediction"].notna() & out["sku_id"].notna()]
    if out.empty:
        raise DataQualityError(
            "No forecast row had a readable product code, target week and quantity together."
        )

    out["horizon"] = out["horizon"].astype("int64")
    out["prediction_lower"] = out["prediction_lower"].fillna(out["prediction"])
    out["prediction_upper"] = out["prediction_upper"].fillna(out["prediction"])
    out["target_week"] = _snap_to_week_start(out["target_week"])

    # One forecast per (product, week). Where a file carries several origins for
    # the same target week, keep the last - the freshest origin in an export
    # ordered by origin, which is how these files are written.
    out = out.drop_duplicates(subset=["sku_id", "target_week"], keep="last")

    log.info(
        "parsed uploaded forecast",
        extra={
            "context": {
                "rows": len(out),
                "skus": int(out["sku_id"].nunique()),
                "has_interval": lower is not None and upper is not None,
            }
        },
    )
    return out.reset_index(drop=True)


def parse_actuals_csv(content: bytes | str) -> pd.DataFrame:
    """Parse an uploaded CSV of actual demand into a normalised frame.

    Returns:
        ``sku_id``, ``week_starting`` and ``units``, cleaned and de-duplicated.

    Raises:
        DataQualityError: with a message written for the person who uploaded the
            file, naming the header it could not find or the rows it could not
            read.
    """
    if isinstance(content, bytes):
        if len(content) > MAX_UPLOAD_BYTES:
            raise DataQualityError(
                f"File is {len(content) / 1e6:.1f} MB; the limit is "
                f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB."
            )
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DataQualityError(
                "The file is not valid UTF-8 text. Export it as CSV (UTF-8) and retry."
            ) from exc
    else:
        text = content

    if not text.strip():
        raise DataQualityError("The file is empty.")

    try:
        # Everything as text, exactly as the ingest layer does: inference on a
        # messy export is what turns one bad row into a wrong column dtype.
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=True)
    except (pd.errors.ParserError, ValueError) as exc:
        raise DataQualityError(f"The file could not be read as CSV: {exc}") from exc

    if frame.empty:
        raise DataQualityError("The file has headers but no rows.")
    if len(frame) > MAX_ROWS:
        raise DataQualityError(f"The file has {len(frame):,} rows; the limit is {MAX_ROWS:,}.")

    resolved = _resolve_columns(frame)
    out = pd.DataFrame(
        {
            "sku_id": frame[resolved["sku_id"]].astype("string").str.strip().str.upper(),
            "week_starting": pd.to_datetime(
                frame[resolved["week_starting"]].astype("string").str.strip(),
                errors="coerce",
                format="mixed",
                dayfirst=False,
            ),
            "units": pd.to_numeric(
                frame[resolved["units"]].astype("string").str.replace(",", "", regex=False),
                errors="coerce",
            ),
        }
    )

    unreadable = int((out["week_starting"].isna() | out["units"].isna()).sum())
    out = out[out["week_starting"].notna() & out["units"].notna() & out["sku_id"].notna()]
    if out.empty:
        raise DataQualityError(
            "No row had a readable product code, date and quantity together. "
            "Check the date format and that the quantity column is numeric."
        )

    # Negative actuals are returns, not demand - same treatment as the pipeline.
    out["units"] = out["units"].clip(lower=0.0)

    if unreadable:
        log.warning(
            "dropped unreadable rows from uploaded actuals",
            extra={"context": {"dropped": unreadable, "kept": len(out)}},
        )

    return out.reset_index(drop=True)


def _snap_to_week_start(dates: pd.Series) -> pd.Series:
    """Snap any date back to the Monday of its week."""
    parsed = pd.to_datetime(dates)
    return parsed - pd.to_timedelta(parsed.dt.dayofweek, unit="D")


def _group_metrics(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """WAPE and bias for the model and the baseline, per level of ``column``."""
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(column, observed=True):
        actual = group["actual"].to_numpy(dtype="float64")
        total = float(np.abs(actual).sum())
        baseline_wape = (
            wape(actual, group["baseline"]) if group["baseline"].notna().any() else float("nan")
        )
        records.append(
            {
                "key": str(key),
                "n_observations": int(len(group)),
                "total_actual": total,
                "wape": wape(actual, group["forecast"]),
                "baseline_wape": baseline_wape,
                "bias_relative": (
                    float((group["forecast"].to_numpy() - actual).sum() / total)
                    if total > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("key").reset_index(drop=True)


def evaluate_actuals(
    actuals: pd.DataFrame,
    forecast: pd.DataFrame,
    backtest: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    holdout: pd.DataFrame | None = None,
    interval_coverage_target: float = 0.8,
    sku_master: pd.DataFrame | None = None,
) -> EvaluationResult:
    """Score forecasts against supplied actuals.

    Args:
        actuals: ``sku_id``, ``week_starting``, ``units``.
        forecast: The forward forecast (``sku_id``, ``target_week``, ``horizon``,
            ``prediction``, ``prediction_lower``, ``prediction_upper``).
        backtest: Backtest predictions, used for weeks the forward forecast does
            not cover.
        panel: The weekly panel, used to reconstruct the seasonal-naive baseline.
        holdout: Optional forecasts from a chronological holdout split. Preferred
            over backtest rows for the same week because the holdout model never
            trained on any of the period being scored.
        interval_coverage_target: Stated coverage, for the comparison.
        sku_master: Optional, to attach category to the breakdown.

    Raises:
        DataQualityError: if nothing could be matched.
    """
    if actuals.empty:
        raise DataQualityError("No actuals were supplied.")

    submitted = len(actuals)
    work = actuals.copy()
    work["sku_id"] = work["sku_id"].astype("string").str.strip().str.upper()
    work["target_week"] = _snap_to_week_start(work["week_starting"])
    work = work.rename(columns={"units": "actual"})

    # One row per (product, week). A duplicate upload should not double-count.
    # Counted separately from unmatched rows: a caller who submitted the same
    # week twice has not supplied anything the forecast failed to cover, and
    # folding the two together would read as a matching failure.
    work = work.drop_duplicates(subset=["sku_id", "target_week"], keep="last")
    deduplicated = len(work)

    # --- 1. Forward forecast: what the client was actually given ------------ #
    forward = pd.DataFrame()
    if not forecast.empty:
        forward = forecast.loc[
            :,
            [
                "sku_id",
                "target_week",
                "horizon",
                "prediction",
                "prediction_lower",
                "prediction_upper",
            ],
        ].copy()
        forward["target_week"] = pd.to_datetime(forward["target_week"])
        forward["source"] = "forward_forecast"

    # --- 1b. Holdout forecasts, if a split has been run --------------------- #
    held_out = pd.DataFrame()
    if holdout is not None and not holdout.empty:
        held_out = holdout.loc[
            :,
            [
                "sku_id",
                "target_week",
                "horizon",
                "prediction",
                "prediction_lower",
                "prediction_upper",
            ],
        ].copy()
        held_out["target_week"] = pd.to_datetime(held_out["target_week"])
        held_out["source"] = "holdout"
        held_out = held_out.drop_duplicates(subset=["sku_id", "target_week"], keep="first")

    # --- 2. Backtest rows for anything the others miss ---------------------- #
    historical = pd.DataFrame()
    if not backtest.empty:
        prediction_column = "ensemble" if "ensemble" in backtest.columns else "prediction"
        lower_column = (
            "ensemble_lower" if "ensemble_lower" in backtest.columns else "prediction_lower"
        )
        upper_column = (
            "ensemble_upper" if "ensemble_upper" in backtest.columns else "prediction_upper"
        )

        historical = backtest.loc[
            :, ["sku_id", "target_week", "horizon", prediction_column, lower_column, upper_column]
        ].rename(
            columns={
                prediction_column: "prediction",
                lower_column: "prediction_lower",
                upper_column: "prediction_upper",
            }
        )
        historical["target_week"] = pd.to_datetime(historical["target_week"])
        historical["source"] = "backtest"
        # A week can appear in several folds; keep one view of it.
        historical = historical.drop_duplicates(subset=["sku_id", "target_week"], keep="last")

    # Order matters: the first frame wins a tie. Forward first because it is what
    # the client actually acted on; then holdout, whose model never saw the
    # period at all; backtest last, as the weakest claim to independence.
    available = pd.concat([forward, held_out, historical], ignore_index=True)
    if available.empty:
        raise DataQualityError(
            "No forecasts are loaded to score against. Run the pipeline and scoring scripts."
        )
    available = available.drop_duplicates(subset=["sku_id", "target_week"], keep="first")

    matched = work.merge(available, on=["sku_id", "target_week"], how="inner")

    if matched.empty:
        weeks = sorted({str(value.date()) for value in available["target_week"].unique()})
        raise DataQualityError(
            "None of the supplied rows matched a forecast. Forecasts exist for weeks "
            f"beginning {weeks[0]} to {weeks[-1]}. Check the product codes and that the "
            "dates fall inside that range."
        )

    # --- 3. Seasonal-naive baseline, reconstructed from history ------------- #
    baseline_source = panel.loc[:, ["sku_id", "week_start", "units"]].copy()
    baseline_source["target_week"] = pd.to_datetime(
        baseline_source["week_start"]
    ) + pd.to_timedelta(SEASONAL_PERIOD_WEEKS * 7, unit="D")
    baseline_source = baseline_source.rename(columns={"units": "baseline"})[
        ["sku_id", "target_week", "baseline"]
    ]
    matched = matched.merge(baseline_source, on=["sku_id", "target_week"], how="left")

    # --- 4. Category, for the breakdown ------------------------------------- #
    if sku_master is not None and not sku_master.empty:
        matched = matched.merge(sku_master.loc[:, ["sku_id", "category"]], on="sku_id", how="left")
    if "category" not in matched.columns:
        matched["category"] = "All"
    matched["category"] = matched["category"].fillna("Unclassified")

    matched = matched.rename(columns={"prediction": "forecast"})

    # --- 5. Metrics ---------------------------------------------------------- #
    actual_values = matched["actual"].to_numpy(dtype="float64")
    model_metrics = evaluate_forecast(actual_values, matched["forecast"])

    has_baseline = matched["baseline"].notna()
    baseline_metrics: ForecastMetrics | None = None
    if has_baseline.any():
        subset = matched[has_baseline]
        # Scored on the same rows as the model, so the comparison is like for like.
        baseline_metrics = evaluate_forecast(
            subset["actual"].to_numpy(dtype="float64"), subset["baseline"]
        )

    coverage = interval_coverage(
        actual_values, matched["prediction_lower"], matched["prediction_upper"]
    )

    # --- 6. Where the error is ----------------------------------------------- #
    matched["absolute_error"] = (matched["forecast"] - matched["actual"]).abs()
    worst = (
        matched.groupby("sku_id", as_index=False)
        .agg(
            n_observations=("actual", "size"),
            total_actual=("actual", "sum"),
            total_forecast=("forecast", "sum"),
            absolute_error=("absolute_error", "sum"),
        )
        .sort_values("absolute_error", ascending=False)
        .head(MAX_WORST_CONTRIBUTORS)
    )
    total_error = float(matched["absolute_error"].sum())
    worst["share_of_total_error"] = (
        worst["absolute_error"] / total_error if total_error > 0 else 0.0
    )

    unmatched = work.merge(
        available[["sku_id", "target_week"]],
        on=["sku_id", "target_week"],
        how="left",
        indicator=True,
    )
    unmatched = unmatched[unmatched["_merge"] == "left_only"]
    unmatched_examples = [
        f"{row.sku_id} @ {pd.Timestamp(row.target_week).date()}"
        for row in unmatched.head(8).itertuples(index=False)
    ]

    result = EvaluationResult(
        rows_submitted=submitted,
        rows_deduplicated=deduplicated,
        rows_matched=len(matched),
        unmatched_examples=unmatched_examples,
        weeks_covered=sorted(
            {str(pd.Timestamp(value).date()) for value in matched["target_week"].unique()}
        ),
        source_counts=matched["source"].value_counts().to_dict(),
        model=model_metrics,
        baseline=baseline_metrics,
        baseline_coverage=float(has_baseline.mean()),
        interval_coverage=coverage,
        by_horizon=_group_metrics(matched, "horizon"),
        by_category=_group_metrics(matched, "category"),
        worst_contributors=worst.reset_index(drop=True),
        matched=matched,
    )

    log.info(
        "evaluated uploaded actuals",
        extra={
            "context": {
                "submitted": submitted,
                "unique": deduplicated,
                "matched": len(matched),
                "wape": round(model_metrics.wape, 4),
                "baseline_wape": (round(baseline_metrics.wape, 4) if baseline_metrics else None),
                "interval_coverage": round(coverage, 4),
                "target_coverage": interval_coverage_target,
            }
        },
    )
    return result
