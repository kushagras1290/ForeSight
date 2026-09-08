"""Request and response contracts for the FORESIGHT scoring service.

Every response is wrapped in a consistent envelope so a client never has to
guess at the shape::

    {"success": true,  "data": {...}, "error": null}
    {"success": false, "data": null,  "error": {"code": "...", "message": "..."}}

Error messages are written for a human reading them in a browser and never
carry internal detail - no stack traces, no file paths, no SQL. The full
diagnostic goes to the structured log instead.
"""

from __future__ import annotations

import datetime as dt
import re
from enum import StrEnum
from typing import Annotated, Any, Final, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Product identifier format, applied after whitespace and case normalisation.
_SKU_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-_]{1,31}$")

__all__ = [
    "ApiError",
    "BatchForecastRequest",
    "ErrorCode",
    "ForecastPoint",
    "HealthResponse",
    "HoldoutSummary",
    "ReadyResponse",
    "Envelope",
    "RiskRecord",
    "SkuForecastResponse",
    "SkuSummary",
]

DataT = TypeVar("DataT")

#: SKU ids follow NBL-<3 letters>-<digits>. Constraining the pattern at the edge
#: means a malformed id is rejected before it reaches any lookup.
SkuId = Annotated[
    str,
    Field(
        min_length=3,
        max_length=32,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9\-_]{1,31}$",
        description="Product identifier, e.g. NBL-LGT-004.",
    ),
]


class ErrorCode(StrEnum):
    """Stable, machine-readable error identifiers."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    ARTIFACTS_UNAVAILABLE = "ARTIFACTS_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"
    BATCH_TOO_LARGE = "BATCH_TOO_LARGE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ApiError(BaseModel):
    """The error half of the envelope."""

    model_config = ConfigDict(frozen=True)

    code: ErrorCode
    message: str = Field(description="Safe, human-readable explanation.")
    details: dict[str, Any] | None = None


class Envelope(BaseModel, Generic[DataT]):
    """Uniform response wrapper."""

    success: bool = True
    data: DataT | None = None
    error: ApiError | None = None


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    """Liveness. Answers only 'is the process up'."""

    status: Literal["ok"] = "ok"
    service: str
    version: str
    time: dt.datetime


class ReadyResponse(BaseModel):
    """Readiness. Answers 'can this process actually serve traffic'."""

    ready: bool
    artifacts_loaded: list[str]
    artifacts_missing: list[str]
    model: str | None = None
    origin_week: dt.date | None = None
    skus: int = 0


# --------------------------------------------------------------------------- #
# Forecast
# --------------------------------------------------------------------------- #
class ForecastPoint(BaseModel):
    """One forecast week for one SKU."""

    model_config = ConfigDict(frozen=True)

    week_starting: dt.date = Field(description="Monday of the forecast week.")
    horizon: int = Field(ge=1, description="Weeks ahead of the forecast origin.")
    forecast_units: float = Field(ge=0.0)
    lower_units: float = Field(ge=0.0, description="Lower bound of the prediction interval.")
    upper_units: float = Field(ge=0.0, description="Upper bound of the prediction interval.")


class RiskRecord(BaseModel):
    """Risk assessment and recommended action for one SKU."""

    sku_id: str
    category: str
    subcategory: str

    on_hand_units: float
    on_order_units: float
    available_units: float
    lead_time_days: float
    inventory_as_of: dt.date

    forecast_lead_time_units: float = Field(
        description="Expected demand over the replenishment lead time."
    )
    forecast_horizon_units: float = Field(description="Expected demand over the full horizon.")
    cover_weeks: float = Field(description="Weeks of forward cover at the forecast rate.")

    stockout_score: float = Field(
        ge=0.0, le=1.0, description="P(demand exceeds stock over lead time)."
    )
    overstock_score: float = Field(ge=0.0, le=1.0)
    stockout_level: str
    overstock_level: str

    action: str = Field(description="reorder_now | markdown_clear | watch_volatile | healthy")
    action_label: str
    action_rationale: str

    recommended_order_units: float
    safety_stock_units: float
    expected_lost_units: float
    excess_units: float
    service_level: float = Field(
        description="Target probability of not stocking out that the order quantity was sized for."
    )

    revenue_at_risk: float = Field(description="Rupees of sales at risk from stocking out.")
    margin_at_risk: float
    locked_capital: float = Field(description="Rupees tied up in stock beyond the cover threshold.")
    value_at_stake: float
    priority_rank: int

    forecast_confidence: str = Field(description="high | medium | low")


class SkuForecastResponse(BaseModel):
    """Everything the service knows about one SKU."""

    sku_id: str
    category: str
    subcategory: str
    model: str
    origin_week: dt.date
    horizon_weeks: int
    forecast: list[ForecastPoint]
    risk: RiskRecord | None = None


class BatchForecastRequest(BaseModel):
    """Batch scoring request."""

    model_config = ConfigDict(extra="forbid")

    # Deliberately NOT typed as `SkuId`. Field-level pattern validation runs
    # before any validator, so a padded id pasted out of a spreadsheet would be
    # rejected before it could be trimmed. The format is enforced below, after
    # normalisation - lenient about whitespace and case, strict about the rest.
    sku_ids: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        min_length=1,
        description=(
            "Product identifiers to score. Whitespace and case are normalised, "
            "and duplicates are collapsed."
        ),
    )

    @field_validator("sku_ids")
    @classmethod
    def _normalise(cls, value: list[str]) -> list[str]:
        """Trim, upper-case, de-duplicate, then validate the format."""
        seen: set[str] = set()
        ordered: list[str] = []
        for raw in value:
            normalised = raw.strip().upper()
            if normalised and normalised not in seen:
                seen.add(normalised)
                ordered.append(normalised)

        if not ordered:
            raise ValueError("no usable product identifiers were supplied")

        malformed = [candidate for candidate in ordered if not _SKU_PATTERN.fullmatch(candidate)]
        if malformed:
            shown = ", ".join(malformed[:5])
            raise ValueError(f"not valid product identifiers: {shown}")

        return ordered


class LivePosition(BaseModel):
    """A stock position supplied by the caller, as of right now.

    Only ``sku_id`` and ``on_hand_units`` are required. Anything omitted falls
    back to the most recent stored snapshot for that product, so a caller with
    only a live on-hand count does not have to reconstruct the rest.
    """

    model_config = ConfigDict(extra="forbid")

    sku_id: SkuId
    on_hand_units: float = Field(ge=0.0, le=10_000_000.0)
    on_order_units: float | None = Field(default=None, ge=0.0, le=10_000_000.0)
    lead_time_days: float | None = Field(
        default=None,
        ge=1.0,
        le=365.0,
        description="Override the stored replenishment lead time.",
    )


class LiveScoreRequest(BaseModel):
    """Re-score risk against stock positions supplied at request time."""

    model_config = ConfigDict(extra="forbid")

    positions: list[LivePosition] = Field(min_length=1)

    @field_validator("positions")
    @classmethod
    def _unique_skus(cls, value: list[LivePosition]) -> list[LivePosition]:
        """One position per product; a duplicate is a caller mistake, not a merge."""
        seen: set[str] = set()
        for position in value:
            normalised = position.sku_id.strip().upper()
            if normalised in seen:
                raise ValueError(f"duplicate position supplied for {normalised}")
            seen.add(normalised)
        return value


class LiveScoreResponse(BaseModel):
    """Freshly computed risk, against the caller's own stock position."""

    scored_at: dt.datetime
    forecast_origin_week: dt.date
    forecast_age_days: int = Field(
        description=(
            "Days since the forecast was produced. The forecast is refreshed weekly; "
            "a large value means it is due a refresh."
        )
    )
    model: str
    requested: int
    returned: int
    not_found: list[str]
    results: list[RiskRecord]


class ActualObservation(BaseModel):
    """One observed week of demand, supplied after the fact."""

    model_config = ConfigDict(extra="forbid")

    sku_id: SkuId
    week_starting: dt.date = Field(
        description="Any date inside the week; it is snapped back to the Monday."
    )
    units: float = Field(ge=0.0, le=10_000_000.0)


class EvaluationRequest(BaseModel):
    """Score the shipped forecast against actuals the client now has."""

    model_config = ConfigDict(extra="forbid")

    actuals: list[ActualObservation] = Field(min_length=1, max_length=100_000)


class MetricBlock(BaseModel):
    """One model's accuracy over the matched rows."""

    n_observations: int
    total_actual: float
    total_predicted: float
    wape: float
    mape: float
    mape_coverage: float
    bias_relative: float
    mae: float
    rmse: float


class EvaluationBreakdown(BaseModel):
    """Accuracy for one level of a grouping."""

    key: str
    n_observations: int
    total_actual: float
    wape: float
    baseline_wape: float | None = None
    bias_relative: float


class ErrorContributor(BaseModel):
    """A product and how much of the total absolute error it accounts for."""

    sku_id: str
    n_observations: int
    total_actual: float
    total_forecast: float
    absolute_error: float
    share_of_total_error: float


class EvaluationResponse(BaseModel):
    """How the forecast actually performed on the supplied actuals."""

    rows_submitted: int
    rows_matched: int
    rows_unmatched: int
    unmatched_examples: list[str]
    weeks_covered: list[str]
    source_counts: dict[str, int]
    model: str

    accuracy: MetricBlock
    baseline: MetricBlock | None = None
    baseline_row_coverage: float = Field(
        description="Share of matched rows a seasonal-naive baseline could be computed for."
    )
    improvement_vs_baseline: float | None = None

    interval_coverage: float
    interval_coverage_target: float

    by_horizon: list[EvaluationBreakdown]
    by_category: list[EvaluationBreakdown]
    worst_contributors: list[ErrorContributor]


class BatchForecastResponse(BaseModel):
    """Batch result, reporting misses explicitly rather than silently dropping them."""

    requested: int
    returned: int
    not_found: list[str]
    results: list[SkuForecastResponse]


# --------------------------------------------------------------------------- #
# Dashboard support
# --------------------------------------------------------------------------- #
class SkuSummary(BaseModel):
    """Compact SKU descriptor for filters and pickers."""

    sku_id: str
    category: str
    subcategory: str


class HistoryPoint(BaseModel):
    """One observed week of demand."""

    model_config = ConfigDict(frozen=True)

    week_starting: dt.date
    units: float
    revenue: float
    promo_days: float


class BacktestPoint(BaseModel):
    """One backtested forecast against the actual that followed."""

    model_config = ConfigDict(frozen=True)

    week_starting: dt.date
    actual: float
    forecast: float
    baseline: float
    lower: float
    upper: float


class SkuDetailResponse(BaseModel):
    """The full SKU view the dashboard's detail panel renders."""

    sku_id: str
    category: str
    subcategory: str
    unit_cost: float
    list_price: float
    history: list[HistoryPoint]
    backtest: list[BacktestPoint]
    forecast: list[ForecastPoint]
    risk: RiskRecord | None = None


class PortfolioSummary(BaseModel):
    """Headline numbers for the dashboard's KPI row and the executive readout."""

    origin_week: dt.date
    horizon_weeks: int
    model: str
    total_skus: int

    reorder_now_skus: int
    markdown_skus: int
    watch_skus: int
    healthy_skus: int
    low_confidence_skus: int

    revenue_at_risk_total: float
    margin_at_risk_total: float
    locked_capital_total: float
    expected_lost_units_total: float
    excess_units_total: float
    reorder_now_order_units: float
    top_10_share_of_revenue_at_risk: float


class AccuracySummary(BaseModel):
    """How the shipped forecast performed on the rolling-origin backtest."""

    selected_model: str
    selected_wape: float
    baseline_wape: float
    naive_wape: float
    gbm_wape: float
    ensemble_wape: float | None = None
    improvement_vs_baseline: float
    interval_coverage: float
    interval_coverage_target: float
    bias_relative: float
    folds: int
    test_observations: int
    horizon_weeks: int
    by_horizon: list[dict[str, Any]]
    by_category: list[dict[str, Any]]
    by_regime: list[dict[str, Any]]


class HoldoutSummary(BaseModel):
    """The chronological 70/30 test: forecasts against outcomes never trained on.

    Distinct from :class:`AccuracySummary`, which reports the rolling-origin
    backtest. This is the simpler and more sceptical test - one split, one
    training run, and a full year of unseen weeks - so it is reported separately
    rather than averaged in.
    """

    available: bool
    reason: str | None = None

    split_week: str | None = None
    holdout_start: str | None = None
    holdout_end: str | None = None
    train_weeks: int = 0
    holdout_weeks: int = 0
    train_fraction: float = 0.0
    observations: int = 0
    model_name: str = "unknown"

    model_wape: float = float("nan")
    baseline_wape: float = float("nan")
    improvement_vs_baseline: float = 0.0
    bias_relative: float = 0.0
    interval_coverage: float = float("nan")
    interval_coverage_target: float = 0.8

    #: One row per holdout week: total demand against total forecast.
    weekly: list[dict[str, Any]] = Field(default_factory=list)
    by_category: list[dict[str, Any]] = Field(default_factory=list)
    by_horizon: list[dict[str, Any]] = Field(default_factory=list)
    #: Per-SKU scorecard, worst first, so a reviewer starts where it went wrong.
    skus: list[dict[str, Any]] = Field(default_factory=list)
