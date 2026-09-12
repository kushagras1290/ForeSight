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

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from foresight.auth import SECURITY_QUESTIONS

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
    UNAUTHORIZED = "UNAUTHORIZED"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    OAUTH_ERROR = "OAUTH_ERROR"


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


class SalesTrendPoint(BaseModel):
    """One week of sales, aggregated across every SKU in scope."""

    model_config = ConfigDict(frozen=True)

    week_starting: dt.date
    units: float
    revenue: float
    sku_count: int


class CategoryTotal(BaseModel):
    """Total units/revenue for one category over the returned window."""

    model_config = ConfigDict(frozen=True)

    category: str
    units: float
    revenue: float
    sku_count: int


class SalesTrendResponse(BaseModel):
    """Portfolio-wide weekly sales trend, optionally scoped to one category.

    Built by aggregating the same weekly panel every SKU-level route reads
    from, so it can never disagree with the per-product history shown
    elsewhere on the dashboard.
    """

    category: str | None = None
    weeks: list[SalesTrendPoint]
    by_category: list[CategoryTotal]


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


# --------------------------------------------------------------------------- #
# Product performance / business insights
# --------------------------------------------------------------------------- #
class ProductPerformanceRow(BaseModel):
    """One SKU's revenue/units rollup, recent trend, and current risk action."""

    model_config = ConfigDict(frozen=True)

    sku_id: str
    category: str
    subcategory: str
    total_revenue: float
    total_units: float
    revenue_share: float = Field(description="This SKU's share of total portfolio revenue.")
    recent_trend_pct: float = Field(
        description="Last 8 weeks' revenue vs. the 8 weeks before that, as a fraction."
    )
    action: str
    action_label: str
    value_at_stake: float


class ProductPerformanceResponse(BaseModel):
    """Portfolio-wide, sortable per-SKU performance leaderboard."""

    category: str | None = None
    rows: list[ProductPerformanceRow]
    total_skus: int


class RevenueConcentrationPoint(BaseModel):
    """Share of total revenue the top X% of SKUs (by revenue) account for."""

    model_config = ConfigDict(frozen=True)

    sku_fraction: float
    sku_count: int
    revenue_share: float


class DeadStockRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku_id: str
    category: str
    subcategory: str
    consecutive_zero_weeks: int
    last_sale_week: str | None = None


class MoverRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku_id: str
    category: str
    recent_revenue: float
    prior_revenue: float
    change_pct: float


class BusinessInsightsResponse(BaseModel):
    """Revenue concentration, dead stock, and top movers.

    No customer entity exists in any extract this project has (every source is
    SKU/week grained), so this is scoped to business insights derivable from
    what's actually here rather than a customer segmentation this data cannot
    support.
    """

    total_revenue: float
    total_skus: int
    revenue_concentration: list[RevenueConcentrationPoint]
    dead_stock: list[DeadStockRow]
    top_gainers: list[MoverRow]
    top_decliners: list[MoverRow]


class PromotionCategoryStat(BaseModel):
    """One category's (or the pooled/all-category) fitted promotional response."""

    model_config = ConfigDict(frozen=True)

    category: str
    has_own_curve: bool
    uplift_by_discount: dict[str, float] = Field(
        description="Multiplicative uplift at sample discount depths, e.g. {'20pct': 1.8}."
    )
    post_promo_dip_factor: float


class PromotionResponseModel(BaseModel):
    """Custom model 5's fitted response, read back from its persisted artifact.

    ``available=False`` (with ``reason``) until
    ``scripts/11_fit_promotion_response.py`` has been run - this is honest
    degradation, not an error, matching :class:`HoldoutSummary`.
    """

    available: bool
    reason: str | None = None
    baseline_window: int = 0
    dip_weeks: int = 0
    pooled: PromotionCategoryStat | None = None
    categories: list[PromotionCategoryStat] = Field(default_factory=list)


class SeasonalIndexPoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    iso_week: int
    index: float


class SeasonalityResponse(BaseModel):
    """The Adaptive Ensemble's fitted seasonal index, read back from its artifact.

    ``available=False`` (with ``reason``) until
    ``scripts/12_extract_seasonal_profile.py`` has been run.
    """

    available: bool
    reason: str | None = None
    global_index: list[SeasonalIndexPoint] = Field(default_factory=list)
    by_category: dict[str, list[SeasonalIndexPoint]] = Field(default_factory=dict)


class ModelBenchmarkRow(BaseModel):
    """One model's row in the all-models comparison - shape varies by model.

    A standard forecasting model has `wape`/`bias_relative`/`n_observations`;
    a measured-effect custom model (not scored on WAPE by design, see
    `reports/model_suite.md` #7) has `wape=None` and its effect described in
    `note` instead.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    description: str
    wape: float | None = None
    bias_relative: float | None = None
    n_observations: int | None = None
    note: str | None = None


class ModelBenchmarkResponse(BaseModel):
    """The full model comparison, read back from `scripts/10_run_all_models.py`'s
    artifact.

    ``available=False`` (with ``reason``) until that script has been run.
    """

    available: bool
    reason: str | None = None
    folds: int = 0
    no_drivers: bool = False
    rows: list[ModelBenchmarkRow] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
#: Letters, digits and underscores only - keeps a username safe to display
#: and to derive from an email's local part without further escaping.
_USERNAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_]{3,32}$")


class RegisterRequest(BaseModel):
    """Create a new password-based account.

    Two independent security questions are required (see
    :mod:`foresight.auth`'s module docstring for why) - both are needed to
    reset the password later, so registration must collect both up front.
    """

    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    username: Annotated[str, Field(pattern=_USERNAME_PATTERN.pattern)]
    password: Annotated[str, Field(min_length=8, max_length=256)]
    display_name: Annotated[str, Field(min_length=1, max_length=80)]
    security_question_1: Literal[SECURITY_QUESTIONS]
    security_answer_1: Annotated[str, Field(min_length=1, max_length=200)]
    security_question_2: Literal[SECURITY_QUESTIONS]
    security_answer_2: Annotated[str, Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def _questions_must_differ(self) -> RegisterRequest:
        if self.security_question_1 == self.security_question_2:
            raise ValueError("Choose two different security questions.")
        return self


class LoginRequest(BaseModel):
    """Sign in with a password. ``identifier`` accepts an email or a username."""

    model_config = ConfigDict(extra="forbid")

    identifier: Annotated[str, Field(min_length=1, max_length=254)]
    password: Annotated[str, Field(min_length=1, max_length=256)]


class UserResponse(BaseModel):
    """The signed-in account, with nothing secret attached."""

    id: int
    email: str
    username: str
    display_name: str


class ChangePasswordRequest(BaseModel):
    """Replace the signed-in account's password."""

    model_config = ConfigDict(extra="forbid")

    current_password: Annotated[str, Field(min_length=1, max_length=256)]
    new_password: Annotated[str, Field(min_length=8, max_length=256)]


class ForgotPasswordQuestionsRequest(BaseModel):
    """Step 1 of password recovery: look up the questions for an account."""

    model_config = ConfigDict(extra="forbid")

    identifier: Annotated[str, Field(min_length=1, max_length=254)]


class SecurityQuestionsResponse(BaseModel):
    """The two security questions on file, in the order they must be answered."""

    questions: Annotated[list[str], Field(min_length=2, max_length=2)]


class ResetPasswordRequest(BaseModel):
    """Step 2 of password recovery: answer both questions to set a new password."""

    model_config = ConfigDict(extra="forbid")

    identifier: Annotated[str, Field(min_length=1, max_length=254)]
    security_answer_1: Annotated[str, Field(min_length=1, max_length=200)]
    security_answer_2: Annotated[str, Field(min_length=1, max_length=200)]
    new_password: Annotated[str, Field(min_length=8, max_length=256)]
