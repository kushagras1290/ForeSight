"""FORESIGHT scoring service (deliverable D6).

A read-only HTTP API returning the demand forecast and inventory risk for any
SKU, singly or in batch. It also backs the planning dashboard (D5), so there is
one source of truth for what the models say - the dashboard cannot drift away
from the API, because it has no numbers of its own.

Run locally::

    uvicorn service.main:app --reload --port 8000

Interactive documentation is served at ``/docs``; the OpenAPI schema at
``/openapi.json`` satisfies D6 acceptance criterion 3.

Design notes
------------
* **Read-only.** Nothing here mutates client data or places orders - explicitly
  out of scope per brief section 4.3. Every route is a GET except batch scoring,
  which is a POST only because the SKU list can exceed a sane URL length.
* **Precomputed.** Requests are lookups against artifacts built by the pipeline,
  so latency does not depend on model inference.
* **Degrades honestly.** Missing artifacts leave the service live but not ready,
  and ``/ready`` names exactly what is absent.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from foresight import __version__
from foresight.config import get_settings
from foresight.evaluate import (
    MAX_UPLOAD_BYTES,
    classify_csv,
    evaluate_actuals,
    parse_actuals_csv,
    parse_forecast_csv,
)
from foresight.exceptions import DataQualityError, RiskScoringError
from foresight.logging_setup import get_logger
from foresight.risk import score_risk
from service.models import (
    AccuracySummary,
    ApiError,
    BacktestPoint,
    BatchForecastRequest,
    BatchForecastResponse,
    Envelope,
    ErrorCode,
    ErrorContributor,
    EvaluationBreakdown,
    EvaluationRequest,
    EvaluationResponse,
    ForecastPoint,
    HealthResponse,
    HistoryPoint,
    HoldoutSummary,
    LiveScoreRequest,
    LiveScoreResponse,
    MetricBlock,
    PortfolioSummary,
    ReadyResponse,
    RiskRecord,
    SkuDetailResponse,
    SkuForecastResponse,
    SkuSummary,
)
from service.store import ArtifactStore, get_store

log = get_logger("service")

settings = get_settings()

# Written as literals on purpose. Starlette has renamed both of these constants
# and referencing the old names emits a DeprecationWarning, while the new names
# do not exist on older versions. The numbers themselves are fixed by RFC 9110.
HTTP_422: Final[int] = 422  # Unprocessable Content
HTTP_413: Final[int] = 413  # Content Too Large

#: Files accepted in one evaluation upload. Generous enough for a forecast plus
#: a year of quarterly actuals exports, low enough that a mis-selected folder
#: fails fast rather than being parsed one file at a time.
MAX_UPLOAD_FILES: Final[int] = 12

# --- Rate limiting ---------------------------------------------------------- #
# A public endpoint with no limit is an availability incident waiting to happen.
# This is a per-client sliding window, deliberately simple: it is in-process, so
# it protects a single instance rather than a fleet. Anything running behind a
# load balancer should enforce limits at the edge as well.
RATE_LIMIT_REQUESTS = 120
RATE_LIMIT_WINDOW_SECONDS = 60
_REQUEST_LOG: dict[str, deque[float]] = defaultdict(deque)
#: Bound the tracking table so a spray of spoofed IPs cannot exhaust memory.
_MAX_TRACKED_CLIENTS = 10_000


# Product identifiers are validated at the edge rather than being allowed to
# fall through to a lookup miss. Anything that cannot be a SKU is rejected as
# malformed (422) instead of reported as "not found" (404), which is both more
# accurate and keeps arbitrary strings out of the lookup path entirely.
SKU_PATH = PathParam(
    min_length=3,
    max_length=32,
    pattern=r"^[A-Za-z0-9][A-Za-z0-9\-_]{1,31}$",
    description="Product identifier, e.g. NBL-LGT-004.",
)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _rate_limited(key: str) -> bool:
    now = time.monotonic()
    window = _REQUEST_LOG[key]
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    while window and window[0] < cutoff:
        window.popleft()
    if not window and len(_REQUEST_LOG) > _MAX_TRACKED_CLIENTS:
        _REQUEST_LOG.pop(key, None)
        return False
    if len(window) >= RATE_LIMIT_REQUESTS:
        return True
    window.append(now)
    return False


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts once at startup."""
    store = get_store(reload=True)
    app.state.store = store
    log.info(
        "service starting",
        extra={"context": {"ready": store.ready, "missing": store.missing, "version": __version__}},
    )
    yield
    log.info("service stopping")


app = FastAPI(
    title=settings.api_title,
    version=__version__,
    lifespan=lifespan,
    description=(
        "Demand forecast and inventory risk for NorthBay Living. "
        "Read-only: this service reports what the models say and never places orders."
    ),
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,  # no cookies or auth are used; keep it off
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=600,
)


# --------------------------------------------------------------------------- #
# Middleware
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Any]]) -> Any:
    """Attach a request id, enforce the rate limit, and log the outcome."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()

    if request.url.path.startswith("/api") and _rate_limited(_client_key(request)):
        log.warning(
            "rate limit exceeded",
            extra={"context": {"request_id": request_id, "path": request.url.path}},
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=Envelope(
                success=False,
                error=ApiError(
                    code=ErrorCode.RATE_LIMITED,
                    message=(
                        f"Too many requests. The limit is {RATE_LIMIT_REQUESTS} per "
                        f"{RATE_LIMIT_WINDOW_SECONDS} seconds."
                    ),
                ),
            ).model_dump(mode="json"),
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS), "X-Request-ID": request_id},
        )

    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.headers["X-Request-ID"] = request_id

    log.info(
        "request",
        extra={
            "context": {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(elapsed_ms, 2),
            }
        },
    )
    return response


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
@app.exception_handler(RequestValidationError)
async def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Turn pydantic's validation output into the standard envelope.

    FastAPI calls exception handlers positionally with ``(request, exc)``, so the
    request parameter is required by the signature even where it is unused.
    """
    problems = [
        {"field": ".".join(str(part) for part in error["loc"][1:]) or "body", "issue": error["msg"]}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=HTTP_422,
        content=Envelope(
            success=False,
            error=ApiError(
                code=ErrorCode.VALIDATION_ERROR,
                message="The request could not be understood. See details.",
                details={"problems": problems},
            ),
        ).model_dump(mode="json"),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Render a raised HTTPException into the standard envelope."""
    code = (
        exc.detail.get("code")
        if isinstance(exc.detail, dict)
        else ErrorCode.NOT_FOUND
        if exc.status_code == 404
        else ErrorCode.INTERNAL_ERROR
    )
    message = exc.detail.get("message") if isinstance(exc.detail, dict) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=Envelope(
            success=False, error=ApiError(code=ErrorCode(code), message=message)
        ).model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, _exc: Exception) -> JSONResponse:
    """Last resort. Log the detail; return nothing internal to the caller.

    The exception itself is captured by ``log.exception`` from the active
    handler context, so it is never interpolated into the response body.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    log.exception(
        "unhandled error",
        extra={"context": {"request_id": request_id, "path": request.url.path}},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=Envelope(
            success=False,
            error=ApiError(
                code=ErrorCode.INTERNAL_ERROR,
                message=f"An unexpected error occurred. Request id {request_id}.",
            ),
        ).model_dump(mode="json"),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _store(request: Request) -> ArtifactStore:
    store: ArtifactStore = request.app.state.store
    if not store.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": ErrorCode.ARTIFACTS_UNAVAILABLE.value,
                "message": (
                    "The service is running but has no scored data to serve. "
                    f"Missing: {', '.join(store.missing)}. "
                    "Run the pipeline and scoring scripts, then restart."
                ),
            },
        )
    return store


def _as_date(value: Any) -> dt.date:
    return pd.Timestamp(value).date() if not isinstance(value, dt.date) else value


def _risk_record(record: dict[str, Any]) -> RiskRecord:
    return RiskRecord(
        sku_id=str(record["sku_id"]),
        category=str(record.get("category", "")),
        subcategory=str(record.get("subcategory", "")),
        on_hand_units=float(record["on_hand_units"]),
        on_order_units=float(record["on_order_units"]),
        available_units=float(record["available_units"]),
        lead_time_days=float(record["lead_time_days"]),
        inventory_as_of=_as_date(record["inventory_as_of"]),
        forecast_lead_time_units=round(float(record["lead_time_demand"]), 2),
        forecast_horizon_units=round(float(record["horizon_demand"]), 2),
        cover_weeks=round(float(record["cover_weeks"]), 2),
        stockout_score=round(float(record["stockout_score"]), 4),
        overstock_score=round(float(record["overstock_score"]), 4),
        stockout_level=str(record["stockout_level"]),
        overstock_level=str(record["overstock_level"]),
        action=str(record["action"]),
        action_label=str(record["action_label"]),
        action_rationale=str(record["action_rationale"]),
        recommended_order_units=float(record["recommended_order_units"]),
        safety_stock_units=round(float(record["safety_stock_units"]), 2),
        expected_lost_units=round(float(record["expected_lost_units"]), 2),
        excess_units=float(record["excess_units"]),
        service_level=float(record.get("service_level", 0.95)),
        revenue_at_risk=float(record["revenue_at_risk"]),
        margin_at_risk=float(record["margin_at_risk"]),
        locked_capital=float(record["locked_capital"]),
        value_at_stake=float(record["value_at_stake"]),
        priority_rank=int(record["priority_rank"]),
        forecast_confidence=str(record.get("forecast_confidence", "high")),
    )


def _forecast_points(rows: list[dict[str, Any]]) -> list[ForecastPoint]:
    return [
        ForecastPoint(
            week_starting=_as_date(row["target_week"]),
            horizon=int(row["horizon"]),
            forecast_units=round(float(row["prediction"]), 2),
            lower_units=round(float(row["prediction_lower"]), 2),
            upper_units=round(float(row["prediction_upper"]), 2),
        )
        for row in rows
    ]


def _sku_or_404(store: ArtifactStore, sku_id: str) -> tuple[str, dict[str, Any]]:
    normalised = sku_id.strip().upper()
    info = store.sku_info(normalised)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": ErrorCode.NOT_FOUND.value,
                "message": (
                    f"No product named '{sku_id}'. "
                    "Call /api/skus for the list of valid identifiers."
                ),
            },
        )
    return normalised, info


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["operations"])
async def health() -> HealthResponse:
    """Liveness probe. Always cheap; never touches artifacts."""
    return HealthResponse(
        service=settings.api_title,
        version=__version__,
        time=dt.datetime.now(tz=dt.UTC),
    )


@app.get("/ready", response_model=ReadyResponse, tags=["operations"])
async def ready(request: Request) -> ReadyResponse:
    """Readiness probe. Reports exactly which artifacts are absent, if any."""
    store: ArtifactStore = request.app.state.store
    return ReadyResponse(
        ready=store.ready,
        artifacts_loaded=store.loaded,
        artifacts_missing=store.missing,
        model=store.model_name if store.ready else None,
        origin_week=store.origin_week,
        skus=len(store.sku_ids),
    )


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #
@app.get("/api/skus", response_model=Envelope[list[SkuSummary]], tags=["catalogue"])
async def list_skus(
    request: Request,
    category: str | None = Query(default=None, max_length=64),
    search: str | None = Query(default=None, max_length=64),
) -> Envelope[list[SkuSummary]]:
    """List products, optionally filtered by category or a free-text search."""
    store = _store(request)
    frame = store.sku_master

    if category:
        frame = frame[frame["category"].str.casefold() == category.strip().casefold()]
    if search:
        needle = search.strip().casefold()
        frame = frame[
            frame["sku_id"].str.casefold().str.contains(needle, regex=False)
            | frame["subcategory"].str.casefold().str.contains(needle, regex=False)
        ]

    return Envelope(
        data=[
            SkuSummary(
                sku_id=str(row["sku_id"]),
                category=str(row["category"]),
                subcategory=str(row["subcategory"]),
            )
            for row in frame.sort_values("sku_id").to_dict(orient="records")
        ]
    )


@app.get("/api/categories", response_model=Envelope[list[str]], tags=["catalogue"])
async def list_categories(request: Request) -> Envelope[list[str]]:
    """Distinct product categories, for dashboard filters."""
    store = _store(request)
    return Envelope(data=sorted(store.sku_master["category"].unique().tolist()))


# --------------------------------------------------------------------------- #
# Forecast and risk
# --------------------------------------------------------------------------- #
@app.get(
    "/api/forecast/{sku_id}",
    response_model=Envelope[SkuForecastResponse],
    tags=["forecast"],
    responses={404: {"description": "Unknown product identifier."}},
)
async def forecast_one(request: Request, sku_id: str = SKU_PATH) -> Envelope[SkuForecastResponse]:
    """Forecast and risk for a single product."""
    store = _store(request)
    normalised, info = _sku_or_404(store, sku_id)

    rows = store.forecast_for(normalised)
    risk = store.risk_for(normalised)

    return Envelope(
        data=SkuForecastResponse(
            sku_id=normalised,
            category=str(info["category"]),
            subcategory=str(info["subcategory"]),
            model=store.model_name,
            origin_week=store.origin_week or dt.date.today(),
            horizon_weeks=store.horizon_weeks,
            forecast=_forecast_points(rows),
            risk=_risk_record(risk) if risk else None,
        )
    )


@app.post(
    "/api/forecast/batch",
    response_model=Envelope[BatchForecastResponse],
    tags=["forecast"],
)
async def forecast_batch(
    request: Request, payload: BatchForecastRequest
) -> Envelope[BatchForecastResponse]:
    """Forecast and risk for many products at once.

    Unknown identifiers are returned in ``not_found`` rather than failing the
    whole request - a batch of 200 should not be rejected because one code was
    mistyped.
    """
    store = _store(request)

    if len(payload.sku_ids) > settings.api_max_batch_size:
        raise HTTPException(
            status_code=HTTP_413,
            detail={
                "code": ErrorCode.BATCH_TOO_LARGE.value,
                "message": (
                    f"Batch of {len(payload.sku_ids)} exceeds the limit of "
                    f"{settings.api_max_batch_size}. Split it into smaller requests."
                ),
            },
        )

    results: list[SkuForecastResponse] = []
    not_found: list[str] = []

    for sku_id in payload.sku_ids:
        info = store.sku_info(sku_id)
        if info is None:
            not_found.append(sku_id)
            continue
        risk = store.risk_for(sku_id)
        results.append(
            SkuForecastResponse(
                sku_id=sku_id,
                category=str(info["category"]),
                subcategory=str(info["subcategory"]),
                model=store.model_name,
                origin_week=store.origin_week or dt.date.today(),
                horizon_weeks=store.horizon_weeks,
                forecast=_forecast_points(store.forecast_for(sku_id)),
                risk=_risk_record(risk) if risk else None,
            )
        )

    return Envelope(
        data=BatchForecastResponse(
            requested=len(payload.sku_ids),
            returned=len(results),
            not_found=not_found,
            results=results,
        )
    )


@app.post("/api/score", response_model=Envelope[LiveScoreResponse], tags=["forecast"])
async def score_live(request: Request, payload: LiveScoreRequest) -> Envelope[LiveScoreResponse]:
    """Re-score risk against stock positions supplied at request time.

    **This is the endpoint to use operationally.** The rest of the API serves the
    weekly batch, which is scored against the stock position as of the last
    snapshot. Stock moves every day; the demand forecast does not. So rather than
    re-running the pipeline to reflect a delivery that landed this morning, send
    the current position here and the risk layer is recomputed against it
    immediately, using the same trained model and the same arithmetic as the
    batch.

    Only ``on_hand_units`` is required per product. Anything omitted is taken
    from the stored snapshot.
    """
    store = _store(request)

    if len(payload.positions) > settings.api_max_batch_size:
        raise HTTPException(
            status_code=HTTP_413,
            detail={
                "code": ErrorCode.BATCH_TOO_LARGE.value,
                "message": (
                    f"{len(payload.positions)} positions exceeds the limit of "
                    f"{settings.api_max_batch_size}."
                ),
            },
        )

    known: list[dict[str, Any]] = []
    not_found: list[str] = []

    for position in payload.positions:
        sku_id = position.sku_id.strip().upper()
        stored = store.risk_for(sku_id)
        if stored is None or not store.forecast_for(sku_id):
            not_found.append(sku_id)
            continue
        known.append(
            {
                "date": pd.Timestamp(store.origin_week or dt.date.today()),
                "sku_id": sku_id,
                "on_hand_units": float(position.on_hand_units),
                # Fall back to the stored snapshot for anything not supplied.
                "on_order_units": float(
                    position.on_order_units
                    if position.on_order_units is not None
                    else stored["on_order_units"]
                ),
                "lead_time_days": float(
                    position.lead_time_days
                    if position.lead_time_days is not None
                    else stored["lead_time_days"]
                ),
                "reorder_point": float(stored.get("reorder_point", 0.0)),
            }
        )

    if not known:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": ErrorCode.NOT_FOUND.value,
                "message": (
                    "None of the supplied products are known to the service. "
                    "Call /api/skus for the list of valid identifiers."
                ),
            },
        )

    inventory = pd.DataFrame.from_records(known)
    sku_ids = set(inventory["sku_id"])
    forecast = store.forecast[store.forecast["sku_id"].isin(sku_ids)]
    master = store.sku_master[store.sku_master["sku_id"].isin(sku_ids)]

    try:
        scored = score_risk(forecast, inventory, master, settings)
    except RiskScoringError as exc:
        # A caller-supplied position that the risk layer cannot use is a bad
        # request, not a server fault.
        raise HTTPException(
            status_code=HTTP_422,
            detail={"code": ErrorCode.VALIDATION_ERROR.value, "message": str(exc)},
        ) from exc

    # Forecast confidence is a property of the product's history, not of the
    # position supplied, so it is carried across from the stored assessment
    # rather than recomputed (and silently defaulted) here.
    confidence = {
        sku_id: (store.risk_for(sku_id) or {}).get("forecast_confidence", "high")
        for sku_id in sku_ids
    }
    scored["forecast_confidence"] = scored["sku_id"].map(confidence).fillna("high")

    origin = store.origin_week or dt.date.today()
    now = dt.datetime.now(tz=dt.UTC)

    log.info(
        "live scoring",
        extra={
            "context": {
                "request_id": getattr(request.state, "request_id", None),
                "requested": len(payload.positions),
                "scored": len(scored),
                "not_found": len(not_found),
            }
        },
    )

    return Envelope(
        data=LiveScoreResponse(
            scored_at=now,
            forecast_origin_week=origin,
            forecast_age_days=(now.date() - origin).days,
            model=store.model_name,
            requested=len(payload.positions),
            returned=len(scored),
            not_found=not_found,
            results=[_risk_record(record) for record in scored.to_dict(orient="records")],
        )
    )


def _run_evaluation(
    store: ArtifactStore,
    actuals: pd.DataFrame,
    *,
    uploaded_forecast: pd.DataFrame | None = None,
) -> EvaluationResponse:
    """Shared body for both evaluation entry points.

    When a forecast is uploaded it replaces the stored one entirely, and the
    backtest and holdout fallbacks are dropped with it. Mixing an external
    forecast with this service's own rows would produce a score belonging to
    neither, so the caller gets exactly what they supplied, measured.
    """
    external = uploaded_forecast is not None and not uploaded_forecast.empty
    empty = pd.DataFrame()
    try:
        result = evaluate_actuals(
            actuals,
            forecast=uploaded_forecast if external else store.forecast,
            backtest=empty if external else store.backtest_predictions,
            panel=store.weekly_panel,
            holdout=None if external else store.holdout_forecast,
            interval_coverage_target=settings.interval_coverage,
            sku_master=store.sku_master,
        )
    except DataQualityError as exc:
        # Everything this raises is written for the person who uploaded the file.
        raise HTTPException(
            status_code=HTTP_422,
            detail={"code": ErrorCode.VALIDATION_ERROR.value, "message": str(exc)},
        ) from exc

    def block(metrics: Any) -> MetricBlock:
        payload = metrics.to_dict()
        return MetricBlock(
            n_observations=int(payload["n_observations"]),
            total_actual=float(payload["total_actual"]),
            total_predicted=float(payload["total_predicted"]),
            wape=float(payload["wape"]),
            mape=float(payload["mape"]),
            mape_coverage=float(payload["mape_coverage"]),
            bias_relative=float(payload["bias_relative"]),
            mae=float(payload["mae"]),
            rmse=float(payload["rmse"]),
        )

    def breakdown(frame: pd.DataFrame) -> list[EvaluationBreakdown]:
        return [
            EvaluationBreakdown(
                key=str(row["key"]),
                n_observations=int(row["n_observations"]),
                total_actual=float(row["total_actual"]),
                wape=float(row["wape"]),
                baseline_wape=(
                    float(row["baseline_wape"]) if pd.notna(row["baseline_wape"]) else None
                ),
                bias_relative=float(row["bias_relative"])
                if pd.notna(row["bias_relative"])
                else 0.0,
            )
            for _, row in frame.iterrows()
        ]

    improvement = result.improvement_vs_baseline

    return EvaluationResponse(
        rows_submitted=result.rows_submitted,
        rows_matched=result.rows_matched,
        rows_unmatched=result.rows_submitted - result.rows_matched,
        unmatched_examples=result.unmatched_examples,
        weeks_covered=result.weeks_covered,
        source_counts=result.source_counts,
        model=store.model_name,
        accuracy=block(result.model),
        baseline=block(result.baseline) if result.baseline is not None else None,
        baseline_row_coverage=result.baseline_coverage,
        improvement_vs_baseline=float(improvement) if np.isfinite(improvement) else None,
        interval_coverage=result.interval_coverage,
        interval_coverage_target=settings.interval_coverage,
        by_horizon=breakdown(result.by_horizon),
        by_category=breakdown(result.by_category),
        worst_contributors=[
            ErrorContributor(
                sku_id=str(row["sku_id"]),
                n_observations=int(row["n_observations"]),
                total_actual=float(row["total_actual"]),
                total_forecast=float(row["total_forecast"]),
                absolute_error=float(row["absolute_error"]),
                share_of_total_error=float(row["share_of_total_error"]),
            )
            for _, row in result.worst_contributors.iterrows()
        ],
    )


@app.post("/api/evaluate", response_model=Envelope[EvaluationResponse], tags=["accuracy"])
async def evaluate(request: Request, payload: EvaluationRequest) -> Envelope[EvaluationResponse]:
    """Score the shipped forecast against actuals supplied as JSON.

    The backtest says how the model performed on history. This says how it is
    performing **now** — once the weeks a plan covered have happened, the client
    has ground truth on the forecast they actually acted on.

    Results are reported against the same seasonal-naive baseline the backtest
    used, so the comparison is like for like.
    """
    store = _store(request)
    actuals = pd.DataFrame(
        [
            {
                "sku_id": item.sku_id,
                "week_starting": pd.Timestamp(item.week_starting),
                "units": float(item.units),
            }
            for item in payload.actuals
        ]
    )
    return Envelope(data=_run_evaluation(store, actuals))


@app.post("/api/evaluate/upload", response_model=Envelope[EvaluationResponse], tags=["accuracy"])
async def evaluate_upload(
    request: Request,
    files: list[UploadFile] = File(
        description="One or more CSVs: actual weekly demand, and optionally a forecast."
    ),
) -> Envelope[EvaluationResponse]:
    """Score a forecast against uploaded actuals.

    Accepts several files at once, and works out what each one is from its own
    headers rather than from the order they arrive in:

    * **Actuals** — a product code, a week date and a quantity. Header spellings
      are matched leniently (``sku``/``product``/``item``, ``week``/``date``,
      ``units``/``qty``/``sales``), because client exports do not use our column
      names and rejecting a file over a header is a poor trade. Several actuals
      files are concatenated, so a year split across quarterly exports can be
      uploaded in one go.
    * **Forecast** — recognised by an unambiguously predictive column
      (``prediction``, ``forecast``, ``yhat``). Supplying one scores the actuals
      against *that* forecast rather than against whatever this service has
      stored, which is what makes a complete external test set measurable end to
      end.

    Files are parsed in memory and never written to disk.
    """
    store = _store(request)

    if not files:
        raise HTTPException(
            status_code=HTTP_422,
            detail={
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": "No file was uploaded.",
            },
        )
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=HTTP_422,
            detail={
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": f"{len(files)} files supplied; the limit is {MAX_UPLOAD_FILES}.",
            },
        )

    actual_frames: list[pd.DataFrame] = []
    uploaded_forecast: pd.DataFrame | None = None
    forecast_filename: str | None = None
    total_bytes = 0

    for upload in files:
        content = await upload.read()
        total_bytes += len(content)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=HTTP_413,
                detail={
                    "code": ErrorCode.BATCH_TOO_LARGE.value,
                    "message": (
                        f"Upload totals {total_bytes / 1e6:.1f} MB; the limit is "
                        f"{MAX_UPLOAD_BYTES / 1e6:.0f} MB across all files."
                    ),
                },
            )

        try:
            kind = classify_csv(content)
            if kind == "forecast":
                if uploaded_forecast is not None:
                    raise HTTPException(
                        status_code=HTTP_422,
                        detail={
                            "code": ErrorCode.VALIDATION_ERROR.value,
                            "message": (
                                "Two forecast files were uploaded "
                                f"({forecast_filename} and {upload.filename}). "
                                "Supply one forecast and any number of actuals files."
                            ),
                        },
                    )
                uploaded_forecast = parse_forecast_csv(content)
                forecast_filename = upload.filename
            else:
                actual_frames.append(parse_actuals_csv(content))
        except DataQualityError as exc:
            raise HTTPException(
                status_code=HTTP_422,
                detail={
                    "code": ErrorCode.VALIDATION_ERROR.value,
                    "message": f"{upload.filename or 'file'}: {exc}",
                },
            ) from exc

    if not actual_frames:
        raise HTTPException(
            status_code=HTTP_422,
            detail={
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": (
                    "Only a forecast was uploaded. Add a file of actual demand — "
                    "a product code, a week date and a quantity — to score it against."
                ),
            },
        )

    actuals = (
        pd.concat(actual_frames, ignore_index=True)
        if len(actual_frames) > 1
        else (actual_frames[0])
    )
    # A week appearing in two uploaded files is one observation, not two.
    actuals = actuals.drop_duplicates(subset=["sku_id", "week_starting"], keep="last")

    log.info(
        "evaluation upload received",
        extra={
            "context": {
                "request_id": getattr(request.state, "request_id", None),
                "files": len(files),
                "filenames": [upload.filename for upload in files],
                "bytes": total_bytes,
                "actual_rows": len(actuals),
                "forecast_rows": 0 if uploaded_forecast is None else len(uploaded_forecast),
                "scored_against": "uploaded_forecast"
                if uploaded_forecast is not None
                else "stored",
            }
        },
    )
    return Envelope(data=_run_evaluation(store, actuals, uploaded_forecast=uploaded_forecast))


@app.get("/api/risk", response_model=Envelope[list[RiskRecord]], tags=["risk"])
async def risk_list(
    request: Request,
    action: str | None = Query(default=None, max_length=32),
    category: str | None = Query(default=None, max_length=64),
    search: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> Envelope[list[RiskRecord]]:
    """The prioritised worklist, filtered and paged.

    Ordered by rupee value at stake, so the first page is always the work that
    matters most.
    """
    store = _store(request)
    frame = store.risk_table

    if action:
        frame = frame[frame["action"] == action.strip().lower()]
    if category:
        frame = frame[frame["category"].str.casefold() == category.strip().casefold()]
    if search:
        needle = search.strip().casefold()
        frame = frame[frame["sku_id"].str.casefold().str.contains(needle, regex=False)]

    page = frame.sort_values("priority_rank").iloc[offset : offset + limit]
    return Envelope(data=[_risk_record(record) for record in page.to_dict(orient="records")])


@app.get("/api/sku/{sku_id}", response_model=Envelope[SkuDetailResponse], tags=["risk"])
async def sku_detail(
    request: Request,
    sku_id: str = SKU_PATH,
    history_weeks: int = Query(default=104, ge=8, le=520),
) -> Envelope[SkuDetailResponse]:
    """Everything about one product: history, backtest, forecast and risk."""
    store = _store(request)
    normalised, info = _sku_or_404(store, sku_id)

    history = store.history_for(normalised, weeks=history_weeks)
    backtest = store.backtest_for(normalised)
    prediction_column = "ensemble" if "ensemble" in backtest.columns else "prediction"

    return Envelope(
        data=SkuDetailResponse(
            sku_id=normalised,
            category=str(info["category"]),
            subcategory=str(info["subcategory"]),
            unit_cost=float(info["unit_cost"]),
            list_price=float(info["list_price"]),
            history=[
                HistoryPoint(
                    week_starting=_as_date(row["week_start"]),
                    units=float(row["units"]),
                    revenue=float(row["revenue"]),
                    promo_days=float(row["promo_days"]),
                )
                for row in history.to_dict(orient="records")
            ],
            backtest=[
                BacktestPoint(
                    week_starting=_as_date(row["target_week"]),
                    actual=float(row["y"]),
                    forecast=round(float(row[prediction_column]), 2),
                    baseline=round(float(row["baseline"]), 2),
                    lower=round(float(row.get("ensemble_lower", row["prediction_lower"])), 2),
                    upper=round(float(row.get("ensemble_upper", row["prediction_upper"])), 2),
                )
                for row in backtest.to_dict(orient="records")
            ],
            forecast=_forecast_points(store.forecast_for(normalised)),
            risk=_risk_record(store.risk_for(normalised)) if store.risk_for(normalised) else None,
        )
    )


# --------------------------------------------------------------------------- #
# Portfolio
# --------------------------------------------------------------------------- #
@app.get("/api/summary", response_model=Envelope[PortfolioSummary], tags=["portfolio"])
async def summary(request: Request) -> Envelope[PortfolioSummary]:
    """Headline portfolio numbers: how many SKUs need what, and the rupee exposure."""
    store = _store(request)
    impact = store.impact
    return Envelope(
        data=PortfolioSummary(
            origin_week=store.origin_week or dt.date.today(),
            horizon_weeks=store.horizon_weeks,
            model=store.model_name,
            total_skus=int(impact["total_skus"]),
            reorder_now_skus=int(impact["reorder_now_skus"]),
            markdown_skus=int(impact["markdown_skus"]),
            watch_skus=int(impact["watch_skus"]),
            healthy_skus=int(impact["healthy_skus"]),
            low_confidence_skus=int(impact["low_confidence_skus"]),
            revenue_at_risk_total=float(impact["revenue_at_risk_total"]),
            margin_at_risk_total=float(impact["margin_at_risk_total"]),
            locked_capital_total=float(impact["locked_capital_total"]),
            expected_lost_units_total=float(impact["expected_lost_units_total"]),
            excess_units_total=float(impact["excess_units_total"]),
            reorder_now_order_units=float(impact["reorder_now_order_units"]),
            top_10_share_of_revenue_at_risk=float(impact["top_10_share_of_revenue_at_risk"]),
        )
    )


@app.get("/api/accuracy", response_model=Envelope[AccuracySummary], tags=["portfolio"])
async def accuracy(request: Request) -> Envelope[AccuracySummary]:
    """How the shipped forecast performed on the rolling-origin backtest.

    Exposed deliberately. The ops team should be able to see how much to trust
    the number they are acting on, without asking anyone.
    """
    store = _store(request)
    metrics = store.metrics

    return Envelope(
        data=AccuracySummary(
            selected_model=str(metrics.get("selected_model", "unknown")),
            selected_wape=float(metrics.get("selected_wape", float("nan"))),
            baseline_wape=float(metrics.get("baseline_wape", float("nan"))),
            naive_wape=float(metrics.get("naive_wape", float("nan"))),
            gbm_wape=float(metrics.get("gbm_wape", float("nan"))),
            ensemble_wape=(float(metrics["ensemble_wape"]) if "ensemble_wape" in metrics else None),
            improvement_vs_baseline=float(metrics.get("selected_improvement_vs_baseline", 0.0)),
            interval_coverage=float(metrics.get("selected_interval_coverage", float("nan"))),
            interval_coverage_target=float(metrics.get("interval_coverage_target", 0.8)),
            bias_relative=float(
                metrics.get("ensemble_bias_relative", metrics.get("gbm_bias_relative", 0.0))
            ),
            folds=int(metrics.get("folds", 0)),
            test_observations=int(metrics.get("test_observations", 0)),
            horizon_weeks=int(metrics.get("horizon_weeks", store.horizon_weeks)),
            by_horizon=store.metrics_by_horizon.to_dict(orient="records"),
            by_category=store.metrics_by_category.to_dict(orient="records"),
            by_regime=store.metrics_by_regime.to_dict(orient="records"),
        )
    )


@app.get("/api/holdout", response_model=Envelope[HoldoutSummary], tags=["accuracy"])
async def holdout(request: Request) -> Envelope[HoldoutSummary]:
    """The chronological 70/30 test: forecasts against outcomes never trained on.

    The backtest at ``/api/accuracy`` walks an origin forward and averages six
    folds. This is the blunter question a sceptic asks instead: train once on the
    first 70% of history, then forecast the remaining year and see what happened.
    Reported separately because it is a different, harsher test - the model has
    30% less history and no chance to re-learn a drift part-way through.

    Returns ``available: false`` rather than an error when the holdout has not
    been run, so the dashboard can show a prompt instead of a failure.
    """
    store = _store(request)
    scored = store.holdout_scored
    metrics = store.holdout_metrics

    if scored.empty:
        return Envelope(
            data=HoldoutSummary(
                available=False,
                reason="No holdout has been run yet. Run scripts/07_holdout_split.py.",
            )
        )

    frame = scored.copy()
    frame["target_week"] = pd.to_datetime(frame["target_week"])
    total_actual = float(frame["actual"].abs().sum())

    def _wape(group: pd.DataFrame) -> float:
        denominator = float(group["actual"].abs().sum())
        if denominator <= 0.0:
            return float("nan")
        return float((group["prediction"] - group["actual"]).abs().sum() / denominator)

    # --- One row per holdout week: the shape a reviewer reads first --------- #
    weekly = (
        frame.groupby("target_week", as_index=False)
        .agg(
            actual=("actual", "sum"),
            predicted=("prediction", "sum"),
            baseline=("baseline", "sum"),
            lower=("prediction_lower", "sum"),
            upper=("prediction_upper", "sum"),
            skus=("sku_id", "nunique"),
            covered=("within_interval", "mean"),
        )
        .sort_values("target_week")
    )
    weekly["week"] = weekly["target_week"].dt.strftime("%Y-%m-%d")
    weekly["error_pct"] = (weekly["predicted"] - weekly["actual"]) / weekly["actual"].replace(
        0.0, float("nan")
    )
    weekly_records = weekly.drop(columns=["target_week"]).to_dict(orient="records")

    # --- Breakdowns --------------------------------------------------------- #
    by_category = [
        {
            "category": str(name),
            "rows": int(len(group)),
            "actual": float(group["actual"].sum()),
            "predicted": float(group["prediction"].sum()),
            "wape": round(_wape(group), 6),
            "accuracy": round(1.0 - _wape(group), 6),
            "share_of_units": round(float(group["actual"].sum()) / total_actual, 6)
            if total_actual > 0
            else 0.0,
        }
        for name, group in frame.groupby("category", sort=True)
    ]

    by_horizon = [
        {
            "horizon": int(name),
            "rows": int(len(group)),
            "wape": round(_wape(group), 6),
            "accuracy": round(1.0 - _wape(group), 6),
            "coverage": round(float(group["within_interval"].mean()), 6),
        }
        for name, group in frame.groupby("horizon", sort=True)
    ]

    # --- Per-SKU scorecard, worst first ------------------------------------- #
    per_sku = frame.groupby(["sku_id", "category"], as_index=False).agg(
        weeks=("actual", "size"),
        actual=("actual", "sum"),
        predicted=("prediction", "sum"),
        abs_error=("abs_error", "sum"),
        coverage=("within_interval", "mean"),
    )
    per_sku["wape"] = per_sku["abs_error"] / per_sku["actual"].replace(0.0, float("nan"))
    per_sku["accuracy"] = 1.0 - per_sku["wape"]
    per_sku["bias"] = (per_sku["predicted"] - per_sku["actual"]) / per_sku["actual"].replace(
        0.0, float("nan")
    )
    # Rank by units mis-forecast rather than by WAPE: a 300% error on a SKU
    # selling two units a week is not the row anyone should look at first.
    per_sku = per_sku.sort_values("abs_error", ascending=False)
    sku_records = (
        per_sku.replace([float("inf"), float("-inf")], float("nan"))
        .where(per_sku.notna(), None)
        .to_dict(orient="records")
    )

    return Envelope(
        data=HoldoutSummary(
            available=True,
            split_week=str(metrics.get("split_week", "")) or None,
            holdout_start=str(metrics.get("holdout_start", "")) or None,
            holdout_end=str(metrics.get("holdout_end", "")) or None,
            train_weeks=int(metrics.get("train_weeks", 0)),
            holdout_weeks=int(metrics.get("holdout_weeks", 0)),
            train_fraction=float(metrics.get("train_fraction", 0.0)),
            observations=int(metrics.get("holdout_observations", len(frame))),
            model_name=str(metrics.get("model", store.model_name)),
            model_wape=float(metrics.get("model_wape", _wape(frame))),
            baseline_wape=float(metrics.get("baseline_wape", float("nan"))),
            improvement_vs_baseline=float(metrics.get("improvement_vs_baseline", 0.0)),
            bias_relative=float(metrics.get("model_bias_relative", 0.0)),
            interval_coverage=float(metrics.get("interval_coverage", float("nan"))),
            interval_coverage_target=float(store.settings.interval_coverage),
            weekly=weekly_records,
            by_category=by_category,
            by_horizon=by_horizon,
            skus=sku_records,
        )
    )


@app.get("/api/grid", response_model=Envelope[list[dict[str, Any]]], tags=["risk"])
async def decisioning_grid(request: Request) -> Envelope[list[dict[str, Any]]]:
    """Every SKU as a point on the stockout-versus-overstock grid.

    Matches the decisioning view in section 8.2 of the brief: position carries
    the meaning, and the marker is sized by the rupee value at stake.
    """
    store = _store(request)
    columns = [
        "sku_id",
        "category",
        "subcategory",
        "stockout_score",
        "overstock_score",
        "action",
        "action_label",
        "value_at_stake",
        "revenue_at_risk",
        "locked_capital",
        "cover_weeks",
        "available_units",
        "forecast_confidence",
    ]
    frame = store.risk_table.loc[:, columns]
    return Envelope(data=frame.to_dict(orient="records"))


# --------------------------------------------------------------------------- #
# Dashboard static files (optional single-deployable mode)
# --------------------------------------------------------------------------- #
_DASHBOARD_DIST = Path(__file__).resolve().parents[1] / "dashboard" / "dist"
if _DASHBOARD_DIST.is_dir():
    # Mounted last so it never shadows an /api or /health route.
    app.mount("/", StaticFiles(directory=_DASHBOARD_DIST, html=True), name="dashboard")
    log.info("serving dashboard build", extra={"context": {"path": str(_DASHBOARD_DIST)}})
