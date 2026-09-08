"""In-memory artifact store backing the scoring service.

Everything the API serves is precomputed by the pipeline and the scoring scripts,
so a request is a lookup rather than a model invocation. That keeps latency flat
and predictable, and means a request can never be the thing that discovers the
model is broken.

The store loads once at startup. If artifacts are missing the process still
starts and reports itself **unready** - a container that exits on a missing file
gives an operator nothing to inspect, whereas ``/ready`` naming the missing
artifacts tells them exactly what to fix.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from foresight.config import Settings, get_settings
from foresight.logging_setup import get_logger

__all__ = ["ArtifactStore", "get_store"]

log = get_logger(__name__)

#: Artifact name -> (path relative to project root, required for readiness).
_ARTIFACTS: dict[str, tuple[str, bool]] = {
    "risk_table": ("artifacts/risk_table.parquet", True),
    "forecast": ("artifacts/forecast.parquet", True),
    "impact": ("artifacts/impact.json", True),
    "metrics": ("artifacts/metrics.json", True),
    "weekly_panel": ("data/processed/weekly_panel.parquet", True),
    "sku_master": ("data/processed/sku_master_clean.parquet", True),
    "backtest_predictions": ("artifacts/backtest_predictions.parquet", False),
    # Produced by scripts/07_holdout_split.py. Optional: the service works
    # without it, but uploaded actuals for the holdout period score against it.
    "holdout_forecast": ("artifacts/holdout_forecast.parquet", False),
    # The same holdout rows with the outcome joined on, and the metrics computed
    # from them. Powers the test-data view, so what is shown on screen is the
    # table the reported score came from rather than a second, re-derived join.
    "holdout_scored": ("artifacts/holdout_scored.parquet", False),
    "holdout_metrics": ("artifacts/holdout_metrics.json", False),
    "metrics_by_horizon": ("artifacts/metrics_by_horizon.parquet", False),
    "metrics_by_category": ("artifacts/metrics_by_category.parquet", False),
    "metrics_by_regime": ("artifacts/metrics_by_regime.parquet", False),
}


@dataclass(slots=True)
class ArtifactStore:
    """Loaded artifacts plus the indexes the API reads from."""

    settings: Settings
    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    risk_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    weekly_panel: pd.DataFrame = field(default_factory=pd.DataFrame)
    sku_master: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdout_forecast: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdout_scored: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics_by_horizon: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics_by_category: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics_by_regime: pd.DataFrame = field(default_factory=pd.DataFrame)

    impact: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    holdout_metrics: dict[str, Any] = field(default_factory=dict)

    # Fast lookups built once at load time.
    _risk_by_sku: dict[str, dict[str, Any]] = field(default_factory=dict)
    _forecast_by_sku: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _sku_index: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        return not self.missing

    @property
    def origin_week(self) -> dt.date | None:
        value = self.impact.get("origin_week")
        return dt.date.fromisoformat(value) if isinstance(value, str) else None

    @property
    def model_name(self) -> str:
        return str(self.impact.get("model") or self.metrics.get("selected_model") or "unknown")

    @property
    def horizon_weeks(self) -> int:
        return int(self.impact.get("horizon_weeks") or self.settings.horizon_weeks)

    @property
    def sku_ids(self) -> list[str]:
        return sorted(self._sku_index)

    # ------------------------------------------------------------------ #
    def risk_for(self, sku_id: str) -> dict[str, Any] | None:
        return self._risk_by_sku.get(sku_id.strip().upper())

    def forecast_for(self, sku_id: str) -> list[dict[str, Any]]:
        return self._forecast_by_sku.get(sku_id.strip().upper(), [])

    def sku_info(self, sku_id: str) -> dict[str, Any] | None:
        return self._sku_index.get(sku_id.strip().upper())

    def history_for(self, sku_id: str, weeks: int | None = None) -> pd.DataFrame:
        normalised = sku_id.strip().upper()
        if self.weekly_panel.empty:
            return pd.DataFrame()
        rows = self.weekly_panel[self.weekly_panel["sku_id"] == normalised].sort_values(
            "week_start"
        )
        return rows.tail(weeks) if weeks else rows

    def backtest_for(self, sku_id: str) -> pd.DataFrame:
        normalised = sku_id.strip().upper()
        if self.backtest_predictions.empty:
            return pd.DataFrame()
        rows = self.backtest_predictions[self.backtest_predictions["sku_id"] == normalised]
        if rows.empty:
            return rows
        # One fold's forecasts per target week would otherwise overlap; keep the
        # most recent origin's view of each week so the chart shows one line.
        return (
            rows.sort_values(["target_week", "origin_week"])
            .groupby("target_week", as_index=False)
            .tail(1)
            .sort_values("target_week")
        )


def _load(settings: Settings) -> ArtifactStore:
    """Read every artifact from disk into memory."""
    root = settings.artifacts_dir.parent
    store = ArtifactStore(settings=settings)

    frames: dict[str, pd.DataFrame] = {}
    payloads: dict[str, dict[str, Any]] = {}

    for name, (relative, required) in _ARTIFACTS.items():
        path: Path = root / relative
        if not path.exists():
            if required:
                store.missing.append(name)
            log.warning(
                "artifact not found",
                extra={"context": {"artifact": name, "path": str(path), "required": required}},
            )
            continue
        try:
            if path.suffix == ".json":
                payloads[name] = json.loads(path.read_text(encoding="utf-8"))
            else:
                frames[name] = pd.read_parquet(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # A corrupt artifact is as unusable as a missing one, and must not
            # take the whole process down at import time.
            if required:
                store.missing.append(name)
            log.error(
                "artifact failed to load",
                extra={"context": {"artifact": name, "path": str(path), "error": str(exc)}},
            )
            continue
        store.loaded.append(name)

    store.risk_table = frames.get("risk_table", pd.DataFrame())
    store.forecast = frames.get("forecast", pd.DataFrame())
    store.weekly_panel = frames.get("weekly_panel", pd.DataFrame())
    store.sku_master = frames.get("sku_master", pd.DataFrame())
    store.backtest_predictions = frames.get("backtest_predictions", pd.DataFrame())
    store.holdout_forecast = frames.get("holdout_forecast", pd.DataFrame())
    store.holdout_scored = frames.get("holdout_scored", pd.DataFrame())
    store.metrics_by_horizon = frames.get("metrics_by_horizon", pd.DataFrame())
    store.metrics_by_category = frames.get("metrics_by_category", pd.DataFrame())
    store.metrics_by_regime = frames.get("metrics_by_regime", pd.DataFrame())
    store.impact = payloads.get("impact", {})
    store.metrics = payloads.get("metrics", {})
    store.holdout_metrics = payloads.get("holdout_metrics", {})

    # --- Build lookups ------------------------------------------------------ #
    if not store.risk_table.empty:
        store._risk_by_sku = {
            str(record["sku_id"]): record for record in store.risk_table.to_dict(orient="records")
        }

    if not store.forecast.empty:
        for record in store.forecast.sort_values(["sku_id", "horizon"]).to_dict(orient="records"):
            store._forecast_by_sku.setdefault(str(record["sku_id"]), []).append(record)

    if not store.sku_master.empty:
        store._sku_index = {
            str(record["sku_id"]): record for record in store.sku_master.to_dict(orient="records")
        }

    log.info(
        "artifact store loaded",
        extra={
            "context": {
                "loaded": store.loaded,
                "missing": store.missing,
                "skus": len(store._sku_index),
                "ready": store.ready,
            }
        },
    )
    return store


_STORE: ArtifactStore | None = None


def get_store(*, reload: bool = False, settings: Settings | None = None) -> ArtifactStore:
    """Return the process-wide artifact store, loading it on first use."""
    global _STORE
    if _STORE is None or reload:
        _STORE = _load(settings or get_settings())
    return _STORE
