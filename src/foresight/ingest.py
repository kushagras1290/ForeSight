"""Ingestion of the raw NorthBay Living extracts (deliverable D1, step 1).

Everything is read as **text**. That is deliberate: letting pandas infer types on
a messy export is how a column that is 99% integers and 1% ``"N/A"`` silently
becomes object dtype, or how ``lead_time_days`` of ``0`` gets treated as valid.
Parsing is the cleaner's job, where each decision is explicit and recorded.

This module also profiles each table on arrival. The profile is what the
data-quality report (D2) is built from, so the issues we claim to have found are
measured rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd

from foresight.config import Settings, get_settings
from foresight.exceptions import DataError, DataQualityError, MissingDataFileError
from foresight.logging_setup import get_logger
from foresight.schemas import (
    RAW_CALENDAR_SCHEMA,
    RAW_INVENTORY_SCHEMA,
    RAW_SALES_SCHEMA,
    RAW_SKU_MASTER_SCHEMA,
    TableSchema,
    validate_dataframe,
)

__all__ = [
    "ColumnProfile",
    "RAW_TABLES",
    "RawExtracts",
    "TableProfile",
    "load_raw_extracts",
    "profile_table",
]

log = get_logger(__name__)

#: Table name -> (filename, presence-only schema). Required: the pipeline
#: cannot run without these four.
RAW_TABLES: Final[dict[str, tuple[str, TableSchema]]] = {
    "sales_daily": ("sales_daily.csv", RAW_SALES_SCHEMA),
    "sku_master": ("sku_master.csv", RAW_SKU_MASTER_SCHEMA),
    "calendar": ("calendar.csv", RAW_CALENDAR_SCHEMA),
    "inventory_snapshots": ("inventory_snapshots.csv", RAW_INVENTORY_SCHEMA),
}


def _optional_raw_tables() -> dict[str, tuple[str, TableSchema]]:
    """Driver extracts the optional extension knows how to read.

    Imported lazily so core never hard-depends on the extension: with
    ``foresight_drivers`` absent this returns nothing and the pipeline runs on
    the four Appendix A extracts alone.
    """
    try:
        from foresight_drivers import OPTIONAL_RAW_TABLES
    except ImportError:
        log.debug("foresight_drivers not installed; driver extracts will be skipped")
        return {}
    return dict(OPTIONAL_RAW_TABLES)


_MISSING_FILE_HINT: Final = (
    "Extracts are not tracked in version control (brief section 16.3). "
    "Run `python scripts/00_generate_data.py` to materialise them into data/raw/, "
    "or copy the four Appendix A CSVs into that directory."
)


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Per-column measurements taken on arrival, before any cleaning."""

    name: str
    non_null: int
    null_count: int
    null_pct: float
    distinct: int
    sample_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TableProfile:
    """Per-table measurements taken on arrival."""

    name: str
    row_count: int
    column_count: int
    exact_duplicate_rows: int
    columns: tuple[ColumnProfile, ...]

    def column(self, name: str) -> ColumnProfile | None:
        for profile in self.columns:
            if profile.name == name:
                return profile
        return None


@dataclass(slots=True)
class RawExtracts:
    """The four raw tables plus their arrival profiles."""

    sales_daily: pd.DataFrame
    sku_master: pd.DataFrame
    calendar: pd.DataFrame
    inventory_snapshots: pd.DataFrame
    profiles: dict[str, TableProfile] = field(default_factory=dict)

    # Optional commercial drivers; empty when the client cannot supply them.
    marketing_spend: pd.DataFrame = field(default_factory=pd.DataFrame)
    web_analytics: pd.DataFrame = field(default_factory=pd.DataFrame)
    market_conditions: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def has_drivers(self) -> bool:
        """True when at least one commercial driver extract was supplied."""
        return not (
            self.marketing_spend.empty and self.web_analytics.empty and self.market_conditions.empty
        )

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "sales_daily": self.sales_daily,
            "sku_master": self.sku_master,
            "calendar": self.calendar,
            "inventory_snapshots": self.inventory_snapshots,
            "marketing_spend": self.marketing_spend,
            "web_analytics": self.web_analytics,
            "market_conditions": self.market_conditions,
        }


def profile_table(name: str, frame: pd.DataFrame, *, sample_size: int = 5) -> TableProfile:
    """Measure a raw table: nulls, cardinality, duplicates, example values."""
    row_count = len(frame)
    columns: list[ColumnProfile] = []

    for column in frame.columns:
        series = frame[column]
        null_count = int(series.isna().sum())
        non_null = row_count - null_count
        # Sorted so the sample is stable across runs and diffable in the report.
        distinct_values = series.dropna().astype(str).unique()
        sample = tuple(sorted(distinct_values.tolist())[:sample_size])
        columns.append(
            ColumnProfile(
                name=str(column),
                non_null=non_null,
                null_count=null_count,
                null_pct=round(100.0 * null_count / row_count, 3) if row_count else 0.0,
                distinct=int(len(distinct_values)),
                sample_values=sample,
            )
        )

    return TableProfile(
        name=name,
        row_count=row_count,
        column_count=frame.shape[1],
        exact_duplicate_rows=int(frame.duplicated().sum()),
        columns=tuple(columns),
    )


def _read_extract(path: Path, name: str, schema: TableSchema) -> pd.DataFrame:
    """Read one extract as text and confirm the expected columns are present."""
    if not path.exists():
        raise MissingDataFileError(path, _MISSING_FILE_HINT)

    frame = pd.read_csv(
        path,
        dtype=str,  # parse nothing; the cleaner owns every conversion
        keep_default_na=True,
        na_values=["", "NA", "N/A", "n/a", "null", "NULL", "None", "-"],
        skipinitialspace=False,  # leading spaces in sku_id are an issue to detect
    )

    if frame.empty:
        raise DataQualityError(f"Extract '{name}' at {path} contains no rows.")

    # Presence-only: raw extracts are text at this stage, so dtype checks would
    # be meaningless. The full contract is enforced after cleaning.
    validate_dataframe(
        frame, schema, check_dtypes=False, check_nulls=False, check_primary_key=False
    )
    return frame


def load_raw_extracts(settings: Settings | None = None) -> RawExtracts:
    """Load and profile all four client extracts.

    Raises:
        MissingDataFileError: if any extract is absent.
        DataQualityError: if an extract is empty.
        SchemaValidationError: if an extract is missing a required column.
    """
    settings = settings or get_settings()
    frames: dict[str, pd.DataFrame] = {}
    profiles: dict[str, TableProfile] = {}

    for name, (filename, schema) in RAW_TABLES.items():
        path = settings.raw_dir / filename
        frame = _read_extract(path, name, schema)
        frames[name] = frame
        profiles[name] = profile_table(name, frame)
        log.info(
            "loaded raw extract",
            extra={
                "context": {
                    "table": name,
                    "rows": len(frame),
                    "columns": frame.shape[1],
                    "exact_duplicates": profiles[name].exact_duplicate_rows,
                }
            },
        )

    # --- Optional driver extracts ------------------------------------------ #
    optional: dict[str, pd.DataFrame] = {}
    for name, (filename, schema) in _optional_raw_tables().items():
        path = settings.raw_dir / filename
        if not path.exists():
            log.info(
                "optional driver extract not supplied; continuing without it",
                extra={"context": {"table": name, "path": str(path)}},
            )
            continue
        try:
            frame = _read_extract(path, name, schema)
        except DataError as exc:
            # A malformed optional table must not take the whole run down; the
            # forecast is simply built without that driver.
            log.warning(
                "optional driver extract unusable; continuing without it",
                extra={"context": {"table": name, "error": str(exc)}},
            )
            continue
        optional[name] = frame
        profiles[name] = profile_table(name, frame)
        log.info(
            "loaded driver extract",
            extra={"context": {"table": name, "rows": len(frame)}},
        )

    return RawExtracts(
        sales_daily=frames["sales_daily"],
        sku_master=frames["sku_master"],
        calendar=frames["calendar"],
        inventory_snapshots=frames["inventory_snapshots"],
        profiles=profiles,
        marketing_spend=optional.get("marketing_spend", pd.DataFrame()),
        web_analytics=optional.get("web_analytics", pd.DataFrame()),
        market_conditions=optional.get("market_conditions", pd.DataFrame()),
    )
