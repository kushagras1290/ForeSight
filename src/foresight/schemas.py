"""Column contracts for every dataset that moves through the pipeline.

Each table declares the columns it must have and the *kind* of value each holds.
Validation happens at two points:

1. **On ingest** - a presence-only check. Raw client extracts are messy by
   definition (brief section 05), so demanding dtypes there would just crash on
   arrival. We only assert the columns exist.
2. **After cleaning** - a full presence + dtype check. If cleaning did its job,
   the analysis-ready tables must satisfy their contract exactly.

Kinds are checked with ``pandas.api.types`` predicates rather than literal dtype
strings, so the same contract holds under pandas 2.x (object-dtype strings) and
pandas 3.x (native ``str`` dtype).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

import pandas as pd

from foresight.exceptions import SchemaValidationError

__all__ = [
    "ANALYSIS_READY_SCHEMA",
    "CANONICAL_CATEGORIES",
    "CLEAN_CALENDAR_SCHEMA",
    "CLEAN_INVENTORY_SCHEMA",
    "CLEAN_SALES_SCHEMA",
    "CLEAN_SKU_MASTER_SCHEMA",
    "ColumnKind",
    "ColumnSpec",
    "RAW_CALENDAR_SCHEMA",
    "RAW_INVENTORY_SCHEMA",
    "RAW_SALES_SCHEMA",
    "RAW_SKU_MASTER_SCHEMA",
    "RiskAction",
    "RiskLevel",
    "TableSchema",
    "WEEKLY_PANEL_SCHEMA",
    "validate_dataframe",
]


class ColumnKind(StrEnum):
    """The family of values a column may hold."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    ANY = "any"


def _matches_kind(series: pd.Series, kind: ColumnKind) -> bool:
    """Return True when ``series`` holds values of ``kind``."""
    types = pd.api.types
    match kind:
        case ColumnKind.ANY:
            return True
        case ColumnKind.STRING:
            # pandas 3 gives a native `str` dtype; pandas 2 gives object.
            return types.is_string_dtype(series.dtype) or types.is_object_dtype(series.dtype)
        case ColumnKind.INTEGER:
            return types.is_integer_dtype(series.dtype)
        case ColumnKind.FLOAT:
            return types.is_float_dtype(series.dtype)
        case ColumnKind.NUMERIC:
            # bool is numeric to pandas but never what a numeric contract means.
            return types.is_numeric_dtype(series.dtype) and not types.is_bool_dtype(series.dtype)
        case ColumnKind.BOOLEAN:
            return types.is_bool_dtype(series.dtype)
        case ColumnKind.DATETIME:
            return types.is_datetime64_any_dtype(series.dtype)
    return False


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """One column's contract."""

    name: str
    kind: ColumnKind = ColumnKind.ANY
    nullable: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class TableSchema:
    """A table's contract: required columns, optional key, optional uniqueness."""

    name: str
    columns: tuple[ColumnSpec, ...]
    primary_key: tuple[str, ...] = field(default=())

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.columns)


def validate_dataframe(
    frame: pd.DataFrame,
    schema: TableSchema,
    *,
    check_dtypes: bool = True,
    check_nulls: bool = True,
    check_primary_key: bool = True,
) -> None:
    """Assert that ``frame`` satisfies ``schema``.

    Args:
        frame: The dataframe to check.
        schema: The contract to check it against.
        check_dtypes: Verify each column's value kind. Disable for raw extracts.
        check_nulls: Verify non-nullable columns contain no nulls.
        check_primary_key: Verify the primary key is unique and non-null.

    Raises:
        SchemaValidationError: with *every* problem found, not just the first -
            fixing extracts one error per run is miserable.
    """
    problems: list[str] = []

    missing = [spec.name for spec in schema.columns if spec.name not in frame.columns]
    if missing:
        problems.append(f"missing column(s): {', '.join(sorted(missing))}")

    for spec in schema.columns:
        if spec.name not in frame.columns:
            continue
        series = frame[spec.name]

        if check_dtypes and not _matches_kind(series, spec.kind):
            problems.append(
                f"column '{spec.name}' should be {spec.kind.value} but is {series.dtype}"
            )

        if check_nulls and not spec.nullable:
            null_count = int(series.isna().sum())
            if null_count:
                problems.append(
                    f"column '{spec.name}' is non-nullable but has {null_count} null(s)"
                )

    if check_primary_key and schema.primary_key:
        key = list(schema.primary_key)
        if all(column in frame.columns for column in key):
            duplicate_count = int(frame.duplicated(subset=key).sum())
            if duplicate_count:
                problems.append(
                    f"primary key {tuple(key)} is not unique: {duplicate_count} duplicate row(s)"
                )
            key_nulls = int(frame[key].isna().any(axis=1).sum())
            if key_nulls:
                problems.append(f"primary key {tuple(key)} has {key_nulls} row(s) with nulls")

    if problems:
        raise SchemaValidationError(schema.name, problems)


# --------------------------------------------------------------------------- #
# Raw extracts - presence only. These arrive dirty; that is the point.
# --------------------------------------------------------------------------- #
RAW_SALES_SCHEMA: Final = TableSchema(
    name="sales_daily (raw)",
    columns=(
        ColumnSpec("date", ColumnKind.ANY, description="Calendar date of the sales record."),
        ColumnSpec("sku_id", ColumnKind.ANY, description="Product identifier."),
        ColumnSpec("units_sold", ColumnKind.ANY, description="Units sold that day."),
        ColumnSpec("revenue", ColumnKind.ANY, description="Revenue for that SKU-day."),
        ColumnSpec("unit_price", ColumnKind.ANY, description="Selling price that day."),
        ColumnSpec("promo_flag", ColumnKind.ANY, description="1 if on promotion, else 0."),
    ),
)

RAW_SKU_MASTER_SCHEMA: Final = TableSchema(
    name="sku_master (raw)",
    columns=(
        ColumnSpec("sku_id", ColumnKind.ANY),
        ColumnSpec("category", ColumnKind.ANY),
        ColumnSpec("subcategory", ColumnKind.ANY),
        ColumnSpec("launch_date", ColumnKind.ANY),
        ColumnSpec("unit_cost", ColumnKind.ANY),
        ColumnSpec("list_price", ColumnKind.ANY),
    ),
)

RAW_CALENDAR_SCHEMA: Final = TableSchema(
    name="calendar (raw)",
    columns=(
        ColumnSpec("date", ColumnKind.ANY),
        ColumnSpec("week", ColumnKind.ANY),
        ColumnSpec("month", ColumnKind.ANY),
        ColumnSpec("season", ColumnKind.ANY),
        ColumnSpec("is_holiday", ColumnKind.ANY),
        ColumnSpec("promo_event", ColumnKind.ANY),
        # NOTE: `planned_discount` is deliberately NOT required here. Many promo
        # calendars name the event without publishing its depth, and rejecting
        # such a calendar would be wrong. The cleaner defaults it to zero, which
        # reads as "no published markdown".
    ),
)

RAW_INVENTORY_SCHEMA: Final = TableSchema(
    name="inventory_snapshots (raw)",
    columns=(
        ColumnSpec("date", ColumnKind.ANY),
        ColumnSpec("sku_id", ColumnKind.ANY),
        ColumnSpec("on_hand_units", ColumnKind.ANY),
        ColumnSpec("on_order_units", ColumnKind.ANY),
        ColumnSpec("lead_time_days", ColumnKind.ANY),
        ColumnSpec("reorder_point", ColumnKind.ANY),
    ),
)

# Contracts for the optional commercial driver extracts live in the
# `foresight_drivers` package, which owns everything about those fields. Core
# never imports them; the pipeline reaches for them only when the extension is
# installed and enabled.


# --------------------------------------------------------------------------- #
# Cleaned tables - full contract. Cleaning must deliver exactly this.
# --------------------------------------------------------------------------- #
CLEAN_SALES_SCHEMA: Final = TableSchema(
    name="sales_daily (clean)",
    columns=(
        ColumnSpec("date", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("sku_id", ColumnKind.STRING, nullable=False),
        ColumnSpec("units_sold", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("revenue", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("unit_price", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("promo_flag", ColumnKind.INTEGER, nullable=False),
    ),
    primary_key=("date", "sku_id"),
)

CLEAN_SKU_MASTER_SCHEMA: Final = TableSchema(
    name="sku_master (clean)",
    columns=(
        ColumnSpec("sku_id", ColumnKind.STRING, nullable=False),
        ColumnSpec("category", ColumnKind.STRING, nullable=False),
        ColumnSpec("subcategory", ColumnKind.STRING, nullable=False),
        ColumnSpec("launch_date", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("unit_cost", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("list_price", ColumnKind.NUMERIC, nullable=False),
    ),
    primary_key=("sku_id",),
)

CLEAN_CALENDAR_SCHEMA: Final = TableSchema(
    name="calendar (clean)",
    columns=(
        ColumnSpec("date", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("iso_year", ColumnKind.INTEGER, nullable=False),
        ColumnSpec("iso_week", ColumnKind.INTEGER, nullable=False),
        ColumnSpec("week_start", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("month", ColumnKind.INTEGER, nullable=False),
        ColumnSpec("season", ColumnKind.STRING, nullable=False),
        ColumnSpec("is_holiday", ColumnKind.INTEGER, nullable=False),
        ColumnSpec("promo_event", ColumnKind.STRING, nullable=False),
        ColumnSpec("planned_discount", ColumnKind.NUMERIC, nullable=False),
    ),
    primary_key=("date",),
)

CLEAN_INVENTORY_SCHEMA: Final = TableSchema(
    name="inventory_snapshots (clean)",
    columns=(
        ColumnSpec("date", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("sku_id", ColumnKind.STRING, nullable=False),
        ColumnSpec("on_hand_units", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("on_order_units", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("lead_time_days", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("reorder_point", ColumnKind.NUMERIC, nullable=False),
    ),
    primary_key=("date", "sku_id"),
)


# --------------------------------------------------------------------------- #
# Modelling tables
# --------------------------------------------------------------------------- #
ANALYSIS_READY_SCHEMA: Final = TableSchema(
    name="analysis_ready (daily)",
    columns=(
        ColumnSpec("date", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("sku_id", ColumnKind.STRING, nullable=False),
        ColumnSpec("units_sold", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("revenue", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("unit_price", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("promo_flag", ColumnKind.INTEGER, nullable=False),
        ColumnSpec("category", ColumnKind.STRING, nullable=False),
        ColumnSpec("subcategory", ColumnKind.STRING, nullable=False),
        ColumnSpec("launch_date", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("unit_cost", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("list_price", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("week_start", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("is_holiday", ColumnKind.INTEGER, nullable=False),
        ColumnSpec("promo_event", ColumnKind.STRING, nullable=False),
        ColumnSpec("season", ColumnKind.STRING, nullable=False),
    ),
    primary_key=("date", "sku_id"),
)

WEEKLY_PANEL_SCHEMA: Final = TableSchema(
    name="weekly_panel",
    columns=(
        ColumnSpec("sku_id", ColumnKind.STRING, nullable=False),
        ColumnSpec("week_start", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("units", ColumnKind.NUMERIC, nullable=False, description="Weekly demand."),
        ColumnSpec("revenue", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("avg_price", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("promo_days", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("holiday_days", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("category", ColumnKind.STRING, nullable=False),
        ColumnSpec("subcategory", ColumnKind.STRING, nullable=False),
        ColumnSpec("unit_cost", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("list_price", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("launch_date", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("weeks_since_launch", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("is_post_launch", ColumnKind.INTEGER, nullable=False),
        # Commercial drivers. Zero-filled when the optional extracts are absent,
        # so the contract holds either way and the model degrades to
        # history-only forecasting rather than failing.
        ColumnSpec("media_spend", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("impressions", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("email_sends", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("sessions", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("add_to_cart", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("discount_pct", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("competitor_price_index", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("weather_anomaly", ColumnKind.NUMERIC, nullable=False),
    ),
    primary_key=("sku_id", "week_start"),
)


# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #
CANONICAL_CATEGORIES: Final[tuple[str, ...]] = (
    "Bedding & Bath",
    "Lighting",
    "Kitchen & Dining",
    "Decor",
    "Storage & Organisation",
    "Small Appliances",
)

#: NorthBay encode the category into the SKU id as ``NBL-<code>-<nnn>``.
#: This is a client naming convention, visible by inspecting sku_master, and the
#: cleaner relies on it to recover categories that are missing or unmappable.
CATEGORY_CODES: Final[dict[str, str]] = {
    "Bedding & Bath": "BTH",
    "Lighting": "LGT",
    "Kitchen & Dining": "KTD",
    "Decor": "DEC",
    "Storage & Organisation": "STO",
    "Small Appliances": "APP",
}

SKU_PREFIX_TO_CATEGORY: Final[dict[str, str]] = {
    code: category for category, code in CATEGORY_CODES.items()
}

#: Value used when a category cannot be mapped or recovered at all.
UNCLASSIFIED_CATEGORY: Final[str] = "Unclassified"

CANONICAL_SUBCATEGORIES: Final[dict[str, tuple[str, ...]]] = {
    "Bedding & Bath": ("Bed Linen", "Towels", "Duvets & Pillows", "Bath Mats"),
    "Lighting": ("Table Lamps", "Floor Lamps", "String Lights", "Ceiling Fixtures"),
    "Kitchen & Dining": ("Cookware", "Dinnerware", "Glassware", "Storage Jars"),
    "Decor": ("Cushions", "Wall Art", "Vases", "Candles & Holders"),
    "Storage & Organisation": ("Baskets", "Shelving", "Wardrobe Organisers", "Boxes & Bins"),
    "Small Appliances": ("Kettles", "Blenders", "Air Purifiers", "Coffee Makers"),
}


class RiskLevel(StrEnum):
    """Severity band attached to each risk dimension."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(StrEnum):
    """The four quadrants of the decisioning grid (brief section 8.2).

    Values match the dashboard's colour tokens and the API contract exactly, so
    a rename here propagates everywhere instead of drifting.
    """

    REORDER_NOW = "reorder_now"
    MARKDOWN_CLEAR = "markdown_clear"
    WATCH_VOLATILE = "watch_volatile"
    HEALTHY = "healthy"


RISK_ACTION_LABELS: Final[dict[RiskAction, str]] = {
    RiskAction.REORDER_NOW: "Reorder now",
    RiskAction.MARKDOWN_CLEAR: "Markdown / clear",
    RiskAction.WATCH_VOLATILE: "Watch / volatile",
    RiskAction.HEALTHY: "Healthy",
}

RISK_ACTION_RATIONALE: Final[dict[RiskAction, str]] = {
    RiskAction.REORDER_NOW: "High stockout risk, low overstock risk - raise a replenishment order before stock runs out.",
    RiskAction.MARKDOWN_CLEAR: "High overstock risk, low stockout risk - promote or discount to free up capital.",
    RiskAction.WATCH_VOLATILE: "High on both - demand is erratic relative to cover; review manually.",
    RiskAction.HEALTHY: "Low on both - no action needed.",
}
