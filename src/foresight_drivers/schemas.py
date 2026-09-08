"""Column contracts for the commercial driver extracts.

Raw contracts are presence-only, matching how ``foresight.schemas`` treats the
four core extracts: these files arrive as text and the cleaner owns every
conversion. The clean contracts are fully typed.
"""

from __future__ import annotations

from typing import Final

from foresight.schemas import ColumnKind, ColumnSpec, TableSchema

__all__ = [
    "CLEAN_MARKETING_SCHEMA",
    "CLEAN_MARKET_CONDITIONS_SCHEMA",
    "CLEAN_WEB_ANALYTICS_SCHEMA",
    "OPTIONAL_RAW_TABLES",
    "RAW_MARKETING_SCHEMA",
    "RAW_MARKET_CONDITIONS_SCHEMA",
    "RAW_WEB_ANALYTICS_SCHEMA",
]


RAW_MARKETING_SCHEMA: Final = TableSchema(
    name="marketing_spend (raw)",
    columns=(
        ColumnSpec("week_start", ColumnKind.ANY, description="Monday of the plan week."),
        ColumnSpec("sku_id", ColumnKind.ANY),
        ColumnSpec("media_spend", ColumnKind.ANY, description="Planned media spend, INR."),
        ColumnSpec("impressions", ColumnKind.ANY),
        ColumnSpec("email_sends", ColumnKind.ANY),
    ),
)

RAW_WEB_ANALYTICS_SCHEMA: Final = TableSchema(
    name="web_analytics (raw)",
    columns=(
        ColumnSpec("date", ColumnKind.ANY),
        ColumnSpec("sku_id", ColumnKind.ANY),
        ColumnSpec("sessions", ColumnKind.ANY, description="Product page sessions."),
        ColumnSpec("add_to_cart", ColumnKind.ANY),
    ),
)

RAW_MARKET_CONDITIONS_SCHEMA: Final = TableSchema(
    name="market_conditions (raw)",
    columns=(
        ColumnSpec("week_start", ColumnKind.ANY),
        ColumnSpec("category", ColumnKind.ANY),
        ColumnSpec("competitor_price_index", ColumnKind.ANY),
        ColumnSpec("weather_anomaly", ColumnKind.ANY),
    ),
)

CLEAN_MARKETING_SCHEMA: Final = TableSchema(
    name="marketing_spend (clean)",
    columns=(
        ColumnSpec("week_start", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("sku_id", ColumnKind.STRING, nullable=False),
        ColumnSpec("media_spend", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("impressions", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("email_sends", ColumnKind.NUMERIC, nullable=False),
    ),
    primary_key=("week_start", "sku_id"),
)

CLEAN_WEB_ANALYTICS_SCHEMA: Final = TableSchema(
    name="web_analytics (clean)",
    columns=(
        ColumnSpec("date", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("sku_id", ColumnKind.STRING, nullable=False),
        ColumnSpec("sessions", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("add_to_cart", ColumnKind.NUMERIC, nullable=False),
    ),
    primary_key=("date", "sku_id"),
)

CLEAN_MARKET_CONDITIONS_SCHEMA: Final = TableSchema(
    name="market_conditions (clean)",
    columns=(
        ColumnSpec("week_start", ColumnKind.DATETIME, nullable=False),
        ColumnSpec("category", ColumnKind.STRING, nullable=False),
        ColumnSpec("competitor_price_index", ColumnKind.NUMERIC, nullable=False),
        ColumnSpec("weather_anomaly", ColumnKind.NUMERIC, nullable=False),
    ),
    primary_key=("week_start", "category"),
)

#: Driver table name -> (filename, presence-only schema). Every one is optional.
OPTIONAL_RAW_TABLES: Final[dict[str, tuple[str, TableSchema]]] = {
    "marketing_spend": ("marketing_spend.csv", RAW_MARKETING_SCHEMA),
    "web_analytics": ("web_analytics.csv", RAW_WEB_ANALYTICS_SCHEMA),
    "market_conditions": ("market_conditions.csv", RAW_MARKET_CONDITIONS_SCHEMA),
}
