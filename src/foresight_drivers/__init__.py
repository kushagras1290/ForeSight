"""Commercial driver extension for Project FORESIGHT.

WHY THIS IS A SEPARATE PACKAGE
------------------------------
The engagement brief specifies four extracts (Appendix A): sales, product
master, calendar and inventory. That is the deliverable, and ``foresight`` on
its own implements exactly it.

This package adds a *second, optional* layer: the commercial data a D2C brand
actually holds beyond those four tables - media spend, storefront analytics and
competitor pricing. It lives apart on purpose:

* the core pipeline stays a faithful implementation of the brief;
* the extension can be removed wholesale without touching core code;
* the accuracy contribution of the extra data can be measured by turning it off,
  rather than argued about.

``foresight`` imports from here through narrow, clearly-marked seams and treats
every one of them as optional. With this package absent, or with
``FORESIGHT_USE_DRIVERS=false``, the system forecasts from history alone and
every contract still holds.

WHAT IT ADDS
------------
``marketing_spend``     weekly media plan per SKU. Committed in advance, so it
                        extends past the sales history and is legitimately known
                        at forecast time.
``web_analytics``       daily sessions and add-to-cart per SKU. Observed only up
                        to the origin, so it is used strictly as a lagged
                        leading indicator.
``market_conditions``   weekly competitor price index and weather anomaly per
                        category. Persistent, so the origin value carries
                        information about the target week.

THE RULE THIS PACKAGE EXISTS TO ENFORCE
---------------------------------------
A driver is only usable for week ``t + h`` if it is genuinely knowable at week
``t``. Media plans and published promo depth are; site traffic and realised
discounts are not. The split is enforced in :mod:`foresight_drivers.features`,
where forward-known drivers and observed-only drivers are built by different
functions and never mixed.
"""

from __future__ import annotations

from foresight_drivers.cleaning import (
    clean_market_conditions,
    clean_marketing,
    clean_web_analytics,
)
from foresight_drivers.features import (
    DRIVER_ORIGIN_FEATURES,
    DRIVER_TARGET_FEATURES,
    attach_target_drivers,
    build_driver_features,
    driver_columns_with_defaults,
)
from foresight_drivers.schemas import (
    CLEAN_MARKET_CONDITIONS_SCHEMA,
    CLEAN_MARKETING_SCHEMA,
    CLEAN_WEB_ANALYTICS_SCHEMA,
    OPTIONAL_RAW_TABLES,
    RAW_MARKET_CONDITIONS_SCHEMA,
    RAW_MARKETING_SCHEMA,
    RAW_WEB_ANALYTICS_SCHEMA,
)

__all__ = [
    "CLEAN_MARKETING_SCHEMA",
    "CLEAN_MARKET_CONDITIONS_SCHEMA",
    "CLEAN_WEB_ANALYTICS_SCHEMA",
    "DRIVER_ORIGIN_FEATURES",
    "DRIVER_TARGET_FEATURES",
    "OPTIONAL_RAW_TABLES",
    "RAW_MARKETING_SCHEMA",
    "RAW_MARKET_CONDITIONS_SCHEMA",
    "RAW_WEB_ANALYTICS_SCHEMA",
    "attach_target_drivers",
    "build_driver_features",
    "clean_market_conditions",
    "clean_marketing",
    "clean_web_analytics",
    "driver_columns_with_defaults",
]

__version__ = "1.0.0"
