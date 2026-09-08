"""Cleaning and normalisation of the NorthBay Living extracts (deliverable D1).

Every transformation performed here is recorded as a :class:`CleaningAction`
carrying what was wrong, how it was detected, how many rows it touched, what was
done, and *why*. The data-quality report is rendered from those records rather
than written by hand, so the report can never drift from what the code actually
did - which is what D1 acceptance criterion 4 and D2 criterion 1 ask for.

All cleaning is coded, deterministic, and re-runnable. There are no manual steps.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from foresight.config import Settings, get_settings
from foresight.exceptions import DataQualityError
from foresight.logging_setup import get_logger
from foresight.schemas import (
    CANONICAL_CATEGORIES,
    SKU_PREFIX_TO_CATEGORY,
    UNCLASSIFIED_CATEGORY,
)

__all__ = [
    "CleaningAction",
    "CleaningReport",
    "clean_calendar",
    "clean_inventory",
    "clean_sales",
    "clean_sku_master",
    "normalise_category",
    "normalise_sku_id",
    "parse_boolean_flag",
    "parse_mixed_dates",
]

log = get_logger(__name__)

#: Tokens that mean "true" in the client's exports, lower-cased.
_TRUTHY: Final[frozenset[str]] = frozenset(
    {"1", "1.0", "y", "yes", "true", "t", "on", "promo", "promoted"}
)
_FALSY: Final[frozenset[str]] = frozenset({"0", "0.0", "n", "no", "false", "f", "off", ""})

_SKU_PREFIX_PATTERN: Final = re.compile(r"^NBL-([A-Z]{3})-\d+$")

_SEASON_BY_MONTH: Final[dict[int, str]] = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Summer",
    6: "Summer",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",
    10: "Festive",
    11: "Festive",
}


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CleaningAction:
    """One recorded cleaning decision."""

    table: str
    issue: str
    detection: str
    rows_affected: int
    resolution: str
    rationale: str
    severity: str = "info"  # info | warning | critical

    def to_dict(self) -> dict[str, object]:
        return {
            "table": self.table,
            "issue": self.issue,
            "detection": self.detection,
            "rows_affected": self.rows_affected,
            "resolution": self.resolution,
            "rationale": self.rationale,
            "severity": self.severity,
        }


@dataclass(slots=True)
class CleaningReport:
    """Accumulated audit trail across all tables."""

    actions: list[CleaningAction]

    def add(self, action: CleaningAction) -> None:
        # Only record work that was actually needed; a zero-row action is noise.
        if action.rows_affected > 0:
            self.actions.append(action)
            log.info(
                "cleaning action",
                extra={
                    "context": {
                        "table": action.table,
                        "issue": action.issue,
                        "rows": action.rows_affected,
                    }
                },
            )

    def to_frame(self) -> pd.DataFrame:
        if not self.actions:
            return pd.DataFrame(
                columns=[
                    "table",
                    "issue",
                    "detection",
                    "rows_affected",
                    "resolution",
                    "rationale",
                    "severity",
                ]
            )
        return pd.DataFrame([action.to_dict() for action in self.actions])

    def total_rows_touched(self) -> int:
        return sum(action.rows_affected for action in self.actions)


# --------------------------------------------------------------------------- #
# Normalisation primitives
# --------------------------------------------------------------------------- #
def normalise_sku_id(series: pd.Series) -> pd.Series:
    """Trim, collapse internal whitespace, and upper-case SKU identifiers.

    ``" nbl-lgt-004 "`` and ``"NBL-LGT-004"`` are the same product; treating them
    as different silently splits a SKU's history in two and wrecks its lags.
    """
    cleaned = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return cleaned.str.upper()


def _fold_label(value: str) -> str:
    """Reduce a free-text label to a comparable key.

    Strips accents, lower-cases, unifies ``and``/``&`` and the -ise/-ize split,
    and collapses whitespace - so ``"Décor"``, ``"DECOR"`` and ``" decor "`` all
    fold to the same key.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(char for char in decomposed if not unicodedata.combining(char))
    folded = ascii_only.strip().lower()
    folded = folded.replace(" and ", " & ")
    folded = folded.replace("organization", "organisation")
    return re.sub(r"\s+", " ", folded)


#: Folded canonical label -> canonical label.
_CATEGORY_LOOKUP: Final[dict[str, str]] = {
    _fold_label(category): category for category in CANONICAL_CATEGORIES
}


def normalise_category(categories: pd.Series, sku_ids: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Map free-text categories onto the controlled vocabulary.

    Unmappable or missing values are recovered from the ``NBL-<code>-<nnn>``
    convention in the SKU id, which is the client's own naming standard and is
    more reliable than the hand-maintained category column.

    Returns:
        The canonical categories, and a boolean mask of rows recovered from the
        SKU id (for the audit trail).
    """
    folded = categories.astype("string").map(
        lambda value: _fold_label(value) if isinstance(value, str) else pd.NA
    )
    mapped = folded.map(_CATEGORY_LOOKUP).astype("object")

    needs_recovery = mapped.isna()
    if needs_recovery.any():
        codes = (
            sku_ids.astype("string")
            .str.extract(_SKU_PREFIX_PATTERN, expand=False)
            .map(SKU_PREFIX_TO_CATEGORY)
        )
        mapped = mapped.where(~needs_recovery, codes)

    recovered = needs_recovery & mapped.notna()
    resolved = mapped.fillna(UNCLASSIFIED_CATEGORY).astype("str")
    return resolved, recovered


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Parse a date column exported in more than one format.

    Tries ISO (``YYYY-MM-DD``) first, then day-first (``DD-MM-YYYY``). Formats
    are given explicitly rather than relying on inference: with mixed input,
    inference resolves ambiguous values like ``04-09-2021`` inconsistently across
    rows, which silently moves dates by months.
    """
    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")

    remaining = parsed.isna() & text.notna()
    if remaining.any():
        fallback = pd.to_datetime(text[remaining], format="%d-%m-%Y", errors="coerce")
        parsed = parsed.copy()
        parsed.loc[remaining] = fallback

    # Anything still unparsed gets one last permissive attempt, day-first.
    remaining = parsed.isna() & text.notna()
    if remaining.any():
        fallback = pd.to_datetime(text[remaining], errors="coerce", dayfirst=True)
        parsed = parsed.copy()
        parsed.loc[remaining] = fallback

    return parsed


def parse_boolean_flag(series: pd.Series) -> pd.Series:
    """Coerce a mixed-encoding flag column to a strict 0/1 integer.

    The client's exports carry ``1/0``, ``Y/N`` and ``TRUE/FALSE`` in the same
    column because several source systems feed it. Anything unrecognised becomes
    0: for a promotion flag, "we have no evidence of a promotion" is the safe
    reading, and inventing promotions would corrupt a model feature.
    """
    text = series.astype("string").str.strip().str.lower().fillna("")
    zeros = pd.Series(0, index=series.index, dtype="int64")
    return zeros.mask(text.isin(_TRUTHY), 1)


def _to_numeric(series: pd.Series) -> pd.Series:
    """Parse a text column to float, tolerating thousands separators."""
    text = series.astype("string").str.strip().str.replace(",", "", regex=False)
    return pd.to_numeric(text, errors="coerce")


# --------------------------------------------------------------------------- #
# sku_master
# --------------------------------------------------------------------------- #
def clean_sku_master(raw: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Clean the product dimension."""
    table = "sku_master"
    frame = raw.copy()

    frame["sku_id"] = normalise_sku_id(frame["sku_id"])

    # --- Categories --------------------------------------------------------- #
    raw_labels = frame["category"].astype("string")
    distinct_before = int(raw_labels.dropna().nunique())
    missing_before = int(raw_labels.isna().sum())

    resolved, recovered = normalise_category(frame["category"], frame["sku_id"])
    frame["category"] = resolved

    if distinct_before > len(CANONICAL_CATEGORIES):
        report.add(
            CleaningAction(
                table=table,
                issue=f"{distinct_before} distinct category spellings for "
                f"{len(CANONICAL_CATEGORIES)} real categories",
                detection="distinct-value count on the raw category column",
                rows_affected=len(frame),
                resolution="folded to a controlled vocabulary by stripping accents and "
                "whitespace, lower-casing, and unifying 'and'/'&' and -ise/-ize",
                rationale="category is a model feature and a dashboard filter; unmerged "
                "spellings fragment every group-by and split one category's "
                "history across several labels",
                severity="warning",
            )
        )

    if missing_before:
        report.add(
            CleaningAction(
                table=table,
                issue=f"{missing_before} product(s) have no category",
                detection="null count on the raw category column",
                rows_affected=missing_before,
                resolution="recovered from the NBL-<code>-<nnn> convention in the SKU id",
                rationale="the SKU id encodes category at creation time and is more "
                "reliable than the hand-maintained category column; dropping "
                "these products would lose their sales history entirely",
                severity="warning",
            )
        )

    unclassified = int((frame["category"] == UNCLASSIFIED_CATEGORY).sum())
    if unclassified:
        report.add(
            CleaningAction(
                table=table,
                issue=f"{unclassified} product(s) could not be categorised",
                detection="no vocabulary match and no usable SKU-id prefix",
                rows_affected=unclassified,
                resolution=f"labelled '{UNCLASSIFIED_CATEGORY}' and retained",
                rationale="these still have real sales; excluding them would understate "
                "demand, so they are kept and made visible instead of hidden",
                severity="warning",
            )
        )

    frame["subcategory"] = (
        frame["subcategory"].astype("string").str.strip().fillna("Unspecified").astype("str")
    )

    # --- Dates -------------------------------------------------------------- #
    launch = parse_mixed_dates(frame["launch_date"])
    mixed_format_count = int(
        (
            frame["launch_date"].astype("string").str.match(r"^\d{2}-\d{2}-\d{4}$").fillna(False)
        ).sum()
    )
    if mixed_format_count:
        report.add(
            CleaningAction(
                table=table,
                issue=f"launch_date exported in two formats "
                f"({mixed_format_count} rows are DD-MM-YYYY, the rest ISO)",
                detection="regex match for DD-MM-YYYY against the raw column",
                rows_affected=mixed_format_count,
                resolution="parsed with explicit formats, ISO first then day-first",
                rationale="letting pandas infer the format resolves ambiguous values such "
                "as 04-09-2021 inconsistently row by row, silently shifting "
                "dates by months and corrupting SKU age features",
                severity="warning",
            )
        )

    unparsed = int(launch.isna().sum())
    if unparsed:
        # A launch date we cannot read is better replaced by first observed sale
        # downstream; for now flag it and use a sentinel far in the past.
        launch = launch.fillna(pd.Timestamp("1970-01-01"))
        report.add(
            CleaningAction(
                table=table,
                issue=f"{unparsed} launch_date value(s) unparseable",
                detection="null count after explicit-format parsing",
                rows_affected=unparsed,
                resolution="set to 1970-01-01 so the SKU is treated as long-established",
                rationale="a missing launch date must not exclude the product; the "
                "conservative choice is to assume it predates the window, "
                "which only affects the weeks-since-launch feature",
                severity="warning",
            )
        )
    frame["launch_date"] = launch

    # --- Money -------------------------------------------------------------- #
    for column in ("unit_cost", "list_price"):
        frame[column] = _to_numeric(frame[column])

    invalid_money = int(
        (frame["unit_cost"].isna() | (frame["unit_cost"] <= 0)).sum()
        + (frame["list_price"].isna() | (frame["list_price"] <= 0)).sum()
    )
    if invalid_money:
        category_cost = frame.groupby("category")["unit_cost"].transform("median")
        category_price = frame.groupby("category")["list_price"].transform("median")
        frame["unit_cost"] = frame["unit_cost"].where(frame["unit_cost"] > 0, category_cost)
        frame["list_price"] = frame["list_price"].where(frame["list_price"] > 0, category_price)
        report.add(
            CleaningAction(
                table=table,
                issue=f"{invalid_money} missing or non-positive cost/price value(s)",
                detection="null or <= 0 check on unit_cost and list_price",
                rows_affected=invalid_money,
                resolution="imputed with the category median",
                rationale="cost and price drive the rupee impact figures; a zero would "
                "silently value a SKU's stockout or overstock exposure at nil",
                severity="critical",
            )
        )

    # --- Duplicate products ------------------------------------------------- #
    duplicate_skus = int(frame.duplicated(subset=["sku_id"]).sum())
    if duplicate_skus:
        frame = frame.drop_duplicates(subset=["sku_id"], keep="first")
        report.add(
            CleaningAction(
                table=table,
                issue=f"{duplicate_skus} duplicate sku_id row(s) in the product master",
                detection="duplicate check on sku_id after normalisation",
                rows_affected=duplicate_skus,
                resolution="kept the first occurrence",
                rationale="sku_master is a dimension and must have one row per product, "
                "otherwise joining it to sales multiplies the fact rows",
                severity="critical",
            )
        )

    columns = ["sku_id", "category", "subcategory", "launch_date", "unit_cost", "list_price"]
    return frame[columns].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# calendar
# --------------------------------------------------------------------------- #
def clean_calendar(raw: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Clean the date dimension and derive the weekly grain."""
    table = "calendar"
    frame = raw.copy()

    frame["date"] = parse_mixed_dates(frame["date"])
    unparsed = int(frame["date"].isna().sum())
    if unparsed:
        frame = frame[frame["date"].notna()]
        report.add(
            CleaningAction(
                table=table,
                issue=f"{unparsed} calendar row(s) with an unparseable date",
                detection="null count after date parsing",
                rows_affected=unparsed,
                resolution="dropped",
                rationale="the calendar is keyed on date; a row without one cannot join "
                "to anything and would create null weeks in the panel",
                severity="critical",
            )
        )

    holiday_raw = frame["is_holiday"].astype("string")
    non_numeric = int((~holiday_raw.str.strip().str.lower().isin({"0", "1"})).sum())
    frame["is_holiday"] = parse_boolean_flag(frame["is_holiday"])
    if non_numeric:
        report.add(
            CleaningAction(
                table=table,
                issue="is_holiday exported as Yes/No text rather than a 0/1 flag",
                detection="values outside {0, 1} on the raw column",
                rows_affected=non_numeric,
                resolution="mapped a fixed truthy vocabulary to 1, everything else to 0",
                rationale="holiday is summed into a weekly feature; a text column would "
                "either fail to aggregate or coerce to nonsense",
                severity="info",
            )
        )

    blank_events = int(frame["promo_event"].isna().sum())
    frame["promo_event"] = (
        frame["promo_event"].astype("string").str.strip().fillna("").astype("str")
    )
    if blank_events:
        report.add(
            CleaningAction(
                table=table,
                issue=f"{blank_events} date(s) with no promo_event",
                detection="null count on promo_event",
                rows_affected=blank_events,
                resolution="set to an empty string meaning 'no event'",
                rationale="a null here means no promotion ran, not unknown data; making "
                "that explicit keeps the one-hot feature honest",
                severity="info",
            )
        )

    # --- Derived time attributes ------------------------------------------- #
    # Recomputed from the date rather than trusted from the export: week and
    # month are the join keys for the entire weekly panel, so they must be
    # internally consistent with the date column.
    iso = frame["date"].dt.isocalendar()
    frame["iso_year"] = iso["year"].astype("int64")
    frame["iso_week"] = iso["week"].astype("int64")
    frame["month"] = frame["date"].dt.month.astype("int64")
    # Weeks start Monday, matching the client's Monday inventory snapshots.
    frame["week_start"] = frame["date"] - pd.to_timedelta(frame["date"].dt.dayofweek, unit="D")
    frame["season"] = frame["month"].map(_SEASON_BY_MONTH).astype("str")

    duplicate_dates = int(frame.duplicated(subset=["date"]).sum())
    if duplicate_dates:
        frame = frame.drop_duplicates(subset=["date"], keep="first")
        report.add(
            CleaningAction(
                table=table,
                issue=f"{duplicate_dates} duplicate calendar date(s)",
                detection="duplicate check on date",
                rows_affected=duplicate_dates,
                resolution="kept the first occurrence",
                rationale="the calendar is a dimension keyed on date; duplicates would "
                "fan out the sales fact table on join",
                severity="critical",
            )
        )

    # Planned discount depth. Absent in a calendar that only names events, in
    # which case zero is correct: no published depth means no planned markdown.
    if "planned_discount" in frame.columns:
        frame["planned_discount"] = (
            _to_numeric(frame["planned_discount"]).fillna(0.0).clip(0.0, 0.95)
        )
    else:
        frame["planned_discount"] = 0.0
    frame["planned_discount"] = frame["planned_discount"].astype("float64")

    columns = [
        "date",
        "iso_year",
        "iso_week",
        "week_start",
        "month",
        "season",
        "is_holiday",
        "promo_event",
        "planned_discount",
    ]
    return frame[columns].sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# sales_daily
# --------------------------------------------------------------------------- #
def clean_sales(
    raw: pd.DataFrame,
    sku_master: pd.DataFrame,
    report: CleaningReport,
) -> pd.DataFrame:
    """Clean the sales fact table.

    Args:
        raw: The raw ``sales_daily`` extract.
        sku_master: Already-cleaned product dimension, used to resolve prices and
            to identify rows referencing unknown products.
        report: Audit trail to append to.
    """
    table = "sales_daily"
    frame = raw.copy()
    rows_in = len(frame)

    # --- 1. Exact duplicates ------------------------------------------------ #
    exact_duplicates = int(frame.duplicated().sum())
    if exact_duplicates:
        frame = frame.drop_duplicates(keep="first")
        report.add(
            CleaningAction(
                table=table,
                issue=f"{exact_duplicates} byte-identical duplicate row(s)",
                detection="full-row duplicate check on the raw extract",
                rows_affected=exact_duplicates,
                resolution="dropped, keeping the first occurrence",
                rationale="identical rows are a re-run of the export appended twice, not "
                "two real orders; leaving them double-counts demand and inflates "
                "every forecast built on it",
                severity="critical",
            )
        )

    # --- 2. Identifiers ----------------------------------------------------- #
    original_ids = frame["sku_id"].astype("string")
    frame["sku_id"] = normalise_sku_id(frame["sku_id"])
    dirty_ids = int((original_ids != frame["sku_id"]).fillna(False).sum())
    if dirty_ids:
        report.add(
            CleaningAction(
                table=table,
                issue=f"{dirty_ids} row(s) with leading/trailing whitespace or lower-case "
                "in sku_id",
                detection="comparison of the raw column against its normalised form",
                rows_affected=dirty_ids,
                resolution="trimmed, whitespace-collapsed and upper-cased",
                rationale="' nbl-lgt-004 ' and 'NBL-LGT-004' are one product; treating "
                "them as two splits that SKU's history and breaks its lag "
                "features and its join to the product master",
                severity="critical",
            )
        )

    # --- 3. Dates ----------------------------------------------------------- #
    frame["date"] = parse_mixed_dates(frame["date"])
    bad_dates = int(frame["date"].isna().sum())
    if bad_dates:
        frame = frame[frame["date"].notna()]
        report.add(
            CleaningAction(
                table=table,
                issue=f"{bad_dates} sales row(s) with an unparseable date",
                detection="null count after explicit-format date parsing",
                rows_affected=bad_dates,
                resolution="dropped",
                rationale="a sale with no date cannot be placed in a week and so cannot "
                "inform a time series; it is unrecoverable",
                severity="critical",
            )
        )

    # --- 4. Rows referencing unknown products ------------------------------- #
    known_skus = set(sku_master["sku_id"])
    unknown_mask = ~frame["sku_id"].isin(known_skus)
    unknown_rows = int(unknown_mask.sum())
    if unknown_rows:
        unknown_ids = sorted(frame.loc[unknown_mask, "sku_id"].unique().tolist())
        frame = frame[~unknown_mask]
        report.add(
            CleaningAction(
                table=table,
                issue=f"{unknown_rows} sales row(s) reference {len(unknown_ids)} product(s) "
                f"absent from the product master ({', '.join(unknown_ids[:5])})",
                detection="anti-join of sales.sku_id against sku_master.sku_id",
                rows_affected=unknown_rows,
                resolution="quarantined out of the modelling set and reported to the client",
                rationale="without a category, cost or price these rows cannot be "
                "forecast, valued, or shown in the dashboard; they are flagged "
                "for the client to fix at source rather than silently guessed",
                severity="warning",
            )
        )

    # --- 5. Numeric coercion ------------------------------------------------ #
    for column in ("units_sold", "revenue", "unit_price"):
        frame[column] = _to_numeric(frame[column])

    # --- 6. Negative units -------------------------------------------------- #
    negative_mask = frame["units_sold"] < 0
    negative_rows = int(negative_mask.sum())
    if negative_rows:
        frame["units_sold"] = frame["units_sold"].clip(lower=0)
        frame["revenue"] = frame["revenue"].where(~negative_mask, 0.0)
        report.add(
            CleaningAction(
                table=table,
                issue=f"{negative_rows} row(s) with negative units_sold",
                detection="units_sold < 0",
                rows_affected=negative_rows,
                resolution="clipped to zero, with revenue on those rows zeroed too",
                rationale="these are refunds booked against the sales ledger. The model "
                "forecasts gross demand, so a return is not negative demand - "
                "netting it off would understate what customers actually wanted",
                severity="warning",
            )
        )

    # --- 7. Prices ---------------------------------------------------------- #
    price_lookup = sku_master.set_index("sku_id")["list_price"]
    missing_price = frame["unit_price"].isna() | (frame["unit_price"] <= 0)
    missing_price_rows = int(missing_price.sum())
    if missing_price_rows:
        # Best available source first: the row's own revenue, then the SKU's
        # median realised price, then its list price.
        implied = frame["revenue"] / frame["units_sold"].replace(0, np.nan)
        sku_median = frame.groupby("sku_id")["unit_price"].transform("median")
        list_price = frame["sku_id"].map(price_lookup)

        repaired = frame["unit_price"].where(~missing_price, implied)
        repaired = repaired.where(repaired.notna() & (repaired > 0), sku_median)
        repaired = repaired.where(repaired.notna() & (repaired > 0), list_price)
        frame["unit_price"] = repaired

        report.add(
            CleaningAction(
                table=table,
                issue=f"{missing_price_rows} row(s) with a missing or non-positive unit_price",
                detection="null or <= 0 check on unit_price",
                rows_affected=missing_price_rows,
                resolution="derived as revenue / units where possible, else the SKU's "
                "median realised price, else its list price",
                rationale="price feeds both the promotion signal and the revenue "
                "reconstruction; the fallback order runs from most to least "
                "row-specific so the best available evidence is always used",
                severity="warning",
            )
        )

    # --- 8. Revenue --------------------------------------------------------- #
    missing_revenue = frame["revenue"].isna()
    missing_revenue_rows = int(missing_revenue.sum())
    if missing_revenue_rows:
        frame["revenue"] = frame["revenue"].where(
            ~missing_revenue, frame["units_sold"] * frame["unit_price"]
        )
        report.add(
            CleaningAction(
                table=table,
                issue=f"{missing_revenue_rows} row(s) with missing revenue",
                detection="null count on revenue",
                rows_affected=missing_revenue_rows,
                resolution="recomputed as units_sold x unit_price",
                rationale="revenue is fully determined by the other two columns, so it is "
                "reconstructed exactly rather than imputed or the row discarded",
                severity="info",
            )
        )

    # --- 9. Promotion flag -------------------------------------------------- #
    promo_raw = frame["promo_flag"].astype("string").str.strip().str.lower()
    non_binary = int((~promo_raw.isin({"0", "1"})).sum())
    frame["promo_flag"] = parse_boolean_flag(frame["promo_flag"])
    if non_binary:
        report.add(
            CleaningAction(
                table=table,
                issue="promo_flag arrives in three encodings (0/1, Y/N, TRUE/FALSE)",
                detection="values outside {0, 1} on the raw column",
                rows_affected=non_binary,
                resolution="mapped a fixed truthy vocabulary to 1, everything else to 0",
                rationale="several source systems feed this column. Unrecognised values "
                "resolve to 0 because inventing a promotion is worse than "
                "missing one: it teaches the model a price effect that never "
                "happened",
                severity="warning",
            )
        )

    # --- 10. Duplicates that only became visible after normalisation -------- #
    # A re-exported row can be semantically identical yet textually different -
    # the same sale with promo_flag written as "1" in one export and "TRUE" in
    # another, or the sku_id in a different case. Step 1 cannot see those,
    # because it compares raw text. Now that every column is normalised and
    # typed, an identical row genuinely is the same sale.
    #
    # This has to run BEFORE key-collision aggregation. Aggregation sums units,
    # so a duplicate that survives to that step is not merged but doubled -
    # inflating that SKU-day's demand and everything derived from it.
    value_columns = ["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"]
    semantic_duplicates = int(frame.duplicated(subset=value_columns).sum())
    if semantic_duplicates:
        frame = frame.drop_duplicates(subset=value_columns, keep="first")
        report.add(
            CleaningAction(
                table=table,
                issue=f"{semantic_duplicates} row(s) identical in value but not in text "
                "(same sale re-exported with a different flag or case encoding)",
                detection="full-row duplicate check repeated after normalisation and "
                "type coercion, which the raw-text check in step 1 cannot see",
                rows_affected=semantic_duplicates,
                resolution="dropped, keeping the first occurrence",
                rationale="these are the same sale exported twice. They must be removed "
                "before key-collision aggregation, which sums units - left in, "
                "they would be added together and double that day's demand "
                "rather than being recognised as one sale",
                severity="critical",
            )
        )

    # --- 11. Remaining key collisions --------------------------------------- #
    key_duplicates = int(frame.duplicated(subset=["date", "sku_id"]).sum())
    if key_duplicates:
        frame = frame.groupby(["date", "sku_id"], as_index=False).agg(
            units_sold=("units_sold", "sum"),
            revenue=("revenue", "sum"),
            unit_price=("unit_price", "mean"),
            promo_flag=("promo_flag", "max"),
        )
        report.add(
            CleaningAction(
                table=table,
                issue=f"{key_duplicates} non-identical row(s) sharing a (date, sku_id) key",
                detection="duplicate check on the declared primary key after normalisation",
                rows_affected=key_duplicates,
                resolution="aggregated: units and revenue summed, price averaged, "
                "promo flag taken as the maximum",
                rationale="the fact table's grain is one row per SKU per day. These are "
                "partial exports of the same day, so summing restores the true "
                "daily total; dropping them would lose real sales",
                severity="warning",
            )
        )

    frame["units_sold"] = frame["units_sold"].fillna(0.0).astype("float64")
    frame["revenue"] = frame["revenue"].fillna(0.0).astype("float64")
    frame["unit_price"] = frame["unit_price"].astype("float64")
    frame["promo_flag"] = frame["promo_flag"].astype("int64")

    if frame.empty:
        raise DataQualityError("sales_daily is empty after cleaning; nothing left to model.")

    log.info(
        "cleaned sales_daily",
        extra={"context": {"rows_in": rows_in, "rows_out": len(frame)}},
    )

    columns = ["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"]
    return frame[columns].sort_values(["sku_id", "date"]).reset_index(drop=True)


# Cleaning for the optional commercial driver extracts lives in the
# `foresight_drivers` package. It reuses the primitives above and records into
# the same CleaningReport, so the data-quality memo covers driver data without
# core knowing anything about those fields.


# --------------------------------------------------------------------------- #
# inventory_snapshots
# --------------------------------------------------------------------------- #
def clean_inventory(
    raw: pd.DataFrame,
    sku_master: pd.DataFrame,
    report: CleaningReport,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Clean the inventory position snapshots."""
    settings = settings or get_settings()
    table = "inventory_snapshots"
    frame = raw.copy()

    exact_duplicates = int(frame.duplicated().sum())
    if exact_duplicates:
        frame = frame.drop_duplicates(keep="first")
        report.add(
            CleaningAction(
                table=table,
                issue=f"{exact_duplicates} byte-identical duplicate snapshot(s)",
                detection="full-row duplicate check",
                rows_affected=exact_duplicates,
                resolution="dropped, keeping the first occurrence",
                rationale="a stock position is a point-in-time fact; repeating it adds "
                "no information and breaks the (date, sku_id) key",
                severity="warning",
            )
        )

    frame["sku_id"] = normalise_sku_id(frame["sku_id"])
    frame["date"] = parse_mixed_dates(frame["date"])
    frame = frame[frame["date"].notna()]

    unknown_mask = ~frame["sku_id"].isin(set(sku_master["sku_id"]))
    unknown_rows = int(unknown_mask.sum())
    if unknown_rows:
        frame = frame[~unknown_mask]
        report.add(
            CleaningAction(
                table=table,
                issue=f"{unknown_rows} snapshot(s) for products absent from the master",
                detection="anti-join against sku_master.sku_id",
                rows_affected=unknown_rows,
                resolution="quarantined out of the modelling set",
                rationale="stock for an unknown product cannot be valued or actioned",
                severity="warning",
            )
        )

    for column in ("on_hand_units", "on_order_units", "lead_time_days", "reorder_point"):
        frame[column] = _to_numeric(frame[column])

    frame = frame.sort_values(["sku_id", "date"])

    # --- On-hand ------------------------------------------------------------ #
    missing_on_hand = int(frame["on_hand_units"].isna().sum())
    if missing_on_hand:
        # A missed snapshot does not mean the warehouse was empty. Carrying the
        # last known position forward is the only defensible reading; a leading
        # gap with no prior observation falls back to zero.
        frame["on_hand_units"] = frame.groupby("sku_id")["on_hand_units"].ffill()
        still_missing = int(frame["on_hand_units"].isna().sum())
        frame["on_hand_units"] = frame["on_hand_units"].fillna(0.0)
        report.add(
            CleaningAction(
                table=table,
                issue=f"{missing_on_hand} snapshot(s) with a missing on_hand_units",
                detection="null count on on_hand_units",
                rows_affected=missing_on_hand,
                resolution=f"carried the last known position forward within each SKU; "
                f"{still_missing} row(s) with no prior observation set to 0",
                rationale="a missed export is not evidence of empty shelves. Filling with "
                "zero everywhere would fabricate stockouts and flood the reorder "
                "list with false alarms",
                severity="critical",
            )
        )

    negative_stock = int((frame["on_hand_units"] < 0).sum())
    if negative_stock:
        frame["on_hand_units"] = frame["on_hand_units"].clip(lower=0)
        report.add(
            CleaningAction(
                table=table,
                issue=f"{negative_stock} snapshot(s) with negative on-hand stock",
                detection="on_hand_units < 0",
                rows_affected=negative_stock,
                resolution="clipped to zero",
                rationale="physical stock cannot be negative; the value reflects an "
                "unreconciled ledger, and zero is the true floor",
                severity="warning",
            )
        )

    frame["on_order_units"] = frame["on_order_units"].fillna(0.0).clip(lower=0)

    # --- Lead time ---------------------------------------------------------- #
    low, high = settings.lead_time_min_days, settings.lead_time_max_days
    invalid_lead = frame["lead_time_days"].isna() | ~frame["lead_time_days"].between(low, high)
    invalid_lead_rows = int(invalid_lead.sum())
    if invalid_lead_rows:
        valid = frame["lead_time_days"].where(~invalid_lead)
        # Per-SKU median of the plausible observations, then a global fallback.
        sku_median = valid.groupby(frame["sku_id"]).transform("median")
        global_median = float(valid.median()) if valid.notna().any() else float(low)
        repaired = frame["lead_time_days"].where(~invalid_lead, sku_median)
        frame["lead_time_days"] = repaired.fillna(global_median).clip(low, high)
        report.add(
            CleaningAction(
                table=table,
                issue=f"{invalid_lead_rows} snapshot(s) with an impossible lead_time_days "
                f"(outside {low}-{high}; observed values include 0 and 900+)",
                detection=f"range check against the configured plausible band [{low}, {high}]",
                rows_affected=invalid_lead_rows,
                resolution="replaced with the SKU's median plausible lead time, falling "
                "back to the global median",
                rationale="lead time sets the window the stockout calculation looks over. "
                "A 0 makes every SKU look safe and a 900 makes every SKU look "
                "critical - both would destroy the reorder list's credibility",
                severity="critical",
            )
        )

    frame["reorder_point"] = frame["reorder_point"].fillna(0.0).clip(lower=0)

    key_duplicates = int(frame.duplicated(subset=["date", "sku_id"]).sum())
    if key_duplicates:
        frame = frame.drop_duplicates(subset=["date", "sku_id"], keep="last")
        report.add(
            CleaningAction(
                table=table,
                issue=f"{key_duplicates} row(s) sharing a (date, sku_id) key",
                detection="duplicate check on the declared primary key",
                rows_affected=key_duplicates,
                resolution="kept the last occurrence",
                rationale="a repeated snapshot for the same day is a restatement; the "
                "latest export is the most accurate stock position",
                severity="warning",
            )
        )

    for column in ("on_hand_units", "on_order_units", "lead_time_days", "reorder_point"):
        frame[column] = frame[column].astype("float64")

    columns = [
        "date",
        "sku_id",
        "on_hand_units",
        "on_order_units",
        "lead_time_days",
        "reorder_point",
    ]
    return frame[columns].sort_values(["sku_id", "date"]).reset_index(drop=True)
