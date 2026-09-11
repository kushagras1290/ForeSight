"""Adapt the client-provided extract onto the Appendix A raw schema.

The brief's four Appendix A tables use lowercase snake_case column names
(``sku_id``, ``units_sold``, ...). The extract as handed over uses different
column names and casing for the same fields (``SKU``, ``Units_Sold``, ...).
This script performs the **rename only** - no value-level cleaning. Every
value-level decision (missing data, duplicates, category recovery, ...)
belongs to ``clean.py`` and must stay auditable there; renaming a column is
not a cleaning decision, so it happens here instead of being buried inside the
cleaner.

One exception, documented rather than hidden: the provided extract's product
``Category`` values use different labels for what are, for four of five
categories, clearly the same real-world category as NorthBay's own controlled
vocabulary (``Storage`` vs ``Storage & Organisation``). Those four are aligned
here, before the label ever reaches ``clean_sku_master``'s exact-match lookup,
because they are a naming-convention difference, not an unknown category. The
fifth, ``Furniture``, has no counterpart anywhere in NorthBay's vocabulary
(``Bedding & Bath``, ``Lighting``, ``Kitchen & Dining``, ``Decor``,
``Storage & Organisation``, ``Small Appliances``) and is deliberately left
unmapped: it is a genuine taxonomy gap, not a labelling quirk, and forcing it
into an unrelated bucket would misreport the source's actual assortment. It
will resolve to ``Unclassified`` downstream and should be reported as a
finding, not silently absorbed.

Usage::

    python scripts/prepare_provided_extract.py
    python scripts/prepare_provided_extract.py --source-dir D:\\zidio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from foresight.config import PROJECT_ROOT
from foresight.exceptions import DataError, ForesightError, MissingDataFileError
from foresight.logging_setup import get_logger

log = get_logger("scripts.prepare_provided_extract")

#: Destination directory this script writes to. A staging area, not data/raw/
#: itself, so preparing the extract never clobbers whichever source is
#: currently live for the pipeline.
PROVIDED_DIR: Path = PROJECT_ROOT / "data" / "sources" / "provided"

#: Column renames, table by table. Appendix A schema (schemas.py RAW_*_SCHEMA)
#: on the right; the provided extract's own column name on the left.
_SALES_RENAME: dict[str, str] = {
    "Date": "date",
    "SKU": "sku_id",
    "Units_Sold": "units_sold",
    "Revenue": "revenue",
    "Price": "unit_price",
    "Promotion": "promo_flag",
}
_SKU_MASTER_RENAME: dict[str, str] = {
    "SKU": "sku_id",
    "Category": "category",
    "Subcategory": "subcategory",
    "Launch_Date": "launch_date",
    "Cost_Price": "unit_cost",
    "Selling_Price": "list_price",
}
_CALENDAR_RENAME: dict[str, str] = {
    "promotion_event": "promo_event",
}
_INVENTORY_RENAME: dict[str, str] = {
    "Snapshot_Date": "date",
    "SKU": "sku_id",
    "Current_Stock": "on_hand_units",
    "On_Order": "on_order_units",
    "Lead_Time_Days": "lead_time_days",
    "Reorder_Point": "reorder_point",
}

#: Naming-convention alignment only - see module docstring for why "Furniture"
#: is deliberately excluded.
_CATEGORY_ALIGNMENT: dict[str, str] = {
    "Home Decor": "Decor",
    "Storage": "Storage & Organisation",
    "Kitchen": "Kitchen & Dining",
}


def _read_source_csv(source_dir: Path, filename: str) -> pd.DataFrame:
    path = source_dir / filename
    if not path.exists():
        raise MissingDataFileError(path, f"Expected the provided extract's {filename!r} at {path}.")
    frame = pd.read_csv(path, dtype=str, keep_default_na=True)
    if frame.empty:
        raise DataError(f"{filename!r} at {path} contains no rows.")
    return frame


def _write_renamed(frame: pd.DataFrame, rename: dict[str, str], destination: Path) -> None:
    missing = [source for source in rename if source not in frame.columns]
    if missing:
        raise DataError(
            f"Expected column(s) {missing} not found in {destination.name}; "
            f"got columns {list(frame.columns)}."
        )
    frame.rename(columns=rename).to_csv(destination, index=False)
    log.info(
        "wrote provided extract table",
        extra={"context": {"table": destination.name, "rows": len(frame)}},
    )


def prepare_sales_daily(source_dir: Path, destination_dir: Path) -> None:
    frame = _read_source_csv(source_dir, "sales_daily.csv")
    _write_renamed(frame, _SALES_RENAME, destination_dir / "sales_daily.csv")


def prepare_sku_master(source_dir: Path, destination_dir: Path) -> None:
    frame = _read_source_csv(source_dir, "sku_master.csv")
    if "Category" in frame.columns:
        aligned_count = frame["Category"].isin(_CATEGORY_ALIGNMENT).sum()
        frame["Category"] = frame["Category"].replace(_CATEGORY_ALIGNMENT)
        log.info(
            "aligned category labels to NorthBay's controlled vocabulary",
            extra={"context": {"rows_aligned": int(aligned_count)}},
        )
    _write_renamed(frame, _SKU_MASTER_RENAME, destination_dir / "sku_master.csv")


def prepare_calendar(source_dir: Path, destination_dir: Path) -> None:
    frame = _read_source_csv(source_dir, "calendar.csv")
    _write_renamed(frame, _CALENDAR_RENAME, destination_dir / "calendar.csv")


def prepare_inventory_snapshots(source_dir: Path, destination_dir: Path) -> None:
    frame = _read_source_csv(source_dir, "inventory_snapshots.csv")
    _write_renamed(frame, _INVENTORY_RENAME, destination_dir / "inventory_snapshots.csv")


def prepare_provided_extract(source_dir: Path, destination_dir: Path) -> None:
    """Rename all four Appendix A tables from the provided extract's raw form."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    prepare_sales_daily(source_dir, destination_dir)
    prepare_sku_master(source_dir, destination_dir)
    prepare_calendar(source_dir, destination_dir)
    prepare_inventory_snapshots(source_dir, destination_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT.parent,
        help="Directory holding the provided extract's raw CSVs (default: repo's parent dir).",
    )
    parser.add_argument(
        "--destination-dir",
        type=Path,
        default=PROVIDED_DIR,
        help="Where to write the renamed extract (default: data/sources/provided/).",
    )
    arguments = parser.parse_args(argv)

    try:
        prepare_provided_extract(arguments.source_dir, arguments.destination_dir)
    except ForesightError as exc:
        log.error("failed to prepare provided extract", extra={"context": {"error": str(exc)}})
        return 1

    log.info(
        "provided extract ready",
        extra={"context": {"destination": str(arguments.destination_dir)}},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
