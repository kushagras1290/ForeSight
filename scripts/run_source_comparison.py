"""Run the full pipeline once per raw-data source and keep both results.

Deliberately thin - no modelling or cleaning logic lives here, only
orchestration, matching this repo's convention that scripts call the library
and nothing else (CODEBASE.md Sec.1). Every step it runs already exists as its
own numbered script; this one just runs each of them twice, once per source in
``data/sources/*``, and snapshots what they wrote before the next source
overwrites it.

Both sources are trained with ``--no-drivers``: only ``synthetic_generated``
has the optional commercial-driver extracts, so comparing WAPE with drivers on
one side and without them on the other would measure the driver extension, not
the data source. The four Appendix A tables are the only fair common ground.

Usage::

    python scripts/run_source_comparison.py
    python scripts/run_source_comparison.py --sources provided
"""

from __future__ import annotations

import argparse
import shutil
import subprocess  # noqa: S404 - fixed argv per call, no shell, no user input
import sys
import time
from pathlib import Path
from typing import Final

from foresight.config import PROJECT_ROOT
from foresight.exceptions import ForesightError
from foresight.logging_setup import get_logger

log = get_logger("scripts.run_source_comparison")

SOURCES_DIR: Final[Path] = PROJECT_ROOT / "data" / "sources"
RAW_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
RUNS_DIR: Final[Path] = PROJECT_ROOT / "runs"

#: The four Appendix A filenames every source must supply.
RAW_TABLE_FILES: Final[tuple[str, ...]] = (
    "sales_daily.csv",
    "sku_master.csv",
    "calendar.csv",
    "inventory_snapshots.csv",
)

#: Numbered scripts run in order for a full training + readout pass.
_PIPELINE_SCRIPTS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("01_run_pipeline.py", ()),
    ("02_run_eda.py", ()),
    ("03_train_backtest.py", ("--no-drivers",)),
    ("04_score_risk.py", ()),
    ("05_build_readout.py", ()),
)

DEFAULT_SOURCES: Final[tuple[str, ...]] = ("synthetic_generated", "provided")


class SourceComparisonError(ForesightError):
    """A step of the per-source pipeline run failed."""


def _swap_in_raw_tables(source: str) -> None:
    source_dir = SOURCES_DIR / source
    for filename in RAW_TABLE_FILES:
        candidate = source_dir / filename
        if not candidate.exists():
            raise SourceComparisonError(
                f"source {source!r} is missing {filename!r} at {candidate}. "
                "Run scripts/prepare_provided_extract.py first if this is 'provided'."
            )
        shutil.copyfile(candidate, RAW_DIR / filename)
    log.info("raw tables swapped in", extra={"context": {"source": source}})


def _run_step(script_name: str, extra_args: tuple[str, ...]) -> None:
    script_path = PROJECT_ROOT / "scripts" / script_name
    argv = [sys.executable, str(script_path), *extra_args]
    started = time.monotonic()
    result = subprocess.run(argv, cwd=PROJECT_ROOT, check=False)
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise SourceComparisonError(
            f"{script_name} exited {result.returncode} after {elapsed:.0f}s"
        )
    log.info(
        "step finished",
        extra={"context": {"script": script_name, "seconds": round(elapsed, 1)}},
    )


def _snapshot_run(source: str) -> None:
    destination = RUNS_DIR / source
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    shutil.copytree(PROJECT_ROOT / "data" / "processed", destination / "processed")
    shutil.copytree(PROJECT_ROOT / "artifacts", destination / "artifacts")

    reports_destination = destination / "reports"
    reports_destination.mkdir()
    for name in ("eda_memo.md", "executive_readout.pdf"):
        source_file = PROJECT_ROOT / "reports" / name
        if source_file.exists():
            shutil.copyfile(source_file, reports_destination / name)
    figures_source = PROJECT_ROOT / "reports" / "figures"
    if figures_source.exists():
        shutil.copytree(figures_source, reports_destination / "figures")

    log.info(
        "run snapshotted", extra={"context": {"source": source, "destination": str(destination)}}
    )


def run_source(source: str) -> None:
    """Swap in one source's raw tables and run the full pipeline against it."""
    log.info("starting source run", extra={"context": {"source": source}})
    _swap_in_raw_tables(source)
    for script_name, extra_args in _PIPELINE_SCRIPTS:
        _run_step(script_name, extra_args)
    _snapshot_run(source)
    log.info("source run complete", extra={"context": {"source": source}})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        choices=list(DEFAULT_SOURCES),
        help="Which sources to run, in order (default: both).",
    )
    arguments = parser.parse_args(argv)

    for source in arguments.sources:
        try:
            run_source(source)
        except ForesightError as exc:
            log.error("source run failed", extra={"context": {"source": source, "error": str(exc)}})
            return 1

    log.info("all source runs complete", extra={"context": {"sources": arguments.sources}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
