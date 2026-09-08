"""Materialise the four client extracts into ``data/raw/``.

Extracts are not tracked in version control (brief section 16.3). This script
reconstructs them deterministically from the configured seed.

If you have been supplied with the extracts directly, skip this step entirely:
drop the four Appendix A CSVs into ``data/raw/`` and run the pipeline. Nothing
downstream depends on how those files got there.

Usage::

    python scripts/00_generate_data.py
"""

from __future__ import annotations

import argparse
import sys

from foresight.config import get_settings
from foresight.datagen import write_extracts
from foresight.exceptions import ForesightError
from foresight.logging_setup import get_logger

log = get_logger("scripts.generate_data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing extracts in data/raw/ instead of stopping.",
    )
    arguments = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_directories()

    existing = sorted(settings.raw_dir.glob("*.csv"))
    if existing and not arguments.force:
        log.warning(
            "extracts already present; nothing written",
            extra={
                "context": {
                    "files": [path.name for path in existing],
                    "hint": "pass --force to overwrite",
                }
            },
        )
        return 0

    try:
        summary = write_extracts(settings)
    except ForesightError as exc:
        log.error("data generation failed", extra={"context": {"error": str(exc)}})
        return 1

    log.info(
        "extracts ready",
        extra={"context": {"path": str(settings.raw_dir), **summary.row_counts}},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
