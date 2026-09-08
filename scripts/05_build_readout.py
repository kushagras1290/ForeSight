"""Build the executive readout (deliverable D7).

Renders an eight-page PDF for the Head of Operations and the Finance lead from
the scored artifacts. Nothing here is written by hand, so the deck can never
disagree with the numbers the models actually produced.

Usage::

    python scripts/05_build_readout.py
"""

from __future__ import annotations

import argparse
import sys

from foresight.config import get_settings
from foresight.exceptions import ForesightError
from foresight.logging_setup import get_logger
from foresight.readout import build_readout

log = get_logger("scripts.build_readout")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    settings = get_settings()

    required = [
        settings.artifacts_dir / "metrics.json",
        settings.artifacts_dir / "impact.json",
        settings.artifacts_dir / "risk_table.parquet",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        log.error(
            "cannot build readout; artifacts missing",
            extra={
                "context": {
                    "missing": missing,
                    "hint": "run scripts/03_train_backtest.py and scripts/04_score_risk.py",
                }
            },
        )
        return 1

    try:
        outputs = build_readout(settings)
    except (ForesightError, FileNotFoundError, OSError, KeyError) as exc:
        log.error("readout generation failed", extra={"context": {"error": str(exc)}})
        return 1

    log.info("readout ready", extra={"context": outputs})
    return 0


if __name__ == "__main__":
    sys.exit(main())
