"""Fit Cross-SKU Substitution (custom model 6) and persist it as an artifact.

Standalone for the same reason `11_fit_promotion_response.py` is: this is a
measured decomposition effect, not a forecasting model scored inside the
rolling-origin backtest, and it is cheap to fit on the full processed weekly
panel without re-running anything else.

Usage::

    python scripts/13_fit_cross_sku_substitution.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from foresight.config import get_settings
from foresight.exceptions import ForesightError
from foresight.logging_setup import get_logger
from foresight.substitution import fit_substitution_effect

log = get_logger("scripts.fit_cross_sku_substitution")

_POOLED_KEY = "__POOLED__"


def main(argv: list[str] | None = None) -> int:
    del argv  # No arguments today; kept for parity with the other scripts.

    settings = get_settings()
    panel_path = settings.processed_dir / "weekly_panel.parquet"
    if not panel_path.exists():
        log.error(
            "processed data missing",
            extra={"context": {"path": str(panel_path), "hint": "run scripts/01_run_pipeline.py"}},
        )
        return 1

    panel = pd.read_parquet(panel_path)
    try:
        model = fit_substitution_effect(panel)
    except ForesightError as exc:
        log.error("fit failed", extra={"context": {"error": str(exc)}})
        return 1

    subcategories = sorted(s for s in model.dip_by_subcategory if s != _POOLED_KEY)
    payload: dict[str, Any] = {
        "diagnostics": model.diagnostics,
        "pooled_dip_factor": model.dip_by_subcategory[_POOLED_KEY],
        "subcategories": [
            {
                "subcategory": subcategory,
                "dip_factor": model.dip_by_subcategory[subcategory],
                "demand_pulled_away_pct": round(
                    (1.0 - model.dip_by_subcategory[subcategory]) * 100.0, 2
                ),
            }
            for subcategory in subcategories
        ],
    }

    out_path: Path = settings.artifacts_dir / "substitution_effect.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        "substitution effect artifact written",
        extra={
            "context": {
                "path": str(out_path),
                "subcategories_with_own_estimate": len(subcategories),
                "pooled_demand_pulled_away_pct": payload["diagnostics"][
                    "pooled_demand_pulled_away_pct"
                ],
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
