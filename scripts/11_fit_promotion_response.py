"""Fit the Promotional Response Decomposition and persist it as a dashboard artifact.

WHY THIS SCRIPT EXISTS
----------------------
``src/foresight/promotions.py`` (custom model 5, "Promotional Response
Decomposition") is fully implemented and unit-tested, but nothing in the
pipeline ever calls it and nothing persists its output - it exists only inside
``tests/test_promotions.py``. This script is the missing last step: fit it once
against the full processed weekly panel and write a small artifact the service
can serve, so the dashboard's Promotion page shows the model's actual fitted
uplift/dip curve rather than static prose.

Deliberately a standalone script rather than a change to
``03_train_backtest.py``: the promotional response is a *reporting*
decomposition (baseline vs. uplift vs. dip), not a forecasting model scored on
WAPE, so it does not belong inside the rolling-origin backtest loop. Fitting it
on the full available history (rather than one fold's training window) is
correct here specifically because nothing downstream treats this artifact as a
forecast - it is read-only business reporting, not a number scored against a
held-out actual.

Usage::

    python scripts/11_fit_promotion_response.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from foresight.config import get_settings
from foresight.exceptions import ForesightError
from foresight.logging_setup import get_logger
from foresight.promotions import PromotionResponse, fit_promotion_response

log = get_logger("scripts.fit_promotion_response")

#: Mirrors ``foresight.promotions._POOLED`` - not exported (it is an internal
#: fitting detail), but the key itself is part of the fitted model's stable
#: dict shape, so reading it back here is reading data, not reaching into
#: implementation.
_POOLED_KEY: Final[str] = "__POOLED__"

#: Representative discount depths shown per category, chosen to bracket what a
#: merchandiser actually plans between (a light and a deep promotion) without
#: implying the fitted curve is only valid at these three points.
_SAMPLE_DISCOUNTS: Final[tuple[float, ...]] = (0.10, 0.20, 0.30)

#: A promotion's shadow starts the week immediately after it ends.
_DIP_REFERENCE_WEEKS_SINCE_PROMO: Final[float] = 1.0


def _category_summary(model: PromotionResponse, category: str) -> dict[str, Any]:
    categories = np.array([category] * len(_SAMPLE_DISCOUNTS))
    discounts = np.array(_SAMPLE_DISCOUNTS, dtype="float64")
    uplifts = model.uplift(categories, discounts)
    dip = model.dip_factor(
        np.array([category]), np.array([_DIP_REFERENCE_WEEKS_SINCE_PROMO])
    )[0]
    return {
        "category": category,
        "has_own_curve": category in model.slope,
        "uplift_by_discount": {
            f"{int(depth * 100)}pct": round(float(uplift), 4)
            for depth, uplift in zip(_SAMPLE_DISCOUNTS, uplifts, strict=True)
        },
        "post_promo_dip_factor": round(float(dip), 4),
    }


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
        model = fit_promotion_response(panel)
    except ForesightError as exc:
        log.error("fit failed", extra={"context": {"error": str(exc)}})
        return 1

    categories = sorted(c for c in model.slope if c != _POOLED_KEY)
    payload: dict[str, Any] = {
        "baseline_window": model.baseline_window,
        "dip_weeks": model.dip_weeks,
        "diagnostics": model.diagnostics,
        "pooled": _category_summary(model, _POOLED_KEY),
        "categories": [_category_summary(model, category) for category in categories],
    }

    out_path: Path = settings.artifacts_dir / "promotion_response.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        "promotion response artifact written",
        extra={
            "context": {
                "path": str(out_path),
                "categories_with_own_curve": len(categories),
                "pooled_uplift_at_20pct": payload["pooled"]["uplift_by_discount"]["20pct"],
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
