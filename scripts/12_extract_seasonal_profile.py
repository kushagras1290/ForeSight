"""Extract the Adaptive Ensemble's seasonal index and persist it as its own artifact.

WHY THIS SCRIPT EXISTS
----------------------
The seasonal-profile component blended into every forecast
(``foresight.custom_model.SeasonalProfile``, one of the three components in the
Adaptive Demand Ensemble) is fit and used purely internally: it is trained
inside ``scripts/03_train_backtest.py`` and saved as part of
``artifacts/adaptive_ensemble.joblib``, but its actual week-of-year index
values never reach their own artifact or API field - only the blended final
forecast numbers do. This script reads the already-trained ensemble back off
disk and writes just its seasonal index out as a small, dashboard-friendly
JSON artifact, so the Seasonality page can show the real per-category curve
the model learned instead of a static EDA figure.

Deliberately does not retrain anything and does not touch
``03_train_backtest.py`` - the index already exists inside the persisted
ensemble the moment training finishes; this is a read-and-reshape step, safe
to re-run any time after training without redoing any of it.

Usage::

    python scripts/12_extract_seasonal_profile.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from foresight.config import get_settings
from foresight.custom_model import load_ensemble
from foresight.exceptions import ForesightError
from foresight.logging_setup import get_logger

log = get_logger("scripts.extract_seasonal_profile")


def main(argv: list[str] | None = None) -> int:
    del argv  # No arguments today; kept for parity with the other scripts.

    settings = get_settings()
    try:
        ensemble = load_ensemble(settings=settings)
    except ForesightError as exc:
        log.error(
            "ensemble not available",
            extra={"context": {"error": str(exc), "hint": "run scripts/03_train_backtest.py"}},
        )
        return 1

    profile = ensemble.profile
    by_category: dict[str, list[dict[str, Any]]] = {}
    for (category, iso_week), value in profile.index_by_category_week.items():
        by_category.setdefault(str(category), []).append(
            {"iso_week": int(iso_week), "index": round(float(value), 4)}
        )
    for points in by_category.values():
        points.sort(key=lambda point: point["iso_week"])

    global_index = [
        {"iso_week": int(iso_week), "index": round(float(value), 4)}
        for iso_week, value in sorted(profile.global_index_by_week.items())
    ]

    payload: dict[str, Any] = {
        "trained_at": ensemble.trained_at,
        "global_index": global_index,
        "by_category": by_category,
    }

    out_path: Path = settings.artifacts_dir / "seasonal_profile.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        "seasonal profile artifact written",
        extra={
            "context": {
                "path": str(out_path),
                "categories": len(by_category),
                "global_weeks": len(global_index),
            }
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
