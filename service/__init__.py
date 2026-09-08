"""FORESIGHT scoring service (deliverable D6).

A read-only FastAPI application serving the demand forecast and inventory risk
produced by the pipeline. It also backs the planning dashboard, so the dashboard
and the API can never disagree about what the models say.
"""

from __future__ import annotations

__all__: list[str] = []
