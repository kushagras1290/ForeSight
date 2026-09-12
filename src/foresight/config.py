"""Validated, environment-driven configuration for Project FORESIGHT.

Everything tunable lives here. Values come from environment variables (or a
local ``.env``), are validated by pydantic at import time, and are frozen
afterwards, so a typo fails loudly at startup instead of silently producing a
wrong forecast.

Usage::

    from foresight.config import get_settings

    settings = get_settings()
    horizon = settings.horizon_weeks

``get_settings`` is cached, so configuration is parsed exactly once per process.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from foresight.exceptions import ConfigurationError

__all__ = ["Settings", "get_settings", "PROJECT_ROOT"]

# src/foresight/config.py -> src/foresight -> src -> <project root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# The generating process needs a full year of history before it can express
# annual seasonality, plus a year of holdout to be worth backtesting.
_MIN_HISTORY_DAYS = 730

# The literal defaults `google_client_id`/`google_client_secret` ship with.
# Compared against at runtime so the service can tell "not configured yet"
# apart from "configured with a real, working Google Cloud OAuth client".
_PLACEHOLDER_GOOGLE_CLIENT_ID = "PLACEHOLDER_GOOGLE_CLIENT_ID.apps.googleusercontent.com"
_PLACEHOLDER_GOOGLE_CLIENT_SECRET = "PLACEHOLDER_GOOGLE_CLIENT_SECRET"


class Settings(BaseSettings):
    """Runtime configuration, validated once at process start.

    Field names map to ``FORESIGHT_``-prefixed environment variables, e.g.
    ``horizon_weeks`` reads ``FORESIGHT_HORIZON_WEEKS``.
    """

    model_config = SettingsConfigDict(
        env_prefix="FORESIGHT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Reproducibility ---------------------------------------------------- #
    random_seed: int = Field(
        default=20260907,
        ge=0,
        le=2**32 - 1,
        description="Master seed. Fixed so every result in the repo is re-creatable.",
    )

    # --- Client extract shape ----------------------------------------------- #
    n_skus: int = Field(default=200, ge=10, le=5000, description="Initial catalogue size.")
    history_start: dt.date = Field(default=dt.date(2023, 1, 2))
    history_end: dt.date = Field(default=dt.date(2026, 8, 30))
    #: SKUs launched at each subsequent year boundary within the history window,
    #: on top of `n_skus`. 0 (default) reproduces the original single-cohort
    #: catalogue exactly - every existing generated dataset used this and stays
    #: reproducible. Set positive to model an assortment that grows every year,
    #: the way a real, ageing catalogue does.
    new_skus_per_year: int = Field(default=0, ge=0, le=2000)
    #: Fraction of the *whole* catalogue (initial + every cohort) given a
    #: genuinely volatile demand archetype - see datagen.py's module docstring
    #: for what that means and why it exists. 0.06 was chosen to reliably
    #: populate the dashboard's "watch - volatile" quadrant with a handful of
    #: SKUs without dominating the catalogue.
    volatile_sku_share: float = Field(default=0.06, ge=0.0, le=0.5)

    # --- Commercial drivers (optional extension) ---------------------------- #
    use_drivers: bool = Field(
        default=True,
        description=(
            "Use the optional foresight_drivers extension when its extracts are "
            "present. Set false to forecast from the four Appendix A tables alone, "
            "which is what the brief specifies and the honest baseline to compare "
            "any driver-derived accuracy gain against."
        ),
    )

    # --- Forecasting -------------------------------------------------------- #
    horizon_weeks: int = Field(
        default=8,
        ge=1,
        le=26,
        description="Forecast horizon in weeks. Brief section 4.2 specifies 6-8.",
    )
    backtest_folds: int = Field(default=6, ge=2, le=24)
    backtest_step_weeks: int = Field(default=4, ge=1, le=26)
    min_train_weeks: int = Field(
        default=60,
        ge=8,
        description="Weeks of history required before a SKU may enter training.",
    )
    interval_coverage: float = Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description="Prediction-interval coverage; 0.8 => 10th/90th quantiles.",
    )

    # --- Risk scoring ------------------------------------------------------- #
    service_level: float = Field(
        default=0.95,
        gt=0.5,
        lt=1.0,
        description="Target probability of not stocking out over the lead time.",
    )
    overstock_cover_weeks: float = Field(default=12.0, gt=0.0, le=104.0)
    risk_high_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    lead_time_min_days: int = Field(default=1, ge=1, le=365)
    lead_time_max_days: int = Field(default=120, ge=1, le=365)

    # --- Service ------------------------------------------------------------ #
    api_title: str = Field(default="FORESIGHT Scoring Service", min_length=1)
    api_max_batch_size: int = Field(default=200, ge=1, le=5000)
    cors_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        description="Comma-separated browser origins allowed to call the API.",
    )

    # --- Authentication ------------------------------------------------------ #
    # Google sign-in ships with a placeholder client id/secret so the button and
    # the callback route exist and can be reviewed end-to-end; they must be
    # replaced with a real Google Cloud OAuth client before the flow can
    # actually authenticate anyone. `google_oauth_configured` below is how the
    # service tells the two states apart.
    google_client_id: str = Field(default=_PLACEHOLDER_GOOGLE_CLIENT_ID)
    google_client_secret: str = Field(default=_PLACEHOLDER_GOOGLE_CLIENT_SECRET)
    google_redirect_uri: str = Field(default="http://localhost:8000/api/auth/google/callback")
    session_ttl_days: int = Field(default=7, ge=1, le=90)
    session_cookie_secure: bool = Field(
        default=False,
        description="Send the session cookie only over HTTPS. Set true in production.",
    )
    session_cookie_name: str = Field(
        default="fs_session",
        min_length=1,
        description=(
            "Name of the session cookie. Change this per deployment when two instances "
            "of this service share a browser's cookie jar (e.g. two ports on localhost) - "
            "cookies are not port-scoped, so two instances using the same name would "
            "silently overwrite each other's session."
        ),
    )

    # --- Multiple deployments of one codebase --------------------------------- #
    data_root: Path | None = Field(
        default=None,
        description=(
            "Where data/, artifacts/ and reports/ live. Defaults to the repo root "
            "(PROJECT_ROOT) when unset. Overriding it lets one codebase serve more than "
            "one independent instance - e.g. a demo instance trained on the brief's "
            "extract, and a client instance with its own data and no shared state - "
            "without checking out a second copy of the code."
        ),
    )

    # --- Logging ------------------------------------------------------------ #
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "console"

    # ----------------------------------------------------------------------- #
    # Validators
    # ----------------------------------------------------------------------- #
    @field_validator("data_root", mode="before")
    @classmethod
    def _blank_data_root_means_unset(cls, value: object) -> object:
        """An empty ``FORESIGHT_DATA_ROOT=`` must mean "use the default", not
        "use the current directory" - ``Path("")`` resolves to ``.`` and would
        silently point every read/write at wherever the process happens to be
        started from.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("log_format", mode="before")
    @classmethod
    def _lower_log_format(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_coherence(self) -> Settings:
        """Reject combinations that are individually valid but jointly wrong."""
        if self.history_end <= self.history_start:
            raise ValueError(
                f"history_end ({self.history_end}) must be after "
                f"history_start ({self.history_start})"
            )

        span_days = (self.history_end - self.history_start).days
        if span_days < _MIN_HISTORY_DAYS:
            raise ValueError(
                f"history spans {span_days} days; at least {_MIN_HISTORY_DAYS} are "
                "needed to learn annual seasonality and still hold out a backtest"
            )

        if self.lead_time_min_days >= self.lead_time_max_days:
            raise ValueError(
                f"lead_time_min_days ({self.lead_time_min_days}) must be below "
                f"lead_time_max_days ({self.lead_time_max_days})"
            )

        # The backtest consumes horizon + (folds - 1) * step weeks of holdout on
        # top of the minimum training window. If history cannot cover that, the
        # backtest would silently shrink and the reported WAPE would not mean
        # what the README claims.
        history_weeks = span_days // 7
        holdout_weeks = self.horizon_weeks + (self.backtest_folds - 1) * self.backtest_step_weeks
        required_weeks = self.min_train_weeks + holdout_weeks
        if history_weeks < required_weeks:
            raise ValueError(
                f"history provides {history_weeks} weeks but the configured backtest "
                f"needs {required_weeks} (min_train_weeks={self.min_train_weeks} + "
                f"horizon={self.horizon_weeks} + "
                f"{self.backtest_folds - 1} x step={self.backtest_step_weeks}). "
                "Lengthen the history or reduce folds/step/min_train_weeks."
            )
        return self

    # ----------------------------------------------------------------------- #
    # Derived values
    # ----------------------------------------------------------------------- #
    @property
    def quantile_levels(self) -> tuple[float, float]:
        """Lower/upper quantiles implied by ``interval_coverage``."""
        tail = (1.0 - self.interval_coverage) / 2.0
        return (round(tail, 6), round(1.0 - tail, 6))

    @property
    def cors_origin_list(self) -> list[str]:
        """``cors_origins`` split into a clean list.

        A literal ``*`` is honoured but should only ever be used for a throwaway
        demo - it lets any site on the internet call the API from a browser.
        """
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # --- Canonical paths ---------------------------------------------------- #
    @property
    def instance_root(self) -> Path:
        """Where this instance's data/artifacts/reports live.

        ``PROJECT_ROOT`` unless ``data_root`` overrides it - see that field's
        docstring. Code (``scripts/``, ``dashboard/dist``) always resolves
        relative to ``PROJECT_ROOT`` regardless, since only one copy of the
        code exists; only this instance's *state* moves.
        """
        return self.data_root or PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return self.instance_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.instance_root / "artifacts"

    @property
    def reports_dir(self) -> Path:
        return self.instance_root / "reports"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def auth_db_path(self) -> Path:
        """SQLite file holding registered users and active sessions."""
        return self.data_dir / "auth.db"

    @property
    def google_oauth_configured(self) -> bool:
        """True once the placeholder Google client id/secret have been replaced."""
        return (
            self.google_client_id != _PLACEHOLDER_GOOGLE_CLIENT_ID
            and self.google_client_secret != _PLACEHOLDER_GOOGLE_CLIENT_SECRET
            and bool(self.google_client_id)
            and bool(self.google_client_secret)
        )

    def ensure_directories(self) -> None:
        """Create every output directory this project writes to."""
        for directory in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.artifacts_dir,
            self.reports_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Raises:
        ConfigurationError: if any value fails validation. The pydantic error is
            re-raised as a project exception with a readable summary, so a bad
            ``.env`` produces an actionable message instead of a stack trace.
    """
    try:
        return Settings()
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(part) for part in err['loc']) or 'config'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ConfigurationError(
            "Invalid FORESIGHT configuration:\n  - " + "\n  - ".join(problems)
        ) from exc
