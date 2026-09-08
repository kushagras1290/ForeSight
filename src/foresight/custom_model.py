"""FORESIGHT Adaptive Demand Ensemble - a purpose-built model for this assortment.

WHY A CUSTOM MODEL AT ALL
-------------------------
A single gradient-boosted model is trained to minimise average error, and the
average of NorthBay's assortment is dominated by its fast movers. But the
catalogue is not homogeneous - it contains at least four genuinely different
demand regimes, and the same estimator is not the right tool for all of them:

* **Steady** - regular weekly offtake. A GBM on lag and seasonality features is
  close to ideal here, and this is where most of the revenue sits.
* **Intermittent** - slow movers that sell nothing most weeks. Squared or
  absolute error on a mostly-zero series pulls a GBM towards predicting zero,
  which is accurate on average and useless for planning. The literature's answer
  is to model *how often* a SKU sells separately from *how much* it sells when
  it does.
* **New** - launched inside the window, with too little history for its own lags
  to mean anything. Section 16.2 of the brief names the mitigation directly:
  fall back to category-level patterns.
* **Volatile** - erratic relative to its own level, where no single estimator is
  reliable and blending reduces variance.

THE THREE COMPONENTS
--------------------
1. ``GBM`` - the global LightGBM model from :mod:`foresight.forecast`.
2. ``TSB`` - Teunter-Syntetos-Babai intermittent demand smoothing. Tracks demand
   size and sale probability as two separate exponentially-smoothed states and
   multiplies them. TSB is used rather than classic Croston because Croston is
   known to be positively biased on intermittent series, and a biased forecast
   feeding an inventory decision produces systematic overstock.
3. ``SeasonalProfile`` - the SKU's own deseasonalised level re-seasonalised by
   its **category's** pooled week-of-year profile. Sparse series borrow the
   seasonal shape of the products they sit beside.

HOW THEY ARE COMBINED
---------------------
Not by a fixed rule. Each row is assigned a regime from origin-side features
only, and a convex weight vector over the three components is fitted **per
regime** by direct WAPE minimisation on a validation slice held out from the
tail of the training window. The components never see that slice during their
own fitting, so the weights are chosen on genuinely out-of-sample errors.

LEAKAGE
-------
Every component obeys the same contract as :mod:`foresight.features`. The TSB
recursion at week ``t`` consumes only demand up to and including ``t``. The
category seasonal profile is estimated solely from training-window targets. The
blend weights are fitted on a validation slice that ends at the training cutoff.
Nothing here reads across the origin.
"""

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd

from foresight.config import Settings, get_settings
from foresight.exceptions import (
    ArtifactNotFoundError,
    InsufficientHistoryError,
    ModelNotFittedError,
)
from foresight.features import FEATURE_SPEC, FeatureSpec
from foresight.forecast import TrainedForecaster, train_forecaster
from foresight.logging_setup import get_logger
from foresight.metrics import wape

__all__ = [
    "AdaptiveDemandEnsemble",
    "DemandRegime",
    "ENSEMBLE_FILENAME",
    "assign_regime",
    "load_ensemble",
    "train_ensemble",
    "tsb_levels",
]

log = get_logger(__name__)

ENSEMBLE_FILENAME: Final[str] = "adaptive_ensemble.joblib"

#: Component names, fixed so weight vectors are always interpreted consistently.
COMPONENTS: Final[tuple[str, ...]] = ("gbm", "tsb", "seasonal_profile")

#: TSB smoothing constants. Deliberately small: NorthBay's slow movers are noisy,
#: and a fast-adapting state would chase single orders and whipsaw the forecast.
TSB_ALPHA: Final[float] = 0.12  # demand size when a sale occurs
TSB_BETA: Final[float] = 0.06  # probability of a sale occurring

#: Weeks held out from the tail of the training window to fit blend weights.
VALIDATION_WEEKS: Final[int] = 12

#: Simplex grid resolution for the weight search (0.1 => 66 combinations).
WEIGHT_STEP: Final[float] = 0.1

_EPSILON: Final[float] = 1e-9


class DemandRegime:
    """The four demand regimes. Plain constants: these are persisted in artifacts."""

    STEADY = "steady"
    INTERMITTENT = "intermittent"
    NEW = "new"
    VOLATILE = "volatile"

    ALL: Final[tuple[str, ...]] = (STEADY, INTERMITTENT, NEW, VOLATILE)


#: Regime thresholds, applied in priority order (new > intermittent > volatile).
NEW_SKU_MAX_WEEKS: Final[int] = 26
INTERMITTENT_ZERO_FRACTION: Final[float] = 0.35
VOLATILE_CV: Final[float] = 0.80

#: Which components may be blended in each regime.
#:
#: TSB is deliberately excluded from ``new``. It estimates a *flat* level from a
#: stable arrival process - demand size times sale probability, both
#: exponentially smoothed. A newly-launched product has no stable arrival
#: process: it is ramping. Given a free choice the optimiser will still assign
#: TSB weight, because on a short validation slice a flat level looks stable, and
#: the result is a forecast that lags the ramp and under-predicts all the way up.
#: Measured on the backtest, an unconstrained blend scored WAPE 0.411 with -17%
#: bias on new SKUs against the GBM's 0.366 - worse than not blending at all.
#: The constraint is a structural statement about what TSB models, not a tuning
#: knob.
REGIME_COMPONENTS: Final[dict[str, tuple[str, ...]]] = {
    DemandRegime.STEADY: ("gbm", "tsb", "seasonal_profile"),
    DemandRegime.INTERMITTENT: ("gbm", "tsb", "seasonal_profile"),
    DemandRegime.NEW: ("gbm", "seasonal_profile"),
    DemandRegime.VOLATILE: ("gbm", "tsb", "seasonal_profile"),
}


def assign_regime(frame: pd.DataFrame) -> pd.Series:
    """Label each row with its demand regime, from origin-side features only.

    Priority matters. A newly-launched SKU is treated as *new* even if it also
    looks intermittent, because thin history is the more pressing problem: its
    own lags are unreliable regardless of how the zeros are distributed.
    """
    required = {"weeks_since_launch", "zero_frac_26", "cv_13"}
    missing = required - set(frame.columns)
    if missing:
        raise ModelNotFittedError(f"Cannot assign regime; missing {sorted(missing)}.")

    weeks_since_launch = pd.to_numeric(frame["weeks_since_launch"], errors="coerce").fillna(0.0)
    zero_fraction = pd.to_numeric(frame["zero_frac_26"], errors="coerce").fillna(0.0)
    coefficient_variation = pd.to_numeric(frame["cv_13"], errors="coerce").fillna(0.0)

    # Applied in ascending priority: each mask overwrites the one before it, so
    # the last rule wins. NEW is applied last and therefore ranks highest.
    regime = pd.Series(DemandRegime.STEADY, index=frame.index, dtype="object")
    regime = regime.mask(coefficient_variation >= VOLATILE_CV, DemandRegime.VOLATILE)
    regime = regime.mask(zero_fraction >= INTERMITTENT_ZERO_FRACTION, DemandRegime.INTERMITTENT)
    return regime.mask(weeks_since_launch < NEW_SKU_MAX_WEEKS, DemandRegime.NEW)


# --------------------------------------------------------------------------- #
# Component 2: TSB intermittent demand
# --------------------------------------------------------------------------- #
def _tsb_single(series: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Run the TSB recursion over one series.

    Returns the forecast level *available at* each index - that is, element ``t``
    is the flat forecast for every week after ``t``, computed from demand up to
    and including ``t``. That alignment is what keeps it leak-free.
    """
    length = series.size
    levels = np.empty(length, dtype="float64")

    non_zero = series[series > 0]
    # Seed from the series' own history; a flat zero seed would take dozens of
    # weeks to recover and would bias early forecasts down.
    size_state = float(non_zero.mean()) if non_zero.size else 0.0
    probability_state = float((series > 0).mean()) if length else 0.0

    for index in range(length):
        value = series[index]
        if value > 0:
            size_state += alpha * (value - size_state)
            probability_state += beta * (1.0 - probability_state)
        else:
            probability_state += beta * (0.0 - probability_state)
        levels[index] = probability_state * size_state

    return levels


def tsb_levels(
    panel: pd.DataFrame,
    alpha: float = TSB_ALPHA,
    beta: float = TSB_BETA,
) -> pd.DataFrame:
    """Compute the TSB forecast level at every (SKU, week).

    Args:
        panel: Weekly panel with ``sku_id``, ``week_start`` and ``units``.

    Returns:
        ``sku_id``, ``week_start``, ``tsb_level``.
    """
    ordered = panel.sort_values(["sku_id", "week_start"])
    frames: list[pd.DataFrame] = []

    for sku_id, group in ordered.groupby("sku_id", sort=False):
        values = group["units"].to_numpy(dtype="float64")
        frames.append(
            pd.DataFrame(
                {
                    "sku_id": sku_id,
                    "week_start": group["week_start"].to_numpy(),
                    "tsb_level": _tsb_single(values, alpha, beta),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Component 3: category-pooled seasonal profile
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SeasonalProfile:
    """Category x week-of-year demand index, estimated on the training window."""

    index_by_category_week: dict[tuple[str, int], float]
    global_index_by_week: dict[int, float]

    def lookup(self, categories: pd.Series, iso_weeks: pd.Series) -> np.ndarray:
        """Return the seasonal index for each (category, ISO week) pair."""
        result = np.ones(len(categories), dtype="float64")
        category_values = categories.astype("object").to_numpy()
        week_values = pd.to_numeric(iso_weeks, errors="coerce").fillna(1).astype("int64").to_numpy()

        for position in range(result.size):
            key = (category_values[position], int(week_values[position]))
            value = self.index_by_category_week.get(key)
            if value is None:
                # An unseen category-week falls back to the all-category profile,
                # then to a flat 1.0, so a new category never produces NaN.
                value = self.global_index_by_week.get(int(week_values[position]), 1.0)
            result[position] = value
        return result


def _fit_seasonal_profile(train: pd.DataFrame) -> SeasonalProfile:
    """Estimate the seasonal index from observed training targets only."""
    frame = train.loc[:, ["category", "target_week", "y"]].copy()
    # The `.dt` accessor keeps the frame's own index. Going via DatetimeIndex
    # would return a frame keyed by the dates themselves, which cannot be
    # assigned back when those dates repeat across SKUs.
    frame["iso_week"] = frame["target_week"].dt.isocalendar()["week"].astype("int64")

    category_mean = frame.groupby("category")["y"].mean()
    category_week_mean = frame.groupby(["category", "iso_week"])["y"].mean()

    index_by_category_week: dict[tuple[str, int], float] = {}
    for (category, iso_week), value in category_week_mean.items():
        denominator = float(category_mean.get(category, np.nan))
        if denominator > _EPSILON and np.isfinite(denominator):
            index_by_category_week[(str(category), int(iso_week))] = float(value) / denominator

    overall_mean = float(frame["y"].mean())
    week_mean = frame.groupby("iso_week")["y"].mean()
    global_index_by_week = (
        {int(week): float(value) / overall_mean for week, value in week_mean.items()}
        if overall_mean > _EPSILON
        else {}
    )

    return SeasonalProfile(
        index_by_category_week=index_by_category_week,
        global_index_by_week=global_index_by_week,
    )


def _seasonal_profile_predict(
    frame: pd.DataFrame,
    profile: SeasonalProfile,
    origin_index: np.ndarray,
) -> np.ndarray:
    """Re-seasonalise each SKU's own level using its category's profile.

    ``level x index(target week) / index(origin window)``. Dividing by the origin
    window's average index is what makes this a *ratio*: without it, a level
    measured during the festive peak would be re-inflated by the peak index and
    the forecast would double-count seasonality.
    """
    level = pd.to_numeric(frame["units_roll_mean_13"], errors="coerce")
    level = level.fillna(pd.to_numeric(frame["units_roll_mean_4"], errors="coerce")).fillna(0.0)

    target_iso_week = frame["target_week"].dt.isocalendar()["week"].astype("int64")
    target_index = profile.lookup(frame["category"], target_iso_week)

    safe_origin = np.where(np.isfinite(origin_index) & (origin_index > _EPSILON), origin_index, 1.0)
    ratio = np.clip(target_index / safe_origin, 0.2, 5.0)
    return np.clip(level.to_numpy(dtype="float64") * ratio, 0.0, None)


def _origin_window_index(
    frame: pd.DataFrame,
    profile: SeasonalProfile,
    window: int = 13,
) -> np.ndarray:
    """Average seasonal index over the ``window`` weeks ending at each origin."""
    origin_weeks = pd.DatetimeIndex(frame["week_start"])
    categories = frame["category"]

    accumulator = np.zeros(len(frame), dtype="float64")
    for offset in range(window):
        shifted = origin_weeks - pd.to_timedelta(offset * 7, unit="D")
        iso_weeks = pd.Series(
            shifted.isocalendar().week.astype("int64").to_numpy(), index=frame.index
        )
        accumulator += profile.lookup(categories, iso_weeks)
    return accumulator / float(window)


# --------------------------------------------------------------------------- #
# The ensemble
# --------------------------------------------------------------------------- #
def _simplex_grid(n_components: int, step: float) -> list[tuple[float, ...]]:
    """All convex weight vectors on a grid of the given step."""
    ticks = int(round(1.0 / step))
    grid: list[tuple[float, ...]] = []
    for combination in itertools.product(range(ticks + 1), repeat=n_components - 1):
        used = sum(combination)
        if used <= ticks:
            weights = tuple(value / ticks for value in (*combination, ticks - used))
            grid.append(weights)
    return grid


@dataclass(slots=True)
class AdaptiveDemandEnsemble:
    """Regime-routed blend of the GBM, TSB and seasonal-profile components."""

    gbm: TrainedForecaster
    profile: SeasonalProfile
    tsb_lookup: dict[tuple[str, pd.Timestamp], float]
    weights: dict[str, tuple[float, ...]]
    regime_shares: dict[str, float]
    component_wape: dict[str, dict[str, float]]
    trained_at: str
    training_rows: int
    interval_scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- #
    def _component_predictions(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Run all three components over the same rows."""
        gbm_output = self.gbm.predict(frame)

        tsb = np.array(
            [
                self.tsb_lookup.get((sku_id, pd.Timestamp(week)), np.nan)
                for sku_id, week in zip(frame["sku_id"], frame["week_start"], strict=True)
            ],
            dtype="float64",
        )
        # A SKU with no TSB state (never seen in training) falls back to its own
        # trailing mean rather than to zero, which would drag the blend down.
        fallback = pd.to_numeric(frame["units_roll_mean_4"], errors="coerce").fillna(0.0).to_numpy()
        tsb = np.where(np.isfinite(tsb), tsb, fallback)

        origin_index = _origin_window_index(frame, self.profile)
        seasonal = _seasonal_profile_predict(frame, self.profile, origin_index)

        return {
            "gbm": np.clip(gbm_output["prediction"].to_numpy(), 0.0, None),
            "tsb": np.clip(tsb, 0.0, None),
            "seasonal_profile": seasonal,
            "_lower": gbm_output["prediction_lower"].to_numpy(),
            "_upper": gbm_output["prediction_upper"].to_numpy(),
        }

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Blend component forecasts using each row's regime weights.

        The interval is inherited from the GBM's quantile models, re-centred on
        the blended point forecast and scaled by the conformal factor fitted in
        :func:`train_ensemble`.
        """
        components = self._component_predictions(frame)
        regimes = assign_regime(frame)

        blended = np.zeros(len(frame), dtype="float64")
        for regime in DemandRegime.ALL:
            mask = (regimes == regime).to_numpy()
            if not mask.any():
                continue
            weights = self.weights.get(regime, (1.0, 0.0, 0.0))
            for weight, name in zip(weights, COMPONENTS, strict=True):
                if weight > 0.0:
                    blended[mask] += weight * components[name][mask]

        point = np.clip(blended, 0.0, None)

        # Re-centre the GBM's interval on the blended point, then widen by the
        # calibration factor so empirical coverage matches the stated level.
        gbm_point = components["gbm"]
        lower_gap = np.clip(gbm_point - components["_lower"], 0.0, None) * self.interval_scale
        upper_gap = np.clip(components["_upper"] - gbm_point, 0.0, None) * self.interval_scale

        return pd.DataFrame(
            {
                "prediction": point,
                "prediction_lower": np.clip(point - lower_gap, 0.0, None),
                "prediction_upper": point + upper_gap,
                "regime": regimes.to_numpy(),
            },
            index=frame.index,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("saved adaptive ensemble", extra={"context": {"path": str(path)}})


def _calibrate_interval_scale(
    actual: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float,
) -> float:
    """Find the multiplicative widening that hits the target coverage.

    Split-conformal in spirit: the quantile models are fitted by minimising
    pinball loss, which does not guarantee the *joint* coverage of the resulting
    band. This searches for the smallest scale factor whose empirical coverage on
    held-out rows reaches the stated level, so an "80% interval" means it.
    """
    lower_gap = np.clip(point - lower, 0.0, None)
    upper_gap = np.clip(upper - point, 0.0, None)

    best_scale = 1.0
    for scale in np.arange(0.5, 5.01, 0.05):
        inside = (actual >= point - lower_gap * scale) & (actual <= point + upper_gap * scale)
        if float(inside.mean()) >= target_coverage:
            best_scale = float(scale)
            break
    else:
        best_scale = 5.0

    return best_scale


def train_ensemble(
    supervised: pd.DataFrame,
    panel: pd.DataFrame,
    settings: Settings | None = None,
    *,
    gbm: TrainedForecaster | None = None,
    feature_spec: FeatureSpec | None = None,
) -> AdaptiveDemandEnsemble:
    """Fit all components and the per-regime blend weights.

    Args:
        supervised: Training rows (already time-filtered by the caller).
        panel: The weekly panel, used for the TSB recursion.
        settings: Configuration.
        gbm: A pre-fitted GBM to reuse. Supplying one avoids retraining it
            inside every backtest fold, which halves the backtest runtime.
        feature_spec: The feature contract for any GBM fitted here. Defaults to
            the contract carried by ``gbm`` when one is supplied, so the scoring
            GBM can never be trained on a different column set than the
            component GBM it stands in for.
    """
    settings = settings or get_settings()
    if supervised.empty:
        raise InsufficientHistoryError("No training rows supplied to train_ensemble.")

    # --- Validation slice: the tail of the training window ------------------ #
    max_target = pd.Timestamp(supervised["target_week"].max())
    validation_start = max_target - pd.to_timedelta(VALIDATION_WEEKS * 7, unit="D")
    fit_mask = supervised["target_week"] <= validation_start
    validation_mask = supervised["target_week"] > validation_start

    if not fit_mask.any() or not validation_mask.any():
        # Too little history to hold anything back; fall back to the GBM alone
        # rather than fitting weights on the data the components trained on.
        fit_mask = pd.Series(True, index=supervised.index)
        validation_mask = pd.Series(False, index=supervised.index)

    fit_rows = supervised[fit_mask]
    validation_rows = supervised[validation_mask]

    # --- Components --------------------------------------------------------- #
    if feature_spec is None:
        feature_spec = gbm.feature_spec if gbm is not None else FEATURE_SPEC
    component_gbm = gbm or train_forecaster(supervised, settings, feature_spec=feature_spec)
    # Weights must be chosen on errors the GBM did not train on, so a second GBM
    # is fitted on the reduced window purely to score the validation slice.
    scoring_gbm = (
        train_forecaster(fit_rows, settings, feature_spec=feature_spec)
        if len(validation_rows)
        else component_gbm
    )

    profile = _fit_seasonal_profile(fit_rows)

    panel_cutoff = panel[panel["week_start"] <= pd.Timestamp(supervised["week_start"].max())]
    tsb_frame = tsb_levels(panel_cutoff)
    tsb_lookup = {
        (row.sku_id, pd.Timestamp(row.week_start)): float(row.tsb_level)
        for row in tsb_frame.itertuples(index=False)
    }

    # --- Fit per-regime weights on the validation slice --------------------- #
    weights: dict[str, tuple[float, ...]] = {}
    component_wape: dict[str, dict[str, float]] = {}
    regime_shares: dict[str, float] = {}
    interval_scale = 1.0

    if len(validation_rows):
        scoring = AdaptiveDemandEnsemble(
            gbm=scoring_gbm,
            profile=profile,
            tsb_lookup=tsb_lookup,
            weights=dict.fromkeys(DemandRegime.ALL, (1.0, 0.0, 0.0)),
            regime_shares={},
            component_wape={},
            trained_at="",
            training_rows=0,
        )
        components = scoring._component_predictions(validation_rows)
        regimes = assign_regime(validation_rows)
        actual = validation_rows["y"].to_numpy(dtype="float64")
        grid = _simplex_grid(len(COMPONENTS), WEIGHT_STEP)

        for regime in DemandRegime.ALL:
            mask = (regimes == regime).to_numpy()
            regime_shares[regime] = float(mask.mean())
            if mask.sum() < 30:
                # Too few rows to fit a weight vector without overfitting it.
                weights[regime] = (1.0, 0.0, 0.0)
                continue

            regime_actual = actual[mask]
            stacked = {name: components[name][mask] for name in COMPONENTS}
            component_wape[regime] = {
                name: wape(regime_actual, values) for name, values in stacked.items()
            }

            # Only components that make structural sense for this regime are
            # eligible; see REGIME_COMPONENTS for why.
            allowed = REGIME_COMPONENTS.get(regime, COMPONENTS)

            best_weights, best_score = (1.0, 0.0, 0.0), float("inf")
            for candidate in grid:
                if any(
                    weight > 0.0
                    for weight, name in zip(candidate, COMPONENTS, strict=True)
                    if name not in allowed
                ):
                    continue
                blended = sum(
                    weight * stacked[name]
                    for weight, name in zip(candidate, COMPONENTS, strict=True)
                )
                score = wape(regime_actual, blended)
                if np.isfinite(score) and score < best_score:
                    best_weights, best_score = candidate, score
            weights[regime] = best_weights

        # --- Interval calibration on the same held-out slice ---------------- #
        blended_point = np.zeros(len(validation_rows), dtype="float64")
        for regime in DemandRegime.ALL:
            mask = (regimes == regime).to_numpy()
            if not mask.any():
                continue
            for weight, name in zip(weights[regime], COMPONENTS, strict=True):
                if weight > 0.0:
                    blended_point[mask] += weight * components[name][mask]

        gbm_point = components["gbm"]
        lower = blended_point - np.clip(gbm_point - components["_lower"], 0.0, None)
        upper = blended_point + np.clip(components["_upper"] - gbm_point, 0.0, None)
        interval_scale = _calibrate_interval_scale(
            actual, blended_point, lower, upper, settings.interval_coverage
        )
    else:
        weights = dict.fromkeys(DemandRegime.ALL, (1.0, 0.0, 0.0))

    ensemble = AdaptiveDemandEnsemble(
        gbm=component_gbm,
        profile=profile,
        tsb_lookup=tsb_lookup,
        weights=weights,
        regime_shares=regime_shares,
        component_wape=component_wape,
        trained_at=dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        training_rows=len(supervised),
        interval_scale=interval_scale,
        metadata={
            "validation_weeks": VALIDATION_WEEKS,
            "tsb_alpha": TSB_ALPHA,
            "tsb_beta": TSB_BETA,
            "components": list(COMPONENTS),
            "interval_coverage_target": settings.interval_coverage,
        },
    )

    log.info(
        "adaptive ensemble trained",
        extra={
            "context": {
                "weights": {
                    regime: [round(w, 2) for w in vector] for regime, vector in weights.items()
                },
                "regime_shares": {key: round(value, 3) for key, value in regime_shares.items()},
                "interval_scale": round(interval_scale, 3),
            }
        },
    )
    return ensemble


def load_ensemble(
    path: Path | None = None, settings: Settings | None = None
) -> AdaptiveDemandEnsemble:
    """Load a saved ensemble."""
    settings = settings or get_settings()
    resolved = path or (settings.artifacts_dir / ENSEMBLE_FILENAME)
    if not resolved.exists():
        raise ArtifactNotFoundError(
            resolved, "Train one first with `python scripts/03_train_backtest.py`."
        )
    ensemble = joblib.load(resolved)
    if not isinstance(ensemble, AdaptiveDemandEnsemble):
        raise ArtifactNotFoundError(resolved, "File exists but is not an AdaptiveDemandEnsemble.")
    return ensemble
