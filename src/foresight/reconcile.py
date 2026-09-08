"""FORESIGHT Hierarchical Demand Reconciler - the second custom model.

WHAT THIS SOLVES THAT THE ENSEMBLE DOES NOT
-------------------------------------------
:mod:`foresight.custom_model` improves the forecast *for one SKU* by choosing a
better estimator for that SKU's demand regime. It works entirely within the
bottom level of the assortment.

This model attacks a different, orthogonal source of error. Demand at a single
SKU is mostly idiosyncratic noise; demand at a category is not, because that
noise cancels across products. This engagement's own backtest measures the gap
directly - the same model on the same data scores far better at category-week
than at SKU-week. Bottom-up planning discards that entirely: it sums 200 noisy
forecasts, so the category plan inherits every SKU's error instead of averaging
them away.

The fix is to forecast each level *independently* - the SKU series, the
subcategory series, the category series and the total - and then combine all of
those forecasts into one coherent set. The aggregate forecasts are more reliable,
so they pull the SKU forecasts toward a total that is more likely to be right.
The SKU forecasts still decide how that total is split. Information flows in both
directions, which is the property bottom-up and top-down each give up.

THE ALGORITHM: MinT
-------------------
Let ``S`` be the summing matrix (see :mod:`foresight.hierarchy`), ``b`` the
unknown true SKU demands and ``y_hat`` the independent base forecasts for all
``n`` nodes. Any coherent reconciliation is a linear map ``y_tilde = S P y_hat``.

Wickramasuriya, Athanasopoulos & Hyndman (2019) show that among all such maps
that are unbiased, the one minimising the trace of the reconciled error
covariance is::

    P = (S' W^-1 S)^-1 S' W^-1

where ``W`` is the covariance of the base forecast errors. This is *Minimum
Trace* reconciliation, and it is optimal rather than heuristic: no other linear
coherent combination of these base forecasts has lower total error variance.

Two properties matter operationally:

* **Coherence is structural.** ``y_tilde = S b_tilde`` for some ``b_tilde``, so
  the SKU forecasts sum to the subcategory forecast, which sums to the category
  forecast, which sums to the total. Planners stop reconciling spreadsheets by
  hand, and the category budget and the SKU buy plan can no longer disagree.
* **Unbiasedness is preserved.** ``P S = I``, so if the base forecasts are
  unbiased the reconciled ones are too. Accuracy is bought without introducing
  the systematic skew that a top-down proration would.

ESTIMATING W - WHERE THE ENGINEERING IS
---------------------------------------
``W`` is an ``n x n`` covariance, and ``n`` here (one total, plus categories,
subcategories and 200 SKUs) is of the same order as the number of residual
observations available. The sample covariance is then unusably ill-conditioned,
and inverting it amplifies noise instead of removing it. Two decisions handle
that:

1. **Shrinkage.** The sample correlation is shrunk toward the identity with the
   Schaefer-Strimmer analytic intensity, which needs no cross-validation and
   degrades gracefully: as evidence thins the intensity approaches one and the
   estimator falls back to weighted least squares by variance, which is the
   standard safe choice. There is no configuration to get wrong.

2. **Pooled correlation, per-horizon scale.** Forecast errors grow with horizon,
   so a single ``W`` across all horizons would misweight both ends. But
   estimating eight separate correlation matrices splits already-thin evidence
   eight ways. Instead the *correlation* is estimated once from residuals
   standardised within each horizon - correlation between two nodes is a
   structural property of the assortment, not of how far ahead one is looking -
   while the *scales* stay horizon-specific. So ``W_h = D_h R D_h``.

Residuals must be out-of-sample or ``W`` describes how well the model fits data
it has already seen, and the reconciliation optimises for the wrong thing.
:func:`harvest_residuals` enforces that by training once at a cutoff and scoring
only origins after it.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

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
from foresight.hierarchy import Hierarchy
from foresight.logging_setup import get_logger

__all__ = [
    "RECONCILER_FILENAME",
    "HierarchicalReconciler",
    "harvest_residuals",
    "load_reconciler",
    "shrunk_correlation",
    "train_reconciler",
]

log = get_logger(__name__)

RECONCILER_FILENAME: Final[str] = "reconciler.joblib"


class _SupportsPredict(Protocol):
    """Anything that turns a supervised frame into a ``prediction`` column."""

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame: ...


#: Added to the correlation diagonal before solving. Guards the one degenerate
#: case shrinkage does not: a node whose residuals are exactly constant, which
#: has zero variance and would otherwise make the matrix singular.
_RIDGE: Final[float] = 1e-8

#: Floor on a node's residual standard deviation. A node that never erred in the
#: residual window is not error-free, it is under-observed; treating it as
#: error-free would give it unbounded weight in the reconciliation.
_MIN_SCALE: Final[float] = 1e-3

#: Minimum residual vectors required before a correlation is worth estimating.
_MIN_OBSERVATIONS: Final[int] = 30

#: Residual observations a single node needs before its own scale is trusted.
_MIN_NODE_OBSERVATIONS: Final[int] = 8

#: Scale multiplier for a node with too little residual evidence. Inflating the
#: assumed error is what stops MinT from trusting a node it cannot assess.
_UNKNOWN_INFLATION: Final[float] = 2.0

#: Share of the residual window held back to choose the blend strength on.
_VALIDATION_FRACTION: Final[float] = 0.35

#: Blend strengths tried. Coarse on purpose - the curve is flat near its optimum,
#: and a finer grid would only fit the validation slice's own noise.
_LAMBDA_GRID: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0)

#: Least of the assortment that must be present at an origin. Ragged coverage is
#: normal - products launch - but if half the catalogue is missing, the caller
#: has passed a partial frame and the aggregates would be reconciling against
#: totals that do not describe it.
_MIN_BOTTOM_COVERAGE: Final[float] = 0.5


# --------------------------------------------------------------------------- #
# Covariance estimation
# --------------------------------------------------------------------------- #
def shrunk_correlation(observations: np.ndarray) -> tuple[np.ndarray, float]:
    """Sample correlation shrunk toward the identity, Schaefer-Strimmer intensity.

    The intensity is derived analytically from the variance of the individual
    correlation estimates, so it needs no tuning and no held-out data::

        lambda* = sum_{i != j} Var(r_ij) / sum_{i != j} r_ij^2

    Read directly: shrink hard when the off-diagonal estimates are themselves
    noisy, shrink little when they are large and stable. If evidence is thin the
    intensity tends to one, the correlation collapses to the identity, and MinT
    degenerates to weighted least squares by variance - a worse estimator, but
    never a broken one.

    Args:
        observations: ``(T, n)`` matrix of residuals, one row per observation.

    Returns:
        The ``(n, n)`` shrunk correlation matrix and the intensity used.

    Raises:
        InsufficientHistoryError: with fewer than two observations.
    """
    matrix = np.asarray(observations, dtype="float64")
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D residual matrix, received shape {matrix.shape}")
    n_obs, n_series = matrix.shape
    if n_obs < 2:
        raise InsufficientHistoryError(
            f"Correlation needs at least 2 observations, received {n_obs}."
        )

    centred = matrix - matrix.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0, ddof=1)
    scale = np.where(scale > _MIN_SCALE, scale, _MIN_SCALE)
    standardised = centred / scale

    # Sample correlation.
    cross = standardised.T @ standardised
    correlation = cross / (n_obs - 1)

    # Var(r_ij) without ever materialising the T x n x n tensor. With
    # w_tij = x_ti * x_tj, the identity
    #     sum_t (w_tij - mean_ij)^2 = sum_t w_tij^2 - (sum_t w_tij)^2 / T
    # turns both terms into plain matrix products.
    squared = standardised**2
    sum_w_squared = squared.T @ squared
    variance = (n_obs / (n_obs - 1) ** 3) * (sum_w_squared - (cross**2) / n_obs)

    off_diagonal = ~np.eye(n_series, dtype=bool)
    denominator = float(np.sum(correlation[off_diagonal] ** 2))
    # No off-diagonal signal at all means there is nothing to shrink toward the
    # identity - the estimate already is the identity, so shrink fully.
    intensity = 1.0 if denominator <= 0.0 else float(np.sum(variance[off_diagonal]) / denominator)
    intensity = float(np.clip(intensity, 0.0, 1.0))

    shrunk = (1.0 - intensity) * correlation
    np.fill_diagonal(shrunk, 1.0)
    return shrunk, intensity


def _mint_projection(
    summing: np.ndarray,
    correlation: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Compute ``P = (S' W^-1 S)^-1 S' W^-1`` for ``W = D R D``.

    ``W`` is never formed or inverted directly. Factoring out the diagonal keeps
    the only genuine linear solves at the size of the correlation matrix and the
    bottom level, and keeps the scaling exact rather than accumulated.
    """
    n_nodes, n_bottom = summing.shape
    safe_scale = np.where(scale > _MIN_SCALE, scale, _MIN_SCALE).reshape(n_nodes, 1)

    ridged = correlation + _RIDGE * np.eye(n_nodes)

    # W^-1 S = D^-1 R^-1 D^-1 S, applied right to left.
    scaled_summing = summing / safe_scale
    solved = np.linalg.solve(ridged, scaled_summing)
    w_inv_summing = solved / safe_scale

    normal = summing.T @ w_inv_summing
    normal += _RIDGE * np.eye(n_bottom)

    projection = np.linalg.solve(normal, w_inv_summing.T)
    return np.asarray(projection, dtype="float64")


def _bottom_up_projection(hierarchy: Hierarchy) -> np.ndarray:
    """The projection that ignores every aggregate and keeps the SKU forecasts.

    This is plain bottom-up planning written as a matrix, and it is the identity
    element of the blend below: at ``lambda = 0`` reconciliation does nothing at
    all, which is what makes the safety net safe.
    """
    projection = np.zeros((hierarchy.n_bottom, hierarchy.n_nodes), dtype="float64")
    for column, node in enumerate(hierarchy.bottom):
        projection[column, hierarchy.node_index[node]] = 1.0
    return projection


def _aggregate_advantage(validation: pd.DataFrame, hierarchy: Hierarchy) -> float | None:
    """How much better a directly-forecast aggregate is than summing its children.

    Positive means the aggregate models know something the bottom level does not,
    which is the only condition under which reconciling toward them can help.
    Negative means the opposite, and the honest answer is to leave the SKU
    forecasts alone.

    Returns ``None`` when the comparison cannot be made at all.
    """
    level_of = dict(zip(hierarchy.nodes, hierarchy.levels, strict=True))
    membership = {
        node: [hierarchy.bottom[column] for column in np.flatnonzero(row)]
        for node, row in zip(hierarchy.nodes, hierarchy.summing, strict=True)
        if level_of[node] != "sku"
    }

    bottom_forecasts = validation[validation["node"].map(level_of) == "sku"]
    if bottom_forecasts.empty:
        return None
    lookup = {
        (str(row.node), pd.Timestamp(row.week_start), int(row.horizon)): float(row.prediction)
        for row in bottom_forecasts.itertuples(index=False)
    }

    direct_error = 0.0
    bottom_up_error = 0.0
    total_actual = 0.0

    aggregates = validation[validation["node"].map(level_of) != "sku"]
    for row in aggregates.itertuples(index=False):
        children = membership.get(str(row.node))
        if not children:
            continue
        key = (pd.Timestamp(row.week_start), int(row.horizon))
        summed = sum(lookup.get((child, *key), 0.0) for child in children)
        actual = float(row.actual)
        direct_error += abs(actual - float(row.prediction))
        bottom_up_error += abs(actual - summed)
        total_actual += abs(actual)

    if total_actual <= 0.0 or bottom_up_error <= 0.0:
        return None
    return (bottom_up_error - direct_error) / bottom_up_error


def _blend_strength(
    mint_by_horizon: dict[int, np.ndarray],
    bottom_up: np.ndarray,
    hierarchy: Hierarchy,
    validation: pd.DataFrame,
    grid: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Choose how far to trust MinT, on origins it was not estimated from.

    Both projections satisfy ``P S = I``, so every convex combination of them
    does too. That is the property that makes this blend legitimate rather than a
    fudge: any ``lambda`` produces a forecast that is still exactly coherent and
    still unbiased. Only the variance changes.

    So ``lambda`` is chosen empirically on held-out origins. If the estimated
    covariance is good, it goes to one and full MinT applies. If it is not - too
    little data, a structural break, base forecasts that are not what ``W`` was
    measured on - it falls toward zero and the reconciliation quietly declines to
    act. A model that can decide not to fire is worth more than one that is
    always confident.

    TWO CHOICES THAT KEEP THE SELECTION HONEST
    ------------------------------------------
    * **One lambda for every horizon**, not one each. Fitting eight separately
      splits an already-thin validation slice eight ways, and a per-horizon
      optimum measured on a handful of origins is mostly noise - measurably so:
      per-horizon selection picked confident weights that then lost on the test
      fold. Pooling gives the single decision eight times the evidence.

    * **The one-standard-error rule.** Rather than the outright winner, take the
      *smallest* lambda whose error is within one standard error of it. Where the
      curve is flat - which is where the winner is least trustworthy - this
      resolves toward doing less. Intervening has to earn its place.

    Returns the chosen lambda and the diagnostics behind the choice.
    """
    empty: dict[str, float] = {}
    if validation.empty:
        return 0.0, empty

    nodes = list(hierarchy.nodes)
    bottom_mask = np.array([level == "sku" for level in hierarchy.levels])
    aggregate_mask = ~bottom_mask

    # Per-horizon blocks, scored together so one lambda is chosen on all of them.
    blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for horizon, group in validation.groupby("horizon", sort=True):
        mint = mint_by_horizon.get(int(horizon))
        if mint is None:
            continue

        predicted = group.pivot_table(
            index="week_start", columns="node", values="prediction", aggfunc="first"
        ).reindex(columns=nodes)
        truth = group.pivot_table(
            index="week_start", columns="node", values="actual", aggfunc="first"
        ).reindex(columns=nodes)

        # Every aggregate must be present - a missing one would enter as a zero
        # and tell MinT the category is forecast to sell nothing. Missing
        # *bottom* rows are fine and expected: an unlaunched SKU is forecast
        # zero, which is true, and it carries no actual to be scored against.
        usable = predicted.loc[:, aggregate_mask].notna().all(axis=1)
        predicted, truth = predicted.loc[usable], truth.loc[usable]
        if predicted.empty:
            continue

        actual_bottom = truth.to_numpy(dtype="float64")[:, bottom_mask]
        observed = ~np.isnan(actual_bottom)
        if not observed.any():
            continue
        blocks.append(
            (predicted.fillna(0.0).to_numpy(dtype="float64"), actual_bottom, observed, mint)
        )

    if not blocks:
        return 0.0, empty

    denominator = sum(float(np.abs(a[o]).sum()) for _, a, o, _ in blocks)
    if denominator <= 0.0:
        return 0.0, empty

    def errors_at(strength: float) -> np.ndarray:
        """Absolute errors of every scored cell, pooled across horizons."""
        parts: list[np.ndarray] = []
        for base, actual_bottom, observed, mint in blocks:
            projection = (1.0 - strength) * bottom_up + strength * mint
            adjusted = np.clip(base @ projection.T, 0.0, None)
            parts.append(np.abs(adjusted[observed] - actual_bottom[observed]))
        return np.concatenate(parts)

    baseline_errors = errors_at(0.0)
    base_wape = float(baseline_errors.sum() / denominator)

    # --- The structural gate ------------------------------------------------ #
    # MinT can only add something if an independently-forecast aggregate beats
    # the free alternative of summing the SKU forecasts. That is not implied by
    # the aggregate having a lower WAPE: summing k roughly independent forecasts
    # shrinks relative error by about sqrt(k) on its own, so an aggregate can
    # look far more accurate while carrying no information the bottom level did
    # not already have.
    #
    # This is tested directly rather than inferred from the blended WAPE,
    # because it is scale-free and so transfers between models. The blended WAPE
    # does not: the residual-harvest model is weaker than the shipped one, MinT
    # helps a weak base forecast more, and the validation slice therefore
    # endorses a strength that the shipped model then loses on. That failure was
    # observed on this engagement, which is why the gate exists.
    advantage = _aggregate_advantage(validation, hierarchy)
    if advantage is not None and advantage <= 0.0:
        return 0.0, {
            "base_wape": round(base_wape, 6),
            "chosen_wape": round(base_wape, 6),
            "aggregate_advantage": round(advantage, 6),
            "gate": "declined: aggregate forecasts do not beat their bottom-up sum",
        }

    scores: dict[float, float] = {0.0: base_wape}
    for candidate in grid:
        scores[float(candidate)] = float(errors_at(float(candidate)).sum() / denominator)

    best_lambda = min(scores, key=lambda key: scores[key])
    best_wape = scores[best_lambda]
    if best_lambda == 0.0:
        return 0.0, {"base_wape": round(base_wape, 6), "chosen_wape": round(base_wape, 6)}

    # Standard error of the mean absolute error at the winning lambda, carried
    # onto the WAPE scale. This is what "within one standard error" is measured
    # against, and it is what stops a fractional win from being taken seriously.
    winning_errors = errors_at(best_lambda)
    standard_error = float(
        np.std(winning_errors - baseline_errors, ddof=1) / np.sqrt(winning_errors.size)
    )
    tolerance = best_wape + standard_error * winning_errors.size / denominator

    chosen = min(
        (value for value, score in scores.items() if score <= tolerance),
        default=best_lambda,
    )
    return chosen, {
        "base_wape": round(base_wape, 6),
        "best_wape": round(best_wape, 6),
        "best_lambda": round(best_lambda, 4),
        "chosen_wape": round(scores[chosen], 6),
        "one_se_tolerance": round(tolerance, 6),
        "aggregate_advantage": round(advantage, 6) if advantage is not None else None,
        "gate": "passed",
    }


# --------------------------------------------------------------------------- #
# Residual harvesting
# --------------------------------------------------------------------------- #
def harvest_residuals(
    supervised: pd.DataFrame,
    cutoff: pd.Timestamp,
    settings: Settings | None = None,
    *,
    feature_spec: FeatureSpec = FEATURE_SPEC,
    params: dict[str, Any] | None = None,
    predictor: Callable[[pd.DataFrame], _SupportsPredict] | None = None,
) -> pd.DataFrame:
    """Produce genuinely out-of-sample base-forecast residuals.

    One model is fitted on everything whose target had been observed by
    ``cutoff``, then used to forecast every origin from ``cutoff`` onward. Each
    of those origins is beyond the training data, so every residual is
    out-of-sample - which is the condition MinT's optimality argument rests on.
    Fitting the model once rather than per origin is what makes a dense residual
    window affordable.

    Args:
        supervised: Supervised rows for one level of the hierarchy.
        cutoff: Training boundary. Rows with ``target_week <= cutoff`` train.
        settings: Configuration.
        feature_spec: Feature contract to train against.
        params: LightGBM overrides.
        predictor: Builds the model to harvest with, given the training rows.
            This matters more than it looks: ``W`` must describe the errors of
            the forecasts actually being reconciled. Harvesting with a plain GBM
            and then reconciling ensemble forecasts optimises the weights for the
            wrong error structure, and measurably makes the result worse.
            Defaults to the GBM.

    Returns:
        Columns ``node``, ``week_start``, ``horizon``, ``target_week``,
        ``actual``, ``prediction``, ``residual``.
    """
    settings = settings or get_settings()

    train = supervised[supervised["target_week"] <= cutoff]
    score = supervised[supervised["week_start"] >= cutoff]
    if train.empty:
        raise InsufficientHistoryError(
            f"No rows with target_week <= {pd.Timestamp(cutoff).date()}; cannot fit "
            "a residual-harvest model."
        )
    if score.empty:
        raise InsufficientHistoryError(
            f"No origins at or after {pd.Timestamp(cutoff).date()} to harvest residuals from."
        )

    if predictor is None:
        model: _SupportsPredict = train_forecaster(
            train, settings, params=params, feature_spec=feature_spec
        )
    else:
        model = predictor(train)
    predictions = model.predict(score)

    residuals = score.loc[:, ["sku_id", "week_start", "horizon", "target_week", "y"]].copy()
    residuals = residuals.rename(columns={"sku_id": "node", "y": "actual"})
    residuals["prediction"] = predictions["prediction"].to_numpy()
    residuals["residual"] = residuals["actual"] - residuals["prediction"]

    log.info(
        "harvested residuals",
        extra={
            "context": {
                "cutoff": str(pd.Timestamp(cutoff).date()),
                "train_rows": len(train),
                "residual_rows": len(residuals),
                "nodes": int(residuals["node"].nunique()),
                "origins": int(residuals["week_start"].nunique()),
            }
        },
    )
    return residuals.reset_index(drop=True)


def _residual_matrices(
    residuals: pd.DataFrame,
    hierarchy: Hierarchy,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Pivot long residuals into one ``(origins, nodes)`` matrix per horizon.

    The panel is ragged by construction - a product launched last quarter has no
    residual at an origin before it existed - so demanding a complete rectangle
    would discard nearly every origin. Two rules keep that honest:

    * **Scale** is estimated from each node's *own* observations. A node with too
      few gets the median scale of the nodes at its level, inflated: unknown
      reliability must read as low reliability, or MinT would hand a node it
      knows nothing about the weight of a node it trusts.
    * **Correlation** is estimated after zero-filling the gaps. A missing
      residual then contributes no covariance rather than a fabricated one, so
      sparse nodes are pulled toward independence - the conservative direction,
      and the one that leaves their forecasts closest to unadjusted.

    Returns the matrices, the per-horizon per-node scales, and the observation
    counts behind each scale.
    """
    nodes = list(hierarchy.nodes)
    level_of = np.array(hierarchy.levels)

    matrices: dict[int, np.ndarray] = {}
    scales: dict[int, np.ndarray] = {}
    counts: dict[int, np.ndarray] = {}

    for horizon, group in residuals.groupby("horizon", sort=True):
        wide = group.pivot_table(
            index="week_start", columns="node", values="residual", aggfunc="first"
        ).reindex(columns=nodes)
        wide = wide.dropna(axis=0, how="all")
        if wide.empty:
            continue

        observed = wide.notna().sum(axis=0).to_numpy(dtype="float64")
        raw_scale = wide.std(axis=0, ddof=1).to_numpy(dtype="float64")

        # Median scale among adequately-observed nodes at the same level. Levels
        # differ by orders of magnitude in volume, so a single global fallback
        # would badly misprice either the total or the slowest SKU.
        reliable = observed >= _MIN_NODE_OBSERVATIONS
        fallback = np.empty_like(raw_scale)
        for level in np.unique(level_of):
            at_level = level_of == level
            pool = raw_scale[at_level & reliable & np.isfinite(raw_scale)]
            fallback[at_level] = np.median(pool) if pool.size else _MIN_SCALE

        scale = np.where(
            reliable & np.isfinite(raw_scale), raw_scale, fallback * _UNKNOWN_INFLATION
        )
        scale = np.where(scale > _MIN_SCALE, scale, _MIN_SCALE)

        matrices[int(horizon)] = wide.to_numpy(dtype="float64")
        scales[int(horizon)] = scale
        counts[int(horizon)] = observed

    if not matrices:
        raise InsufficientHistoryError(
            "No horizon produced any residuals; cannot estimate the error covariance."
        )
    return matrices, scales, counts


def _pooled_correlation(
    matrices: dict[int, np.ndarray],
    scales: dict[int, np.ndarray],
    counts: dict[int, np.ndarray],
    n_nodes: int,
) -> tuple[np.ndarray, float, int]:
    """One correlation matrix, estimated only where there is evidence for it.

    Nodes with thin residual coverage are given an identity row rather than an
    estimated one. That is not a shortcut - filling their gaps with zeros, the
    obvious alternative, is actively harmful here: unlaunched SKUs are missing at
    *the same* origins, so zero-filling makes them all move together and
    manufactures a strong, entirely fictitious correlation. The shrinkage
    estimator then reads that fiction as signal and shrinks less, which is the
    opposite of what thin data should produce.

    Returns the full correlation, the shrinkage intensity, and how many nodes
    were actually estimated.
    """
    reliable = np.ones(n_nodes, dtype=bool)
    for observed in counts.values():
        reliable &= observed >= _MIN_NODE_OBSERVATIONS

    correlation = np.eye(n_nodes, dtype="float64")
    if reliable.sum() < 2:
        return correlation, 1.0, int(reliable.sum())

    pooled: list[np.ndarray] = []
    for horizon, matrix in matrices.items():
        subset = matrix[:, reliable] / scales[horizon][reliable]
        # Only origins where every reliable node reported. Among reliable nodes
        # these are the large majority, so little is lost and nothing is invented.
        complete = ~np.isnan(subset).any(axis=1)
        if complete.any():
            pooled.append(subset[complete])

    if not pooled:
        return correlation, 1.0, int(reliable.sum())

    stacked = np.vstack(pooled)
    if stacked.shape[0] < _MIN_OBSERVATIONS:
        return correlation, 1.0, int(reliable.sum())

    estimated, intensity = shrunk_correlation(stacked)
    block = np.ix_(reliable, reliable)
    correlation[block] = estimated
    return correlation, intensity, int(reliable.sum())


# --------------------------------------------------------------------------- #
# The fitted model
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class HierarchicalReconciler:
    """A fitted MinT reconciler plus the aggregate-level forecaster it needs.

    The object is self-contained: it carries the hierarchy it was fitted for, the
    model that forecasts the aggregate nodes, and one projection matrix per
    horizon. Handing it bottom-level forecasts is enough to get reconciled ones
    back, so train and serve cannot disagree about the structure.
    """

    hierarchy: Hierarchy
    aggregate_forecaster: TrainedForecaster
    projections: dict[int, np.ndarray]
    scales: dict[int, np.ndarray]
    shrinkage: float
    blend: dict[int, float]
    trained_at: str
    residual_observations: dict[int, int]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def horizons(self) -> tuple[int, ...]:
        return tuple(sorted(self.projections))

    def predict_aggregates(self, aggregate_frame: pd.DataFrame) -> pd.DataFrame:
        """Base forecasts for every aggregate node in ``aggregate_frame``."""
        if aggregate_frame.empty:
            raise InsufficientHistoryError("No aggregate rows supplied to predict_aggregates.")
        predictions = self.aggregate_forecaster.predict(aggregate_frame)
        out = aggregate_frame.loc[:, ["sku_id", "week_start", "horizon", "target_week"]].copy()
        out = out.rename(columns={"sku_id": "node"})
        out["prediction"] = predictions["prediction"].to_numpy()
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def reconcile(
        self,
        bottom: pd.DataFrame,
        aggregates: pd.DataFrame,
        *,
        prediction_column: str = "prediction",
    ) -> pd.DataFrame:
        """Combine base forecasts from every level into one coherent set.

        Args:
            bottom: SKU-level forecasts. Needs ``sku_id``, ``week_start``,
                ``horizon`` and ``prediction_column``. Whatever produced these -
                the GBM, the adaptive ensemble, anything - is irrelevant here;
                the reconciliation is a post-process, which is why the two custom
                models compose rather than compete.
            aggregates: Node-level forecasts from
                :meth:`predict_aggregates`.
            prediction_column: Name of the column holding the point forecast.

        Returns:
            ``bottom`` with a ``reconciled`` column added, plus ``adjustment``
            (reconciled minus base) so a planner can see and audit what the
            reconciliation changed and by how much.

        Raises:
            ModelNotFittedError: if a horizon in ``bottom`` has no projection.
            InsufficientHistoryError: if any SKU is missing at some origin.
        """
        if bottom.empty:
            raise InsufficientHistoryError("No bottom-level forecasts supplied to reconcile.")
        for column in ("sku_id", "week_start", "horizon", prediction_column):
            if column not in bottom.columns:
                raise ValueError(f"bottom forecasts are missing required column {column!r}")

        node_position = self.hierarchy.node_index
        bottom_nodes = self.hierarchy.bottom
        # Position of each SKU within the bottom vector `b`, and the row it
        # occupies in the full node vector. Both are built once.
        bottom_slot = {node: index for index, node in enumerate(bottom_nodes)}
        bottom_rows = np.array([node_position[node] for node in bottom_nodes])
        n_bottom = len(bottom_nodes)

        aggregate_lookup: dict[tuple[pd.Timestamp, int], dict[str, float]] = {}
        if not aggregates.empty:
            for row in aggregates.itertuples(index=False):
                key = (pd.Timestamp(row.week_start), int(row.horizon))
                aggregate_lookup.setdefault(key, {})[str(row.node)] = float(row.prediction)

        # Reset the index so a row's label is its position; the reconciled values
        # are scattered back by position, and a caller's non-unique index would
        # otherwise silently misalign them.
        working = bottom.reset_index(drop=True).copy()
        working["week_start"] = pd.to_datetime(working["week_start"])
        working["horizon"] = working["horizon"].astype("int64")
        reconciled = np.full(len(working), np.nan, dtype="float64")

        summing = self.hierarchy.summing

        for (origin, horizon), group in working.groupby(["week_start", "horizon"], sort=False):
            horizon = int(horizon)
            projection = self.projections.get(horizon)
            if projection is None:
                raise ModelNotFittedError(
                    f"No reconciliation projection for horizon {horizon}. The "
                    f"reconciler was fitted for horizons {list(self.horizons)}."
                )

            row_positions = group.index.to_numpy()
            slots = np.array(
                [bottom_slot.get(sku, -1) for sku in group["sku_id"].astype(str)], dtype="int64"
            )
            unknown = int((slots < 0).sum())
            if unknown:
                raise InsufficientHistoryError(
                    f"Origin {pd.Timestamp(origin).date()} horizon {horizon} carries "
                    f"{unknown} forecast(s) for SKUs absent from the hierarchy. Rebuild "
                    "the hierarchy from the current SKU master before reconciling."
                )
            present = np.unique(slots)
            if present.size != slots.size:
                raise InsufficientHistoryError(
                    f"Origin {pd.Timestamp(origin).date()} horizon {horizon} carries "
                    "more than one forecast for the same SKU; reconciliation needs "
                    "each SKU exactly once."
                )
            coverage = present.size / n_bottom
            if coverage < _MIN_BOTTOM_COVERAGE:
                raise InsufficientHistoryError(
                    f"Origin {pd.Timestamp(origin).date()} horizon {horizon} covers only "
                    f"{present.size} of {n_bottom} SKUs ({coverage:.0%}). That is too "
                    "little of the assortment for the aggregate levels to mean anything; "
                    "the forecast frame and the hierarchy are probably out of step."
                )

            # SKUs absent from this origin have not launched yet, so their demand
            # is genuinely zero rather than unknown - the aggregates they roll
            # into are correct with those zeros included. They are forced back to
            # zero after reconciliation so MinT cannot put stock against a product
            # that does not exist.
            base_bottom = np.zeros(n_bottom, dtype="float64")
            base_bottom[slots] = group[prediction_column].to_numpy(dtype="float64")
            absent = np.ones(n_bottom, dtype=bool)
            absent[slots] = False

            # Assemble the full base vector. Any aggregate the model did not
            # forecast falls back to the bottom-up sum, which contributes no new
            # information but keeps the system well-posed.
            base = summing @ base_bottom
            for node, value in aggregate_lookup.get((pd.Timestamp(origin), horizon), {}).items():
                index = node_position.get(node)
                if index is not None and self.hierarchy.levels[index] != "sku":
                    base[index] = value
            base[bottom_rows] = base_bottom

            adjusted = np.clip(projection @ base, 0.0, None)
            adjusted[absent] = 0.0
            reconciled[row_positions] = adjusted[slots]

        if np.isnan(reconciled).any():
            raise ModelNotFittedError(
                f"{int(np.isnan(reconciled).sum())} rows were left unreconciled; "
                "the grouping did not cover every input row."
            )

        working["reconciled"] = reconciled
        working["adjustment"] = working["reconciled"] - working[prediction_column]

        log.info(
            "reconciled forecasts",
            extra={
                "context": {
                    "rows": len(working),
                    "origins": int(working["week_start"].nunique()),
                    "mean_abs_adjustment": round(float(working["adjustment"].abs().mean()), 4),
                    "nodes": self.hierarchy.n_nodes,
                }
            },
        )
        return working

    # ------------------------------------------------------------------ #
    def coherence_check(self, reconciled: pd.DataFrame) -> float:
        """Worst aggregate-versus-children mismatch across all origins.

        Should be at floating-point noise. Anything larger means the output is
        not a coherent plan and must not ship.
        """
        bottom_slot = {node: index for index, node in enumerate(self.hierarchy.bottom)}
        n_bottom = len(bottom_slot)
        worst = 0.0
        for _, group in reconciled.groupby(["week_start", "horizon"], sort=False):
            slots = np.array(
                [bottom_slot.get(sku, -1) for sku in group["sku_id"].astype(str)], dtype="int64"
            )
            if (slots < 0).any():
                continue
            # Unlaunched SKUs stay zero, exactly as reconcile() left them.
            bottom_values = np.zeros(n_bottom, dtype="float64")
            bottom_values[slots] = group["reconciled"].to_numpy(dtype="float64")
            worst = max(
                worst, self.hierarchy.coherence_error(self.hierarchy.summing @ bottom_values)
            )
        return worst

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("saved reconciler", extra={"context": {"path": str(path)}})


def load_reconciler(path: Path) -> HierarchicalReconciler:
    """Load a persisted reconciler.

    Raises:
        ArtifactNotFoundError: if the file is absent.
    """
    if not path.exists():
        raise ArtifactNotFoundError(
            f"No reconciler at {path}. Run scripts/08_train_reconciler.py first."
        )
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_reconciler(
    bottom_supervised: pd.DataFrame,
    aggregate_supervised: pd.DataFrame,
    hierarchy: Hierarchy,
    settings: Settings | None = None,
    *,
    feature_spec: FeatureSpec = FEATURE_SPEC,
    residual_fraction: float = 0.3,
    params: dict[str, Any] | None = None,
    bottom_predictor: Callable[[pd.DataFrame], _SupportsPredict] | None = None,
) -> HierarchicalReconciler:
    """Fit the aggregate forecaster and the MinT projection matrices.

    Args:
        bottom_supervised: Supervised rows for the SKU level.
        aggregate_supervised: Supervised rows for every node above it, built from
            :func:`foresight.hierarchy.aggregate_panel`.
        hierarchy: The structure being reconciled.
        settings: Configuration.
        feature_spec: Feature contract for both levels' models.
        residual_fraction: Share of the timeline reserved for out-of-sample
            residuals. Larger gives a better-estimated covariance but leaves the
            harvest model less history, so its residuals stop resembling the
            shipped model's.
        params: LightGBM overrides.
        bottom_predictor: Builds the SKU-level model whose errors define ``W``.
            Pass the same kind of model whose forecasts will be reconciled - if
            the adaptive ensemble is what ships, harvest with the ensemble.

    Returns:
        A fitted :class:`HierarchicalReconciler`.
    """
    settings = settings or get_settings()
    if not 0.0 < residual_fraction < 1.0:
        raise ValueError(f"residual_fraction must be in (0, 1), received {residual_fraction}")

    weeks = np.sort(bottom_supervised["week_start"].unique())
    if len(weeks) < _MIN_OBSERVATIONS:
        raise InsufficientHistoryError(
            f"Only {len(weeks)} origin weeks available; the error covariance needs "
            f"at least {_MIN_OBSERVATIONS}."
        )
    cutoff = pd.Timestamp(weeks[int(len(weeks) * (1.0 - residual_fraction))])

    log.info(
        "training hierarchical reconciler",
        extra={
            "context": {
                "cutoff": str(cutoff.date()),
                "nodes": hierarchy.n_nodes,
                "bottom_rows": len(bottom_supervised),
                "aggregate_rows": len(aggregate_supervised),
            }
        },
    )

    # --- Residuals from both levels, on identical origins ------------------- #
    bottom_residuals = harvest_residuals(
        bottom_supervised,
        cutoff,
        settings,
        feature_spec=feature_spec,
        params=params,
        predictor=bottom_predictor,
    )
    aggregate_residuals = harvest_residuals(
        aggregate_supervised, cutoff, settings, feature_spec=feature_spec, params=params
    )
    residuals = pd.concat([bottom_residuals, aggregate_residuals], ignore_index=True)

    # --- Split the residual window: estimate on the front, validate on the tail #
    origins = np.sort(residuals["week_start"].unique())
    split_at = pd.Timestamp(origins[int(len(origins) * (1.0 - _VALIDATION_FRACTION))])
    estimation = residuals[residuals["week_start"] < split_at]
    validation = residuals[residuals["week_start"] >= split_at]
    if estimation.empty:
        estimation, validation = residuals, residuals.iloc[0:0]

    matrices, scales, counts = _residual_matrices(estimation, hierarchy)
    correlation, shrinkage, estimated_nodes = _pooled_correlation(
        matrices, scales, counts, hierarchy.n_nodes
    )

    # --- Projections, blended back toward bottom-up by a validated amount ---- #
    bottom_up = _bottom_up_projection(hierarchy)
    grid = np.asarray(_LAMBDA_GRID, dtype="float64")

    mint_by_horizon = {
        horizon: _mint_projection(hierarchy.summing, correlation, scale)
        for horizon, scale in scales.items()
    }
    strength, validation_scores = _blend_strength(
        mint_by_horizon, bottom_up, hierarchy, validation, grid
    )

    projections = {
        horizon: (1.0 - strength) * bottom_up + strength * mint
        for horizon, mint in mint_by_horizon.items()
    }
    blend = dict.fromkeys(mint_by_horizon, strength)

    # --- The shipped aggregate model: refit on everything ------------------- #
    aggregate_forecaster = train_forecaster(
        aggregate_supervised, settings, params=params, feature_spec=feature_spec
    )

    # --- Diagnostics the model documents quote ------------------------------ #
    level_of = dict(zip(hierarchy.nodes, hierarchy.levels, strict=True))
    residuals["level"] = residuals["node"].map(level_of)
    by_level = (
        residuals.assign(
            abs_error=residuals["residual"].abs(), abs_actual=residuals["actual"].abs()
        )
        .groupby("level", sort=False)[["abs_error", "abs_actual"]]
        .sum()
    )
    base_wape = (by_level["abs_error"] / by_level["abs_actual"].replace(0.0, np.nan)).to_dict()

    reconciler = HierarchicalReconciler(
        hierarchy=hierarchy,
        aggregate_forecaster=aggregate_forecaster,
        projections=projections,
        scales=scales,
        shrinkage=shrinkage,
        blend=blend,
        trained_at=dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        residual_observations={horizon: int(m.shape[0]) for horizon, m in matrices.items()},
        diagnostics={
            "residual_cutoff": str(cutoff.date()),
            "validation_split": str(split_at.date()),
            "shrinkage_intensity": round(shrinkage, 6),
            "nodes_in_correlation": estimated_nodes,
            "blend_lambda": round(strength, 4),
            "mean_blend": round(strength, 4),
            "validation_scores": validation_scores,
            "base_wape_by_level": {
                level: round(float(value), 6)
                for level, value in base_wape.items()
                if np.isfinite(value)
            },
            "n_nodes": hierarchy.n_nodes,
            "n_bottom": hierarchy.n_bottom,
            "nodes_with_thin_residuals": {
                str(horizon): int((observed < _MIN_NODE_OBSERVATIONS).sum())
                for horizon, observed in counts.items()
            },
        },
    )

    log.info(
        "reconciler fitted",
        extra={
            "context": {
                "shrinkage": round(shrinkage, 4),
                "mean_blend": reconciler.diagnostics["mean_blend"],
                "nodes_in_correlation": estimated_nodes,
                "horizons": list(reconciler.horizons),
                "base_wape_by_level": {
                    level: round(float(value), 4)
                    for level, value in base_wape.items()
                    if np.isfinite(value)
                },
            }
        },
    )
    return reconciler
