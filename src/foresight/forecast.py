"""Demand forecasting model (deliverable D3).

A single global LightGBM model across all SKUs, trained in **direct
multi-horizon** form: the horizon is an input feature, and each training row is
one (origin week, horizon) pair. That was chosen over the alternatives for
reasons that matter here:

* **Global, not per-SKU.** 200 series of ~190 weeks each are far too short to fit
  200 independent models. Pooling lets a SKU with thin history borrow the
  seasonal shape of its category, which is precisely the mitigation the brief
  asks for in section 16.2.
* **Direct, not recursive.** A recursive model feeds its own predictions back in
  and compounds error across the horizon. Direct forecasting predicts week
  ``t + h`` in one shot, so nothing is ever conditioned on a guess.
* **L1 objective.** WAPE is the metric this engagement is judged on, and WAPE is
  an absolute-error criterion. Training on squared error would optimise
  something the client never sees, and would drag forecasts upward on the spiky
  festive weeks that dominate this assortment.

Uncertainty comes from two extra models fitted at the 10th and 90th percentile,
giving the 80% interval the client needs to judge how much to trust a number.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from foresight.config import Settings, get_settings
from foresight.exceptions import (
    ArtifactNotFoundError,
    InsufficientHistoryError,
    ModelNotFittedError,
)
from foresight.features import FEATURE_SPEC, FeatureSpec
from foresight.logging_setup import get_logger

__all__ = ["TrainedForecaster", "DEFAULT_PARAMS", "load_forecaster", "train_forecaster"]

log = get_logger(__name__)

MODEL_FILENAME: Final[str] = "forecast_model.joblib"

#: Conservative, deliberately un-tuned defaults. Depth and leaf count are kept
#: modest because the panel is wide but short - ~280k rows across only 200
#: series - and an unconstrained tree will memorise individual SKU-weeks.
DEFAULT_PARAMS: Final[dict[str, Any]] = {
    "objective": "l1",
    "n_estimators": 700,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}


@dataclass(slots=True)
class TrainedForecaster:
    """A fitted point model plus its two interval models.

    Carries the feature contract and the exact categorical levels seen during
    training, so serving cannot silently drift away from training.
    """

    point_model: lgb.LGBMRegressor
    lower_model: lgb.LGBMRegressor
    upper_model: lgb.LGBMRegressor
    feature_spec: FeatureSpec
    category_levels: dict[str, list[str]]
    quantiles: tuple[float, float]
    params: dict[str, Any]
    trained_at: str
    training_rows: int
    horizons: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------- #
    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Select and type the feature columns exactly as at training time."""
        missing = [
            column for column in self.feature_spec.all_features if column not in frame.columns
        ]
        if missing:
            raise ModelNotFittedError(
                f"Input is missing required feature column(s): {', '.join(missing)}"
            )

        prepared = frame.loc[:, list(self.feature_spec.all_features)].copy()
        for column in self.feature_spec.categorical:
            # Fixed categories from training. An unseen level becomes NaN, which
            # LightGBM handles, rather than silently shifting every category code
            # and corrupting every prediction for that column.
            prepared[column] = pd.Categorical(
                prepared[column].astype("object"),
                categories=self.category_levels[column],
            )
        for column in self.feature_spec.numeric:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce").astype("float64")
        return prepared

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict point demand and the prediction interval.

        Returns:
            A frame with ``prediction``, ``prediction_lower`` and
            ``prediction_upper``, aligned to ``frame``'s index.

        Demand cannot be negative, so all three are clipped at zero. The three
        are then sorted per row: the quantile models are fitted independently and
        can cross on hard rows, which would otherwise emit an interval whose
        lower bound sits above its upper bound.
        """
        prepared = self._prepare(frame)

        point = np.clip(self.point_model.predict(prepared), 0.0, None)
        lower = np.clip(self.lower_model.predict(prepared), 0.0, None)
        upper = np.clip(self.upper_model.predict(prepared), 0.0, None)

        stacked = np.sort(np.vstack([lower, point, upper]), axis=0)

        return pd.DataFrame(
            {
                "prediction": point,
                "prediction_lower": stacked[0],
                "prediction_upper": stacked[2],
            },
            index=frame.index,
        )

    # ----------------------------------------------------------------- #
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("saved forecaster", extra={"context": {"path": str(path)}})

    @property
    def feature_importance(self) -> pd.DataFrame:
        """Gain-based importance of each feature in the point model."""
        return (
            pd.DataFrame(
                {
                    "feature": self.point_model.feature_name_,
                    "gain": self.point_model.booster_.feature_importance(importance_type="gain"),
                    "split": self.point_model.booster_.feature_importance(importance_type="split"),
                }
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )


def _fit_one(
    features: pd.DataFrame,
    target: np.ndarray,
    params: dict[str, Any],
    seed: int,
    categorical: tuple[str, ...],
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(random_state=seed, **params)
    model.fit(features, target, categorical_feature=list(categorical))
    return model


def train_forecaster(
    supervised: pd.DataFrame,
    settings: Settings | None = None,
    *,
    params: dict[str, Any] | None = None,
    feature_spec: FeatureSpec = FEATURE_SPEC,
) -> TrainedForecaster:
    """Fit the point model and both interval models on a supervised frame.

    Args:
        supervised: Rows produced by
            :func:`foresight.features.build_supervised_frame`, already filtered
            to the training window by the caller. This function does no time
            filtering of its own - the backtest owns that, so there is exactly
            one place where the train/test split can be got wrong.
        settings: Configuration.
        params: LightGBM overrides.
        feature_spec: The feature contract to train against.
    """
    settings = settings or get_settings()
    if supervised.empty:
        raise InsufficientHistoryError("No training rows supplied to train_forecaster.")

    resolved_params = {**DEFAULT_PARAMS, **(params or {})}
    lower_quantile, upper_quantile = settings.quantile_levels

    category_levels = {
        column: sorted(supervised[column].astype("object").dropna().unique().tolist())
        for column in feature_spec.categorical
    }

    features = supervised.loc[:, list(feature_spec.all_features)].copy()
    for column in feature_spec.categorical:
        features[column] = pd.Categorical(
            features[column].astype("object"), categories=category_levels[column]
        )
    for column in feature_spec.numeric:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype("float64")

    target = supervised["y"].to_numpy(dtype="float64")

    log.info(
        "training forecaster",
        extra={
            "context": {
                "rows": len(features),
                "features": len(feature_spec.all_features),
                "objective": resolved_params["objective"],
            }
        },
    )

    categorical = feature_spec.categorical
    point_model = _fit_one(features, target, resolved_params, settings.random_seed, categorical)

    quantile_params = {**resolved_params, "objective": "quantile"}
    lower_model = _fit_one(
        features,
        target,
        {**quantile_params, "alpha": lower_quantile},
        settings.random_seed,
        categorical,
    )
    upper_model = _fit_one(
        features,
        target,
        {**quantile_params, "alpha": upper_quantile},
        settings.random_seed,
        categorical,
    )

    horizons = tuple(sorted(int(value) for value in supervised["horizon"].unique()))

    forecaster = TrainedForecaster(
        point_model=point_model,
        lower_model=lower_model,
        upper_model=upper_model,
        feature_spec=feature_spec,
        category_levels=category_levels,
        quantiles=(lower_quantile, upper_quantile),
        params=resolved_params,
        trained_at=dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        training_rows=len(features),
        horizons=horizons,
        metadata={
            "interval_coverage": settings.interval_coverage,
            "seed": settings.random_seed,
            "train_target_week_max": str(pd.Timestamp(supervised["target_week"].max()).date())
            if "target_week" in supervised.columns
            else None,
        },
    )
    log.info(
        "forecaster trained",
        extra={"context": {"rows": forecaster.training_rows, "horizons": list(horizons)}},
    )
    return forecaster


def load_forecaster(
    path: Path | None = None, settings: Settings | None = None
) -> TrainedForecaster:
    """Load a saved forecaster.

    Raises:
        ArtifactNotFoundError: if no model has been trained yet.
    """
    settings = settings or get_settings()
    resolved = path or (settings.artifacts_dir / MODEL_FILENAME)
    if not resolved.exists():
        raise ArtifactNotFoundError(
            resolved, "Train one first with `python scripts/03_train_backtest.py`."
        )
    forecaster = joblib.load(resolved)
    if not isinstance(forecaster, TrainedForecaster):
        raise ArtifactNotFoundError(resolved, "File exists but is not a TrainedForecaster.")
    return forecaster
