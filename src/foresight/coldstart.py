"""FORESIGHT Cold-Start Propagation - the third custom model.

THE FAILURE THIS EXISTS FOR
---------------------------
Section 16.2 of the brief names it directly: a SKU launched inside the forecast
window has no history, and every technique in the rest of this project is built
on history. Lags are missing. Rolling means are undefined. TSB has no series to
smooth. The seasonal profile has no level to scale. A gradient-boosted model
handed a row of nulls does not abstain - it predicts the average of whatever
rows looked vaguely similar during training, which is how new products get
ordered in quantities nobody can defend.

This is also the regime the backtest scores worst on, so it is not a hypothetical
concern.

THE IDEA
--------
A new product has no history, but it is not information-free. We know what it
*is*: its category, its subcategory, its list price, its margin, where it sits in
the price ladder of the products beside it. And we have watched dozens of similar
products launch already. Their trajectories are the evidence.

So the forecast is assembled from three separately-estimated pieces::

    forecast = scale  x  launch curve(age)  x  seasonal index(week, category)
               ^^^^^     ^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
               how much  what shape a launch   when in the year the target
               it sells  of this kind takes    week falls

Separating them matters. Scale is a property of the product, shape is a property
of launches in its category, seasonality is a property of the calendar. Estimated
jointly they would be confounded - a product that launched into December would
have its seasonal lift baked into its "scale" forever.

CREDIBILITY, NOT A CUTOFF
-------------------------
The obvious design is a switch: use donors below thirteen weeks of history, use
the SKU's own data above. That puts a cliff in the middle of the plan - a product
can jump by a third on the week it crosses the threshold, for no reason a planner
can explain to a buyer.

Instead the two estimates are blended by a credibility weight taken straight from
actuarial practice (Buhlmann-Straub)::

    Z = n / (n + k)

where ``n`` is the weeks actually observed for this SKU and ``k`` is how many
weeks of its own evidence are worth as much as the donor pool. At launch ``Z``
is zero and the forecast is entirely borrowed. Every week that passes moves
weight onto the product's own record, and by the time it has a year of history
the donors contribute nothing measurable. The model dissolves into the standard
pipeline instead of handing over abruptly, and there is no threshold to tune or
to argue about.

ONE CORRECTION THAT IS EASY TO GET WRONG
----------------------------------------
A young SKU's observed mean is not its mature level - it is a sample from the
early, steep part of the ramp. Blending it directly against donors' *mature*
levels would systematically under-forecast every new product. So the observed
mean is first divided by the average curve value over exactly the ages that were
observed, which converts it to a mature-equivalent scale before the two
estimates ever meet.

LEAKAGE
-------
Same contract as the rest of the project. The launch library, the price
elasticity and the seasonal index are all fitted from the training window alone,
and the per-SKU observed level at origin ``t`` uses only weeks up to and
including ``t``. The target week contributes its calendar position and nothing
else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from foresight.exceptions import InsufficientHistoryError
from foresight.logging_setup import get_logger

__all__ = [
    "ColdStartModel",
    "LaunchLibrary",
    "fit_cold_start",
]

log = get_logger(__name__)

#: Ages (weeks since launch) the launch curve is estimated over. Beyond this a
#: product is treated as mature and the curve is flat at 1.0.
MAX_CURVE_AGE: Final[int] = 26

#: The age window defining a product's "mature level". Starts after the launch
#: ramp has flattened and ends before a full year has passed, so the window never
#: spans two of the same season.
MATURE_AGE_LO: Final[int] = 13
MATURE_AGE_HI: Final[int] = 26

#: A donor must have been observed through the whole mature window, otherwise its
#: normalising level is itself a guess.
_MIN_DONOR_AGE: Final[int] = MATURE_AGE_HI

#: Weeks of a SKU's own history worth as much as the entire donor pool. Eight is
#: roughly two months - long enough to see through a launch promotion, short
#: enough that a genuinely mispriced borrow is corrected within a quarter.
CREDIBILITY_WEEKS: Final[float] = 8.0

#: Donors pooled per target SKU. Too few and one odd launch dominates; too many
#: and the pool stops resembling the target.
DEFAULT_DONORS: Final[int] = 12

#: Bounds on the fitted price elasticity of mature volume. Demand falling with
#: price is the only sign economics allows, and magnitudes outside this range are
#: an artefact of a thin category rather than a real relationship.
_ELASTICITY_BOUNDS: Final[tuple[float, float]] = (-3.0, 0.0)

#: Age 0 is a partial week - a product launched on a Thursday sells for two days.
#: Including it would bias every curve downward at exactly the point planners
#: care most about.
_FIRST_FULL_AGE: Final[int] = 1

_EPSILON: Final[float] = 1e-6

#: Attributes describing what a product *is*, independent of how it has sold.
_ATTRIBUTES: Final[tuple[str, ...]] = ("log_list_price", "margin_rate", "price_rank")

#: Distance added when a donor sits in a different subcategory / category. The
#: category penalty is large enough to make cross-category borrowing a last
#: resort rather than a routine outcome.
_SUBCATEGORY_PENALTY: Final[float] = 1.0
_CATEGORY_PENALTY: Final[float] = 4.0


# --------------------------------------------------------------------------- #
# Launch library
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LaunchLibrary:
    """Normalised launch trajectories, pooled overall and per category.

    Attributes:
        overall: Curve value by age, index ``0 .. MAX_CURVE_AGE``. A value of
            0.6 at age 3 means a typical product sells 60% of its eventual
            mature weekly rate three weeks after launch.
        by_category: Per-category curves, used when a category had enough
            launches to estimate its own shape.
        donors: How many launches each category's curve was built from.
    """

    overall: np.ndarray
    by_category: dict[str, np.ndarray]
    donors: dict[str, int]

    def curve_for(self, category: str) -> np.ndarray:
        return self.by_category.get(category, self.overall)

    def value_at(self, category: str, age: np.ndarray) -> np.ndarray:
        """Curve value at each age, flat at 1.0 once a product is mature."""
        curve = self.curve_for(category)
        clipped = np.clip(np.asarray(age, dtype="int64"), 0, MAX_CURVE_AGE)
        values = curve[clipped]
        return np.where(np.asarray(age) > MAX_CURVE_AGE, 1.0, values)


def _mature_level(units: np.ndarray, ages: np.ndarray) -> float:
    """Mean weekly demand across the mature age window."""
    window = (ages >= MATURE_AGE_LO) & (ages <= MATURE_AGE_HI)
    if not window.any():
        return float("nan")
    return float(np.mean(units[window]))


def _build_launch_library(panel: pd.DataFrame, min_category_donors: int = 5) -> LaunchLibrary:
    """Estimate normalised launch curves from every SKU old enough to have one.

    The pooled statistic is the median, not the mean: one donor whose launch was
    supported by an unusually heavy promotion would otherwise drag the whole
    curve up and every future launch would be over-ordered.
    """
    ages_all = panel["weeks_since_launch"].to_numpy(dtype="float64")
    usable = panel.loc[(ages_all >= 0) & (ages_all <= MAX_CURVE_AGE)].copy()
    usable["age"] = usable["weeks_since_launch"].astype("int64")

    curves: dict[str, list[np.ndarray]] = {}
    category_of: dict[str, str] = {}

    for sku_id, group in usable.groupby("sku_id", sort=False):
        ages = group["age"].to_numpy(dtype="int64")
        units = group["units"].to_numpy(dtype="float64")
        if ages.max() < _MIN_DONOR_AGE:
            continue
        level = _mature_level(units, ages)
        if not np.isfinite(level) or level <= _EPSILON:
            continue

        curve = np.full(MAX_CURVE_AGE + 1, np.nan, dtype="float64")
        curve[ages] = units / level
        curves[str(sku_id)] = curve
        category_of[str(sku_id)] = str(group["category"].iloc[0])

    if not curves:
        raise InsufficientHistoryError(
            "No SKU has enough post-launch history to estimate a launch curve. "
            f"At least one product must be observed to age {_MIN_DONOR_AGE} weeks."
        )

    stacked = np.vstack(list(curves.values()))
    overall = np.nanmedian(stacked, axis=0)
    # Ages nobody was observed at, and the mature tail, are flat at 1.0.
    overall = np.where(np.isfinite(overall), overall, 1.0)
    overall[MATURE_AGE_LO:] = np.where(
        np.isfinite(overall[MATURE_AGE_LO:]), overall[MATURE_AGE_LO:], 1.0
    )

    by_category: dict[str, np.ndarray] = {}
    donors: dict[str, int] = {}
    frame_category = pd.Series(category_of)
    for category, members in frame_category.groupby(frame_category):
        skus = list(members.index)
        donors[str(category)] = len(skus)
        if len(skus) < min_category_donors:
            continue
        subset = np.vstack([curves[sku] for sku in skus])
        pooled = np.nanmedian(subset, axis=0)
        by_category[str(category)] = np.where(np.isfinite(pooled), pooled, overall)

    log.info(
        "built launch library",
        extra={
            "context": {
                "donor_skus": len(curves),
                "category_curves": len(by_category),
                "curve_at_age_1": round(float(overall[1]), 3),
                "curve_at_age_8": round(float(overall[min(8, MAX_CURVE_AGE)]), 3),
            }
        },
    )
    return LaunchLibrary(overall=overall, by_category=by_category, donors=donors)


# --------------------------------------------------------------------------- #
# Attributes and price
# --------------------------------------------------------------------------- #
def _attribute_table(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per SKU describing what it is, standardised for distance."""
    first = (
        panel.sort_values(["sku_id", "week_start"])
        .groupby("sku_id", as_index=False)
        .agg(
            category=("category", "first"),
            subcategory=("subcategory", "first"),
            list_price=("list_price", "median"),
            unit_cost=("unit_cost", "median"),
        )
    )
    first["log_list_price"] = np.log1p(first["list_price"].clip(lower=0.0))
    first["margin_rate"] = (first["list_price"] - first["unit_cost"]) / (
        first["list_price"] + _EPSILON
    )
    # Where a product sits in the price ladder of its own subcategory. This is
    # what makes "the premium one" comparable across subcategories whose absolute
    # prices differ by an order of magnitude.
    first["price_rank"] = first.groupby("subcategory")["list_price"].rank(pct=True)

    for column in _ATTRIBUTES:
        values = first[column].to_numpy(dtype="float64")
        spread = float(np.nanstd(values))
        centre = float(np.nanmean(values))
        first[column] = (values - centre) / (spread if spread > _EPSILON else 1.0)

    return first.set_index("sku_id")


def _fit_price_elasticity(levels: pd.DataFrame) -> float:
    """Pooled elasticity of mature volume with respect to list price.

    Fitted within category by removing each category's mean from both variables,
    so the slope measures "the dearer product in this aisle sells less" rather
    than "expensive categories are smaller", which is a different claim and not
    the one being used.
    """
    usable = levels[(levels["mature_level"] > 0.0) & (levels["list_price"] > 0.0)]
    if len(usable) < 10 or usable["category"].nunique() == 0:
        return 0.0

    log_volume = np.log(usable["mature_level"].to_numpy(dtype="float64"))
    log_price = np.log(usable["list_price"].to_numpy(dtype="float64"))
    category = usable["category"].to_numpy()

    frame = pd.DataFrame({"category": category, "volume": log_volume, "price": log_price})
    centred = frame.assign(
        volume=frame["volume"] - frame.groupby("category")["volume"].transform("mean"),
        price=frame["price"] - frame.groupby("category")["price"].transform("mean"),
    )

    variance = float(np.sum(centred["price"] ** 2))
    if variance <= _EPSILON:
        return 0.0
    slope = float(np.sum(centred["price"] * centred["volume"]) / variance)
    return float(np.clip(slope, *_ELASTICITY_BOUNDS))


def _fit_seasonal_index(panel: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Multiplicative week-of-year index per category, normalised to mean one.

    Estimated on the training window only. A category with a thin week falls back
    to 1.0 rather than to a one-observation estimate.
    """
    frame = panel.loc[:, ["category", "week_start", "units"]].copy()
    frame["iso_week"] = pd.to_datetime(frame["week_start"]).dt.isocalendar()["week"].astype("int64")

    weekly = frame.groupby(["category", "iso_week"], as_index=False)["units"].mean()
    overall = (
        frame.groupby("category", as_index=False)["units"]
        .mean()
        .rename(columns={"units": "category_mean"})
    )
    merged = weekly.merge(overall, on="category", how="left")
    merged["index"] = merged["units"] / merged["category_mean"].replace(0.0, np.nan)
    merged["index"] = merged["index"].fillna(1.0).clip(0.25, 4.0)

    return {
        (str(row.category), int(row.iso_week)): float(row.index)
        for row in merged.itertuples(index=False)
    }


# --------------------------------------------------------------------------- #
# The fitted model
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ColdStartModel:
    """Forecasts young SKUs by borrowing from launches that already happened."""

    library: LaunchLibrary
    attributes: pd.DataFrame
    mature_levels: pd.Series
    seasonal_index: dict[tuple[str, int], float]
    price_elasticity: float
    n_donors: int
    credibility_weeks: float
    diagnostics: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def _donor_prior(self, sku_id: str) -> tuple[float, float]:
        """Donor-implied mature level for ``sku_id``, and the donors' mean price.

        Returns ``(nan, nan)`` when the SKU is unknown or has no usable donors,
        which the caller treats as "cannot borrow" rather than as a zero.
        """
        if sku_id not in self.attributes.index:
            return float("nan"), float("nan")

        target = self.attributes.loc[sku_id]
        pool = self.attributes.drop(index=sku_id, errors="ignore")
        pool = pool[pool.index.isin(self.mature_levels.index)]
        if pool.empty:
            return float("nan"), float("nan")

        difference = pool.loc[:, list(_ATTRIBUTES)].to_numpy(dtype="float64") - np.asarray(
            [target[column] for column in _ATTRIBUTES], dtype="float64"
        )
        distance = np.sqrt(np.sum(difference**2, axis=1))
        distance = distance + np.where(
            pool["subcategory"].to_numpy() != target["subcategory"], _SUBCATEGORY_PENALTY, 0.0
        )
        distance = distance + np.where(
            pool["category"].to_numpy() != target["category"], _CATEGORY_PENALTY, 0.0
        )

        take = min(self.n_donors, len(pool))
        chosen = np.argpartition(distance, take - 1)[:take]
        weights = 1.0 / (1.0 + distance[chosen])
        total = float(np.sum(weights))
        if total <= _EPSILON:
            return float("nan"), float("nan")

        donor_ids = pool.index[chosen]
        donor_levels = self.mature_levels.reindex(donor_ids).to_numpy(dtype="float64")
        donor_prices = pool.iloc[chosen]["list_price"].to_numpy(dtype="float64")

        finite = np.isfinite(donor_levels) & np.isfinite(donor_prices)
        if not finite.any():
            return float("nan"), float("nan")

        weights = weights[finite]
        level = float(np.sum(weights * donor_levels[finite]) / np.sum(weights))
        price = float(np.sum(weights * donor_prices[finite]) / np.sum(weights))
        return level, price

    # ------------------------------------------------------------------ #
    def predict(self, frame: pd.DataFrame, panel: pd.DataFrame) -> np.ndarray:
        """Forecast every row of ``frame``.

        Args:
            frame: Rows carrying ``sku_id``, ``week_start``, ``horizon``,
                ``target_week``, ``weeks_since_launch`` and ``category``.
            panel: The weekly panel, read only at or before each row's origin so
                the SKU's own observed level can be measured.

        Returns:
            One forecast per row, in row order. Rows whose SKU cannot be matched
            to donors and has no history of its own come back as ``nan``, which
            the caller must handle explicitly rather than silently treat as zero.
        """
        if frame.empty:
            return np.zeros(0, dtype="float64")

        observed = self._observed_levels(frame, panel)

        skus = frame["sku_id"].astype(str).to_numpy()
        categories = frame["category"].astype(str).to_numpy()
        origin_age = frame["weeks_since_launch"].to_numpy(dtype="float64")
        horizons = frame["horizon"].to_numpy(dtype="float64")
        target_age = np.clip(origin_age + horizons, 0.0, None)

        target_weeks = pd.to_datetime(frame["target_week"])
        iso_weeks = target_weeks.dt.isocalendar()["week"].astype("int64").to_numpy()

        # --- Scale, per unique SKU ------------------------------------------ #
        priors: dict[str, tuple[float, float]] = {}
        for sku in pd.unique(skus):
            priors[sku] = self._donor_prior(sku)

        target_prices = self.attributes["list_price"].reindex(pd.unique(skus))

        scale = np.empty(len(frame), dtype="float64")
        for index, sku in enumerate(skus):
            donor_level, donor_price = priors[sku]

            # Price-adjust the borrowed level: a product priced above the donors
            # it borrowed from should not inherit their volume unchanged.
            if np.isfinite(donor_level) and np.isfinite(donor_price) and donor_price > 0.0:
                own_price = float(target_prices.get(sku, np.nan))
                if np.isfinite(own_price) and own_price > 0.0:
                    ratio = own_price / donor_price
                    donor_level = donor_level * float(ratio**self.price_elasticity)

            own_level, own_weeks = observed[index]
            if not np.isfinite(own_level) and not np.isfinite(donor_level):
                scale[index] = np.nan
                continue
            if not np.isfinite(donor_level):
                scale[index] = own_level
                continue
            if not np.isfinite(own_level):
                scale[index] = donor_level
                continue

            credibility = own_weeks / (own_weeks + self.credibility_weeks)
            scale[index] = credibility * own_level + (1.0 - credibility) * donor_level

        # --- Shape and season ------------------------------------------------ #
        shape = np.empty(len(frame), dtype="float64")
        for category in pd.unique(categories):
            mask = categories == category
            shape[mask] = self.library.value_at(str(category), target_age[mask].astype("int64"))

        season = np.array(
            [
                self.seasonal_index.get((str(category), int(week)), 1.0)
                for category, week in zip(categories, iso_weeks, strict=True)
            ],
            dtype="float64",
        )

        return np.clip(scale * shape * season, 0.0, None)

    # ------------------------------------------------------------------ #
    def _observed_levels(
        self, frame: pd.DataFrame, panel: pd.DataFrame
    ) -> list[tuple[float, float]]:
        """Mature-equivalent level from each SKU's own history up to its origin.

        The correction that makes this comparable to a donor level: a young SKU's
        raw mean sits on the launch ramp, so it is divided by the average curve
        value over exactly the ages it was observed at. Without that division
        every new product would be under-forecast by the depth of its own ramp.
        """
        history = panel.loc[:, ["sku_id", "week_start", "units", "weeks_since_launch", "category"]]
        history = history[history["weeks_since_launch"] >= _FIRST_FULL_AGE]

        grouped: dict[str, pd.DataFrame] = {
            str(sku): group.sort_values("week_start")
            for sku, group in history.groupby("sku_id", sort=False)
        }

        results: list[tuple[float, float]] = []
        for row in frame.itertuples(index=False):
            sku = str(row.sku_id)
            group = grouped.get(sku)
            if group is None:
                results.append((float("nan"), 0.0))
                continue

            # Strictly at or before the origin: nothing after it may be read.
            visible = group[group["week_start"] <= pd.Timestamp(row.week_start)]
            if visible.empty:
                results.append((float("nan"), 0.0))
                continue

            units = visible["units"].to_numpy(dtype="float64")
            ages = visible["weeks_since_launch"].to_numpy(dtype="float64")
            curve = self.library.value_at(str(row.category), ages.astype("int64"))
            mean_curve = float(np.mean(curve))
            if mean_curve <= _EPSILON:
                results.append((float("nan"), 0.0))
                continue

            results.append((float(np.mean(units)) / mean_curve, float(len(units))))

        return results


def fit_cold_start(
    panel: pd.DataFrame,
    *,
    n_donors: int = DEFAULT_DONORS,
    credibility_weeks: float = CREDIBILITY_WEEKS,
) -> ColdStartModel:
    """Fit the cold-start model on a training-window panel.

    The caller is responsible for passing only weeks inside the training window;
    this function does no time filtering of its own, for the same reason
    :func:`foresight.forecast.train_forecaster` does not - there should be
    exactly one place in the project where a train/test boundary can be got
    wrong, and it is the backtest.

    Args:
        panel: Weekly panel restricted to the training window.
        n_donors: Similar launches pooled per target SKU.
        credibility_weeks: Weeks of a SKU's own history worth as much as the
            whole donor pool.

    Raises:
        InsufficientHistoryError: if no product is old enough to serve as a donor.
    """
    if panel.empty:
        raise InsufficientHistoryError("Weekly panel is empty; cannot fit the cold-start model.")

    library = _build_launch_library(panel)
    attributes = _attribute_table(panel)

    levels: dict[str, float] = {}
    for sku_id, group in panel.groupby("sku_id", sort=False):
        ages = group["weeks_since_launch"].to_numpy(dtype="float64")
        units = group["units"].to_numpy(dtype="float64")
        level = _mature_level(units, ages)
        if np.isfinite(level) and level > 0.0:
            levels[str(sku_id)] = level
    mature_levels = pd.Series(levels, dtype="float64", name="mature_level")

    if mature_levels.empty:
        raise InsufficientHistoryError(
            "No SKU reached the mature age window, so there is nothing to borrow from."
        )

    elasticity_frame = attributes.loc[:, ["category", "list_price"]].join(
        mature_levels, how="inner"
    )
    elasticity = _fit_price_elasticity(elasticity_frame)
    seasonal_index = _fit_seasonal_index(panel)

    model = ColdStartModel(
        library=library,
        attributes=attributes,
        mature_levels=mature_levels,
        seasonal_index=seasonal_index,
        price_elasticity=elasticity,
        n_donors=n_donors,
        credibility_weeks=credibility_weeks,
        diagnostics={
            "donor_skus": float(len(mature_levels)),
            "price_elasticity": round(elasticity, 4),
            "curve_age_1": round(float(library.overall[1]), 4),
            "curve_age_4": round(float(library.overall[min(4, MAX_CURVE_AGE)]), 4),
            "curve_age_13": round(float(library.overall[min(13, MAX_CURVE_AGE)]), 4),
            "category_curves": float(len(library.by_category)),
        },
    )

    log.info(
        "fitted cold-start model",
        extra={
            "context": {
                "donor_skus": len(mature_levels),
                "price_elasticity": round(elasticity, 4),
                "credibility_weeks": credibility_weeks,
                "n_donors": n_donors,
            }
        },
    )
    return model
