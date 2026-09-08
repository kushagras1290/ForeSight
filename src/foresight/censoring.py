"""FORESIGHT Censored Demand Recovery - the fourth custom model.

THE FAILURE THIS EXISTS FOR
---------------------------
Every other model in this project learns from the sales ledger. But a sales
ledger does not record demand. It records::

    sales = min(demand, what was on the shelf)

Those are the same number only while stock lasts. On this dataset 11% of
SKU-weeks end at zero on-hand, so for roughly one week in nine the ledger is a
*lower bound* on demand and the model is being taught the wrong number - and it
is taught it precisely on the weeks demand was highest, because that is when
stock runs out.

Left uncorrected this closes a loop that gets worse every cycle:

    stockout -> recorded sales fall -> forecast falls -> reorder falls ->
    stockout sooner

The forecast becomes a self-fulfilling prophecy, the SKU is quietly demoted, and
nothing in the reporting shows it happening, because from the model's point of
view its predictions look *more* accurate every cycle. It is predicting its own
past decisions rather than demand.

THE CORRECTION
--------------
Recover demand from the two facts the extracts do carry: how much was available
and how much sold.

1. **Availability.** Inventory snapshots give on-hand at the start of each week;
   differencing successive snapshots against sales recovers receipts, and hence
   how much could have been sold at all. Assuming demand arrives evenly through
   the week - which is the standard assumption, and the honest one absent
   intra-week data - a product with ``A`` units available against an expected
   ``d`` units of demand is in stock for ``min(1, A / d)`` of that week.

2. **Inversion.** If only ``a`` of the week was sellable, the ``s`` units
   recorded represent ``s / a`` units of underlying demand. When ``a`` reaches
   zero nothing at all could sell, ``s / a`` says nothing, and the estimate falls
   back to the product's own uncensored demand level scaled by season.

3. **Iteration.** Expected demand is needed to estimate availability, and
   availability is needed to estimate demand. Starting from uncensored weeks
   only and alternating a few times is a small EM: each pass has a cleaner
   demand estimate to work from, and it converges quickly because the uncensored
   weeks - the large majority - never move.

GUARDRAILS
----------
Two, because an imputation that runs away is worse than no imputation:

* Recovered demand is never below observed sales. Whatever else is uncertain,
  the units that sold definitely sold.
* Uplift is capped. A single bad availability estimate cannot manufacture a
  demand spike that then propagates into a reorder recommendation.

HOW THIS IS EVALUATED HONESTLY
------------------------------
Scoring a forecast against an imputed target would be circular - a forecast that
happened to match the imputation would score well whether or not either was
right. So the two uses are deliberately separated:

* **Training** uses recovered demand, because true demand is what the business
  needs predicted.
* **Scoring** uses only weeks that were *not* censored, where the observation is
  the truth and no imputation is involved.

The imputation therefore cannot flatter any reported number. It can only be
judged by whether models trained on it forecast uncensored weeks better - which
is exactly the question worth asking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from foresight.exceptions import DataQualityError
from foresight.logging_setup import get_logger

__all__ = [
    "CensoringSummary",
    "attach_availability",
    "recover_censored_demand",
]

log = get_logger(__name__)

#: Availability at or above this counts the week as fully served. Not 1.0,
#: because a product that ran out in the last hours of a week lost essentially
#: no demand and correcting it would add noise for nothing.
UNCENSORED_THRESHOLD: Final[float] = 0.97

#: Availability at or below this counts the week as a total stockout, where the
#: observed figure carries no information about the level of demand.
BLACKOUT_THRESHOLD: Final[float] = 0.05

#: Floor on the availability divisor. Without it, a week estimated at 1%
#: availability would multiply observed sales a hundredfold.
_MIN_AVAILABILITY: Final[float] = 0.15

#: Hard ceiling on recovered / observed. Uncertainty in availability is real;
#: a five-fold correction is already at the edge of credible, and anything past
#: it is far more likely to be a bad inventory snapshot than genuine lost demand.
MAX_UPLIFT: Final[float] = 5.0

#: EM passes. The uncensored majority does not move between passes, so the
#: estimate is stable well before this.
DEFAULT_ITERATIONS: Final[int] = 3

_EPSILON: Final[float] = 1e-6


@dataclass(frozen=True, slots=True)
class CensoringSummary:
    """What the recovery found and changed, for the data-quality memo."""

    total_weeks: int
    censored_weeks: int
    blackout_weeks: int
    observed_units: float
    recovered_units: float
    skus_affected: int
    iterations: int

    @property
    def censored_share(self) -> float:
        return self.censored_weeks / self.total_weeks if self.total_weeks else 0.0

    @property
    def uplift_share(self) -> float:
        """Recovered demand above observed sales, as a share of observed."""
        if self.observed_units <= 0.0:
            return 0.0
        return (self.recovered_units - self.observed_units) / self.observed_units

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_weeks": self.total_weeks,
            "censored_weeks": self.censored_weeks,
            "blackout_weeks": self.blackout_weeks,
            "censored_share": round(self.censored_share, 6),
            "observed_units": round(self.observed_units, 2),
            "recovered_units": round(self.recovered_units, 2),
            "uplift_share": round(self.uplift_share, 6),
            "skus_affected": self.skus_affected,
            "iterations": self.iterations,
        }


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def attach_availability(
    panel: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    expected_demand: pd.Series | None = None,
) -> pd.DataFrame:
    """Estimate what fraction of each SKU-week was sellable.

    Args:
        panel: Weekly panel with ``sku_id``, ``week_start`` and ``units``.
        inventory: Cleaned inventory snapshots with ``date``, ``sku_id`` and
            ``on_hand_units``. Snapshots are taken at week starts.
        expected_demand: Demand rate per panel row to compare stock against.
            Defaults to each SKU's own mean observed sales, which is the right
            starting point but is itself depressed by censoring - hence the
            iteration in :func:`recover_censored_demand`.

    Returns:
        ``panel`` plus ``on_hand_start``, ``available_units`` and
        ``availability``.

    Raises:
        DataQualityError: if the inventory extract lacks a required column.
    """
    required = {"date", "sku_id", "on_hand_units"}
    missing = required - set(inventory.columns)
    if missing:
        raise DataQualityError(
            f"Inventory extract is missing column(s) needed for censoring "
            f"detection: {', '.join(sorted(missing))}"
        )

    snapshots = inventory.loc[:, ["sku_id", "date", "on_hand_units"]].rename(
        columns={"date": "week_start", "on_hand_units": "on_hand_start"}
    )
    snapshots["week_start"] = pd.to_datetime(snapshots["week_start"])
    snapshots = snapshots.groupby(["sku_id", "week_start"], as_index=False)["on_hand_start"].last()

    frame = panel.copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    frame = frame.merge(snapshots, on=["sku_id", "week_start"], how="left", validate="one_to_one")
    frame = frame.sort_values(["sku_id", "week_start"]).reset_index(drop=True)

    # Next week's opening stock is this week's closing stock. Receipts during the
    # week are whatever closed the gap that sales alone cannot explain; they are
    # units that were genuinely sellable and must be counted as such.
    grouped = frame.groupby("sku_id", sort=False)["on_hand_start"]
    on_hand_end = grouped.shift(-1)
    receipts = (on_hand_end - frame["on_hand_start"] + frame["units"]).clip(lower=0.0)

    # The last week of each SKU has no following snapshot, so receipts are
    # unknowable. Assume none rather than guessing: it makes availability a
    # lower bound, which errs toward leaving the observation alone.
    receipts = receipts.fillna(0.0)

    frame["available_units"] = (frame["on_hand_start"].fillna(0.0) + receipts).clip(lower=0.0)

    if expected_demand is None:
        expected = frame.groupby("sku_id", sort=False)["units"].transform("mean")
    else:
        expected = pd.Series(
            np.asarray(expected_demand, dtype="float64"), index=frame.index, dtype="float64"
        )
    expected = expected.clip(lower=_EPSILON)

    availability = (frame["available_units"] / expected).clip(0.0, 1.0)

    # A SKU with no inventory record at all is not evidence of a stockout - it is
    # a gap in the extract. Treat it as fully available so the recovery leaves it
    # untouched, and let the cleaning report flag the missing rows.
    availability = availability.where(frame["on_hand_start"].notna(), 1.0)

    # A week that sold more than its estimated availability was clearly better
    # supplied than the estimate suggests. Trust the ledger over the snapshot.
    sold_share = frame["units"] / frame["available_units"].clip(lower=_EPSILON)
    availability = np.where(
        (frame["units"] > 0.0) & (sold_share <= 1.0) & (availability < 1.0),
        np.maximum(availability, np.minimum(1.0, sold_share)),
        availability,
    )

    frame["availability"] = np.clip(availability, 0.0, 1.0)
    return frame


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #
def _seasonal_fallback(frame: pd.DataFrame) -> np.ndarray:
    """Expected demand for weeks where nothing could sell.

    Built from uncensored weeks only, as the SKU's own level times its
    category's week-of-year shape. A blackout week carries no information about
    its own demand, so the estimate has to come entirely from elsewhere.
    """
    clean = frame[frame["availability"] >= UNCENSORED_THRESHOLD]
    if clean.empty:
        return frame.groupby("sku_id", sort=False)["units"].transform("mean").to_numpy()

    sku_level = clean.groupby("sku_id")["units"].mean()
    global_level = float(clean["units"].mean())

    iso = pd.to_datetime(clean["week_start"]).dt.isocalendar()["week"].astype("int64")
    seasonal = clean.assign(iso_week=iso.to_numpy())
    category_mean = seasonal.groupby("category")["units"].transform("mean")
    seasonal = seasonal.assign(ratio=seasonal["units"] / category_mean.replace(0.0, np.nan))
    index = seasonal.groupby(["category", "iso_week"])["ratio"].mean().clip(0.25, 4.0).to_dict()

    frame_iso = pd.to_datetime(frame["week_start"]).dt.isocalendar()["week"].astype("int64")
    levels = frame["sku_id"].map(sku_level).fillna(global_level).to_numpy(dtype="float64")
    shape = np.array(
        [
            index.get((category, int(week)), 1.0)
            for category, week in zip(frame["category"], frame_iso, strict=True)
        ],
        dtype="float64",
    )
    return levels * shape


def recover_censored_demand(
    panel: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    iterations: int = DEFAULT_ITERATIONS,
    max_uplift: float = MAX_UPLIFT,
) -> tuple[pd.DataFrame, CensoringSummary]:
    """Recover true demand from censored sales.

    Args:
        panel: The weekly panel. Must carry ``sku_id``, ``week_start``,
            ``units`` and ``category``.
        inventory: Cleaned inventory snapshots.
        iterations: EM passes alternating between the demand estimate and the
            availability estimate.
        max_uplift: Ceiling on recovered / observed for a single week.

    Returns:
        The panel with ``availability``, ``is_censored``, ``units_demand`` and
        ``censoring_uplift`` added, and a summary for the data-quality memo.

    Raises:
        DataQualityError: if the panel is empty or the inventory is unusable.
    """
    if panel.empty:
        raise DataQualityError("Weekly panel is empty; nothing to recover.")
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, received {iterations}")

    frame = attach_availability(panel, inventory)
    demand = frame["units"].to_numpy(dtype="float64").copy()

    for pass_index in range(iterations):
        blackout = frame["availability"].to_numpy() <= BLACKOUT_THRESHOLD
        divisor = np.maximum(frame["availability"].to_numpy(), _MIN_AVAILABILITY)

        inverted = frame["units"].to_numpy(dtype="float64") / divisor
        fallback = _seasonal_fallback(frame.assign(units_demand=demand))

        recovered = np.where(blackout, fallback, inverted)

        # Guardrails: never below what sold, never a runaway multiple of it.
        observed = frame["units"].to_numpy(dtype="float64")
        ceiling = np.where(observed > 0.0, observed * max_uplift, fallback * max_uplift)
        demand = np.clip(np.maximum(recovered, observed), 0.0, np.maximum(ceiling, _EPSILON))

        # Weeks that were fully served are the ledger's own number, untouched.
        served = frame["availability"].to_numpy() >= UNCENSORED_THRESHOLD
        demand = np.where(served, observed, demand)

        if pass_index < iterations - 1:
            # Re-estimate availability against the improved demand rate. This is
            # the step that matters: the first pass compares stock to a demand
            # mean that censoring itself pushed down, so it under-detects.
            expected = (
                pd.Series(demand, index=frame.index)
                .groupby(frame["sku_id"].to_numpy(), sort=False)
                .transform("mean")
            )
            frame = attach_availability(panel, inventory, expected_demand=expected)

    frame["units_demand"] = demand
    frame["is_censored"] = (frame["availability"] < UNCENSORED_THRESHOLD).astype("int64")
    frame["censoring_uplift"] = frame["units_demand"] - frame["units"]

    censored_mask = frame["is_censored"] == 1
    summary = CensoringSummary(
        total_weeks=int(len(frame)),
        censored_weeks=int(censored_mask.sum()),
        blackout_weeks=int((frame["availability"] <= BLACKOUT_THRESHOLD).sum()),
        observed_units=float(frame["units"].sum()),
        recovered_units=float(frame["units_demand"].sum()),
        skus_affected=int(frame.loc[censored_mask, "sku_id"].nunique()),
        iterations=iterations,
    )

    log.info(
        "recovered censored demand",
        extra={
            "context": {
                **summary.to_dict(),
                "max_single_week_uplift": round(float(frame["censoring_uplift"].max()), 2),
            }
        },
    )
    return frame, summary
