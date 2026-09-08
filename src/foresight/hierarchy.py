"""The aggregation structure of the assortment, and how to build series for it.

WHY THIS EXISTS
---------------
A SKU-week forecast is the hardest thing in the catalogue to predict, because a
single product's weekly demand is mostly noise. Aggregate that same demand and
the noise cancels: on this engagement's own backtest, category-week forecasts
run at roughly 90% accuracy against 72% at SKU-week, from the *same* model on
the *same* data. The information is real, and bottom-up planning throws it away
- it sums 200 noisy forecasts and inherits all 200 errors.

This module builds the structure needed to exploit that. It produces:

* the **summing matrix** ``S`` that expresses every aggregate as a sum of SKUs,
* an **aggregate panel** carrying category, subcategory and total series in the
  exact schema the SKU panel uses, so the existing feature pipeline forecasts
  them without a single special case.

:mod:`foresight.reconcile` then combines forecasts across those levels.

THE HIERARCHY
-------------
Four levels, strictly nested::

    TOTAL                       1 node
    +- CAT::<category>          one per category
       +- SUB::<cat>|<sub>      one per (category, subcategory)
          +- <sku_id>           the bottom level - 200 nodes

Subcategory nodes carry their parent category in the key because subcategory
names are only unique *within* a category; keying on the bare name would silently
merge two unrelated groups into one series.

AGGREGATING A PANEL
-------------------
Each column is aggregated the way its own units demand, not with a blanket
``sum``. Quantities add; prices are volume-weighted, because the average selling
price of a category is what its customers actually paid across the mix, not the
unweighted mean of its price tags; ages take the maximum, so an aggregate node is
as old as its oldest member and is never mistaken for a new launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from foresight.exceptions import DataQualityError
from foresight.logging_setup import get_logger

__all__ = [
    "CATEGORY_PREFIX",
    "Hierarchy",
    "LEVELS",
    "SUBCATEGORY_PREFIX",
    "TOTAL_NODE",
    "aggregate_panel",
    "build_hierarchy",
]

log = get_logger(__name__)

TOTAL_NODE: Final[str] = "TOTAL"
CATEGORY_PREFIX: Final[str] = "CAT::"
SUBCATEGORY_PREFIX: Final[str] = "SUB::"

#: Level names, ordered from the top of the hierarchy down.
LEVELS: Final[tuple[str, ...]] = ("total", "category", "subcategory", "sku")

#: Sentinel used for the category/subcategory attributes of nodes that sit above
#: those levels. LightGBM treats it as one more category level, which is exactly
#: right: "this series is the whole business" is a real, distinct value.
_ALL: Final[str] = "__ALL__"


def _category_node(category: str) -> str:
    return f"{CATEGORY_PREFIX}{category}"


def _subcategory_node(category: str, subcategory: str) -> str:
    return f"{SUBCATEGORY_PREFIX}{category}|{subcategory}"


@dataclass(frozen=True, slots=True)
class Hierarchy:
    """The summing structure of the assortment.

    Attributes:
        nodes: Every series in the hierarchy, ordered top-down (total, then
            categories, then subcategories, then SKUs). Length ``n``.
        bottom: The SKU ids, in the order they appear as columns of ``S``.
            Length ``m``.
        levels: The level name of each node, parallel to ``nodes``.
        summing: The ``(n, m)`` 0/1 matrix with ``S[i, j] == 1`` when SKU ``j``
            is part of node ``i``. Bottom rows form an identity block, so
            ``S @ b`` reproduces the SKU forecasts unchanged alongside every
            aggregate they imply.
    """

    nodes: tuple[str, ...]
    bottom: tuple[str, ...]
    levels: tuple[str, ...]
    summing: np.ndarray

    # ------------------------------------------------------------------ #
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_bottom(self) -> int:
        return len(self.bottom)

    @property
    def node_index(self) -> dict[str, int]:
        """Row position of each node in ``summing``."""
        return {node: index for index, node in enumerate(self.nodes)}

    @property
    def aggregate_nodes(self) -> tuple[str, ...]:
        """Every node above the bottom level."""
        return tuple(
            node for node, level in zip(self.nodes, self.levels, strict=True) if level != "sku"
        )

    def level_of(self, node: str) -> str:
        position = self.node_index.get(node)
        if position is None:
            raise KeyError(f"{node!r} is not part of this hierarchy")
        return self.levels[position]

    def coherence_error(self, values: np.ndarray) -> float:
        """Largest absolute violation of ``aggregate == sum of its children``.

        A reconciled forecast must satisfy this by construction; anything above
        floating-point noise means the reconciliation is wrong, so this is the
        check that proves the output is usable as a plan.
        """
        vector = np.asarray(values, dtype="float64").reshape(-1)
        if vector.shape[0] != self.n_nodes:
            raise ValueError(f"expected {self.n_nodes} values, received {vector.shape[0]}")
        bottom_mask = np.array([level == "sku" for level in self.levels])
        implied = self.summing @ vector[bottom_mask]
        return float(np.max(np.abs(vector - implied)))


def build_hierarchy(sku_master: pd.DataFrame) -> Hierarchy:
    """Build the four-level hierarchy from the SKU dimension.

    Args:
        sku_master: Must carry ``sku_id``, ``category`` and ``subcategory``.

    Raises:
        DataQualityError: if the dimension is empty, has duplicate SKU ids, or
            contains an id that collides with a reserved aggregate-node name.
    """
    required = {"sku_id", "category", "subcategory"}
    missing = required - set(sku_master.columns)
    if missing:
        raise DataQualityError(
            f"SKU master is missing column(s) needed to build the hierarchy: "
            f"{', '.join(sorted(missing))}"
        )
    if sku_master.empty:
        raise DataQualityError("SKU master is empty; cannot build a hierarchy.")

    frame = (
        sku_master.loc[:, ["sku_id", "category", "subcategory"]]
        .astype({"sku_id": "object", "category": "object", "subcategory": "object"})
        .sort_values("sku_id")
        .reset_index(drop=True)
    )

    duplicates = frame["sku_id"].duplicated()
    if duplicates.any():
        offenders = sorted(frame.loc[duplicates, "sku_id"].unique().tolist())[:5]
        raise DataQualityError(
            f"SKU master has {int(duplicates.sum())} duplicate sku_id(s), e.g. "
            f"{', '.join(offenders)}. The summing matrix would double-count them."
        )

    reserved = frame["sku_id"].str.startswith((CATEGORY_PREFIX, SUBCATEGORY_PREFIX)) | (
        frame["sku_id"] == TOTAL_NODE
    )
    if reserved.any():
        offenders = sorted(frame.loc[reserved, "sku_id"].unique().tolist())[:5]
        raise DataQualityError(
            f"sku_id(s) collide with reserved aggregate-node names: {', '.join(offenders)}. "
            f"Rename them, or change TOTAL_NODE/{CATEGORY_PREFIX}/{SUBCATEGORY_PREFIX}."
        )

    bottom = tuple(frame["sku_id"].tolist())
    bottom_position = {sku: index for index, sku in enumerate(bottom)}

    categories = sorted(frame["category"].unique().tolist())
    subcategory_pairs = sorted(
        {(row.category, row.subcategory) for row in frame.itertuples(index=False)}
    )

    nodes: list[str] = [TOTAL_NODE]
    levels: list[str] = ["total"]
    members: list[list[int]] = [list(range(len(bottom)))]

    for category in categories:
        nodes.append(_category_node(category))
        levels.append("category")
        members.append(
            [
                bottom_position[row.sku_id]
                for row in frame.itertuples(index=False)
                if row.category == category
            ]
        )

    for category, subcategory in subcategory_pairs:
        nodes.append(_subcategory_node(category, subcategory))
        levels.append("subcategory")
        members.append(
            [
                bottom_position[row.sku_id]
                for row in frame.itertuples(index=False)
                if row.category == category and row.subcategory == subcategory
            ]
        )

    for sku in bottom:
        nodes.append(sku)
        levels.append("sku")
        members.append([bottom_position[sku]])

    summing = np.zeros((len(nodes), len(bottom)), dtype="float64")
    for row_index, member_indices in enumerate(members):
        summing[row_index, member_indices] = 1.0

    hierarchy = Hierarchy(
        nodes=tuple(nodes),
        bottom=bottom,
        levels=tuple(levels),
        summing=summing,
    )

    log.info(
        "built hierarchy",
        extra={
            "context": {
                "nodes": hierarchy.n_nodes,
                "skus": hierarchy.n_bottom,
                "categories": len(categories),
                "subcategories": len(subcategory_pairs),
            }
        },
    )
    return hierarchy


# --------------------------------------------------------------------------- #
# Aggregate series
# --------------------------------------------------------------------------- #
#: Columns that add across SKUs.
_ADDITIVE: Final[tuple[str, ...]] = (
    "units",
    "revenue",
    "media_spend",
    "impressions",
    "email_sends",
    "sessions",
    "add_to_cart",
)

#: Columns averaged by unit volume - what customers actually paid or saw across
#: the mix, rather than an unweighted average over unequal products.
_VOLUME_WEIGHTED: Final[tuple[str, ...]] = (
    "list_price",
    "unit_cost",
    "promo_days",
    "discount_pct",
    "competitor_price_index",
    "weather_anomaly",
)

#: Columns describing the week itself, identical for every SKU in it.
_WEEK_LEVEL: Final[tuple[str, ...]] = ("holiday_days",)


def _aggregate_group(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse a SKU-week frame to one row per week, column by column."""
    grouped = frame.groupby("week_start", sort=True)

    parts: dict[str, pd.Series] = {}
    for column in _ADDITIVE:
        if column in frame.columns:
            parts[column] = grouped[column].sum()

    # Weight by units, falling back to a plain mean in weeks that sold nothing -
    # a zero-demand week still has a real list price, and dropping it would leave
    # a gap in the series.
    weights = frame["units"].clip(lower=0.0)
    weighted = frame.assign(_weight=weights)
    weight_total = weighted.groupby("week_start", sort=True)["_weight"].sum()
    for column in _VOLUME_WEIGHTED:
        if column not in frame.columns:
            continue
        product = weighted.assign(_product=weighted[column] * weighted["_weight"])
        numerator = product.groupby("week_start", sort=True)["_product"].sum()
        plain = grouped[column].mean()
        parts[column] = np.where(
            weight_total > 0.0, numerator / weight_total.replace(0.0, np.nan), plain
        )
        parts[column] = pd.Series(parts[column], index=plain.index).fillna(plain)

    for column in _WEEK_LEVEL:
        if column in frame.columns:
            parts[column] = grouped[column].max()

    aggregated = pd.DataFrame(parts)

    # Realised average selling price of the aggregate: total revenue over total
    # units. Undefined in a zero-demand week, where the list price is the honest
    # stand-in for "what this would have sold at".
    if "revenue" in aggregated.columns and "units" in aggregated.columns:
        units = aggregated["units"].to_numpy(dtype="float64")
        revenue = aggregated["revenue"].to_numpy(dtype="float64")
        fallback = aggregated.get("list_price")
        fallback_values = (
            fallback.to_numpy(dtype="float64") if fallback is not None else np.zeros_like(units)
        )
        aggregated["avg_price"] = np.where(
            units > 0.0, revenue / np.where(units > 0.0, units, 1.0), fallback_values
        )

    if "weeks_since_launch" in frame.columns:
        aggregated["weeks_since_launch"] = grouped["weeks_since_launch"].max()
    if "launch_date" in frame.columns:
        aggregated["launch_date"] = grouped["launch_date"].min()
    if "is_post_launch" in frame.columns:
        aggregated["is_post_launch"] = grouped["is_post_launch"].max()

    return aggregated.reset_index()


def aggregate_panel(panel: pd.DataFrame, hierarchy: Hierarchy) -> pd.DataFrame:
    """Build weekly series for every node above the SKU level.

    The result uses the *same* schema as the SKU panel, with ``sku_id`` holding
    the node id. That is what lets :func:`foresight.features.build_base_features`
    and the GBM forecast aggregates with no branch of their own: to the model an
    aggregate node is simply a product with more volume and a longer history.

    Args:
        panel: The weekly SKU panel.
        hierarchy: The structure to aggregate over.

    Returns:
        One row per (aggregate node, week), carrying ``node_level`` alongside the
        panel's own columns.
    """
    if panel.empty:
        raise DataQualityError("Weekly panel is empty; nothing to aggregate.")

    frames: list[pd.DataFrame] = []

    total = _aggregate_group(panel)
    total["sku_id"] = TOTAL_NODE
    total["category"] = _ALL
    total["subcategory"] = _ALL
    total["node_level"] = "total"
    frames.append(total)

    for category, group in panel.groupby("category", sort=True):
        node = _aggregate_group(group)
        node["sku_id"] = _category_node(str(category))
        node["category"] = category
        node["subcategory"] = _ALL
        node["node_level"] = "category"
        frames.append(node)

    for (category, subcategory), group in panel.groupby(["category", "subcategory"], sort=True):
        node = _aggregate_group(group)
        node["sku_id"] = _subcategory_node(str(category), str(subcategory))
        node["category"] = category
        node["subcategory"] = subcategory
        node["node_level"] = "subcategory"
        frames.append(node)

    aggregated = pd.concat(frames, ignore_index=True)

    # Keep the column order and dtypes aligned with the SKU panel so downstream
    # code cannot tell the two apart.
    for column in panel.columns:
        if column not in aggregated.columns:
            aggregated[column] = 0.0
    ordered = [*panel.columns, "node_level"]
    aggregated = aggregated.loc[:, ordered]

    expected_nodes = len(hierarchy.aggregate_nodes)
    actual_nodes = int(aggregated["sku_id"].nunique())
    if actual_nodes != expected_nodes:
        raise DataQualityError(
            f"Aggregated {actual_nodes} nodes but the hierarchy declares "
            f"{expected_nodes}. The panel and the SKU master disagree about the "
            "assortment; reconcile them before forecasting."
        )

    log.info(
        "built aggregate panel",
        extra={
            "context": {
                "rows": len(aggregated),
                "nodes": actual_nodes,
                "weeks": int(aggregated["week_start"].nunique()),
            }
        },
    )
    return aggregated.sort_values(["sku_id", "week_start"]).reset_index(drop=True)
