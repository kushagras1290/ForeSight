"""Project FORESIGHT - demand & inventory intelligence for NorthBay Living.

A four-week data science engagement delivering:

* ``pipeline``  - reproducible ingest + clean into an analysis-ready dataset (D1)
* ``eda``       - data-quality and exploratory findings (D2)
* ``forecast``  - weekly SKU-level demand forecast, backtested against a
                  seasonal-naive baseline (D3)
* ``risk``      - stockout / overstock scoring with recommended actions (D4)

Five purpose-built models sit on top of that base, each addressing a different
and independent way SKU-level demand forecasting fails:

* ``custom_model``  - Adaptive Demand Ensemble. Routes each SKU to the estimator
                      that suits its demand regime instead of forcing one model
                      on a catalogue that is not homogeneous.
* ``reconcile``     - Hierarchical Demand Reconciler. Combines forecasts made
                      independently at SKU, subcategory, category and total level
                      into one coherent set, so the SKU plan borrows the accuracy
                      that only exists in aggregate.
* ``coldstart``     - Cold-Start Propagation. Forecasts products with no history
                      by borrowing the launch trajectories of similar products,
                      handing over to their own data as evidence accumulates.
* ``censoring``     - Censored Demand Recovery. Recovers true demand from sales
                      that were capped by a stockout, so the model is not taught
                      to forecast its own past shortages.
* ``promotions``    - Promotional Response Decomposition. Separates baseline
                      demand from promotional uplift and the pull-forward dip
                      that follows.

The dashboard (D5) and scoring service (D6) live in ``dashboard/`` and
``service/`` at the repository root.
"""

from __future__ import annotations

__version__ = "1.1.0"
__all__ = ["__version__"]
