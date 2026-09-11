# Data source comparison — self-generated vs. the provided extract

**Project FORESIGHT — Demand & Inventory Intelligence**
**Client:** NorthBay Living | **Companion to:** D1 (pipeline), D2 (EDA), D3 (forecast)

## Why this exists

The brief (`Zidio_Project_Data_1.1.pdf`, §05) states: *"You receive four
simulated extracts... A synthetic dataset accompanies this brief. Do not spend
the engagement fabricating data."* This repository shipped with two things
that both look like "the data": `scripts/00_generate_data.py`, which
synthesises a 200-SKU, ~3.7-year extract from a fixed seed, and a second,
smaller extract (50 SKUs, 2024-01-01 to 2025-12-22) sitting at the project
root whose column names and table grain match the brief's Appendix A
description — the actual client-provided extract.

Rather than assume one is right, both were run through the identical,
unmodified pipeline (`01_run_pipeline.py` → `02_run_eda.py` →
`03_train_backtest.py --no-drivers` → `04_score_risk.py` →
`05_build_readout.py`), and the results were compared head-to-head. Both runs
used `--no-drivers`: the self-generated source has optional commercial-driver
extracts (marketing spend, web analytics, market conditions) that the provided
extract does not, and including them on only one side would measure the driver
extension rather than the data source. Full outputs for both runs are kept at
`runs/synthetic_generated/` and `runs/provided/` for inspection; nothing was
deleted.

## Results, head-to-head

| | `synthetic_generated` | `provided` |
|---|---:|---:|
| SKUs | 200 | 50 |
| History | 2023-01-02 → 2026-08-30 (~3.7 yrs) | 2024-01-01 → 2025-12-22 (2 yrs) |
| Selected model | Adaptive Ensemble | Adaptive Ensemble |
| **WAPE (backtest)** | **23.8%** | **9.0%** |
| Seasonal-naive baseline WAPE | 34.5% | 11.8% |
| Improvement vs. baseline | 31.0% | 23.6% |
| 80% interval coverage (target 80%) | 81.3% | 82.5% |
| Sales at risk (stockout) | ₹1.43 Cr | ₹30.7 L |
| Capital locked (overstock) | ₹5.97 Cr | ₹71.9 L |

(Full figures: `runs/synthetic_generated/artifacts/metrics.json` /
`impact.json` vs. `runs/provided/artifacts/metrics.json` / `impact.json`.)

## Reading the numbers honestly

The provided extract posts a *lower absolute error* (9.0% vs 23.8% WAPE) but a
*smaller relative improvement over its own baseline* (23.6% vs 31.0%). Both
are true at once and say different things: the provided extract's demand is
simply smoother and easier to predict — its seasonal-naive baseline is
already strong at 11.8% WAPE — so there is less room for a model to add value
on top of it. The self-generated extract has noisier, more promotion-driven
demand (deliberately, since it was built to exercise seasonality, promotions
and cold-start SKUs), which is harder to predict in absolute terms but gives
the model more to legitimately win back from the baseline.

One data-quality finding is specific to the provided extract and worth
recording here rather than only in the EDA memo: its `sku_master` category
labels (`Furniture`, `Home Decor`, `Kitchen`, `Lighting`, `Storage`) don't
match NorthBay's controlled vocabulary one-for-one.
`scripts/prepare_provided_extract.py` aligns the four that are pure
naming-convention differences (`Home Decor`→`Decor`, `Storage`→
`Storage & Organisation`, `Kitchen`→`Kitchen & Dining`) before the extract
ever reaches the cleaner. `Furniture` (10 of 50 SKUs) has no counterpart in
NorthBay's six categories at all — that's a genuine catalogue gap, not a
label quirk, so it is left unmapped and resolves to `Unclassified`, exactly
as `eda_memo.md`'s data-quality table now reports.

## Decision: `provided` is canonical

The provided extract is promoted to the live pipeline — it is the actual
client-supplied data the brief describes serving, it produces materially
better absolute accuracy (9.0% vs. 23.8% WAPE), and its interval coverage sits
closer to the 80% target. `data/raw/`, `data/processed/`, `artifacts/`, and
`reports/eda_memo.md`/`figures/` all currently reflect this run — no
restoration step was needed since it was the run performed last and nothing
downstream has touched those paths since.

The self-generated extract is kept, unmodified, at
`data/sources/synthetic_generated/` and its full run at
`runs/synthetic_generated/`, and remains available to reproduce via
`scripts/run_source_comparison.py --sources synthetic_generated` — its larger
SKU count and longer history make it a reasonable stress-test dataset for
future model work even though it is not the client's own data.

## Reproducing this comparison

```bash
python scripts/prepare_provided_extract.py     # rename the provided extract onto Appendix A
python scripts/run_source_comparison.py        # both sources, snapshotted under runs/
```
