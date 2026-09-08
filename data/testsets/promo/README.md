# Test set: Heavy promotional period

**Does it call the peak when a third of the range goes on offer?**

The same real year, with four two-week promotional bursts applied across a third of the assortment, driving several times baseline demand.

A model trained mostly on unpromoted weeks tends to under-call these peaks, and that shows up as a *negative bias* rather than as scattered error - the signature to look for is the bias figure moving sharply while accuracy falls less than you would expect. This is the case the promotional response model exists for.

## Files

| File | What it is |
|---|---|
| `actuals.csv` | What demand turned out to be. |
| `forecast.csv` | The forecast under test, so the pair scores end to end. |

Upload **both files together** on the dashboard's *Forecast accuracy* tab.

## What it scored

| Measure | Value |
|---|---:|
| Accuracy | **70.5%** |
| WAPE | 0.2949 |
| Seasonal-naive WAPE | 0.3796 |
| Bias | -13.8% |
| Interval coverage | 74.5% (80% stated) |
| Product-weeks scored | 10,760 |

### Against the control

The `steady` set scores **74.9%** on the same forecast and the
same weeks. This scenario costs **-4.4 points** of accuracy across
the whole portfolio and moves bias by **-5.9%**.

### Where the damage actually is

An aggregate figure dilutes a localised failure. Scored only on
the promoted products, during their promotion weeks:

| Measure | Value |
|---|---:|
| Accuracy | **29.6%** |
| Bias | -70.1% |
| Interval coverage | 9.2% |
| Product-weeks affected | 502 |

That is the number to quote about this scenario. The portfolio figure
above is real, but it mostly measures how much of the assortment was
left alone.

## How this set was bent

66 of 200 products are promoted across 8 weeks, in four two-week bursts, each at an uplift drawn between 2.0x and 4.5x.

## How it was built

The base is the project's **chronological holdout** - a year of real demand
the model never trained on, with the forecasts it genuinely made for those
weeks. `steady` is that period untouched; the other sets take the same real
base and bend it, so any difference from the control is caused by the stress
and nothing else.

Regenerate with `python scripts/09_generate_test_sets.py`.
