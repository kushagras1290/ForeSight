# Test set: Structural break

**How badly does it fail when the world changes underneath it?**

The same real year, with one category made to surge as though a product went viral or a competitor left the market, and another to collapse as though supply was disrupted. Both build gradually from the midpoint of the period, and nothing in the history predicts either.

Accuracy is expected to fall, and it should. The useful questions are by how much, how fast the prediction intervals stop covering, and whether the failure is visible in the numbers or silent. A model that degrades gracefully is usable in a crisis; one that does not is dangerous precisely when it matters most.

## Files

| File | What it is |
|---|---|
| `actuals.csv` | What demand turned out to be. |
| `forecast.csv` | The forecast under test, so the pair scores end to end. |

Upload **both files together** on the dashboard's *Forecast accuracy* tab.

## What it scored

| Measure | Value |
|---|---:|
| Accuracy | **71.0%** |
| WAPE | 0.2897 |
| Seasonal-naive WAPE | 0.3763 |
| Bias | -10.4% |
| Interval coverage | 71.5% (80% stated) |
| Product-weeks scored | 10,760 |

### Against the control

The `steady` set scores **74.9%** on the same forecast and the
same weeks. This scenario costs **-3.8 points** of accuracy across
the whole portfolio and moves bias by **-2.6%**.

### Where the damage actually is

An aggregate figure dilutes a localised failure. Scored only on
the two shocked categories, after the onset:

| Measure | Value |
|---|---:|
| Accuracy | **47.8%** |
| Bias | -26.8% |
| Interval coverage | 42.4% |
| Product-weeks affected | 1,880 |

That is the number to quote about this scenario. The portfolio figure
above is real, but it mostly measures how much of the assortment was
left alone.

## How this set was bent

`Bedding & Bath` ramps up to +80% and `Lighting` falls to -60%, both starting 2026-02-09 and building week by week. Every other category is left exactly as it happened.

## How it was built

The base is the project's **chronological holdout** - a year of real demand
the model never trained on, with the forecasts it genuinely made for those
weeks. `steady` is that period untouched; the other sets take the same real
base and bend it, so any difference from the control is caused by the stress
and nothing else.

Regenerate with `python scripts/09_generate_test_sets.py`.
