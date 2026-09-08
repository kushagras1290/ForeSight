# Test set: Business as usual

**Does the reported accuracy actually transfer?**

The chronological holdout exactly as it happened - a full year of real demand the model never trained on, against the forecasts it genuinely made for those weeks. Nothing is altered.

This is the control. It should reproduce the holdout figure, and if it does not, something is wrong with the measurement rather than with the world. It is also the reference the other two sets are read against: their drop from this number is caused entirely by the stress applied, because everything else is identical.

## Files

| File | What it is |
|---|---|
| `actuals.csv` | What demand turned out to be. |
| `forecast.csv` | The forecast under test, so the pair scores end to end. |

Upload **both files together** on the dashboard's *Forecast accuracy* tab.

## What it scored

| Measure | Value |
|---|---:|
| Accuracy | **74.9%** |
| WAPE | 0.2514 |
| Seasonal-naive WAPE | 0.3444 |
| Bias | -7.9% |
| Interval coverage | 77.5% (80% stated) |
| Product-weeks scored | 10,760 |

## How it was built

The base is the project's **chronological holdout** - a year of real demand
the model never trained on, with the forecasts it genuinely made for those
weeks. `steady` is that period untouched; the other sets take the same real
base and bend it, so any difference from the control is caused by the stress
and nothing else.

Regenerate with `python scripts/09_generate_test_sets.py`.
