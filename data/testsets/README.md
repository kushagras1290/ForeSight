# Test datasets

Three independent test sets for the forward forecast, each answering a
different question. Upload `actuals.csv` and `forecast.csv` together on the
dashboard's *Forecast accuracy* tab.

| Set | Question | Accuracy | Bias | Coverage |
|---|---|---:|---:|---:|
| [`steady`](steady/README.md) | Does the reported accuracy actually transfer? | **74.9%** | -7.9% | 78% |
| [`shock`](shock/README.md) | How badly does it fail when the world changes underneath it? | **71.0%** | -10.4% | 71% |
| [`promo`](promo/README.md) | Does it call the peak when a third of the range goes on offer? | **70.5%** | -13.8% | 75% |

The spread between them is the point. A forecast that scores the same on
all three is not being tested hard enough; one that collapses on `shock`
is behaving exactly as a history-based model should, and the size of the
collapse is what tells you how much headroom to leave when planning.

Regenerate with `python scripts/09_generate_test_sets.py`.
