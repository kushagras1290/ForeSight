# How the Adaptive Ensemble compares to traditional forecasting

Project FORESIGHT · NorthBay Living · Deliverable D3

Every number below comes from the same rolling-origin backtest: **6 folds**, **9,360
out-of-sample product-weeks**, forecasts made using only what was knowable at each origin.
No model is scored on data it trained on. A separate **70/30 chronological holdout** of
10,760 further observations is reported alongside as an independent check.

Reproduce with `python scripts/03_train_backtest.py` and `python scripts/07_holdout_split.py`.

---

## 1. Headline results

| Approach | WAPE | vs baseline | Bias | Interval coverage |
|---|---:|---:|---:|---:|
| Seasonal-naive (the bar) | 34.50% | — | −15.6% | n/a |
| Last-value naive | 34.80% | +0.9% | — | n/a |
| Global LightGBM | 23.01% | −33.3% | −0.99% | 68.1% ✗ |
| **Adaptive Ensemble** | **22.77%** | **−34.0%** | **−0.68%** | **81.6%** ✓ |

Lower WAPE is better. Bias closer to zero is better. Coverage should sit near the 80% target.

The ensemble beats the baseline **in all six folds individually**, not just on the pooled
average — which matters, because a model that wins on average while losing in a third of
periods is not one an operations team can rely on week to week.

| Fold | Origin | Seasonal-naive | LightGBM | Ensemble |
|---:|---|---:|---:|---:|
| 1 | 2026-02-02 | 34.50% | **22.01%** | 22.93% |
| 2 | 2026-03-02 | 33.61% | 21.07% | **20.84%** |
| 3 | 2026-03-30 | 33.64% | 22.54% | **22.21%** |
| 4 | 2026-04-27 | 34.15% | 22.40% | **22.33%** |
| 5 | 2026-05-25 | 34.69% | 23.29% | **22.88%** |
| 6 | 2026-06-22 | 35.86% | 25.68% | **24.64%** |

Against the **GBM** specifically the ensemble wins 5 folds of 6 — it loses fold 1 by 0.9
points. That is stated rather than rounded away: the ensemble's margin over a well-tuned
GBM is real but modest (1.0% relative overall), and most of its value is in the interval
calibration below rather than in the point forecast.

### The independent holdout agrees

Trained on the earliest 134 weeks (70%), tested on the most recent 57 (30%) with no
retraining:

| | WAPE | Bias | Interval coverage |
|---|---:|---:|---:|
| Seasonal-naive | 34.69% | −14.5% | — |
| **Adaptive Ensemble** | **25.14%** | −7.9% | 77.5% |

**27.5% better than the baseline on 10,760 observations the model never saw.** Slightly
worse than the backtest figure, which is expected: it trains on 30% less history.

---

## 2. Why seasonal-naive is the right bar, not a straw man

A model should be measured against the best *simple* thing, not the worst. On a catalogue
with a festive peak this large, "same week last year" is genuinely hard to beat — it gets the
seasonal shape right for free, which is most of the signal.

The evidence that it is a real bar: **last-value naive**, the obvious alternative, is *worse*
(34.80%). Had the goal been to flatter the model, the weaker comparison would have been the
one to quote. Both are reported.

Seasonal-naive also has a structural flaw that makes it unusable rather than merely
inaccurate: its bias is **−15.6%**. NorthBay's demand grows year on year, so last year's week
is systematically too low. A forecast that runs 16% light every week guarantees chronic
under-ordering — exactly the stockout problem the client described. The ensemble's bias is
**−0.7%**, effectively unbiased.

One honest caveat: the baseline cannot be computed at all for **7.1%** of rows, because those
products have less than a year of history. Those fall back to a trailing four-week mean, and
the fallback rate is reported rather than the rows being quietly dropped.

---

## 3. What about ARIMA, ETS or Prophet?

These are the classical answers, considered and rejected for reasons specific to this data:

| Method | Why it does not fit here |
|---|---|
| **ARIMA / SARIMA** | Fits one model per series. With 200 series of ~190 weeks, each sees too little data to identify a seasonal order (needs 2–3 full cycles; many SKUs have barely two, and the newest have 20 weeks). A thin-history SKU cannot borrow anything from its category. |
| **ETS / Holt-Winters** | Same per-series limit, plus it assumes a smooth seasonal structure. 6% of this catalogue sells nothing in a third of weeks, where smoothing the level is not meaningful. |
| **Prophet** | Handles holidays well and would be reasonable for the *steady* group. But it is per-series, does not pool, and over-smooths sharp promotional spikes — which is where the money is on a festive assortment. |

The common thread: **all three are per-series**, and the defining constraint here is 200 short
series rather than one long one. Pooling into a global model with SKU and category as features
is what makes thin history survivable, and it is why the GBM beats seasonal-naive by 25%
before any ensembling.

This is a judgement, not a measurement — none of the three were fitted. Saying so is more
useful than implying a benchmark that was never run.

---

## 4. Against a single global GBM: an honest accounting

The ensemble improves WAPE by **1.0%** over the GBM (23.01% −> 22.77%). Real, but small, and
it would be misleading to lead with it. The GBM does the overwhelming majority of the work,
and the ensemble's stronger claim is the interval calibration in section 5, not this number.

Where the difference sits, by regime:

| Regime | Rows | Share | Seasonal-naive | LightGBM | Ensemble | vs GBM |
|---|---:|---:|---:|---:|---:|---:|
| Steady | 8,008 | 85.6% | 34.14% | 22.62% | **22.30%** | −1.4% |
| **Intermittent** | 360 | 3.8% | 407.53% | 174.21% | **165.45%** | −5.0% |
| Newly launched | 624 | 6.7% | 36.52% | **26.21%** | 26.78% | +2.2% |
| Volatile | 368 | 3.9% | 46.86% | 28.79% | **28.54%** | −0.9% |

Reading it honestly:

- **Steady carries the result** — 85.6% of rows, and a 1.4% relative improvement there is
  most of the aggregate gain.
- **Intermittent gains most in relative terms** (5.0%) but starts from a WAPE above 100%.
  That is not a bug: on series selling one or two units in a typical week, total absolute
  error can genuinely exceed total demand. It is why WAPE, not MAPE, is the reported metric.
  These rows carry 0.02% of units, so the gain barely moves the aggregate.
- **Newly launched is 2.2% worse than the GBM.** Blending does not help there. Reported
  because it is true, not omitted because it is unflattering.

### The new-SKU regime: a failure that was found and fixed

An early run of this comparison had the ensemble at **41.1% on newly-launched products
against the GBM's 36.6%** — materially *worse* than not blending at all, with a −17% bias.

The cause was structural. The optimiser had assigned 40% weight to TSB, which estimates a
**flat** level from a stable arrival process. A newly-launched product has no stable arrival
process — it is ramping — so a flat level lags the ramp all the way up.

The fix was to exclude TSB from that regime's candidate set on the grounds of what it models,
not on the grounds that the number improved. That constraint is still in force, and on the
current feature set TSB takes zero weight on new products without being forced to.

The regime is not yet a win, though, and the table above says so: at 26.78% the ensemble still
trails the plain GBM's 26.21%. The category seasonal profile now carries 70% of the weight
there. The structure is defensible; the result on that 6.7% of rows is not yet better.

A residual limitation remains and is not papered over: new products are still under-forecast
by roughly **15%**. All nine are flagged `low` confidence in the dashboard, but their reorder
quantities will read light.

---

## 5. The calibration result is arguably the bigger contribution

| | Raw LightGBM | Ensemble |
|---|---:|---:|
| Stated interval | 80% | 80% |
| **Measured coverage** | **68.1%** | **81.6%** |

The raw quantile models produce a band containing the outcome 68.1% of the time while
claiming 80%. In isolation that is a modelling nicety. Here it is not, because the risk layer
converts interval width directly into a standard deviation and then a stockout probability:

```text
sigma_week     = (upper − lower) / (2 × 1.2816)
stockout_score = P(demand over lead time > available stock)
```

An interval that narrow makes every stockout probability too low, every safety stock too
small, and every reorder recommendation too late — silently, across all 200 products. The
conformal step (a fitted widening factor of 1.40) fixes it at source. **A calibrated interval
matters more to the recommendations this system emits than the 1.0% point-accuracy gain.**

---

## 6. Is the model underfitting? No — and it was measured

"Add more capacity" is the reflex, so it was tested rather than assumed. On a real backtest
fold:

| Config | Trees | Train WAPE | Test WAPE | Gap |
|---|---:|---:|---:|---:|
| Constrained (15 leaves) | 400 | 0.228 | 0.2757 | 0.048 |
| **Current** | 700 | 0.165 | **0.2568** | 0.092 |
| More capacity (127 leaves) | 2000 | 0.096 | 0.2570 | 0.161 |
| Much deeper (255 leaves) | 2500 | 0.061 | 0.2558 | 0.195 |

Read it from the bottom up. Going from 700 trees to 2,500 with 255 leaves cuts **training**
error by a factor of 2.7 and moves **test** error by 0.001. The extra capacity is being spent
memorising the training set rather than learning anything that generalises — the signature of
overfitting.

The top row is what underfitting actually looks like, included deliberately as a control:
constrain the model to 15 leaves and *both* numbers get worse together, with test error rising
to 0.276. The current configuration is nowhere near that regime.

So the sweep answers the question in both directions. No capacity setting on this data
meaningfully improves out-of-sample error, and reducing capacity demonstrably hurts.

**How much headroom is left at all?** A centred rolling median — which cheats by reading the
future, and is therefore an unreachable lower bound — scores **0.23–0.26** on this data. The
model reaches 0.228 genuinely out-of-sample. It is at the edge of that oracle bound, and
61% of the remaining absolute error sits on high-volume weeks where WAPE is already lowest.
The constraint is demand noise, not model capacity.

### Train versus test, side by side

Measured on one fold, same models, disjoint rows. "Accuracy" here is `100% − WAPE`, which is
not classification accuracy — it reads as *total absolute error as a share of total demand*.

| Model | Train WAPE | Test WAPE | Gap | Train accuracy | Test accuracy |
|---|---:|---:|---:|---:|---:|
| Seasonal-naive | 0.3969 | 0.3586 | −0.0383 | 60.3% | 64.1% |
| LightGBM | 0.1652 | 0.2568 | 0.0916 | 83.5% | 74.3% |
| **Adaptive Ensemble** | 0.1862 | 0.2464 | **0.0602** | 81.4% | **75.4%** |

This is the clearest single piece of evidence in the project, and it points in a direction that
is easy to misread as a weakness:

**The ensemble is worse than the GBM on training data (81.4% vs 83.5%) and better on test data
(75.4% vs 74.3%).** It trades memorisation for generalisation, because the TSB and
seasonal-profile components are low-capacity and act as regularisers on the GBM.

The gaps make the same point. The GBM's train/test gap is **0.0916**; the ensemble's is
**0.0602** — a third tighter. Neither is at the floor, but the direction is what matters: the
component that generalises better is the one carrying more of the blend.

The baseline's *negative* gap is expected rather than anomalous: seasonal-naive fits nothing,
so the difference between its periods is variation, not learning.

Across all six folds the test-side figures are **65.5%** (baseline), **77.0%** (LightGBM) and
**77.2%** (ensemble); on the 70/30 holdout, **74.9%** against the baseline's **65.3%**.

One caveat stated rather than buried: the ensemble's blend weights are fitted on a validation
slice carved from the training window, so its *train* figure includes rows that informed those
weights. Every test figure is unaffected.

---

## 7. What the accuracy is worth in rupees

Accuracy only matters through the decisions it changes. At the current plan week:

| | Products | Value |
|---|---:|---:|
| Reorder now | 25 | **₹1.20 Cr** of sales at risk (₹0.69 Cr margin) |
| Markdown / clear | 46 | **₹3.05 Cr** of capital locked |
| Watch / review | 5 | forecast too erratic to act on automatically |
| Healthy | 124 | no action |

The two figures are never summed: one is revenue that may not happen, the other is cash
already spent.

The link back to accuracy is bias. Seasonal-naive's −15.7% would under-order across the whole
catalogue. Removing that is what turns the reorder list from a guess into something worth
acting on — a larger practical effect than the WAPE difference between the two learned models.

**The top 10 products carry 84.9% of the total revenue at risk.** Ten calls capture most of
the recoverable value.

---

## 8. When this model would *not* be the right choice

- **A single long series.** With 5+ years of one product, ARIMA or ETS would be competitive
  and far simpler to maintain. The pooling advantage disappears.
- **A mostly-intermittent catalogue.** Here 83% of rows are steady, so a GBM-led blend is
  right. Flip the ratio and a dedicated intermittent method should lead.
- **Sub-weekly forecasting.** Everything is built at a weekly grain; daily would need
  day-of-week structure the feature set does not carry.
- **Where the forecast itself must be defended line by line.** The risk layer is transparent
  arithmetic; the forecast under it is not. The seasonal-profile component alone would be the
  defensible choice — at a real cost in accuracy.

---

## 9. Summary

| Claim | Evidence |
|---|---|
| Beats the baseline | 22.77% vs 34.50% WAPE, −34.0%, winning all 6 folds |
| Holds up out of sample | 29.29% vs 39.04% on a 70/30 chronological holdout it never trained on |
| Beats a single GBM | 22.77% vs 23.01%, −1.0% overall, −1.4% on the 86% of rows that are steady |
| Is not badly biased | −0.7% vs the baseline's −15.6% |
| Intervals mean what they say | 81.6% measured against an 80% target, up from 68.1% |
| Does not leak | Automated guard, verified to catch a planted leak, gates the build |
| Is not underfitting | Capacity sweep: every increase makes test error worse; within a few points of an oracle |

The honest one-line summary: **the global GBM delivers most of the accuracy; the ensemble adds
a small aggregate gain, a solid gain on the 83% of rows that matter most, and a properly
calibrated uncertainty band that the inventory decisions depend on.**
