# The FORESIGHT model suite — five models, five failure modes

**Project FORESIGHT · NorthBay Living · Deliverable D3**

This document covers all five purpose-built models in the project: what each one
is for, how it works, and — separately and honestly — what each one was actually
measured to be worth on NorthBay's data.

Two of the five improve the headline forecast. Two do not, on this dataset, and
this document says so and explains why. One of those two produced a finding that
is arguably more useful than an accuracy gain would have been. Reporting that
plainly is the point: a suite of models that all claim to help is a suite nobody
has measured.

---

## 1. Why five models and not one

SKU-level demand forecasting does not fail in one way. It fails in several, and
the failures are independent — fixing one does nothing for the others. Each model
here exists for exactly one of them.

| # | Model | The failure it addresses | Where it acts |
|---|---|---|---|
| 1 | **Adaptive Demand Ensemble** | One estimator is forced on a catalogue containing four genuinely different demand regimes | The forecast itself |
| 2 | **Hierarchical Demand Reconciler** | SKU forecasts and the category plan are produced separately and contradict each other | Post-process across levels |
| 3 | **Cold-Start Propagation** | A newly launched product has no history, and every other method is built on history | Substitutes where history is absent |
| 4 | **Censored Demand Recovery** | Sales during a stockout are a lower bound on demand, not demand | The training target |
| 5 | **Promotional Response Decomposition** | A promoted week is a different process, not a bigger number | Target-week adjustment + reporting |

They are deliberately orthogonal. Model 1 works within a single SKU's time
series. Model 2 works across aggregation levels. Model 3 works in product
attribute space. Model 4 corrects the historical record before anything is
fitted. Model 5 works on the calendar of committed commercial activity. None of
them is a variation on another, and no two would be expected to help the same
row for the same reason.

### How they compose

```
   inventory snapshots
           |
           v
   [4] Censored Demand Recovery      corrects the target the others learn from
           |
           v
   weekly panel  ->  [1] Adaptive Demand Ensemble  ->  base SKU forecast
                              |                                |
        [3] Cold-Start  ------+ (where history is absent)       |
                                                                v
                              [5] Promotional Response  (target-week adjustment)
                                                                |
                                                                v
                              [2] Hierarchical Reconciler  (coherence across levels)
                                                                |
                                                                v
                                                        planning output
```

Model 4 runs first because it changes what "demand" means for everything
downstream. Models 2 and 5 are post-processes, which is what allows them to sit
on top of any base forecast rather than competing with it.

---

## 2. Model 1 — Adaptive Demand Ensemble

**Status: ships. This is the selected forecast.**

Full detail in [model_architecture.md](model_architecture.md) and
[model_comparison.md](model_comparison.md); summarised here for completeness.

### What it does

Routes every (SKU, week, horizon) row to one of four demand regimes — steady,
intermittent, new, volatile — from origin-side features only, then blends three
components with weights fitted **per regime** by direct WAPE minimisation on a
validation slice held out from the tail of the training window.

The three components:

1. **GBM** — global LightGBM over lag, rolling, calendar and price features.
2. **TSB** (Teunter–Syntetos–Babai) — intermittent smoothing, tracking demand
   size and sale probability as separate states. Chosen over Croston because
   Croston is positively biased on intermittent series, and a biased forecast
   feeding a reorder decision produces systematic overstock.
3. **Seasonal profile** — the SKU's own deseasonalised level re-seasonalised by
   its *category's* pooled week-of-year shape, so sparse series borrow the
   seasonal form of the products beside them.

### Measured result

Six rolling origins, 9,360 out-of-sample observations, all candidates scored on
identical folds and identical rows.

| Forecast | WAPE | Accuracy | vs seasonal-naive |
|---|---|---|---|
| Seasonal-naive (contractual baseline) | 0.3450 | 65.50% | — |
| Naive last-value | 0.3480 | 65.20% | −0.9% |
| Global LightGBM | 0.2301 | 76.99% | +33.3% |
| **Adaptive Demand Ensemble** | **0.2277** | **77.23%** | **+34.0%** |

The ensemble beats the baseline in **6 of 6 folds**. Relative bias is −0.68%,
against −0.99% for the GBM alone.

### The contribution that is easy to miss

Interval calibration. The raw GBM's 80% prediction intervals covered only
**68.1%** of actuals — a planner setting safety stock from them would be short
roughly one week in three while believing they were covered one week in five.
Split-conformal widening brings the ensemble to **81.6%** coverage against an
80% target.

For an inventory decision this matters more than the point forecast. A point
forecast that is 2% better changes an order quantity slightly; an interval that
is honest about its own uncertainty changes whether the safety stock is right at
all.

### Feature scope: what the extra data is worth

The brief's Appendix A specifies four extracts. Optional commercial driver data
(media spend, storefront analytics, market conditions) lives in a separate
package, `foresight_drivers`, and can be switched off entirely.

| Feature set | Features | WAPE | Accuracy | vs baseline |
|---|---|---|---|---|
| Four Appendix A tables only | 44 | 0.2380 | 76.20% | +31.0% |
| + commercial drivers | 60 | 0.2277 | 77.23% | +34.0% |

The 16 driver features buy **4.3% relative WAPE reduction — 1.03 accuracy
points**. Real, and worth having, but the honest headline is the other way
round: roughly **99% of the achievable accuracy is reachable inside the brief's
own scope**. Nobody needs to go and buy data to make this work.

---

## 3. Model 2 — Hierarchical Demand Reconciler (MinT)

**Status: ships, and correctly declines to act on this dataset. Read on — this
is the most interesting result in the suite.**

### The idea

Forecast the SKU series, the subcategory series, the category series and the
total *independently*, then combine all of them into one coherent set. With `S`
the summing matrix and `y_hat` the base forecasts, any coherent reconciliation is
a linear map `y_tilde = S P y_hat`. Wickramasuriya, Athanasopoulos & Hyndman
(2019) show the unbiased map minimising total reconciled error variance is:

```
P = (S' W^-1 S)^-1 S' W^-1
```

where `W` is the covariance of base forecast errors. This is *Minimum Trace*
reconciliation — optimal rather than heuristic.

The hierarchy here is 231 nodes: 1 total, 6 categories, 24 subcategories, 200
SKUs.

### The premise, and why it was wrong

The motivating observation was striking. Measured on the same model and the same
data:

| Level | Base forecast WAPE |
|---|---|
| SKU | 0.2465 |
| Subcategory | 0.1484 |
| **Category** | **0.1131** |
| Total | 0.1724 |

Category-level forecasts appeared to be more than twice as accurate as SKU-level
ones. The obvious inference is that aggregate forecasts carry information the SKU
forecasts lack, and reconciling toward them should help.

**That inference is wrong, and the model was built to test it rather than assume
it.** Summing `k` roughly independent forecasts shrinks *relative* error by about
`sqrt(k)` for free. A category containing 33 SKUs would show a lower WAPE than
its members even if the category forecast were nothing more than the sum of them.
The table above is consistent with the aggregates containing genuinely extra
information, and equally consistent with them containing none at all.

So the project measures the question directly: **does an independently-forecast
aggregate beat the free alternative of simply summing the SKU forecasts?**

| Level | Direct forecast | Bottom-up sum | Direct advantage |
|---|---|---|---|
| Category | 0.0917 | **0.0801** | **−14.5%** |
| Subcategory | 0.1435 | **0.1260** | **−13.9%** |
| Total | 0.0911 | **0.0555** | **−64.1%** |

The answer is unambiguous. The aggregate models are **worse** than free
summation, by 14% at category level and 64% at the total. Every bit of the
apparent category-level advantage was the `sqrt(k)` artefact. There is no extra
information to borrow, and reconciling toward these aggregates can only add
noise.

### What the model does about it

Three layers of defence, each added in response to a measured failure rather than
anticipated in the abstract:

1. **Matched residuals.** `W` must describe the errors of the forecasts actually
   being reconciled. Harvesting residuals with a plain GBM and then reconciling
   *ensemble* forecasts optimises for the wrong error structure — measured cost:
   the reconciliation went from −5.5% to roughly neutral once this was fixed.

2. **Honest correlation.** Missing residuals are not zero-filled. Products launch
   part-way through history, so unlaunched SKUs are missing at *the same*
   origins; zero-filling makes them move together and manufactures a strong,
   entirely fictitious correlation, which the shrinkage estimator then reads as
   signal. Thin nodes get an identity row instead. Measured effect: shrinkage
   intensity moved from an implausible 0.099 to 0.152–0.218.

3. **A structural gate.** Reconciliation is blended with plain bottom-up at
   strength `lambda`, chosen on held-out origins. Because *both* projections
   satisfy `P S = I`, every value of `lambda` yields a forecast that is still
   exactly coherent and still unbiased — only the variance changes. That is what
   makes the blend legitimate rather than a fudge. `lambda` is then chosen with a
   one-standard-error rule, and gated on the aggregate-advantage test above: if
   the aggregates do not beat their bottom-up sum, `lambda` is forced to zero and
   the SKU forecasts pass through untouched.

On NorthBay's data the gate fires. Measured on the same six folds and 9,360
rows as everything else:

| Forecast | WAPE | Accuracy | Bias |
|---|---|---|---|
| Adaptive Ensemble (input) | 0.227749 | 77.23% | −0.0068 |
| **After reconciliation** | **0.227749** | **77.23%** | **−0.0068** |

Gain: **exactly 0.0%**. Chosen `lambda`: **0.0** at every horizon and every fold.
Aggregate advantage measured on the validation slice: **−19.5%**, so the gate
records `declined: aggregate forecasts do not beat their bottom-up sum`. Maximum
coherence error across all origins and horizons: **0.0**.

**The reconciler leaves the forecast exactly as it found it** — not
approximately, identically — while still guaranteeing the output is coherent.

### So what is it worth?

Three things, none of them an accuracy gain:

1. **Exact coherence, guaranteed.** Maximum aggregate-versus-children mismatch
   across every origin and horizon: **0.0**. The SKU buy plan sums exactly to the
   subcategory plan, which sums to the category budget, which sums to the total.
   Planners currently reconcile that by hand in spreadsheets. It is a real
   operational deliverable regardless of WAPE.

2. **A model that can decline to fire.** The gate is the deliverable as much as
   the algebra is. It converted a technique that was *actively costing 5.5%
   accuracy* into one that provably cannot do harm, and it did so on evidence
   rather than on judgement.

3. **A reusable finding.** "Do our aggregate forecasts beat summation?" is now a
   measured diagnostic that re-runs on every refresh. NorthBay's demand shows
   little cross-SKU correlation today. If that changes — shared supply shocks,
   substitution between products, category-wide promotions, a supplier
   disruption — the correlation appears in `W`, the gate opens, and the same code
   starts contributing without anyone rebuilding it.

**Recommendation:** ship it, with the gate. Do not report an accuracy improvement
from it, because there is not one.

---

## 4. Model 3 — Cold-Start Propagation

**Status: ships as a fallback for products with no history. Not an accuracy
improvement for young-but-observed SKUs, and not claimed as one.**

### What it does

Section 16.2 of the brief names the problem: a SKU launched inside the forecast
window has no history, and every other technique here is built on history.

A new product has no history but is not information-free. Its category,
subcategory, price and margin are known, and dozens of similar products have
already launched. The forecast is assembled from three separately-estimated
pieces:

```
forecast = scale  x  launch curve(age)  x  seasonal index(week, category)
           ^^^^^     ^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           product   launches in this      where the target week falls
           property  category              in the year
```

Estimating them separately matters: a product that launched into December would
otherwise have the seasonal lift baked into its "scale" permanently.

**Measured launch curve** (share of eventual mature weekly rate):

| Weeks since launch | 1 | 4 | 13 |
|---|---|---|---|
| Share of mature rate | **0.23** | **0.54** | **1.03** |

A product sells under a quarter of its eventual rate in week one and takes about
three months to reach it. Fitted from 54 donor products, with 4 categories having
enough launches for their own curve.

**Price elasticity of mature volume: −0.155**, fitted within category and
constrained non-positive — a dearer product selling *more* is a sampling artefact,
and letting it through would recommend raising prices to grow volume.

### Credibility, not a cutoff

The obvious design is a switch: donors below 13 weeks, own data above. That puts
a cliff in the middle of the plan — a product can jump by a third the week it
crosses the threshold, for no reason anyone can explain to a buyer.

Instead the estimates blend by an actuarial credibility weight
(Bühlmann–Straub): `Z = n / (n + k)`, with `n` the weeks observed and `k = 8`.
At launch `Z = 0` and the forecast is entirely borrowed; every week moves weight
onto the product's own record; by a year it borrows nothing measurable. The model
dissolves into the standard pipeline instead of handing over abruptly, and there
is no threshold to tune or argue about.

One correction that is easy to get wrong: a young SKU's observed mean is a sample
from the *steep* part of the ramp, not its mature level. Blending it directly
against donors' mature levels would under-forecast every new product by the depth
of its own launch curve. The observed mean is therefore divided by the average
curve value over exactly the ages observed, converting it to a mature-equivalent
scale before the two estimates ever meet.

### Measured result — and the honest scope

Broken out by age, so the claim is scoped to the population it holds for rather
than averaged across products whose own lags are already informative:

| Age at forecast | Rows | Units | Cold-start | Ensemble | GBM |
|---|---|---|---|---|---|
| 0–4 weeks | 120 | 1.4% | 0.8277 | 0.4210 | **0.3844** |
| 5–8 weeks | 120 | 1.4% | 0.7332 | 0.3100 | **0.3097** |
| 9–13 weeks | 160 | 2.8% | 0.6131 | 0.1779 | **0.1776** |
| 14–26 weeks | 224 | 3.6% | 0.5896 | **0.2608** | 0.2608 |
| 27+ weeks | 8,736 | 90.9% | 0.4180 | **0.2237** | 0.2269 |

**Cold-start loses to the GBM at every age, including the youngest band.** That
is the measured result and it is stated without softening. The suite does not
route young SKUs to it.

Two things explain it, and both are worth knowing:

* **The GBM is already good at new products here.** It sees
  `weeks_since_launch` as a feature and has pooled launch behaviour across the
  whole catalogue, so it reconstructs much of what cold-start reconstructs
  explicitly — but with 60 features rather than three factors. The `new` regime
  is simply not a weak spot in this dataset.
* **Cold-start is deliberately simple.** It is level × ramp × season, and its
  0.418 on mature SKUs is about what a three-factor level model should score.
  That simplicity is the point for its intended case, and a liability everywhere
  else.

So its scope is narrow and specific: a product with **zero** weeks of history.
There the GBM's lag, rolling and trend features are all null, and it returns
whatever its mostly-missing feature row happens to land on — a number with no
traceable justification. Cold-start returns a defensible one built from named
donor products and a measured launch curve.

That case cannot appear in a backtest at all: a product with no history has no
panel rows to score. Its absence from the table above is not an oversight; it is
the reason the model exists, and the reason it cannot be justified by a benchmark.

**Recommendation:** ship it as an explicit zero-history fallback and as the
source of the launch-curve and elasticity estimates, which are useful in their
own right for planning a launch buy. Do not route observed SKUs to it, and do not
claim an accuracy improvement — there is none.

---

## 5. Model 4 — Censored Demand Recovery

**Status: ships. Active on 632 SKU-weeks.**

### The failure it prevents

Every other model learns from the sales ledger. But a sales ledger does not
record demand — it records `min(demand, what was on the shelf)`. Those are the
same number only while stock lasts, and they diverge precisely when demand was
highest, because that is when stock runs out.

Left uncorrected this closes a loop that worsens every cycle:

```
stockout -> recorded sales fall -> forecast falls -> reorder falls -> stockout sooner
```

The forecast becomes a self-fulfilling prophecy and the SKU is quietly demoted.
Nothing in the reporting shows it, because from the model's point of view its
predictions look *more* accurate each cycle — it is successfully predicting its
own past decisions.

### How it works

1. **Availability.** Inventory snapshots give opening on-hand; differencing
   successive snapshots against sales recovers receipts, and hence what could
   have been sold at all. Assuming demand arrives evenly through the week — the
   standard assumption, and the honest one absent intra-week data — a product
   with `A` units available against expected demand `d` is in stock for
   `min(1, A/d)` of that week.

2. **Inversion.** If only `a` of the week was sellable, the `s` units recorded
   represent `s/a` of underlying demand. When `a` reaches zero nothing could
   sell, `s/a` says nothing, and the estimate falls back to the product's own
   uncensored level scaled by season.

3. **Iteration.** Expected demand is needed to estimate availability and vice
   versa. Alternating from uncensored weeks only is a small EM; it converges
   quickly because the uncensored majority never moves. Three passes.

**Two guardrails**, because a runaway imputation is worse than none:
recovered demand is never below observed sales, and uplift is capped at 5×.

### Measured result

| Metric | Value |
|---|---|
| SKU-weeks examined | 33,366 |
| Censored weeks detected | **632 (1.89%)** |
| Total blackout weeks (nothing sellable) | 55 |
| SKUs affected | 100 |
| Observed units | 2,463,717 |
| Recovered units | 2,468,331 |
| Demand recovered | **+0.19%** |
| Mean uplift on a censored week | 7.3 units |

The raw extract shows 4,247 SKU-weeks opening at zero on-hand — 11% — but most of
those received stock mid-week and were not censored at all. Counting them would
have overstated the problem five-fold; recovering receipts from the snapshot
differences is what distinguishes the two.

### How this is evaluated honestly

Scoring a forecast against an imputed target would be circular. So the two uses
are separated:

* **Training** uses recovered demand, because true demand is what the business
  needs predicted.
* **Scoring** uses only weeks that were *not* censored, where the observation is
  the truth and no imputation is involved.

The imputation therefore cannot flatter any reported number in this project.

**Recommendation:** ship it. The measured effect on this history is small
(+0.19% of units) because NorthBay is well stocked. Its value is as a
structural guard: the death-spiral it prevents is slow, invisible in ordinary
reporting, and expensive once established. On a client with genuine service-level
problems the same machinery would be doing considerably more work.

---

## 6. Model 5 — Promotional Response Decomposition

**Status: ships for scenario planning and reporting.**

### What it does

A promoted week is a different process: price sensitivity invisible at list price
becomes the dominant driver, and demand is pulled forward from surrounding weeks.
A model fitted across blended history learns the average of those states — it
under-predicts the peak, because most weeks are not promoted, and over-predicts
the recovery, because it cannot know a promotion just ended.

Three pieces, each estimated separately:

* **Baseline** — each SKU's trailing median over its recent *unpromoted* weeks,
  rescaled by its category's week-of-year index. Trailing, never centred: a
  centred window would read across the promotion it is supposed to be the
  counterfactual for, and absorb the very uplift being measured.
* **Uplift** — `log(units / baseline)` regressed on discount depth, pooled within
  category. The slope is constrained non-negative.
* **Dip** — the mean ratio of observed to baseline in the weeks immediately after
  a promotion ends.

### Measured result

Fitted on **5,363 promoted SKU-weeks**, with all 6 categories having enough
observations for their own curve.

| Quantity | Value |
|---|---|
| Median discount depth in history | 10.3% |
| Median observed uplift | **2.0×** |
| Pooled discount slope | 3.87 |
| Fitted uplift at 20% off | **2.82×** |
| Fitted uplift at 40% off | **6.12×** |
| Post-promotion dip factor | **1.00 (none detected)** |

Per-category slopes range from 3.67 (Bedding & Bath) to 4.45 (Storage &
Organisation) — Storage responds roughly 20% more strongly to the same discount
than Bedding does, which is directly actionable when allocating promotional
budget.

**The dip machinery found no pull-forward in this history** (factor 1.00 across
every category). That is reported as measured rather than quietly dropped: on
this data, post-promotion weeks recover to baseline immediately. The estimator
remains in place and will pick pull-forward up on a client whose history contains
it.

### What this adds over the GBM's own promo features

The GBM already sees whether a promotion is scheduled for the target week and how
deep the published discount is. This model is not a replacement, and does not
claim to beat it on WAPE. It adds four things a tree ensemble structurally
cannot:

1. **Extrapolation.** A tree cannot predict outside the discount depths it was
   trained on. Asked about a 40% event when history tops out around 25%, it
   returns its 25% answer. The parametric curve gives a defensible number past
   the edge of the data — exactly the range where decisions are most expensive
   and least informed.
2. **The post-promotion dip.** Pull-forward requires knowing a promotion *just
   ended* — a property of the week before the target, not the target week. It is
   measured here directly.
3. **Decomposition.** The output is three numbers, not one: *"you would have sold
   400; the promotion adds 260."* That is a sentence a merchandiser can act on
   and challenge. A single blended forecast is not.
4. **Scenario planning.** "Should we run 20% or 30%?" requires a response curve.
   Now there is one.

**Recommendation:** ship it as a planning and reporting layer. Its value is
decision support, not WAPE.

---

## 7. The honest scorecard

| # | Model | Improves headline accuracy? | What it actually delivers |
|---|---|---|---|
| 1 | Adaptive Demand Ensemble | **Yes — +34.0% vs baseline** | The shipped forecast, plus honest 81.6% interval coverage |
| 2 | Hierarchical Reconciler | **No — correctly declines** | Exact coherence (error 0.0); a gate that prevents a −5.5% loss; a reusable diagnostic |
| 3 | Cold-Start Propagation | **No — loses to the GBM at every age** | A defensible forecast where the GBM has no inputs at all; a measured launch curve and price elasticity |
| 4 | Censored Demand Recovery | Marginally (+0.19% of units) | Prevents the stockout death spiral; corrects 632 weeks |
| 5 | Promotional Response | Not measured on WAPE | Discount elasticity, uplift decomposition, scenario planning |

Two of five improve the number. That is not a disappointing result — it is what
measuring honestly looks like. The reconciler in particular would have shipped as
a 5.5% *regression* dressed up as sophistication if the aggregate-advantage test
had not been run, and the test itself is now a permanent part of the refresh.

### Accuracy at the level decisions are actually made

WAPE at SKU-week is the hardest possible framing and the one this project reports
as its headline. Planners do not commit at SKU-week:

| Aggregation | Accuracy |
|---|---|
| SKU × week (the headline) | 77.2% |
| SKU × month | 86.8% |
| **SKU × full 8-week horizon** | **89.1%** |
| Category × week | 92.2% |
| Portfolio × week | 94.7% |

A buy decision covers a SKU across the whole lead time, which is the third row:
**89.1%**.

---

## 8. The leakage contract, shared by all five

Every model obeys the same rule. Features are either history-derived (functions
of the SKU's own past, evaluated at and inclusive of the origin week) or
forward-known (attributes of the target week taken from a published planning
artefact — the promo calendar, the committed media plan). Nothing else is
permitted.

This is enforced empirically, not by code review. `assert_no_leakage` rebuilds
every feature against a copy of the panel whose future has been overwritten with
noise, and fails the build if a single feature value at or before the cutoff
moves. It corrupts **12 observed columns** — demand, price, promo days, sessions,
cart adds, realised discount, media spend, impressions, email sends, competitor
index, weather anomaly, revenue — and checks all **58 numeric features** across
180,192 comparable rows. Forward-committed plans are deliberately *not* corrupted,
because being knowable in advance is exactly what qualifies them.

A model that cannot pass it never gets trained. The guard caught one real leak
during development, in the promotional baseline's fallback path.

Model-specific contracts:

* **Model 1** — TSB consumes demand only up to the origin; the category seasonal
  profile is fitted on training-window targets; blend weights are fitted on a
  validation slice ending at the training cutoff.
* **Model 2** — residuals defining `W` are harvested strictly out-of-sample, by
  training once at a cutoff and scoring only origins after it.
* **Model 3** — the launch library, price elasticity and seasonal index are fitted
  on the training window; the per-SKU observed level at origin `t` reads only
  weeks up to and including `t`.
* **Model 4** — availability uses only inventory snapshots and sales; the recovery
  is applied to the historical record and never to a target week's outcome.
* **Model 5** — the baseline is trailing-only; at forecast time the sole
  promotional input is the *published* discount for the target week.

---

## 9. What would change on client data

The two models that decline to act here are not dormant by design — they are
gated on conditions this dataset does not meet.

* **Model 2 would activate** if cross-SKU demand correlation rises: shared supply
  shocks, substitution between products, category-wide events, a supplier
  disruption. All of these show up in `W`, the aggregate-advantage test flips
  positive, the gate opens, and the same code starts contributing. No rebuild.
* **Model 3 matters more** the faster the assortment turns over. NorthBay
  launched roughly 10% of its catalogue in the final year. A brand launching 40%
  a year would find cold-start carrying a large share of its planning volume.
* **Model 4 matters more** the worse service levels are. At 1.9% censoring its
  effect is small; at 10% it would be one of the largest single corrections
  available.
* **Model 5's dip** is currently 1.00. Categories with heavy stock-up behaviour —
  bulk consumables, gifting around festivals — typically show real pull-forward,
  and the estimator will report it when present.

---

## 10. Reproducing every number here

```bash
python scripts/00_generate_data.py --force
python scripts/01_run_pipeline.py

# Model 1, both feature scopes
python scripts/03_train_backtest.py --no-drivers --tag basetables
python scripts/03_train_backtest.py --tag drivers

# Models 2 and 3, scored on identical folds
python scripts/08_hierarchy_backtest.py
```

Outputs land in `artifacts/`: `metrics_basetables.json`, `metrics_drivers.json`,
`hierarchy_metrics.json`. Models 4 and 5 are exercised by the test suite and by
the pipeline's cleaning report. Every figure in this document is read from those
files; none is quoted from memory.
