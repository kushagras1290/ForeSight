# Explaining the FORESIGHT codebase

**A speaking guide.** This is written so you can read it once and then confidently
walk anyone through the system — a reviewer, a teammate, an interviewer, a client.

It gives you, for each part: what it does, **the sentence to say about it**, and
the reasoning to fall back on if you get pushed. Section 8 has the hard questions
and the answers.

---

## 1. The 60-second version

> "FORESIGHT forecasts weekly demand for 200 SKUs eight weeks ahead, then turns
> those forecasts into reorder and markdown decisions with a rupee value attached.
>
> The forecast is 77.2% accurate at SKU-week level, against 65.5% for the
> seasonal-naive baseline the brief specifies — a 34% reduction in error, winning
> in all six backtest folds.
>
> There are five purpose-built models, each targeting a different way demand
> forecasting fails. Two of them measurably improve accuracy. Two of them don't,
> on this data — and the project says so, with numbers, instead of quietly
> shipping them."

That last sentence is the one that lands. Lead with it if the audience is
technical.

---

## 2. The shape of the repo — what to point at

```text
ForeSight/
├─ src/foresight/          the library — all logic lives here
├─ src/foresight_drivers/  optional extension for extra commercial data
├─ scripts/                00–08, numbered in run order
├─ service/                FastAPI app
├─ dashboard/              React + Vite planning UI
├─ reports/                EDA memo, model docs, executive readout
├─ artifacts/              trained models, metrics, forecasts  (generated)
└─ data/                   raw / interim / processed  (generated)
```

**Say this:**

> "Two rules govern the layout. First, the scripts contain no logic — they parse
> arguments, load data, call one or two library functions, and write results.
> That's what makes the library testable without running the pipeline.
>
> Second, the core library never imports the driver extension at module level.
> The extra commercial data is genuinely optional, so the core has to run without
> it. Where it's needed, the import is inside a function behind a flag."

---

## 3. The data flow — tell it as a story

```text
data/raw/*.csv                     7 client extracts
      │
      ▼  ingest.py                 read + validate against declared schemas
      ▼  clean.py                  11 documented cleaning steps, each logged
      ▼  pipeline.py               daily grid → weekly panel          (D1)
data/processed/weekly_panel.parquet    ← the spine of everything
      │
      ├─▶ eda.py                   data-quality memo + figures        (D2)
      ▼  features.py               leak-safe features at each origin
      ▼  backtest.py               rolling origin, 6 folds            (D3)
      ▼  risk.py                   stockout / overstock → ₹ impact    (D4)
artifacts/                         models, metrics, forecast, risk table
      │
      ▼  service/ ──────▶ dashboard/                                  (D5, D6)
      ▼  readout.py                executive PDF                      (D7)
```

**Say this:**

> "Everything hangs off one table: the weekly panel, one row per SKU per week.
> Raw extracts get validated, cleaned, and collapsed into it. Every model reads
> it. If a number ever looks wrong, you open that parquet file and look.
>
> The important thing about the cleaning layer is that every step records *why*
> it changed something, not just what. That record becomes the data-quality memo
> — so the memo can't drift out of sync with the code, because it's generated
> from it."

**If asked "what was actually wrong with the data?"** — the pipeline caught 1,389
injected duplicate rows: 453 byte-identical, and 936 key collisions that were
being silently *summed* into inflated demand. The fix was semantic deduplication
before key aggregation. That one ordering decision is worth mentioning; reversing
those two steps double-counts.

---

## 4. The five models — one sentence each

This is the centrepiece. Explain the *framing* first, then the models.

**Say this:**

> "SKU-level demand forecasting doesn't fail in one way. It fails in several, and
> fixing one does nothing for the others. So rather than one model, there are five
> — each targeting exactly one failure mode, and each reported with what it was
> measured to be worth."

| # | Model | The failure it addresses | The verdict |
|---|---|---|---|
| 1 | **Adaptive Demand Ensemble** | One estimator forced on four genuinely different demand regimes | **Ships — +34.0% vs baseline** |
| 2 | **Hierarchical Reconciler** | SKU forecasts and the category plan contradict each other | Ships, **declines to act**; guarantees exact coherence |
| 3 | **Cold-Start Propagation** | A new product has no history to forecast from | Ships as a zero-history fallback; **no accuracy gain** |
| 4 | **Censored Demand Recovery** | Sales during a stockout are a lower bound, not demand | Ships — corrects 632 SKU-weeks |
| 5 | **Promotional Response** | A promoted week is a different process, not a bigger number | Ships for scenario planning |

### Model 1 — Adaptive Demand Ensemble (`custom_model.py`)

> "The catalogue isn't homogeneous. Some SKUs sell steadily every week, some sell
> nothing most weeks, some just launched, some are erratic. One model trained to
> minimise average error optimises for the fast movers and treats everything else
> as noise.
>
> So every row is routed to one of four regimes from origin-side features only,
> and three components are blended with weights fitted *per regime*: a LightGBM,
> a TSB intermittent-demand smoother, and a category seasonal profile.
>
> The detail I'd point to is TSB rather than Croston. Croston is known to be
> positively biased on intermittent series, and a biased forecast feeding a
> reorder decision produces systematic overstock. That's a rupee consequence, not
> a statistical nicety."

**The under-appreciated result:** the raw GBM's 80% prediction intervals covered
only 68% of actuals. Conformal calibration brought that to 81.6%. For inventory
that matters more than the point forecast — an interval that lies about its own
uncertainty makes the safety stock wrong.

### Model 2 — Hierarchical Reconciler (`reconcile.py`, `hierarchy.py`)

**This is the best story in the project. Take your time on it.**

> "The premise was that category-level forecasts are far more accurate than
> SKU-level ones — 0.113 WAPE against 0.247, measured. So reconciling SKU
> forecasts toward the category total should pull them right. That's MinT,
> minimum-trace reconciliation, and it's a well-established technique.
>
> But I tested the premise instead of assuming it, and it was false.
>
> Summing k roughly independent forecasts shrinks relative error by about √k for
> free. So a category made of 33 SKUs will *always* show a lower WAPE than its
> members — even if the category forecast is nothing more than their sum. The
> apparent advantage was an arithmetic artefact.
>
> The test that settles it: does an independently-forecast category beat simply
> adding up the SKU forecasts? Answer — no. It's 14.5% *worse* at category level
> and 64% worse at the total.
>
> Applied at full strength, reconciliation cost 5.5% accuracy. It now gates on
> that test, sets its blend weight to zero, and passes the forecast through
> untouched. Gain: exactly 0.0%."

**If asked "so why ship it at all?"** Three reasons:

1. **Exact coherence** — max mismatch 0.0. The SKU buy plan sums exactly to the
   category budget. Planners currently reconcile that by hand.
2. **A model that can decline to fire** is the deliverable. Without the gate this
   would have shipped as a 5.5% regression dressed up as sophistication.
3. **It re-arms automatically.** If cross-SKU correlation ever appears —
   substitution, shared supply shocks, category-wide promotions — the gate opens
   and the same code starts contributing. No rebuild.

**The engineering detail worth mentioning:** a subtle bug I hit was zero-filling
missing residuals. Products launch mid-history, so unlaunched SKUs are missing at
*the same* origins — zero-filling makes them move together and manufactures a
completely fictitious correlation, which the shrinkage estimator then reads as
real signal. Shrinkage intensity was an implausible 0.099. Giving thin nodes an
identity row instead moved it to a believable 0.15–0.22.

### Model 3 — Cold-Start Propagation (`coldstart.py`)

> "The brief names this in section 16.2: a SKU launched inside the forecast
> window has no history, and everything else here is built on history.
>
> A new product isn't information-free though — we know its category, price and
> margin, and dozens of similar products have already launched. So the forecast
> is scale × launch curve × seasonal index, each estimated separately. Estimating
> them jointly would confound them: a product launched into December would have
> the seasonal lift baked into its 'scale' permanently.
>
> The measured launch curve says a product sells 23% of its eventual weekly rate
> in week one, 54% by week four, and reaches its mature rate around week 13."

**The design detail to highlight** — credibility, not a cutoff:

> "The obvious design is a switch: donors below 13 weeks, own data above. That
> puts a cliff in the middle of the plan — a product can jump a third the week it
> crosses the threshold, for no reason you can explain to a buyer.
>
> Instead it blends by an actuarial credibility weight, Z = n/(n+k). At launch
> it's entirely borrowed; every week moves weight onto the product's own record.
> The model dissolves into the standard pipeline instead of handing over
> abruptly, and there's no threshold to argue about."

**Be honest about the result:** it loses to the GBM at every age band, including
0–4 weeks (0.83 vs 0.38 WAPE). Say so directly. The reason is that the GBM
already sees `weeks_since_launch` and has pooled launch behaviour across the
catalogue. Cold-start's real scope is a SKU with *zero* history — which can't be
backtested, because it has no rows to score. That's exactly why it needs a model
rather than a benchmark.

### Model 4 — Censored Demand Recovery (`censoring.py`)

**This one lands well with commercial audiences. It's a vivid failure.**

> "Every other model learns from the sales ledger. But a sales ledger doesn't
> record demand — it records the *minimum* of demand and what was on the shelf.
> Those are the same number only while stock lasts, and they diverge exactly when
> demand was highest.
>
> Left alone that closes a loop: stockout, recorded sales fall, forecast falls,
> reorder falls, stockout sooner. The forecast becomes self-fulfilling and the
> SKU gets quietly demoted — and nothing in the reporting shows it, because from
> the model's point of view its predictions look *more* accurate every cycle.
> It's successfully predicting its own past decisions.
>
> The fix recovers availability from inventory snapshots — differencing
> successive snapshots against sales recovers receipts — and inverts the
> censoring. 632 SKU-weeks corrected, across 100 products."

**The nuance that shows rigour:** the raw extract shows 4,247 SKU-weeks opening at
zero on hand — 11%. But most of those received stock mid-week and weren't
censored at all. Counting them would have overstated the problem five-fold.

**And the honesty control:** training uses recovered demand; *scoring* uses only
uncensored weeks. Scoring against your own imputation is circular.

### Model 5 — Promotional Response (`promotions.py`)

> "A promoted week is a different process. A model fitted across blended history
> learns the average of both states — it under-predicts the peak, because most
> weeks aren't promoted, and over-predicts the recovery.
>
> This decomposes demand into baseline, uplift, and the dip that follows. Fitted
> on 5,363 promoted SKU-weeks: 20% off gives a 2.8× lift, 40% off gives 6.1×.
> Storage & Organisation responds about 20% more strongly than Bedding & Bath to
> the same discount — which is directly actionable when allocating promo budget."

**If asked "doesn't the GBM already see the discount?"** Yes, and say so. Then:

> "It does, and this doesn't beat it on WAPE — I don't claim it does. It adds four
> things a tree structurally can't: extrapolation past the discount depths in the
> training data; the post-promotion dip, which depends on the week *before* the
> target; a decomposition a merchandiser can actually challenge — 'you'd have sold
> 400, the promo adds 260'; and scenario planning, because 20%-or-30% needs a
> response curve."

**Report the null result:** the dip came out at 1.00 — no pull-forward in this
history. Say that plainly rather than hiding it.

---

## 5. The four invariants — the "how do I know it's right" answer

If someone asks what stops this being wrong, these are the four things.

### 1. Features never read the future — and it's a build gate

> "Every feature is built at an origin week and used to predict forward. Only two
> families are allowed: functions of the SKU's own past, and forward-known
> planning artefacts like the published promo calendar. The *realised* promo flag
> from the sales ledger is deliberately excluded for the target week — that's an
> outcome, not a plan.
>
> This isn't enforced by code review. `assert_no_leakage` rebuilds every feature
> against a copy of the panel whose future has been overwritten with noise. If any
> value at or before the cutoff moves, it read the future. It corrupts 12 observed
> columns and checks all 58 features across 180,192 rows, and on failure **no
> model is trained and nothing is written**.
>
> It's verified to catch a planted leak, not just to pass. And it caught a real
> one during development — in the promotional baseline's fallback path."

### 2. The split filters on the target week, not the origin week

```python
train = supervised[supervised["target_week"] <= origin]   # correct
train = supervised[supervised["week_start"] <= origin]    # WRONG
```

> "This is the most common bug in forecasting codebases and it has no symptom —
> the score just gets better. A row with origin O−2 and horizon 8 has its target
> at O+6, six weeks past the split. Training on it teaches the model outcomes from
> after the forecast date. It lives in exactly one function so there's one place
> to get it wrong."

### 3. Train and serve share one contract

> "A fitted model carries its own feature spec, and serving code asks the *model*
> what it needs rather than reading configuration. Otherwise a model trained with
> the driver data can be served a frame built without it — which surfaces as a
> missing-column error if you're lucky and a silently wrong forecast if you're
> not."

### 4. Never score against an imputed value

> "Imputation may inform a model. It may never be a benchmark."

---

## 6. The numbers to have at hand

| Quantity | Value |
|---|---|
| Ensemble WAPE / accuracy | 0.2277 / **77.2%** |
| Seasonal-naive baseline | 0.3450 / 65.5% |
| Improvement | **+34.0%**, winning 6/6 folds |
| Out-of-sample observations | 9,360 across 6 rolling origins |
| Forecast bias | −0.68% (baseline −15.7%) |
| Interval coverage | 80% stated, **81.6%** measured |
| Accuracy at SKU × full 8-week horizon | **89.1%** |
| Accuracy at category × week | 92.2% |
| Four Appendix A tables only | 0.2380 / 76.2% |
| With commercial drivers | 0.2277 / 77.2% — **+1.03 points** |

**The line to use about the extra data:**

> "The optional commercial data buys about one accuracy point. Real, worth having,
> but the honest headline is the other way round — roughly 99% of the achievable
> accuracy is reachable inside the brief's own scope. Nobody needs to go and buy
> data to make this work."

**On aggregation, if someone says 77% sounds low:**

> "SKU-week is the hardest possible framing, and it's what I report as the
> headline deliberately. But nobody commits stock at SKU-week. A buy decision
> covers a SKU across the whole lead time — that's 89.1%. Category planning is
> 92.2%."

---

## 7. Where each deliverable lives

| # | Deliverable | Point at |
|---|---|---|
| D1 | Reproducible pipeline | `src/foresight/pipeline.py`, `clean.py` |
| D2 | Data-quality & EDA memo | `reports/eda_memo.md` |
| D3 | Backtested forecast | `forecast.py`, `custom_model.py`, `backtest.py` |
| D4 | Risk scoring & ₹ impact | `src/foresight/risk.py` |
| D5 | Planning dashboard | `dashboard/` |
| D6 | Scoring service | `service/` |
| D7 | Executive readout | `reports/executive_readout.pdf` |

**On the risk layer, say:**

> "It's deliberately transparent arithmetic rather than a second model. An ops
> manager has to be able to challenge a reorder recommendation and get a straight
> answer. Both axes are genuine probabilities on opposite tails — probability
> demand over the lead time exceeds stock, and probability demand over twelve
> weeks falls short of it."

---

## 8. Hard questions, and how to answer them

**"Two of your five models don't help. Why ship them?"**

> "Because measuring honestly is the point. The reconciler would have shipped as a
> 5.5% accuracy regression if I hadn't run the test — and the test is now a
> permanent part of the refresh. It also delivers exact coherence, which is a real
> operational win independent of accuracy, and it re-arms itself if the data ever
> develops the correlation structure it needs. Cold-start covers a case nothing
> else can serve at all. I'd rather ship a model that knows when to stay quiet
> than one that's always confident."

**"Is the model underfitting?"**

> "No, and it was measured rather than assumed. Adding capacity halves training
> error while test error gets worse — train 0.213 / test 0.273 currently, versus
> 0.083 / 0.282 with more capacity. That's an overfitting signature, not
> underfitting. Early stopping picks about 367 trees and test error is flat from
> 400 to 2,500."

**"Why not ARIMA / Prophet / a neural net?"**

> "200 series of ~190 weeks each are far too short to fit individually, which is
> what per-series ARIMA or ETS requires. A global model pools across SKUs so a
> slow mover borrows structure from the fast ones. And it's direct multi-horizon —
> the horizon is a feature — because a recursive model compounds its own error
> across eight weeks."

**"How do I know the backtest isn't leaking?"**

Point at invariants 1 and 2 in §5. The empirical guard is the strong answer —
it's behavioural, not a code review.

**"What would you do next?"**

> "Three things. Wire the promotional decomposition into the dashboard so
> merchandisers can run discount scenarios. Get real inventory data with worse
> service levels, where censoring recovery would be doing far more work than the
> 1.9% it corrects here. And re-run the aggregate-advantage test each refresh — if
> NorthBay's demand ever develops cross-SKU correlation, the reconciler starts
> earning its place automatically."

**"What's the weakest part?"**

Answer it straight — this is a credibility question, not a trap.

> "Cold-start. It's the one model that doesn't beat its alternative anywhere I can
> measure. I kept it because the zero-history case is real and unservable
> otherwise, and because the launch curve and price elasticity it produces are
> useful for planning a launch buy on their own. But I'd want real launch data
> before claiming much for it."

---

## 9. Running it live, if you're demoing

```bash
python scripts/00_generate_data.py --force
python scripts/01_run_pipeline.py       # ~4 seconds, shows the cleaning report
python scripts/03_train_backtest.py     # ~12 min — leakage gate then 6 folds
python scripts/04_score_risk.py         # forecast + risk + rupee impact
uvicorn service.main:app --port 8000    # dashboard on :8000, API docs on /docs
```

**Two things worth showing live:**

1. `scripts/01_run_pipeline.py` — the cleaning log scrolls past with a reason for
   every action. It makes the data-quality work visible in about four seconds.
2. The leakage guard line in `03`: `leakage check passed | columns_corrupted=12
   features_checked=58 rows_compared=180192`. That single log line is the
   strongest correctness claim in the project.

**Quality gates, if asked:**

```bash
python -m pytest tests/ -q        # 260 tests
python -m ruff check src/ scripts/ service/
```
