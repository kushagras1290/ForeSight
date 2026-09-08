# Project FORESIGHT

**Demand & Inventory Intelligence for NorthBay Living**

A weekly, SKU level demand forecast and an early-warning system that tells the
operations team what to reorder, what to clear, and what to leave alone — with the
rupee value of each attached.

---

## The result, up front

| | |
|---|---|
| **Forecast error (WAPE)** | **22.8%** vs **34.5%** for the seasonal-naive baseline — **34.0% better**, winning in all 6 backtest folds |
| **Held-out test** | **25.1%** vs **34.7%** — **27.5% better** on a 30% slice the model never trained on |
| **Forecast bias** | **−0.7%** vs the baseline's **−15.6%** |
| **Prediction interval** | 80% stated, **81.6% measured** coverage |
| **Accuracy at the decision level** | **89.1%** for a SKU across the full 8-week lead time |
| **Sales at risk** | **₹1.42 Cr** across 20 products likely to stock out |
| **Capital locked** | **₹6.03 Cr** across 37 overstocked products |
| **Validation** | Rolling-origin, 6 folds, 9,360 out-of-sample product-weeks, plus a 70/30 chronological holdout of 10,760 more |

Full numbers in [`artifacts/metrics.json`](artifacts/metrics.json). The model
selection and the comparison against traditional methods are written up in
[`reports/model_comparison.md`](reports/model_comparison.md); all five models and
what each was measured to be worth are in
[`reports/model_suite.md`](reports/model_suite.md).

---

## Run it

Requires **Python 3.11+** and **Node 20+**.

```bash
# 1. Install
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Run the whole analysis (~10 minutes, most of it training)
python scripts/00_generate_data.py     # materialise data/raw/
python scripts/01_run_pipeline.py      # clean -> analysis-ready datasets     (D1)
python scripts/02_run_eda.py           # data-quality & EDA memo + figures    (D2)
python scripts/03_train_backtest.py    # features, leakage gate, backtest     (D3)
python scripts/04_score_risk.py        # forecast + risk + rupee impact       (D4)
python scripts/05_build_readout.py     # executive readout PDF                (D7)
python scripts/08_hierarchy_backtest.py   # models 2 & 3, scored on the same folds
python scripts/07_holdout_split.py        # 70/30 chronological test split
python scripts/09_generate_test_sets.py   # three ready-made test datasets

# Optional: what the extra commercial data is worth, measured
python scripts/03_train_backtest.py --no-drivers --tag basetables   # brief scope only
python scripts/03_train_backtest.py --tag drivers                   # with drivers

# 3. Serve the API and the dashboard
cd dashboard && npm ci && npm run build && cd ..
uvicorn service.main:app --port 8000
```

Then open <http://localhost:8000> for the dashboard (D5) and
<http://localhost:8000/docs> for the API (D6).

For dashboard development with hot reload, run `npm run dev` in `dashboard/`
alongside the service and use port 5173 — `/api` is proxied, so no CORS setup is
needed.

### The data

Extracts are not tracked in version control (brief section 16.3).
`scripts/00_generate_data.py` materialises a dataset matching the Appendix A
schema into `data/raw/`, deterministically from a fixed seed, which is what makes
every number above reproducible.

**The pipeline is source-agnostic.** It reads `data/raw/*.csv` and knows nothing
about how those files got there, so supplying the real extracts means dropping
the four CSVs into that directory — no code changes:

```bash
python scripts/06_refresh.py --validate-only   # check they are readable first
python scripts/06_refresh.py                   # refresh the plan
```

---

## Testing it on held-out data

The backtest scores the model on history. To test it yourself on data the model
has **never been trained on**, produce a chronological 70/30 split:

```bash
python scripts/07_holdout_split.py            # 70% train, 30% held back
python scripts/07_holdout_split.py --holdout-fraction 0.2
```

This trains on the earliest 70% of weeks, holds the most recent 30% back
entirely, forecasts them, and writes:

| File | What it is |
|---|---|
| `data/holdout/holdout_actuals.csv` | the true demand — **upload this** |
| `data/holdout/holdout_forecast.csv` | what the model predicted |
| `artifacts/holdout_metrics.json` | the honest score vs the baseline |

The result is on the dashboard's **Test data** tab: every week of the test period
charted against what actually sold, broken down by category and by how far ahead
the forecast was made, and a per-product scorecard ranked by units mis-forecast.
That tab reads the same table the score was computed from, so what is on screen
and what is reported cannot disagree.

To score it yourself instead, open **Forecast accuracy** → *Calibrate on your own
data* and drop the CSVs in. The service scores them and reports WAPE, bias,
interval coverage and the biggest error contributors — independently of anything
the backtest claimed.

**The split is by time, never at random.** A random 70/30 split would put later
weeks into training and earlier weeks into test, so the model would be asked to
"predict" a past it had already been shown, and the accuracy would be fiction.

One honest detail: as the origin rolls forward through the holdout, *features*
do use observed holdout demand — because in production you genuinely do see last
week before forecasting next week. What never happens is the model **training**
on a holdout week. The split is on model fitting, not on observation.

### Three ready-made test sets

```bash
python scripts/09_generate_test_sets.py
```

Writes three test sets to `data/testsets/`, each answering a different question.
All three are built on the real holdout — a year of genuine demand the model
never trained on — so `steady` is that period untouched and the other two bend
the same base, which means any difference from the control is caused by the
stress and nothing else.

| Set | Question | Portfolio accuracy | On the affected rows |
|---|---|---:|---:|
| `steady` | Does the reported accuracy transfer? | **74.9%** | — (control) |
| `shock` | What happens when the world changes? | 71.0% | **47.8%** |
| `promo` | Does it call a promotional peak? | 70.5% | **29.6%** |

The right-hand column is the one that matters. A shock confined to two of six
categories moves the portfolio figure by four points however severe it is inside
them, so each set is also scored on **just the rows it touched**. On promoted
products during their promotion weeks the forecast runs **70% light** and its
intervals cover **9%** of outcomes — a concrete demonstration of why the
promotional response model exists.

Each set ships with `actuals.csv`, `forecast.csv` and a README stating what it
scored. Upload both files together on the dashboard.

### Bringing your own actuals

You do not need any of the above. Any CSV with a product code, a week and a
quantity works — headers are matched leniently, so `sku`/`product`/`item`,
`week`/`date` and `units`/`qty`/`sales` all resolve:

```csv
sku_id,week_starting,units
NBL-LGT-004,2026-08-31,42
```

**Upload several files at once.** The service works out what each one is from its
own headers, so order does not matter and nothing has to be labelled:

- *Actuals only* — scored against the forecast this service already holds.
- *Actuals plus a forecast* — scored against **your** forecast instead, so an
  external test set is measured end to end with nothing borrowed. A forecast is
  recognised by a `prediction`, `forecast` or `yhat` column.
- *Several actuals files* — concatenated, so a year split across quarterly
  exports goes in together.

```bash
curl -F "files=@actuals.csv" -F "files=@forecast.csv" \
     http://localhost:8000/api/evaluate/upload
```

---

## Running it on new data

Two different rhythms, deliberately separated.

### Weekly — new sales and inventory extracts land

```bash
python scripts/06_refresh.py --validate-only    # pre-flight: shape, nulls, duplicates
python scripts/06_refresh.py                    # re-clean, re-score, rebuild the readout
python scripts/06_refresh.py --retrain          # occasionally: also retrain (~10 min)
```

A refresh **reuses the trained model** and takes seconds. The model learns demand
*patterns*, which move slowly; the plan depends on recent demand and current
stock, which move weekly. `06_refresh.py` warns when the data has run more than
13 weeks past the model's training cutoff and a retrain is worth considering.

### Daily — stock moves, but the forecast does not

Do **not** re-run the pipeline for a delivery that landed this morning. Post the
live position instead:

```bash
curl -X POST http://localhost:8000/api/score \
  -H 'Content-Type: application/json' \
  -d '{"positions": [{"sku_id": "NBL-LGT-004", "on_hand_units": 120, "on_order_units": 40}]}'
```

The risk layer is recomputed immediately against that position, using the same
model and the same arithmetic as the batch. Only `on_hand_units` is required;
anything omitted falls back to the stored snapshot. The response carries
`forecast_age_days` so a caller can see how stale the underlying forecast is.

---

## What was built

| # | Deliverable | Where |
|---|---|---|
| D1 | Reproducible data pipeline | [`src/foresight/pipeline.py`](src/foresight/pipeline.py), [`clean.py`](src/foresight/clean.py) |
| D2 | Data-quality & EDA memo | [`reports/eda_memo.md`](reports/eda_memo.md) |
| D3 | Demand forecast, backtested | [`src/foresight/forecast.py`](src/foresight/forecast.py), [`custom_model.py`](src/foresight/custom_model.py), [`backtest.py`](src/foresight/backtest.py) |
| D4 | Risk scoring & decisioning | [`src/foresight/risk.py`](src/foresight/risk.py) |
| D5 | Planning dashboard | [`dashboard/`](dashboard/) |
| D6 | Scoring service | [`service/`](service/) |
| D7 | Executive readout | [`reports/executive_readout.pdf`](reports/executive_readout.pdf) |

### Five purpose-built models

SKU-level demand forecasting does not fail in one way, and fixing one failure
does nothing for the others. Each model targets exactly one, and each is reported
with what it was *measured* to be worth — including the two that turned out to be
worth nothing on this dataset.

| # | Model | Failure it addresses | Measured verdict |
|---|---|---|---|
| 1 | [Adaptive Demand Ensemble](src/foresight/custom_model.py) | One estimator forced on four different demand regimes | **Ships — +34.0% vs baseline** |
| 2 | [Hierarchical Reconciler](src/foresight/reconcile.py) | SKU forecasts and the category plan contradict each other | Ships, **correctly declines to act**; guarantees exact coherence |
| 3 | [Cold-Start Propagation](src/foresight/coldstart.py) | A new product has no history to forecast from | Ships as a zero-history fallback; **no accuracy gain** |
| 4 | [Censored Demand Recovery](src/foresight/censoring.py) | Sales during a stockout are a lower bound, not demand | Ships — corrects 632 SKU-weeks |
| 5 | [Promotional Response](src/foresight/promotions.py) | A promoted week is a different process, not a bigger number | Ships for scenario planning and reporting |

Supporting documents:

- [**All five models, and what each is actually worth**](reports/model_suite.md)
- [**How the custom ensemble works**](reports/model_architecture.md)
- [**How it compares to traditional forecasting**](reports/model_comparison.md)

The reconciler is the one worth reading about. It was built on the premise that
category forecasts (WAPE 0.113) are far more accurate than SKU forecasts (0.247)
and should pull them toward a better total. Measuring the premise directly showed
it was false — the gap is an arithmetic artefact of summing, and an
independently-forecast category is actually **14.5% worse** than simply adding up
its SKUs. Applied at full strength it cost 5.5% accuracy. It now gates on that
test and passes the forecast through untouched. See
[`reports/model_suite.md` §3](reports/model_suite.md).

---

## How it works

```text
data/raw/*.csv                 four client extracts, read as text
      │
      ▼  ingest + clean        every decision recorded with its rationale
data/processed/                analysis-ready daily + weekly panels
      │
      ▼  features              strictly lagged to the forecast origin
      │                        ── leakage guard runs here; a leak fails the build
      ▼  model                 LightGBM + TSB + seasonal profile, regime-routed
      │
      ▼  backtest              rolling origin, 6 folds, vs seasonal-naive
      │
      ▼  risk                  stockout / overstock probabilities → actions → ₹
artifacts/                     model, metrics, forecast, risk table
      │
      ▼
  FastAPI service  ──────────►  React dashboard
```

### The forecast

A **global** model across all 200 SKUs in **direct multi-horizon** form (the
horizon is a feature), blended per demand regime with two specialist components.
Global because 200 series of ~190 weeks are far too short to fit individually;
direct because a recursive model compounds its own error across the horizon; L1
objective because WAPE is what the client judges it on.

### The risk layer

Deliberately transparent arithmetic rather than a second model — an ops manager
has to be able to challenge a reorder recommendation and get a straight answer.
Both axes are genuine probabilities on opposite tails:

```text
stockout_score  = P(demand over the lead time  > stock available)
overstock_score = P(demand over 12 weeks       < stock available)
```

Those two, plus a forecast-volatility override, produce the four quadrants of the
decisioning grid. `risk.py` explains why the two axes are framed this way — the
obvious framing makes the fourth quadrant mathematically unreachable.

---

## Honesty controls

Brief section 7.1 makes leak-free forecasting the non-negotiable rule of this
engagement. Five things enforce it rather than assert it:

1. **The leakage guard is a build gate.** `assert_no_leakage` rebuilds every
   feature against a copy of the panel whose future has been overwritten with
   noise. If any feature value at or before the cutoff moves, it read the future.
   It corrupts 12 observed columns and checks all 58 numeric features across
   180,192 comparable rows. It runs inside `scripts/03_train_backtest.py`, and on
   failure **no model is trained and nothing is written**. It is verified to
   catch a planted centred-rolling-window leak, not just to pass — and it caught
   a real one during development, in the promotional baseline's fallback path.

2. **The train/test split filters on the target week, not the origin week.**
   `train: target_week <= origin`. Filtering on the origin admits rows whose
   outcomes lie beyond it — the classic bug that inflates backtest scores with no
   visible symptom. It lives in one function, `_split_fold`.

3. **The baseline can win.** `selected_model` picks whichever candidate posts the
   lowest backtest WAPE, including seasonal-naive. If no learned model beats it,
   the baseline ships and CI records it as a warning.

4. **Models can decline to fire.** The hierarchical reconciler gates on a direct
   test of whether aggregate forecasts beat their own bottom-up sum. On this data
   they do not, so it sets its blend weight to zero and passes the forecast
   through unchanged. Without that test it would have shipped a 5.5% accuracy
   regression that looked like sophistication.

5. **Imputed values are never scored against.** Censored demand recovery supplies
   the *training* target, but weeks it touched are excluded from *evaluation*, so
   an imputation can never flatter a reported number.

### Known limitations

Stated here rather than left to be discovered:

- Average error is ~23% of demand at SKU-week. Good for weekly SKU-level
  forecasting, but not precision — the output is a prioritised starting point,
  not an instruction.
- **Newly-launched products are under-forecast by about 15%.** They are still
  ramping and the model lags the ramp. All 9 are flagged `low` confidence in the
  dashboard, but their reorder quantities will read light.
- Intermittent SKUs are forecast far worse (WAPE > 100%). They carry 0.02% of
  units, and are reported separately rather than hidden in the average.
- Demand during a stockout is recovered rather than taken at face value
  (`censoring.py`), but the recovery rests on an even-arrival assumption inside
  the week. Where a stockout coincided with a demand spike, the correction will
  still be conservative.
- `sigma` over the lead time assumes weekly forecast errors are independent. They
  are somewhat correlated, so the true uncertainty is a little wider than shown.
- Ensemble blend weights are refitted from scratch each run and are not smoothed
  across runs, so they can move when the validation window shifts.
- Inventory is a weekly snapshot. Use `POST /api/score` for a live position.

### The model is not underfitting — this was measured

A capacity sweep on a real backtest fold, reported because "add more capacity" is
the reflex and here it is wrong:

| Config | Trees | Train WAPE | Test WAPE | Gap |
|---|---:|---:|---:|---:|
| Constrained (15 leaves) | 400 | 0.228 | 0.276 | 0.048 |
| **Current** | 700 | 0.165 | **0.257** | 0.092 |
| More capacity (127 leaves) | 2000 | 0.096 | 0.257 | 0.161 |
| Much deeper (255 leaves) | 2500 | 0.061 | 0.256 | 0.195 |

Going from 700 trees to 2,500 with 255 leaves cuts **training** error by a factor
of 2.7 and moves **test** error by 0.001. The extra capacity is being spent
memorising, not generalising — the signature of overfitting.

The top row is the control: constrain the model to 15 leaves and *both* numbers
get worse together, which is what underfitting actually looks like. The current
setting is nowhere near it.

For scale: a centred rolling median that *cheats by reading the future* scores
0.23–0.26 on this data. The model reaches **0.228** genuinely out-of-sample
across the full backtest, so it is at the edge of that oracle bound. The
remaining error is demand noise, not model capacity.

---

## Project layout

```text
ForeSight/
├─ src/foresight/          pipeline, features, the five models, risk, reporting
├─ src/foresight_drivers/  optional commercial-driver extension (separable)
├─ scripts/                00–09, the numbered run order
├─ service/                FastAPI scoring service (D6)
├─ dashboard/              React + Vite planning dashboard (D5)
├─ notebooks/              exploratory narrative
├─ reports/                EDA memo, executive readout, model documents
├─ artifacts/              trained models, metrics, forecast, risk table
├─ data/                   raw / interim / processed / holdout / testsets  (not tracked)
├─ Dockerfile              single deployable: API + dashboard
└─ render.yaml             deployment blueprint
```

---

## Configuration

Everything tunable is an environment variable, validated by pydantic at startup —
a bad value fails immediately with a readable message rather than silently
producing a wrong forecast. See [`.env.example`](.env.example).

| Variable | Default | Effect |
|---|---|---|
| `FORESIGHT_HORIZON_WEEKS` | 8 | Forecast horizon |
| `FORESIGHT_SERVICE_LEVEL` | 0.95 | Target probability of not stocking out |
| `FORESIGHT_OVERSTOCK_COVER_WEEKS` | 12 | Cover above which stock counts as excess |
| `FORESIGHT_RISK_HIGH_THRESHOLD` | 0.5 | Quadrant boundary on the decisioning grid |
| `FORESIGHT_INTERVAL_COVERAGE` | 0.8 | Prediction-interval coverage |
| `FORESIGHT_RANDOM_SEED` | 20260907 | Fixed, so results are re-creatable |

---

## Deployment

```bash
docker build -t foresight .          # LightGBM only, a few minutes
docker run --rm -p 8000:8000 foresight
```

The image builds the dashboard, installs the service, and bakes in a scored plan,
so one container serves both from the same origin. Pass
`--build-arg FULL_TRAIN=true` to bake in the adaptive ensemble instead.
[`render.yaml`](render.yaml) deploys the same image as a Render Blueprint.

---

## Verifying it

```bash
ruff check src service scripts        # lint
python scripts/06_refresh.py --validate-only   # extracts readable?
cd dashboard && npx tsc -b --noEmit   # dashboard typecheck
```

The strongest check is the end-to-end run itself: a clean clone, the six
commands above, and the same headline numbers. That is what
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) executes on every push —
including the leakage gate and an assertion that the selected model beat the
baseline.

---

*Zidio Development · Data Science & Analytics engagement · Confidential*
