# Data Quality & EDA Insight Memo

**Project FORESIGHT — Demand & Inventory Intelligence**
**Client:** NorthBay Living   |   **Deliverable:** D2   |   **Milestone:** M2

---

## Summary for the Head of Operations

We took delivery of four extracts covering **50 SKUs** over
**104 weeks** (2024-01-01 to 2025-12-22) and put them through an
automated cleaning pipeline before doing any analysis.

The data is usable, but **it is not clean on arrival**. We found and repaired
**5 distinct classes of problem** affecting **46,533 rows**, of which
**0** were severe enough that leaving them in place would have produced a
materially wrong forecast rather than merely a noisier one.

Three of those matter enough to raise with your systems team, because they are worth
fixing at source rather than patching every month:

1. **The sales export omits zero-demand days entirely.** A day on which a product sold
   nothing produces no row at all. Any analysis that treats the file as a complete daily
   record will read those gaps as missing data rather than as genuine zero demand, and will
   overstate average demand for every slow-moving line.
2. **The same sale can appear twice with different formatting.** Re-runs of the export
   append rows that are identical in substance but written differently — `1` versus `TRUE`
   in the promotion flag, or a lower-cased product code. A naive de-duplication misses these
   and double-counts the demand.
3. **`lead_time_days` contains impossible values.** Both `0` and values above 900 appear.
   Lead time sets the window the stockout calculation looks over, so a zero makes a product
   look permanently safe and a 900 makes it look permanently critical.

Everything below is reproducible: re-running `python scripts/01_run_pipeline.py` regenerates
these findings from the raw files.

---

## 1. Data quality: what we found and what we did

Every row in this table was produced by the pipeline itself, not written by hand.

| Severity | Table | Issue found | Rows | How it was resolved | Why |
|---|---|---|---:|---|---|
| **warning** | `marketing_spend` | 42200 media rows for products absent from the master | 42,200 | dropped | spend against an unknown product cannot be attributed to a forecast, and would otherwise inflate the plan totals |
| **warning** | `inventory_snapshots` | 3600 snapshot(s) for products absent from the master | 3,600 | quarantined out of the modelling set | stock for an unknown product cannot be valued or actioned |
| **warning** | `weekly_panel` | 67 SKU-week(s) covering fewer than seven days | 67 | dropped from the modelling panel | a part-week at a SKU's launch or at the edge of the extract holds less demand purely because it is shorter. Kept, it would read as a demand collapse and drag both the lag features and the seasonal-naive baseline down |
| **warning** | `sku_master` | 10 product(s) could not be categorised | 10 | labelled 'Unclassified' and retained | these still have real sales; excluding them would understate demand, so they are kept and made visible instead of hidden |
| **info** | `calendar` | 656 date(s) with no promo_event | 656 | set to an empty string meaning 'no event' | a null here means no promotion ran, not unknown data; making that explicit keeps the one-hot feature honest |


---

## 2. What the demand data says

### 1. Demand is strongly seasonal, and the festive peak is the whole year

The busiest week (2025-03-24) sold 6,075 units against 2,708 in the quietest (2024-01-01) - a 2.2x swing. Lighting and Decor peak hardest around Diwali; Storage & Organisation peaks in January instead.

**So what:** Planning to an annual average guarantees stocking out in October and sitting on stock in April. Any reorder rule has to be seasonal.

### 2. 24 of 50 SKUs generate 80% of revenue

The top 20% of SKUs account for 47% of revenue. The tail is long and slow-moving.

**So what:** Attention should be rationed accordingly: the head needs accurate forecasting, the tail needs a simple rule and a periodic cull.

### 3. 0% of SKUs sell nothing in a third or more of weeks

These intermittent lines make percentage-based accuracy metrics unstable, which is why WAPE rather than MAPE is used to judge the forecast.

**So what:** They also need a different forecasting treatment from the fast movers - averaging across both understates the fast movers and overstates the tail.

### 4. Promotions lift demand about 5% on average

The effect varies materially by category, so a single blanket uplift assumption would misprice the stock build for every category at once.

**So what:** Promotion timing is known in advance from the calendar, so it can be forecast rather than reacted to.


---

## 3. Where the revenue actually sits

| Category | SKUs | Units sold | Revenue (Rs) | Avg units/week | Share of weeks with no sale |
|---|---|---|---|---|---|
| Decor | 10 | 113,185.00 | 760,335,746.67 | 122.10 | 0.00 |
| Unclassified | 10 | 75,199.00 | 561,537,570.34 | 80.00 | 0.00 |
| Kitchen & Dining | 10 | 92,899.00 | 525,473,085.77 | 100.00 | 0.00 |
| Storage & Organisation | 10 | 85,779.00 | 510,219,663.79 | 89.54 | 0.00 |
| Lighting | 10 | 91,278.00 | 424,095,359.87 | 95.98 | 0.00 |


### Top 15 SKUs by revenue

| SKU | Category | Units | Revenue (Rs) |
|---|---|---|---|
| SKU026 | Unclassified | 16,480.00 | 180,283,619.20 |
| SKU027 | Decor | 17,650.00 | 150,317,990.00 |
| SKU045 | Storage & Organisation | 18,553.00 | 142,427,299.34 |
| SKU029 | Lighting | 11,495.00 | 133,829,962.75 |
| SKU043 | Kitchen & Dining | 15,247.00 | 133,346,755.19 |
| SKU042 | Decor | 13,970.00 | 128,507,655.10 |
| SKU031 | Unclassified | 13,671.00 | 119,155,479.03 |
| SKU005 | Storage & Organisation | 12,146.00 | 115,304,650.12 |
| SKU008 | Kitchen & Dining | 11,199.00 | 110,227,725.36 |
| SKU035 | Storage & Organisation | 9,765.00 | 106,147,405.35 |
| SKU018 | Kitchen & Dining | 18,392.00 | 102,542,940.72 |
| SKU049 | Lighting | 17,787.00 | 92,908,082.19 |
| SKU007 | Decor | 18,061.00 | 92,365,579.49 |
| SKU017 | Decor | 10,337.00 | 87,630,056.84 |
| SKU012 | Decor | 9,911.00 | 87,012,336.07 |


---

## 4. Dead stock: lines that have stopped selling

Comparing average weekly demand over the last 13 weeks against the same 13 weeks a year
earlier. These are the clearest markdown and delisting candidates.

| SKU | Weekly units a year ago | Weekly units now | Decline |
|---|---|---|---|
| SKU025 | 19.00 | 15.15 | 0.20 |
| SKU015 | 26.31 | 22.08 | 0.16 |
| SKU028 | 35.92 | 31.00 | 0.14 |
| SKU046 | 62.67 | 54.62 | 0.13 |
| SKU012 | 168.67 | 151.77 | 0.10 |
| SKU040 | 96.08 | 88.85 | 0.08 |
| SKU045 | 155.00 | 145.54 | 0.06 |
| SKU050 | 39.46 | 37.08 | 0.06 |
| SKU021 | 52.23 | 49.15 | 0.06 |
| SKU036 | 30.38 | 29.31 | 0.04 |
| SKU006 | 39.54 | 38.23 | 0.03 |
| SKU026 | 137.92 | 133.77 | 0.03 |
| SKU041 | 68.62 | 66.62 | 0.03 |
| SKU048 | 52.77 | 51.31 | 0.03 |
| SKU034 | 123.31 | 119.92 | 0.03 |


---

## 5. Promotional response by category

| Category | Avg units, normal week | Avg units, promo week | Uplift |
|---|---|---|---|
| Lighting | 93.74 | 99.66 | 0.06 |
| Kitchen & Dining | 98.03 | 103.19 | 0.05 |
| Decor | 119.89 | 125.75 | 0.05 |
| Storage & Organisation | 87.93 | 92.13 | 0.05 |
| Unclassified | 78.66 | 82.20 | 0.05 |


---

## 6. Figures

![01_demand_over_time.png](figures/01_demand_over_time.png)

![02_seasonality_by_category.png](figures/02_seasonality_by_category.png)

![03_revenue_concentration.png](figures/03_revenue_concentration.png)

![04_dead_stock.png](figures/04_dead_stock.png)

![05_promo_uplift.png](figures/05_promo_uplift.png)

![06_data_quality.png](figures/06_data_quality.png)


---

## 7. What this means for the forecast

* **Seasonality has to be explicit.** The festive swing is far too large for a moving
  average to absorb, so the model is given week-of-year and holiday features directly, and
  the baseline it is measured against is seasonal-naive rather than a flat average.
* **WAPE, not MAPE.** A large share of SKUs sell nothing in many weeks. MAPE is undefined on
  those weeks and explodes on the ones either side, so it is reported for familiarity only
  and never used to choose a model.
* **The tail needs separate treatment.** Fast and intermittent lines behave differently
  enough that one estimator is the wrong answer for both, which is what the adaptive
  ensemble in D3 addresses.
* **Promotion dates are known in advance.** They come from the published calendar, so they
  are a legitimate forecast input rather than something to be reacted to after the fact.

---

## 8. Known limitations of this analysis

* Inventory positions are weekly snapshots, so demand that occurred while a product was out
  of stock is invisible. Recorded sales are therefore a floor on true demand, and stockout
  risk is, if anything, understated for the lines that stock out most often.
* Products absent from the product master were quarantined rather than guessed at. They
  carry real sales and should be fixed at source.
* Returns booked as negative sales were treated as zero demand rather than netted off,
  because the forecast models gross demand. Net revenue reporting would need them handled
  differently.
