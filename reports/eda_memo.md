# Data Quality & EDA Insight Memo

**Project FORESIGHT — Demand & Inventory Intelligence**
**Client:** NorthBay Living   |   **Deliverable:** D2   |   **Milestone:** M2

---

## Summary for the Head of Operations

We took delivery of four extracts covering **200 SKUs** over
**191 weeks** (2023-01-02 to 2026-08-24) and put them through an
automated cleaning pipeline before doing any analysis.

The data is usable, but **it is not clean on arrival**. We found and repaired
**18 distinct classes of problem** affecting **190,931 rows**, of which
**6** were severe enough that leaving them in place would have produced a
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
| **critical** | `sales_daily` | 64902 SKU-day combination(s) absent from the extract | 64,902 | materialised as explicit zero-demand rows | the export only carries rows where a sale occurred, so a zero-demand day is simply missing. Lags, rolling means and the seasonal-naive baseline all index by position in time - without the zeros every one of them silently shifts and the model learns a demand level that never existed |
| **critical** | `sales_daily` | 2025 row(s) with leading/trailing whitespace or lower-case in sku_id | 2,025 | trimmed, whitespace-collapsed and upper-cased | ' nbl-lgt-004 ' and 'NBL-LGT-004' are one product; treating them as two splits that SKU's history and breaks its lag features and its join to the product master |
| **critical** | `inventory_snapshots` | 955 snapshot(s) with an impossible lead_time_days (outside 1-120; observed values include 0 and 900+) | 955 | replaced with the SKU's median plausible lead time, falling back to the global median | lead time sets the window the stockout calculation looks over. A 0 makes every SKU look safe and a 900 makes every SKU look critical - both would destroy the reorder list's credibility |
| **critical** | `sales_daily` | 919 row(s) identical in value but not in text (same sale re-exported with a different flag or case encoding) | 919 | dropped, keeping the first occurrence | these are the same sale exported twice. They must be removed before key-collision aggregation, which sums units - left in, they would be added together and double that day's demand rather than being recognised as one sale |
| **critical** | `inventory_snapshots` | 764 snapshot(s) with a missing on_hand_units | 764 | carried the last known position forward within each SKU; 5 row(s) with no prior observation set to 0 | a missed export is not evidence of empty shelves. Filling with zero everywhere would fabricate stockouts and flood the reorder list with false alarms |
| **critical** | `sales_daily` | 420 byte-identical duplicate row(s) | 420 | dropped, keeping the first occurrence | identical rows are a re-run of the export appended twice, not two real orders; leaving them double-counts demand and inflates every forecast built on it |
| **warning** | `sales_daily` | promo_flag arrives in three encodings (0/1, Y/N, TRUE/FALSE) | 112,580 | mapped a fixed truthy vocabulary to 1, everything else to 0 | several source systems feed this column. Unrecognised values resolve to 0 because inventing a promotion is worse than missing one: it teaches the model a price effect that never happened |
| **warning** | `sales_daily` | 1688 row(s) with a missing or non-positive unit_price | 1,688 | derived as revenue / units where possible, else the SKU's median realised price, else its list price | price feeds both the promotion signal and the revenue reconstruction; the fallback order runs from most to least row-specific so the best available evidence is always used |
| **warning** | `sales_daily` | 675 sales row(s) reference 3 product(s) absent from the product master (NBL-XXX-001, NBL-XXX-002, NBL-XXX-003) | 675 | quarantined out of the modelling set and reported to the client | without a category, cost or price these rows cannot be forecast, valued, or shown in the dashboard; they are flagged for the client to fix at source rather than silently guessed |
| **warning** | `sales_daily` | 506 row(s) with negative units_sold | 506 | clipped to zero, with revenue on those rows zeroed too | these are refunds booked against the sales ledger. The model forecasts gross demand, so a return is not negative demand - netting it off would understate what customers actually wanted |
| **warning** | `sku_master` | 24 distinct category spellings for 6 real categories | 200 | folded to a controlled vocabulary by stripping accents and whitespace, lower-casing, and unifying 'and'/'&' and -ise/-ize | category is a model feature and a dashboard filter; unmerged spellings fragment every group-by and split one category's history across several labels |
| **warning** | `weekly_panel` | 48 SKU-week(s) covering fewer than seven days | 48 | dropped from the modelling panel | a part-week at a SKU's launch or at the edge of the extract holds less demand purely because it is shorter. Kept, it would read as a demand collapse and drag both the lag features and the seasonal-naive baseline down |
| **warning** | `sku_master` | launch_date exported in two formats (45 rows are DD-MM-YYYY, the rest ISO) | 45 | parsed with explicit formats, ISO first then day-first | letting pandas infer the format resolves ambiguous values such as 04-09-2021 inconsistently row by row, silently shifting dates by months and corrupting SKU age features |
| **warning** | `sales_daily` | 11 non-identical row(s) sharing a (date, sku_id) key | 11 | aggregated: units and revenue summed, price averaged, promo flag taken as the maximum | the fact table's grain is one row per SKU per day. These are partial exports of the same day, so summing restores the true daily total; dropping them would lose real sales |
| **warning** | `sku_master` | 3 product(s) have no category | 3 | recovered from the NBL-<code>-<nnn> convention in the SKU id | the SKU id encodes category at creation time and is more reliable than the hand-maintained category column; dropping these products would lose their sales history entirely |
| **info** | `sales_daily` | 2516 row(s) with missing revenue | 2,516 | recomputed as units_sold x unit_price | revenue is fully determined by the other two columns, so it is reconstructed exactly rather than imputed or the row discarded |
| **info** | `calendar` | is_holiday exported as Yes/No text rather than a 0/1 flag | 1,477 | mapped a fixed truthy vocabulary to 1, everything else to 0 | holiday is summed into a weekly feature; a text column would either fail to aggregate or coerce to nonsense |
| **info** | `calendar` | 1197 date(s) with no promo_event | 1,197 | set to an empty string meaning 'no event' | a null here means no promotion ran, not unknown data; making that explicit keeps the one-hot feature honest |


---

## 2. What the demand data says

### 1. Demand is strongly seasonal, and the festive peak is the whole year

The busiest week (2024-10-28) sold 40,958 units against 5,980 in the quietest (2023-05-08) - a 6.8x swing. Lighting and Decor peak hardest around Diwali; Storage & Organisation peaks in January instead.

**So what:** Planning to an annual average guarantees stocking out in October and sitting on stock in April. Any reorder rule has to be seasonal.

### 2. 62 of 200 SKUs generate 80% of revenue

The top 20% of SKUs account for 69% of revenue. The tail is long and slow-moving.

**So what:** Attention should be rationed accordingly: the head needs accurate forecasting, the tail needs a simple rule and a periodic cull.

### 3. 7% of SKUs sell nothing in a third or more of weeks

These intermittent lines make percentage-based accuracy metrics unstable, which is why WAPE rather than MAPE is used to judge the forecast.

**So what:** They also need a different forecasting treatment from the fast movers - averaging across both understates the fast movers and overstates the tail.

### 4. Promotions lift demand about 134% on average

The effect varies materially by category, so a single blanket uplift assumption would misprice the stock build for every category at once.

**So what:** Promotion timing is known in advance from the calendar, so it can be forecast rather than reacted to.


---

## 3. Where the revenue actually sits

| Category | SKUs | Units sold | Revenue (Rs) | Avg units/week | Share of weeks with no sale |
|---|---|---|---|---|---|
| Small Appliances | 27 | 387,222.00 | 2,367,088,841.58 | 80.40 | 0.04 |
| Kitchen & Dining | 45 | 695,969.00 | 1,288,881,973.55 | 97.27 | 0.02 |
| Bedding & Bath | 39 | 430,650.00 | 1,144,289,504.09 | 68.54 | 0.01 |
| Lighting | 30 | 278,698.00 | 805,201,772.45 | 51.96 | 0.03 |
| Decor | 40 | 432,939.00 | 460,827,308.35 | 63.71 | 0.01 |
| Storage & Organisation | 19 | 238,239.00 | 217,983,151.64 | 80.68 | 0.01 |


### Top 15 SKUs by revenue

| SKU | Category | Units | Revenue (Rs) |
|---|---|---|---|
| NBL-BTH-014 | Bedding & Bath | 96,901.00 | 346,043,549.46 |
| NBL-APP-001 | Small Appliances | 55,711.00 | 332,302,101.66 |
| NBL-APP-011 | Small Appliances | 29,145.00 | 306,234,967.05 |
| NBL-APP-015 | Small Appliances | 29,560.00 | 249,990,693.60 |
| NBL-APP-022 | Small Appliances | 26,519.00 | 249,441,161.47 |
| NBL-APP-024 | Small Appliances | 28,368.00 | 210,156,979.77 |
| NBL-KTD-028 | Kitchen & Dining | 41,277.00 | 175,103,189.36 |
| NBL-APP-009 | Small Appliances | 25,248.00 | 173,667,608.55 |
| NBL-APP-014 | Small Appliances | 81,367.00 | 168,448,422.68 |
| NBL-KTD-045 | Kitchen & Dining | 64,610.00 | 143,051,062.70 |
| NBL-KTD-022 | Kitchen & Dining | 30,341.00 | 117,735,834.47 |
| NBL-KTD-006 | Kitchen & Dining | 115,807.00 | 114,808,506.26 |
| NBL-APP-010 | Small Appliances | 11,584.00 | 109,340,746.02 |
| NBL-APP-004 | Small Appliances | 11,908.00 | 92,150,763.22 |
| NBL-KTD-037 | Kitchen & Dining | 66,160.00 | 88,087,408.80 |


---

## 4. Dead stock: lines that have stopped selling

Comparing average weekly demand over the last 13 weeks against the same 13 weeks a year
earlier. These are the clearest markdown and delisting candidates.

| SKU | Weekly units a year ago | Weekly units now | Decline |
|---|---|---|---|
| NBL-APP-013 | 1.54 | 0.15 | 0.90 |
| NBL-KTD-043 | 0.62 | 0.08 | 0.88 |
| NBL-KTD-017 | 1.23 | 0.23 | 0.81 |
| NBL-LGT-024 | 1.08 | 0.23 | 0.79 |
| NBL-APP-005 | 4.62 | 1.08 | 0.77 |
| NBL-LGT-026 | 1.23 | 0.31 | 0.75 |
| NBL-KTD-036 | 1.23 | 0.31 | 0.75 |
| NBL-APP-002 | 2.08 | 0.62 | 0.70 |
| NBL-STO-015 | 31.77 | 9.62 | 0.70 |
| NBL-BTH-008 | 3.62 | 1.15 | 0.68 |
| NBL-APP-027 | 0.46 | 0.15 | 0.67 |
| NBL-LGT-019 | 6.23 | 2.08 | 0.67 |
| NBL-APP-023 | 13.85 | 4.77 | 0.66 |
| NBL-DEC-040 | 9.46 | 3.31 | 0.65 |
| NBL-APP-025 | 1.23 | 0.46 | 0.62 |


---

## 5. Promotional response by category

| Category | Avg units, normal week | Avg units, promo week | Uplift |
|---|---|---|---|
| Lighting | 40.59 | 107.73 | 1.65 |
| Kitchen & Dining | 78.29 | 196.38 | 1.51 |
| Bedding & Bath | 53.67 | 132.45 | 1.47 |
| Storage & Organisation | 64.27 | 156.74 | 1.44 |
| Small Appliances | 64.42 | 154.14 | 1.39 |
| Decor | 59.73 | 93.65 | 0.57 |


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
