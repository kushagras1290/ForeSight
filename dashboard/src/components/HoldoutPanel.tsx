/**
 * The test set: what the model predicted for a year it had never seen.
 *
 * The *Forecast accuracy* tab reports the rolling-origin backtest, which walks
 * an origin forward and averages six folds. This tab answers the blunter
 * question a sceptic asks instead — train once on the first 70% of history, then
 * forecast the remaining year and show what happened, week by week.
 *
 * It is the harsher test of the two, and it is presented as such rather than
 * blended into the headline: the model here has 30% less history and no chance
 * to re-learn a drift part-way through.
 */

import { useMemo, useState } from 'react';
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import type { HoldoutSummary } from '../lib/api';
import { formatPercent, formatShortDate, formatSignedPercent, formatUnits } from '../lib/format';

interface HoldoutPanelProps {
  holdout: HoldoutSummary;
}

/** Rows shown in the per-product table before "show all" is used. */
const TABLE_PREVIEW = 25;

const OK = 'var(--signal-ok)';
const WARN = 'var(--signal-warn)';
const CRITICAL = 'var(--signal-critical)';

/** Accuracy bands, so a colour never has to be interpreted from its hue alone. */
function accuracyColour(accuracy: number): string {
  if (accuracy >= 0.8) return OK;
  if (accuracy >= 0.6) return WARN;
  return CRITICAL;
}

function toNumber(value: number | string | null | undefined): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function HoldoutPanel({ holdout }: HoldoutPanelProps) {
  const [showAll, setShowAll] = useState(false);

  const weekly = useMemo(
    () =>
      holdout.weekly.map((row) => ({
        week: formatShortDate(row.week),
        iso: row.week,
        actual: Math.round(row.actual),
        predicted: Math.round(row.predicted),
        baseline: Math.round(row.baseline),
        band: [Math.round(row.lower), Math.round(row.upper)] as [number, number],
      })),
    [holdout.weekly],
  );

  const categories = useMemo(
    () =>
      holdout.by_category
        .map((row) => ({
          category: String(row.category),
          accuracy: toNumber(row.accuracy),
          units: toNumber(row.actual),
        }))
        .sort((left, right) => right.accuracy - left.accuracy),
    [holdout.by_category],
  );

  const horizons = useMemo(
    () =>
      holdout.by_horizon.map((row) => ({
        horizon: `+${toNumber(row.horizon)}`,
        accuracy: toNumber(row.accuracy),
        coverage: toNumber(row.coverage),
      })),
    [holdout.by_horizon],
  );

  const visibleSkus = showAll ? holdout.skus : holdout.skus.slice(0, TABLE_PREVIEW);

  if (!holdout.available) {
    return (
      <div className="callout callout--warn">
        <span className="callout__mark" aria-hidden="true">
          ◆
        </span>
        <span>
          {holdout.reason ?? 'No test split has been run yet.'} Run{' '}
          <code>python scripts/07_holdout_split.py</code> and reload this page.
        </span>
      </div>
    );
  }

  const accuracy = 1 - holdout.model_wape;
  const baselineAccuracy = 1 - holdout.baseline_wape;
  const coverageGap = holdout.interval_coverage - holdout.interval_coverage_target;
  const coverageIsGood = Math.abs(coverageGap) <= 0.05;

  const stats = [
    {
      label: 'Accuracy on unseen data',
      value: formatPercent(accuracy),
      detail: `${formatPercent(baselineAccuracy)} for the seasonal-naive baseline`,
      accent: accuracyColour(accuracy),
    },
    {
      label: 'Better than baseline by',
      value: formatPercent(holdout.improvement_vs_baseline),
      detail: 'relative reduction in forecast error',
      accent: holdout.improvement_vs_baseline > 0 ? OK : CRITICAL,
    },
    {
      label: 'Forecast bias',
      value: formatSignedPercent(holdout.bias_relative),
      detail: holdout.bias_relative < 0 ? 'runs light on average' : 'runs heavy on average',
      accent: Math.abs(holdout.bias_relative) <= 0.1 ? OK : WARN,
    },
    {
      label: 'Interval coverage',
      value: formatPercent(holdout.interval_coverage),
      detail: `${formatPercent(holdout.interval_coverage_target, 0)} was stated`,
      accent: coverageIsGood ? OK : WARN,
    },
    {
      label: 'Weeks tested',
      value: String(holdout.holdout_weeks),
      detail: `${holdout.observations.toLocaleString('en-IN')} product-weeks scored`,
      accent: 'var(--harbour-700)',
    },
    {
      label: 'Trained on',
      value: `${holdout.train_weeks} weeks`,
      detail: `${formatPercent(holdout.train_fraction, 0)} of history, ending ${
        holdout.split_week ? formatShortDate(holdout.split_week) : '—'
      }`,
      accent: 'var(--harbour-700)',
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)' }}>
      <div className="callout">
        <span className="callout__mark" aria-hidden="true">
          ◆
        </span>
        <span>
          The model was trained on the <strong>first {holdout.train_weeks} weeks</strong> of history
          and then asked to forecast the{' '}
          <strong>{holdout.holdout_weeks} weeks after it</strong> —{' '}
          {holdout.holdout_start ? formatShortDate(holdout.holdout_start) : '—'} to{' '}
          {holdout.holdout_end ? formatShortDate(holdout.holdout_end) : '—'} — with no retraining
          along the way. It had never seen any of it. This is a harder test than the backtest on the
          previous tab, and the numbers below are lower because of it.
        </span>
      </div>

      <div className="stat-grid">
        {stats.map((stat) => (
          <div key={stat.label} className="stat">
            <p className="stat__label">{stat.label}</p>
            <p className="stat__value" style={{ color: stat.accent }}>
              {stat.value}
            </p>
            <p className="stat__detail">{stat.detail}</p>
          </div>
        ))}
      </div>

      {/* --- Week by week ----------------------------------------------------- */}
      <section>
        <div className="section__head">
          <div>
            <p className="eyebrow">Every week of the test period</p>
            <h3 className="section__title">Forecast against what actually sold</h3>
            <p className="section__note">
              Total demand across all {holdout.skus.length} products. The shaded band is the
              forecast&apos;s own 80% uncertainty range.
            </p>
          </div>
        </div>

        <div className="chart-card">
          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart data={weekly} margin={{ top: 8, right: 8, bottom: 4, left: 8 }}>
              <CartesianGrid stroke="var(--rule)" vertical={false} />
              <XAxis
                dataKey="week"
                tick={{ fontSize: 11, fill: 'var(--ink-3)' }}
                tickLine={false}
                axisLine={{ stroke: 'var(--rule)' }}
                minTickGap={24}
              />
              <YAxis
                tick={{ fontSize: 11, fill: 'var(--ink-3)' }}
                tickLine={false}
                axisLine={false}
                width={64}
                tickFormatter={(value: unknown) => formatUnits(Number(value))}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--surface-raised)',
                  border: '1px solid var(--rule)',
                  borderRadius: 8,
                  fontSize: 12,
                }}
                formatter={(value: unknown, name: unknown) => [
                  formatUnits(Number(value)),
                  String(name),
                ]}
              />
              <Area
                dataKey="band"
                name="80% range"
                stroke="none"
                fill="var(--harbour-100)"
                fillOpacity={0.55}
                isAnimationActive={false}
              />
              <Line
                dataKey="actual"
                name="Actual"
                stroke="var(--ink)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                dataKey="predicted"
                name="Forecast"
                stroke="var(--harbour-500)"
                strokeWidth={2}
                strokeDasharray="5 3"
                dot={false}
                isAnimationActive={false}
              />
              <Line
                dataKey="baseline"
                name="Seasonal-naive"
                stroke="var(--ink-3)"
                strokeWidth={1}
                strokeDasharray="2 4"
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>

          <div className="chart-legend">
            <span className="chart-legend__item">
              <span className="chart-legend__swatch" style={{ background: 'var(--ink)' }} />
              Actual demand
            </span>
            <span className="chart-legend__item">
              <span
                className="chart-legend__swatch"
                style={{ background: 'var(--harbour-500)' }}
              />
              Forecast
            </span>
            <span className="chart-legend__item">
              <span className="chart-legend__swatch" style={{ background: 'var(--ink-3)' }} />
              Seasonal-naive baseline
            </span>
          </div>
        </div>
      </section>

      {/* --- Breakdowns ------------------------------------------------------- */}
      <div className="two-col">
        <section>
          <div className="section__head">
            <div>
              <p className="eyebrow">By category</p>
              <h3 className="section__title">Where it held up</h3>
            </div>
          </div>
          <div className="chart-card">
            <ResponsiveContainer width="100%" height={Math.max(200, categories.length * 42)}>
              <BarChart
                data={categories}
                layout="vertical"
                margin={{ top: 4, right: 40, bottom: 4, left: 8 }}
              >
                <CartesianGrid stroke="var(--rule)" horizontal={false} />
                <XAxis
                  type="number"
                  domain={[0, 1]}
                  tick={{ fontSize: 11, fill: 'var(--ink-3)' }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value: unknown) => formatPercent(Number(value), 0)}
                />
                <YAxis
                  type="category"
                  dataKey="category"
                  width={150}
                  tick={{ fontSize: 11, fill: 'var(--ink-2)' }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  cursor={{ fill: 'var(--surface-sunk)' }}
                  contentStyle={{
                    background: 'var(--surface-raised)',
                    border: '1px solid var(--rule)',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(value: unknown) => [formatPercent(Number(value)), 'Accuracy']}
                />
                <Bar dataKey="accuracy" radius={[0, 3, 3, 0]} isAnimationActive={false}>
                  {categories.map((row) => (
                    <Cell key={row.category} fill={accuracyColour(row.accuracy)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section>
          <div className="section__head">
            <div>
              <p className="eyebrow">By weeks ahead</p>
              <h3 className="section__title">How far out it stays useful</h3>
            </div>
          </div>
          <div className="chart-card">
            <ResponsiveContainer width="100%" height={Math.max(200, horizons.length * 42)}>
              <BarChart
                data={horizons}
                layout="vertical"
                margin={{ top: 4, right: 40, bottom: 4, left: 8 }}
              >
                <CartesianGrid stroke="var(--rule)" horizontal={false} />
                <XAxis
                  type="number"
                  domain={[0, 1]}
                  tick={{ fontSize: 11, fill: 'var(--ink-3)' }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value: unknown) => formatPercent(Number(value), 0)}
                />
                <YAxis
                  type="category"
                  dataKey="horizon"
                  width={60}
                  tick={{ fontSize: 11, fill: 'var(--ink-2)' }}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip
                  cursor={{ fill: 'var(--surface-sunk)' }}
                  contentStyle={{
                    background: 'var(--surface-raised)',
                    border: '1px solid var(--rule)',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  formatter={(value: unknown) => [formatPercent(Number(value)), 'Accuracy']}
                />
                <Bar dataKey="accuracy" radius={[0, 3, 3, 0]} isAnimationActive={false}>
                  {horizons.map((row) => (
                    <Cell key={row.horizon} fill={accuracyColour(row.accuracy)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      {/* --- Per-product scorecard -------------------------------------------- */}
      <section>
        <div className="section__head">
          <div>
            <p className="eyebrow">Per product</p>
            <h3 className="section__title">Where the forecast missed most</h3>
            <p className="section__note">
              Ranked by units mis-forecast, not by percentage — a large error on a product selling
              two units a week is not the row to look at first.
            </p>
          </div>
        </div>

        <div className="card" style={{ overflow: 'hidden' }}>
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Product</th>
                  <th scope="col" className="numeric">
                    Actual
                  </th>
                  <th scope="col" className="numeric">
                    Forecast
                  </th>
                  <th scope="col" className="numeric">
                    Units missed
                  </th>
                  <th scope="col" className="numeric">
                    Accuracy
                  </th>
                  <th scope="col" className="numeric">
                    Bias
                  </th>
                  <th scope="col" className="numeric">
                    In range
                  </th>
                </tr>
              </thead>
              <tbody>
                {visibleSkus.map((row) => {
                  const skuAccuracy = row.accuracy;
                  return (
                    <tr key={row.sku_id} style={{ cursor: 'default' }}>
                      <td>
                        <div className="sku-cell">
                          <span className="sku-cell__id">{row.sku_id}</span>
                          <span className="sku-cell__meta">{row.category}</span>
                        </div>
                      </td>
                      <td className="numeric">{formatUnits(row.actual)}</td>
                      <td className="numeric">{formatUnits(row.predicted)}</td>
                      <td className="numeric">{formatUnits(row.abs_error)}</td>
                      <td
                        className="numeric"
                        style={{
                          color:
                            skuAccuracy === null ? 'var(--ink-3)' : accuracyColour(skuAccuracy),
                          fontWeight: 600,
                        }}
                      >
                        {skuAccuracy === null ? '—' : formatPercent(skuAccuracy)}
                      </td>
                      <td className="numeric">
                        {row.bias === null ? '—' : formatSignedPercent(row.bias)}
                      </td>
                      <td className="numeric">{formatPercent(row.coverage, 0)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {holdout.skus.length > TABLE_PREVIEW ? (
            <div className="toolbar" style={{ borderTop: '1px solid var(--rule)' }}>
              <button
                type="button"
                className="button button--ghost"
                onClick={() => setShowAll((value) => !value)}
              >
                {showAll
                  ? `Show top ${TABLE_PREVIEW}`
                  : `Show all ${holdout.skus.length} products`}
              </button>
            </div>
          ) : null}
        </div>
      </section>
    </div>
  );
}
