/**
 * How much to trust the forecast.
 *
 * Exposed to the ops team rather than hidden in a report, on the principle that
 * anyone being asked to act on a number is entitled to see its track record.
 * Every figure here comes from the rolling-origin backtest: forecasts made using
 * only what was knowable at the time, scored against what actually happened.
 *
 * The comparison against the seasonal-naive baseline is shown first and always,
 * including when the baseline wins — brief section 7.1 treats that as a finding
 * to report rather than a failure to hide.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import type { AccuracySummary } from '../lib/api';
import { formatPercent, formatSignedPercent, formatWape } from '../lib/format';
import { AccuracyUpload } from './AccuracyUpload';

interface AccuracyPanelProps {
  accuracy: AccuracySummary;
}

const MODEL_LABELS: Record<string, string> = {
  adaptive_ensemble: 'Adaptive Ensemble',
  lightgbm: 'LightGBM',
  seasonal_naive: 'Seasonal-naive baseline',
  naive: 'Last-value naive',
};

const REGIME_LABELS: Record<string, string> = {
  steady: 'Steady',
  intermittent: 'Intermittent',
  volatile: 'Volatile',
  new: 'Newly launched',
};

export function AccuracyPanel({ accuracy }: AccuracyPanelProps) {
  const rows = [
    {
      key: 'seasonal_naive',
      wape: accuracy.baseline_wape,
      note: 'Same week last year. The bar every model must clear.',
    },
    {
      key: 'naive',
      wape: accuracy.naive_wape,
      note: 'Carry this week forward unchanged. Shown for context.',
    },
    {
      key: 'lightgbm',
      wape: accuracy.gbm_wape,
      note: 'Global gradient-boosted model across all products.',
    },
    ...(accuracy.ensemble_wape !== null
      ? [
          {
            key: 'adaptive_ensemble',
            wape: accuracy.ensemble_wape,
            note: 'Regime-routed blend of the three component models.',
          },
        ]
      : []),
  ].sort((left, right) => left.wape - right.wape);

  const horizonData = accuracy.by_horizon.map((row) => ({
    horizon: `+${row.horizon as number}`,
    model: Number(row.ensemble_wape ?? row.wape ?? 0),
    baseline: Number(row.baseline_wape ?? 0),
  }));

  const categoryData = accuracy.by_category
    .map((row) => ({
      category: String(row.category),
      model: Number(row.ensemble_wape ?? row.wape ?? 0),
      baseline: Number(row.baseline_wape ?? 0),
    }))
    .sort((left, right) => left.model - right.model);

  const coverageGap = accuracy.interval_coverage - accuracy.interval_coverage_target;
  const coverageIsGood = Math.abs(coverageGap) <= 0.05;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)' }}>
      <div className="callout">
        <span className="callout__mark" aria-hidden="true">
          ◆
        </span>
        <span>
          Every figure below comes from <strong>rolling-origin backtesting</strong> across{' '}
          <strong>{accuracy.folds} folds</strong> and{' '}
          <strong>{accuracy.test_observations.toLocaleString('en-IN')}</strong> out-of-sample
          product-weeks. At each fold the model was trained only on data whose outcome was
          already known at that date, then asked to forecast the following{' '}
          {accuracy.horizon_weeks} weeks. Nothing here is scored on data the model trained on.
        </span>
      </div>

      {/* --- Model comparison ------------------------------------------------ */}
      <section>
        <div className="section__head">
          <div>
            <p className="eyebrow">Model selection</p>
            <h3 className="section__title">Which forecast ships, and why</h3>
          </div>
        </div>
        <div className="card" style={{ overflow: 'hidden' }}>
          <table className="model-table">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col" className="numeric">
                  WAPE
                </th>
                <th scope="col" className="numeric">
                  vs baseline
                </th>
                <th scope="col">What it is</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const isSelected = row.key === accuracy.selected_model;
                const improvement =
                  accuracy.baseline_wape > 0
                    ? (accuracy.baseline_wape - row.wape) / accuracy.baseline_wape
                    : 0;
                return (
                  <tr key={row.key} data-selected={isSelected}>
                    <td>
                      {MODEL_LABELS[row.key] ?? row.key}
                      {isSelected ? <span className="winner-tag">selected</span> : null}
                    </td>
                    <td className="numeric">{formatWape(row.wape)}</td>
                    <td
                      className="numeric"
                      style={{
                        color:
                          row.key === 'seasonal_naive'
                            ? 'var(--ink-3)'
                            : improvement > 0
                              ? 'var(--signal-ok)'
                              : 'var(--signal-critical)',
                      }}
                    >
                      {row.key === 'seasonal_naive' ? '—' : formatSignedPercent(-improvement)}
                    </td>
                    <td style={{ color: 'var(--ink-3)', fontSize: 12 }}>{row.note}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* --- Honesty checks -------------------------------------------------- */}
      <div className="stat-grid">
        <div className="stat">
          <div className="stat__label">Forecast bias</div>
          <div className="stat__value tabular">{formatSignedPercent(accuracy.bias_relative)}</div>
          <div className="stat__hint">
            {Math.abs(accuracy.bias_relative) < 0.05
              ? 'balanced — no systematic over- or under-forecasting'
              : accuracy.bias_relative < 0
                ? 'runs light; risks under-ordering'
                : 'runs heavy; risks over-ordering'}
          </div>
        </div>
        <div className="stat">
          <div className="stat__label">Interval coverage</div>
          <div
            className="stat__value tabular"
            style={{ color: coverageIsGood ? 'var(--signal-ok)' : 'var(--signal-warn-ink)' }}
          >
            {formatPercent(accuracy.interval_coverage)}
          </div>
          <div className="stat__hint">
            target {formatPercent(accuracy.interval_coverage_target, 0)} —{' '}
            {coverageIsGood
              ? 'well calibrated'
              : coverageGap < 0
                ? 'intervals are too narrow'
                : 'intervals are wider than needed'}
          </div>
        </div>
        <div className="stat">
          <div className="stat__label">Backtest folds</div>
          <div className="stat__value tabular">{accuracy.folds}</div>
          <div className="stat__hint">rolling origins, never a random split</div>
        </div>
        <div className="stat">
          <div className="stat__label">Out-of-sample rows</div>
          <div className="stat__value tabular">
            {accuracy.test_observations.toLocaleString('en-IN')}
          </div>
          <div className="stat__hint">product-weeks scored</div>
        </div>
      </div>

      {/* --- Breakdown charts ------------------------------------------------ */}
      <div className="two-col">
        <div className="chart-card">
          <p className="eyebrow">Accuracy by how far ahead</p>
          <p className="section__note" style={{ marginBottom: 'var(--space-3)' }}>
            Error should grow with distance. A flat line across the horizon would suggest the
            model is not really using recent information.
          </p>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={horizonData} margin={{ top: 12, right: 8, bottom: 4, left: -18 }}>
              <CartesianGrid stroke="var(--rule)" strokeWidth={0.6} vertical={false} />
              <XAxis
                dataKey="horizon"
                tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                stroke="var(--rule)"
              />
              <YAxis
                tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`}
                tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                stroke="var(--rule)"
                width={46}
              />
              <Tooltip
                // Recharts types these callbacks with wide unions, so the
                // parameters are taken as `unknown` and narrowed here rather
                // than fighting the library's generics.
                formatter={(value: unknown, name: unknown) => [
                  formatWape(Number(value)),
                  name === 'model' ? 'Model' : 'Baseline',
                ]}
                labelFormatter={(label: unknown) => `${String(label)} weeks ahead`}
                contentStyle={{
                  background: 'var(--harbour-900)',
                  border: 'none',
                  borderRadius: 8,
                  fontSize: 12,
                  color: '#fff',
                }}
                itemStyle={{ color: '#fff' }}
                labelStyle={{ color: 'rgba(247,244,238,0.7)' }}
              />
              <Bar dataKey="baseline" fill="var(--series-baseline)" radius={[2, 2, 0, 0]} />
              <Bar dataKey="model" fill="var(--harbour-700)" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
          <div className="chart-legend">
            <span className="chart-legend__item">
              <span
                className="grid-legend__swatch"
                style={{ background: 'var(--harbour-700)', borderRadius: 2 }}
              />
              Model
            </span>
            <span className="chart-legend__item">
              <span
                className="grid-legend__swatch"
                style={{ background: 'var(--series-baseline)', borderRadius: 2 }}
              />
              Seasonal-naive
            </span>
          </div>
        </div>

        <div className="chart-card">
          <p className="eyebrow">Accuracy by category</p>
          <p className="section__note" style={{ marginBottom: 'var(--space-3)' }}>
            Model error per category, lowest first. The categories at the bottom are the
            hardest to forecast and the ones worth investigating next.
          </p>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart
              data={categoryData}
              layout="vertical"
              margin={{ top: 6, right: 42, bottom: 4, left: 8 }}
            >
              <CartesianGrid stroke="var(--rule)" strokeWidth={0.6} horizontal={false} />
              <XAxis
                type="number"
                tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`}
                tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                stroke="var(--rule)"
              />
              <YAxis
                type="category"
                dataKey="category"
                tick={{ fontSize: 9.5, fill: 'var(--ink-2)' }}
                stroke="var(--rule)"
                width={128}
              />
              <Tooltip
                formatter={(value: unknown) => [formatWape(Number(value)), 'Model WAPE']}
                contentStyle={{
                  background: 'var(--harbour-900)',
                  border: 'none',
                  borderRadius: 8,
                  fontSize: 12,
                  color: '#fff',
                }}
                itemStyle={{ color: '#fff' }}
                labelStyle={{ color: 'rgba(247,244,238,0.7)' }}
              />
              <Bar dataKey="model" radius={[0, 3, 3, 0]} barSize={15}>
                {categoryData.map((entry) => (
                  <Cell key={entry.category} fill="var(--harbour-700)" />
                ))}
                <LabelList
                  dataKey="model"
                  position="right"
                  formatter={(value: unknown) => formatWape(Number(value))}
                  style={{ fontSize: 9.5, fill: 'var(--ink-3)' }}
                />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* --- Regime breakdown ------------------------------------------------ */}
      {accuracy.by_regime.length > 0 ? (
        <section>
          <div className="section__head">
            <div>
              <p className="eyebrow">Where the ensemble earns its keep</p>
              <h3 className="section__title">Accuracy by demand regime</h3>
              <p className="section__note">
                Products are routed to different component models depending on how they sell.
                This is where routing helps and where it does not.
              </p>
            </div>
          </div>
          <div className="card" style={{ overflow: 'hidden' }}>
            <table className="model-table">
              <thead>
                <tr>
                  <th scope="col">Demand regime</th>
                  <th scope="col" className="numeric">
                    Product-weeks
                  </th>
                  <th scope="col" className="numeric">
                    Baseline
                  </th>
                  <th scope="col" className="numeric">
                    LightGBM
                  </th>
                  <th scope="col" className="numeric">
                    Ensemble
                  </th>
                </tr>
              </thead>
              <tbody>
                {accuracy.by_regime.map((row) => {
                  const ensembleWape = Number(row.ensemble_wape ?? 0);
                  const gbmWape = Number(row.gbm_wape ?? 0);
                  const better = ensembleWape < gbmWape;
                  return (
                    <tr key={String(row.regime)}>
                      <td>{REGIME_LABELS[String(row.regime)] ?? String(row.regime)}</td>
                      <td className="numeric">
                        {Number(row.n_observations ?? 0).toLocaleString('en-IN')}
                      </td>
                      <td className="numeric" style={{ color: 'var(--ink-3)' }}>
                        {formatWape(Number(row.baseline_wape ?? 0))}
                      </td>
                      <td className="numeric">{formatWape(gbmWape)}</td>
                      <td
                        className="numeric"
                        style={{
                          fontWeight: 600,
                          color: better ? 'var(--signal-ok)' : 'var(--ink)',
                        }}
                      >
                        {formatWape(ensembleWape)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <AccuracyUpload intervalTarget={accuracy.interval_coverage_target} />
    </div>
  );
}
