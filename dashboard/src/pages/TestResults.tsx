/**
 * Full exploration of an uploaded-actuals evaluation — the "real testing
 * data" counterpart to the Risk Dashboard: every scored product, not just a
 * top-few highlight reel, reached from the "Open full test results" button
 * on the Demand Forecast page.
 *
 * Deliberately not a persisted, bookmarkable resource: the result lives only
 * in navigation state (the upload that produced it lives in the browser
 * doing the upload, not on the server), so landing here directly - a refresh,
 * a shared link - shows an empty state pointing back to where to start.
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
import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

import type { ErrorContributor, EvaluationResponse } from '../lib/api';
import { EvaluationSummary } from '../components/EvaluationSummary';
import { Empty } from '../components/States';
import { formatPercent, formatUnits, formatWape } from '../lib/format';

interface LocationState {
  result?: EvaluationResponse;
  intervalTarget?: number;
}

type SortKey = 'share_of_total_error' | 'total_actual' | 'total_forecast' | 'sku_id';

function sortContributors(rows: ErrorContributor[], key: SortKey): ErrorContributor[] {
  const sorted = [...rows];
  if (key === 'sku_id') {
    sorted.sort((left, right) => left.sku_id.localeCompare(right.sku_id));
  } else {
    sorted.sort((left, right) => right[key] - left[key]);
  }
  return sorted;
}

export function TestResults() {
  const location = useLocation();
  const { result, intervalTarget } = (location.state as LocationState | null) ?? {};
  const [sortKey, setSortKey] = useState<SortKey>('share_of_total_error');

  if (!result) {
    return (
      <section className="section">
        <Empty
          mark="↑"
          title="No test results to show"
          body="Upload a file of actual sales on the Demand Forecast page to see the full breakdown here."
          action={
            <Link to="/demand-forecast" className="button">
              Go to Demand Forecast
            </Link>
          }
        />
      </section>
    );
  }

  const horizonData = result.by_horizon.map((row) => ({
    horizon: row.key,
    model: row.wape,
    baseline: row.baseline_wape,
  }));

  const categoryData = [...result.by_category].sort((left, right) => right.wape - left.wape);
  const contributors = sortContributors(result.worst_contributors, sortKey);

  return (
    <>
      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Real testing data</p>
          <h2 className="section__title">Full test results</h2>
          <p className="section__note">
            Every product your upload could be matched against, worst error first.
          </p>
        </div>
        <EvaluationSummary result={result} intervalTarget={intervalTarget ?? 0.8} />
      </section>

      <section className="section">
        <div className="two-col">
          <div className="chart-card">
            <p className="eyebrow">Accuracy by how far ahead</p>
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
                  className="chart-legend__swatch"
                  style={{ background: 'var(--harbour-700)' }}
                />
                Model
              </span>
              <span className="chart-legend__item">
                <span
                  className="chart-legend__swatch"
                  style={{ background: 'var(--series-baseline)' }}
                />
                Seasonal-naive
              </span>
            </div>
          </div>

          <div className="chart-card">
            <p className="eyebrow">Accuracy by category</p>
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
                  dataKey="key"
                  tick={{ fontSize: 9.5, fill: 'var(--ink-2)' }}
                  stroke="var(--rule)"
                  width={128}
                />
                <Tooltip
                  formatter={(value: unknown) => [formatWape(Number(value)), 'WAPE']}
                  contentStyle={{
                    background: 'var(--harbour-900)',
                    border: 'none',
                    borderRadius: 8,
                    fontSize: 12,
                    color: '#fff',
                  }}
                  itemStyle={{ color: '#fff' }}
                />
                <Bar dataKey="wape" radius={[0, 3, 3, 0]} barSize={15}>
                  {categoryData.map((entry) => (
                    <Cell key={entry.key} fill="var(--harbour-700)" />
                  ))}
                  <LabelList
                    dataKey="wape"
                    position="right"
                    formatter={(value: unknown) => formatWape(Number(value))}
                    style={{ fontSize: 9.5, fill: 'var(--ink-3)' }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Every product scored</p>
          <h2 className="section__title">Where the error concentrates</h2>
        </div>

        <div className="toolbar">
          <select
            className="select"
            aria-label="Sort by"
            value={sortKey}
            onChange={(event) => setSortKey(event.target.value as SortKey)}
          >
            <option value="share_of_total_error">Sort by share of error</option>
            <option value="total_actual">Sort by actual units</option>
            <option value="total_forecast">Sort by forecast units</option>
            <option value="sku_id">Sort by product code</option>
          </select>
          <span className="toolbar__count">{contributors.length} products</span>
        </div>

        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th>Product</th>
                <th className="numeric">Weeks scored</th>
                <th className="numeric">Actual</th>
                <th className="numeric">Forecast</th>
                <th className="numeric">Share of error</th>
              </tr>
            </thead>
            <tbody>
              {contributors.map((row) => (
                <tr key={row.sku_id}>
                  <td className="mono">{row.sku_id}</td>
                  <td className="numeric">{formatUnits(row.n_observations)}</td>
                  <td className="numeric">{formatUnits(row.total_actual)}</td>
                  <td className="numeric">{formatUnits(row.total_forecast)}</td>
                  <td className="numeric at-stake">{formatPercent(row.share_of_total_error, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
