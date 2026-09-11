/**
 * Portfolio-wide sales trend: weekly units and revenue, optionally scoped to
 * one category, plus how revenue splits across categories. Reads `/api/sales/
 * trend`, which aggregates the same weekly panel every SKU-level view reads
 * from — so nothing here can disagree with a single product's own history.
 */

import { useEffect, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  ComposedChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { ApiError, api, type SalesTrendResponse } from '../lib/api';
import { useCoreData } from '../components/Layout';
import { ErrorState, LoadingChart } from '../components/States';
import { NoDataYet } from '../components/NoDataYet';
import { formatDate, formatInr, formatShortDate, formatUnits } from '../lib/format';

interface TrendTooltipEntry {
  dataKey?: string | number;
  value?: number;
}

function TrendTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TrendTooltipEntry[];
  label?: string;
}) {
  if (!active || !payload?.length || !label) return null;
  const units = payload.find((entry) => entry.dataKey === 'units')?.value;

  return (
    <div className="tooltip">
      <div className="tooltip__title">Week of {formatDate(label)}</div>
      {typeof units === 'number' ? (
        <div className="tooltip__row">
          <span>Units</span>
          <strong>{formatUnits(units)}</strong>
        </div>
      ) : null}
    </div>
  );
}

function CategoryTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: { category: string; revenue: number; units: number } }>;
}) {
  const row = active && payload?.length ? payload[0]?.payload : undefined;
  if (!row) return null;

  return (
    <div className="tooltip">
      <div className="tooltip__title">{row.category}</div>
      <div className="tooltip__row">
        <span>Revenue</span>
        <strong>{formatInr(row.revenue)}</strong>
      </div>
      <div className="tooltip__row">
        <span>Units</span>
        <strong>{formatUnits(row.units)}</strong>
      </div>
    </div>
  );
}

export function SalesAnalytics() {
  const { core, ready } = useCoreData();
  const [category, setCategory] = useState('');
  const [trend, setTrend] = useState<SalesTrendResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    if (!ready?.ready) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .salesTrend({ category: category || undefined })
      .then((result) => {
        if (!cancelled) setTrend(result);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof ApiError ? cause.message : 'Could not load the sales trend.');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [category, reloadToken, ready?.ready]);

  if (!ready?.ready) {
    return (
      <section className="section">
        <NoDataYet
          title="No sales data trained yet"
          body="This instance has no data to analyse. Once real extracts are run through the pipeline, weekly trends and category breakdowns will show up here."
          missing={ready?.artifacts_missing}
        />
      </section>
    );
  }

  return (
    <>
      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Sales analytics</p>
          <h2 className="section__title">Weekly demand across the portfolio</h2>
          <p className="section__note">
            Units and revenue, aggregated from the same weekly panel every forecast is built on.
          </p>
        </div>

        <div className="toolbar">
          <select
            className="select"
            aria-label="Filter by category"
            value={category}
            onChange={(event) => setCategory(event.target.value)}
          >
            <option value="">All categories</option>
            {(core?.categories ?? []).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>

        {error ? (
          <ErrorState message={error} onRetry={() => setReloadToken((token) => token + 1)} />
        ) : loading || !trend ? (
          <LoadingChart height={320} />
        ) : (
          <div className="chart-card">
            <p className="eyebrow">
              Units and revenue per week{category ? ` — ${category}` : ''}
            </p>
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart
                data={trend.weeks}
                margin={{ top: 16, right: 8, bottom: 4, left: -14 }}
              >
                <CartesianGrid stroke="var(--rule)" strokeWidth={0.6} vertical={false} />
                <XAxis
                  dataKey="week_starting"
                  tickFormatter={formatShortDate}
                  tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                  stroke="var(--rule)"
                  minTickGap={44}
                />
                <YAxis
                  yAxisId="units"
                  tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                  stroke="var(--rule)"
                  width={52}
                />
                <Tooltip content={<TrendTooltip />} cursor={{ stroke: 'var(--rule-strong)' }} />
                <Line
                  yAxisId="units"
                  dataKey="units"
                  name="units"
                  stroke="var(--series-history)"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
            <div className="chart-legend">
              <span className="chart-legend__item">
                <span
                  className="chart-legend__line"
                  style={{ borderColor: 'var(--series-history)' }}
                />
                Units sold per week
              </span>
            </div>
          </div>
        )}
      </section>

      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Where revenue sits</p>
          <h2 className="section__title">Revenue by category</h2>
        </div>

        {!loading && trend ? (
          <div className="chart-card">
            <ResponsiveContainer width="100%" height={Math.max(180, trend.by_category.length * 34)}>
              <BarChart
                data={trend.by_category}
                layout="vertical"
                margin={{ top: 4, right: 24, bottom: 4, left: 8 }}
              >
                <CartesianGrid stroke="var(--rule)" strokeWidth={0.6} horizontal={false} />
                <XAxis
                  type="number"
                  tickFormatter={(value: number) => formatInr(value, 1)}
                  tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                  stroke="var(--rule)"
                />
                <YAxis
                  type="category"
                  dataKey="category"
                  width={140}
                  tick={{ fontSize: 11, fill: 'var(--ink-2)' }}
                  stroke="var(--rule)"
                />
                <Tooltip content={<CategoryTooltip />} cursor={{ fill: 'var(--surface-sunk)' }} />
                <Bar dataKey="revenue" fill="var(--harbour-500)" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <LoadingChart height={220} />
        )}
      </section>
    </>
  );
}
