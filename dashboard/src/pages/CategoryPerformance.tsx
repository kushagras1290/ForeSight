/**
 * Category Performance Dashboard: revenue/units by category, and — once one
 * is picked — that category's weekly trend plus its risk-action mix.
 * Composed entirely from two existing endpoints (`/api/sales/trend` and
 * `/api/risk`, both already scoped by `category`); no new endpoint needed.
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

import { ACTION_ORDER, ACTION_STYLE, ApiError, api, type RiskAction, type SalesTrendResponse } from '../lib/api';
import { useCoreData } from '../components/Layout';
import { ErrorState, LoadingChart } from '../components/States';
import { NoDataYet } from '../components/NoDataYet';
import { formatDate, formatInr, formatShortDate, formatUnits } from '../lib/format';

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

function TrendTooltip({
  active,
  label,
  payload,
}: {
  active?: boolean;
  label?: string;
  payload?: Array<{ value?: number }>;
}) {
  if (!active || !label || !payload?.length) return null;
  return (
    <div className="tooltip">
      <div className="tooltip__title">Week of {formatDate(label)}</div>
      <div className="tooltip__row">
        <span>Units</span>
        <strong>{formatUnits(payload[0]?.value ?? 0)}</strong>
      </div>
    </div>
  );
}

export function CategoryPerformance() {
  const { core, ready } = useCoreData();
  const [category, setCategory] = useState('');
  const [trend, setTrend] = useState<SalesTrendResponse | null>(null);
  const [actionCounts, setActionCounts] = useState<Record<RiskAction, number> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    if (!ready?.ready) return;
    let cancelled = false;
    setLoading(true);
    setError(null);

    const requests: [ReturnType<typeof api.salesTrend>, ReturnType<typeof api.risk> | null] = [
      api.salesTrend({ category: category || undefined }),
      category ? api.risk({ category, limit: 1000 }) : null,
    ];

    Promise.all([requests[0], requests[1] ?? Promise.resolve(null)])
      .then(([trendResult, riskRows]) => {
        if (cancelled) return;
        setTrend(trendResult);
        if (riskRows) {
          const counts: Record<RiskAction, number> = {
            reorder_now: 0,
            markdown_clear: 0,
            watch_volatile: 0,
            healthy: 0,
          };
          for (const row of riskRows) counts[row.action] += 1;
          setActionCounts(counts);
        } else {
          setActionCounts(null);
        }
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof ApiError ? cause.message : 'Could not load category data.');
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
          title="No category data trained yet"
          body="This instance has no data to break down by category. Once real extracts are run through the pipeline, category totals and risk mix will show up here."
          missing={ready?.artifacts_missing}
        />
      </section>
    );
  }

  return (
    <>
      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Category performance</p>
          <h2 className="section__title">Where revenue and risk sit, by category</h2>
        </div>

        <div className="toolbar">
          <select
            className="select"
            aria-label="Focus on a category"
            value={category}
            onChange={(event) => setCategory(event.target.value)}
          >
            <option value="">All categories — overview</option>
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
          <LoadingChart height={260} />
        ) : (
          <div className="chart-card">
            <p className="eyebrow">Revenue by category</p>
            <ResponsiveContainer
              width="100%"
              height={Math.max(180, trend.by_category.length * 34)}
            >
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
        )}
      </section>

      {category && trend ? (
        <section className="section">
          <div className="section__head">
            <p className="eyebrow">{category}</p>
            <h2 className="section__title">Weekly trend and risk mix</h2>
          </div>

          <div className="chart-card">
            <ResponsiveContainer width="100%" height={260}>
              <ComposedChart data={trend.weeks} margin={{ top: 16, right: 8, bottom: 4, left: -14 }}>
                <CartesianGrid stroke="var(--rule)" strokeWidth={0.6} vertical={false} />
                <XAxis
                  dataKey="week_starting"
                  tickFormatter={formatShortDate}
                  tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                  stroke="var(--rule)"
                  minTickGap={44}
                />
                <YAxis tick={{ fontSize: 10, fill: 'var(--ink-3)' }} stroke="var(--rule)" width={52} />
                <Tooltip content={<TrendTooltip />} cursor={{ stroke: 'var(--rule-strong)' }} />
                <Line
                  dataKey="units"
                  stroke="var(--series-history)"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {actionCounts ? (
            <div className="stat-grid" style={{ marginTop: 'var(--space-4)' }}>
              {ACTION_ORDER.map((action) => {
                const style = ACTION_STYLE[action];
                return (
                  <div className="stat" key={action}>
                    <div className="stat__label">{style.label}</div>
                    <div className="stat__value tabular" style={{ color: style.ink }}>
                      {actionCounts[action]} SKUs
                    </div>
                  </div>
                );
              })}
            </div>
          ) : null}
        </section>
      ) : null}
    </>
  );
}
