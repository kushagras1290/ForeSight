/**
 * Seasonality Dashboard: the Adaptive Ensemble's actual fitted seasonal
 * index, global and per category. Reads `/api/seasonality`, backed by
 * `artifacts/seasonal_profile.json` (written by
 * `scripts/12_extract_seasonal_profile.py`), extracted straight from the
 * ensemble that already blends this into every forecast — not a separate
 * static EDA figure.
 */

import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  ComposedChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { ApiError, api, type SeasonalIndexPoint } from '../lib/api';
import { useCoreData } from '../components/Layout';
import { Empty, ErrorState, LoadingChart } from '../components/States';
import { NoDataYet } from '../components/NoDataYet';

interface ChartRow {
  iso_week: number;
  global: number | null;
  category: number | null;
}

function SeasonalityTooltip({
  active,
  label,
  payload,
}: {
  active?: boolean;
  label?: number;
  payload?: Array<{ dataKey?: string; value?: number }>;
}) {
  if (!active || label === undefined || !payload?.length) return null;
  return (
    <div className="tooltip">
      <div className="tooltip__title">Week {label} of the year</div>
      {payload.map((entry) =>
        typeof entry.value === 'number' ? (
          <div className="tooltip__row" key={entry.dataKey}>
            <span>{entry.dataKey === 'global' ? 'All categories' : 'This category'}</span>
            <strong>{entry.value.toFixed(2)}×</strong>
          </div>
        ) : null,
      )}
    </div>
  );
}

export function SeasonalityDashboard() {
  const { core, ready } = useCoreData();
  const [category, setCategory] = useState('');
  const [globalIndex, setGlobalIndex] = useState<SeasonalIndexPoint[]>([]);
  const [byCategory, setByCategory] = useState<Record<string, SeasonalIndexPoint[]>>({});
  const [available, setAvailable] = useState(true);
  const [reason, setReason] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    if (!ready?.ready) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .seasonality()
      .then((result) => {
        if (cancelled) return;
        setAvailable(result.available);
        setReason(result.reason);
        setGlobalIndex(result.global_index);
        setByCategory(result.by_category);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof ApiError ? cause.message : 'Could not load seasonality data.');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [reloadToken, ready?.ready]);

  const chartData: ChartRow[] = useMemo(() => {
    const categoryPoints = category ? (byCategory[category] ?? []) : [];
    const categoryByWeek = new Map(categoryPoints.map((point) => [point.iso_week, point.index]));
    return globalIndex.map((point) => ({
      iso_week: point.iso_week,
      global: point.index,
      category: categoryByWeek.get(point.iso_week) ?? null,
    }));
  }, [globalIndex, byCategory, category]);

  if (!ready?.ready) {
    return (
      <section className="section">
        <NoDataYet
          title="No seasonality data trained yet"
          body="This instance has no seasonal profile fitted. Once real data has been run through the pipeline, this page shows the model's actual seasonal curve."
          missing={ready?.artifacts_missing}
        />
      </section>
    );
  }

  return (
    <section className="section">
      <div className="section__head">
        <p className="eyebrow">Seasonality dashboard</p>
        <h2 className="section__title">The model's actual seasonal curve</h2>
        <p className="section__note">
          The week-of-year index the Adaptive Ensemble blends into every forecast — 1.0 is an
          average week.
        </p>
      </div>

      {error ? (
        <ErrorState message={error} onRetry={() => setReloadToken((token) => token + 1)} />
      ) : loading ? (
        <LoadingChart height={300} />
      ) : !available ? (
        <Empty
          title="Not extracted yet"
          body={reason ?? 'Run scripts/12_extract_seasonal_profile.py to populate this page.'}
        />
      ) : (
        <>
          <div className="toolbar">
            <select
              className="select"
              aria-label="Overlay a category"
              value={category}
              onChange={(event) => setCategory(event.target.value)}
            >
              <option value="">Global index only</option>
              {(core?.categories ?? []).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>

          <div className="chart-card">
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={chartData} margin={{ top: 16, right: 8, bottom: 4, left: -14 }}>
                <CartesianGrid stroke="var(--rule)" strokeWidth={0.6} vertical={false} />
                <XAxis
                  dataKey="iso_week"
                  tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                  stroke="var(--rule)"
                  label={{ value: 'ISO week', position: 'insideBottom', offset: -2, fontSize: 10 }}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: 'var(--ink-3)' }}
                  stroke="var(--rule)"
                  width={44}
                  domain={['auto', 'auto']}
                />
                <Tooltip content={<SeasonalityTooltip />} cursor={{ stroke: 'var(--rule-strong)' }} />
                <Line
                  dataKey="global"
                  name="global"
                  stroke="var(--series-history)"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
                {category ? (
                  <Line
                    dataKey="category"
                    name="category"
                    stroke="var(--harbour-500)"
                    strokeWidth={2}
                    strokeDasharray="4 3"
                    dot={false}
                    isAnimationActive={false}
                    connectNulls
                  />
                ) : null}
              </ComposedChart>
            </ResponsiveContainer>
            <div className="chart-legend">
              <span className="chart-legend__item">
                <span className="chart-legend__line" style={{ borderColor: 'var(--series-history)' }} />
                All categories
              </span>
              {category ? (
                <span className="chart-legend__item">
                  <span className="chart-legend__line" style={{ borderColor: 'var(--harbour-500)' }} />
                  {category}
                </span>
              ) : null}
            </div>
          </div>
        </>
      )}
    </section>
  );
}
