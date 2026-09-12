/**
 * Promotion Dashboard: custom model 5's fitted promotional uplift and
 * post-promotion dip, per category. Reads `/api/promotions`, backed by
 * `artifacts/promotion_response.json` (written by
 * `scripts/11_fit_promotion_response.py`) — a real fitted curve, not a
 * static EDA figure.
 */

import { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import { ApiError, api, type PromotionCategoryStat } from '../lib/api';
import { useCoreData } from '../components/Layout';
import { Empty, ErrorState, LoadingChart } from '../components/States';
import { NoDataYet } from '../components/NoDataYet';
import { formatPercent } from '../lib/format';

const REFERENCE_DISCOUNT = '20pct';

function UpliftTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: { category: string; uplift: number } }>;
}) {
  const row = active && payload?.length ? payload[0]?.payload : undefined;
  if (!row) return null;
  return (
    <div className="tooltip">
      <div className="tooltip__title">{row.category}</div>
      <div className="tooltip__row">
        <span>Uplift at 20% off</span>
        <strong>{row.uplift.toFixed(2)}×</strong>
      </div>
    </div>
  );
}

export function PromotionDashboard() {
  const { ready } = useCoreData();
  const [category, setCategory] = useState('');
  const [pooled, setPooled] = useState<PromotionCategoryStat | null>(null);
  const [categories, setCategories] = useState<PromotionCategoryStat[]>([]);
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
      .promotions()
      .then((result) => {
        if (cancelled) return;
        setAvailable(result.available);
        setReason(result.reason);
        setPooled(result.pooled);
        setCategories(result.categories);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof ApiError ? cause.message : 'Could not load promotion data.');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [reloadToken, ready?.ready]);

  if (!ready?.ready) {
    return (
      <section className="section">
        <NoDataYet
          title="No promotion data trained yet"
          body="This instance has no promotional response fitted. Once real data has been run through the pipeline, this page shows the fitted uplift curve."
          missing={ready?.artifacts_missing}
        />
      </section>
    );
  }

  const selected = categories.find((entry) => entry.category === category) ?? null;
  const chartData = categories
    .map((entry) => ({
      category: entry.category,
      uplift: entry.uplift_by_discount[REFERENCE_DISCOUNT] ?? 1.0,
    }))
    .sort((a, b) => b.uplift - a.uplift);

  return (
    <section className="section">
      <div className="section__head">
        <p className="eyebrow">Promotion dashboard</p>
        <h2 className="section__title">What a promotion actually buys you</h2>
        <p className="section__note">
          Custom model 5's fitted promotional uplift and post-promotion dip, per category.
        </p>
      </div>

      {error ? (
        <ErrorState message={error} onRetry={() => setReloadToken((token) => token + 1)} />
      ) : loading ? (
        <LoadingChart height={260} />
      ) : !available ? (
        <Empty
          title="Not fitted yet"
          body={reason ?? 'Run scripts/11_fit_promotion_response.py to populate this page.'}
        />
      ) : (
        <>
          <div className="toolbar">
            <select
              className="select"
              aria-label="Focus on a category"
              value={category}
              onChange={(event) => setCategory(event.target.value)}
            >
              <option value="">All categories — overview</option>
              {categories.map((entry) => (
                <option key={entry.category} value={entry.category}>
                  {entry.category}
                </option>
              ))}
            </select>
          </div>

          {!category ? (
            <div className="chart-card">
              <p className="eyebrow">Uplift at 20% off, by category</p>
              <ResponsiveContainer width="100%" height={Math.max(180, chartData.length * 34)}>
                <BarChart
                  data={chartData}
                  layout="vertical"
                  margin={{ top: 4, right: 24, bottom: 4, left: 8 }}
                >
                  <CartesianGrid stroke="var(--rule)" strokeWidth={0.6} horizontal={false} />
                  <XAxis
                    type="number"
                    tickFormatter={(value: number) => `${value.toFixed(1)}×`}
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
                  <Tooltip content={<UpliftTooltip />} cursor={{ fill: 'var(--surface-sunk)' }} />
                  <Bar dataKey="uplift" fill="var(--harbour-500)" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
              {pooled ? (
                <p className="section__note" style={{ marginTop: 'var(--space-3)' }}>
                  Pooled across every category: {pooled.uplift_by_discount[REFERENCE_DISCOUNT]?.toFixed(2)}×
                  at 20% off, dip factor {pooled.post_promo_dip_factor.toFixed(2)} the week after.
                </p>
              ) : null}
            </div>
          ) : selected ? (
            <div className="stat-grid">
              {Object.entries(selected.uplift_by_discount).map(([depth, uplift]) => (
                <div className="stat" key={depth}>
                  <div className="stat__label">Uplift at {depth.replace('pct', '%')} off</div>
                  <div className="stat__value tabular">{uplift.toFixed(2)}×</div>
                </div>
              ))}
              <div className="stat">
                <div className="stat__label">Post-promo dip</div>
                <div className="stat__value tabular">
                  {formatPercent(1 - selected.post_promo_dip_factor, 0)} below baseline
                </div>
                <div className="stat__hint">for the week(s) right after the promotion ends</div>
              </div>
              <div className="stat">
                <div className="stat__label">Curve fitted</div>
                <div className="stat__value" style={{ fontSize: 14 }}>
                  {selected.has_own_curve ? 'This category, directly' : 'Pooled (too few promotions)'}
                </div>
              </div>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}
