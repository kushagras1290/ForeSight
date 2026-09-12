/**
 * Product Performance Dashboard: a portfolio-wide, sortable leaderboard — the
 * entry point for "which products actually matter" that the old dashboard
 * didn't have (Product Details only ever showed one SKU at a time). Reads
 * `/api/products/performance`, which rolls up the same weekly panel and risk
 * table every other page reads from.
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { ACTION_STYLE, ApiError, api, type ProductPerformanceRow, type RiskAction } from '../lib/api';
import { useCoreData } from '../components/Layout';
import { ErrorState, Loading } from '../components/States';
import { NoDataYet } from '../components/NoDataYet';
import { Empty } from '../components/States';
import { formatInr, formatSignedPercent, formatUnits } from '../lib/format';

const SORT_OPTIONS: Array<{ value: string; label: string }> = [
  { value: 'revenue_desc', label: 'Revenue, high to low' },
  { value: 'units_desc', label: 'Units, high to low' },
  { value: 'trend_desc', label: 'Trending up' },
  { value: 'trend_asc', label: 'Trending down' },
  { value: 'value_at_stake_desc', label: 'Value at stake, high to low' },
];

function actionStyleFor(action: ProductPerformanceRow['action']) {
  return action === 'unknown' ? null : ACTION_STYLE[action as RiskAction];
}

export function ProductPerformance() {
  const { core, ready } = useCoreData();
  const navigate = useNavigate();

  const [category, setCategory] = useState('');
  const [sort, setSort] = useState('revenue_desc');
  const [rows, setRows] = useState<ProductPerformanceRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    if (!ready?.ready) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .productPerformance({ category: category || undefined, sort, limit: 500 })
      .then((result) => {
        if (!cancelled) setRows(result.rows);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof ApiError ? cause.message : 'Could not load performance data.');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [category, sort, reloadToken, ready?.ready]);

  if (!ready?.ready) {
    return (
      <section className="section">
        <NoDataYet
          title="No performance data trained yet"
          body="This instance has no forecast or risk data to rank. Once real data has been run through the pipeline, every product will show up here, ranked."
          missing={ready?.artifacts_missing}
        />
      </section>
    );
  }

  return (
    <section className="section">
      <div className="section__head">
        <p className="eyebrow">Product performance</p>
        <h2 className="section__title">Which products actually matter</h2>
        <p className="section__note">
          Revenue, units, recent trend, and current risk action — one row per SKU.
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

        <select
          className="select"
          aria-label="Sort by"
          value={sort}
          onChange={(event) => setSort(event.target.value)}
        >
          {SORT_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>

        <span className="toolbar__count" aria-live="polite">
          {loading ? 'Loading…' : `${rows.length} products`}
        </span>
      </div>

      {error ? (
        <ErrorState message={error} onRetry={() => setReloadToken((token) => token + 1)} />
      ) : loading ? (
        <Loading rows={8} label="Loading performance data" />
      ) : rows.length === 0 ? (
        <Empty title="No products in this view" body="Try a different category." />
      ) : (
        <div className="table-scroll">
          <table className="table">
            <caption className="visually-hidden">
              Per-SKU revenue, units, trend, and risk action, sorted per the toolbar above.
            </caption>
            <thead>
              <tr>
                <th scope="col">Product</th>
                <th scope="col">Action</th>
                <th scope="col" className="numeric">
                  Revenue
                </th>
                <th scope="col" className="numeric">
                  Units
                </th>
                <th scope="col" className="numeric">
                  Trend
                </th>
                <th scope="col" className="numeric">
                  At stake
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const style = actionStyleFor(row.action);
                return (
                  <tr
                    key={row.sku_id}
                    onClick={() => navigate(`/products/${row.sku_id}`)}
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        navigate(`/products/${row.sku_id}`);
                      }
                    }}
                  >
                    <td>
                      <span className="sku-cell">
                        <span className="sku-cell__id">{row.sku_id}</span>
                        <span className="sku-cell__cat">{row.subcategory}</span>
                      </span>
                    </td>
                    <td>
                      {style ? (
                        <span
                          className="pill"
                          style={
                            {
                              '--pill-colour': style.colour,
                              '--pill-soft': style.soft,
                              '--pill-ink': style.ink,
                            } as React.CSSProperties
                          }
                        >
                          <span className="pill__icon" aria-hidden="true">
                            {style.icon}
                          </span>
                          {style.short}
                        </span>
                      ) : (
                        <span className="pill">{row.action_label}</span>
                      )}
                    </td>
                    <td className="numeric">{formatInr(row.total_revenue)}</td>
                    <td className="numeric">{formatUnits(row.total_units)}</td>
                    <td className="numeric">{formatSignedPercent(row.recent_trend_pct)}</td>
                    <td className="numeric">{formatInr(row.value_at_stake)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
