/**
 * Business Insights: revenue concentration, dead stock, and top movers.
 *
 * Scoped deliberately to *business* insights, not customer insights — no
 * customer entity exists in any extract this project has (every source is
 * SKU/week grained), so a customer-segmentation page would have to invent
 * data the brief doesn't provide. Reads `/api/insights/business`, a cheap
 * aggregation over the same weekly panel every other page reads from.
 */

import { useEffect, useState } from 'react';

import { ApiError, api, type BusinessInsightsResponse } from '../lib/api';
import { useCoreData } from '../components/Layout';
import { Empty, ErrorState, Loading } from '../components/States';
import { NoDataYet } from '../components/NoDataYet';
import { formatInr, formatPercent, formatSignedPercent } from '../lib/format';

export function BusinessInsights() {
  const { ready } = useCoreData();
  const [data, setData] = useState<BusinessInsightsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    if (!ready?.ready) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .businessInsights()
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof ApiError ? cause.message : 'Could not load business insights.');
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
          title="No business insights yet"
          body="This instance has no sales data to analyse. Once real extracts are run through the pipeline, revenue concentration, dead stock, and top movers will show up here."
          missing={ready?.artifacts_missing}
        />
      </section>
    );
  }

  return (
    <>
      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Business insights</p>
          <h2 className="section__title">Where the portfolio's revenue really sits</h2>
        </div>

        {error ? (
          <ErrorState message={error} onRetry={() => setReloadToken((token) => token + 1)} />
        ) : loading || !data ? (
          <Loading rows={4} label="Loading business insights" />
        ) : (
          <div className="stat-grid">
            {data.revenue_concentration.map((point) => (
              <div className="stat" key={point.sku_fraction}>
                <div className="stat__label">
                  Top {formatPercent(point.sku_fraction, 0)} of SKUs ({point.sku_count})
                </div>
                <div className="stat__value tabular">
                  {formatPercent(point.revenue_share, 0)} of revenue
                </div>
              </div>
            ))}
            <div className="stat">
              <div className="stat__label">Total portfolio revenue</div>
              <div className="stat__value tabular">{formatInr(data.total_revenue)}</div>
              <div className="stat__hint">{data.total_skus} SKUs</div>
            </div>
          </div>
        )}
      </section>

      {data ? (
        <>
          <section className="section">
            <div className="section__head">
              <p className="eyebrow">Dead stock</p>
              <h2 className="section__title">Products that have stopped selling</h2>
              <p className="section__note">
                Eight or more consecutive weeks with zero recorded demand, as of the most recent
                week in the panel.
              </p>
            </div>

            {data.dead_stock.length === 0 ? (
              <Empty
                mark="✓"
                title="Nothing flagged"
                body="No SKU has gone eight or more weeks without a sale."
              />
            ) : (
              <div className="table-scroll">
                <table className="table">
                  <thead>
                    <tr>
                      <th scope="col">Product</th>
                      <th scope="col" className="numeric">
                        Weeks with no sale
                      </th>
                      <th scope="col">Last sale</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.dead_stock.map((row) => (
                      <tr key={row.sku_id}>
                        <td>
                          <span className="sku-cell">
                            <span className="sku-cell__id">{row.sku_id}</span>
                            <span className="sku-cell__cat">{row.subcategory}</span>
                          </span>
                        </td>
                        <td className="numeric">{row.consecutive_zero_weeks}</td>
                        <td>{row.last_sale_week ?? 'Never'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="section">
            <div className="section__head">
              <p className="eyebrow">Top movers</p>
              <h2 className="section__title">Biggest changes, last 8 weeks vs. the 8 before</h2>
            </div>

            <div className="card-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
              <div>
                <p className="eyebrow">Gainers</p>
                <div className="table-scroll">
                  <table className="table">
                    <thead>
                      <tr>
                        <th scope="col">Product</th>
                        <th scope="col" className="numeric">
                          Change
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.top_gainers.map((row) => (
                        <tr key={row.sku_id}>
                          <td>{row.sku_id}</td>
                          <td className="numeric" style={{ color: 'var(--signal-ok)' }}>
                            {formatSignedPercent(row.change_pct)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div>
                <p className="eyebrow">Decliners</p>
                <div className="table-scroll">
                  <table className="table">
                    <thead>
                      <tr>
                        <th scope="col">Product</th>
                        <th scope="col" className="numeric">
                          Change
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.top_decliners.map((row) => (
                        <tr key={row.sku_id}>
                          <td>{row.sku_id}</td>
                          <td className="numeric" style={{ color: 'var(--signal-critical)' }}>
                            {formatSignedPercent(row.change_pct)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </section>
        </>
      ) : null}
    </>
  );
}
