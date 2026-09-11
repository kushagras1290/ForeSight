/**
 * Inventory Dashboard: stock position and cover for every product, plus a
 * live "what if" re-score against a stock count that hasn't made it into a
 * snapshot yet. The what-if hits `POST /api/score` directly — the same
 * arithmetic as the batch risk table, just against a number the caller
 * supplies instead of the last stored snapshot.
 */

import { useEffect, useState } from 'react';

import { ApiError, api, type LiveScoreResponse, type RiskRecord } from '../lib/api';
import { useCoreData } from '../components/Layout';
import { ErrorState, Loading } from '../components/States';
import { NoDataYet } from '../components/NoDataYet';
import { formatDate, formatInr, formatUnits, formatWeeks } from '../lib/format';

export function InventoryDashboard() {
  const { ready } = useCoreData();

  const [rows, setRows] = useState<RiskRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ready?.ready) return;
    let cancelled = false;
    api
      .risk({ limit: 500 })
      .then((result) => {
        if (!cancelled) {
          setRows([...result].sort((left, right) => left.cover_weeks - right.cover_weeks));
        }
      })
      .catch((cause: unknown) => {
        if (!cancelled) {
          setError(cause instanceof ApiError ? cause.message : 'Could not load inventory.');
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [ready?.ready]);

  // --- What-if scoring form ------------------------------------------------ //
  const [whatIfSku, setWhatIfSku] = useState('');
  const [whatIfStock, setWhatIfStock] = useState('');
  const [whatIfResult, setWhatIfResult] = useState<LiveScoreResponse | null>(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const [whatIfError, setWhatIfError] = useState<string | null>(null);

  const runWhatIf = () => {
    const skuId = whatIfSku.trim();
    const onHand = Number(whatIfStock);
    if (!skuId || !Number.isFinite(onHand) || onHand < 0) {
      setWhatIfError('Enter a product code and a non-negative stock count.');
      return;
    }
    setWhatIfLoading(true);
    setWhatIfError(null);
    api
      .score([{ sku_id: skuId, on_hand_units: onHand }])
      .then(setWhatIfResult)
      .catch((cause: unknown) => {
        setWhatIfResult(null);
        setWhatIfError(cause instanceof ApiError ? cause.message : 'Could not re-score that SKU.');
      })
      .finally(() => setWhatIfLoading(false));
  };

  if (!ready?.ready) {
    return (
      <section className="section">
        <NoDataYet
          title="No inventory data trained yet"
          body="This instance has no stock positions to show or re-score against. Once real data has been run through the pipeline, this page comes alive."
          missing={ready?.artifacts_missing}
        />
      </section>
    );
  }

  return (
    <>
      <section className="section">
        <div className="section__head">
          <p className="eyebrow">What if</p>
          <h2 className="section__title">Re-score against a stock count you have right now</h2>
          <p className="section__note">
            The weekly snapshot goes stale between refreshes. This hits the live model with
            whatever count you have on hand today.
          </p>
        </div>

        <form
          className="toolbar"
          onSubmit={(event) => {
            event.preventDefault();
            runWhatIf();
          }}
        >
          <input
            className="input"
            aria-label="Product code"
            placeholder="Product code"
            value={whatIfSku}
            onChange={(event) => setWhatIfSku(event.target.value)}
          />
          <input
            className="input"
            type="number"
            min={0}
            aria-label="Units on hand"
            placeholder="Units on hand"
            value={whatIfStock}
            onChange={(event) => setWhatIfStock(event.target.value)}
            style={{ minWidth: 120 }}
          />
          <button type="submit" className="button">
            Re-score
          </button>
        </form>

        {whatIfError ? <ErrorState message={whatIfError} /> : null}
        {whatIfLoading ? <Loading rows={1} label="Scoring" /> : null}
        {whatIfResult && whatIfResult.results[0] ? (
          <div className="stat-grid" style={{ marginTop: 'var(--space-4)' }}>
            <div className="stat">
              <div className="stat__label">Action</div>
              <div className="stat__value">{whatIfResult.results[0].action_label}</div>
              <div className="stat__hint">{whatIfResult.results[0].action_rationale}</div>
            </div>
            <div className="stat">
              <div className="stat__label">Cover</div>
              <div className="stat__value tabular">
                {formatWeeks(whatIfResult.results[0].cover_weeks)}
              </div>
            </div>
            <div className="stat">
              <div className="stat__label">Recommended order</div>
              <div className="stat__value tabular">
                {formatUnits(whatIfResult.results[0].recommended_order_units)}
              </div>
            </div>
            <div className="stat">
              <div className="stat__label">Forecast age</div>
              <div className="stat__value tabular">{whatIfResult.forecast_age_days} days</div>
              <div className="stat__hint">as of {formatDate(whatIfResult.forecast_origin_week)}</div>
            </div>
          </div>
        ) : null}
      </section>

      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Stock position</p>
          <h2 className="section__title">Every product, soonest cover first</h2>
        </div>

        {error ? (
          <ErrorState message={error} />
        ) : loading ? (
          <Loading rows={8} />
        ) : (
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>Product</th>
                  <th className="numeric">On hand</th>
                  <th className="numeric">On order</th>
                  <th className="numeric">Lead time</th>
                  <th className="numeric">Cover</th>
                  <th>Action</th>
                  <th className="numeric">Value at stake</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 200).map((row) => (
                  <tr key={row.sku_id}>
                    <td>
                      <div className="sku-cell">
                        <span className="sku-cell__id">{row.sku_id}</span>
                        <span className="sku-cell__cat">{row.category}</span>
                      </div>
                    </td>
                    <td className="numeric">{formatUnits(row.on_hand_units)}</td>
                    <td className="numeric">{formatUnits(row.on_order_units)}</td>
                    <td className="numeric">{formatUnits(row.lead_time_days)} d</td>
                    <td className="numeric">{formatWeeks(row.cover_weeks)}</td>
                    <td>{row.action_label}</td>
                    <td className="numeric at-stake">{formatInr(row.value_at_stake)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
