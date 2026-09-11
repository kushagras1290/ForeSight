/**
 * Risk Dashboard: the decisioning view, promoted from the old single-page
 * app's "Plan" tab to its own route. Same three pieces (filter legend,
 * stockout/overstock grid, priority worklist), unchanged, because they were
 * already built for exactly this question — what do I reorder, what do I
 * clear.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { ApiError, api, type RiskAction, type RiskRecord, type SkuDetail } from '../lib/api';
import { useCoreData } from '../components/Layout';
import { DecisionGrid } from '../components/DecisionGrid';
import { FilterBar } from '../components/FilterBar';
import { ErrorState } from '../components/States';
import { NoDataYet } from '../components/NoDataYet';
import { SkuDetailDrawer } from '../components/SkuDetail';
import { Worklist } from '../components/Worklist';

const RISK_HIGH_THRESHOLD = 0.5;
const SEARCH_DEBOUNCE_MS = 250;

export function RiskDashboard() {
  const { core, ready } = useCoreData();

  const [action, setAction] = useState<RiskAction | ''>('');
  const [category, setCategory] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [search, setSearch] = useState('');

  const [rows, setRows] = useState<RiskRecord[]>([]);
  const [rowsLoading, setRowsLoading] = useState(true);
  const [rowsError, setRowsError] = useState<string | null>(null);

  const [selectedSku, setSelectedSku] = useState<string | null>(null);
  const [detail, setDetail] = useState<SkuDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  useEffect(() => {
    const timer = setTimeout(() => setSearch(searchInput.trim()), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [searchInput]);

  const requestSequence = useRef(0);

  useEffect(() => {
    if (!ready?.ready) return;
    const sequence = ++requestSequence.current;
    setRowsLoading(true);
    setRowsError(null);

    api
      .risk({ action, category, search, limit: 500 })
      .then((result) => {
        if (sequence !== requestSequence.current) return;
        setRows(result);
      })
      .catch((cause: unknown) => {
        if (sequence !== requestSequence.current) return;
        setRowsError(cause instanceof ApiError ? cause.message : 'Could not load the worklist.');
      })
      .finally(() => {
        if (sequence === requestSequence.current) setRowsLoading(false);
      });
  }, [action, category, search, ready?.ready]);

  const loadDetail = useCallback((skuId: string) => {
    setDetailLoading(true);
    setDetailError(null);
    setDetail(null);
    api
      .sku(skuId)
      .then(setDetail)
      .catch((cause: unknown) => {
        setDetailError(cause instanceof ApiError ? cause.message : 'Could not load this product.');
      })
      .finally(() => setDetailLoading(false));
  }, []);

  const handleSelect = useCallback(
    (skuId: string) => {
      setSelectedSku(skuId);
      loadDetail(skuId);
    },
    [loadDetail],
  );

  const closeDetail = useCallback(() => {
    setSelectedSku(null);
    setDetail(null);
    setDetailError(null);
  }, []);

  const clearFilters = useCallback(() => {
    setAction('');
    setCategory('');
    setSearchInput('');
    setSearch('');
  }, []);

  const hasFilters = action !== '' || category !== '' || search !== '';

  if (!ready?.ready || !core) {
    return (
      <section className="section">
        <NoDataYet
          title="No risk scoring trained yet"
          body="This instance has no forecast or inventory data to plot. Once real data has been run through the pipeline, every product will show up here."
          missing={ready?.artifacts_missing}
        />
      </section>
    );
  }

  return (
    <>
      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Decisioning view</p>
          <h2 className="section__title">Where every product sits</h2>
          <p className="section__note">
            Risk of running out, against risk of being left holding stock.
          </p>
        </div>

        <FilterBar summary={core.summary} active={action} onChange={setAction} />

        <div className="grid-panel">
          <DecisionGrid
            points={core.grid}
            threshold={RISK_HIGH_THRESHOLD}
            activeAction={action}
            selectedSku={selectedSku}
            onSelect={handleSelect}
          />
        </div>
      </section>

      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Worklist</p>
          <h2 className="section__title">What to do, in priority order</h2>
        </div>

        <div className="toolbar">
          <select
            className="select"
            aria-label="Filter by category"
            value={category}
            onChange={(event) => setCategory(event.target.value)}
          >
            <option value="">All categories</option>
            {core.categories.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>

          <input
            className="input"
            type="search"
            aria-label="Search by product code"
            placeholder="Search product code…"
            value={searchInput}
            onChange={(event) => setSearchInput(event.target.value)}
          />

          {hasFilters ? (
            <button type="button" className="button button--ghost" onClick={clearFilters}>
              Clear
            </button>
          ) : null}

          <span className="toolbar__count" aria-live="polite">
            {rowsLoading ? 'Loading…' : `${rows.length} products`}
          </span>
        </div>

        {rowsError ? (
          <ErrorState message={rowsError} />
        ) : (
          <Worklist
            rows={rows}
            loading={rowsLoading}
            selectedSku={selectedSku}
            onSelect={handleSelect}
            onClearFilters={clearFilters}
            hasFilters={hasFilters}
          />
        )}
      </section>

      {selectedSku ? (
        <SkuDetailDrawer
          skuId={selectedSku}
          detail={detail}
          loading={detailLoading}
          error={detailError}
          onClose={closeDetail}
          onRetry={() => loadDetail(selectedSku)}
        />
      ) : null}
    </>
  );
}
