/**
 * Shared shape for the three risk-action pages split out of the old single
 * Risk Dashboard: Stockout Risk, Overstock, and Watchlist. Each is the same
 * grid + worklist + drawer, differing only by which `action` it's locked to
 * — so this is the one place that logic lives, and each page is a thin
 * wrapper naming its own action, copy, and page-eyebrow.
 *
 * Unlike the combined Risk Dashboard, there is no action toggle here — the
 * action is the page. The grid still shows every product (so a Stockout Risk
 * page has the same context a planner had on the combined page), it just
 * comes pre-highlighted via `activeAction`; only the worklist and its count
 * are filtered to this page's action specifically.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { ApiError, api, type RiskAction, type RiskRecord, type SkuDetail } from '../lib/api';
import { useCoreData } from './Layout';
import { DecisionGrid } from './DecisionGrid';
import { ErrorState } from './States';
import { NoDataYet } from './NoDataYet';
import { SkuDetailDrawer } from './SkuDetail';
import { Worklist } from './Worklist';

const RISK_HIGH_THRESHOLD = 0.5;
const SEARCH_DEBOUNCE_MS = 250;

interface RiskActionPageProps {
  action: RiskAction;
  eyebrow: string;
  title: string;
  note: string;
  emptyTitle: string;
  emptyBody: string;
}

export function RiskActionPage({
  action,
  eyebrow,
  title,
  note,
  emptyTitle,
  emptyBody,
}: RiskActionPageProps) {
  const { core, ready } = useCoreData();

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
    setCategory('');
    setSearchInput('');
    setSearch('');
  }, []);

  const hasFilters = category !== '' || search !== '';

  if (!ready?.ready || !core) {
    return (
      <section className="section">
        <NoDataYet title={emptyTitle} body={emptyBody} missing={ready?.artifacts_missing} />
      </section>
    );
  }

  return (
    <>
      <section className="section">
        <div className="section__head">
          <p className="eyebrow">{eyebrow}</p>
          <h2 className="section__title">{title}</h2>
          <p className="section__note">{note}</p>
        </div>

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
