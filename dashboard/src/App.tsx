/**
 * FORESIGHT planning dashboard (deliverable D5).
 *
 * Structured around the one question the Head of Operations actually asked:
 * *what do I reorder and what do I clear?* The Plan tab answers it — the grid
 * for triage, the cards for filtering, the list for working through. The
 * Accuracy tab exists so that answer can be trusted rather than taken on faith.
 *
 * State is deliberately kept in this component. The app has one screen, a handful
 * of filters and no routing, so a state library would be ceremony without
 * benefit.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  ApiError,
  api,
  fetchReady,
  type AccuracySummary,
  type GridPoint,
  type HoldoutSummary,
  type PortfolioSummary,
  type ReadyState,
  type RiskAction,
  type RiskRecord,
  type SkuDetail,
} from './lib/api';
import { AccuracyPanel } from './components/AccuracyPanel';
import { DecisionGrid } from './components/DecisionGrid';
import { HoldoutPanel } from './components/HoldoutPanel';
import { FilterBar } from './components/FilterBar';
import { Header } from './components/Header';
import { SkuDetailDrawer } from './components/SkuDetail';
import { ErrorState, LoadingChart, LoadingTiles, NotReady } from './components/States';
import { StatBand } from './components/StatBand';
import { Worklist } from './components/Worklist';

import './styles/theme.css';
import './styles/app.css';

/** Matches `risk_high_threshold` in the Python configuration. */
const RISK_HIGH_THRESHOLD = 0.5;
const SEARCH_DEBOUNCE_MS = 250;

type Tab = 'plan' | 'accuracy' | 'test';

interface CoreData {
  summary: PortfolioSummary;
  accuracy: AccuracySummary;
  grid: GridPoint[];
  categories: string[];
  holdout: HoldoutSummary;
}

export default function App() {
  const [ready, setReady] = useState<ReadyState | null>(null);
  const [core, setCore] = useState<CoreData | null>(null);
  const [coreError, setCoreError] = useState<string | null>(null);

  const [tab, setTab] = useState<Tab>('plan');
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

  /** Bumped to force a refetch after a retry. */
  const [reloadToken, setReloadToken] = useState(0);

  // --- Debounce the search box so a filter runs per pause, not per keystroke.
  useEffect(() => {
    const timer = setTimeout(() => setSearch(searchInput.trim()), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [searchInput]);

  // --- Readiness, then core data ----------------------------------------- //
  useEffect(() => {
    let cancelled = false;

    async function load() {
      setCoreError(null);
      try {
        const readyState = await fetchReady();
        if (cancelled) return;
        setReady(readyState);
        if (!readyState.ready) return;

        const [summary, accuracy, grid, categories, holdout] = await Promise.all([
          api.summary(),
          api.accuracy(),
          api.grid(),
          api.categories(),
          api.holdout(),
        ]);
        if (cancelled) return;
        setCore({ summary, accuracy, grid, categories, holdout });
      } catch (cause) {
        if (cancelled) return;
        setCoreError(
          cause instanceof ApiError ? cause.message : 'An unexpected error occurred.',
        );
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [reloadToken]);

  // --- Worklist ----------------------------------------------------------- //
  // Tracks the newest request so a slow earlier response cannot overwrite a
  // faster later one and show results for a filter the user has moved on from.
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
        setRowsError(
          cause instanceof ApiError ? cause.message : 'Could not load the worklist.',
        );
      })
      .finally(() => {
        if (sequence === requestSequence.current) setRowsLoading(false);
      });
  }, [ready, action, category, search, reloadToken]);

  // --- Detail ------------------------------------------------------------- //
  const loadDetail = useCallback((skuId: string) => {
    setDetailLoading(true);
    setDetailError(null);
    setDetail(null);
    api
      .sku(skuId)
      .then(setDetail)
      .catch((cause: unknown) => {
        setDetailError(
          cause instanceof ApiError ? cause.message : 'Could not load this product.',
        );
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

  // The grid always shows the whole catalogue; the active filter dims rather
  // than removes, so the selection stays in the context of the full portfolio.
  const gridPoints = useMemo(() => core?.grid ?? [], [core]);

  // --- Render ------------------------------------------------------------- //
  if (coreError) {
    return (
      <div className="app">
        <Header summary={null} accuracy={null} />
        <main className="page shell">
          <ErrorState
            title="Cannot reach the scoring service"
            message={coreError}
            hint={'uvicorn service.main:app --port 8000'}
            onRetry={() => setReloadToken((token) => token + 1)}
          />
        </main>
      </div>
    );
  }

  if (ready && !ready.ready) {
    return (
      <div className="app">
        <Header summary={null} accuracy={null} />
        <main className="page shell">
          <NotReady
            missing={ready.artifacts_missing}
            onRetry={() => setReloadToken((token) => token + 1)}
          />
        </main>
      </div>
    );
  }

  return (
    <div className="app">
      <Header summary={core?.summary ?? null} accuracy={core?.accuracy ?? null} />

      <main className="page shell">
        {!core ? (
          <>
            <LoadingTiles />
            <div className="section">
              <LoadingChart height={380} />
            </div>
          </>
        ) : (
          <>
            <StatBand summary={core.summary} />

            <div className="tabs" role="tablist" aria-label="Dashboard views">
              <button
                type="button"
                role="tab"
                className="tab"
                aria-selected={tab === 'plan'}
                onClick={() => setTab('plan')}
              >
                Plan
              </button>
              <button
                type="button"
                role="tab"
                className="tab"
                aria-selected={tab === 'accuracy'}
                onClick={() => setTab('accuracy')}
              >
                Forecast accuracy
              </button>
              <button
                type="button"
                role="tab"
                className="tab"
                aria-selected={tab === 'test'}
                onClick={() => setTab('test')}
              >
                Test data
              </button>
            </div>

            {tab === 'plan' ? (
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
                      points={gridPoints}
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
                      <button
                        type="button"
                        className="button button--ghost"
                        onClick={clearFilters}
                      >
                        Clear
                      </button>
                    ) : null}

                    <span className="toolbar__count" aria-live="polite">
                      {rowsLoading ? 'Loading…' : `${rows.length} products`}
                    </span>
                  </div>

                  {rowsError ? (
                    <ErrorState
                      message={rowsError}
                      onRetry={() => setReloadToken((token) => token + 1)}
                    />
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
              </>
            ) : tab === 'accuracy' ? (
              <section className="section">
                <AccuracyPanel accuracy={core.accuracy} />
              </section>
            ) : (
              <section className="section">
                <HoldoutPanel holdout={core.holdout} />
              </section>
            )}
          </>
        )}
      </main>

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
    </div>
  );
}
