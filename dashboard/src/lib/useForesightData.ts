/**
 * Shared readiness + core-data loading, used once by `Layout` and read by
 * every page through the router's outlet context.
 *
 * Extracted from the single-page `App.tsx` this dashboard used to be: the
 * fetch sequence (readiness, then the five core requests in parallel) does
 * not change per page, so it happens once at the layout level instead of
 * once per route.
 */

import { useEffect, useState } from 'react';

import {
  ApiError,
  api,
  fetchReady,
  type AccuracySummary,
  type GridPoint,
  type HoldoutSummary,
  type PortfolioSummary,
  type ReadyState,
} from './api';

export interface CoreData {
  summary: PortfolioSummary;
  accuracy: AccuracySummary;
  grid: GridPoint[];
  categories: string[];
  holdout: HoldoutSummary;
}

export interface ForesightData {
  ready: ReadyState | null;
  core: CoreData | null;
  coreError: string | null;
  reload: () => void;
}

export function useForesightData(): ForesightData {
  const [ready, setReady] = useState<ReadyState | null>(null);
  const [core, setCore] = useState<CoreData | null>(null);
  const [coreError, setCoreError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

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
        setCoreError(cause instanceof ApiError ? cause.message : 'An unexpected error occurred.');
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [reloadToken]);

  return { ready, core, coreError, reload: () => setReloadToken((token) => token + 1) };
}
