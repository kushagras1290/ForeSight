/**
 * App shell: header, primary navigation, and the initial-load / unreachable-
 * service gates. Loads core data once via `useForesightData` and hands it
 * down through the router's outlet context, so no page re-fetches readiness.
 *
 * Deliberately does NOT block the whole app behind "no data yet" - a fresh
 * instance with nothing trained should still look and navigate like a real
 * product, not vanish behind a single "come back later" screen. Each page
 * reads `ready`/`core` from `useCoreData()` and renders its own empty state
 * when there is nothing to show yet; only two states are handled here,
 * because they are the only two that mean "no page could do anything useful
 * regardless of what it shows": the service is unreachable, or the very
 * first readiness check hasn't come back.
 */

import { NavLink, Outlet, useOutletContext } from 'react-router-dom';

import { useAuth } from '../lib/AuthContext';
import type { CoreData, ForesightData } from '../lib/useForesightData';
import { useForesightData } from '../lib/useForesightData';
import { ErrorState, Loading } from './States';
import { Header } from './Header';
import type { ReadyState } from '../lib/api';

interface NavItem {
  to: string;
  label: string;
}

const NAV_ITEMS: NavItem[] = [
  { to: '/', label: 'Home' },
  { to: '/sales-analytics', label: 'Sales Analytics' },
  { to: '/demand-forecast', label: 'Demand Forecast' },
  { to: '/inventory', label: 'Inventory' },
  { to: '/risk', label: 'Risk Dashboard' },
  { to: '/products', label: 'Product Details' },
  { to: '/executive-summary', label: 'Executive Summary' },
];

/**
 * What every page receives via `useOutletContext<PageContext>()`.
 *
 * `core` is `null` until both a trained model exists (`ready.ready`) and its
 * data has finished loading - pages should treat `!ready?.ready` as "nothing
 * trained yet" (a stable, permanent-until-someone-runs-the-pipeline state)
 * and `ready?.ready && !core` as "loading" (transient).
 */
export interface PageContext {
  core: CoreData | null;
  ready: ReadyState | null;
  reload: () => void;
}

export function useCoreData(): PageContext {
  return useOutletContext<PageContext>();
}

export function Layout() {
  const { ready, core, coreError, reload }: ForesightData = useForesightData();
  const { user } = useAuth();

  if (coreError) {
    return (
      <div className="app">
        <Header summary={null} accuracy={null} />
        <main className="page shell">
          <ErrorState
            title="Cannot reach the scoring service"
            message={coreError}
            hint={'uvicorn service.main:app --port 8000'}
            onRetry={reload}
          />
        </main>
      </div>
    );
  }

  // Only the very first check, before we even know if a model exists. This
  // is the one loading state that has to block navigation - the nav itself
  // never changes based on readiness, but rendering it before the first
  // `/ready` response would mean showing a shell with unknown behaviour.
  if (!ready) {
    return (
      <div className="app">
        <Header summary={null} accuracy={null} />
        <main className="page shell">
          <Loading rows={4} label="Loading" />
        </main>
      </div>
    );
  }

  return (
    <div className="app">
      <Header summary={core?.summary ?? null} accuracy={core?.accuracy ?? null} />

      <nav className="nav" aria-label="Dashboard sections">
        <div className="shell nav__inner">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) => `nav__link${isActive ? ' active' : ''}`}
            >
              {item.label}
            </NavLink>
          ))}
          <NavLink
            to="/account"
            className={({ isActive }) => `nav__link nav__link--account${isActive ? ' active' : ''}`}
          >
            {user?.display_name ?? 'Account'}
          </NavLink>
        </div>
      </nav>

      <main className="page shell">
        <Outlet context={{ core: core ?? null, ready, reload } satisfies PageContext} />
      </main>
    </div>
  );
}
