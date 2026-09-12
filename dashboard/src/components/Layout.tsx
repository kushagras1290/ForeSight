/**
 * App shell: header, collapsible grouped sidebar, and the initial-load /
 * unreachable-service gates. Loads core data once via `useForesightData` and
 * hands it down through the router's outlet context, so no page re-fetches
 * readiness.
 *
 * The sidebar replaced a flat top-nav row once the page count grew past
 * what a single row could hold (14 destinations, grouped into 6 sections).
 * Its collapsed/expanded state is a per-viewer UI preference, not data, so it
 * lives in `localStorage` rather than anywhere the server would need to know
 * about it — wrapped in try/catch since a private-browsing tab or blocked
 * site data can throw on either read or write.
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

import { useEffect, useState } from 'react';
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

interface NavGroup {
  label: string;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    label: 'Overview',
    items: [
      { to: '/', label: 'Home' },
      { to: '/executive-dashboard', label: 'Executive Dashboard' },
      { to: '/executive-recommendation', label: 'Executive Recommendation' },
    ],
  },
  {
    label: 'Sales & performance',
    items: [
      { to: '/sales-analytics', label: 'Sales Analytics' },
      { to: '/product-performance', label: 'Product Performance' },
      { to: '/category-performance', label: 'Category Performance' },
    ],
  },
  {
    label: 'Forecast',
    items: [
      { to: '/demand-forecast', label: 'Forecast Dashboard' },
      { to: '/model-benchmark', label: 'Model Benchmark' },
    ],
  },
  {
    label: 'Inventory & risk',
    items: [
      { to: '/inventory', label: 'Inventory Dashboard' },
      { to: '/risk/stockout', label: 'Stockout Risk' },
      { to: '/risk/overstock', label: 'Overstock' },
      { to: '/risk/watchlist', label: 'Watchlist' },
    ],
  },
  {
    label: 'Promotions & seasonality',
    items: [
      { to: '/promotions', label: 'Promotion Dashboard' },
      { to: '/seasonality', label: 'Seasonality Dashboard' },
    ],
  },
  {
    label: 'Insights',
    items: [{ to: '/business-insights', label: 'Business Insights' }],
  },
];

const SIDEBAR_COLLAPSED_KEY = 'foresight.sidebar.collapsed';

function readStoredCollapsed(): boolean {
  try {
    return window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === '1';
  } catch {
    return false;
  }
}

function writeStoredCollapsed(collapsed: boolean): void {
  try {
    window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? '1' : '0');
  } catch {
    // Private browsing or blocked site data - the toggle still works for this
    // tab, it just won't be remembered next visit.
  }
}

/** First letters of each word, for the collapsed sidebar's compact label. */
function initials(label: string): string {
  return label
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0])
    .join('')
    .toUpperCase();
}

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
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    setCollapsed(readStoredCollapsed());
  }, []);

  const toggleCollapsed = () => {
    setCollapsed((previous) => {
      const next = !previous;
      writeStoredCollapsed(next);
      return next;
    });
  };

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
  // is the one loading state that has to block navigation - the sidebar
  // itself never changes based on readiness, but rendering it before the
  // first `/ready` response would mean showing a shell with unknown
  // behaviour.
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
      <Header
        summary={core?.summary ?? null}
        accuracy={core?.accuracy ?? null}
        onMenuToggle={() => setMobileOpen((open) => !open)}
      />

      <div className="app-body">
        {mobileOpen ? (
          <button
            type="button"
            className="sidebar-backdrop"
            aria-label="Close navigation"
            onClick={() => setMobileOpen(false)}
          />
        ) : null}

        <nav
          className={`sidebar${collapsed ? ' sidebar--collapsed' : ''}${mobileOpen ? ' sidebar--open' : ''}`}
          aria-label="Dashboard sections"
        >
          <button
            type="button"
            className="sidebar__toggle"
            onClick={toggleCollapsed}
            aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'}
            title={collapsed ? 'Expand navigation' : 'Collapse navigation'}
          >
            {collapsed ? '»' : '«'}
          </button>

          <div className="sidebar__scroll">
            {NAV_GROUPS.map((group) => (
              <div className="sidebar__group" key={group.label}>
                {!collapsed ? <p className="sidebar__group-label">{group.label}</p> : null}
                {group.items.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.to === '/'}
                    className={({ isActive }) => `sidebar__link${isActive ? ' active' : ''}`}
                    title={collapsed ? item.label : undefined}
                    onClick={() => setMobileOpen(false)}
                  >
                    <span className="sidebar__link-text">
                      {collapsed ? initials(item.label) : item.label}
                    </span>
                  </NavLink>
                ))}
              </div>
            ))}

            <div className="sidebar__group">
              <NavLink
                to="/account"
                className={({ isActive }) => `sidebar__link sidebar__link--account${isActive ? ' active' : ''}`}
                title={collapsed ? (user?.display_name ?? 'Account') : undefined}
                onClick={() => setMobileOpen(false)}
              >
                <span className="sidebar__link-text">
                  {collapsed ? initials(user?.display_name ?? 'Account') : (user?.display_name ?? 'Account')}
                </span>
              </NavLink>
            </div>
          </div>
        </nav>

        <main className="page shell">
          <Outlet context={{ core: core ?? null, ready, reload } satisfies PageContext} />
        </main>
      </div>
    </div>
  );
}
