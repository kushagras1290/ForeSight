/**
 * Route guard: redirects to /login unless a session is present.
 *
 * Wraps the entire dashboard route subtree in `App.tsx`, so no page has to
 * fetch or check auth itself - one gate in one place, matching how `Layout`
 * already centralises the readiness check for core data.
 */

import { Navigate, Outlet, useLocation } from 'react-router-dom';

import { useAuth } from '../lib/AuthContext';
import { Loading } from './States';

export function RequireAuth() {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="app">
        <main className="page shell" style={{ paddingTop: 'var(--space-7)' }}>
          <Loading rows={4} label="Checking your session" />
        </main>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" state={{ from: location.pathname }} replace />;
  }

  return <Outlet />;
}
