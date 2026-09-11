/**
 * Signed-in user, fetched once and shared with the whole app.
 *
 * `RequireAuth` reads `user`/`loading` to gate every dashboard route; `Login`,
 * `Register` and `Account` call `refresh`/`setUser` after a successful
 * request so the rest of the app finds out immediately, without a second
 * round trip to `/api/auth/me`.
 */

import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react';

import { api, type AuthUser } from './api';

interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean;
  setUser: (user: AuthUser | null) => void;
  refresh: () => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      setUser(await api.auth.me());
    } catch {
      // A 401 ("not signed in") is expected and unremarkable here; any other
      // failure (network, 5xx) still leaves the app treating the visitor as
      // signed out, which is the safe default either way.
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const logout = useCallback(async () => {
    try {
      await api.auth.logout();
    } finally {
      setUser(null);
    }
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, setUser, refresh, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider.');
  }
  return context;
}
