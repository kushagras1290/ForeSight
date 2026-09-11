/**
 * Sign in: Google, or email/username + password.
 *
 * A not-yet-configured Google client (see foresight.config.google_oauth_
 * configured) still shows the button - clicking it surfaces the backend's
 * "not configured" error rather than hiding the option entirely, so the flow
 * is reviewable end to end before real credentials exist.
 */

import { useState, type FormEvent } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';

import { ApiError, api } from '../lib/api';
import { useAuth } from '../lib/AuthContext';
import { PasswordField } from '../components/PasswordField';
import { ErrorState } from '../components/States';

export function Login() {
  const { setUser } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const redirectTo = (location.state as { from?: string } | null)?.from ?? '/';

  const [identifier, setIdentifier] = useState('');
  const [password, setPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const user = await api.auth.login(identifier.trim(), password);
      setUser(user);
      navigate(redirectTo, { replace: true });
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not sign in.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <p className="header__mark" style={{ color: 'var(--ink)' }}>
          FORESIGHT
        </p>
        <h1 className="auth-card__title">Sign in</h1>
        <p className="auth-card__subtitle">NorthBay Living · Demand &amp; Inventory Intelligence</p>

        <a className="button auth-google-button" href={api.auth.googleStartUrl}>
          <span aria-hidden="true">G</span> Continue with Google
        </a>

        <div className="auth-divider">
          <span>or</span>
        </div>

        {error ? <ErrorState message={error} /> : null}

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="login-identifier" className="field__label">
              Email or username
            </label>
            <input
              id="login-identifier"
              className="input"
              value={identifier}
              onChange={(event) => setIdentifier(event.target.value)}
              autoComplete="username"
              required
            />
          </div>

          <PasswordField
            id="login-password"
            label="Password"
            value={password}
            onChange={setPassword}
            autoComplete="current-password"
            required
          />

          <button type="submit" className="button auth-submit" disabled={submitting}>
            {submitting ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        <div className="auth-links">
          <Link to="/forgot-password">Forgot password?</Link>
          <Link to="/register">Create an account</Link>
        </div>
      </div>
    </div>
  );
}
