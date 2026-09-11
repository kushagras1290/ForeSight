/**
 * Account page: who you're signed in as, and changing your password.
 *
 * Lives inside the same protected route tree as the rest of the dashboard
 * (via RequireAuth + Layout), so it gets the header and nav for free.
 */

import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';

import { ApiError, api } from '../lib/api';
import { useAuth } from '../lib/AuthContext';
import { PasswordField } from '../components/PasswordField';
import { ErrorState } from '../components/States';

export function Account() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  if (!user) return null;

  const handleChangePassword = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setSuccess(false);

    if (newPassword !== confirmPassword) {
      setError('New passwords do not match.');
      return;
    }

    setSubmitting(true);
    try {
      await api.auth.changePassword(currentPassword, newPassword);
      setSuccess(true);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not change the password.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleLogout = async () => {
    await logout();
    navigate('/login', { replace: true });
  };

  return (
    <>
      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Account</p>
          <h2 className="section__title">{user.display_name}</h2>
        </div>
        <div className="stat-grid">
          <div className="stat">
            <div className="stat__label">Email</div>
            <div className="stat__value" style={{ fontSize: 14 }}>
              {user.email}
            </div>
          </div>
          <div className="stat">
            <div className="stat__label">Username</div>
            <div className="stat__value" style={{ fontSize: 14 }}>
              {user.username}
            </div>
          </div>
        </div>
        <button
          type="button"
          className="button button--ghost"
          style={{ marginTop: 'var(--space-4)' }}
          onClick={() => void handleLogout()}
        >
          Log out
        </button>
      </section>

      <section className="section">
        <div className="section__head">
          <p className="eyebrow">Security</p>
          <h2 className="section__title">Change password</h2>
        </div>

        {error ? <ErrorState message={error} /> : null}
        {success ? (
          <div className="callout" style={{ maxWidth: 'none' }}>
            Password changed.
          </div>
        ) : null}

        <form
          className="auth-form"
          style={{ maxWidth: 420 }}
          onSubmit={(event) => void handleChangePassword(event)}
        >
          <PasswordField
            id="account-current-password"
            label="Current password"
            value={currentPassword}
            onChange={setCurrentPassword}
            autoComplete="current-password"
            required
          />
          <PasswordField
            id="account-new-password"
            label="New password"
            value={newPassword}
            onChange={setNewPassword}
            autoComplete="new-password"
            minLength={8}
            required
          />
          <PasswordField
            id="account-confirm-password"
            label="Confirm new password"
            value={confirmPassword}
            onChange={setConfirmPassword}
            autoComplete="new-password"
            minLength={8}
            required
          />
          <button type="submit" className="button auth-submit" disabled={submitting}>
            {submitting ? 'Changing…' : 'Change password'}
          </button>
        </form>
      </section>
    </>
  );
}
