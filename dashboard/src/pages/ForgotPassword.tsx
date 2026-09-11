/**
 * Password recovery: identifier -> both security questions -> new password.
 *
 * Two steps, not one form, because the questions themselves are not known
 * until the identifier is looked up - the account is not always the one
 * making the request, so the questions can't be pre-filled from local state.
 */

import { useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';

import { ApiError, api } from '../lib/api';
import { useAuth } from '../lib/AuthContext';
import { PasswordField } from '../components/PasswordField';
import { ErrorState } from '../components/States';

type Step = 'identify' | 'answer';

export function ForgotPassword() {
  const { setUser } = useAuth();
  const navigate = useNavigate();

  const [step, setStep] = useState<Step>('identify');
  const [identifier, setIdentifier] = useState('');
  const [questions, setQuestions] = useState<string[]>([]);
  const [answer1, setAnswer1] = useState('');
  const [answer2, setAnswer2] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleIdentify = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const result = await api.auth.forgotPasswordQuestions(identifier.trim());
      setQuestions(result.questions);
      setStep('answer');
    } catch (cause) {
      setError(
        cause instanceof ApiError ? cause.message : 'Could not find an account with that identifier.',
      );
    } finally {
      setSubmitting(false);
    }
  };

  const handleReset = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setSubmitting(true);
    try {
      const user = await api.auth.resetPassword({
        identifier: identifier.trim(),
        security_answer_1: answer1,
        security_answer_2: answer2,
        new_password: newPassword,
      });
      setUser(user);
      navigate('/', { replace: true });
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not reset the password.');
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
        <h1 className="auth-card__title">Reset your password</h1>

        {error ? <ErrorState message={error} /> : null}

        {step === 'identify' ? (
          <form className="auth-form" onSubmit={handleIdentify}>
            <p className="section__note">
              Enter your email or username and we'll ask your security questions.
            </p>
            <div className="field">
              <label htmlFor="forgot-identifier" className="field__label">
                Email or username
              </label>
              <input
                id="forgot-identifier"
                className="input"
                value={identifier}
                onChange={(event) => setIdentifier(event.target.value)}
                autoComplete="username"
                required
              />
            </div>
            <button type="submit" className="button auth-submit" disabled={submitting}>
              {submitting ? 'Looking up…' : 'Continue'}
            </button>
          </form>
        ) : (
          <form className="auth-form" onSubmit={handleReset}>
            <div className="field">
              <label htmlFor="forgot-answer-1" className="field__label">
                {questions[0]}
              </label>
              <input
                id="forgot-answer-1"
                className="input"
                value={answer1}
                onChange={(event) => setAnswer1(event.target.value)}
                required
              />
            </div>
            <div className="field">
              <label htmlFor="forgot-answer-2" className="field__label">
                {questions[1]}
              </label>
              <input
                id="forgot-answer-2"
                className="input"
                value={answer2}
                onChange={(event) => setAnswer2(event.target.value)}
                required
              />
            </div>

            <PasswordField
              id="forgot-new-password"
              label="New password"
              value={newPassword}
              onChange={setNewPassword}
              autoComplete="new-password"
              minLength={8}
              required
            />
            <PasswordField
              id="forgot-confirm-password"
              label="Confirm new password"
              value={confirmPassword}
              onChange={setConfirmPassword}
              autoComplete="new-password"
              minLength={8}
              required
            />

            <button type="submit" className="button auth-submit" disabled={submitting}>
              {submitting ? 'Resetting…' : 'Reset password'}
            </button>
          </form>
        )}

        <div className="auth-links">
          <Link to="/login">Back to sign in</Link>
        </div>
      </div>
    </div>
  );
}
