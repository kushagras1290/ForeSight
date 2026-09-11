/**
 * Create an account: email/username/password plus two distinct security
 * questions, used later by the forgot-password flow. See foresight.auth's
 * module docstring for why two questions rather than one.
 */

import { useEffect, useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';

import { ApiError, api } from '../lib/api';
import { useAuth } from '../lib/AuthContext';
import { PasswordField } from '../components/PasswordField';
import { ErrorState, Loading } from '../components/States';

export function Register() {
  const { setUser } = useAuth();
  const navigate = useNavigate();

  const [questions, setQuestions] = useState<string[]>([]);
  const [questionsLoading, setQuestionsLoading] = useState(true);
  const [questionsError, setQuestionsError] = useState<string | null>(null);

  useEffect(() => {
    api.auth
      .securityQuestions()
      .then(setQuestions)
      .catch(() => setQuestionsError('Could not load security questions. Refresh to try again.'))
      .finally(() => setQuestionsLoading(false));
  }, []);

  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [question1, setQuestion1] = useState('');
  const [answer1, setAnswer1] = useState('');
  const [question2, setQuestion2] = useState('');
  const [answer2, setAnswer2] = useState('');

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const secondQuestionOptions = questions.filter((question) => question !== question1);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }
    if (!question1 || !question2) {
      setError('Choose both security questions.');
      return;
    }

    setSubmitting(true);
    try {
      const user = await api.auth.register({
        email: email.trim(),
        username: username.trim(),
        password,
        display_name: displayName.trim(),
        security_question_1: question1,
        security_answer_1: answer1,
        security_question_2: question2,
        security_answer_2: answer2,
      });
      setUser(user);
      navigate('/', { replace: true });
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Could not create the account.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-shell">
      <div className="auth-card auth-card--wide">
        <p className="header__mark" style={{ color: 'var(--ink)' }}>
          FORESIGHT
        </p>
        <h1 className="auth-card__title">Create an account</h1>

        {error ? <ErrorState message={error} /> : null}
        {questionsError ? <ErrorState message={questionsError} /> : null}

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="field">
            <label htmlFor="register-email" className="field__label">
              Email
            </label>
            <input
              id="register-email"
              className="input"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete="email"
              required
            />
          </div>

          <div className="two-col">
            <div className="field">
              <label htmlFor="register-username" className="field__label">
                Username
              </label>
              <input
                id="register-username"
                className="input"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                autoComplete="username"
                pattern="[A-Za-z0-9_]{3,32}"
                title="3-32 letters, digits or underscores"
                required
              />
            </div>
            <div className="field">
              <label htmlFor="register-display-name" className="field__label">
                Display name
              </label>
              <input
                id="register-display-name"
                className="input"
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                autoComplete="name"
                required
              />
            </div>
          </div>

          <div className="two-col">
            <PasswordField
              id="register-password"
              label="Password"
              value={password}
              onChange={setPassword}
              autoComplete="new-password"
              minLength={8}
              required
            />
            <PasswordField
              id="register-confirm-password"
              label="Confirm password"
              value={confirmPassword}
              onChange={setConfirmPassword}
              autoComplete="new-password"
              minLength={8}
              required
            />
          </div>

          <p className="section__note" style={{ marginTop: 'var(--space-2)' }}>
            Two security questions, used to reset your password later.
          </p>

          {questionsLoading ? (
            <Loading rows={2} label="Loading security questions" />
          ) : (
            <>
              <div className="two-col">
                <div className="field">
                  <label htmlFor="register-question-1" className="field__label">
                    Security question 1
                  </label>
                  <select
                    id="register-question-1"
                    className="select"
                    value={question1}
                    onChange={(event) => setQuestion1(event.target.value)}
                    required
                  >
                    <option value="" disabled>
                      Choose a question…
                    </option>
                    {questions.map((question) => (
                      <option key={question} value={question}>
                        {question}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="register-answer-1" className="field__label">
                    Answer 1
                  </label>
                  <input
                    id="register-answer-1"
                    className="input"
                    value={answer1}
                    onChange={(event) => setAnswer1(event.target.value)}
                    required
                  />
                </div>
              </div>

              <div className="two-col">
                <div className="field">
                  <label htmlFor="register-question-2" className="field__label">
                    Security question 2
                  </label>
                  <select
                    id="register-question-2"
                    className="select"
                    value={question2}
                    onChange={(event) => setQuestion2(event.target.value)}
                    required
                    disabled={!question1}
                  >
                    <option value="" disabled>
                      Choose a different question…
                    </option>
                    {secondQuestionOptions.map((question) => (
                      <option key={question} value={question}>
                        {question}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="register-answer-2" className="field__label">
                    Answer 2
                  </label>
                  <input
                    id="register-answer-2"
                    className="input"
                    value={answer2}
                    onChange={(event) => setAnswer2(event.target.value)}
                    required
                  />
                </div>
              </div>
            </>
          )}

          <button type="submit" className="button auth-submit" disabled={submitting}>
            {submitting ? 'Creating account…' : 'Create account'}
          </button>
        </form>

        <div className="auth-links">
          <Link to="/login">Already have an account? Sign in</Link>
        </div>
      </div>
    </div>
  );
}
