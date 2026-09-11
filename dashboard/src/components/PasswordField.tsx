/**
 * A password input with a show/hide toggle.
 *
 * Plain `<input type="password">` gives no way to check what you just typed
 * before submitting, which is exactly when a typo is most costly (setting a
 * new password, answering a security question). The toggle never changes
 * what is submitted - only whether the browser masks it on screen.
 */

import { useId, useState } from 'react';

interface PasswordFieldProps {
  id?: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  autoComplete?: string;
  placeholder?: string;
  required?: boolean;
  minLength?: number;
}

export function PasswordField({
  id,
  label,
  value,
  onChange,
  autoComplete,
  placeholder,
  required,
  minLength,
}: PasswordFieldProps) {
  const generatedId = useId();
  const inputId = id ?? generatedId;
  const [visible, setVisible] = useState(false);

  return (
    <div className="field">
      <label htmlFor={inputId} className="field__label">
        {label}
      </label>
      <div className="password-field">
        <input
          id={inputId}
          className="input password-field__input"
          type={visible ? 'text' : 'password'}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          autoComplete={autoComplete}
          placeholder={placeholder}
          required={required}
          minLength={minLength}
        />
        <button
          type="button"
          className="password-field__toggle"
          onClick={() => setVisible((current) => !current)}
          aria-label={visible ? 'Hide password' : 'Show password'}
          aria-pressed={visible}
        >
          {visible ? '🙈' : '👁'}
        </button>
      </div>
    </div>
  );
}
