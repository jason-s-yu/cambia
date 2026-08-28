import React, { useState } from 'react';
import { EYEBROW } from '../eyebrow';

/**
 * Native input attributes pass straight through to the <input> (id, name,
 * autoComplete, required, inputMode, aria-*), so forms keep their
 * autofill and validation semantics. The wrapper owns `style`.
 *
 * A field with no `label` has no accessible name of its own: `aria-label` is
 * the passthrough that gives it one, and the placeholder is not a substitute
 * (cambia-876, DL-3 review F6).
 */
export interface InputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'onChange' | 'style' | 'type' | 'value' | 'defaultValue'> {
  /** Uppercase eyebrow label above the field. */
  label?: string;
  value?: string;
  defaultValue?: string;
  placeholder?: string;
  type?: string;
  /** Monospace for codes, seeds and ids. */
  mono?: boolean;
  /** Error message below the field. */
  error?: string;
  disabled?: boolean;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  style?: React.CSSProperties;
}

/** Text input on an inset well; gold focus ring, danger-token error state. */
const Input: React.FC<InputProps> = ({ label, value, defaultValue, placeholder, type = 'text', mono = false, error, disabled = false, onChange, style, ...rest }) => {
  const [focus, setFocus] = useState(false);
  return (
    <label style={{ display: 'block', ...style }}>
      {label && (
        <span style={{ display: 'block', marginBottom: 6, ...EYEBROW }}>{label}</span>
      )}
      <input
        {...rest}
        type={type}
        value={value}
        defaultValue={defaultValue}
        placeholder={placeholder}
        disabled={disabled}
        onChange={onChange}
        onFocus={() => setFocus(true)}
        onBlur={() => setFocus(false)}
        style={{
          width: '100%',
          height: 'var(--control-h-md)',
          padding: '0 10px',
          fontFamily: mono ? 'var(--ds-font-mono)' : 'var(--font-sans)',
          fontSize: 'var(--text-md)',
          fontVariantNumeric: 'tabular-nums',
          color: disabled ? 'var(--text-disabled)' : 'var(--text-primary)',
          background: 'var(--surface-inset)',
          border: '1px solid ' + (error ? 'var(--status-danger)' : focus ? 'var(--focus-ring-color)' : 'var(--border-default)'),
          borderRadius: 'var(--ds-radius-sm)',
          outline: 'none',
          boxShadow: focus ? 'var(--focus-ring)' : 'none',
          opacity: disabled ? 0.55 : 1,
          transition: 'border-color var(--dur-fast) var(--ds-ease-out), box-shadow var(--dur-fast) var(--ds-ease-out)'
        }}
      />
      {error && <span style={{ display: 'block', marginTop: 5, fontSize: 'var(--ds-text-xs)', color: 'var(--status-danger)', fontWeight: 'var(--weight-medium)' }}>{error}</span>}
    </label>
  );
};

export default Input;
