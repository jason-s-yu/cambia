import React, { useState } from 'react';

export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectProps {
  label?: string;
  value?: string;
  defaultValue?: string;
  options?: Array<string | SelectOption>;
  disabled?: boolean;
  onChange?: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  style?: React.CSSProperties;
}

/** Styled native select with eyebrow label. */
const Select: React.FC<SelectProps> = ({ label, value, defaultValue, options = [], disabled = false, onChange, style }) => {
  const [focus, setFocus] = useState(false);
  return (
    <label style={{ display: 'block', position: 'relative', ...style }}>
      {label && (
        <span
          style={{
            display: 'block',
            marginBottom: 6,
            fontSize: 'var(--text-2xs)',
            fontWeight: 'var(--weight-bold)',
            letterSpacing: 'var(--tracking-caps)',
            textTransform: 'uppercase',
            color: 'var(--text-tertiary)'
          }}
        >
          {label}
        </span>
      )}
      <span style={{ position: 'relative', display: 'block' }}>
        <select
          value={value}
          defaultValue={defaultValue}
          disabled={disabled}
          onChange={onChange}
          onFocus={() => setFocus(true)}
          onBlur={() => setFocus(false)}
          style={{
            width: '100%',
            height: 'var(--control-h-md)',
            padding: '0 32px 0 10px',
            fontFamily: 'var(--font-sans)',
            fontSize: 'var(--text-md)',
            fontWeight: 'var(--weight-medium)',
            color: disabled ? 'var(--text-disabled)' : 'var(--text-primary)',
            background: 'var(--surface-inset)',
            border: '1px solid ' + (focus ? 'var(--focus-ring-color)' : 'var(--border-default)'),
            borderRadius: 'var(--ds-radius-sm)',
            outline: 'none',
            appearance: 'none',
            WebkitAppearance: 'none',
            boxShadow: focus ? 'var(--focus-ring)' : 'none',
            opacity: disabled ? 0.55 : 1,
            cursor: disabled ? 'not-allowed' : 'pointer'
          }}
        >
          {options.map((o) => {
            const opt = typeof o === 'string' ? { value: o, label: o } : o;
            return (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            );
          })}
        </select>
        <span style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--text-tertiary)', fontSize: 11 }}>▼</span>
      </span>
    </label>
  );
};

export default Select;
