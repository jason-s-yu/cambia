import React from 'react';

export function Select({ label, value, defaultValue, options = [], disabled = false, onChange, style }) {
  const [focus, setFocus] = React.useState(false);
  return (
    <label style={{ display: 'block', position: 'relative', ...style }}>
      {label && (
        <span style={{ display: 'block', marginBottom: 6, fontSize: 'var(--text-2xs)', fontWeight: 'var(--weight-black)', letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)' }}>{label}</span>
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
            width: '100%', height: 'var(--control-h-md)', padding: '0 34px 0 12px',
            fontFamily: 'var(--font-ui)', fontSize: 'var(--text-md)', fontWeight: 'var(--weight-medium)',
            color: 'var(--text-primary)', background: 'var(--surface-inset)',
            border: '1.5px solid ' + (focus ? 'var(--honey-500)' : 'var(--border-default)'),
            borderRadius: 'var(--radius-sm)', outline: 'none', appearance: 'none', WebkitAppearance: 'none',
            boxShadow: focus ? 'var(--focus-ring)' : 'none',
            opacity: disabled ? 0.45 : 1, cursor: disabled ? 'not-allowed' : 'pointer',
          }}
        >
          {options.map((o) => {
            const opt = typeof o === 'string' ? { value: o, label: o } : o;
            return <option key={opt.value} value={opt.value}>{opt.label}</option>;
          })}
        </select>
        <span style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', pointerEvents: 'none', color: 'var(--text-tertiary)', fontSize: 11 }}>▼</span>
      </span>
    </label>
  );
}
