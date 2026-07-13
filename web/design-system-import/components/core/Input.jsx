import React from 'react';

export function Input({ label, value, defaultValue, placeholder, type = 'text', mono = false, error, disabled = false, onChange, style }) {
  const [focus, setFocus] = React.useState(false);
  return (
    <label style={{ display: 'block', ...style }}>
      {label && (
        <span style={{ display: 'block', marginBottom: 6, fontSize: 'var(--text-2xs)', fontWeight: 'var(--weight-black)', letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)' }}>{label}</span>
      )}
      <input
        type={type}
        value={value}
        defaultValue={defaultValue}
        placeholder={placeholder}
        disabled={disabled}
        onChange={onChange}
        onFocus={() => setFocus(true)}
        onBlur={() => setFocus(false)}
        style={{
          width: '100%', height: 'var(--control-h-md)', padding: '0 12px',
          fontFamily: mono ? 'var(--font-mono)' : 'var(--font-ui)', fontSize: 'var(--text-md)',
          color: 'var(--text-primary)', background: 'var(--surface-inset)',
          border: '1.5px solid ' + (error ? 'var(--berry-500)' : focus ? 'var(--honey-500)' : 'var(--border-default)'),
          borderRadius: 'var(--radius-sm)', outline: 'none',
          boxShadow: focus ? 'var(--focus-ring)' : 'none',
          opacity: disabled ? 0.45 : 1,
          transition: 'border-color var(--dur-fast) var(--ease-out), box-shadow var(--dur-fast) var(--ease-out)',
        }}
      />
      {error && <span style={{ display: 'block', marginTop: 5, fontSize: 'var(--text-xs)', color: 'var(--berry-400)', fontWeight: 'var(--weight-bold)' }}>{error}</span>}
    </label>
  );
}
