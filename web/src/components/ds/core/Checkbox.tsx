import React, { useState } from 'react';

export interface CheckboxProps {
  label?: string;
  /** Secondary line, e.g. the house-rule explanation. */
  description?: string;
  checked?: boolean;
  defaultChecked?: boolean;
  disabled?: boolean;
  onChange?: (checked: boolean) => void;
  style?: React.CSSProperties;
}

/** Chunky game-piece checkbox with label + optional description (house rules). */
const Checkbox: React.FC<CheckboxProps> = ({ label, description, checked, defaultChecked, disabled = false, onChange, style }) => {
  const [internal, setInternal] = useState(!!defaultChecked);
  const isOn = checked !== undefined ? checked : internal;
  const toggle = () => {
    if (disabled) return;
    if (checked === undefined) setInternal(!internal);
    if (onChange) onChange(!isOn);
  };
  return (
    <div
      onClick={toggle}
      style={{ display: 'flex', gap: 10, alignItems: 'flex-start', cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.45 : 1, ...style }}
    >
      <span
        style={{
          width: 22,
          height: 22,
          flex: 'none',
          marginTop: 1,
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: isOn ? 'var(--ember-500)' : 'var(--surface-inset)',
          border: 'var(--line-thick) solid ' + (isOn ? 'var(--outline-ink)' : 'var(--border-strong)'),
          borderRadius: 7,
          boxShadow: isOn ? 'var(--shadow-piece)' : 'none',
          color: 'var(--text-on-ember)',
          fontSize: 14,
          fontWeight: 800,
          lineHeight: 1,
          transition: 'background var(--dur-fast) var(--ds-ease-out)'
        }}
      >
        {isOn ? '✓' : ''}
      </span>
      <span style={{ flex: 1, minWidth: 0 }}>
        {label && <span style={{ display: 'block', fontWeight: 'var(--weight-bold)', fontSize: 'var(--text-md)', lineHeight: 1.3 }}>{label}</span>}
        {description && <span style={{ display: 'block', fontSize: 'var(--ds-text-xs)', color: 'var(--text-secondary)', marginTop: 2 }}>{description}</span>}
      </span>
    </div>
  );
};

export default Checkbox;
