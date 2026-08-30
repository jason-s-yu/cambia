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

/** Flat checkbox with label and optional description (house rules). */
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
      style={{ display: 'flex', gap: 10, alignItems: 'flex-start', cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.55 : 1, ...style }}
    >
      <span
        style={{
          width: 18,
          height: 18,
          flex: 'none',
          marginTop: 2,
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: isOn ? 'var(--accent-gold)' : 'var(--surface-inset)',
          border: '1px solid ' + (isOn ? 'var(--accent-gold)' : 'var(--border-strong)'),
          borderRadius: 'var(--ds-radius-sm)',
          color: 'var(--text-on-gold)',
          fontSize: 12,
          fontWeight: 'var(--weight-black)',
          lineHeight: 1,
          transition: 'background var(--dur-fast) var(--ds-ease-out), border-color var(--dur-fast) var(--ds-ease-out)'
        }}
      >
        {isOn ? '✓' : ''}
      </span>
      <span style={{ flex: 1, minWidth: 0 }}>
        {label && <span style={{ display: 'block', fontWeight: 'var(--weight-medium)', fontSize: 'var(--text-md)', lineHeight: 1.35 }}>{label}</span>}
        {/* Body copy, so a step above the scale's floor: the house-rule
            explanations are the sentences a host actually reads (cambia-1097). */}
        {description && <span style={{ display: 'block', fontSize: 'var(--ds-text-sm)', lineHeight: 'var(--ds-leading-snug)', color: 'var(--text-secondary)', marginTop: 2 }}>{description}</span>}
      </span>
    </div>
  );
};

export default Checkbox;
