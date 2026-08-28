import React, { useState } from 'react';

export interface SwitchProps {
  label?: string;
  checked?: boolean;
  defaultChecked?: boolean;
  disabled?: boolean;
  onChange?: (checked: boolean) => void;
  style?: React.CSSProperties;
}

/** Flat on/off toggle (gold when on). Use Checkbox for multi-line rule settings. */
const Switch: React.FC<SwitchProps> = ({ label, checked, defaultChecked, disabled = false, onChange, style }) => {
  const [internal, setInternal] = useState(!!defaultChecked);
  const isOn = checked !== undefined ? checked : internal;
  const toggle = () => {
    if (disabled) return;
    if (checked === undefined) setInternal(!internal);
    if (onChange) onChange(!isOn);
  };
  return (
    <div onClick={toggle} style={{ display: 'inline-flex', gap: 10, alignItems: 'center', cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.55 : 1, ...style }}>
      <span
        style={{
          width: 38,
          height: 22,
          flex: 'none',
          position: 'relative',
          boxSizing: 'border-box',
          background: isOn ? 'var(--accent-gold)' : 'var(--surface-inset)',
          border: '1px solid ' + (isOn ? 'var(--accent-gold)' : 'var(--border-strong)'),
          borderRadius: 'var(--radius-pill)',
          transition: 'background var(--dur-fast) var(--ds-ease-out), border-color var(--dur-fast) var(--ds-ease-out)'
        }}
      >
        <span
          style={{
            position: 'absolute',
            top: 3,
            left: isOn ? 19 : 3,
            width: 14,
            height: 14,
            background: isOn ? 'var(--text-on-gold)' : 'var(--text-tertiary)',
            borderRadius: '50%',
            transition: 'left var(--dur-fast) var(--ds-ease-out), background var(--dur-fast) var(--ds-ease-out)'
          }}
        ></span>
      </span>
      {label && <span style={{ fontWeight: 'var(--weight-medium)', fontSize: 'var(--text-md)' }}>{label}</span>}
    </div>
  );
};

export default Switch;
