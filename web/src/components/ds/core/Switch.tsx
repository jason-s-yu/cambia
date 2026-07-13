import React, { useState } from 'react';

export interface SwitchProps {
  label?: string;
  checked?: boolean;
  defaultChecked?: boolean;
  disabled?: boolean;
  onChange?: (checked: boolean) => void;
  style?: React.CSSProperties;
}

/** On/off toggle (moss when on). Use Checkbox for multi-line rule settings. */
const Switch: React.FC<SwitchProps> = ({ label, checked, defaultChecked, disabled = false, onChange, style }) => {
  const [internal, setInternal] = useState(!!defaultChecked);
  const isOn = checked !== undefined ? checked : internal;
  const toggle = () => {
    if (disabled) return;
    if (checked === undefined) setInternal(!internal);
    if (onChange) onChange(!isOn);
  };
  return (
    <div onClick={toggle} style={{ display: 'inline-flex', gap: 10, alignItems: 'center', cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.45 : 1, ...style }}>
      <span
        style={{
          width: 44,
          height: 24,
          flex: 'none',
          position: 'relative',
          background: isOn ? 'var(--moss-500)' : 'var(--surface-inset)',
          border: 'var(--line-thick) solid ' + (isOn ? 'var(--outline-ink)' : 'var(--border-strong)'),
          borderRadius: 'var(--radius-pill)',
          transition: 'background var(--dur-fast) var(--ds-ease-out)'
        }}
      >
        <span
          style={{
            position: 'absolute',
            top: 2,
            left: isOn ? 22 : 2,
            width: 16,
            height: 16,
            background: 'var(--parchment-100)',
            borderRadius: '50%',
            border: '1.5px solid var(--outline-ink)',
            transition: 'left var(--dur-fast) var(--ds-ease-out)'
          }}
        ></span>
      </span>
      {label && <span style={{ fontWeight: 'var(--weight-bold)', fontSize: 'var(--text-md)' }}>{label}</span>}
    </div>
  );
};

export default Switch;
