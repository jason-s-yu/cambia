import React from 'react';

let injected = false;
function injectKeyframes() {
  if (injected || typeof document === 'undefined') return;
  const el = document.createElement('style');
  el.textContent = '@keyframes cambia-spin { to { transform: rotate(360deg); } }';
  document.head.appendChild(el);
  injected = true;
}

export function Spinner({ size = 24, label, style }) {
  injectKeyframes();
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 10, ...style }}>
      <span style={{
        width: size, height: size, flex: 'none', borderRadius: '50%',
        border: Math.max(2, Math.round(size / 9)) + 'px solid var(--border-default)',
        borderTopColor: 'var(--ember-500)',
        animation: 'cambia-spin 0.9s linear infinite',
      }}></span>
      {label && <span style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', fontWeight: 'var(--weight-bold)' }}>{label}</span>}
    </span>
  );
}
