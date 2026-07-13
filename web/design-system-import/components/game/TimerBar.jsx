import React from 'react';

export function TimerBar({ totalSec = 30, remainingSec = 30, label, style }) {
  const frac = Math.max(0, Math.min(1, remainingSec / totalSec));
  const low = frac <= 0.25;
  const mm = Math.floor(remainingSec / 60);
  const ss = String(Math.max(0, Math.floor(remainingSec % 60))).padStart(2, '0');
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, ...style }}>
      {label && <span style={{ fontSize: 'var(--text-2xs)', fontWeight: 800, letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)', flex: 'none' }}>{label}</span>}
      <div style={{ flex: 1, height: 10, background: 'var(--surface-inset)', border: '1.5px solid var(--border-default)', borderRadius: 'var(--radius-pill)', overflow: 'hidden' }}>
        <div style={{
          width: (frac * 100) + '%', height: '100%',
          background: low ? 'var(--berry-500)' : 'var(--honey-500)',
          borderRadius: 'var(--radius-pill)',
          transition: 'width 1s linear, background var(--dur-med) var(--ease-out)',
        }}></div>
      </div>
      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 'var(--text-sm)', color: low ? 'var(--berry-400)' : 'var(--honey-400)', flex: 'none' }}>{mm}:{ss}</span>
    </div>
  );
}
