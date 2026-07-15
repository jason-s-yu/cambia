import React, { useEffect, useState } from 'react';

export interface TimerBarProps {
  totalSec?: number;
  remainingSec?: number;
  /**
   * Absolute server-clock epoch-ms deadline for a live countdown (cambia-488). When set
   * (non-null), remainingSec is ignored and the bar ticks down in real time:
   * remaining = (deadlineMs + clockOffsetMs - Date.now()) / 1000, clamped to [0, totalSec].
   * Leave null/undefined to fall back to the static informational render (remainingSec as-is,
   * used when the game has no turn timer configured).
   */
  deadlineMs?: number | null;
  /**
   * serverNow - clientNow offset in ms, captured from the same event that carried deadlineMs.
   * Corrects for client clock skew so the countdown tracks the server's actual deadline.
   */
  clockOffsetMs?: number;
  label?: string;
  style?: React.CSSProperties;
}

/** Turn timer: honey bar that turns berry in the last quarter; mono countdown. */
const TimerBar: React.FC<TimerBarProps> = ({
  totalSec = 30,
  remainingSec = 30,
  deadlineMs = null,
  clockOffsetMs = 0,
  label,
  style
}) => {
  const isLive = deadlineMs != null;

  const [liveRemainingSec, setLiveRemainingSec] = useState(() =>
    deadlineMs != null ? Math.max(0, (deadlineMs + clockOffsetMs - Date.now()) / 1000) : remainingSec
  );

  useEffect(() => {
    if (deadlineMs == null) return;
    const tick = () => Math.max(0, (deadlineMs + clockOffsetMs - Date.now()) / 1000);
    setLiveRemainingSec(tick());
    const id = setInterval(() => {
      const next = tick();
      setLiveRemainingSec(next);
      if (next <= 0) clearInterval(id);
    }, 250);
    return () => clearInterval(id);
  }, [deadlineMs, clockOffsetMs]);

  const effectiveRemaining = isLive ? liveRemainingSec : remainingSec;
  const frac = Math.max(0, Math.min(1, effectiveRemaining / totalSec));
  const low = frac <= 0.25;
  const mm = Math.floor(effectiveRemaining / 60);
  const ss = String(Math.max(0, Math.floor(effectiveRemaining % 60))).padStart(2, '0');
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, ...style }}>
      {label && (
        <span style={{ fontSize: 'var(--text-2xs)', fontWeight: 800, letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)', flex: 'none' }}>
          {label}
        </span>
      )}
      <div style={{ flex: 1, height: 10, background: 'var(--surface-inset)', border: '1.5px solid var(--border-default)', borderRadius: 'var(--radius-pill)', overflow: 'hidden' }}>
        <div
          style={{
            width: frac * 100 + '%',
            height: '100%',
            background: low ? 'var(--berry-500)' : 'var(--honey-500)',
            borderRadius: 'var(--radius-pill)',
            transition: 'width 1s linear, background var(--dur-med) var(--ds-ease-out)'
          }}
        ></div>
      </div>
      <span style={{ fontFamily: 'var(--ds-font-mono)', fontWeight: 700, fontSize: 'var(--ds-text-sm)', color: low ? 'var(--berry-400)' : 'var(--honey-400)', flex: 'none' }}>
        {mm}:{ss}
      </span>
    </div>
  );
};

export default TimerBar;
