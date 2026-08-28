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
  /**
   * Set when the bar sits directly on the felt. The neutral text and inset tiers flip with
   * the theme and lose contrast on green, so the label, track and countdown switch to the
   * on-felt tokens; the low-time signal then comes from the fill alone (cambia-848).
   */
  onFelt?: boolean;
  style?: React.CSSProperties;
}

/** Turn timer: gold bar that turns danger-red in the last quarter; tabular countdown. */
const TimerBar: React.FC<TimerBarProps> = ({
  totalSec = 30,
  remainingSec = 30,
  deadlineMs = null,
  clockOffsetMs = 0,
  label,
  onFelt = false,
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
  const labelColor = onFelt ? 'var(--text-on-felt-muted)' : 'var(--text-tertiary)';
  const trackBg = onFelt ? 'var(--surface-felt-deep)' : 'var(--surface-inset)';
  const trackBorder = onFelt ? 'var(--border-on-felt)' : 'var(--border-default)';
  const countColor = onFelt ? 'var(--text-on-green)' : low ? 'var(--status-danger)' : 'var(--text-primary)';
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, ...style }}>
      {label && (
        <span style={{ fontSize: 'var(--text-2xs)', fontWeight: 'var(--weight-bold)', letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: labelColor, flex: 'none' }}>
          {label}
        </span>
      )}
      <div style={{ flex: 1, height: 6, background: trackBg, border: '1px solid ' + trackBorder, borderRadius: 'var(--radius-pill)', overflow: 'hidden' }}>
        <div
          style={{
            width: frac * 100 + '%',
            height: '100%',
            background: low ? 'var(--accent-danger)' : 'var(--accent-gold)',
            borderRadius: 'var(--radius-pill)',
            transition: 'width 1s linear, background var(--dur-med) var(--ds-ease-out)'
          }}
        ></div>
      </div>
      <span style={{ fontWeight: 'var(--weight-bold)', fontVariantNumeric: 'tabular-nums', fontSize: 'var(--ds-text-sm)', color: countColor, flex: 'none' }}>
        {mm}:{ss}
      </span>
    </div>
  );
};

export default TimerBar;
