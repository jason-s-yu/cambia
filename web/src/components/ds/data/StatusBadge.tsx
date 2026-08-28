import React from 'react';

export type Status = 'created' | 'starting' | 'running' | 'stopping' | 'stopped' | 'crashed' | 'queued' | 'succeeded' | 'failed';

export interface StatusBadgeProps {
  status?: Status;
  style?: React.CSSProperties;
}

interface StatusSpec {
  color: string;
  bg: string;
  border: string;
}

const SUCCESS: StatusSpec = { color: 'var(--status-success)', bg: 'var(--status-success-bg)', border: 'var(--status-success-border)' };
const INFO: StatusSpec = { color: 'var(--status-info)', bg: 'var(--status-info-bg)', border: 'var(--status-info-border)' };
const WARNING: StatusSpec = { color: 'var(--status-warning)', bg: 'var(--status-warning-bg)', border: 'var(--status-warning-border)' };
const DANGER: StatusSpec = { color: 'var(--status-danger)', bg: 'var(--status-danger-bg)', border: 'var(--status-danger-border)' };

const MAP: Record<Status, StatusSpec> = {
  running: SUCCESS,
  succeeded: SUCCESS,
  created: INFO,
  queued: INFO,
  starting: WARNING,
  stopping: WARNING,
  stopped: { color: 'var(--text-secondary)', bg: 'var(--surface-2)', border: 'var(--border-default)' },
  crashed: DANGER,
  failed: DANGER
};

/** Process/eval status pill (training dashboard): uppercase label + dot. */
const StatusBadge: React.FC<StatusBadgeProps> = ({ status = 'created', style }) => {
  const m = MAP[status] || MAP.created;
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '2px 9px',
        borderRadius: 'var(--radius-pill)',
        background: m.bg,
        border: '1px solid ' + m.border,
        color: m.color,
        fontFamily: 'var(--font-sans)',
        fontSize: 'var(--text-2xs)',
        fontWeight: 'var(--weight-bold)',
        textTransform: 'uppercase',
        letterSpacing: 'var(--tracking-caps)',
        whiteSpace: 'nowrap',
        ...style
      }}
    >
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'currentColor', flex: 'none' }}></span>
      {status}
    </span>
  );
};

export default StatusBadge;
