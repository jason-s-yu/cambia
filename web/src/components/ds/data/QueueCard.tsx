import React from 'react';
import Button from '../core/Button';
import Badge from '../core/Badge';

export interface QueueCardProps {
  name?: string;
  tagline?: string;
  players?: number;
  rounds?: number;
  /** Estimated match length in minutes. */
  minutes?: number;
  /** Player-readable rating pool label, e.g. "H2H Ranked pool" or "FFA-4 pool". */
  pool?: string;
  ranked?: boolean;
  onPlay?: () => void;
  style?: React.CSSProperties;
}

/** Matchmaking queue card (h2h_rapid, ffa4_standard, ...) with Play action. */
const QueueCard: React.FC<QueueCardProps> = ({
  name = 'H2H Rapid',
  tagline,
  players = 2,
  rounds = 8,
  minutes = 20,
  pool,
  ranked = true,
  onPlay,
  style
}) => {
  return (
    <div
      style={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border-default)',
        borderRadius: 'var(--ds-radius-lg)',
        padding: 'var(--space-4) var(--space-5)',
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
        ...style
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 10 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 'var(--ds-text-lg)', fontWeight: 'var(--weight-bold)', letterSpacing: 'var(--ds-tracking-tight)', lineHeight: 'var(--ds-leading-tight)' }}>{name}</div>
          {tagline && <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)', marginTop: 2 }}>{tagline}</div>}
        </div>
        {ranked ? <Badge tone='warning'>Ranked</Badge> : <Badge>Casual</Badge>}
      </div>
      {/* The facts row is what a player compares queues on, so it reads at the
          body size rather than the scale's floor (cambia-1097). The pool label
          sits on its own line below it, never inside the wrap row: a short pool
          name would otherwise ride the facts line on one card and drop below it
          on the next, and the cards would come out different heights. */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 0, fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 14px', fontVariantNumeric: 'tabular-nums' }}>
          <span>{players}p</span>
          <span>
            {rounds} {rounds === 1 ? 'round' : 'rounds'}
          </span>
          <span>~{minutes} min</span>
        </div>
        {pool && <div style={{ color: 'var(--text-tertiary)' }}>{pool}</div>}
      </div>
      <Button variant='secondary' fullWidth onClick={onPlay}>
        Play
      </Button>
    </div>
  );
};

export default QueueCard;
