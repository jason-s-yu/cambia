import React from 'react';
import Badge from '@/components/ds/core/Badge';
import Button from '@/components/ds/core/Button';
import IconButton from '@/components/ds/core/IconButton';
import Panel from '@/components/ds/chrome/Panel';
import type { ActiveSession } from '@/types';
import { gameModeLabel } from '@/utils/gameMode';

export interface DsResumeBannerProps {
  session: ActiveSession;
  /** Navigate back into the lobby/game. */
  onResume: () => void;
  /** Hide the banner for this tab (e.g. the user left on purpose). */
  onDismiss: () => void;
}

/** Headline copy per session shape: a seat in a running game is the urgent case. */
function headline(session: ActiveSession): string {
  if (session.phase === 'in_game') {
    return session.seated ? 'Game in progress' : 'Game running in your lobby';
  }
  if (session.phase === 'searching') {
    return 'Searching for a match';
  }
  return 'Lobby open';
}

function badgeTone(session: ActiveSession): 'warning' | 'info' | 'neutral' {
  if (session.phase === 'in_game') return 'warning';
  if (session.phase === 'searching') return 'info';
  return 'neutral';
}

function badgeLabel(session: ActiveSession): string {
  if (session.phase === 'in_game') return 'In game';
  if (session.phase === 'searching') return 'Searching';
  return 'In lobby';
}

/**
 * Home-screen affordance for returning to the lobby or game the server still counts you in
 * (cambia-783). Rendered only when GET /lobby/active reports a session, so it is absent for a
 * user with nothing to resume. The border steps up to strong for attention; the gold accent
 * stays on the resume button alone.
 */
const DsResumeBanner: React.FC<DsResumeBannerProps> = ({ session, onResume, onDismiss }) => {
  const lobbyLabel = session.name || `Lobby ${session.lobbyId.substring(0, 6)}`;
  const players = `${session.playerCount} ${session.playerCount === 1 ? 'player' : 'players'}`;

  return (
    <Panel style={{ borderColor: 'var(--border-strong)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
        <Badge tone={badgeTone(session)} dot>{badgeLabel(session)}</Badge>
        <div style={{ flex: 1, minWidth: 200, lineHeight: 'var(--ds-leading-snug)' }}>
          <div style={{ fontSize: 'var(--ds-text-lg)', fontWeight: 'var(--weight-bold)', letterSpacing: 'var(--ds-tracking-tight)', color: 'var(--text-primary)' }}>
            {headline(session)}
          </div>
          <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
            {lobbyLabel} · {gameModeLabel(session.gameMode)} · {players}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginLeft: 'auto' }}>
          <Button variant='primary' onClick={onResume}>
            {session.phase === 'in_game' && session.seated ? 'Rejoin game' : 'Back to lobby'}
          </Button>
          <IconButton variant='ghost' title='Dismiss' onClick={onDismiss}>
            <svg width='14' height='14' viewBox='0 0 14 14' fill='none' stroke='currentColor' strokeWidth='2' strokeLinecap='round' aria-hidden='true'>
              <path d='M2 2 L12 12 M12 2 L2 12' />
            </svg>
          </IconButton>
        </div>
      </div>
    </Panel>
  );
};

export default DsResumeBanner;
