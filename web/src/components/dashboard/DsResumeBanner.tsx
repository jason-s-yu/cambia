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
    return session.seated ? 'Your game is in progress' : 'A game is running in your lobby';
  }
  if (session.phase === 'searching') {
    return 'Your lobby is searching for a match';
  }
  return 'You are still in a lobby';
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
 * user with nothing to resume.
 */
const DsResumeBanner: React.FC<DsResumeBannerProps> = ({ session, onResume, onDismiss }) => {
  const lobbyLabel = session.name || `Lobby ${session.lobbyId.substring(0, 6)}`;
  const players = `${session.playerCount} ${session.playerCount === 1 ? 'player' : 'players'}`;

  return (
    <Panel style={{ borderColor: 'var(--border-strong)' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
        <Badge tone={badgeTone(session)} dot>{badgeLabel(session)}</Badge>
        <div style={{ flex: 1, minWidth: 200, lineHeight: 1.3 }}>
          <div style={{ fontFamily: 'var(--font-display)', fontSize: 'var(--ds-text-lg)' }}>{headline(session)}</div>
          <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>
            {lobbyLabel} · {gameModeLabel(session.gameMode)} · {players}
          </div>
        </div>
        <Button variant='primary' onClick={onResume}>
          {session.phase === 'in_game' && session.seated ? 'Rejoin game' : 'Return to lobby'}
        </Button>
        <IconButton size='sm' variant='ghost' title='Dismiss' onClick={onDismiss}>
          <svg width='14' height='14' viewBox='0 0 14 14' fill='none' stroke='currentColor' strokeWidth='2' strokeLinecap='round'>
            <path d='M2 2 L12 12 M12 2 L2 12' />
          </svg>
        </IconButton>
      </div>
    </Panel>
  );
};

export default DsResumeBanner;
