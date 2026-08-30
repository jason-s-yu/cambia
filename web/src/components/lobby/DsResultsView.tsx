// src/components/lobby/DsResultsView.tsx
// End-of-game results (cambia-484, restyled in cambia-848) for the post_game
// (casual single game) and match_end (ranked circuit) phases. Reads standings
// from matchState/lobbyDetails. "Back to lobby" calls onReturnToLobby, which
// LobbyPage sends to the hub as a return_to_lobby frame (cambia-1238); the
// results stay up until the server's phase_change answers it.
// Casual (post_game) standings have no matchState (ranked-only, see
// hub.buildLobbySnapshot), so final scores fall back to gameStore's finalScores,
// captured off the game_end event (cambia-510).
//
// When the finished table is still in the store the results render as an
// overlay above it, so the last board state stays visible under the scrim.
// Without one (a reload straight into post_game) the card sits on the ground.
//
// Over the table the card is a modal: aria-modal, focus moves to the primary
// action on mount, Tab cycles inside the card, and the table underneath is
// inert so its cards and leave control drop out of the tab order (cambia-848).
import React, { useEffect, useMemo, useRef } from 'react';
import { useCurrentLobbyStore, type LobbyPhase } from '@/stores/lobbyStore';
import { useAuthStore } from '@/stores/authStore';
import { useGameStore, selectFinalScores } from '@/stores/gameStore';
import type { ClientGameAction, ObfGameState } from '@/types/game';
import Button from '@/components/ds/core/Button';
import Badge from '@/components/ds/core/Badge';
import { EYEBROW } from '@/components/ds/eyebrow';
import ScorePill from '@/components/ds/game/ScorePill';
import DsGameTable from '@/components/game/DsGameTable';

interface DsResultsViewProps {
  phase: LobbyPhase;
  onReturnToLobby: () => void;
  onLeave: () => void;
  /** The finished table, rendered under the results when still available. */
  gameState?: ObfGameState | null;
  sendMessage?: (msg: ClientGameAction) => void;
}

const DsResultsView: React.FC<DsResultsViewProps> = ({ phase, onReturnToLobby, onLeave, gameState, sendMessage }) => {
  const matchState = useCurrentLobbyStore((s) => s.matchState);
  const lobbyPlayers = useCurrentLobbyStore((s) => s.lobbyDetails?.lobby_status?.users);
  const selfId = useAuthStore((s) => s.user?.id);
  const authName = useAuthStore((s) => s.user?.username);
  const finalScores = useGameStore(selectFinalScores);

  const isMatchEnd = phase === 'match_end';
  const title = isMatchEnd ? 'Final standings' : 'Game over';
  const overTable = !!gameState && !!sendMessage;

  const cardRef = useRef<HTMLElement>(null);
  useEffect(() => {
    cardRef.current?.querySelector<HTMLElement>('button')?.focus();
  }, []);

  // Keep Tab inside the card while it covers the table. The table is inert, but the app
  // chrome above it is not, and a modal must not hand focus to what its scrim hides.
  const trapTab = (e: React.KeyboardEvent<HTMLElement>) => {
    if (!overTable || e.key !== 'Tab' || !cardRef.current) return;
    const focusable = Array.from(cardRef.current.querySelectorAll<HTMLElement>('button:not([disabled]), [href], [tabindex]:not([tabindex="-1"])'));
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && (active === first || !cardRef.current.contains(active))) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && (active === last || !cardRef.current.contains(active))) {
      e.preventDefault();
      first.focus();
    }
  };

  // Display names: lobby roster, then the game snapshot (its username can arrive empty),
  // then the signed-in user's own name, then a seat number.
  const names = useMemo(() => {
    const m = new Map<string, string>();
    (lobbyPlayers ?? []).forEach((u) => { if (u.username) m.set(u.id, u.username); });
    (gameState?.players ?? []).forEach((p, i) => {
      if (!m.get(p.playerId)) m.set(p.playerId, p.username || (p.playerId === selfId && authName) || `Player ${i + 1}`);
    });
    if (selfId && !m.get(selfId) && authName) m.set(selfId, authName);
    return m;
  }, [lobbyPlayers, gameState, selfId, authName]);

  const cumulative = matchState?.cumulativeScores;
  const standings = useMemo(() => {
    if (cumulative && Object.keys(cumulative).length > 0) {
      return Object.keys(cumulative)
        .map((id) => ({ id, name: names.get(id) ?? id.substring(0, 6), score: cumulative[id] }))
        .sort((a, b) => a.score - b.score);
    }
    if (finalScores && Object.keys(finalScores).length > 0) {
      return Object.keys(finalScores)
        .map((id) => ({ id, name: names.get(id) ?? id.substring(0, 6), score: finalScores[id] }))
        .sort((a, b) => a.score - b.score);
    }
    return (lobbyPlayers ?? []).map((u) => ({ id: u.id, name: u.username, score: null as number | null }));
  }, [cumulative, finalScores, names, lobbyPlayers]);

  const ratingChanges = matchState?.ratingChanges;
  const scored = standings.filter((r) => r.score !== null);
  const winner = scored[0] ?? null;
  const own = standings.find((r) => r.id === selfId) ?? null;
  const ownWon = !!winner && winner.id === selfId;
  const caller = gameState?.cambiaCalled ? names.get(gameState.cambiaCallerId ?? '') : undefined;
  // Ids, not display names: two seats can carry the same name, and a seat whose
  // name fell back to a placeholder never matches (cambia-876, DL-4 review F12).
  const callerIsSelf = !!selfId && gameState?.cambiaCallerId === selfId;

  const card = (
    <section
      ref={cardRef}
      role='dialog'
      aria-modal={overTable ? 'true' : undefined}
      aria-labelledby='results-title'
      onKeyDown={trapTab}
      style={{
        width: '100%',
        maxWidth: 520,
        margin: 'auto',
        background: 'var(--surface-1)',
        border: '1px solid var(--border-default)',
        borderRadius: 'var(--ds-radius-lg)',
        boxShadow: gameState ? 'var(--shadow-overlay)' : 'none',
        color: 'var(--text-primary)',
        overflow: 'hidden'
      }}
    >
      <div style={{ padding: '18px 20px 14px', display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span style={EYEBROW}>{isMatchEnd ? 'Circuit' : 'Casual game'}</span>
          {isMatchEnd && matchState?.isRanked && <Badge tone='gold'>Ranked</Badge>}
          {matchState && <Badge tone='info'>Round {matchState.currentRound}/{matchState.totalRounds}</Badge>}
        </div>
        <h1 id='results-title' style={{ margin: 0, fontSize: 'var(--ds-text-2xl)', fontWeight: 'var(--weight-bold)', letterSpacing: 'var(--ds-tracking-tight)', lineHeight: 'var(--ds-leading-tight)' }}>{title}</h1>
        {winner && (
          <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>
            {ownWon ? 'You win.' : `${winner.name} wins.`}{caller ? ` Cambia was called by ${callerIsSelf ? 'you' : caller}.` : ''}
          </div>
        )}
      </div>

      {own && own.score !== null && (
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, padding: '0 20px 14px' }}>
          <span style={{ fontSize: 'var(--ds-text-4xl)', fontWeight: 'var(--weight-black)', letterSpacing: 'var(--ds-tracking-tight)', lineHeight: 1, fontVariantNumeric: 'tabular-nums', color: ownWon ? 'var(--accent-gold)' : 'var(--text-primary)' }}>
            {own.score}
          </span>
          <span style={EYEBROW}>{isMatchEnd ? 'Your total' : 'Your score'}</span>
        </div>
      )}

      <div style={{ borderTop: '1px solid var(--border-subtle)', padding: '12px 20px 16px' }}>
        <div style={{ ...EYEBROW, marginBottom: 6 }}>{isMatchEnd ? 'Standings' : 'Scores'}</div>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {standings.map((row, i) => (
            <div key={row.id} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 0', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
              <span style={{ fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)', width: 16, fontVariantNumeric: 'tabular-nums' }}>{i + 1}</span>
              <span style={{ fontWeight: 'var(--weight-medium)', flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: row.id === selfId ? 'var(--accent-gold)' : 'var(--text-primary)' }}>
                {row.name}{row.id === selfId ? ' (you)' : ''}
              </span>
              {winner && row.id === winner.id && <Badge tone='success'>Winner</Badge>}
              {row.score !== null && <span style={{ fontWeight: 'var(--weight-bold)', fontVariantNumeric: 'tabular-nums', minWidth: 28, textAlign: 'right' }}>{row.score}</span>}
            </div>
          ))}
          {standings.length === 0 && (
            <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-tertiary)' }}>No results yet.</div>
          )}
        </div>
      </div>

      {isMatchEnd && ratingChanges && Object.keys(ratingChanges).length > 0 && (
        <div style={{ borderTop: '1px solid var(--border-subtle)', padding: '12px 20px 16px' }}>
          <div style={{ ...EYEBROW, marginBottom: 6 }}>Rating changes</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {Object.entries(ratingChanges).map(([id, change]) => {
              const before = Math.round(change.before);
              const after = Math.round(change.after);
              const diff = after - before;
              const sign = diff >= 0 ? '+' : '';
              return (
                <div key={id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, fontSize: 'var(--ds-text-sm)' }}>
                  <span style={{ fontWeight: 'var(--weight-medium)', minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{names.get(id) ?? id.substring(0, 6)}</span>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>{before} -&gt; {after}</span>
                    <ScorePill value={`${sign}${diff}`} tone={diff >= 0 ? 'success' : 'danger'} />
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', padding: '14px 20px', borderTop: '1px solid var(--border-subtle)', background: 'var(--surface-2)' }}>
        <Button variant='primary' onClick={onReturnToLobby}>Back to lobby</Button>
        <Button variant='ghost' onClick={onLeave}>Leave lobby</Button>
      </div>
    </section>
  );

  if (gameState && sendMessage) {
    return (
      <>
        <div inert style={{ display: 'contents' }}>
          <DsGameTable gameState={gameState} phase={phase} sendMessage={sendMessage} onLeave={onLeave} />
        </div>
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 100,
            display: 'flex',
            padding: '24px 16px',
            overflowY: 'auto',
            background: 'var(--surface-overlay)'
          }}
        >
          {card}
        </div>
      </>
    );
  }

  return (
    <div style={{ flex: 1, display: 'flex', padding: '24px 16px' }}>
      {card}
    </div>
  );
};

export default DsResultsView;
