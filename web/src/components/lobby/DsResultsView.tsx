// src/components/lobby/DsResultsView.tsx
// End-of-game results (cambia-484, restyled in cambia-848) for the post_game
// (casual single game) and match_end (ranked circuit) phases. Reads standings
// from matchState/lobbyDetails. "Back to lobby" calls onReturnToLobby, which
// LobbyPage sends to the hub as a return_to_lobby frame (cambia-1238); the
// results stay up until the server's phase_change answers it.
//
// The hub only admits that frame from the host, widened to any seated player
// where the host role belongs to the system (Hub.mayReturnToLobby,
// cambia-1238) - a matchmade lobby has no player host to gate on. A seat the
// hub would refuse never sees the button: before cambia-1516 every seat saw
// it and a non-host click came back as an error envelope. Those seats see a
// waiting label instead, naming the results timer where one is actually
// armed - only post_game arms one; match_end does not (see
// service/doc/lobby_actions.md "Post-game exit").
// Casual (post_game) standings have no matchState (ranked-only, see
// hub.buildLobbySnapshot), so final scores fall back to gameStore's finalScores,
// captured off the game_end event (cambia-510).
//
// A game the service ended on an internal error is the one case this card
// shows no result at all (cambia-1831): the results frames carry a `reason`,
// and the scores beside it were read off whatever state the fault left, so the
// card reports the error and draws no scoreboard from them. The controls stay -
// the table is over either way and the seat still needs its way out.
//
// When the finished table is still in the store the results render as an
// overlay above it, so the last board state stays visible under the scrim.
// Without one (a reload straight into post_game) the card sits on the ground.
//
// Over the table the card is a modal: aria-modal, Tab cycles inside the card,
// and the table underneath is inert so its cards and leave control drop out of
// the tab order (cambia-848). Focus opens on the primary action where the seat
// has one and on the card itself where it does not: taking the first button
// instead put a non-host seat's opening focus on "Leave lobby" once cambia-1516
// replaced the primary with a waiting label, so the reflex Enter on a dialog
// that had just appeared left the lobby, unasked (cambia-1239 review).
import React, { useEffect, useMemo, useRef } from 'react';
import { useCurrentLobbyStore, type LobbyPhase } from '@/stores/lobbyStore';
import { useAuthStore } from '@/stores/authStore';
import { useGameStore, selectFinalScores, selectFinalHands, selectEndReason } from '@/stores/gameStore';
import type { ClientGameAction, ObfGameState } from '@/types/game';
import Button from '@/components/ds/core/Button';
import Badge from '@/components/ds/core/Badge';
import { EYEBROW } from '@/components/ds/eyebrow';
import PlayingCard from '@/components/ds/game/PlayingCard';
import ScorePill from '@/components/ds/game/ScorePill';
import { toDsCardFace, cardFaceName } from '@/components/game/dsCardMap';
import { roundCounterLabel } from '@/lib/roundCounter';
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
  const circuitEnabled = useCurrentLobbyStore((s) => s.lobbyDetails?.circuit?.enabled);
  const yourIsHost = useCurrentLobbyStore((s) => s.lobbyDetails?.your_is_host);
  const systemHost = useCurrentLobbyStore((s) => s.lobbyDetails?.system_host);
  // Mirrors Hub.mayReturnToLobby: the host, or (a matchmade lobby has none) any seat of a
  // system-hosted one. Everyone here is already a seat, since the results are their own.
  const canReturnToLobby = !!yourIsHost || !!systemHost;
  // Shown only where a circuit is playing the rounds it counts (lib/roundCounter.ts). The card
  // used to badge Round 0/8 over the standings of the one game a matchmade lobby plays
  // (cambia-1126 item 2).
  const roundCounter = roundCounterLabel({
    circuitEnabled,
    totalRounds: matchState?.totalRounds,
    currentRound: matchState?.currentRound
  });
  const selfId = useAuthStore((s) => s.user?.id);
  const authName = useAuthStore((s) => s.user?.username);
  const finalScores = useGameStore(selectFinalScores);
  // The round-end reveal (RULES.md 3C, cambia-1542), off game_end or off the game_results a
  // reload into the results is answered with. Keyed by seat so a standings row can draw the hand
  // that produced its score; a seat that forfeited is absent and draws none.
  const finalHands = useGameStore(selectFinalHands);
  const handsBySeat = useMemo(() => {
    const m = new Map<string, { id: string; rank: string; suit: string }[]>();
    (finalHands ?? []).forEach((h) => m.set(h.playerId, h.cards));
    return m;
  }, [finalHands]);

  // A game the service ended on an internal error carries a reason on its results frames
  // (cambia-1831). The scores that arrive with it are read off whatever state the fault left, so
  // this card reports the error and shows no scoreboard at all: a standings list drawn from them
  // would be read as the game's result, which is the one thing it is not. Any reason at all does
  // this, known copy or not; the copy below is what this client has for the one the service sends.
  const endReason = useGameStore(selectEndReason);
  const endedInError = !!endReason;

  const isMatchEnd = phase === 'match_end';
  const title = endedInError ? 'Game ended by an error' : isMatchEnd ? 'Final standings' : 'Game over';
  const overTable = !!gameState && !!sendMessage;

  const cardRef = useRef<HTMLElement>(null);
  useEffect(() => {
    // The primary action marks itself, so a card that renders none opens on the card (tabIndex -1)
    // rather than on whatever control happens to be drawn first. Nothing is armed under Enter, and
    // the reader still lands inside the dialog.
    const primary = cardRef.current?.querySelector<HTMLElement>('[data-autofocus]');
    (primary ?? cardRef.current)?.focus();
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
    // The card itself holds focus when no primary action rendered, and it is the start of the
    // cycle as much as the first control is, so shift+Tab from it wraps rather than leaving.
    if (e.shiftKey && (active === first || active === cardRef.current || !cardRef.current.contains(active))) {
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
  // The hands only go beside a score they actually explain: a single round's. A circuit's
  // standings are totals across every round played, and the reveal is one round's, so a hand
  // under a cumulative total would be read as its cause and would not be.
  const showsOneRound = !(cumulative && Object.keys(cumulative).length > 0);
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
      // Focusable only as the fallback target above: -1 keeps it out of the tab cycle, which is
      // also why trapTab's own query passes over it.
      tabIndex={-1}
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
          {endedInError && <Badge tone='danger'>Internal error</Badge>}
          {isMatchEnd && matchState?.isRanked && <Badge tone='gold'>Ranked</Badge>}
          {roundCounter && <Badge tone='info'>{roundCounter}</Badge>}
        </div>
        <h1 id='results-title' style={{ margin: 0, fontSize: 'var(--ds-text-2xl)', fontWeight: 'var(--weight-bold)', letterSpacing: 'var(--ds-tracking-tight)', lineHeight: 'var(--ds-leading-tight)' }}>{title}</h1>
        {winner && !endedInError && (
          <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>
            {ownWon ? 'You win.' : `${winner.name} wins.`}{caller ? ` Cambia was called by ${callerIsSelf ? 'you' : caller}.` : ''}
          </div>
        )}
      </div>

      {endedInError && (
        <div data-testid='results-internal-error' style={{ borderTop: '1px solid var(--border-subtle)', padding: '12px 20px 16px', fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)', lineHeight: 'var(--ds-leading-normal)' }}>
          The server hit an internal error and had to end this game. The scores it was holding came
          from an interrupted state, so they are not shown as a result and this game was not rated.
        </div>
      )}

      {!endedInError && own && own.score !== null && (
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, padding: '0 20px 14px' }}>
          <span style={{ fontSize: 'var(--ds-text-4xl)', fontWeight: 'var(--weight-black)', letterSpacing: 'var(--ds-tracking-tight)', lineHeight: 1, fontVariantNumeric: 'tabular-nums', color: ownWon ? 'var(--accent-gold)' : 'var(--text-primary)' }}>
            {own.score}
          </span>
          <span style={EYEBROW}>{isMatchEnd ? 'Your total' : 'Your score'}</span>
        </div>
      )}

      {!endedInError && (
      <div style={{ borderTop: '1px solid var(--border-subtle)', padding: '12px 20px 16px' }}>
        <div style={{ ...EYEBROW, marginBottom: 6 }}>{isMatchEnd ? 'Standings' : 'Scores'}</div>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {standings.map((row, i) => {
            const hand = showsOneRound ? handsBySeat.get(row.id) : undefined;
            return (
              <div key={row.id} style={{ padding: '8px 0', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)', width: 16, fontVariantNumeric: 'tabular-nums' }}>{i + 1}</span>
                  <span style={{ fontWeight: 'var(--weight-medium)', flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: row.id === selfId ? 'var(--accent-gold)' : 'var(--text-primary)' }}>
                    {row.name}{row.id === selfId ? ' (you)' : ''}
                  </span>
                  {winner && row.id === winner.id && <Badge tone='success'>Winner</Badge>}
                  {row.score !== null && <span style={{ fontWeight: 'var(--weight-bold)', fontVariantNumeric: 'tabular-nums', minWidth: 28, textAlign: 'right' }}>{row.score}</span>}
                </div>
                {/* The hand that made that score, turned up (RULES.md 3C, cambia-1542). Indented
                    to sit under the name rather than the rank number. */}
                {hand && hand.length > 0 && (
                  <div data-testid={`final-hand-${row.id}`} style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginTop: 8, marginLeft: 26 }}>
                    {hand.map((card) => {
                      const face = toDsCardFace({ ...card, known: true });
                      if (!face) return null;
                      return (
                        <PlayingCard
                          key={card.id}
                          rank={face.rank}
                          suit={face.suit}
                          size='sm'
                          label={cardFaceName(face) ?? face.rank}
                        />
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
          {standings.length === 0 && (
            <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-tertiary)' }}>No results yet.</div>
          )}
        </div>
      </div>
      )}

      {!endedInError && isMatchEnd && ratingChanges && Object.keys(ratingChanges).length > 0 && (
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

      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', padding: '14px 20px', borderTop: '1px solid var(--border-subtle)', background: 'var(--surface-2)' }}>
        {canReturnToLobby ? (
          <Button variant='primary' autoFocus onClick={onReturnToLobby}>Back to lobby</Button>
        ) : (
          // post_game arms a results timer (returnToLobby fires on its own after
          // PostGameDuration); match_end arms none, so it names no timer that is not there
          // (cambia-1516).
          <span style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>
            {isMatchEnd
              ? 'Waiting for the host to return to the lobby.'
              : 'Waiting for the host. The results timer will return everyone to the lobby shortly.'}
          </span>
        )}
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
