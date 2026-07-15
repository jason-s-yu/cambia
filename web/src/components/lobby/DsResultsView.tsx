// src/components/lobby/DsResultsView.tsx
// DS-styled end-of-game results (cambia-484) for the post_game (casual single
// game) and match_end (ranked circuit) phases. Reads standings from
// matchState/lobbyDetails and preserves the legacy return-to-lobby behaviour
// (LobbyPage set phase back to 'open'). No WS message is sent from here.
// Casual (post_game) standings have no matchState (that's ranked-only, see
// hub.buildLobbySnapshot), so final scores fall back to gameStore's finalScores,
// captured off the game_end event (cambia-510).
import React, { useMemo } from 'react';
import { useCurrentLobbyStore, type LobbyPhase } from '@/stores/lobbyStore';
import { useAuthStore } from '@/stores/authStore';
import { useGameStore, selectFinalScores } from '@/stores/gameStore';
import Panel from '@/components/ds/chrome/Panel';
import Button from '@/components/ds/core/Button';
import Badge from '@/components/ds/core/Badge';
import ScorePill from '@/components/ds/game/ScorePill';

interface DsResultsViewProps {
  phase: LobbyPhase;
  onReturnToLobby: () => void;
  onLeave: () => void;
}

const DsResultsView: React.FC<DsResultsViewProps> = ({ phase, onReturnToLobby, onLeave }) => {
  const matchState = useCurrentLobbyStore((s) => s.matchState);
  const lobbyPlayers = useCurrentLobbyStore((s) => s.lobbyDetails?.lobby_status?.users);
  const selfId = useAuthStore((s) => s.user?.id);
  const finalScores = useGameStore(selectFinalScores);

  const isMatchEnd = phase === 'match_end';
  const title = isMatchEnd ? 'Final standings' : 'Game over';

  const names = useMemo(() => {
    const m = new Map<string, string>();
    (lobbyPlayers ?? []).forEach((u) => m.set(u.id, u.username));
    return m;
  }, [lobbyPlayers]);

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

  return (
    <div style={{ maxWidth: 640, margin: '0 auto', padding: 22, display: 'flex', flexDirection: 'column', gap: 18 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: 'var(--ds-text-2xl)', fontWeight: 'var(--weight-regular)' }}>{title}</h1>
        {isMatchEnd && matchState?.isRanked && <Badge tone='gold'>ranked</Badge>}
        {matchState && <Badge tone='info' mono>{matchState.currentRound}/{matchState.totalRounds}</Badge>}
      </div>

      <Panel title={isMatchEnd ? 'Standings · lower is better' : 'Players'}>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {standings.map((row, i) => (
            <div key={row.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '9px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
              <span style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)', width: 16 }}>{i + 1}</span>
              <span style={{ fontWeight: 'var(--weight-bold)', flex: 1, color: row.id === selfId ? 'var(--honey-400)' : 'var(--text-primary)' }}>
                {row.name}{row.id === selfId ? ' (you)' : ''}
              </span>
              {i === 0 && row.score !== null && <Badge tone='success'>winner</Badge>}
              {row.score !== null && <span style={{ fontFamily: 'var(--ds-font-mono)', fontWeight: 'var(--weight-bold)' }}>{row.score}</span>}
            </div>
          ))}
          {standings.length === 0 && (
            <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-tertiary)' }}>No results available.</div>
          )}
        </div>
      </Panel>

      {isMatchEnd && ratingChanges && Object.keys(ratingChanges).length > 0 && (
        <Panel title='Rating changes'>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {Object.entries(ratingChanges).map(([id, change]) => {
              const diff = Math.round(change.after - change.before);
              const sign = diff >= 0 ? '+' : '';
              return (
                <div key={id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, fontSize: 'var(--ds-text-sm)' }}>
                  <span style={{ fontWeight: 'var(--weight-bold)' }}>{names.get(id) ?? id.substring(0, 6)}</span>
                  <ScorePill value={`${Math.round(change.before)} -> ${Math.round(change.after)} (${sign}${diff})`} tone={diff >= 0 ? 'moss' : 'berry'} />
                </div>
              );
            })}
          </div>
        </Panel>
      )}

      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <Button variant='primary' onClick={onReturnToLobby}>Back to lobby</Button>
        <Button variant='ghost' onClick={onLeave}>Leave lobby</Button>
      </div>
    </div>
  );
};

export default DsResultsView;
