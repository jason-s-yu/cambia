// src/components/lobby/DsLobbyView.tsx
// DS-styled pre-game lobby (cambia-484). Re-skins the legacy LobbyPage lobby UI
// (PlayerList / ReadyButton / HostControls / ChatWindow / settings / countdown)
// onto the design-system LobbyScreen, wired to useCurrentLobbyStore. Every
// outgoing WS message matches what the legacy components sent: `ready`/`unready`
// (ReadyButton), `start_game` (HostControls), `chat` (ChatWindow), `update_rules`
// (settings, via DsMatchSettings). No protocol change.
import React, { useEffect, useMemo, useState } from 'react';
import { useCurrentLobbyStore, type LobbyPhase } from '@/stores/lobbyStore';
import { useAuthStore } from '@/stores/authStore';
import Panel from '@/components/ds/chrome/Panel';
import Button from '@/components/ds/core/Button';
import Badge from '@/components/ds/core/Badge';
import PlayerSeat from '@/components/ds/game/PlayerSeat';
import DsMatchSettings from './DsMatchSettings';

interface DsLobbyViewProps {
  lobbyId: string;
  phase: LobbyPhase;
  sendMessage: (message: { type: string; body?: unknown }) => void;
  onLeave: () => void;
}

/** Live seconds remaining for the start countdown, or null when inactive. */
function useCountdownRemaining(): number | null {
  const startTime = useCurrentLobbyStore((s) => s.countdownStartTime);
  const duration = useCurrentLobbyStore((s) => s.countdownDuration);
  const [remaining, setRemaining] = useState<number | null>(null);

  useEffect(() => {
    if (!startTime || !duration || duration <= 0) {
      setRemaining(null);
      return;
    }
    const end = startTime + duration * 1000;
    const tick = () => setRemaining(Math.max(0, Math.ceil((end - Date.now()) / 1000)));
    tick();
    const id = window.setInterval(tick, 250);
    return () => window.clearInterval(id);
  }, [startTime, duration]);

  return remaining;
}

const DsLobbyView: React.FC<DsLobbyViewProps> = ({ lobbyId, phase, sendMessage, onLeave }) => {
  const lobbyDetails = useCurrentLobbyStore((s) => s.lobbyDetails);
  const chatMessages = useCurrentLobbyStore((s) => s.chatMessages);
  const selfId = useAuthStore((s) => s.user?.id);
  const selfName = useAuthStore((s) => s.user?.username) || 'You';
  const remaining = useCountdownRemaining();
  const [draft, setDraft] = useState('');
  const [copied, setCopied] = useState(false);

  const players = useMemo(() => lobbyDetails?.lobby_status?.users ?? [], [lobbyDetails]);
  const isHost = lobbyDetails?.your_is_host ?? false;
  const self = players.find((p) => p.id === selfId);
  const isReady = self?.is_ready ?? false;
  const allReady = players.length > 0 && players.every((p) => p.is_ready);
  const canStart = players.length >= 2 && allReady;
  const waiting = players.filter((p) => !p.is_ready).map((p) => p.username);
  const shortId = lobbyId.substring(0, 8);

  const toggleReady = () => sendMessage({ type: isReady ? 'unready' : 'ready' });
  const startGame = () => sendMessage({ type: 'start_game' });

  const sendChat = () => {
    const msg = draft.trim();
    if (!msg) return;
    sendMessage({ type: 'chat', body: { msg } });
    setDraft('');
  };

  const copyInvite = () => {
    const url = typeof window !== 'undefined' ? window.location.href : lobbyId;
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(url).then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }).catch(() => undefined);
    }
  };

  const banner = (() => {
    if (remaining !== null && remaining > 0) {
      return { tone: 'gold' as const, text: `Game starting in ${remaining}s…` };
    }
    if (phase === 'searching') return { tone: 'info' as const, text: 'Searching for a match…' };
    if (phase === 'ready_check') return { tone: 'gold' as const, text: 'Match found — ready up to begin.' };
    if (players.length < 2) return { tone: 'info' as const, text: 'Waiting for more players to join.' };
    if (allReady) return { tone: 'moss' as const, text: 'Everyone is ready.' };
    return { tone: 'gold' as const, text: `Waiting on ${waiting.join(', ')} — starting when everyone is ready.` };
  })();

  const bannerColors: Record<string, { bg: string; border: string; color: string }> = {
    gold: { bg: 'rgba(223,174,71,0.1)', border: 'var(--honey-600)', color: 'var(--honey-400)' },
    info: { bg: 'rgba(92,127,163,0.14)', border: 'var(--dusk-600)', color: 'var(--dusk-400)' },
    moss: { bg: 'rgba(79,138,94,0.16)', border: 'var(--moss-600)', color: 'var(--moss-400)' }
  };
  const bc = bannerColors[banner.tone];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 340px) minmax(0, 1fr) minmax(240px, 300px)', gap: 20, padding: 22, maxWidth: 1280, margin: '0 auto', width: '100%', alignItems: 'start' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: 'var(--ds-text-2xl)', fontWeight: 'var(--weight-regular)' }}>Lobby</h1>
            <Badge tone='info'>{lobbyDetails?.type ?? 'private'}</Badge>
          </div>
          <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>
            <span>code: {shortId}</span>
            <Button size='sm' variant='ghost' onClick={copyInvite}>{copied ? 'Copied' : 'Copy invite'}</Button>
          </div>
        </div>

        <Panel title={`Players · ${players.length}`}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {players.map((p) => (
              <div key={p.id} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <PlayerSeat username={p.username} isYou={p.id === selfId} state={p.is_ready ? 'ready' : undefined} style={{ flex: 1 }} />
                {p.is_host && <Badge tone='ember'>HOST</Badge>}
                {!p.is_ready && !p.is_host && <Badge>not ready</Badge>}
              </div>
            ))}
            {players.length === 0 && (
              <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-tertiary)' }}>No players yet.</div>
            )}
          </div>

          <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ padding: '8px 12px', borderRadius: 'var(--ds-radius-md)', background: bc.bg, border: `1.5px solid ${bc.border}`, fontSize: 'var(--ds-text-sm)', color: bc.color, fontWeight: 'var(--weight-bold)' }}>
              {banner.text}
            </div>
            {self && (
              <Button variant={isReady ? 'secondary' : 'primary'} fullWidth onClick={toggleReady}>
                {isReady ? 'Mark as not ready' : 'Mark as ready'}
              </Button>
            )}
            {isHost && (
              <Button variant='gold' fullWidth disabled={!canStart} onClick={startGame}>
                Start game
              </Button>
            )}
            <Button variant='ghost' fullWidth onClick={onLeave}>Leave lobby</Button>
          </div>
        </Panel>
      </div>

      {lobbyDetails ? (
        <DsMatchSettings currentSettings={lobbyDetails} isHost={isHost} sendMessage={sendMessage} />
      ) : (
        <Panel title='Match settings'>
          <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-tertiary)' }}>Loading settings…</div>
        </Panel>
      )}

      <Panel title='Lobby chat' style={{ display: 'flex', flexDirection: 'column', minHeight: 420 }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8, overflowY: 'auto' }}>
          {chatMessages.length === 0 && (
            <div style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-tertiary)' }}>No messages yet.</div>
          )}
          {chatMessages.map((c, i) => {
            const mine = c.user_id === selfId;
            return (
              <div key={`${c.user_id}-${c.ts}-${i}`} style={{ fontSize: 'var(--ds-text-sm)', lineHeight: 1.4 }}>
                <span style={{ fontWeight: 'var(--weight-black)', color: mine ? 'var(--ember-400)' : 'var(--dusk-400)' }}>{c.username}</span>
                <span style={{ color: 'var(--text-secondary)' }}> {c.msg}</span>
              </div>
            );
          })}
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') sendChat(); }}
            placeholder={`Message as ${selfName}…`}
            aria-label='Chat message input'
            style={{ flex: 1, height: 'var(--control-h-sm)', padding: '0 10px', fontFamily: 'var(--font-ui)', fontSize: 'var(--ds-text-sm)', color: 'var(--text-primary)', background: 'var(--surface-inset)', border: '1.5px solid var(--border-default)', borderRadius: 'var(--ds-radius-sm)', outline: 'none' }}
          />
          <Button size='sm' variant='secondary' onClick={sendChat}>Send</Button>
        </div>
      </Panel>
    </div>
  );
};

export default DsLobbyView;
