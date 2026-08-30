// src/components/lobby/DsLobbyView.tsx
// DS-styled pre-game lobby (cambia-484), swept onto the flat card-room language
// in cambia-847. Re-skins the legacy LobbyPage lobby UI (PlayerList /
// ReadyButton / HostControls / ChatWindow / settings / countdown) onto the
// design-system primitives, wired to useCurrentLobbyStore. Every outgoing WS
// message matches what the legacy components sent: `ready`/`unready`
// (ReadyButton), `start_game` (HostControls), `chat` (ChatWindow),
// `update_rules` (settings, via DsMatchSettings). No protocol change.
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useCurrentLobbyStore, type LobbyPhase } from '@/stores/lobbyStore';
import { useAuthStore } from '@/stores/authStore';
import Panel from '@/components/ds/chrome/Panel';
import { EYEBROW } from '@/components/ds/eyebrow';
import Button from '@/components/ds/core/Button';
import Badge from '@/components/ds/core/Badge';
import Input from '@/components/ds/core/Input';
import PlayerSeat from '@/components/ds/game/PlayerSeat';
import DsMatchSettings from './DsMatchSettings';
import DsLobbyStatus, { type LobbyStatusTone } from './DsLobbyStatus';
import { humanizeId } from '@/utils/gameMode';

interface DsLobbyViewProps {
  lobbyId: string;
  phase: LobbyPhase;
  sendMessage: (message: { type: string; body?: unknown }) => void;
  onLeave: () => void;
}

interface LobbyStatusSpec {
  tone: LobbyStatusTone;
  text: string;
  value?: string;
}

const MUTED: React.CSSProperties = {
  fontSize: 'var(--ds-text-sm)',
  color: 'var(--text-tertiary)'
};

const TYPE_LABELS: Record<string, string> = {
  private: 'Private',
  public: 'Public',
  matchmaking: 'Matchmaking'
};

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

/** How long the dropped-frame notice stays up. Matches the table's notice line. */
const DROP_NOTICE_MS = 4500;

/**
 * True while a lobby frame the hub discarded and the client could not resend is worth telling
 * the player about (cambia-913 F4). useSocket counts them; the copy lives here, as it does on
 * the table: a ready or a chat line that never landed leaves a lobby that simply does not react.
 */
function useDroppedNotice(): boolean {
  const nonce = useCurrentLobbyStore((s) => s.droppedActionNonce);
  const seen = useRef(nonce);
  const [showing, setShowing] = useState(false);

  useEffect(() => {
    if (nonce === seen.current) return;
    seen.current = nonce;
    setShowing(true);
    const t = window.setTimeout(() => setShowing(false), DROP_NOTICE_MS);
    return () => window.clearTimeout(t);
  }, [nonce]);

  return showing;
}

const DsLobbyView: React.FC<DsLobbyViewProps> = ({ lobbyId, phase, sendMessage, onLeave }) => {
  const lobbyDetails = useCurrentLobbyStore((s) => s.lobbyDetails);
  const chatMessages = useCurrentLobbyStore((s) => s.chatMessages);
  const selfId = useAuthStore((s) => s.user?.id);
  const remaining = useCountdownRemaining();
  const dropped = useDroppedNotice();
  const [draft, setDraft] = useState('');
  const [copied, setCopied] = useState(false);
  const chatListRef = useRef<HTMLDivElement>(null);

  const players = useMemo(() => lobbyDetails?.lobby_status?.users ?? [], [lobbyDetails]);
  // A matchmade lobby is run by its queue: nobody holds the host role, the rules are the
  // queue's, and the ready check starts the game (cambia-1087). The service already reports
  // your_is_host false and marks no seat is_host for one, so this only keeps a stale snapshot
  // from leaving a Host badge or a Start game button on screen over a service that refuses
  // both. A party that is still searching is not system-hosted: its leader keeps the search.
  const systemHost = lobbyDetails?.system_host ?? false;
  const isHost = (lobbyDetails?.your_is_host ?? false) && !systemHost;
  const self = players.find((p) => p.id === selfId);
  const isReady = self?.is_ready ?? false;
  const allReady = players.length > 0 && players.every((p) => p.is_ready);
  const canStart = players.length >= 2 && allReady;
  const waiting = players.filter((p) => !p.is_ready).map((p) => p.username);
  const shortId = lobbyId.substring(0, 8);
  const lobbyType = lobbyDetails?.type ?? 'private';
  // A lobby type the map does not carry is title-cased rather than printed raw: the wire value is
  // an id, and the badge used to put a bare `matchmaking_ranked`-shaped string on screen the way
  // the pool ids of cambia-1086 did (cambia-1099 K1). Empty renders no badge rather than an empty
  // one, since there is nothing to name.
  const typeLabel = TYPE_LABELS[lobbyType] ?? humanizeId(lobbyType);

  // The chat list is height-bounded, so each new line pins the scroll to the newest
  // message; otherwise anything past the fold would land unseen.
  useEffect(() => {
    const el = chatListRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [chatMessages.length]);

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

  // With auto-start on, the server begins the countdown the moment every seat is
  // ready (lobby.MarkUserReadyUnsafe returns true on allReady && AutoStart), so
  // nobody is waiting on the host (cambia-876, DL-3 review F8).
  const autoStart = !!(lobbyDetails?.lobbySettings ?? lobbyDetails?.settings)?.autoStart;

  const status: LobbyStatusSpec = (() => {
    // 'Starting' carries the announcement; the seconds are the aria-hidden
    // readout, so the countdown does not narrate itself once a second.
    if (remaining !== null && remaining > 0) {
      return { tone: 'gold', text: 'Starting', value: `${remaining}s` };
    }
    if (phase === 'searching') return { tone: 'info', text: 'Searching for a match' };
    if (phase === 'ready_check') return { tone: 'gold', text: 'Match found. Ready up to begin.' };
    if (players.length < 2) return { tone: 'info', text: 'Waiting for players. Share the invite link.' };
    if (allReady) return { tone: 'success', text: isHost || autoStart || systemHost ? 'All players ready.' : 'All players ready. Waiting on the host.' };
    return { tone: 'info', text: `Waiting on ${waiting.join(', ')}.` };
  })();

  return (
    <div className='grid w-full max-w-[1280px] mx-auto items-start gap-5 p-4 md:p-6 grid-cols-1 lg:grid-cols-[minmax(300px,380px)_minmax(0,1fr)_minmax(240px,280px)]'>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)', minWidth: 0 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <h1
              style={{
                margin: 0,
                fontSize: 'var(--ds-text-2xl)',
                fontWeight: 'var(--weight-bold)',
                letterSpacing: 'var(--ds-tracking-tight)',
                lineHeight: 'var(--ds-leading-tight)',
                color: 'var(--text-primary)'
              }}
            >
              Lobby
            </h1>
            {typeLabel && <Badge tone='neutral'>{typeLabel}</Badge>}
          </div>
          <div style={{ marginTop: 'var(--space-2)', display: 'flex', alignItems: 'center', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
            {/* Not 'Invite code': nothing takes a code. Joining goes through the
                dashboard list or the copied link, so the id is labelled as what it
                is (cambia-876, DL-3 review F10). */}
            <span style={EYEBROW}>Lobby id</span>
            <Badge mono>{shortId}</Badge>
            <Button size='sm' variant='ghost' onClick={copyInvite}>{copied ? 'Link copied' : 'Copy link'}</Button>
          </div>
        </div>

        <Panel title='Players' action={<Badge tone='neutral'>{players.length} seated</Badge>}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
            {players.map((p) => (
              <div key={p.id} style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                {/* Two cells, one shape per row: the seat takes the free width and truncates a
                    long name inside the pill; the badge group keeps its natural width. */}
                <PlayerSeat
                  username={p.username}
                  isYou={p.id === selfId}
                  state={p.is_ready ? 'ready' : undefined}
                  style={{ flex: '1 1 0', minWidth: 0 }}
                />
                {((p.is_host && !systemHost) || !p.is_ready) && (
                  <div style={{ display: 'flex', gap: 'var(--space-2)', flex: 'none' }}>
                    {p.is_host && !systemHost && <Badge tone='gold'>Host</Badge>}
                    {!p.is_ready && <Badge tone='neutral'>Not ready</Badge>}
                  </div>
                )}
              </div>
            ))}
            {players.length === 0 && <div style={MUTED}>No one seated yet.</div>}
          </div>

          <div style={{ marginTop: 'var(--space-4)', display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
            <DsLobbyStatus tone={status.tone} text={status.text} value={status.value} />
            {dropped && <DsLobbyStatus tone='warning' text='That did not go through.' />}
            {self && (
              <Button variant={isReady ? 'secondary' : 'primary'} fullWidth onClick={toggleReady}>
                {isReady ? 'Unready' : 'Ready up'}
              </Button>
            )}
            {isHost && (
              <Button variant='primary' fullWidth disabled={!canStart} onClick={startGame}>
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
        <Panel title='Rule sheet'>
          <div style={MUTED}>Loading rules</div>
        </Panel>
      )}

      <Panel title='Lobby chat' style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <div ref={chatListRef} style={{ display: 'flex', flexDirection: 'column', gap: 6, minHeight: 240, maxHeight: 360, overflowY: 'auto' }}>
          {chatMessages.length === 0 && <div style={MUTED}>No messages yet.</div>}
          {chatMessages.map((c, i) => {
            const mine = c.user_id === selfId;
            return (
              <div key={`${c.user_id}-${c.ts}-${i}`} style={{ fontSize: 'var(--ds-text-sm)', lineHeight: 'var(--ds-leading-snug)', overflowWrap: 'anywhere' }}>
                <span style={{ fontWeight: 'var(--weight-bold)', color: mine ? 'var(--accent-gold-text)' : 'var(--text-primary)' }}>{c.username}</span>
                <span style={{ color: 'var(--text-secondary)' }}> {c.msg}</span>
              </div>
            );
          })}
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendChat();
          }}
          style={{ display: 'flex', gap: 'var(--space-2)', marginTop: 'var(--space-3)' }}
        >
          <Input
            aria-label='Chat message'
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder='Message the lobby'
            style={{ flex: 1, minWidth: 0 }}
          />
          <Button variant='secondary'>Send</Button>
        </form>
      </Panel>
    </div>
  );
};

export default DsLobbyView;
