import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '@/components/ds/core/Button';
import Badge from '@/components/ds/core/Badge';
import Input from '@/components/ds/core/Input';
import Select from '@/components/ds/core/Select';
import Checkbox from '@/components/ds/core/Checkbox';
import Switch from '@/components/ds/core/Switch';
import PlayerSeat from '@/components/ds/game/PlayerSeat';
import Panel from '@/components/ds/chrome/Panel';
import { useAuthStore } from '@/stores/authStore';

interface Seat {
  name: string;
  you?: boolean;
  ready?: boolean;
  host?: boolean;
}

interface ChatLine {
  who: string;
  msg: string;
}

const SEATS: Array<Seat | null> = [
  { name: 'Juniper', you: true, ready: true, host: true },
  { name: 'Maple', ready: true },
  { name: 'Bram', ready: false },
  null
];

const INITIAL_CHAT: ChatLine[] = [
  { who: 'Maple', msg: 'jokers on?' },
  { who: 'Juniper', msg: 'always. full 54.' },
  { who: 'Bram', msg: 'one sec, grabbing tea' }
];

/**
 * DS preview: pre-game lobby (cambia-438). Seats, chat and settings are typed
 * mock data. Wiring to useCurrentLobbyStore/HouseRules needs a live hub
 * WebSocket session, absent in a standalone preview; the "you" identity and
 * outgoing chat sender read the authenticated username where present.
 */
const LobbyScreen: React.FC = () => {
  const navigate = useNavigate();
  const authUser = useAuthStore((state) => state.user);
  const me = authUser?.username || 'Juniper';
  const [chat, setChat] = useState<ChatLine[]>(INITIAL_CHAT);
  const [draft, setDraft] = useState('');

  const send = () => {
    if (!draft.trim()) return;
    setChat([...chat, { who: me, msg: draft.trim() }]);
    setDraft('');
  };

  const start = () => navigate('/ds/game');
  const leave = () => navigate('/ds/home');

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 340px) minmax(0, 1fr) minmax(240px, 300px)', gap: 20, padding: 22, maxWidth: 1280, margin: '0 auto', width: '100%', alignItems: 'start' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: 'var(--ds-text-2xl)', fontWeight: 'var(--weight-regular)' }}>Friday night circuit</h1>
            <Badge tone='info'>private</Badge>
          </div>
          <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-sm)', color: 'var(--text-secondary)' }}>
            <span>code: FNC-4821</span>
            <Button size='sm' variant='ghost'>Copy</Button>
          </div>
        </div>
        <Panel title='Players · 3/4'>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {SEATS.map((s, i) => s ? (
              <div key={s.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <PlayerSeat username={s.name} isYou={s.you} state={s.ready ? 'ready' : undefined} style={{ flex: 1 }} />
                {s.host && <Badge tone='ember'>HOST</Badge>}
                {!s.ready && !s.host && <Badge>not ready</Badge>}
              </div>
            ) : (
              <button
                key={'empty' + i}
                style={{
                  height: 46,
                  borderRadius: 'var(--radius-pill)',
                  cursor: 'pointer',
                  border: '2px dashed var(--border-strong)',
                  background: 'transparent',
                  color: 'var(--text-tertiary)',
                  fontFamily: 'var(--font-ui)',
                  fontWeight: 'var(--weight-bold)',
                  fontSize: 'var(--ds-text-sm)'
                }}
              >+ Invite a friend</button>
            ))}
          </div>
          <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ padding: '8px 12px', borderRadius: 'var(--ds-radius-md)', background: 'rgba(223,174,71,0.1)', border: '1.5px solid var(--honey-600)', fontSize: 'var(--ds-text-sm)', color: 'var(--honey-400)', fontWeight: 'var(--weight-bold)' }}>
              Waiting on Bram — starting when everyone is ready.
            </div>
            <Button variant='gold' fullWidth onClick={start}>Start anyway (host)</Button>
            <Button variant='ghost' fullWidth onClick={leave}>Leave lobby</Button>
          </div>
        </Panel>
      </div>

      <Panel title='Match settings · host only'>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: 14, marginBottom: 16 }}>
          <Select label='Game mode' defaultValue='circuit_4p' options={[
            { value: 'head_to_head', label: 'Head to head' },
            { value: 'group', label: 'Free-for-all' },
            { value: 'circuit_4p', label: 'Circuit · 4 players' }
          ]} />
          <Select label='Rounds' defaultValue='8' options={[
            { value: '8', label: '8 — Quick (~55 min)' },
            { value: '12', label: '12 — Standard (~85 min)' },
            { value: '20', label: '20 — Championship (~2.5 h)' }
          ]} />
          <Input label='Turn timer (sec)' mono defaultValue='30' />
          <Input label='Penalty draw count' mono defaultValue='2' />
        </div>
        <div style={{ fontSize: 'var(--text-2xs)', fontWeight: 'var(--weight-black)', letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)', margin: '2px 0 10px' }}>House rules</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px 18px' }}>
          <Checkbox defaultChecked label='Draw from discard pile' description='allowDrawFromDiscardPile — top card is public information' />
          <Checkbox defaultChecked label='Replace abilities' description='allowReplaceAbilities — keep ability cards for later payoffs' />
          <Checkbox defaultChecked label='Snap race' description='snapRace — only the first snap wins' />
          <Checkbox label='Lock caller hand' description="lockCallerHand — protect the Cambia caller's hand" />
          <Checkbox defaultChecked label='Use jokers' description='useJokers — full 54-card deck' />
          <Checkbox label='Forfeit on disconnect' description='forfeitOnDisconnect — otherwise AI plays defensively' />
        </div>
        <div style={{ borderTop: '1.5px solid var(--border-subtle)', marginTop: 16, paddingTop: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
            <span style={{ fontSize: 'var(--text-2xs)', fontWeight: 'var(--weight-black)', letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)' }}>Circuit scoring</span>
            <Switch defaultChecked label='Enabled' />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 14 }}>
            <Input label='Target score' mono defaultValue='100' />
            <Input label='Win bonus' mono defaultValue='−5 / −2' />
            <Input label='False Cambia penalty' mono defaultValue='+5' />
          </div>
          <p style={{ margin: '10px 0 0', fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)' }}>Ranked circuits enforce: draw from discard ✓, replace abilities ✓, lock caller hand ✗.</p>
        </div>
      </Panel>

      <Panel title='Lobby chat' style={{ display: 'flex', flexDirection: 'column', minHeight: 420 }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8, overflowY: 'auto' }}>
          {chat.map((c, i) => (
            <div key={i} style={{ fontSize: 'var(--ds-text-sm)', lineHeight: 1.4 }}>
              <span style={{ fontWeight: 'var(--weight-black)', color: c.who === me ? 'var(--ember-400)' : 'var(--dusk-400)' }}>{c.who}</span>
              <span style={{ color: 'var(--text-secondary)' }}> {c.msg}</span>
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') send(); }}
            placeholder='Say something…'
            style={{ flex: 1, height: 'var(--control-h-sm)', padding: '0 10px', fontFamily: 'var(--font-ui)', fontSize: 'var(--ds-text-sm)', color: 'var(--text-primary)', background: 'var(--surface-inset)', border: '1.5px solid var(--border-default)', borderRadius: 'var(--ds-radius-sm)', outline: 'none' }}
          />
          <Button size='sm' variant='secondary' onClick={send}>Send</Button>
        </div>
      </Panel>
    </div>
  );
};

export default LobbyScreen;
