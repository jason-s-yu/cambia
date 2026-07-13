import React from 'react';
import { useNavigate } from 'react-router-dom';
import Button from '@/components/ds/core/Button';
import Badge from '@/components/ds/core/Badge';
import QueueCard from '@/components/ds/data/QueueCard';
import TierBadge from '@/components/ds/data/TierBadge';
import StatRow from '@/components/ds/data/StatRow';
import Panel from '@/components/ds/chrome/Panel';
import { useAuthStore } from '@/stores/authStore';

interface PublicLobby {
  name: string;
  mode: string;
  players: string;
  jokers: boolean;
}

interface Friend {
  name: string;
  online: boolean;
  note: string;
}

const PUBLIC_LOBBIES: PublicLobby[] = [
  { name: 'Friday night circuit', mode: 'FFA-4 · casual', players: '3/4', jokers: true },
  { name: 'no jokers, fight me', mode: 'H2H · casual', players: '1/2', jokers: false },
  { name: 'learning lobby — be nice', mode: 'FFA-6 · casual', players: '4/6', jokers: true }
];

const FRIENDS: Friend[] = [
  { name: 'Maple', online: true, note: 'In queue · FFA-4' },
  { name: 'Bram', online: true, note: 'In match · round 5/8' },
  { name: 'Sorrel', online: false, note: 'Last seen 2h ago' }
];

/**
 * DS preview: matchmaking home (cambia-438). Queue cards, public lobbies and
 * friends are typed mock data (no store exposes queue defs, lobby display
 * names or a friends list). The ratings panel reads the authenticated user's
 * Glicko/OpenSkill fields where present, falling back to the kit's sample.
 */
const HomeScreen: React.FC = () => {
  const navigate = useNavigate();
  const authUser = useAuthStore((state) => state.user);

  const play = () => navigate('/ds/lobby');
  const createLobby = () => navigate('/ds/lobby');

  const ratingValue = authUser?.elo !== undefined ? `${Math.round(authUser.elo)}` : '1520';
  const ratingBand = authUser?.rd !== undefined ? `± ${Math.round(authUser.rd)}` : '± 140';
  const ffaValue = authUser?.open_skill_mu !== undefined ? authUser.open_skill_mu.toFixed(1) : 'Placement';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 20, padding: 22, maxWidth: 1240, margin: '0 auto', width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20, minWidth: 0 }}>
        <div>
          <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: 'var(--ds-text-3xl)', fontWeight: 'var(--weight-regular)', lineHeight: 'var(--ds-leading-tight)' }}>Lowest score wins.</h1>
          <p style={{ margin: '6px 0 0', color: 'var(--text-secondary)', fontSize: 'var(--text-md)' }}>Pick a queue — the forest remembers your discards.</p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 14 }}>
          <QueueCard name='H2H Rapid' tagline='Two players, eight rounds' players={2} rounds={8} minutes={40} pool='Glicko-2' primary onPlay={play} />
          <QueueCard name='FFA-4 Standard' tagline='Four players, eight rounds' players={4} rounds={8} minutes={55} pool='OpenSkill' primary onPlay={play} />
          <QueueCard name='Quick Play' tagline='One game, no rank shown' players={2} rounds={1} minutes={5} ranked={false} onPlay={play} />
          <QueueCard name='H2H Blitz' tagline='Short and sharp' players={2} rounds={4} minutes={20} pool='Glicko-2' onPlay={play} />
          <QueueCard name='H2H Classical' tagline='Full replication depth' players={2} rounds={16} minutes={80} pool='Glicko-2' onPlay={play} />
          <QueueCard name='FFA-4 Classical' tagline='Twelve rounds of nerve' players={4} rounds={12} minutes={85} pool='OpenSkill' onPlay={play} />
        </div>
        <Panel title='Public lobbies' action={<Button size='sm' onClick={createLobby}>Create lobby</Button>}>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {PUBLIC_LOBBIES.map((l, i) => (
              <div key={l.name} style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '10px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                <span style={{ fontWeight: 'var(--weight-bold)', flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{l.name}</span>
                <Badge>{l.mode}</Badge>
                {!l.jokers && <Badge tone='info'>no jokers</Badge>}
                <span style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-xs)', color: 'var(--text-secondary)', width: 34, textAlign: 'right' }}>{l.players}</span>
                <Button size='sm' variant='secondary' onClick={createLobby}>Join</Button>
              </div>
            ))}
          </div>
        </Panel>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        <Panel title='Your ratings'>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <TierBadge tier='gold' />
            <span style={{ fontFamily: 'var(--ds-font-mono)', fontWeight: 'var(--weight-bold)', fontSize: 'var(--ds-text-lg)' }}>{ratingValue} <span style={{ color: 'var(--text-tertiary)', fontSize: 'var(--ds-text-xs)' }}>{ratingBand}</span></span>
          </div>
          <StatRow label='H2H Ranked · Glicko-2' value={ratingValue} delta='+12' />
          <StatRow label='FFA-4 Ranked · OpenSkill' value={ffaValue} unit='7/10 games' />
          <StatRow label='Season peak' value='Gold II' />
          <p style={{ margin: '10px 0 0', fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)' }}>Season 4 ends in 23 days. Peak tier is kept as a badge.</p>
        </Panel>
        <Panel title='Friends'>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {FRIENDS.map((f, i) => (
              <div key={f.name} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', flex: 'none', background: f.online ? 'var(--moss-500)' : 'var(--border-strong)' }}></span>
                <span style={{ lineHeight: 1.2, flex: 1 }}>
                  <span style={{ display: 'block', fontWeight: 'var(--weight-bold)', fontSize: 'var(--ds-text-sm)' }}>{f.name}</span>
                  <span style={{ display: 'block', fontSize: 'var(--text-2xs)', color: 'var(--text-tertiary)' }}>{f.note}</span>
                </span>
                {f.online && <Button size='sm' variant='ghost'>Invite</Button>}
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
};

export default HomeScreen;
