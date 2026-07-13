const { Button, Badge, QueueCard, TierBadge, StatRow, PlayerSeat } = window.Cambia_cc8727;

const PUBLIC_LOBBIES = [
  { name: 'Friday night circuit', mode: 'FFA-4 · casual', players: '3/4', jokers: true },
  { name: 'no jokers, fight me', mode: 'H2H · casual', players: '1/2', jokers: false },
  { name: 'learning lobby — be nice', mode: 'FFA-6 · casual', players: '4/6', jokers: true },
];

const FRIENDS = [
  { name: 'Maple', online: true, note: 'In queue · FFA-4' },
  { name: 'Bram', online: true, note: 'In match · round 5/8' },
  { name: 'Sorrel', online: false, note: 'Last seen 2h ago' },
];

function HomeScreen({ onPlay, onCreateLobby }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 20, padding: 22, maxWidth: 1240, margin: '0 auto', width: '100%' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20, minWidth: 0 }}>
        <div>
          <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: 'var(--text-3xl)', fontWeight: 400, lineHeight: 'var(--leading-tight)' }}>Lowest score wins.</h1>
          <p style={{ margin: '6px 0 0', color: 'var(--text-secondary)', fontSize: 'var(--text-md)' }}>Pick a queue — the forest remembers your discards.</p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 14 }}>
          <QueueCard name="H2H Rapid" tagline="Two players, eight rounds" players={2} rounds={8} minutes={40} pool="Glicko-2" primary onPlay={onPlay} />
          <QueueCard name="FFA-4 Standard" tagline="Four players, eight rounds" players={4} rounds={8} minutes={55} pool="OpenSkill" primary onPlay={onPlay} />
          <QueueCard name="Quick Play" tagline="One game, no rank shown" players={2} rounds={1} minutes={5} ranked={false} onPlay={onPlay} />
          <QueueCard name="H2H Blitz" tagline="Short and sharp" players={2} rounds={4} minutes={20} pool="Glicko-2" onPlay={onPlay} />
          <QueueCard name="H2H Classical" tagline="Full replication depth" players={2} rounds={16} minutes={80} pool="Glicko-2" onPlay={onPlay} />
          <QueueCard name="FFA-4 Classical" tagline="Twelve rounds of nerve" players={4} rounds={12} minutes={85} pool="OpenSkill" onPlay={onPlay} />
        </div>
        <window.Panel title="Public lobbies" action={<Button size="sm" onClick={onCreateLobby}>Create lobby</Button>}>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {PUBLIC_LOBBIES.map((l, i) => (
              <div key={l.name} style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '10px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                <span style={{ fontWeight: 700, flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{l.name}</span>
                <Badge>{l.mode}</Badge>
                {!l.jokers && <Badge tone="info">no jokers</Badge>}
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--text-secondary)', width: 34, textAlign: 'right' }}>{l.players}</span>
                <Button size="sm" variant="secondary" onClick={onCreateLobby}>Join</Button>
              </div>
            ))}
          </div>
        </window.Panel>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        <window.Panel title="Your ratings">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <TierBadge tier="gold" />
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 'var(--text-lg)' }}>1520 <span style={{ color: 'var(--text-tertiary)', fontSize: 'var(--text-xs)' }}>± 140</span></span>
          </div>
          <StatRow label="H2H Ranked · Glicko-2" value="1520" delta="+12" />
          <StatRow label="FFA-4 Ranked · OpenSkill" value="Placement" unit="7/10 games" />
          <StatRow label="Season peak" value="Gold II" />
          <p style={{ margin: '10px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)' }}>Season 4 ends in 23 days. Peak tier is kept as a badge.</p>
        </window.Panel>
        <window.Panel title="Friends">
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {FRIENDS.map((f, i) => (
              <div key={f.name} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '9px 2px', borderTop: i ? '1px solid var(--border-subtle)' : 'none' }}>
                <span style={{ width: 8, height: 8, borderRadius: '50%', flex: 'none', background: f.online ? 'var(--moss-500)' : 'var(--border-strong)' }}></span>
                <span style={{ lineHeight: 1.2, flex: 1 }}>
                  <span style={{ display: 'block', fontWeight: 700, fontSize: 'var(--text-sm)' }}>{f.name}</span>
                  <span style={{ display: 'block', fontSize: 'var(--text-2xs)', color: 'var(--text-tertiary)' }}>{f.note}</span>
                </span>
                {f.online && <Button size="sm" variant="ghost">Invite</Button>}
              </div>
            ))}
          </div>
        </window.Panel>
      </div>
    </div>
  );
}

window.HomeScreen = HomeScreen;
