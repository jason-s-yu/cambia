const { Badge, TierBadge } = window.Cambia_cc8727;

const H2H_ROWS = [
  { rank: 1, name: 'Thistledown', tier: 'grandmaster', rating: '2141 ± 62', games: 412, peak: 'grandmaster' },
  { rank: 2, name: 'OldTomBarrow', tier: 'grandmaster', rating: '2098 ± 71', games: 287, peak: 'grandmaster' },
  { rank: 3, name: 'redking_enjoyer', tier: 'master', rating: '1993 ± 58', games: 530, peak: 'grandmaster' },
  { rank: 4, name: 'Maple', tier: 'master', rating: '1961 ± 66', games: 198, peak: 'master' },
  { rank: 5, name: 'quietsnap', tier: 'master', rating: '1904 ± 90', games: 121, peak: 'master' },
  { rank: 6, name: 'Bram', tier: 'diamond', rating: '1842 ± 74', games: 265, peak: 'master' },
  { rank: 7, name: 'FennelTheBold', tier: 'diamond', rating: '1815 ± 81', games: 176, peak: 'diamond' },
  { rank: 8, name: 'sevens_and_eights', tier: 'diamond', rating: '1788 ± 69', games: 340, peak: 'diamond' },
];
const YOU_ROW = { rank: 1042, name: 'Juniper', tier: 'gold', rating: '1520 ± 140', games: 47, peak: 'gold' };

function LbRow({ r, you }) {
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: '56px 1fr 150px 130px 70px 130px', alignItems: 'center',
      gap: 10, padding: '9px 14px',
      background: you ? 'rgba(223,174,71,0.08)' : 'transparent',
      borderTop: '1px solid var(--border-subtle)',
      border: you ? '1.5px solid var(--honey-600)' : undefined,
      borderRadius: you ? 'var(--radius-md)' : 0,
    }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: r.rank <= 3 ? 'var(--honey-400)' : 'var(--text-tertiary)' }}>#{r.rank}</span>
      <span style={{ display: 'flex', alignItems: 'center', gap: 9, fontWeight: 700, minWidth: 0 }}>
        <span style={{ width: 24, height: 24, flex: 'none', borderRadius: '50%', background: 'var(--surface-inset)', border: '1.5px solid var(--outline-ink)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, color: 'var(--text-primary)' }}>{r.name[0].toUpperCase()}</span>
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.name}{you ? ' (you)' : ''}</span>
      </span>
      <TierBadge tier={r.tier} size="sm" />
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-sm)' }}>{r.rating}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>{r.games}</span>
      <TierBadge tier={r.peak} size="sm" />
    </div>
  );
}

function LeaderboardScreen() {
  const [pool, setPool] = React.useState('h2h');
  return (
    <div style={{ padding: 22, maxWidth: 1000, margin: '0 auto', width: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: 'var(--text-3xl)', fontWeight: 400 }}>Leaderboard</h1>
          <p style={{ margin: '4px 0 0', color: 'var(--text-secondary)' }}>Season 4 · recalibrated monthly · percentile tiers</p>
        </div>
        <div style={{ display: 'flex', gap: 6, background: 'var(--surface-inset)', border: '1.5px solid var(--border-default)', borderRadius: 'var(--radius-pill)', padding: 4 }}>
          {[['h2h', 'H2H Ranked · Glicko-2'], ['ffa4', 'FFA-4 Ranked · OpenSkill']].map(([id, label]) => (
            <button key={id} onClick={() => setPool(id)} style={{
              padding: '6px 16px', borderRadius: 'var(--radius-pill)', cursor: 'pointer',
              fontFamily: 'var(--font-ui)', fontWeight: 700, fontSize: 'var(--text-sm)', whiteSpace: 'nowrap',
              background: pool === id ? 'var(--ember-500)' : 'transparent',
              color: pool === id ? 'var(--text-on-ember)' : 'var(--text-secondary)',
              border: pool === id ? '1.5px solid var(--outline-ink)' : '1.5px solid transparent',
            }}>{label}</button>
          ))}
        </div>
      </div>
      <window.Panel style={{ marginTop: 18, padding: '6px 4px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '56px 1fr 150px 130px 70px 130px', gap: 10, padding: '8px 14px', fontSize: 'var(--text-2xs)', fontWeight: 800, letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)' }}>
          <span>Rank</span><span>Player</span><span>Tier</span><span>Rating</span><span>Games</span><span>Peak</span>
        </div>
        {H2H_ROWS.map((r) => <LbRow key={r.rank} r={r} />)}
        <div style={{ padding: '10px 14px', textAlign: 'center', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)' }}>···</div>
        <LbRow r={YOU_ROW} you />
      </window.Panel>
      {pool === 'ffa4' && (
        <p style={{ marginTop: 10, fontSize: 'var(--text-xs)', color: 'var(--text-tertiary)' }}>FFA-4 shows OpenSkill μ (25.0 start, β = 8.0). Sample data mirrors the H2H board in this kit.</p>
      )}
    </div>
  );
}

window.LeaderboardScreen = LeaderboardScreen;
