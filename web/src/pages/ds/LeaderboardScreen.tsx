import React, { useState } from 'react';
import TierBadge, { type Tier } from '@/components/ds/data/TierBadge';
import Panel from '@/components/ds/chrome/Panel';

interface LbRowData {
  rank: number;
  name: string;
  tier: Tier;
  rating: string;
  games: number;
  peak: Tier;
}

const GRID = '56px 1fr 150px 130px 70px 130px';

const H2H_ROWS: LbRowData[] = [
  { rank: 1, name: 'Thistledown', tier: 'grandmaster', rating: '2141 ± 62', games: 412, peak: 'grandmaster' },
  { rank: 2, name: 'OldTomBarrow', tier: 'grandmaster', rating: '2098 ± 71', games: 287, peak: 'grandmaster' },
  { rank: 3, name: 'redking_enjoyer', tier: 'master', rating: '1993 ± 58', games: 530, peak: 'grandmaster' },
  { rank: 4, name: 'Maple', tier: 'master', rating: '1961 ± 66', games: 198, peak: 'master' },
  { rank: 5, name: 'quietsnap', tier: 'master', rating: '1904 ± 90', games: 121, peak: 'master' },
  { rank: 6, name: 'Bram', tier: 'diamond', rating: '1842 ± 74', games: 265, peak: 'master' },
  { rank: 7, name: 'FennelTheBold', tier: 'diamond', rating: '1815 ± 81', games: 176, peak: 'diamond' },
  { rank: 8, name: 'sevens_and_eights', tier: 'diamond', rating: '1788 ± 69', games: 340, peak: 'diamond' }
];

const YOU_ROW: LbRowData = { rank: 1042, name: 'Juniper', tier: 'gold', rating: '1520 ± 140', games: 47, peak: 'gold' };

interface LbRowProps {
  r: LbRowData;
  you?: boolean;
}

const LbRow: React.FC<LbRowProps> = ({ r, you = false }) => {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: GRID,
        alignItems: 'center',
        gap: 10,
        padding: '9px 14px',
        background: you ? 'rgba(223,174,71,0.08)' : 'transparent',
        borderTop: '1px solid var(--border-subtle)',
        border: you ? '1.5px solid var(--honey-600)' : undefined,
        borderRadius: you ? 'var(--ds-radius-md)' : 0
      }}
    >
      <span style={{ fontFamily: 'var(--ds-font-mono)', fontWeight: 'var(--weight-bold)', color: r.rank <= 3 ? 'var(--honey-400)' : 'var(--text-tertiary)' }}>#{r.rank}</span>
      <span style={{ display: 'flex', alignItems: 'center', gap: 9, fontWeight: 'var(--weight-bold)', minWidth: 0 }}>
        <span style={{ width: 24, height: 24, flex: 'none', borderRadius: '50%', background: 'var(--surface-inset)', border: '1.5px solid var(--outline-ink)', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, color: 'var(--text-primary)' }}>{r.name[0].toUpperCase()}</span>
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.name}{you ? ' (you)' : ''}</span>
      </span>
      <TierBadge tier={r.tier} size='sm' />
      <span style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-sm)' }}>{r.rating}</span>
      <span style={{ fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-xs)', color: 'var(--text-secondary)' }}>{r.games}</span>
      <TierBadge tier={r.peak} size='sm' />
    </div>
  );
};

const POOLS: Array<[string, string]> = [
  ['h2h', 'H2H Ranked · Glicko-2'],
  ['ffa4', 'FFA-4 Ranked · OpenSkill']
];

/**
 * DS preview: seasonal leaderboard (cambia-438). The web client exposes no
 * leaderboard store or ranking service, so rows are typed mock data mirroring
 * the kit sample across both rating pools.
 */
const LeaderboardScreen: React.FC = () => {
  const [pool, setPool] = useState('h2h');
  return (
    <div style={{ padding: 22, maxWidth: 1000, margin: '0 auto', width: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <h1 style={{ margin: 0, fontFamily: 'var(--font-display)', fontSize: 'var(--ds-text-3xl)', fontWeight: 'var(--weight-regular)' }}>Leaderboard</h1>
          <p style={{ margin: '4px 0 0', color: 'var(--text-secondary)' }}>Season 4 · recalibrated monthly · percentile tiers</p>
        </div>
        <div style={{ display: 'flex', gap: 6, background: 'var(--surface-inset)', border: '1.5px solid var(--border-default)', borderRadius: 'var(--radius-pill)', padding: 4 }}>
          {POOLS.map(([id, label]) => (
            <button
              key={id}
              onClick={() => setPool(id)}
              style={{
                padding: '6px 16px',
                borderRadius: 'var(--radius-pill)',
                cursor: 'pointer',
                fontFamily: 'var(--font-ui)',
                fontWeight: 'var(--weight-bold)',
                fontSize: 'var(--ds-text-sm)',
                whiteSpace: 'nowrap',
                background: pool === id ? 'var(--ember-500)' : 'transparent',
                color: pool === id ? 'var(--text-on-ember)' : 'var(--text-secondary)',
                border: pool === id ? '1.5px solid var(--outline-ink)' : '1.5px solid transparent'
              }}
            >{label}</button>
          ))}
        </div>
      </div>
      <Panel style={{ marginTop: 18, padding: '6px 4px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: GRID, gap: 10, padding: '8px 14px', fontSize: 'var(--text-2xs)', fontWeight: 'var(--weight-black)', letterSpacing: 'var(--tracking-caps)', textTransform: 'uppercase', color: 'var(--text-tertiary)' }}>
          <span>Rank</span><span>Player</span><span>Tier</span><span>Rating</span><span>Games</span><span>Peak</span>
        </div>
        {H2H_ROWS.map((r) => <LbRow key={r.rank} r={r} />)}
        <div style={{ padding: '10px 14px', textAlign: 'center', color: 'var(--text-tertiary)', fontFamily: 'var(--ds-font-mono)', fontSize: 'var(--ds-text-xs)' }}>···</div>
        <LbRow r={YOU_ROW} you />
      </Panel>
      {pool === 'ffa4' && (
        <p style={{ marginTop: 10, fontSize: 'var(--ds-text-xs)', color: 'var(--text-tertiary)' }}>FFA-4 shows OpenSkill μ (25.0 start, β = 8.0). Sample data mirrors the H2H board in this kit.</p>
      )}
    </div>
  );
};

export default LeaderboardScreen;
