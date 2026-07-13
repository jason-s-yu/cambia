import React from 'react';

export type Tier = 'bronze' | 'silver' | 'gold' | 'platinum' | 'diamond' | 'master' | 'grandmaster';

export interface TierBadgeProps {
  tier?: Tier;
  showLabel?: boolean;
  size?: 'sm' | 'md';
  style?: React.CSSProperties;
}

const TIERS: Tier[] = ['bronze', 'silver', 'gold', 'platinum', 'diamond', 'master', 'grandmaster'];

/** Rank tier shield chip (Bronze -> Grandmaster), tier token colors. */
const TierBadge: React.FC<TierBadgeProps> = ({ tier = 'bronze', showLabel = true, size = 'md', style }) => {
  const t = TIERS.includes(tier) ? tier : 'bronze';
  const px = size === 'sm' ? 16 : 22;
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7, ...style }}>
      <span
        style={{
          width: px,
          height: px,
          flex: 'none',
          background: 'var(--tier-' + t + ')',
          border: '1.5px solid var(--outline-ink)',
          borderRadius: '35% 35% 45% 45%',
          boxShadow: '0 1.5px 0 var(--outline-ink)'
        }}
      ></span>
      {showLabel && <span style={{ fontWeight: 800, fontSize: size === 'sm' ? 'var(--ds-text-xs)' : 'var(--ds-text-sm)', textTransform: 'capitalize' }}>{t}</span>}
    </span>
  );
};

export default TierBadge;
