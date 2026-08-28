import React from 'react';

export type Tier = 'bronze' | 'silver' | 'gold' | 'platinum' | 'diamond' | 'master' | 'grandmaster';

export interface TierBadgeProps {
  tier?: Tier;
  showLabel?: boolean;
  size?: 'sm' | 'md';
  style?: React.CSSProperties;
}

const TIERS: Tier[] = ['bronze', 'silver', 'gold', 'platinum', 'diamond', 'master', 'grandmaster'];

/** Rank tier chip (Bronze -> Grandmaster), flat tier token colors. */
const TierBadge: React.FC<TierBadgeProps> = ({ tier = 'bronze', showLabel = true, size = 'md', style }) => {
  const t = TIERS.includes(tier) ? tier : 'bronze';
  const px = size === 'sm' ? 14 : 18;
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7, ...style }}>
      <span
        style={{
          width: px,
          height: px,
          flex: 'none',
          background: 'var(--tier-' + t + ')',
          borderRadius: 'var(--ds-radius-sm)'
        }}
      ></span>
      {showLabel && (
        <span
          style={{
            fontWeight: 'var(--weight-bold)',
            fontSize: size === 'sm' ? 'var(--ds-text-xs)' : 'var(--ds-text-sm)',
            color: 'var(--tier-' + t + ')',
            textTransform: 'capitalize'
          }}
        >
          {t}
        </span>
      )}
    </span>
  );
};

export default TierBadge;
