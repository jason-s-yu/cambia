import * as React from 'react';

/** Rank tier shield chip (Bronze → Grandmaster), tier token colors. */
export interface TierBadgeProps {
  tier?: 'bronze' | 'silver' | 'gold' | 'platinum' | 'diamond' | 'master' | 'grandmaster';
  showLabel?: boolean;
  size?: 'sm' | 'md';
  style?: React.CSSProperties;
}
export declare function TierBadge(props: TierBadgeProps): JSX.Element;
