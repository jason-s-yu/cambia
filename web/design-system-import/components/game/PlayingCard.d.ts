import * as React from 'react';

/**
 * Playing card: parchment face with ink/berry pips, or ember lattice back.
 * @startingPoint section="Game" subtitle="Card faces, backs, joker, selection state" viewport="700x260"
 */
export interface PlayingCardProps {
  /** 'A','2'–'10','J','Q','K' or 'JOKER'. Ignored when faceDown. */
  rank?: string;
  suit?: 'spades' | 'hearts' | 'diamonds' | 'clubs';
  faceDown?: boolean;
  size?: 'sm' | 'md' | 'lg';
  /** Lifted with honey ring (targeting / chosen). */
  selected?: boolean;
  onClick?: () => void;
  style?: React.CSSProperties;
}
export declare function PlayingCard(props: PlayingCardProps): JSX.Element;
