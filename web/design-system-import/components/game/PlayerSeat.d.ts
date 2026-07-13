import * as React from 'react';

/** Player chip: initial avatar + name + state line. Honey ring = their turn; berry ring = called Cambia. */
export interface PlayerSeatProps {
  username?: string;
  state?: 'turn' | 'ready' | 'cambia' | 'disconnected';
  isYou?: boolean;
  /** Displayed rating (mono). */
  rating?: number | string;
  /** Card count in hand (mono). */
  handSize?: number;
  compact?: boolean;
  style?: React.CSSProperties;
}
export declare function PlayerSeat(props: PlayerSeatProps): JSX.Element;
