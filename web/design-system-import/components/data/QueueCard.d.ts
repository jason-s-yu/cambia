import * as React from 'react';

/** Matchmaking queue card (h2h_rapid, ffa4_standard, …) with Play action. */
export interface QueueCardProps {
  name?: string;
  tagline?: string;
  players?: number;
  rounds?: number;
  /** Estimated match length in minutes. */
  minutes?: number;
  /** Rating pool label, e.g. "Glicko-2" or "OpenSkill". */
  pool?: string;
  /** Primary queue: honey border + PRIMARY badge + ember Play. */
  primary?: boolean;
  ranked?: boolean;
  onPlay?: () => void;
  style?: React.CSSProperties;
}
export declare function QueueCard(props: QueueCardProps): JSX.Element;
