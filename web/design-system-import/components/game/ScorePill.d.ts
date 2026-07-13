import * as React from 'react';

/** Mono number in an inset pill: scores, round counters, ratings, timers. */
export interface ScorePillProps {
  /** Uppercase micro-label, e.g. "ROUND", "TOTAL". */
  label?: string;
  value?: React.ReactNode;
  tone?: 'neutral' | 'gold' | 'berry' | 'moss';
  big?: boolean;
  style?: React.CSSProperties;
}
export declare function ScorePill(props: ScorePillProps): JSX.Element;
