import * as React from 'react';

/** Pill badge for statuses, counts and labels. */
export interface BadgeProps {
  tone?: 'neutral' | 'success' | 'danger' | 'warning' | 'info' | 'gold' | 'ember';
  /** Leading status dot. */
  dot?: boolean;
  /** Space Mono contents (counts, ratings). */
  mono?: boolean;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export declare function Badge(props: BadgeProps): JSX.Element;
