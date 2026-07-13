import * as React from 'react';

/** Label/value stat line (mono value), for profiles and dashboards. */
export interface StatRowProps {
  label?: string;
  value?: React.ReactNode;
  unit?: string;
  /** e.g. "+12" / "−8"; colored by deltaTone. */
  delta?: string;
  deltaTone?: 'moss' | 'berry';
  style?: React.CSSProperties;
}
export declare function StatRow(props: StatRowProps): JSX.Element;
