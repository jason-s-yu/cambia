import * as React from 'react';

/** Loading spinner (ember arc) with optional label. */
export interface SpinnerProps {
  /** Diameter in px. Default 24. */
  size?: number;
  label?: string;
  style?: React.CSSProperties;
}
export declare function Spinner(props: SpinnerProps): JSX.Element;
