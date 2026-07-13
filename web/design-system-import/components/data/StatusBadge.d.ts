import * as React from 'react';

/** Process/eval status pill (training dashboard): mono uppercase + dot. */
export interface StatusBadgeProps {
  status?: 'created' | 'starting' | 'running' | 'stopping' | 'stopped' | 'crashed' | 'queued' | 'succeeded' | 'failed';
  style?: React.CSSProperties;
}
export declare function StatusBadge(props: StatusBadgeProps): JSX.Element;
