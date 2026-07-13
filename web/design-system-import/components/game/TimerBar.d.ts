import * as React from 'react';

/** Turn timer: honey bar that turns berry in the last quarter; mono countdown. */
export interface TimerBarProps {
  totalSec?: number;
  remainingSec?: number;
  label?: string;
  style?: React.CSSProperties;
}
export declare function TimerBar(props: TimerBarProps): JSX.Element;
