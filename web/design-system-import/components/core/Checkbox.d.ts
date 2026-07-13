import * as React from 'react';

/** Chunky game-piece checkbox with label + optional description (house rules). */
export interface CheckboxProps {
  label?: string;
  /** Secondary line, e.g. the house-rule explanation. */
  description?: string;
  checked?: boolean;
  defaultChecked?: boolean;
  disabled?: boolean;
  onChange?: (checked: boolean) => void;
  style?: React.CSSProperties;
}
export declare function Checkbox(props: CheckboxProps): JSX.Element;
