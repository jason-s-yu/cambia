import * as React from 'react';

/** Text input on inset surface; honey focus ring, berry error state. */
export interface InputProps {
  /** Uppercase eyebrow label above the field. */
  label?: string;
  value?: string;
  defaultValue?: string;
  placeholder?: string;
  type?: string;
  /** Space Mono for codes/seeds/numbers. */
  mono?: boolean;
  /** Error message below the field (berry). */
  error?: string;
  disabled?: boolean;
  onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void;
  style?: React.CSSProperties;
}
export declare function Input(props: InputProps): JSX.Element;
