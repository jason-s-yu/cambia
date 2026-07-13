import * as React from 'react';

/** Styled native select with eyebrow label. */
export interface SelectProps {
  label?: string;
  value?: string;
  defaultValue?: string;
  options?: Array<string | { value: string; label: string }>;
  disabled?: boolean;
  onChange?: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  style?: React.CSSProperties;
}
export declare function Select(props: SelectProps): JSX.Element;
