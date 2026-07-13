import * as React from 'react';

/**
 * Tabletop-piece action button: ink outline, hard offset shadow, presses DOWN.
 * @startingPoint section="Core" subtitle="Primary, cambia, gold, secondary and ghost buttons" viewport="700x220"
 */
export interface ButtonProps {
  /** 'primary' ember (default) · 'cambia' berry red (Cambia call / destructive) · 'gold' honey (ranked/win) · 'secondary' · 'ghost' */
  variant?: 'primary' | 'secondary' | 'ghost' | 'cambia' | 'gold';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  fullWidth?: boolean;
  onClick?: () => void;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export declare function Button(props: ButtonProps): JSX.Element;
