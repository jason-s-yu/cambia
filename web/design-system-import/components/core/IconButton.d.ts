import * as React from 'react';

/** Square icon-only button; pass a Lucide <i data-lucide> or inline SVG as children. */
export interface IconButtonProps {
  size?: 'sm' | 'md' | 'lg';
  variant?: 'secondary' | 'ghost';
  disabled?: boolean;
  onClick?: () => void;
  /** Accessible label (also tooltip). */
  title?: string;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export declare function IconButton(props: IconButtonProps): JSX.Element;
