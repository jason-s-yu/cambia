import * as React from 'react';

/** Centered dialog with display-serif title; inline=true renders the panel without the fixed scrim (specimens/embeds). */
export interface ModalProps {
  open?: boolean;
  title?: React.ReactNode;
  onClose?: () => void;
  /** Action row (Buttons), right-aligned on a raised strip. */
  footer?: React.ReactNode;
  /** Render panel only, no fixed overlay. */
  inline?: boolean;
  width?: number;
  children?: React.ReactNode;
}
export declare function Modal(props: ModalProps): JSX.Element | null;
