import React from 'react';

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

/** Centered dialog with display-serif title; inline=true renders the panel without the fixed scrim (specimens/embeds). */
const Modal: React.FC<ModalProps> = ({ open = true, title, onClose, footer, inline = false, width = 440, children }) => {
  if (!open) return null;
  const panel = (
    <div
      style={{
        width: inline ? '100%' : width,
        maxWidth: '92vw',
        background: 'var(--surface-card)',
        border: 'var(--line-thick) solid var(--outline-ink)',
        borderRadius: 'var(--ds-radius-xl)',
        boxShadow: 'var(--shadow-overlay)',
        overflow: 'hidden'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 20px 12px' }}>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: 'var(--ds-text-xl)', lineHeight: 1.15 }}>{title}</div>
        {onClose && (
          <button
            onClick={onClose}
            aria-label="Close"
            style={{
              width: 30,
              height: 30,
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              background: 'transparent',
              border: 'none',
              borderRadius: 'var(--ds-radius-sm)',
              color: 'var(--text-tertiary)',
              fontSize: 16,
              cursor: 'pointer'
            }}
          >
            ✕
          </button>
        )}
      </div>
      <div style={{ padding: '0 20px 18px', fontSize: 'var(--text-md)', color: 'var(--text-secondary)' }}>{children}</div>
      {footer && (
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, padding: '14px 20px', borderTop: '1.5px solid var(--border-subtle)', background: 'var(--surface-raised)' }}>
          {footer}
        </div>
      )}
    </div>
  );
  if (inline) return panel;
  return (
    <div
      style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--surface-overlay)' }}
      onClick={onClose}
    >
      <div onClick={(e) => e.stopPropagation()}>{panel}</div>
    </div>
  );
};

export default Modal;
