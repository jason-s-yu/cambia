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

/** Centered dialog on a raised flat surface; inline=true renders the panel without the fixed scrim (specimens/embeds). */
const Modal: React.FC<ModalProps> = ({ open = true, title, onClose, footer, inline = false, width = 440, children }) => {
  if (!open) return null;
  const panel = (
    <div
      style={{
        width: inline ? '100%' : width,
        maxWidth: '92vw',
        background: 'var(--surface-1)',
        border: '1px solid var(--border-default)',
        borderRadius: 'var(--ds-radius-lg)',
        boxShadow: inline ? 'none' : 'var(--shadow-overlay)',
        overflow: 'hidden'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 20px 12px' }}>
        <div style={{ fontSize: 'var(--ds-text-lg)', fontWeight: 'var(--weight-bold)', letterSpacing: 'var(--ds-tracking-tight)', color: 'var(--text-primary)', lineHeight: 'var(--ds-leading-tight)' }}>{title}</div>
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
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, padding: '14px 20px', borderTop: '1px solid var(--border-subtle)', background: 'var(--surface-2)' }}>
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
