// src/components/ds/core/Modal.tsx
// Centered dialog on a raised flat surface. As a modal it carries the dialog
// contract DsResultsView established in cambia-848: role=dialog + aria-modal,
// an accessible name from the title heading, focus into the panel on open and
// back to the opener on close, Tab kept inside, Escape closes, and the page
// behind the scrim marked inert so its controls leave the tab order. The panel
// is portalled to the body because the inert page is the tree the modal is
// rendered from, and inert cannot be lifted by a descendant (cambia-892, DL-7).
// inline=true renders the panel only, no scrim and no modal behaviour
// (specimens and embeds).
import React, { useCallback, useEffect, useId, useRef } from 'react';
import { createPortal } from 'react-dom';

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

const FOCUSABLE = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])'
].join(', ');

/** Centered dialog on a raised flat surface; inline=true renders the panel without the fixed scrim (specimens/embeds). */
const Modal: React.FC<ModalProps> = ({ open = true, title, onClose, footer, inline = false, width = 440, children }) => {
  const panelRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const isModal = open && !inline;

  // Read through a ref so a new onClose identity does not re-run the open
  // effect, which would re-take focus and re-read the opener.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  // One effect for focus and inert: on close the page has to come back out of
  // inert before the opener is focused, since focus does not land on an inert
  // element. Splitting them runs the cleanups in the wrong order.
  useEffect(() => {
    if (!isModal) return;
    const opener = document.activeElement as HTMLElement | null;
    const page = document.getElementById('root');
    page?.setAttribute('inert', '');
    const first = panelRef.current?.querySelector<HTMLElement>(FOCUSABLE);
    (first ?? panelRef.current)?.focus();
    // Escape listens on the document rather than the panel: a scrim click puts
    // focus on the body, and the key still has to reach the dialog from there.
    const onEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && onCloseRef.current) {
        e.stopPropagation();
        onCloseRef.current();
      }
    };
    document.addEventListener('keydown', onEscape, true);
    return () => {
      document.removeEventListener('keydown', onEscape, true);
      page?.removeAttribute('inert');
      if (opener && document.contains(opener)) opener.focus();
    };
  }, [isModal]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (!isModal) return;
      if (e.key !== 'Tab' || !panelRef.current) return;
      const focusable = Array.from(panelRef.current.querySelectorAll<HTMLElement>(FOCUSABLE));
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === first || !panelRef.current.contains(active))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && (active === last || !panelRef.current.contains(active))) {
        e.preventDefault();
        first.focus();
      }
    },
    [isModal]
  );

  if (!open) return null;

  const panel = (
    <div
      ref={panelRef}
      role='dialog'
      aria-modal={isModal ? 'true' : undefined}
      aria-labelledby={title ? titleId : undefined}
      aria-label={title ? undefined : 'Dialog'}
      tabIndex={-1}
      onKeyDown={onKeyDown}
      style={{
        width: inline ? '100%' : width,
        maxWidth: '92vw',
        background: 'var(--surface-1)',
        border: '1px solid var(--border-default)',
        borderRadius: 'var(--ds-radius-lg)',
        boxShadow: inline ? 'none' : 'var(--shadow-overlay)',
        outline: 'none',
        overflow: 'hidden'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 20px 12px' }}>
        {/* The title is the dialog's name, so it is a heading with an id, not a
            styled div; an omitted title falls back to aria-label. */}
        {title
          ? <h2 id={titleId} style={{ margin: 0, fontSize: 'var(--ds-text-lg)', fontWeight: 'var(--weight-bold)', letterSpacing: 'var(--ds-tracking-tight)', color: 'var(--text-primary)', lineHeight: 'var(--ds-leading-tight)' }}>{title}</h2>
          : <span></span>}
        {onClose && (
          <button
            type='button'
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
  return createPortal(
    <div
      style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--surface-overlay)' }}
      onClick={onClose}
    >
      <div onClick={(e) => e.stopPropagation()}>{panel}</div>
    </div>,
    document.body
  );
};

export default Modal;
