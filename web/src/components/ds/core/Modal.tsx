// src/components/ds/core/Modal.tsx
// Centered dialog on a raised flat surface. As a modal it carries the dialog
// contract DsResultsView established in cambia-848: role=dialog + aria-modal,
// an accessible name from the title heading, focus into the panel on open and
// back to the opener on close, Tab kept inside, Escape closes, and the page
// behind the scrim marked inert so its controls leave the tab order. The panel
// is portalled to the body because the inert page is the tree the modal is
// rendered from, and inert cannot be lifted by a descendant (cambia-892, DL-7).
// inline=true renders the panel only: no scrim, no modal behaviour and no
// dialog semantics, since a specimen embedded in a page is not a dialog
// (cambia-914, DL-8 R4).
import React, { useCallback, useEffect, useId, useRef } from 'react';
import { createPortal } from 'react-dom';
import { getAppRoot } from '@/lib/appRoot';
import { pickInitialFocus, type ModalFocusTarget } from '@/lib/modalFocus';

export interface ModalProps {
  open?: boolean;
  title?: React.ReactNode;
  onClose?: () => void;
  /** Action row (Buttons), right-aligned on a raised strip. */
  footer?: React.ReactNode;
  /**
   * Where focus lands on open: 'confirm' (last footer control), 'dismiss'
   * (first), 'body' (first field), 'panel' (nothing armed), or a ref to any
   * control in the panel. Default: the first footer control that is not
   * destructive, then the first body control, then the panel.
   */
  initialFocus?: ModalFocusTarget | React.RefObject<HTMLElement | null>;
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

// Open modals, innermost last. Escape is listened for at document capture, so
// without a stack every mounted modal answered the same keypress and a nested
// dialog closed its parent along with itself (cambia-914, DL-8 R2).
const modalStack: symbol[] = [];

// inert on the app mount is one attribute shared by every open modal, so it is
// refcounted: the first modal sets it, only the last one clears it. Setting and
// clearing per modal let an inner dialog's unmount hand the page back while an
// outer dialog still covered it (cambia-914, DL-8 R2).
let inertDepth = 0;

function acquireInert(): void {
  const page = getAppRoot();
  if (!page) return;
  if (inertDepth === 0) page.setAttribute('inert', '');
  inertDepth++;
}

function releaseInert(): void {
  const page = getAppRoot();
  if (!page) return;
  inertDepth = Math.max(0, inertDepth - 1);
  if (inertDepth === 0) page.removeAttribute('inert');
}

/** Centered dialog on a raised flat surface; inline=true renders the panel without the fixed scrim (specimens/embeds). */
const Modal: React.FC<ModalProps> = ({ open = true, title, onClose, footer, initialFocus, inline = false, width = 440, children }) => {
  const panelRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const footerRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const isModal = open && !inline;

  // Read through a ref so a new onClose identity does not re-run the open
  // effect, which would re-take focus and re-read the opener.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  // Same reason: the target is read once, when the dialog opens.
  const initialFocusRef = useRef(initialFocus);
  initialFocusRef.current = initialFocus;

  // One effect for focus and inert: on close the page has to come back out of
  // inert before the opener is focused, since focus does not land on an inert
  // element. Splitting them runs the cleanups in the wrong order.
  useEffect(() => {
    if (!isModal) return;
    const token = Symbol('ds-modal');
    modalStack.push(token);
    const opener = document.activeElement as HTMLElement | null;
    acquireInert();
    // Focus lands in the panel, not on the Close X, and the caller says where:
    // the old rule took the last control in the action row, which reads as the
    // confirm action only while the confirm action is drawn last, and armed a
    // trailing destructive control under Enter (cambia-935, F6; the
    // confirm-on-open precedent from cambia-914 DL-8 R9 survives as an explicit
    // initialFocus='confirm' at the call site that wanted it).
    const footerControls = footerRef.current ? Array.from(footerRef.current.querySelectorAll<HTMLElement>(FOCUSABLE)) : [];
    const bodyControls = bodyRef.current ? Array.from(bodyRef.current.querySelectorAll<HTMLElement>(FOCUSABLE)) : [];
    const target = initialFocusRef.current;
    const initial = (typeof target === 'object' && target !== null ? target.current : null)
      ?? pickInitialFocus(
        typeof target === 'string' ? target : undefined,
        { footer: footerControls, body: bodyControls, panel: panelRef.current },
        (el) => el.closest('[data-destructive]') !== null
      );
    initial?.focus();
    // Escape listens on the document rather than the panel: a scrim click puts
    // focus on the body, and the key still has to reach the dialog from there.
    // Only the topmost modal answers it.
    const onEscape = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (modalStack[modalStack.length - 1] !== token) return;
      if (!onCloseRef.current) return;
      e.stopPropagation();
      onCloseRef.current();
    };
    document.addEventListener('keydown', onEscape, true);
    return () => {
      document.removeEventListener('keydown', onEscape, true);
      // Unmount order is not guaranteed to be LIFO, so the token is removed by
      // identity rather than popped.
      const i = modalStack.lastIndexOf(token);
      if (i >= 0) modalStack.splice(i, 1);
      releaseInert();
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
      role={inline ? undefined : 'dialog'}
      aria-modal={isModal ? 'true' : undefined}
      aria-labelledby={!inline && title ? titleId : undefined}
      aria-label={!inline && !title ? 'Dialog' : undefined}
      tabIndex={inline ? undefined : -1}
      onKeyDown={onKeyDown}
      style={{
        width: inline ? '100%' : width,
        maxWidth: '92vw',
        // A dialog taller than the viewport clipped its own footer against
        // overflow:hidden while the page behind it was inert, leaving no way to
        // reach the actions. The panel is capped to the viewport less its
        // gutter and the body scrolls instead (cambia-914, DL-8 R5).
        maxHeight: inline ? undefined : 'calc(100dvh - 2 * var(--space-6))',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--surface-1)',
        border: '1px solid var(--border-default)',
        borderRadius: 'var(--ds-radius-lg)',
        boxShadow: inline ? 'none' : 'var(--shadow-overlay)',
        outline: 'none',
        overflow: 'hidden'
      }}
    >
      <div style={{ flex: '0 0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 20px 12px' }}>
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
      <div ref={bodyRef} style={{ flex: '1 1 auto', minHeight: 0, overflowY: 'auto', padding: '0 20px 18px', fontSize: 'var(--text-md)', color: 'var(--text-secondary)' }}>{children}</div>
      {footer && (
        <div ref={footerRef} style={{ flex: '0 0 auto', display: 'flex', justifyContent: 'flex-end', gap: 10, padding: '14px 20px', borderTop: '1px solid var(--border-subtle)', background: 'var(--surface-2)' }}>
          {footer}
        </div>
      )}
    </div>
  );
  if (inline) return panel;
  return createPortal(
    <div
      style={{ position: 'fixed', inset: 0, zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 'var(--space-6)', background: 'var(--surface-overlay)' }}
      onClick={onClose}
    >
      <div onClick={(e) => e.stopPropagation()} style={{ display: 'flex', minHeight: 0, maxHeight: '100%' }}>{panel}</div>
    </div>,
    document.body
  );
};

export default Modal;
