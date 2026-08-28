import React from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
}

/**
 * Centered dialog on the overlay scrim. A detached layer, so it is the one
 * surface here that carries the overlay shadow; no blur behind it.
 * Legacy isOpen/onClose API kept for the training dialogs; new markup
 * composes ds/core/Modal.
 */
const Modal: React.FC<ModalProps> = ({ isOpen, onClose, title, children }) => {
  if (!isOpen) return null;

  return (
    <div
      className='fixed inset-0 z-50 flex items-center justify-center bg-surface-overlay p-4'
      onClick={onClose}
    >
      <div
        role='dialog'
        aria-modal='true'
        aria-label={title}
        className='relative w-full max-w-md bg-surface-1 border border-border-default rounded-ds-lg shadow-ds-overlay text-text-primary'
        onClick={(e) => e.stopPropagation()}
      >
        <div className='flex items-center justify-between gap-3 px-5 pt-4 pb-3'>
          <h2 className='m-0 text-ds-lg font-ds-bold tracking-ds-tight leading-ds-tight'>{title}</h2>
          <button
            type='button'
            onClick={onClose}
            aria-label='Close'
            className='inline-flex items-center justify-center w-[30px] h-[30px] flex-none rounded-ds-sm bg-transparent border-0 text-text-tertiary hover:text-text-primary hover:bg-[var(--interactive-hover)] cursor-pointer'
          >
            <svg width='14' height='14' viewBox='0 0 14 14' fill='none' stroke='currentColor' strokeWidth='2' strokeLinecap='round' aria-hidden='true'>
              <path d='M2 2 L12 12 M12 2 L2 12' />
            </svg>
          </button>
        </div>
        <div className='px-5 pb-5 text-ds-md text-text-secondary'>
          {children}
        </div>
      </div>
    </div>
  );
};

export default Modal;
