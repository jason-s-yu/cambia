import React from 'react';

interface ErrorMessageProps {
  message: string | null;
  onClear?: () => void;
}

/** Inline error strip on the danger status tint, with an optional dismiss. */
const ErrorMessage: React.FC<ErrorMessageProps> = ({ message, onClear }) => {
  if (!message) return null;

  return (
    <div
      role='alert'
      className='flex items-start gap-3 px-3.5 py-2.5 mb-4 rounded-ds-md border text-ds-sm font-ds-medium'
      style={{ background: 'var(--status-danger-bg)', borderColor: 'var(--status-danger-border)', color: 'var(--status-danger)' }}
    >
      <span className='flex-1 min-w-0 leading-ds-snug'>{message}</span>
      {onClear && (
        <button
          type='button'
          onClick={onClear}
          aria-label='Dismiss error'
          className='inline-flex items-center justify-center w-5 h-5 flex-none rounded-ds-sm bg-transparent border-0 cursor-pointer opacity-70 hover:opacity-100'
          style={{ color: 'inherit' }}
        >
          <svg width='12' height='12' viewBox='0 0 14 14' fill='none' stroke='currentColor' strokeWidth='2' strokeLinecap='round' aria-hidden='true'>
            <path d='M2 2 L12 12 M12 2 L2 12' />
          </svg>
        </button>
      )}
    </div>
  );
};

export default ErrorMessage;
