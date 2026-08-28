// src/components/common/Input.tsx
import React, { forwardRef } from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string | null;
}

const MARGIN_RE = /^!?mb-\d+$/;

/**
 * Attribute-passthrough text input for forms (id, name, autoComplete,
 * required, ref). Same visual spec as ds/core/Input: uppercase eyebrow
 * label, inset well, 1px border, gold focus ring, danger border and message
 * on error. The wrapper keeps the legacy mb-4 default so existing stacked
 * forms hold their rhythm; an mb-* class in className moves to the wrapper
 * and overrides it, every other class lands on the input itself.
 */
const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, id, error, className = '', ...props }, ref) => {
    const classes = className.split(/\s+/).filter(Boolean);
    const margin = classes.find((c) => MARGIN_RE.test(c)) ?? 'mb-4';
    const own = classes.filter((c) => !MARGIN_RE.test(c)).join(' ');
    const base = 'block w-full h-[var(--control-h-md)] px-2.5 bg-surface-inset text-text-primary text-ds-md font-sans tabular-nums rounded-ds-sm border outline-none placeholder:text-text-tertiary disabled:opacity-55 disabled:cursor-not-allowed transition-[border-color,box-shadow] duration-[var(--dur-fast)] focus:border-[var(--focus-ring-color)] focus:shadow-[var(--focus-ring)]';
    const tone = error ? 'border-status-danger' : 'border-border-default';

    return (
      <div className={margin}>
        {label && (
          <label htmlFor={id} className='block mb-1.5 text-2xs font-ds-bold tracking-caps uppercase text-text-tertiary'>
            {label}
          </label>
        )}
        <input
          ref={ref}
          id={id}
          aria-invalid={error ? true : undefined}
          className={`${base} ${tone} ${own}`}
          {...props}
        />
        {error && <p className='mt-1.5 mb-0 text-ds-xs font-ds-medium text-status-danger'>{error}</p>}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;
