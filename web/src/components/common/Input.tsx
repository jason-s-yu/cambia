// src/components/common/Input.tsx
import React, { forwardRef } from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string | null;
}

const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, id, error, className = '', ...props }, ref) => {
    const baseStyle = 'block w-full px-3 py-2 border rounded-md shadow-sm focus:outline-none sm:text-sm';
    const normalStyle = 'border-gray-300 dark:border-gray-600 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white';
    const errorStyle = 'border-red-500 focus:ring-red-500 focus:border-red-500 dark:border-red-600';
    // Apply mb-4 only if className doesn't contain specific margin overrides
    const marginStyle = !/!?mb-\d+/.test(className) ? 'mb-4' : '';

    return (
      <div className={marginStyle}>
        {label && (
          <label htmlFor={id} className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            {label}
          </label>
        )}
        <input
          ref={ref}
          id={id}
          className={`${baseStyle} ${error ? errorStyle : normalStyle} ${className}`}
          {...props}
        />
        {error && <p className="mt-1 text-sm text-red-600 dark:text-red-400">{error}</p>}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;