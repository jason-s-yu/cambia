import React from 'react';

interface ErrorMessageProps {
  message: string | null;
  onClear?: () => void;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ message, onClear }) => {
  if (!message) return null;

  return (
    <div className="bg-red-100 border border-red-400 text-red-700 dark:bg-red-900 dark:border-red-700 dark:text-red-200 px-4 py-3 rounded relative mb-4" role="alert">
      <strong className="font-bold">Error: </strong>
      <span className="block sm:inline">{message}</span>
      {onClear && (
         <button
             onClick={onClear}
             className="absolute top-0 bottom-0 right-0 px-4 py-3 text-red-500 hover:text-red-700 dark:text-red-300 dark:hover:text-red-100"
             aria-label="Clear error"
         >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
         </button>
      )}
    </div>
  );
};

export default ErrorMessage;