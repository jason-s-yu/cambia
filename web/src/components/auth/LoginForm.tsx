import React, { useState } from 'react';
import { useAuthStore } from '@/stores/authStore';
import Input from '@/components/common/Input';
import Button from '@/components/common/Button';
import ErrorMessage from '@/components/common/ErrorMessage';

const LoginForm: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const login = useAuthStore((state) => state.login);
  const loginAsGuest = useAuthStore((state) => state.loginAsGuest);
  const isLoading = useAuthStore((state) => state.isLoading);
  const error = useAuthStore((state) => state.error);
  const clearError = useAuthStore((state) => state.clearError);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError(); // Clear previous errors before attempting login
    if (!email || !password) {
      useAuthStore.setState({ error: 'Enter your email and password.' });
      return;
    }
    await login({ email, password });
    // Navigation is handled by the App component based on isAuthenticated state
  };

  const handleGuestLogin = async () => {
    clearError();
    await loginAsGuest();
    // Navigation is handled by the App component based on isAuthenticated state
  };

  return (
    <form onSubmit={handleSubmit} className='flex flex-col gap-4'>
      <ErrorMessage message={error} onClear={clearError} />
      <Input
        label='Email'
        id='email'
        type='email'
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        required
        autoComplete='email'
        placeholder='you@example.com'
        disabled={isLoading}
        className='mb-0'
      />
      <Input
        label='Password'
        id='password'
        type='password'
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        required
        autoComplete='current-password'
        disabled={isLoading}
        className='mb-0'
      />
      <Button type='submit' className='w-full mt-2' isLoading={isLoading} disabled={isLoading}>
        Sign in
      </Button>
      <div className='flex items-center gap-3 text-2xs font-ds-bold tracking-caps uppercase text-text-tertiary'>
        <span className='flex-1 border-t border-border-subtle' />
        or
        <span className='flex-1 border-t border-border-subtle' />
      </div>
      <div>
        <Button
          type='button'
          variant='secondary'
          className='w-full'
          onClick={handleGuestLogin}
          isLoading={isLoading}
          disabled={isLoading}
        >
          Play as guest
        </Button>
        <p className='mt-2 mb-0 text-center text-ds-xs text-text-tertiary'>
          No account needed. Claim it later from your profile.
        </p>
      </div>
    </form>
  );
};

export default LoginForm;
