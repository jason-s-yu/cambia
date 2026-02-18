import React, { useState } from 'react';
import { useAuthStore } from '@/stores/authStore';
import Input from '@/components/common/Input';
import Button from '@/components/common/Button';
import ErrorMessage from '@/components/common/ErrorMessage';

const LoginForm: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const login = useAuthStore((state) => state.login);
  const isLoading = useAuthStore((state) => state.isLoading);
  const error = useAuthStore((state) => state.error);
  const clearError = useAuthStore((state) => state.clearError);


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError(); // Clear previous errors before attempting login
    if (!email || !password) {
        useAuthStore.setState({ error: 'Please enter both email and password.' });
        return;
    }
    await login({ email, password });
    // Navigation is handled by the App component based on isAuthenticated state
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
       <ErrorMessage message={error} onClear={clearError} />
      <Input
        label="Email Address"
        id="email"
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        required
        autoComplete="email"
        placeholder="you@example.com"
        disabled={isLoading}
      />
      <Input
        label="Password"
        id="password"
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        required
        autoComplete="current-password"
        placeholder="••••••••"
        disabled={isLoading}
      />
      <div>
        <Button type="submit" className="w-full justify-center" isLoading={isLoading} disabled={isLoading}>
          Sign In
        </Button>
      </div>
    </form>
  );
};

export default LoginForm;