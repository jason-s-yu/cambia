import React, { useState } from 'react';
import { useAuthStore } from '@/stores/authStore';
import Input from '@/components/common/Input';
import Button from '@/components/common/Button';
import ErrorMessage from '@/components/common/ErrorMessage';

const RegisterForm: React.FC = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const register = useAuthStore((state) => state.register);
  const isLoading = useAuthStore((state) => state.isLoading);
  const error = useAuthStore((state) => state.error);
  const clearError = useAuthStore((state) => state.clearError);
  const [validationError, setValidationError] = useState<string | null>(null);


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    clearError();
    setValidationError(null);

    if (!username || !email || !password || !confirmPassword) {
        setValidationError('Please fill in all fields.');
        return;
    }
    if (password !== confirmPassword) {
      setValidationError('Passwords do not match.');
      return;
    }

    await register({ username, email, password });
    // Navigation handled by App component or potentially triggered by register success
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <ErrorMessage message={error || validationError} onClear={() => { clearError(); setValidationError(null); }} />
       <Input
        label="Username"
        id="username"
        type="text"
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        required
        autoComplete="username"
        placeholder="Choose a username"
        disabled={isLoading}
      />
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
        autoComplete="new-password"
        placeholder="Create a password"
        disabled={isLoading}
      />
       <Input
        label="Confirm Password"
        id="confirm-password"
        type="password"
        value={confirmPassword}
        onChange={(e) => setConfirmPassword(e.target.value)}
        required
        autoComplete="new-password"
        placeholder="Confirm your password"
        disabled={isLoading}
        error={validationError && validationError.includes('Passwords do not match') ? validationError : undefined}
      />
      <div>
        <Button type="submit" className="w-full justify-center mt-2" isLoading={isLoading} disabled={isLoading}>
          Register
        </Button>
      </div>
    </form>
  );
};

export default RegisterForm;