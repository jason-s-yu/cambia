// src/components/ErrorBoundary.tsx
// React error boundary (cambia-1235). Before this, no boundary existed anywhere in the tree (a
// grep for ErrorBoundary|componentDidCatch|getDerivedStateFromError across src returned nothing),
// so a single render throw inside DsGameTable unmounted the whole React tree to a blank page
// mid-game while the socket stayed open and the server kept running the turn timer. The table
// renders live server state through optional-chained shapes (DsGameTable.tsx), so an unexpected
// payload shape is a real, expected failure mode, not a hypothetical one.
//
// Two independent instances wrap the tree (App.tsx for the app shell, LobbyPage.tsx for the
// table) rather than one at the root: a single root boundary would swap the entire app, nav
// included, for the recovery panel on a table-only failure, taking the exit with it. Nesting the
// table's own boundary inside the shell means a table throw is caught before it reaches the
// outer boundary, so the shell (and its own leave/reload controls) stays mounted and reachable.
//
// A class component because getDerivedStateFromError/componentDidCatch have no hook equivalent.
import React from 'react';
import Button from '@/components/ds/core/Button';
import { EYEBROW } from '@/components/ds/eyebrow';

export interface ErrorBoundaryLeaveAction {
  label: string;
  onClick: () => void;
}

export interface ErrorBoundaryProps {
  children: React.ReactNode;
  /** What failed, named in the panel's copy, e.g. "table" or "app". Plain text, no flavor. */
  what: string;
  /** Secondary recovery control shown alongside Reload, e.g. leaving the table. */
  leaveAction: ErrorBoundaryLeaveAction;
}

interface ErrorBoundaryState {
  error: Error | null;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // React invokes componentDidCatch exactly once per caught render error (the commit that
    // trips getDerivedStateFromError), not once per render of the fallback that follows, so this
    // is the single log point regardless of how many times the boundary re-renders while showing
    // the recovery panel (AC3).
    console.error(`[ErrorBoundary:${this.props.what}]`, error, info.componentStack);
  }

  handleReload = (): void => {
    window.location.reload();
  };

  render(): React.ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'center',
          padding: 'var(--space-12) var(--space-5)'
        }}
      >
        <div
          role='alert'
          style={{
            width: '100%',
            maxWidth: 480,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 'var(--space-4)',
            padding: 'var(--space-6) var(--space-5)',
            background: 'var(--surface-1)',
            border: '1px solid var(--status-danger-border)',
            borderRadius: 'var(--ds-radius-lg)',
            textAlign: 'center'
          }}
        >
          <span style={{ ...EYEBROW, color: 'var(--status-danger)' }}>Error</span>
          <span style={{ fontSize: 'var(--text-md)', color: 'var(--text-primary)', lineHeight: 'var(--ds-leading-snug)' }}>
            The {this.props.what} hit an error and stopped rendering.
          </span>
          <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap', justifyContent: 'center' }}>
            <Button variant='primary' onClick={this.handleReload} testId='error-boundary-reload'>
              Reload
            </Button>
            <Button variant='ghost' onClick={this.props.leaveAction.onClick} testId='error-boundary-leave'>
              {this.props.leaveAction.label}
            </Button>
          </div>
        </div>
      </div>
    );
  }
}

export default ErrorBoundary;
