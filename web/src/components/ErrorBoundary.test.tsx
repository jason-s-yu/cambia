// src/components/ErrorBoundary.test.tsx
// Render-level coverage for the error boundary (cambia-1235): the recovery panel on a caught
// render throw, the single log point (AC3), Reload wired to a real page reload, and the two-
// boundary composition that keeps a shell alive when a nested (table) boundary is the one that
// actually catches (AC2/AC4).
import React from 'react';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import ErrorBoundary from './ErrorBoundary';

/** Throws on every render; the deliberately-throwing child the render tests mount. */
function Boom(): React.ReactNode {
  throw new Error('boom');
}

// React itself logs a caught render error to console.error in development (a second, separate
// message from our own componentDidCatch log), so every test here spies on console.error and
// filters to this boundary's own tagged calls rather than asserting a raw call count.
let errorSpy: ReturnType<typeof vi.spyOn>;
function ownLogCalls(): unknown[][] {
  return errorSpy.mock.calls.filter((call: unknown[]) => typeof call[0] === 'string' && call[0].startsWith('[ErrorBoundary:'));
}

beforeEach(() => {
  errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  errorSpy.mockRestore();
});

describe('ErrorBoundary recovery panel', () => {
  it('renders children when nothing has thrown', () => {
    render(
      <ErrorBoundary what='table' leaveAction={{ label: 'Leave table', onClick: vi.fn() }}>
        <div data-testid='child'>Table content</div>
      </ErrorBoundary>
    );

    expect(screen.getByTestId('child')).toBeInTheDocument();
  });

  it('catches a render throw and names the failure with plain text, no flavor copy', () => {
    render(
      <ErrorBoundary what='table' leaveAction={{ label: 'Leave table', onClick: vi.fn() }}>
        <Boom />
      </ErrorBoundary>
    );

    const panel = screen.getByRole('alert');
    expect(within(panel).getByText('The table hit an error and stopped rendering.')).toBeInTheDocument();
  });

  it('offers Reload and the caller-supplied leave control', async () => {
    const user = userEvent.setup();
    const onLeave = vi.fn();
    render(
      <ErrorBoundary what='table' leaveAction={{ label: 'Leave table', onClick: onLeave }}>
        <Boom />
      </ErrorBoundary>
    );

    expect(screen.getByTestId('error-boundary-reload')).toHaveTextContent('Reload');
    const leaveButton = screen.getByTestId('error-boundary-leave');
    expect(leaveButton).toHaveTextContent('Leave table');

    await user.click(leaveButton);
    expect(onLeave).toHaveBeenCalledTimes(1);
  });

  it('reloads the page from the Reload control', async () => {
    const user = userEvent.setup();
    // jsdom's window.location.reload is non-configurable, so vi.spyOn can't stub it in place;
    // the whole location object is swapped for the duration of this test instead.
    const originalLocation = window.location;
    const reloadMock = vi.fn();
    Reflect.deleteProperty(window, 'location');
    (window as unknown as { location: Location }).location = { ...originalLocation, reload: reloadMock } as unknown as Location;

    render(
      <ErrorBoundary what='table' leaveAction={{ label: 'Leave table', onClick: vi.fn() }}>
        <Boom />
      </ErrorBoundary>
    );
    await user.click(screen.getByTestId('error-boundary-reload'));

    expect(reloadMock).toHaveBeenCalledTimes(1);
    (window as unknown as { location: Location }).location = originalLocation;
  });

  it('logs the error and component stack exactly once, not once per subsequent render of the fallback', () => {
    const { rerender } = render(
      <ErrorBoundary what='table' leaveAction={{ label: 'Leave table', onClick: vi.fn() }}>
        <Boom />
      </ErrorBoundary>
    );
    expect(ownLogCalls()).toHaveLength(1);
    const [tag, error, componentStack] = ownLogCalls()[0];
    expect(tag).toBe('[ErrorBoundary:table]');
    expect(error).toBeInstanceOf(Error);
    expect(typeof componentStack).toBe('string');

    // A parent re-render (e.g. an unrelated state update elsewhere in the app) re-renders the
    // boundary while it is still showing the recovery panel. It must not log again: the boundary
    // already caught this error and is not re-catching it on every commit.
    rerender(
      <ErrorBoundary what='table' leaveAction={{ label: 'Leave table', onClick: vi.fn() }}>
        <Boom />
      </ErrorBoundary>
    );
    expect(ownLogCalls()).toHaveLength(1);
  });
});

describe('ErrorBoundary composition (app shell + table)', () => {
  it('keeps the shell mounted and reachable when only the nested table boundary catches', async () => {
    const user = userEvent.setup();
    const appLeave = vi.fn();
    const tableLeave = vi.fn();

    render(
      <ErrorBoundary what='app' leaveAction={{ label: 'Go to home', onClick: appLeave }}>
        <nav data-testid='shell-nav'>Shell nav</nav>
        <ErrorBoundary what='table' leaveAction={{ label: 'Leave table', onClick: tableLeave }}>
          <Boom />
        </ErrorBoundary>
      </ErrorBoundary>
    );

    // The shell survives: it never unmounted, because the throw was caught by the inner
    // boundary before it reached the outer one.
    expect(screen.getByTestId('shell-nav')).toBeInTheDocument();

    // The table's own recovery panel is what rendered, not the app shell's.
    expect(screen.getByText('The table hit an error and stopped rendering.')).toBeInTheDocument();
    expect(screen.queryByText('The app hit an error and stopped rendering.')).not.toBeInTheDocument();

    const leaveButton = screen.getByTestId('error-boundary-leave');
    expect(leaveButton).toHaveTextContent('Leave table');
    await user.click(leaveButton);
    expect(tableLeave).toHaveBeenCalledTimes(1);
    expect(appLeave).not.toHaveBeenCalled();
  });
});
