import React from 'react';
import Switch from '../core/Switch';
import Button from '../core/Button';
import Wordmark from './Wordmark';

export interface TopBarNavItem {
  label: string;
  /** Route path this tab navigates to, e.g. '/dashboard'. */
  path: string;
}

export interface TopBarUser {
  name: string;
  /** Formatted rating line, e.g. "1520 ± 140". */
  rating: string;
}

export interface TopBarProps {
  items: TopBarNavItem[];
  /** Current router pathname; drives the active tab highlight. */
  activePath: string;
  onNav: (path: string) => void;
  /** True when the light theme is active (drives the day/night switch). */
  light: boolean;
  onToggleTheme: (light: boolean) => void;
  user: TopBarUser;
  onLogout: () => void;
  style?: React.CSSProperties;
}

/** A tab is active on an exact path match or when the path is a nested child. */
function isActive(activePath: string, path: string): boolean {
  return activePath === path || activePath.startsWith(path + '/');
}

/**
 * Platform top bar: wordmark, primary nav, theme switch, profile chip and
 * logout. Presentational: navigation, theme and identity are wired by the
 * containing layout (AppLayout).
 */
const TopBar: React.FC<TopBarProps> = ({ items, activePath, onNav, light, onToggleTheme, user, onLogout, style }) => {
  return (
    <div
      style={{
        height: 'var(--topbar-h)',
        display: 'flex',
        alignItems: 'center',
        gap: 22,
        padding: '0 20px',
        background: 'var(--surface-1)',
        borderBottom: '1px solid var(--border-subtle)',
        flex: 'none',
        ...style
      }}
    >
      <a onClick={() => onNav(items[0]?.path ?? '/dashboard')} style={{ cursor: 'pointer', color: 'var(--text-primary)', textDecoration: 'none' }}>
        <Wordmark />
      </a>
      <nav style={{ display: 'flex', gap: 4 }}>
        {items.map((item) => {
          const active = isActive(activePath, item.path);
          return (
            <button
              key={item.path}
              onClick={() => onNav(item.path)}
              style={{
                padding: '6px 12px',
                fontFamily: 'var(--font-sans)',
                fontSize: 'var(--text-md)',
                fontWeight: active ? 'var(--weight-bold)' : 'var(--weight-medium)',
                cursor: 'pointer',
                borderRadius: 'var(--ds-radius-md)',
                background: active ? 'var(--surface-2)' : 'transparent',
                color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
                border: '1px solid ' + (active ? 'var(--border-default)' : 'transparent'),
                transition: 'background var(--dur-fast) var(--ds-ease-out), color var(--dur-fast) var(--ds-ease-out)'
              }}
            >
              {item.label}
            </button>
          );
        })}
      </nav>
      <div style={{ flex: 1 }}></div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          color: 'var(--text-secondary)',
          fontSize: 'var(--ds-text-xs)',
          fontWeight: 'var(--weight-bold)'
        }}
      >
        <span>&#9790;</span>
        <Switch checked={light} onChange={onToggleTheme} />
        <span>&#9728;</span>
      </div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 9,
          padding: '4px 12px 4px 5px',
          background: 'var(--surface-2)',
          border: '1px solid var(--border-default)',
          borderRadius: 'var(--radius-pill)'
        }}
      >
        <span
          style={{
            width: 24,
            height: 24,
            borderRadius: '50%',
            background: 'var(--accent-gold)',
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 'var(--weight-black)',
            fontSize: 12,
            color: 'var(--text-on-gold)'
          }}
        >
          {(user.name[0] || '?').toUpperCase()}
        </span>
        <span style={{ lineHeight: 1.2 }}>
          <span style={{ display: 'block', fontWeight: 'var(--weight-medium)', fontSize: 'var(--ds-text-sm)', color: 'var(--text-primary)' }}>{user.name}</span>
          <span style={{ display: 'block', fontSize: 10, fontVariantNumeric: 'tabular-nums', color: 'var(--text-tertiary)' }}>{user.rating}</span>
        </span>
      </div>
      <Button variant="secondary" size="sm" onClick={onLogout}>
        Logout
      </Button>
    </div>
  );
};

export default TopBar;
