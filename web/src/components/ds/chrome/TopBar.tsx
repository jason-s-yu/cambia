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
        padding: '0 22px',
        background: 'var(--surface-card)',
        borderBottom: '1.5px solid var(--border-default)',
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
                padding: '7px 14px',
                fontFamily: 'var(--font-ui)',
                fontSize: 'var(--text-md)',
                fontWeight: 'var(--weight-bold)',
                cursor: 'pointer',
                borderRadius: 'var(--ds-radius-md)',
                background: active ? 'var(--surface-raised)' : 'transparent',
                color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
                border: active ? '1.5px solid var(--border-strong)' : '1.5px solid transparent'
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
          padding: '5px 12px 5px 6px',
          background: 'var(--surface-raised)',
          border: '1.5px solid var(--border-default)',
          borderRadius: 'var(--radius-pill)'
        }}
      >
        <span
          style={{
            width: 26,
            height: 26,
            borderRadius: '50%',
            background: 'var(--dusk-500)',
            border: '1.5px solid var(--outline-ink)',
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 'var(--weight-black)',
            fontSize: 12,
            color: 'var(--text-on-ember)'
          }}
        >
          {(user.name[0] || '?').toUpperCase()}
        </span>
        <span style={{ lineHeight: 1.15 }}>
          <span style={{ display: 'block', fontWeight: 'var(--weight-bold)', fontSize: 'var(--ds-text-sm)' }}>{user.name}</span>
          <span style={{ display: 'block', fontFamily: 'var(--ds-font-mono)', fontSize: 10, color: 'var(--text-tertiary)' }}>{user.rating}</span>
        </span>
      </div>
      <Button variant="secondary" size="sm" onClick={onLogout}>
        Logout
      </Button>
    </div>
  );
};

export default TopBar;
