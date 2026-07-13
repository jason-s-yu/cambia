import React from 'react';
import Switch from '../core/Switch';
import Wordmark from './Wordmark';

export type TopBarTab = 'home' | 'leaderboard';

export interface TopBarUser {
  name: string;
  /** Formatted rating line, e.g. "1520 ± 140". */
  rating: string;
}

export interface TopBarProps {
  active: TopBarTab;
  onNav: (tab: TopBarTab) => void;
  /** True when the light theme is active (drives the day/night switch). */
  light: boolean;
  onToggleTheme: (light: boolean) => void;
  user: TopBarUser;
  style?: React.CSSProperties;
}

const TABS: Array<[TopBarTab, string]> = [
  ['home', 'Play'],
  ['leaderboard', 'Leaderboard']
];

/**
 * Platform top bar: wordmark, primary nav, theme switch and profile chip.
 *
 * Adapted from design-system-import/ui_kits/platform/shared.jsx TopBar for the
 * additive DS preview (cambia-438). Presentational: navigation, theme and the
 * profile identity are wired by the containing layout.
 */
const TopBar: React.FC<TopBarProps> = ({ active, onNav, light, onToggleTheme, user, style }) => {
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
      <a onClick={() => onNav('home')} style={{ cursor: 'pointer', color: 'var(--text-primary)', textDecoration: 'none' }}>
        <Wordmark />
      </a>
      <nav style={{ display: 'flex', gap: 4 }}>
        {TABS.map(([id, label]) => (
          <button
            key={id}
            onClick={() => onNav(id)}
            style={{
              padding: '7px 14px',
              fontFamily: 'var(--font-ui)',
              fontSize: 'var(--text-md)',
              fontWeight: 'var(--weight-bold)',
              cursor: 'pointer',
              borderRadius: 'var(--ds-radius-md)',
              background: active === id ? 'var(--surface-raised)' : 'transparent',
              color: active === id ? 'var(--text-primary)' : 'var(--text-secondary)',
              border: active === id ? '1.5px solid var(--border-strong)' : '1.5px solid transparent'
            }}
          >
            {label}
          </button>
        ))}
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
    </div>
  );
};

export default TopBar;
