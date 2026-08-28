import React, { useState } from 'react';
import Button from '../core/Button';
import IconButton from '../core/IconButton';
import Wordmark from './Wordmark';

export interface TopBarNavItem {
  label: string;
  /** Route path this tab navigates to, e.g. '/dashboard'. */
  path: string;
}

export interface TopBarUser {
  name: string;
  /** Formatted rating line, e.g. "1520 ± 140" or "Unrated". */
  rating: string;
}

export interface TopBarProps {
  items: TopBarNavItem[];
  /** Current router pathname; drives the active tab highlight. */
  activePath: string;
  onNav: (path: string) => void;
  /** True when the light theme is active (drives the sun/moon toggle). */
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

const SunIcon: React.FC = () => (
  <svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' strokeWidth='2' strokeLinecap='round' strokeLinejoin='round' aria-hidden='true'>
    <circle cx='12' cy='12' r='4' />
    <path d='M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41' />
  </svg>
);

const MoonIcon: React.FC = () => (
  <svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' strokeWidth='2' strokeLinecap='round' strokeLinejoin='round' aria-hidden='true'>
    <path d='M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z' />
  </svg>
);

const LogoutIcon: React.FC = () => (
  <svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' strokeWidth='2' strokeLinecap='round' strokeLinejoin='round' aria-hidden='true'>
    <path d='M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4' />
    <path d='M16 17l5-5-5-5M21 12H9' />
  </svg>
);

/** One nav tab. Active: raised surface + border; inactive: ghost with a hover wash. */
const NavTab: React.FC<{ label: string; active: boolean; onClick: () => void }> = ({ label, active, onClick }) => {
  const [hover, setHover] = useState(false);
  return (
    <button
      onClick={onClick}
      aria-current={active ? 'page' : undefined}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        height: 'var(--control-h-sm)',
        padding: '0 12px',
        flex: 'none',
        fontFamily: 'var(--font-sans)',
        fontSize: 'var(--ds-text-sm)',
        fontWeight: active ? 'var(--weight-bold)' : 'var(--weight-medium)',
        whiteSpace: 'nowrap',
        cursor: 'pointer',
        borderRadius: 'var(--ds-radius-md)',
        background: active ? 'var(--surface-2)' : hover ? 'var(--interactive-hover)' : 'transparent',
        color: active || hover ? 'var(--text-primary)' : 'var(--text-secondary)',
        border: '1px solid ' + (active ? 'var(--border-default)' : 'transparent'),
        transition: 'background var(--dur-fast) var(--ds-ease-out), color var(--dur-fast) var(--ds-ease-out)'
      }}
    >
      {label}
    </button>
  );
};

/**
 * Platform top bar: wordmark, primary nav, theme toggle, identity chip and
 * log out. Presentational: navigation, theme and identity are wired by the
 * containing layout (AppLayout).
 *
 * Above the md breakpoint the nav sits inline beside the wordmark. Below it
 * the tabs drop to a second, horizontally scrollable row and the log out
 * control collapses to an icon, so the bar fits a 390px viewport without
 * truncating the identity chip. Between md and lg the bar tightens its gaps:
 * the identity chip is the only item that shrinks (name ellipsis), every
 * other item keeps its intrinsic width, so the log out label never wraps.
 */
const TopBar: React.FC<TopBarProps> = ({ items, activePath, onNav, light, onToggleTheme, user, onLogout, style }) => {
  const tabs = items.map((item) => (
    <NavTab key={item.path} label={item.label} active={isActive(activePath, item.path)} onClick={() => onNav(item.path)} />
  ));

  return (
    <header
      style={{
        background: 'var(--surface-1)',
        borderBottom: '1px solid var(--border-subtle)',
        flex: 'none',
        ...style
      }}
    >
      <div className='flex items-center gap-2 md:gap-3 lg:gap-5 px-4 md:px-5' style={{ height: 'var(--topbar-h)' }}>
        <a
          onClick={() => onNav(items[0]?.path ?? '/dashboard')}
          aria-label='Cambia home'
          style={{ cursor: 'pointer', color: 'var(--text-primary)', textDecoration: 'none', flex: 'none' }}
        >
          <Wordmark />
        </a>
        <nav aria-label='Primary' className='hidden md:flex items-center gap-1'>
          {tabs}
        </nav>
        <div style={{ flex: 1 }}></div>
        <IconButton variant='ghost' title={light ? 'Switch to dark theme' : 'Switch to light theme'} onClick={() => onToggleTheme(!light)}>
          {light ? <MoonIcon /> : <SunIcon />}
        </IconButton>
        <div
          title={user.name + ', ' + user.rating}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            minWidth: 0,
            height: 'var(--control-h-md)',
            padding: '0 12px 0 5px',
            background: 'var(--surface-2)',
            border: '1px solid var(--border-default)',
            borderRadius: 'var(--radius-pill)'
          }}
        >
          <span
            aria-hidden='true'
            style={{
              width: 24,
              height: 24,
              flex: 'none',
              borderRadius: '50%',
              background: 'var(--accent-gold)',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 'var(--weight-black)',
              fontSize: 'var(--ds-text-xs)',
              color: 'var(--text-on-gold)'
            }}
          >
            {(user.name[0] || '?').toUpperCase()}
          </span>
          <span style={{ display: 'flex', flexDirection: 'column', minWidth: 0, lineHeight: 'var(--ds-leading-tight)' }}>
            <span
              className='max-w-[104px] md:max-w-[160px]'
              style={{
                fontWeight: 'var(--weight-medium)',
                fontSize: 'var(--ds-text-sm)',
                color: 'var(--text-primary)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
              }}
            >
              {user.name}
            </span>
            <span
              style={{
                fontSize: 'var(--text-2xs)',
                fontVariantNumeric: 'tabular-nums',
                color: 'var(--text-tertiary)',
                whiteSpace: 'nowrap'
              }}
            >
              {user.rating}
            </span>
          </span>
        </div>
        <span className='hidden md:inline-flex' style={{ flex: 'none' }}>
          <Button variant='secondary' onClick={onLogout}>
            Log out
          </Button>
        </span>
        <span className='inline-flex md:hidden'>
          <IconButton variant='secondary' title='Log out' onClick={onLogout}>
            <LogoutIcon />
          </IconButton>
        </span>
      </div>
      <nav aria-label='Primary' className='flex md:hidden items-center gap-1 px-3 pb-2 overflow-x-auto'>
        {tabs}
      </nav>
    </header>
  );
};

export default TopBar;
