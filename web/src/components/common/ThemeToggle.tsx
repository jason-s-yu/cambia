import React from 'react';
import { useUiStore, type Theme } from '@/stores/uiStore';
import IconButton from '@/components/ds/core/IconButton';

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

const SystemIcon: React.FC = () => (
  <svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' strokeWidth='2' strokeLinecap='round' strokeLinejoin='round' aria-hidden='true'>
    <rect x='3' y='4' width='18' height='12' rx='2' />
    <path d='M8 20h8M12 16v4' />
  </svg>
);

const ORDER: Theme[] = ['light', 'dark', 'system'];
const LABEL: Record<Theme, string> = { light: 'Light', dark: 'Dark', system: 'System' };

/** Cycles the stored preference light -> dark -> system. Entry pages only; the app shell has its own toggle. */
const ThemeToggle: React.FC = () => {
  const theme = useUiStore((state) => state.theme);
  const setTheme = useUiStore((state) => state.setTheme);

  const cycleTheme = () => {
    const nextIndex = (ORDER.indexOf(theme) + 1) % ORDER.length;
    setTheme(ORDER[nextIndex]);
  };

  const icon = theme === 'light' ? <SunIcon /> : theme === 'dark' ? <MoonIcon /> : <SystemIcon />;

  return (
    <IconButton variant='ghost' onClick={cycleTheme} title={`Theme: ${LABEL[theme]}`}>
      {icon}
    </IconButton>
  );
};

export default ThemeToggle;
