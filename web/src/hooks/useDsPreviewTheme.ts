import { useEffect, useState } from 'react';
import { useUiStore } from '@/stores/uiStore';

/**
 * Mirrors the real uiStore theme onto the `data-theme` attribute the design
 * tokens key light mode off (design-system-import/tokens/colors.css), for the
 * additive DS preview routes (cambia-438).
 *
 * The rest of the app themes via a `.dark` class on <html> (Tailwind); the DS
 * tokens instead read `data-theme="light"` (absence = dark). This hook keeps
 * the attribute in sync with uiStore while a DS route is mounted and clears it
 * on unmount so it never leaks into the Tailwind screens.
 *
 * Returns the effective light flag and a toggle that writes back to uiStore.
 */
export function useDsPreviewTheme(): { light: boolean; toggle: (light: boolean) => void } {
  const theme = useUiStore((state) => state.theme);
  const setTheme = useUiStore((state) => state.setTheme);
  const [light, setLight] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const apply = () => {
      const isLight = theme === 'light' ? true : theme === 'dark' ? false : !mediaQuery.matches;
      setLight(isLight);
      if (isLight) document.documentElement.setAttribute('data-theme', 'light');
      else document.documentElement.removeAttribute('data-theme');
    };
    apply();
    mediaQuery.addEventListener('change', apply);
    return () => {
      mediaQuery.removeEventListener('change', apply);
      document.documentElement.removeAttribute('data-theme');
    };
  }, [theme]);

  const toggle = (nextLight: boolean) => setTheme(nextLight ? 'light' : 'dark');

  return { light, toggle };
}
