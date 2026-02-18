// src/hooks/useTheme.ts
import { useEffect } from 'react';
import { useUiStore, type Theme } from '@/stores/uiStore';

/**
 * Manages the application theme (light/dark/system).
 * Reads preference from Zustand store (persisted to localStorage).
 * Detects system preference using `matchMedia`.
 * Applies/removes the 'dark' class to the `<html>` element.
 * Listens for system theme changes and updates the UI if theme is set to 'system'.
 */
export function useTheme() {
	const storedTheme = useUiStore((state) => state.theme);
	const setTheme = useUiStore((state) => state.setTheme);

	useEffect(() => {
		const root = window.document.documentElement;
		const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

		/** Applies the correct theme class based on the stored or system preference. */
		const applyTheme = (themeToApply: Theme) => {
			let isDark: boolean;

			if (themeToApply === 'system') {
				isDark = mediaQuery.matches;
			} else {
				isDark = themeToApply === 'dark';
			}
			// console.log(`Applying theme: ${themeToApply}, effective dark: ${isDark}`);
			root.classList.toggle('dark', isDark);
		};

		// Apply the theme initially based on the stored preference.
		applyTheme(storedTheme);

		/** Listener for changes in the system's preferred color scheme. */
		const handleSystemThemeChange = (e: MediaQueryListEvent) => {
			// Only re-apply theme if the current setting is 'system'.
			if (useUiStore.getState().theme === 'system') {
				// console.log(`System theme changed (matches dark: ${e.matches}). Re-applying 'system' theme.`);
				applyTheme('system');
			}
		};

		// Add listener for system theme changes.
		mediaQuery.addEventListener('change', handleSystemThemeChange);

		// Cleanup listener on unmount or when storedTheme changes.
		return () => mediaQuery.removeEventListener('change', handleSystemThemeChange);

	}, [storedTheme]); // Re-run effect if the user manually changes the storedTheme preference.

	// Return theme state and setter for convenience, although setTheme is globally available via store hook.
	return { theme: storedTheme, setTheme };
}