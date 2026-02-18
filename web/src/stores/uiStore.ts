// src/stores/uiStore.ts
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

export type Theme = 'light' | 'dark' | 'system';

interface UiState {
	theme: Theme;
	setTheme: (theme: Theme) => void;
}

/** Manages UI-related preferences, persisted to localStorage */
export const useUiStore = create<UiState>()(
	persist(
		(set) => ({
			theme: 'system', // Default to system preference
			setTheme: (theme) => set({ theme })
		}),
		{
			name: 'ui-preferences', // Key for localStorage item
			storage: createJSONStorage(() => localStorage) // Use localStorage for persistence
		}
	)
);