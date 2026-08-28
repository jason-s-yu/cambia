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
			// Dark is the product default (cambia-845): the token layer's :root
			// values are the dark set, so an unset attribute already renders
			// dark and a fresh visitor sees no theme flash. 'system' and
			// 'light' remain selectable.
			theme: 'dark',
			setTheme: (theme) => set({ theme })
		}),
		{
			name: 'ui-preferences', // Key for localStorage item
			storage: createJSONStorage(() => localStorage) // Use localStorage for persistence
		}
	)
);