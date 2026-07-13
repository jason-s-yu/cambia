// src/utils/gameMode.ts

/**
 * Human-readable labels for the game mode values the service accepts
 * (see service/internal/handlers/lobby.go validGameModes).
 */
const GAME_MODE_LABELS: Record<string, string> = {
  head_to_head: 'Head to Head',
  group_of_4: 'Free-for-All (4p)',
  circuit_4p: 'Circuit · 4 Players',
};

/**
 * Maps a raw gameMode value (e.g. "head_to_head") to a human label.
 * Falls back to a title-cased, underscore-stripped rendering of the raw
 * value for modes not yet in the known map, so new modes never render blank.
 */
export function gameModeLabel(mode?: string | null): string {
  if (!mode) return 'Unknown mode';
  const known = GAME_MODE_LABELS[mode];
  if (known) return known;
  return mode
    .split('_')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
