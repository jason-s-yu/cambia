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
 * Renders a service id ("head_to_head", "ranked-h2h") as title-cased words.
 *
 * The fallback any id -> label map needs: a value the map does not carry is still a wire id, and
 * putting one in the DOM raw shows an identifier where a name belongs (cambia-1086 did it with
 * pool ids, DsLobbyView with lobby types). Splits on both separators the service uses, and
 * answers an absent value with '' so the caller decides whether an unnamed thing renders at all.
 */
export function humanizeId(value?: string | null): string {
  if (!value) return '';
  return value
    .split(/[_-]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Maps a raw gameMode value (e.g. "head_to_head") to a human label.
 * Falls back to a title-cased, separator-stripped rendering of the raw
 * value for modes not yet in the known map, so new modes never render blank.
 */
export function gameModeLabel(mode?: string | null): string {
  if (!mode) return 'Unknown mode';
  return GAME_MODE_LABELS[mode] ?? humanizeId(mode);
}
