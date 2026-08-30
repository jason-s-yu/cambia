// src/lib/lobbyPreset.ts
// Which ruleset a lobby is on, as the rule sheet reads it (cambia-1123).
//
// The sheet used to answer this by comparing house rules against GET /lobby/presets and taking
// the first match. It cannot be answered that way: MATCHMAKING.md 5.2 fixes one ruleset for
// every ranked queue, so all six queue presets hold byte-identical houseRules and differ only in
// player and round count, neither of which is a rule on the sheet. Creating a lobby from H2H
// Rapid therefore opened a sheet reading H2H Quick, whichever the service happened to list first.
//
// The service now records the id (lobby.PresetID, sent as lobby_state.preset_id and returned as
// presetId on POST /lobby/create) and that recorded id is the answer. Value matching survives
// only as the fallback for a lobby carrying no recorded id, and it takes the game mode into
// account there, or a 4-player preset names a 2-player lobby's rules.
//
// Types are structural rather than imported: this module is loaded directly by node --test
// (web/scripts/test-lobby-preset.mjs), which strips types but resolves no bundler aliases. It
// also reflects what actually arrives - the rule objects come off the wire as JSON.

/** One entry of GET /lobby/presets, in the shape this module reads it. */
export interface PresetOption {
  id: string;
  /** Display name, which the sheet shows in place of the id. Always sent (lobby/presets.go). */
  name: string;
  /** One line about the ruleset, shown under the host's select. */
  description: string;
  /** Empty for the default preset, which fixes no player count. */
  gameMode?: string;
  houseRules: unknown;
  settings: { autoStart?: boolean };
}

/** The saved lobby state a preset is resolved against. */
export interface LobbyRuleSubject {
  /** The id the service recorded, from lobby_state.preset_id or the create response. */
  presetId?: string | null;
  gameMode?: string | null;
  houseRules?: unknown;
  settings?: unknown;
}

/** Value of the Ruleset select once the sheet is on no preset. Not a preset id. */
export const CUSTOM_PRESET_VALUE = '__custom__';

/** Reads a key off a value that may be null, undefined, or not an object at all. */
function at(source: unknown, key: string): unknown {
  if (!source || typeof source !== 'object') return undefined;
  return (source as Record<string, unknown>)[key];
}

/** The keys of a value that may not be an object. */
function keysOf(source: unknown): string[] {
  if (!source || typeof source !== 'object') return [];
  return Object.keys(source as Record<string, unknown>);
}

/**
 * Whether a rule sheet holds exactly the rules a preset names. Compared field by field over the
 * preset's own keys rather than by serializing both sides: a buffer takes its key order from
 * whichever message delivered it, and key order is not a rule difference.
 *
 * Circuit settings are not compared, and not because it would be inconvenient: a preset cannot
 * express a round count, so it says nothing about circuit scoring and a lobby that turns it on
 * has not left the preset.
 */
export function presetMatchesRules(preset: PresetOption, houseRules: unknown, settings: unknown): boolean {
  const keys = keysOf(preset.houseRules);
  // A preset carrying no rules at all would otherwise match every sheet ever rendered.
  if (keys.length === 0) return false;
  return keys.every((key) => at(houseRules, key) === at(preset.houseRules, key)) &&
    at(settings, 'autoStart') === preset.settings.autoStart;
}

/**
 * Whether a preset could be the one a lobby is playing, by value. The game mode has to agree:
 * the queue presets are rule-identical, so without this gate the first 2-player preset in the
 * list answers for a 4-player lobby. A preset that fixes no game mode (the default) fits any
 * lobby.
 */
export function presetFitsLobby(preset: PresetOption, subject: LobbyRuleSubject): boolean {
  if (preset.gameMode && preset.gameMode !== subject.gameMode) return false;
  return presetMatchesRules(preset, subject.houseRules, subject.settings);
}

/**
 * The preset a saved lobby is on, or null for a sheet that is nobody's preset.
 *
 * The recorded id wins outright wherever it names a preset in the list. It is the only thing
 * that can tell two rule-identical presets apart, and the service clears it the moment an edit
 * departs from the preset, so an id that is still there is still true. Value matching runs only
 * when there is no id to honour: a lobby created before the service recorded them, or one whose
 * id names a queue that has since left the config.
 */
export function resolvePresetId(presets: PresetOption[], subject: LobbyRuleSubject): string | null {
  const recorded = subject.presetId;
  if (recorded && presets.some((p) => p.id === recorded)) return recorded;
  return presets.find((p) => presetFitsLobby(p, subject))?.id ?? null;
}

/** Label for a sheet that is nobody's preset, in the select and on a read-only row alike. */
export const CUSTOM_PRESET_LABEL = 'Custom';

/** One entry of the host's Ruleset select. */
export interface RulesetChoice {
  value: string;
  label: string;
}

/**
 * What the rule sheet's Ruleset row shows.
 *
 * `none` is having no preset list to answer with, which is not the same as being on no preset:
 * an unreachable GET /lobby/presets drops the row rather than calling the lobby Custom, having
 * read nothing that would support saying so.
 */
export type RulesetRow =
  | { kind: 'none' }
  | { kind: 'name'; name: string }
  | { kind: 'select'; value: string; options: RulesetChoice[]; description: string };

/** The sheet a Ruleset row is decided against. */
export interface RulesetRowInput {
  /** GET /lobby/presets, empty when it could not be read. */
  presets: PresetOption[];
  /** Whether the viewer may change the ruleset: the host of an unlocked lobby, and nobody else. */
  canEdit: boolean;
  /** The saved lobby, which is what names a sheet the viewer cannot change. */
  saved: LobbyRuleSubject;
  /** The host's current pick and the buffer it filled. Read only when `canEdit`. */
  selectedId?: string | null;
  houseRules?: unknown;
  settings?: unknown;
}

/**
 * Which Ruleset row the rule sheet renders (cambia-1123).
 *
 * A locked or non-host sheet gets the name, not a dropped row. The row used to be gated on the
 * viewer being able to change it, which took it off every matchmade lobby: those are locked by
 * definition, so the one sheet whose ruleset the player never chose - and most needs named - was
 * the one that never named it.
 *
 * The read-only name comes from the resolved id and is not re-checked against the rules, unlike
 * the host's select. There is no buffer here to have departed from the preset, and the recorded
 * id outranks the rules by construction: the queue presets are rule-identical, so a value check
 * could only ever disagree with the id by naming the wrong preset.
 */
export function rulesetRow(input: RulesetRowInput): RulesetRow {
  const { presets, canEdit, saved, selectedId, houseRules, settings } = input;
  if (presets.length === 0) return { kind: 'none' };

  if (!canEdit) {
    const preset = presets.find((p) => p.id === resolvePresetId(presets, saved));
    // The name alone. A preset's description is written for a host picking one out of the New
    // lobby dialog ("A custom lobby plays a single round"), which is not what a seated player
    // reading a sheet they cannot change is asking.
    return { kind: 'name', name: preset ? preset.name : CUSTOM_PRESET_LABEL };
  }

  const active = presets.find((p) => p.id === selectedId);
  const onPreset = !!active && presetMatchesRules(active, houseRules, settings);
  return {
    kind: 'select',
    value: onPreset && active ? active.id : CUSTOM_PRESET_VALUE,
    options: [
      ...presets.map((p) => ({ value: p.id, label: p.name })),
      ...(onPreset ? [] : [{ value: CUSTOM_PRESET_VALUE, label: CUSTOM_PRESET_LABEL }])
    ],
    description: onPreset && active ? active.description : ''
  };
}
