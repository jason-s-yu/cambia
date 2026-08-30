// src/lib/lobbyRuleBuffer.ts
// The rule sheet's edit buffer, as a reducer (cambia-1099 Q3).
//
// Two facts seed the sheet a host edits: the saved lobby, which arrives over the WS, and the
// preset naming it, which is resolved against GET /lobby/presets and so lands one fetch later.
// Both used to run through a single effect keyed on [currentSettings, savedPresetId], which made
// the presets fetch resolving a reason to reset the buffer: a host who started typing before that
// request came back watched the sheet snap back to the saved lobby under them.
//
// The two are separate here. A lobby whose rules actually moved reseeds the buffer, which is the
// case that effect existed for. A preset id arriving against a lobby that has not moved only
// names the sheet, and it does not name one the host has already edited.
//
// Seeding is decided by value and not by object identity: the sheet is handed a fresh LobbyState
// on every lobby event, a player joining included, and a new object holding the same rules is not
// a rule change to reset an edit for.
//
// Types are generic rather than imported, for the reason lobbyPreset.ts states: node --test loads
// this module directly (web/scripts/test-lobby-rule-buffer.mjs), stripping types and resolving no
// bundler aliases.

/** Whether Save has just run. The button reads Saved for as long as it is 'saved'. */
export type SaveStatus = 'idle' | 'saved';

/** The three rule objects the sheet holds, in the shape the lobby state carries them. */
export interface RuleSeed<H, C, S> {
  houseRules: H;
  circuit: C;
  settings: S;
}

export interface RuleBuffer<H, C, S> extends RuleSeed<H, C, S> {
  /** The preset the buffer is on, or null for a sheet that is nobody's preset. */
  presetId: string | null;
  saveStatus: SaveStatus;
  /**
   * Whether the host has moved anything since the buffer was seeded. Not derivable from the
   * values: switching between two rule-identical presets moves no rule and is still an edit.
   */
  touched: boolean;
  /** The saved lobby the buffer was seeded from, which is what a 'saved' action is measured against. */
  seed: RuleSeed<H, C, S>;
}

export type RuleAction<H, C, S> =
  /** The saved lobby, with the preset id resolved against whatever preset list has arrived. */
  | { type: 'saved'; lobby: RuleSeed<H, C, S>; presetId: string | null }
  | { type: 'houseRules'; houseRules: H }
  | { type: 'circuit'; circuit: C }
  | { type: 'settings'; settings: S }
  /** A preset picked from the Ruleset select: it fills the rules and names itself. */
  | { type: 'preset'; presetId: string; houseRules: H; settings: S }
  | { type: 'saveStatus'; status: SaveStatus };

function jsonEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Whether two saved lobbies hold the same rules. Key order is not a rule difference, but both
 * sides here are built from the same lobby state by the same code, so ordering cannot differ; a
 * difference that somehow was one would reseed, which is what the sheet did unconditionally
 * before.
 */
function sameRules<H, C, S>(a: RuleSeed<H, C, S>, b: RuleSeed<H, C, S>): boolean {
  return jsonEqual(a.houseRules, b.houseRules) &&
    jsonEqual(a.circuit, b.circuit) &&
    jsonEqual(a.settings, b.settings);
}

/** A buffer holding exactly what the lobby holds, edited by nobody yet. */
export function seedRuleBuffer<H, C, S>(lobby: RuleSeed<H, C, S>, presetId: string | null): RuleBuffer<H, C, S> {
  return {
    houseRules: lobby.houseRules,
    circuit: lobby.circuit,
    settings: lobby.settings,
    presetId,
    saveStatus: 'idle',
    touched: false,
    seed: lobby
  };
}

export function ruleBufferReducer<H, C, S>(state: RuleBuffer<H, C, S>, action: RuleAction<H, C, S>): RuleBuffer<H, C, S> {
  switch (action.type) {
    case 'saved':
      // The lobby moved: the buffer is reseeded from it, the ruleset naming it included, since
      // leaving the select pointed at a preset the sheet no longer holds would name a ruleset
      // nobody is playing.
      if (!sameRules(state.seed, action.lobby)) return seedRuleBuffer(action.lobby, action.presetId);
      // It did not move, so there is nothing here to reset from. All this can carry is the
      // resolved id, and only onto a sheet the host has not since edited or renamed: an edited
      // one is named by its rules, which is Custom until they are a preset's again.
      if (state.touched || state.presetId === action.presetId) return state;
      return { ...state, presetId: action.presetId };
    case 'houseRules':
      return { ...state, houseRules: action.houseRules, saveStatus: 'idle', touched: true };
    case 'circuit':
      return { ...state, circuit: action.circuit, saveStatus: 'idle', touched: true };
    case 'settings':
      return { ...state, settings: action.settings, saveStatus: 'idle', touched: true };
    case 'preset':
      // Circuit scoring is untouched on purpose: a preset cannot express a round count, so it
      // says nothing about circuit scoring and applying one does not turn it off.
      return {
        ...state,
        houseRules: action.houseRules,
        settings: action.settings,
        presetId: action.presetId,
        saveStatus: 'idle',
        touched: true
      };
    case 'saveStatus':
      return state.saveStatus === action.status ? state : { ...state, saveStatus: action.status };
    default:
      return state;
  }
}
