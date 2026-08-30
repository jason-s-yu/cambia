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

/** A rule sheet plus the ruleset it names, which is what Save sends and what it is measured against. */
export interface RuleBaseline<H, C, S> extends RuleSeed<H, C, S> {
  presetId: string | null;
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
  /**
   * The sheet as Save last sent it, held until the service echoes it back, and null the rest of
   * the time (cambia-1126 item 3).
   *
   * Save is offered whenever the buffer differs from what the lobby holds, and update_rules takes
   * a round trip to move what the lobby holds. Measured against the saved lobby alone, the button
   * therefore stayed live on the sheet that had just been sent, so Save read Saved and was still
   * a control: every further click resent the same rules. Measured against this, a sent sheet is
   * nothing left to save, and Save comes back exactly when the host moves a rule off it.
   */
  submitted: RuleBaseline<H, C, S> | null;
}

export type RuleAction<H, C, S> =
  /** The saved lobby, with the preset id resolved against whatever preset list has arrived. */
  | { type: 'saved'; lobby: RuleSeed<H, C, S>; presetId: string | null }
  | { type: 'houseRules'; houseRules: H }
  | { type: 'circuit'; circuit: C }
  | { type: 'settings'; settings: S }
  /** A preset picked from the Ruleset select: it fills the rules and names itself. */
  | { type: 'preset'; presetId: string; houseRules: H; settings: S }
  /** Save has gone out. `presetId` is the ruleset id that travelled with it, null for a custom sheet. */
  | { type: 'submitted'; presetId: string | null }
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
    seed: lobby,
    submitted: null
  };
}

/**
 * What Save is measured against: the sheet last sent while its echo is still outstanding, and the
 * saved lobby the rest of the time.
 *
 * The lobby is passed in rather than read off `seed`, because the two can differ by a ruleset id:
 * a lobby whose rules did not move does not reseed the buffer, and the id resolved against a
 * late-arriving preset list is carried onto it separately.
 */
function saveBaseline<H, C, S>(
  buffer: RuleBuffer<H, C, S>,
  lobby: RuleSeed<H, C, S>,
  presetId: string | null
): RuleBaseline<H, C, S> {
  return buffer.submitted ?? { ...lobby, presetId };
}

/**
 * Whether Save has anything to send, which is what decides the button.
 *
 * `onPreset` is whether the buffer still holds the rules of the preset it names, decided by the
 * caller against the preset list (lib/lobbyPreset.ts). Switching between two rule-identical
 * presets moves no rule, so no value comparison can see it; it is still a change worth saving,
 * since the lobby would otherwise keep naming the ruleset the host just replaced (cambia-1123).
 */
export function ruleBufferHasChanges<H, C, S>(
  buffer: RuleBuffer<H, C, S>,
  lobby: RuleSeed<H, C, S>,
  presetId: string | null,
  onPreset: boolean
): boolean {
  const baseline = saveBaseline(buffer, lobby, presetId);
  return !jsonEqual(buffer.houseRules, baseline.houseRules) ||
    !jsonEqual(buffer.circuit, baseline.circuit) ||
    !jsonEqual(buffer.settings, baseline.settings) ||
    (onPreset && buffer.presetId !== baseline.presetId);
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
    // Every edit drops `submitted`: the host has moved the sheet off what was sent, so Save is
    // measured against the lobby again and is on offer again with it.
    case 'houseRules':
      return { ...state, houseRules: action.houseRules, saveStatus: 'idle', touched: true, submitted: null };
    case 'circuit':
      return { ...state, circuit: action.circuit, saveStatus: 'idle', touched: true, submitted: null };
    case 'settings':
      return { ...state, settings: action.settings, saveStatus: 'idle', touched: true, submitted: null };
    case 'preset':
      // Circuit scoring is untouched on purpose: a preset cannot express a round count, so it
      // says nothing about circuit scoring and applying one does not turn it off.
      return {
        ...state,
        houseRules: action.houseRules,
        settings: action.settings,
        presetId: action.presetId,
        saveStatus: 'idle',
        touched: true,
        submitted: null
      };
    case 'submitted':
      // The sheet as it went out. Not the lobby: nothing has confirmed these rules yet, and a
      // 'saved' carrying them is what reseeds the buffer for real.
      return {
        ...state,
        saveStatus: 'saved',
        submitted: {
          houseRules: state.houseRules,
          circuit: state.circuit,
          settings: state.settings,
          presetId: action.presetId
        }
      };
    case 'saveStatus':
      return state.saveStatus === action.status ? state : { ...state, saveStatus: action.status };
    default:
      return state;
  }
}
