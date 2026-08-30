// src/components/lobby/DsMatchSettings.tsx
// DS-styled rule sheet for the live lobby (cambia-484, swept onto the flat
// card-room language in cambia-847). Re-skins the legacy LobbySettingsPanel /
// LobbySettingsView pair: hosts edit a local buffer and Save emits the exact
// `update_rules` WS message the legacy panel sent
// ({ rules: { houseRules, circuit, settings } }); non-hosts read the same
// sheet as plain values (no disabled controls: guests are its readers).
// No WS protocol change.
import React, { useEffect, useMemo, useReducer, useState } from 'react';
import type { LobbyState, HouseRules, CircuitSettings, LobbySettings, LobbyPreset } from '@/types';
import { presetMatchesRules, resolvePresetId, rulesetRow } from '@/lib/lobbyPreset';
import { ruleBufferReducer, seedRuleBuffer } from '@/lib/lobbyRuleBuffer';
import Panel from '@/components/ds/chrome/Panel';
import { EYEBROW } from '@/components/ds/eyebrow';
import Input from '@/components/ds/core/Input';
import Checkbox from '@/components/ds/core/Checkbox';
import Select from '@/components/ds/core/Select';
import Switch from '@/components/ds/core/Switch';
import Badge from '@/components/ds/core/Badge';
import Button from '@/components/ds/core/Button';
import IconButton from '@/components/ds/core/IconButton';
import DsCircuitInfoModal from './DsCircuitInfoModal';
import { getLobbyPresets } from '@/services/lobbyService';
import { gameModeLabel } from '@/utils/gameMode';

interface DsMatchSettingsProps {
  currentSettings: LobbyState;
  isHost: boolean;
  sendMessage: (message: { type: string; body?: unknown }) => void;
}

function jsonEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/**
 * Group heading (cambia-1097). Was the shared EYEBROW, the same style the field
 * labels under it use, so Pace, Deal and Play sat at the weight of the rows they
 * titled and the sheet had to be read to be navigated. Sentence case, above body
 * size, in primary text: one rank under the panel's own heading and two above
 * the eyebrow labels. Size and not weight alone, because the read-only sheet
 * draws its rule names at the body size in --weight-medium, close enough that a
 * bolded heading at the same size would not have separated from them.
 */
const GROUP_TITLE: React.CSSProperties = {
  fontSize: 'var(--ds-text-lg)',
  fontWeight: 'var(--weight-bold)',
  letterSpacing: 'var(--ds-tracking-tight)',
  lineHeight: 'var(--ds-leading-tight)',
  color: 'var(--text-primary)'
};

/** Explanatory line under a group heading. Body copy, so not the smallest tier. */
const HINT: React.CSSProperties = {
  fontSize: 'var(--ds-text-sm)',
  lineHeight: 'var(--ds-leading-snug)',
  color: 'var(--text-tertiary)'
};

const FIELD_GRID: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
  gap: 'var(--space-3)'
};

// Four fields: a fixed pair of columns keeps the group a 2x2 block at every
// width instead of auto-fit's 3+1 with the last field orphaned.
const DEAL_GRID: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
  gap: 'var(--space-3)'
};

const RULE_GRID: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
  gap: 'var(--space-3) var(--space-5)'
};

// Wider than the space-4 it opened at: the divider plus the air around it is
// what makes a group read as a block rather than as one more row (cambia-1097).
const DIVIDER: React.CSSProperties = {
  borderTop: '1px solid var(--border-default)',
  marginTop: 'var(--space-6)',
  paddingTop: 'var(--space-5)'
};

const FLAG_LABEL: React.CSSProperties = {
  display: 'block',
  fontWeight: 'var(--weight-medium)',
  fontSize: 'var(--text-md)',
  lineHeight: 1.35
};

/** The (i) glyph: a hairline ring, a dot and a stem, drawn in the button's own color. */
const InfoIcon: React.FC = () => (
  <svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' strokeWidth='2' strokeLinecap='round' strokeLinejoin='round' aria-hidden='true'>
    <circle cx='12' cy='12' r='9' />
    <path d='M12 11v5' />
    <path d='M12 7.75h.01' />
  </svg>
);

/**
 * One titled group of the rule sheet: heading, optional hint, fields.
 * `info` sits with the title rather than in `action`, since it explains the
 * group and does not set it.
 *
 * The hint took its own line in cambia-1097. Trailing the heading, it pushed the
 * group's only landmark into the middle of a sentence and put two type sizes on
 * one row; under it, the heading is the leftmost thing in the block.
 */
const RuleGroup: React.FC<{ title: string; hint?: string; info?: React.ReactNode; action?: React.ReactNode; first?: boolean; children?: React.ReactNode }> = ({ title, hint, info, action, first = false, children }) => (
  <div style={first ? undefined : DIVIDER}>
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 'var(--space-3)', marginBottom: hint ? 2 : 'var(--space-3)', flexWrap: 'wrap' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', minWidth: 0 }}>
        <span style={GROUP_TITLE}>{title}</span>
        {info}
      </div>
      {action}
    </div>
    {hint && <p style={{ ...HINT, margin: '0 0 var(--space-3)' }}>{hint}</p>}
    {children}
  </div>
);

/**
 * A value the sheet states rather than a control it offers. Holds the height of the input or
 * select it stands in for, so a read-only sheet keeps the rhythm of an editable one.
 */
const READ_ONLY_VALUE: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  minHeight: 'var(--control-h-md)',
  fontSize: 'var(--ds-text-lg)',
  fontWeight: 'var(--weight-bold)',
  color: 'var(--text-primary)'
};

/** Read-only numeric rule: the Input's eyebrow, then the value carrying the weight. */
const RuleValue: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div>
    <span style={{ display: 'block', marginBottom: 6, ...EYEBROW }}>{label}</span>
    <span style={{ ...READ_ONLY_VALUE, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
  </div>
);

/** On/off state for the read-only sheet. Status color reports state, never a control. */
const OnOff: React.FC<{ on: boolean }> = ({ on }) => <Badge tone={on ? 'success' : 'neutral'}>{on ? 'On' : 'Off'}</Badge>;

/** Read-only boolean rule: label and explanation, state at the trailing edge. */
const RuleFlag: React.FC<{ label: string; description: string; on: boolean }> = ({ label, description, on }) => (
  <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
    <span style={{ flex: 1, minWidth: 0 }}>
      <span style={FLAG_LABEL}>{label}</span>
      <span style={{ display: 'block', fontSize: 'var(--ds-text-sm)', lineHeight: 'var(--ds-leading-snug)', color: 'var(--text-secondary)', marginTop: 2 }}>{description}</span>
    </span>
    <span style={{ flex: 'none' }}>
      <OnOff on={on} />
    </span>
  </div>
);

const DsMatchSettings: React.FC<DsMatchSettingsProps> = ({ currentSettings, isHost, sendMessage }) => {
  // The saved lobby's rules, which are what the host's buffer is seeded from and compared against.
  // A lobby carries its auto-start setting under either name depending on which message delivered
  // it, so the sheet reads one effective value rather than each call site picking.
  const savedLobby = useMemo(
    () => ({
      houseRules: currentSettings.houseRules,
      circuit: currentSettings.circuit,
      settings: currentSettings.lobbySettings ?? currentSettings.settings ?? { autoStart: false }
    }),
    [currentSettings]
  );

  // The buffer the host edits and Save sends, and the rules for reseeding it, in lib/
  // lobbyRuleBuffer.ts. It is a reducer rather than five useStates because the interesting part is
  // what a late-arriving preset id may do to a sheet already being typed into (cambia-1099 Q3).
  const [buffer, dispatch] = useReducer(
    ruleBufferReducer<HouseRules, CircuitSettings, LobbySettings>,
    savedLobby,
    (lobby) => seedRuleBuffer(lobby, null)
  );
  const { houseRules, circuit, settings: lobbySettings, saveStatus, presetId } = buffer;
  const [circuitInfoOpen, setCircuitInfoOpen] = useState(false);

  // A ranked or matchmade lobby has its rules fixed by the queue it entered: the service
  // rejects update_rules for one regardless (hub.go), so this only keeps the host from editing
  // a control that would fail on Save (cambia-966). The service says so itself, in
  // lobby_state.rules_locked, because this side cannot work it out: a standing public or private
  // lobby that queued its party into a ranked queue is locked by its mode, and no mode reaches
  // the client (cambia-1099 K2). The type/mode reading stays as the fallback for a lobby object
  // built from the REST create response, which carries neither flag nor a lock to report.
  const locked = currentSettings.rulesLocked ?? (currentSettings.type === 'matchmaking' || currentSettings.mode === 'ranked');
  const canEdit = isHost && !locked;

  // Ruleset presets (cambia-1088): one named ruleset fills the whole sheet. Fetched for every
  // viewer, not just an editing host: a locked or read-only sheet has no preset to apply but it
  // still has one to name, and without the list there is nothing to name it with (cambia-1123).
  // GET /lobby/presets is unauthenticated and static, so a guest costs it nothing. An unreachable
  // endpoint leaves the list empty, which drops the row and leaves the field-by-field sheet
  // exactly as it was.
  const [presets, setPresets] = useState<LobbyPreset[]>([]);

  useEffect(() => {
    let cancelled = false;
    getLobbyPresets()
      .then((list) => {
        if (!cancelled) setPresets(list);
      })
      .catch(() => {
        if (!cancelled) setPresets([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Which preset the lobby's saved rules are: the id the service recorded, and only then a value
  // match (lib/lobbyPreset.ts). Recognising one by value alone named the wrong preset every time
  // for a queue ruleset, since all six queue presets hold the same rules (cambia-1123).
  const savedSubject = useMemo(
    () => ({
      presetId: currentSettings.presetId,
      gameMode: currentSettings.gameMode,
      houseRules: savedLobby.houseRules,
      settings: savedLobby.settings
    }),
    [currentSettings, savedLobby]
  );
  const savedPresetId = useMemo(() => resolvePresetId(presets, savedSubject), [presets, savedSubject]);

  // The saved lobby and the ruleset naming it, handed to the buffer together. The preset id is
  // resolved against a list fetched at mount, so it lands after the lobby does and this runs a
  // second time with the same lobby; reseeding on that pass is what discarded a host's in-flight
  // edits (cambia-1099 Q3), and the reducer is where that no longer happens.
  useEffect(() => {
    dispatch({ type: 'saved', lobby: savedLobby, presetId: savedPresetId });
  }, [savedLobby, savedPresetId]);

  const setRule = <K extends keyof HouseRules>(key: K, value: HouseRules[K]) => {
    dispatch({ type: 'houseRules', houseRules: { ...houseRules, [key]: value } });
  };
  // initialViewCount is bounded by cardsPerPlayer server-side (internal/game/rules.go, cambia-817):
  // the pregame peek cannot cover more cards than the hand holds, and an over-large value rejects
  // the whole update_rules message. Lowering the deal size therefore pulls the peek down with it
  // instead of leaving the panel holding a combination the server refuses.
  const setCardsPerPlayer = (value: number) => {
    dispatch({
      type: 'houseRules',
      houseRules: {
        ...houseRules,
        cardsPerPlayer: value,
        initialViewCount: Math.min(houseRules?.initialViewCount ?? 2, value)
      }
    });
  };
  const setCircuitRule = <K extends keyof CircuitSettings['rules']>(key: K, value: CircuitSettings['rules'][K]) => {
    dispatch({ type: 'circuit', circuit: { ...circuit, rules: { ...circuit.rules, [key]: value } } });
  };

  const activePreset = presets.find((p) => p.id === presetId);
  const onPreset = !!activePreset && presetMatchesRules(activePreset, houseRules, lobbySettings);

  // Switching between two rule-identical presets moves no rule, so the sheet-versus-saved
  // comparison cannot see it. It is still a change worth saving: the lobby would otherwise keep
  // naming the ruleset the host just replaced (cambia-1123).
  const hasChanges = useMemo(() => {
    return !jsonEqual(houseRules, savedLobby.houseRules) ||
      !jsonEqual(circuit, savedLobby.circuit) ||
      !jsonEqual(lobbySettings, savedLobby.settings) ||
      (onPreset && presetId !== savedPresetId);
  }, [houseRules, circuit, lobbySettings, savedLobby, onPreset, presetId, savedPresetId]);

  // Selector for the host of an unlocked lobby, the ruleset's name for everyone else, nothing at
  // all when the preset list could not be read (lib/lobbyPreset.ts).
  const ruleset = rulesetRow({
    presets,
    canEdit,
    saved: savedSubject,
    selectedId: presetId,
    houseRules,
    settings: lobbySettings
  });

  // Applying a preset fills the buffer, leaving Save to send the same expanded sheet it always
  // has. The service accepts a presetId on update_rules and expands it identically, so nothing
  // about the save path had to change to carry one.
  const applyPreset = (id: string) => {
    const p = presets.find((x) => x.id === id);
    if (!p) return;
    dispatch({ type: 'preset', presetId: p.id, houseRules: p.houseRules, settings: p.settings });
  };

  // Save carries the preset id alongside the expanded sheet whenever the buffer is still on one.
  // The service records it and echoes it back as lobby_state.preset_id, which is what lets the
  // sheet name the ruleset the host chose rather than the first one holding those rules; without
  // it, an update_rules that moves a rule clears the recorded id and the lobby goes back to being
  // recognised by value (cambia-1123). The expanded sheet still travels, and lands on top of the
  // preset server-side, so a departed sheet saves exactly as it did before.
  const save = () => {
    if (!isHost || locked) return;
    const rules: Record<string, unknown> = { houseRules, circuit, settings: lobbySettings };
    if (onPreset && activePreset) rules.presetId = activePreset.id;
    sendMessage({ type: 'update_rules', body: { rules } });
    dispatch({ type: 'saveStatus', status: 'saved' });
    setTimeout(() => dispatch({ type: 'saveStatus', status: 'idle' }), 2000);
  };

  const ro = !isHost || locked;
  const num = (v: number | undefined) => (v === undefined || v === null ? '' : String(v));

  // Numeric house rules are range-checked server-side (internal/game/rules.go) and an
  // out-of-range value rejects the whole update_rules message, so the panel clamps to the same
  // bounds rather than letting a typo discard every other edited setting. An empty or
  // unparseable field falls back to the rule's own default.
  const clamped = (raw: string, min: number, max: number, fallback: number) => {
    const parsed = parseInt(raw, 10);
    if (Number.isNaN(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  };

  // Hosts edit; everyone else reads. A read-only sheet renders the value in text-primary
  // rather than a disabled control, whose dimmed text falls under 2:1 in both themes.
  const numField = (label: string, value: number | undefined, onChange: (raw: string) => void) =>
    ro ? (
      <RuleValue label={label} value={num(value)} />
    ) : (
      <Input label={label} type='number' value={num(value)} onChange={(e) => onChange(e.target.value)} />
    );
  const flag = (label: string, description: string, on: boolean, onChange: (v: boolean) => void) =>
    ro ? (
      <RuleFlag label={label} description={description} on={on} />
    ) : (
      <Checkbox checked={on} onChange={onChange} label={label} description={description} />
    );

  return (
    <Panel
      title='Rule sheet'
      action={<Badge tone='info'>{gameModeLabel(currentSettings.gameMode)}</Badge>}
      style={{ minWidth: 0 }}
    >
      {ruleset.kind === 'select' && (
        <RuleGroup title='Ruleset' hint='Fills the sheet below' first>
          <Select
            value={ruleset.value}
            onChange={(e) => applyPreset(e.target.value)}
            options={ruleset.options}
          />
          <p style={{ ...HINT, margin: '6px 0 0' }}>
            {ruleset.description || 'Custom rules. Pick a preset to refill the sheet.'}
          </p>
        </RuleGroup>
      )}

      {/* Read-only twin of the selector above: the same ruleset name, no control (cambia-1123).
          Drawn at the weight RuleValue gives a read-only rule's value, since that is what it is -
          the sheet's one named value, and the only line naming what the queue paired the players
          under. No hint: 'Fills the sheet below' describes picking one, which is not on offer. */}
      {ruleset.kind === 'name' && (
        <RuleGroup title='Ruleset' first>
          <span style={READ_ONLY_VALUE}>{ruleset.name}</span>
        </RuleGroup>
      )}

      <RuleGroup title='Pace' hint='Set 0 to turn the clock or the cap off' first={ruleset.kind === 'none'}>
        <div style={FIELD_GRID}>
          {numField('Turn clock (sec)', houseRules?.turnTimerSec, (raw) => setRule('turnTimerSec', clamped(raw, 0, 86400, 0)))}
          {numField('Turn cap', houseRules?.maxGameTurns, (raw) => setRule('maxGameTurns', clamped(raw, 0, 65535, 46)))}
        </div>
      </RuleGroup>

      <RuleGroup title='Deal' hint='Opening peek is capped at the hand size'>
        <div style={DEAL_GRID}>
          {numField('Cards per hand', houseRules?.cardsPerPlayer, (raw) => setCardsPerPlayer(clamped(raw, 1, 6, 4)))}
          {numField('Opening peek', houseRules?.initialViewCount, (raw) => setRule('initialViewCount', clamped(raw, 0, houseRules?.cardsPerPlayer ?? 4, 2)))}
          {numField('Decks', houseRules?.numDecks, (raw) => setRule('numDecks', clamped(raw, 1, 4, 1)))}
          {numField('Jokers per deck', houseRules?.numJokers, (raw) => setRule('numJokers', clamped(raw, 0, 2, 2)))}
        </div>
      </RuleGroup>

      <RuleGroup title='Play'>
        <div style={{ ...FIELD_GRID, marginBottom: 'var(--space-4)' }}>
          {numField('Cambia from round', houseRules?.cambiaAllowedRound, (raw) => setRule('cambiaAllowedRound', clamped(raw, 0, 255, 0)))}
          {numField('Snap penalty (cards)', houseRules?.penaltyDrawCount, (raw) => setRule('penaltyDrawCount', clamped(raw, 0, 6, 2)))}
        </div>
        <div style={RULE_GRID}>
          {flag('Draw from discard', 'The discard pile is a legal draw.', !!houseRules?.allowDrawFromDiscardPile, (v) => setRule('allowDrawFromDiscardPile', v))}
          {flag('Replace abilities', 'A card discarded by replacing still fires its ability.', !!houseRules?.allowReplaceAbilities, (v) => setRule('allowReplaceAbilities', v))}
          {flag('Snap other hands', "Snap a matching card out of another player's hand.", houseRules?.allowOpponentSnapping ?? true, (v) => setRule('allowOpponentSnapping', v))}
          {flag('Snap race', 'Only the first snap counts. Later snaps take the penalty.', !!houseRules?.snapRace, (v) => setRule('snapRace', v))}
          {flag("Lock caller's hand", "The Cambia caller's hand is safe from snaps and swaps.", houseRules?.lockCallerHand ?? true, (v) => setRule('lockCallerHand', v))}
        </div>
        {/* The forfeit rule and the window it waits out are one setting in two controls, so they
            share a row rather than sitting in the flag grid and the numeric grid apart. */}
        <div style={{ ...RULE_GRID, alignItems: 'start', marginTop: 'var(--space-3)' }}>
          {flag('Forfeit on disconnect', 'A dropped player forfeits instead of rejoining.', !!houseRules?.forfeitOnDisconnect, (v) => setRule('forfeitOnDisconnect', v))}
          {numField('Reconnect grace (sec)', houseRules?.disconnectGraceSec, (raw) => setRule('disconnectGraceSec', clamped(raw, 0, 3600, 60)))}
        </div>
      </RuleGroup>

      {/* Not 'Rounds accumulate toward a target' (cambia-1117 D1): nothing reads
          CircuitRules.TargetScore, and engine/circuit.go ends a circuit on its round
          count. The explainer behind the (i) already says so; the hint used to
          contradict it. */}
      <RuleGroup
        title='Circuit scoring'
        hint='Rounds accumulate; lowest total wins'
        info={
          <IconButton
            size='sm'
            variant='ghost'
            title='About circuit scoring'
            onClick={() => setCircuitInfoOpen(true)}
            style={{ color: 'var(--text-secondary)' }}
          >
            <InfoIcon />
          </IconButton>
        }
        action={
          ro ? (
            <OnOff on={!!circuit?.enabled} />
          ) : (
            <Switch
              checked={!!circuit?.enabled}
              onChange={(v) => dispatch({ type: 'circuit', circuit: { ...circuit, enabled: v } })}
              label={circuit?.enabled ? 'On' : 'Off'}
            />
          )
        }
      >
        {circuit?.enabled && (
          <div style={FIELD_GRID}>
            {numField('Target score', circuit?.rules?.targetScore, (raw) => setCircuitRule('targetScore', parseInt(raw, 10) || 100))}
            {numField('Win bonus', circuit?.rules?.winBonus, (raw) => setCircuitRule('winBonus', parseInt(raw, 10) || -1))}
            {numField('False Cambia penalty', circuit?.rules?.falseCambiaPenalty, (raw) => setCircuitRule('falseCambiaPenalty', parseInt(raw, 10) || 1))}
          </div>
        )}
        {/* Mounted here and not beside the Panel so the explainer travels with the row that
            opens it. Modal portals to the body and renders nothing while closed. */}
        <DsCircuitInfoModal open={circuitInfoOpen} onClose={() => setCircuitInfoOpen(false)} />
      </RuleGroup>

      <div style={{ ...DIVIDER, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
        {ro ? (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 10 }}>
            <span style={FLAG_LABEL}>Auto-start when all ready</span>
            <OnOff on={!!lobbySettings?.autoStart} />
          </span>
        ) : (
          <Switch
            checked={!!lobbySettings?.autoStart}
            onChange={(v) => dispatch({ type: 'settings', settings: { ...lobbySettings, autoStart: v } })}
            label='Auto-start when all ready'
          />
        )}
        {/* A locked sheet reads the same for everyone, host or not: the queue fixed these rules
            when it paired the players, and a matchmade lobby has no player host to point at
            anyway (cambia-1087). Pointing at Create lobby is the answer to what the copy leaves
            people wanting, which is a table they can set their own rules on. */}
        {locked ? (
          <span style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-tertiary)' }}>
            The queue sets these rules. Use Create lobby for your own.
          </span>
        ) : isHost ? (
          <Button variant='primary' size='sm' disabled={!hasChanges} onClick={save}>
            {saveStatus === 'saved' ? 'Saved' : 'Save rules'}
          </Button>
        ) : (
          <span style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-tertiary)' }}>
            Host sets the rules
          </span>
        )}
      </div>
    </Panel>
  );
};

export default DsMatchSettings;
