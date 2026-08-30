// src/components/lobby/DsMatchSettings.tsx
// DS-styled rule sheet for the live lobby (cambia-484, swept onto the flat
// card-room language in cambia-847). Re-skins the legacy LobbySettingsPanel /
// LobbySettingsView pair: hosts edit a local buffer and Save emits the exact
// `update_rules` WS message the legacy panel sent
// ({ rules: { houseRules, circuit, settings } }); non-hosts read the same
// sheet as plain values (no disabled controls: guests are its readers).
// No WS protocol change.
import React, { useEffect, useMemo, useState } from 'react';
import type { LobbyState, HouseRules, CircuitSettings, LobbySettings, LobbyPreset } from '@/types';
import Panel from '@/components/ds/chrome/Panel';
import { EYEBROW } from '@/components/ds/eyebrow';
import Input from '@/components/ds/core/Input';
import Checkbox from '@/components/ds/core/Checkbox';
import Select from '@/components/ds/core/Select';
import Switch from '@/components/ds/core/Switch';
import Badge from '@/components/ds/core/Badge';
import Button from '@/components/ds/core/Button';
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

/** Value of the Ruleset select once the sheet no longer matches any preset. Not a preset id. */
const CUSTOM_PRESET_VALUE = '__custom__';

/**
 * Whether a rule sheet is still exactly the preset it was filled from (cambia-1088). Compared
 * field by field over the preset's own keys rather than by serializing both sides: the buffer
 * takes its key order from whichever message delivered it, and a key-order difference is not a
 * rule difference. Circuit settings are not part of a preset and so are not compared - a preset
 * cannot express a round count, so it has nothing to say about circuit scoring.
 */
function presetMatches(preset: LobbyPreset, rules: HouseRules, settings: LobbySettings): boolean {
  const keys = Object.keys(preset.houseRules) as (keyof HouseRules)[];
  return keys.every((k) => rules?.[k] === preset.houseRules[k]) &&
    settings?.autoStart === preset.settings.autoStart;
}

const HINT: React.CSSProperties = {
  fontSize: 'var(--ds-text-xs)',
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

const DIVIDER: React.CSSProperties = {
  borderTop: '1px solid var(--border-subtle)',
  marginTop: 'var(--space-4)',
  paddingTop: 'var(--space-4)'
};

const FLAG_LABEL: React.CSSProperties = {
  display: 'block',
  fontWeight: 'var(--weight-medium)',
  fontSize: 'var(--text-md)',
  lineHeight: 1.35
};

/** One titled group of the rule sheet: eyebrow, optional hint, fields. */
const RuleGroup: React.FC<{ title: string; hint?: string; action?: React.ReactNode; first?: boolean; children?: React.ReactNode }> = ({ title, hint, action, first = false, children }) => (
  <div style={first ? undefined : DIVIDER}>
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 'var(--space-3)', marginBottom: 'var(--space-3)', flexWrap: 'wrap' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
        <span style={EYEBROW}>{title}</span>
        {hint && <span style={HINT}>{hint}</span>}
      </div>
      {action}
    </div>
    {children}
  </div>
);

/** Read-only numeric rule: the Input's eyebrow, then the value carrying the weight. */
const RuleValue: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div>
    <span style={{ display: 'block', marginBottom: 6, ...EYEBROW }}>{label}</span>
    <span
      style={{
        display: 'flex',
        alignItems: 'center',
        minHeight: 'var(--control-h-md)',
        fontSize: 'var(--ds-text-lg)',
        fontWeight: 'var(--weight-bold)',
        fontVariantNumeric: 'tabular-nums',
        color: 'var(--text-primary)'
      }}
    >
      {value}
    </span>
  </div>
);

/** On/off state for the read-only sheet. Status color reports state, never a control. */
const OnOff: React.FC<{ on: boolean }> = ({ on }) => <Badge tone={on ? 'success' : 'neutral'}>{on ? 'On' : 'Off'}</Badge>;

/** Read-only boolean rule: label and explanation, state at the trailing edge. */
const RuleFlag: React.FC<{ label: string; description: string; on: boolean }> = ({ label, description, on }) => (
  <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
    <span style={{ flex: 1, minWidth: 0 }}>
      <span style={FLAG_LABEL}>{label}</span>
      <span style={{ display: 'block', fontSize: 'var(--ds-text-xs)', color: 'var(--text-secondary)', marginTop: 2 }}>{description}</span>
    </span>
    <span style={{ flex: 'none' }}>
      <OnOff on={on} />
    </span>
  </div>
);

const DsMatchSettings: React.FC<DsMatchSettingsProps> = ({ currentSettings, isHost, sendMessage }) => {
  const initialLobbySettings = currentSettings.lobbySettings ?? currentSettings.settings ?? { autoStart: false };

  const [houseRules, setHouseRules] = useState<HouseRules>(currentSettings.houseRules);
  const [circuit, setCircuit] = useState<CircuitSettings>(currentSettings.circuit);
  const [lobbySettings, setLobbySettings] = useState<LobbySettings>(initialLobbySettings);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saved'>('idle');

  useEffect(() => {
    setHouseRules(currentSettings.houseRules);
    setCircuit(currentSettings.circuit);
    setLobbySettings(currentSettings.lobbySettings ?? currentSettings.settings ?? { autoStart: false });
    setSaveStatus('idle');
  }, [currentSettings]);

  const setRule = <K extends keyof HouseRules>(key: K, value: HouseRules[K]) => {
    setHouseRules((prev) => ({ ...prev, [key]: value }));
    setSaveStatus('idle');
  };
  // initialViewCount is bounded by cardsPerPlayer server-side (internal/game/rules.go, cambia-817):
  // the pregame peek cannot cover more cards than the hand holds, and an over-large value rejects
  // the whole update_rules message. Lowering the deal size therefore pulls the peek down with it
  // instead of leaving the panel holding a combination the server refuses.
  const setCardsPerPlayer = (value: number) => {
    setHouseRules((prev) => ({
      ...prev,
      cardsPerPlayer: value,
      initialViewCount: Math.min(prev.initialViewCount ?? 2, value)
    }));
    setSaveStatus('idle');
  };
  const setCircuitRule = <K extends keyof CircuitSettings['rules']>(key: K, value: CircuitSettings['rules'][K]) => {
    setCircuit((prev) => ({ ...prev, rules: { ...prev.rules, [key]: value } }));
    setSaveStatus('idle');
  };

  const hasChanges = useMemo(() => {
    const effective = currentSettings.lobbySettings ?? currentSettings.settings ?? { autoStart: false };
    return !jsonEqual(houseRules, currentSettings.houseRules) ||
      !jsonEqual(circuit, currentSettings.circuit) ||
      !jsonEqual(lobbySettings, effective);
  }, [houseRules, circuit, lobbySettings, currentSettings]);

  // A ranked or matchmade lobby has its rules fixed by the queue it entered: the service
  // rejects update_rules for one regardless (hub.go), so this only keeps the host from editing
  // a control that would 400 on Save (cambia-966). mode can be briefly stale right after the
  // WS connects (buildLobbySnapshot does not send it outside a multi-round match_state), so
  // type carries the check on its own - every ranked queue today is also a matchmaking lobby.
  const locked = currentSettings.type === 'matchmaking' || currentSettings.mode === 'ranked';
  const canEdit = isHost && !locked;

  // Ruleset presets (cambia-1088): one named ruleset fills the whole sheet. Only an editing
  // host fetches them - a locked or read-only sheet has nothing to apply one to. An unreachable
  // endpoint leaves the list empty, which drops the control and leaves the field-by-field sheet
  // exactly as it was.
  const [presets, setPresets] = useState<LobbyPreset[]>([]);
  const [presetId, setPresetId] = useState<string | null>(null);

  useEffect(() => {
    if (!canEdit) return;
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
  }, [canEdit]);

  // Which preset the lobby's saved rules correspond to, re-derived whenever they arrive. First
  // hit wins: the six queue presets are rule-identical to each other, differing only in player
  // count and round count, neither of which is a rule on this sheet. The id is then held rather
  // than re-derived per keystroke, so editing a field reports a departure from the preset the
  // host chose instead of silently jumping to another one that happens to match.
  useEffect(() => {
    if (presets.length === 0) return;
    const effective = currentSettings.lobbySettings ?? currentSettings.settings ?? { autoStart: false };
    setPresetId((prev) => {
      const held = prev ? presets.find((p) => p.id === prev) : undefined;
      if (held && presetMatches(held, currentSettings.houseRules, effective)) return prev;
      return presets.find((p) => presetMatches(p, currentSettings.houseRules, effective))?.id ?? null;
    });
  }, [presets, currentSettings]);

  const activePreset = presets.find((p) => p.id === presetId);
  const onPreset = !!activePreset && presetMatches(activePreset, houseRules, lobbySettings);
  const presetValue = activePreset && onPreset ? activePreset.id : CUSTOM_PRESET_VALUE;
  const presetOptions = [
    ...presets.map((p) => ({ value: p.id, label: p.name })),
    ...(onPreset ? [] : [{ value: CUSTOM_PRESET_VALUE, label: 'Custom' }])
  ];
  const showPresets = canEdit && presets.length > 0;

  // Applying a preset fills the buffer, leaving Save to send the same expanded sheet it always
  // has. The service accepts a presetId on update_rules and expands it identically, so nothing
  // about the save path had to change to carry one.
  const applyPreset = (id: string) => {
    const p = presets.find((x) => x.id === id);
    if (!p) return;
    setHouseRules(p.houseRules);
    setLobbySettings(p.settings);
    setPresetId(p.id);
    setSaveStatus('idle');
  };

  const save = () => {
    if (!isHost || locked) return;
    sendMessage({
      type: 'update_rules',
      body: { rules: { houseRules, circuit, settings: lobbySettings } }
    });
    setSaveStatus('saved');
    setTimeout(() => setSaveStatus('idle'), 2000);
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
      {showPresets && (
        <RuleGroup title='Ruleset' hint='Fills the sheet below' first>
          <Select
            value={presetValue}
            onChange={(e) => applyPreset(e.target.value)}
            options={presetOptions}
          />
          <p style={{ ...HINT, margin: '6px 0 0' }}>
            {onPreset && activePreset ? activePreset.description : 'Custom rules. Pick a preset to refill the sheet.'}
          </p>
        </RuleGroup>
      )}

      <RuleGroup title='Pace' hint='Set 0 to turn the clock or the cap off' first={!showPresets}>
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

      <RuleGroup
        title='Circuit scoring'
        hint='Rounds accumulate toward a target'
        action={
          ro ? (
            <OnOff on={!!circuit?.enabled} />
          ) : (
            <Switch
              checked={!!circuit?.enabled}
              onChange={(v) => { setCircuit((prev) => ({ ...prev, enabled: v })); setSaveStatus('idle'); }}
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
            onChange={(v) => { setLobbySettings((prev) => ({ ...prev, autoStart: v })); setSaveStatus('idle'); }}
            label='Auto-start when all ready'
          />
        )}
        {isHost && !locked ? (
          <Button variant='primary' size='sm' disabled={!hasChanges} onClick={save}>
            {saveStatus === 'saved' ? 'Saved' : 'Save rules'}
          </Button>
        ) : (
          <span style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-tertiary)' }}>
            {isHost ? 'Rules are locked for ranked play' : 'Host sets the rules'}
          </span>
        )}
      </div>
    </Panel>
  );
};

export default DsMatchSettings;
