// src/components/lobby/DsMatchSettings.tsx
// DS-styled rule sheet for the live lobby (cambia-484, swept onto the flat
// card-room language in cambia-847). Re-skins the legacy LobbySettingsPanel /
// LobbySettingsView pair: hosts edit a local buffer and Save emits the exact
// `update_rules` WS message the legacy panel sent
// ({ rules: { houseRules, circuit, settings } }); non-hosts see the same fields
// disabled. No WS protocol change.
import React, { useEffect, useMemo, useState } from 'react';
import type { LobbyState, HouseRules, CircuitSettings, LobbySettings } from '@/types';
import Panel from '@/components/ds/chrome/Panel';
import Input from '@/components/ds/core/Input';
import Checkbox from '@/components/ds/core/Checkbox';
import Switch from '@/components/ds/core/Switch';
import Badge from '@/components/ds/core/Badge';
import Button from '@/components/ds/core/Button';
import { gameModeLabel } from '@/utils/gameMode';

interface DsMatchSettingsProps {
  currentSettings: LobbyState;
  isHost: boolean;
  sendMessage: (message: { type: string; body?: unknown }) => void;
}

function jsonEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

const EYEBROW: React.CSSProperties = {
  fontSize: 'var(--text-2xs)',
  fontWeight: 'var(--weight-bold)',
  letterSpacing: 'var(--tracking-caps)',
  textTransform: 'uppercase',
  color: 'var(--text-tertiary)'
};

const HINT: React.CSSProperties = {
  fontSize: 'var(--ds-text-xs)',
  color: 'var(--text-tertiary)'
};

const FIELD_GRID: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
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

  const save = () => {
    if (!isHost) return;
    sendMessage({
      type: 'update_rules',
      body: { rules: { houseRules, circuit, settings: lobbySettings } }
    });
    setSaveStatus('saved');
    setTimeout(() => setSaveStatus('idle'), 2000);
  };

  const ro = !isHost;
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

  return (
    <Panel
      title='Rule sheet'
      action={<Badge tone='info'>{gameModeLabel(currentSettings.gameMode)}</Badge>}
      style={{ minWidth: 0 }}
    >
      <RuleGroup title='Pace' hint='0 turns the clock or the cap off' first>
        <div style={FIELD_GRID}>
          <Input
            label='Turn clock (sec)'
            type='number'
            disabled={ro}
            value={num(houseRules?.turnTimerSec)}
            onChange={(e) => setRule('turnTimerSec', clamped(e.target.value, 0, 86400, 0))}
          />
          <Input
            label='Turn cap'
            type='number'
            disabled={ro}
            value={num(houseRules?.maxGameTurns)}
            onChange={(e) => setRule('maxGameTurns', clamped(e.target.value, 0, 65535, 46))}
          />
        </div>
      </RuleGroup>

      <RuleGroup title='Deal' hint='The peek never exceeds the hand'>
        <div style={FIELD_GRID}>
          <Input
            label='Cards per hand'
            type='number'
            disabled={ro}
            value={num(houseRules?.cardsPerPlayer)}
            onChange={(e) => setCardsPerPlayer(clamped(e.target.value, 1, 6, 4))}
          />
          <Input
            label='Opening peek'
            type='number'
            disabled={ro}
            value={num(houseRules?.initialViewCount)}
            onChange={(e) => setRule('initialViewCount', clamped(e.target.value, 0, houseRules?.cardsPerPlayer ?? 4, 2))}
          />
          <Input
            label='Decks'
            type='number'
            disabled={ro}
            value={num(houseRules?.numDecks)}
            onChange={(e) => setRule('numDecks', clamped(e.target.value, 1, 4, 1))}
          />
          <Input
            label='Jokers per deck'
            type='number'
            disabled={ro}
            value={num(houseRules?.numJokers)}
            onChange={(e) => setRule('numJokers', clamped(e.target.value, 0, 2, 2))}
          />
        </div>
      </RuleGroup>

      <RuleGroup title='Play'>
        <div style={{ ...FIELD_GRID, marginBottom: 'var(--space-4)' }}>
          <Input
            label='Cambia from round'
            type='number'
            disabled={ro}
            value={num(houseRules?.cambiaAllowedRound)}
            onChange={(e) => setRule('cambiaAllowedRound', clamped(e.target.value, 0, 255, 0))}
          />
          <Input
            label='Snap penalty (cards)'
            type='number'
            disabled={ro}
            value={num(houseRules?.penaltyDrawCount)}
            onChange={(e) => setRule('penaltyDrawCount', clamped(e.target.value, 0, 6, 2))}
          />
        </div>
        <div style={RULE_GRID}>
          <Checkbox
            disabled={ro}
            checked={!!houseRules?.allowDrawFromDiscardPile}
            onChange={(v) => setRule('allowDrawFromDiscardPile', v)}
            label='Draw from discard'
            description='The discard pile is a legal draw.'
          />
          <Checkbox
            disabled={ro}
            checked={!!houseRules?.allowReplaceAbilities}
            onChange={(v) => setRule('allowReplaceAbilities', v)}
            label='Replace abilities'
            description='A card discarded by replacing still fires its ability.'
          />
          <Checkbox
            disabled={ro}
            checked={houseRules?.allowOpponentSnapping ?? true}
            onChange={(v) => setRule('allowOpponentSnapping', v)}
            label='Snap other hands'
            description="Snap a matching card out of another player's hand."
          />
          <Checkbox
            disabled={ro}
            checked={!!houseRules?.snapRace}
            onChange={(v) => setRule('snapRace', v)}
            label='Snap race'
            description='Only the first snap counts. Later snaps take the penalty.'
          />
          <Checkbox
            disabled={ro}
            checked={houseRules?.lockCallerHand ?? true}
            onChange={(v) => setRule('lockCallerHand', v)}
            label="Lock caller's hand"
            description="The Cambia caller's hand is safe from snaps and swaps."
          />
          <Checkbox
            disabled={ro}
            checked={!!houseRules?.forfeitOnDisconnect}
            onChange={(v) => setRule('forfeitOnDisconnect', v)}
            label='Forfeit on disconnect'
            description='A dropped player forfeits instead of rejoining.'
          />
        </div>
      </RuleGroup>

      <RuleGroup
        title='Circuit scoring'
        hint='Rounds accumulate toward a target'
        action={
          <Switch
            disabled={ro}
            checked={!!circuit?.enabled}
            onChange={(v) => { setCircuit((prev) => ({ ...prev, enabled: v })); setSaveStatus('idle'); }}
            label={circuit?.enabled ? 'On' : 'Off'}
          />
        }
      >
        {circuit?.enabled && (
          <div style={FIELD_GRID}>
            <Input
              label='Target score'
              type='number'
              disabled={ro}
              value={num(circuit?.rules?.targetScore)}
              onChange={(e) => setCircuitRule('targetScore', parseInt(e.target.value, 10) || 100)}
            />
            <Input
              label='Win bonus'
              type='number'
              disabled={ro}
              value={num(circuit?.rules?.winBonus)}
              onChange={(e) => setCircuitRule('winBonus', parseInt(e.target.value, 10) || -1)}
            />
            <Input
              label='False Cambia penalty'
              type='number'
              disabled={ro}
              value={num(circuit?.rules?.falseCambiaPenalty)}
              onChange={(e) => setCircuitRule('falseCambiaPenalty', parseInt(e.target.value, 10) || 1)}
            />
          </div>
        )}
      </RuleGroup>

      <div style={{ ...DIVIDER, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
        <Switch
          disabled={ro}
          checked={!!lobbySettings?.autoStart}
          onChange={(v) => { setLobbySettings((prev) => ({ ...prev, autoStart: v })); setSaveStatus('idle'); }}
          label='Auto-start when all ready'
        />
        {isHost ? (
          <Button variant='primary' size='sm' disabled={!hasChanges} onClick={save}>
            {saveStatus === 'saved' ? 'Saved' : 'Save rules'}
          </Button>
        ) : (
          <span style={{ fontSize: 'var(--ds-text-sm)', color: 'var(--text-tertiary)' }}>Host sets the rules</span>
        )}
      </div>
    </Panel>
  );
};

export default DsMatchSettings;
