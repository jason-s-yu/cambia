// src/components/lobby/DsMatchSettings.tsx
// DS-styled match-settings panel for the live lobby (cambia-484). Re-skins the
// legacy LobbySettingsPanel / LobbySettingsView pair: hosts edit a local buffer
// and Save emits the exact `update_rules` WS message the legacy panel sent
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

interface DsMatchSettingsProps {
  currentSettings: LobbyState;
  isHost: boolean;
  sendMessage: (message: { type: string; body?: unknown }) => void;
}

function jsonEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

const CAPS_LABEL: React.CSSProperties = {
  fontSize: 'var(--text-2xs)',
  fontWeight: 'var(--weight-black)',
  letterSpacing: 'var(--tracking-caps)',
  textTransform: 'uppercase',
  color: 'var(--text-tertiary)'
};

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

  return (
    <Panel
      title={isHost ? 'Match settings · host' : 'Match settings'}
      action={<Badge tone='info'>{currentSettings.gameMode || 'unknown'}</Badge>}
    >
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 14, marginBottom: 16 }}>
        <Input
          label='Turn timer (sec)'
          mono
          type='number'
          disabled={ro}
          value={num(houseRules?.turnTimerSec)}
          onChange={(e) => setRule('turnTimerSec', parseInt(e.target.value, 10) || 0)}
        />
        <Input
          label='Penalty draw count'
          mono
          type='number'
          disabled={ro}
          value={num(houseRules?.penaltyDrawCount)}
          onChange={(e) => setRule('penaltyDrawCount', parseInt(e.target.value, 10) || 0)}
        />
      </div>

      <div style={{ ...CAPS_LABEL, margin: '2px 0 10px' }}>House rules</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(190px, 1fr))', gap: '12px 18px' }}>
        <Checkbox
          disabled={ro}
          checked={!!houseRules?.allowDrawFromDiscardPile}
          onChange={(v) => setRule('allowDrawFromDiscardPile', v)}
          label='Draw from discard pile'
          description='allowDrawFromDiscardPile'
        />
        <Checkbox
          disabled={ro}
          checked={!!houseRules?.allowReplaceAbilities}
          onChange={(v) => setRule('allowReplaceAbilities', v)}
          label='Replace abilities'
          description='allowReplaceAbilities'
        />
        <Checkbox
          disabled={ro}
          checked={houseRules?.allowOpponentSnapping ?? true}
          onChange={(v) => setRule('allowOpponentSnapping', v)}
          label='Opponent snapping'
          description='allowOpponentSnapping'
        />
        <Checkbox
          disabled={ro}
          checked={!!houseRules?.snapRace}
          onChange={(v) => setRule('snapRace', v)}
          label='Snap race'
          description='snapRace — only the first snap wins'
        />
        <Checkbox
          disabled={ro}
          checked={!!houseRules?.forfeitOnDisconnect}
          onChange={(v) => setRule('forfeitOnDisconnect', v)}
          label='Forfeit on disconnect'
          description='forfeitOnDisconnect'
        />
      </div>

      <div style={{ borderTop: '1.5px solid var(--border-subtle)', marginTop: 16, paddingTop: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
          <span style={CAPS_LABEL}>Circuit scoring</span>
          <Switch
            disabled={ro}
            checked={!!circuit?.enabled}
            onChange={(v) => { setCircuit((prev) => ({ ...prev, enabled: v })); setSaveStatus('idle'); }}
            label={circuit?.enabled ? 'Enabled' : 'Off'}
          />
        </div>
        {circuit?.enabled && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 14 }}>
            <Input
              label='Target score'
              mono
              type='number'
              disabled={ro}
              value={num(circuit?.rules?.targetScore)}
              onChange={(e) => setCircuitRule('targetScore', parseInt(e.target.value, 10) || 100)}
            />
            <Input
              label='Win bonus'
              mono
              type='number'
              disabled={ro}
              value={num(circuit?.rules?.winBonus)}
              onChange={(e) => setCircuitRule('winBonus', parseInt(e.target.value, 10) || -1)}
            />
            <Input
              label='False Cambia penalty'
              mono
              type='number'
              disabled={ro}
              value={num(circuit?.rules?.falseCambiaPenalty)}
              onChange={(e) => setCircuitRule('falseCambiaPenalty', parseInt(e.target.value, 10) || 1)}
            />
          </div>
        )}
      </div>

      <div style={{ borderTop: '1.5px solid var(--border-subtle)', marginTop: 16, paddingTop: 14, display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
        <Switch
          disabled={ro}
          checked={!!lobbySettings?.autoStart}
          onChange={(v) => { setLobbySettings((prev) => ({ ...prev, autoStart: v })); setSaveStatus('idle'); }}
          label='Auto start when everyone is ready'
        />
        {isHost && (
          <Button variant='secondary' size='sm' disabled={!hasChanges} onClick={save}>
            {saveStatus === 'saved' ? 'Saved' : 'Save settings'}
          </Button>
        )}
      </div>
    </Panel>
  );
};

export default DsMatchSettings;
