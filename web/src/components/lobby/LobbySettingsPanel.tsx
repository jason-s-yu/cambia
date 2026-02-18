/* eslint-disable @typescript-eslint/no-explicit-any */
// web/src/components/lobby/LobbySettingsPanel.tsx
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import type { LobbyState, HouseRules, CircuitSettings, LobbySettings } from '@/types';
import Button from '../common/Button';

interface LobbySettingsPanelProps {
	currentSettings: LobbyState;
	sendMessage: (message: { type: string; rules?: any; }) => void;
}

function deepEqual(obj1: any, obj2: any): boolean {
	return JSON.stringify(obj1) === JSON.stringify(obj2);
}

const LobbySettingsPanel: React.FC<LobbySettingsPanelProps> = ({ currentSettings, sendMessage }) => {
	// Initialize state based on potentially different keys from props
	const initialLobbySettings = currentSettings.lobbySettings ?? currentSettings.settings ?? { autoStart: false };

	const [editedHouseRules, setEditedHouseRules] = useState<HouseRules>(currentSettings.houseRules);
	const [editedCircuit, setEditedCircuit] = useState<CircuitSettings>(currentSettings.circuit);
	const [editedLobbySettings, setEditedLobbySettings] = useState<LobbySettings>(initialLobbySettings);
	const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

	useEffect(() => {
		// When the underlying store data changes, reset local edits
		setEditedHouseRules(currentSettings.houseRules);
		setEditedCircuit(currentSettings.circuit);
		setEditedLobbySettings(currentSettings.lobbySettings ?? currentSettings.settings ?? { autoStart: false });
		setSaveStatus('idle');
	}, [currentSettings]);


	const handleHouseRuleChange = useCallback(<K extends keyof HouseRules>(key: K, value: HouseRules[K]) => {
		setEditedHouseRules((prev: HouseRules) => ({ ...prev, [key]: value }));
		setSaveStatus('idle');
	}, []);

	const handleCircuitRuleChange = useCallback(<K extends keyof CircuitSettings['rules']>(key: K, value: CircuitSettings['rules'][K]) => {
		setEditedCircuit((prev: CircuitSettings) => ({
			...prev,
			rules: {
				...prev.rules,
				[key]: value,
			},
		}));
		setSaveStatus('idle');
	}, []);

	const handleCircuitEnabledChange = useCallback((enabled: boolean) => {
		setEditedCircuit((prev: CircuitSettings) => ({ ...prev, enabled }));
		setSaveStatus('idle');
	}, []);

	const handleLobbySettingChange = useCallback(<K extends keyof LobbySettings>(key: K, value: LobbySettings[K]) => {
		setEditedLobbySettings((prev: LobbySettings) => ({ ...prev, [key]: value }));
		setSaveStatus('idle');
	}, []);


	const handleSaveChanges = () => {
		const rulesPayload = {
			houseRules: editedHouseRules,
			circuit: editedCircuit,
			settings: editedLobbySettings // Use 'settings' key for WS message based on lobby_actions.md
		};

		console.log('[LobbySettingsPanel] Sending update_rules message with payload:', rulesPayload);
		setSaveStatus('saving');
		try {
			sendMessage({ type: 'update_rules', rules: rulesPayload });
			setSaveStatus('saved');
			setTimeout(() => setSaveStatus('idle'), 2000);
		} catch (error) {
			console.error('Failed to send update_rules message:', error);
			setSaveStatus('error');
		}
	};

	const hasChanges = useMemo(() => {
		const currentLobbyEffective = currentSettings.lobbySettings ?? currentSettings.settings ?? { autoStart: false };
		return !deepEqual(editedHouseRules, currentSettings.houseRules) ||
			!deepEqual(editedCircuit, currentSettings.circuit) ||
			!deepEqual(editedLobbySettings, currentLobbyEffective);
	}, [editedHouseRules, editedCircuit, editedLobbySettings, currentSettings]);

	// Render logic remains the same...
	return (
		<div className="bg-white dark:bg-gray-800 shadow rounded-lg p-4 space-y-3">
			<h3 className="text-lg font-semibold mb-2 text-gray-800 dark:text-gray-100">Lobby Settings</h3>

			{/* General */}
			<h4 className="text-md font-semibold pt-2 border-t dark:border-gray-700 text-gray-700 dark:text-gray-200">General</h4>
			<div className="flex items-center justify-between">
				<label htmlFor="autoStart" className="text-sm font-medium text-gray-700 dark:text-gray-300">
					Auto Start Game When Ready
				</label>
				<input
					id="autoStart"
					type="checkbox"
					checked={editedLobbySettings.autoStart}
					onChange={(e) => handleLobbySettingChange('autoStart', e.target.checked)}
					className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
				/>
			</div>

			{/* House Rules */}
			<h4 className="text-md font-semibold pt-2 border-t dark:border-gray-700 text-gray-700 dark:text-gray-200">House Rules</h4>
			<div className="flex items-center justify-between">
				<label htmlFor="turnTimerSec" className="text-sm font-medium text-gray-700 dark:text-gray-300" title="Seconds per turn (0 to disable)">
					Turn Timer (sec)
				</label>
				<input
					id="turnTimerSec"
					type="number"
					min="0"
					max="120"
					value={editedHouseRules.turnTimerSec ?? 0}
					onChange={(e) => handleHouseRuleChange('turnTimerSec', parseInt(e.target.value, 10) || 0)}
					className="w-20 px-2 py-1 border border-gray-300 rounded-md text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white"
				/>
			</div>
			<div className="flex items-center justify-between">
				<label htmlFor="penaltyDrawCount" className="text-sm font-medium text-gray-700 dark:text-gray-300" title="Cards drawn on incorrect snap">
					Penalty Draw Count
				</label>
				<input
					id="penaltyDrawCount"
					type="number"
					min="0"
					max="5"
					value={editedHouseRules.penaltyDrawCount}
					onChange={(e) => handleHouseRuleChange('penaltyDrawCount', parseInt(e.target.value, 10) || 0)}
					className="w-20 px-2 py-1 border border-gray-300 rounded-md text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white"
				/>
			</div>
			<div className="flex items-center justify-between">
				<label htmlFor="allowReplaceAbilities" className="text-sm font-medium text-gray-700 dark:text-gray-300" title="Allow special card ability when replacing instead of discarding">
					Allow Special Ability on Replace
				</label>
				<input
					id="allowReplaceAbilities"
					type="checkbox"
					checked={editedHouseRules.allowReplaceAbilities}
					onChange={(e) => handleHouseRuleChange('allowReplaceAbilities', e.target.checked)}
					className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
				/>
			</div>
			<div className="flex items-center justify-between">
				<label htmlFor="allowDrawFromDiscardPile" className="text-sm font-medium text-gray-700 dark:text-gray-300" title="Allow drawing the top card from discard pile">
					Allow Draw From Discard
				</label>
				<input
					id="allowDrawFromDiscardPile"
					type="checkbox"
					checked={editedHouseRules.allowDrawFromDiscardPile}
					onChange={(e) => handleHouseRuleChange('allowDrawFromDiscardPile', e.target.checked)}
					className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
				/>
			</div>
			<div className="flex items-center justify-between">
				<label htmlFor="snapRace" className="text-sm font-medium text-gray-700 dark:text-gray-300" title="Only the first player to snap successfully gets it">
					Snap Race (First Only)
				</label>
				<input
					id="snapRace"
					type="checkbox"
					checked={editedHouseRules.snapRace}
					onChange={(e) => handleHouseRuleChange('snapRace', e.target.checked)}
					className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
				/>
			</div>


			{/* Circuit Settings */}
			<h4 className="text-md font-semibold pt-2 border-t dark:border-gray-700 text-gray-700 dark:text-gray-200">Circuit Mode</h4>
			<div className="flex items-center justify-between">
				<label htmlFor="circuitEnabled" className="text-sm font-medium text-gray-700 dark:text-gray-300">
					Enable Circuit Mode
				</label>
				<input
					id="circuitEnabled"
					type="checkbox"
					checked={editedCircuit.enabled}
					onChange={(e) => handleCircuitEnabledChange(e.target.checked)}
					className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
				/>
			</div>
			{editedCircuit.enabled && (
				<div className='pl-4 space-y-2 border-l ml-2 dark:border-gray-600'>
					<div className="flex items-center justify-between">
						<label htmlFor="targetScore" className="text-xs font-medium text-gray-600 dark:text-gray-400">Target Score</label>
						<input
							id="targetScore" type="number" min="10" max="500"
							value={editedCircuit.rules.targetScore}
							onChange={(e) => handleCircuitRuleChange('targetScore', parseInt(e.target.value, 10) || 100)}
							className="w-20 px-2 py-1 border border-gray-300 rounded-md text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white"
						/>
					</div>
					<div className="flex items-center justify-between">
						<label htmlFor="winBonus" className="text-xs font-medium text-gray-600 dark:text-gray-400">Win Bonus</label>
						<input
							id="winBonus" type="number" min="-10" max="0"
							value={editedCircuit.rules.winBonus}
							onChange={(e) => handleCircuitRuleChange('winBonus', parseInt(e.target.value, 10) || -1)}
							className="w-20 px-2 py-1 border border-gray-300 rounded-md text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white"
						/>
					</div>
					<div className="flex items-center justify-between">
						<label htmlFor="falseCambiaPenalty" className="text-xs font-medium text-gray-600 dark:text-gray-400">False Cambia Penalty</label>
						<input
							id="falseCambiaPenalty" type="number" min="0" max="20"
							value={editedCircuit.rules.falseCambiaPenalty}
							onChange={(e) => handleCircuitRuleChange('falseCambiaPenalty', parseInt(e.target.value, 10) || 1)}
							className="w-20 px-2 py-1 border border-gray-300 rounded-md text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white"
						/>
					</div>
				</div>
			)}

			{/* Save Button */}
			<div className='pt-3 border-t dark:border-gray-700 mt-3'>
				<Button
					onClick={handleSaveChanges}
					variant="primary"
					size='sm'
					className="w-full justify-center"
					disabled={!hasChanges || saveStatus === 'saving'}
					isLoading={saveStatus === 'saving'}
				>
					{saveStatus === 'saved' ? 'Settings Saved!' : (hasChanges ? 'Save Settings Changes' : 'Settings Saved')}
				</Button>
				{saveStatus === 'error' && <p className="text-red-500 text-xs mt-1 text-center">Failed to save settings.</p>}
			</div>
		</div>
	);
};

export default LobbySettingsPanel;