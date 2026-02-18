// src/components/lobby/LobbySettingsView.tsx
import React from 'react';
import type { LobbyState } from '@/types/index';

interface LobbySettingsViewProps {
	currentSettings: LobbyState;
}

/** Formats a boolean value into a user-friendly string */
const formatBoolean = (value: boolean | undefined): string => {
	return value === true ? 'Enabled' : (value === false ? 'Disabled' : 'N/A');
};

/** Formats a number, handling undefined or null */
const formatNumber = (value: number | undefined | null, unit: string = ''): string => {
	if (value === undefined || value === null) return 'N/A';
	return `${value}${unit}`;
};

const LobbySettingsView: React.FC<LobbySettingsViewProps> = ({ currentSettings }) => {
	const { houseRules, circuit, lobbySettings, settings } = currentSettings;
	const effectiveLobbySettings = lobbySettings ?? settings ?? { autoStart: false }; // Combine sources

	return (
		<div className="bg-white dark:bg-gray-800 shadow rounded-lg p-4 space-y-3 mt-4">
			<h3 className="text-lg font-semibold mb-2 text-gray-800 dark:text-gray-100">Lobby Settings</h3>

			{/* General */}
			<h4 className="text-md font-semibold pt-2 border-t dark:border-gray-700 text-gray-700 dark:text-gray-200">General</h4>
			<div className="flex justify-between text-sm">
				<span className="font-medium text-gray-700 dark:text-gray-300">Auto Start Game:</span>
				<span className="text-gray-600 dark:text-gray-400">{formatBoolean(effectiveLobbySettings?.autoStart)}</span>
			</div>

			{/* House Rules */}
			<h4 className="text-md font-semibold pt-2 border-t dark:border-gray-700 text-gray-700 dark:text-gray-200">House Rules</h4>
			<div className="flex justify-between text-sm">
				<span className="font-medium text-gray-700 dark:text-gray-300">Turn Timer:</span>
				<span className="text-gray-600 dark:text-gray-400">{houseRules?.turnTimerSec === 0 ? 'Disabled' : formatNumber(houseRules?.turnTimerSec, 's')}</span>
			</div>
			<div className="flex justify-between text-sm">
				<span className="font-medium text-gray-700 dark:text-gray-300">Penalty Draw Count:</span>
				<span className="text-gray-600 dark:text-gray-400">{formatNumber(houseRules?.penaltyDrawCount)}</span>
			</div>
			<div className="flex justify-between text-sm">
				<span className="font-medium text-gray-700 dark:text-gray-300">Allow Special on Replace:</span>
				<span className="text-gray-600 dark:text-gray-400">{formatBoolean(houseRules?.allowReplaceAbilities)}</span>
			</div>
			<div className="flex justify-between text-sm">
				<span className="font-medium text-gray-700 dark:text-gray-300">Allow Draw From Discard:</span>
				<span className="text-gray-600 dark:text-gray-400">{formatBoolean(houseRules?.allowDrawFromDiscardPile)}</span>
			</div>
			<div className="flex justify-between text-sm">
				<span className="font-medium text-gray-700 dark:text-gray-300">Snap Race (First Only):</span>
				<span className="text-gray-600 dark:text-gray-400">{formatBoolean(houseRules?.snapRace)}</span>
			</div>

			{/* Circuit Settings */}
			<h4 className="text-md font-semibold pt-2 border-t dark:border-gray-700 text-gray-700 dark:text-gray-200">Circuit Mode</h4>
			<div className="flex justify-between text-sm">
				<span className="font-medium text-gray-700 dark:text-gray-300">Circuit Mode Enabled:</span>
				<span className="text-gray-600 dark:text-gray-400">{formatBoolean(circuit?.enabled)}</span>
			</div>
			{circuit?.enabled && (
				<div className='pl-4 space-y-1 border-l ml-2 dark:border-gray-600'>
					<div className="flex justify-between text-xs">
						<span className="font-medium text-gray-600 dark:text-gray-400">Target Score:</span>
						<span className="text-gray-500 dark:text-gray-500">{formatNumber(circuit?.rules?.targetScore)}</span>
					</div>
					<div className="flex justify-between text-xs">
						<span className="font-medium text-gray-600 dark:text-gray-400">Win Bonus:</span>
						<span className="text-gray-500 dark:text-gray-500">{formatNumber(circuit?.rules?.winBonus)}</span>
					</div>
					<div className="flex justify-between text-xs">
						<span className="font-medium text-gray-600 dark:text-gray-400">False Cambia Penalty:</span>
						<span className="text-gray-500 dark:text-gray-500">{formatNumber(circuit?.rules?.falseCambiaPenalty)}</span>
					</div>
				</div>
			)}
		</div>
	);
};

export default LobbySettingsView;