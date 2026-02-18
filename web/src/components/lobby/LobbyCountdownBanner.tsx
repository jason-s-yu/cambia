// src/components/lobby/LobbyCountdownBanner.tsx
import React, { useState, useEffect } from 'react';
import { useCurrentLobbyStore } from '@/stores/lobbyStore'; // Import the main store hook

const LobbyCountdownBanner: React.FC = () => {
	// Select primitive values directly from the store
	const startTime = useCurrentLobbyStore((state) => state.countdownStartTime);
	const duration = useCurrentLobbyStore((state) => state.countdownDuration);
	const [remaining, setRemaining] = useState<number | null>(null);

	useEffect(() => {
		let intervalId: number | undefined;

		if (startTime && duration && duration > 0) {
			const endTime = startTime + duration * 1000;

			const updateRemaining = () => {
				const now = Date.now();
				const newRemaining = Math.max(0, Math.ceil((endTime - now) / 1000));
				setRemaining(newRemaining);

				if (newRemaining === 0) {
					clearInterval(intervalId);
				}
			};

			updateRemaining(); // Initial update
			intervalId = setInterval(updateRemaining, 1000); // Update every second

		} else {
			setRemaining(null); // Clear remaining time if countdown stops
		}

		return () => {
			if (intervalId) {
				clearInterval(intervalId);
			}
		};
	}, [startTime, duration]);

	if (remaining === null || remaining <= 0) {
		return null; // Don't render if no active countdown or it finished
	}

	return (
		<div className="bg-blue-100 dark:bg-blue-900 border border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-200 px-4 py-2 rounded-md text-center mb-4 shadow">
			<p className="font-semibold">
				Game starting in: <span className="text-xl font-bold">{remaining}</span> second{remaining !== 1 ? 's' : ''}...
			</p>
		</div>
	);
};

export default LobbyCountdownBanner;