// src/components/dashboard/LobbyList.tsx
import React, { useEffect, useState } from 'react';
import { useLobbyListStore } from '@/stores/lobbyStore';
import { useNavigate } from 'react-router-dom';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import ErrorMessage from '@/components/common/ErrorMessage';
import Button from '../common/Button';
import { joinLobby } from '@/services/lobbyService';

const LobbyList: React.FC = () => {
	const { lobbies, isLoading, error, fetchLobbies, clearError } = useLobbyListStore();
	const navigate = useNavigate();
	const [joinError, setJoinError] = useState<string | null>(null);

	useEffect(() => {
		fetchLobbies();
		// setup interval polling
		// const intervalId = setInterval(fetchLobbies, 15000);
		// return () => clearInterval(intervalId);
	}, [fetchLobbies]);

	const publicLobbies = Object.values(lobbies).filter(entry => entry.lobby?.type === 'public');

	const handleJoin = async (lobbyId: string) => {
		setJoinError(null);
		try {
			await joinLobby(lobbyId);
			navigate(`/lobby/${lobbyId}`);
		} catch (err: any) {
			setJoinError(err.response?.data?.message || err.message || 'Failed to join lobby.');
		}
	};

	return (
		<div className="bg-white dark:bg-gray-800 shadow rounded-lg p-4 min-h-[10rem]">
			<ErrorMessage message={error} onClear={clearError} />
			<ErrorMessage message={joinError} onClear={() => setJoinError(null)} />
			{isLoading && (
				<div className="flex justify-center items-center h-40">
					<LoadingSpinner />
					<p className='ml-2 text-gray-500 dark:text-gray-400'>Loading lobbies...</p>
				</div>
			)}
			{!isLoading && error && (
				<div className="text-center py-4">
					<p className="text-red-500">Could not load lobbies.</p>
					<Button onClick={fetchLobbies} variant="secondary" size="sm" className="mt-2">Retry</Button>
				</div>
			)}
			{!isLoading && !error && publicLobbies.length === 0 && (
				<p className="text-gray-500 dark:text-gray-400 text-center py-4">No public lobbies found. Create one!</p>
			)}
			{!isLoading && !error && publicLobbies.length > 0 && (
				<ul className="space-y-3 max-h-96 overflow-y-auto pr-2">
					{publicLobbies.map((entry) => {
						const lobby = entry.lobby;
						const playerCount = entry.playerCount;
						const maxPlayers = entry.maxPlayers;

						return (
							<li key={lobby.id} className="flex justify-between items-center p-3 bg-gray-50 dark:bg-gray-700 rounded hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors">
								<div className='truncate mr-2'>
									<span className="font-medium text-gray-800 dark:text-gray-100">Lobby {lobby.id.substring(0, 6)}...</span>
									<span className="text-sm text-gray-500 dark:text-gray-400 ml-2 capitalize">({lobby.gameMode?.replace(/_/g, ' ') || 'Unknown Mode'})</span>
									<span className="text-sm text-gray-500 dark:text-gray-400 ml-2">
										({playerCount}/{maxPlayers} players)
									</span>
								</div>
								<Button size='sm' variant='secondary' className='shrink-0' disabled={playerCount >= maxPlayers} onClick={() => handleJoin(lobby.id)}>
									{playerCount >= maxPlayers ? 'Full' : 'Join'}
								</Button>
							</li>
						);
					})}
				</ul>
			)}
		</div>
	);
};

export default LobbyList;