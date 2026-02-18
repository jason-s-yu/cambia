// src/components/dashboard/CreateLobbyButton.tsx
import React, { useState, useEffect } from 'react';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import Button from '@/components/common/Button';
import Modal from '@/components/common/Modal';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../common/ErrorMessage';
import type { LobbyState } from '@/types/index';

const CreateLobbyButton: React.FC = () => {
	const [isModalOpen, setIsModalOpen] = useState(false);
	const [lobbyType, setLobbyType] = useState<'private' | 'public'>('public');
	const [gameMode, setGameMode] = useState<string>('head_to_head');

	const {
		createAndJoinLobby,
		isLoading,
		error,
		setError,
		currentLobbyId
	} = useCurrentLobbyStore();
	const navigate = useNavigate();
	const previousLobbyIdRef = React.useRef<string | null>(null);


	const handleOpenModal = () => {
		setError(null);
		setIsModalOpen(true);
	};

	const handleCreateLobby = async () => {
		setError(null);
		const settings: Partial<LobbyState> = { type: lobbyType, gameMode };
		console.log('[CreateLobbyButton] Attempting to create lobby with settings: ', settings);
		await createAndJoinLobby(settings);
		// Navigation is handled by the useEffect below
	};

	// Effect to navigate when currentLobbyId changes after creation attempt
	useEffect(() => {
		const previousLobbyId = previousLobbyIdRef.current;
		previousLobbyIdRef.current = currentLobbyId;

		if (isModalOpen && currentLobbyId && currentLobbyId !== previousLobbyId) {
			console.log(`[CreateLobbyButton Effect] Detected store currentLobbyId = ${currentLobbyId}. Navigating...`);
			setIsModalOpen(false);
			navigate(`/lobby/${currentLobbyId}`);
		}
	}, [currentLobbyId, isModalOpen, navigate]);


	const handleCloseModal = () => {
		if (!isLoading) {
			setIsModalOpen(false);
			setError(null);
		}
	};

	return (
		<>
			<Button onClick={handleOpenModal} variant='primary'>
				Create Lobby
			</Button>

			<Modal isOpen={isModalOpen} onClose={handleCloseModal} title="Create New Lobby">
				<div className="space-y-4">
					{!isLoading && error && <ErrorMessage message={error} onClear={() => setError(null)} />}

					<div>
						<label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Lobby Type</label>
						<select
							value={lobbyType}
							onChange={(e) => setLobbyType(e.target.value as 'private' | 'public')}
							disabled={isLoading}
							className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
						>
							<option value="public">Public</option>
							<option value="private">Private</option>
						</select>
					</div>

					<div>
						<label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Game Mode</label>
						<select
							value={gameMode}
							onChange={(e) => setGameMode(e.target.value)}
							disabled={isLoading}
							className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
						>
							<option value="head_to_head">Head to Head (2p)</option>
							<option value="group_of_4">Group of 4 (4p)</option>
						</select>
					</div>

					<div className="flex justify-end space-x-3 pt-4">
						<Button variant="secondary" onClick={handleCloseModal} disabled={isLoading}>
							Cancel
						</Button>
						<Button variant="primary" onClick={handleCreateLobby} isLoading={isLoading} disabled={isLoading}>
							{isLoading ? 'Creating...' : 'Create Lobby'}
						</Button>
					</div>
				</div>
			</Modal>
		</>
	);
};

export default CreateLobbyButton;