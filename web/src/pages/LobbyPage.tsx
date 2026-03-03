// src/pages/LobbyPage.tsx
import React, { useEffect, useMemo, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import { useSocket } from '@/hooks/useSocket';
import { useGameStore, selectGameState } from '@/stores/gameStore';
import GameBoard from '@/components/game/GameBoard';
import PlayerList from '@/components/lobby/PlayerList';
import ChatWindow from '@/components/lobby/ChatWindow';
import ReadyButton from '@/components/lobby/ReadyButton';
import HostControls from '@/components/lobby/HostControls';
import LobbySettingsPanel from '@/components/lobby/LobbySettingsPanel';
import LobbySettingsView from '@/components/lobby/LobbySettingsView';
import LobbyCountdownBanner from '@/components/lobby/LobbyCountdownBanner';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import ErrorMessage from '@/components/common/ErrorMessage';
import Button from '@/components/common/Button';

// Memoize child components
const MemoizedPlayerList = React.memo(PlayerList);
const MemoizedChatWindow = React.memo(ChatWindow);
const MemoizedLobbySettingsPanel = React.memo(LobbySettingsPanel);
const MemoizedLobbySettingsView = React.memo(LobbySettingsView);
const MemoizedReadyButton = React.memo(ReadyButton);
const MemoizedHostControls = React.memo(HostControls);
const MemoizedCountdownBanner = React.memo(LobbyCountdownBanner);

const LobbyPage: React.FC = () => {
	const { lobbyId: urlLobbyId } = useParams<{ lobbyId: string }>();
	const navigate = useNavigate();
	const hasAttemptedConnectionRef = useRef(false);

	// --- State Selectors ---
	const {
		currentLobbyId: storeLobbyId,
		lobbyDetails,
		isLoading: isStoreLoading,
		error: storeError,
		chatMessages,
		isConnected,
		setCurrentLobbyId,
		leaveLobby,
		clearError: clearStoreError,
		countdownStartTime,
		countdownDuration,
		phase,
	} = useCurrentLobbyStore();

	const gameState = useGameStore(selectGameState);

	// --- Validate URL Lobby ID ---
	const isValidLobbyId = useMemo(() => urlLobbyId && typeof urlLobbyId === 'string' && urlLobbyId.length > 5, [urlLobbyId]);

	// --- WebSocket Hook ---
	const { sendMessage, closeSocket } = useSocket(isValidLobbyId ? urlLobbyId : null);

	// --- Effect to Synchronize URL ID with Store ID & Trigger Connection ---
	useEffect(() => {
		if (isValidLobbyId && urlLobbyId) {
			if (urlLobbyId !== storeLobbyId) {
				hasAttemptedConnectionRef.current = false;
				setCurrentLobbyId(urlLobbyId);
			} else if (!isConnected && !isStoreLoading && !hasAttemptedConnectionRef.current) {
				hasAttemptedConnectionRef.current = true;
			}
		} else {
			if (storeLobbyId !== null) {
				setCurrentLobbyId(null);
			}
		}
	}, [urlLobbyId, isValidLobbyId, storeLobbyId, isConnected, isStoreLoading, setCurrentLobbyId]);

	// --- Derived State ---
	const players = useMemo(() => lobbyDetails?.lobby_status?.users ?? [], [lobbyDetails]);
	const isHost = useMemo(() => lobbyDetails?.your_is_host ?? false, [lobbyDetails]);
	const lobbyShortId = useMemo(() => urlLobbyId ? urlLobbyId.substring(0, 8) : '...', [urlLobbyId]);

	// --- Event Handlers ---
	const handleLeaveLobbyClick = () => {
		closeSocket();
		leaveLobby();
		navigate('/dashboard', { replace: true });
	};

	// --- Redirect Logic ---
	const shouldRedirectToDash = !isValidLobbyId || (!!storeError && !isStoreLoading && !isConnected);

	useEffect(() => {
		if (shouldRedirectToDash) {
			if (storeError) clearStoreError();
			closeSocket();
			leaveLobby();
			navigate('/dashboard', { replace: true });
		}
	}, [shouldRedirectToDash, isValidLobbyId, storeError, isStoreLoading, isConnected, navigate, clearStoreError, closeSocket, leaveLobby]);

	// --- Loading State ---
	const isLoading = isStoreLoading || (storeLobbyId === urlLobbyId && isValidLobbyId && !isConnected && !storeError);

	// --- Render Logic ---
	if (!isValidLobbyId) return <div className="flex items-center justify-center h-screen"><LoadingSpinner /></div>;
	if (isLoading) return (
		<div className="flex flex-col items-center justify-center h-full pt-10">
			<LoadingSpinner />
			<p className="mt-2 text-gray-600 dark:text-gray-400">
				{isStoreLoading ? 'Processing...' : `Connecting to lobby ${lobbyShortId}...`}
			</p>
		</div>
	);
	if (storeError) return (
		<div className="text-center pt-10">
			<ErrorMessage message={storeError} onClear={clearStoreError} />
			<Button onClick={handleLeaveLobbyClick} className="mt-4">Back to Dashboard</Button>
		</div>
	);
	if (isConnected && !lobbyDetails) return (
		<div className="flex flex-col items-center justify-center h-full pt-10">
			<LoadingSpinner />
			<p className="mt-2 text-gray-600 dark:text-gray-400">Waiting for lobby data...</p>
			<Button onClick={handleLeaveLobbyClick} className="mt-4" variant="secondary">Leave Lobby</Button>
		</div>
	);
	if (!isConnected && !isLoading && !lobbyDetails) return (
		<div className="flex flex-col items-center justify-center h-full pt-10">
			<p className="text-yellow-600 dark:text-yellow-400 mb-4">Attempting to connect to lobby...</p>
			<LoadingSpinner size="sm" />
		</div>
	);
	if (!lobbyDetails) return (
		<div className="flex flex-col items-center justify-center h-full pt-10">
			<ErrorMessage message="Lobby data is missing. Please try rejoining." />
			<Button onClick={handleLeaveLobbyClick} className="mt-4">Back to Dashboard</Button>
		</div>
	);

	// --- Phase: In Game ---
	if (phase === 'in_game' && gameState) {
		return (
			<div className="w-full h-full flex flex-col items-center justify-center">
				<h2 className="text-xl font-semibold mb-4 text-center text-gray-800 dark:text-gray-100">
					Game: {lobbyShortId}
				</h2>
				<GameBoard gameState={gameState} sendMessage={sendMessage} />
			</div>
		);
	}

	// --- Phase: Post Game ---
	if (phase === 'post_game') {
		return (
			<div className="flex flex-col items-center justify-center h-full pt-10">
				<h2 className="text-2xl font-semibold mb-4 text-gray-800 dark:text-gray-100">Game Over</h2>
				<p className="text-gray-600 dark:text-gray-400 mb-6">Returning to lobby...</p>
				<Button onClick={() => useCurrentLobbyStore.getState().setPhase('open')} variant="secondary">
					Back to Lobby
				</Button>
			</div>
		);
	}

	// --- Main Lobby UI Render ---
	return (
		<div className="flex flex-col md:flex-row gap-4 h-full max-h-[calc(100vh-150px)]">
			{/* Left Panel */}
			<div className="md:w-1/3 flex flex-col gap-4 overflow-y-auto p-1">
				<div className='flex justify-between items-center sticky top-0 bg-gray-100 dark:bg-gray-800 p-2 z-10 rounded-t-lg shadow-sm'>
					<h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 truncate" title={`Lobby ID: ${lobbyDetails.id}`}>Lobby: {lobbyShortId}</h2>
					<Button onClick={handleLeaveLobbyClick} variant='secondary' size='sm'>Leave Lobby</Button>
				</div>
				{countdownStartTime && countdownDuration ? (
					<MemoizedCountdownBanner />
				) : null}
				<MemoizedPlayerList players={players} />
				{isHost ? (
					<MemoizedLobbySettingsPanel currentSettings={lobbyDetails} sendMessage={sendMessage} />
				) : (
					<MemoizedLobbySettingsView currentSettings={lobbyDetails} />
				)}
				<MemoizedReadyButton sendMessage={sendMessage} players={players} />
				{isHost && <MemoizedHostControls sendMessage={sendMessage} players={players} />}
			</div>
			{/* Right Panel (Chat) */}
			<div className="md:w-2/3 flex flex-col h-full">
				<MemoizedChatWindow messages={chatMessages} sendMessage={sendMessage} />
			</div>
		</div>
	);
};

export default LobbyPage;
