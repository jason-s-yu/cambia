// src/pages/GamePage.tsx
import React from 'react';
import { useParams } from 'react-router-dom';
import { useGameSocket } from '@/hooks/useGameSocket';
import { useGameStore, selectGameState, selectIsLoading, selectGameError } from '@/stores/gameStore';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import ErrorMessage from '@/components/common/ErrorMessage';
import GameBoard from '@/components/game/GameBoard'; // Create this next

const GamePage: React.FC = () => {
	const { gameId } = useParams<{ gameId: string }>();
	const { sendMessage, closeSocket, isConnected } = useGameSocket(gameId); // Initialize socket hook
	const gameState = useGameStore(selectGameState);
	const isLoading = useGameStore(selectIsLoading);
	const error = useGameStore(selectGameError);
	const clearError = useGameStore(state => state.clearError);

	// Handle cleanup on unmount? Maybe not needed if hook handles it.
	// useEffect(() => {
	//   return () => {
	//     closeSocket();
	//     // clearGameState(); // Decide if state should clear immediately on leaving page
	//   };
	// }, [closeSocket, clearGameState]);

	if (isLoading && !gameState) {
		return (
			<div className="flex flex-col items-center justify-center h-full pt-10">
				<LoadingSpinner />
				<p className="mt-2 text-gray-600 dark:text-gray-400">Connecting to game {gameId?.substring(0, 8)}...</p>
			</div>
		);
	}

	if (error) {
		return (
			<div className="text-center pt-10">
				<ErrorMessage message={`Game Error: ${error}`} onClear={clearError} />
				{/* Add button to go back to dashboard? */}
			</div>
		);
	}

	if (!isConnected || !gameState) {
		// Show loading or specific message if connection failed silently
		return (
			<div className="flex flex-col items-center justify-center h-full pt-10">
				<p className="text-yellow-500 dark:text-yellow-400 mb-2">Connecting...</p>
				<LoadingSpinner size="sm" />
				{/* Add a retry button? */}
			</div>
		);
	}

	// Once connected and gameState is available
	return (
		<div className="w-full h-full flex flex-col items-center justify-center">
			<h2 className="text-xl font-semibold mb-4 text-center">Game: {gameState.gameId.substring(0, 8)}</h2>
			<GameBoard gameState={gameState} sendMessage={sendMessage} />
			{/* TODO: Add DrawnCardDisplay component */}
		</div>
	);
};

export default GamePage;