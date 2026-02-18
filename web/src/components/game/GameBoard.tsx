// src/components/game/GameBoard.tsx
import React from 'react';
import type { ObfGameState, ClientGameAction } from '@/types/game';
import PlayerHand from './PlayerHand'; // To be created
import OpponentHand from './OpponentHand'; // To be created
import Deck from './Deck'; // To be created
import DiscardPile from './DiscardPile'; // To be created
import GameInfoPanel from './GameInfoPanel'; // To be created
import ActionControls from './ActionControls'; // To be created
import DrawnCardDisplay from './DrawnCardDisplay'; // To be created
import { useGameStore } from '@/stores/gameStore';
import { useAuthStore } from '@/stores/authStore';

interface GameBoardProps {
	gameState: ObfGameState;
	sendMessage: (action: ClientGameAction) => void;
}

const GameBoard: React.FC<GameBoardProps> = ({ gameState, sendMessage }) => {
	const selfId = useAuthStore((state) => state.user?.id);
	const displayedDrawnCard = useGameStore((state) => state.displayedDrawnCard);
	const clearDisplayedDrawnCard = useGameStore((state) => state.clearDisplayedDrawnCard);

	// Find self and opponents
	const selfState = gameState.players.find(p => p.playerId === selfId);
	const opponents = gameState.players.filter(p => p.playerId !== selfId);

	if (!selfState) {
		// This shouldn't happen if the player is connected to the game
		return <div className='text-red-500'>Error: Could not find player state for self.</div>;
	}

	// Basic layout logic (replace with more sophisticated table layout)
	// Example: Arrange opponents around the center
	const getOpponentPosition = (index: number, totalOpponents: number) => {
		// Simple linear layout for now
		// TODO: Implement circular/table layout
		const positions = [
			'top-0 left-1/2 -translate-x-1/2', // Top center
			'top-1/2 left-0 -translate-y-1/2', // Middle left
			'top-1/2 right-0 -translate-y-1/2' // Middle right
		];
		return positions[index % positions.length] || 'top-0 right-0'; // Fallback
	};

	return (
		<div className="relative w-full max-w-4xl aspect-[4/3] bg-green-700 dark:bg-green-900 border-4 border-yellow-700 dark:border-yellow-500 rounded-lg p-4 shadow-lg flex flex-col items-center justify-between">

			{/* Opponents */}
			<div className="absolute inset-0">
				{opponents.map((opponent, index) => (
					<div key={opponent.playerId} className={`absolute ${getOpponentPosition(index, opponents.length)} m-2`}>
						<OpponentHand playerState={opponent} />
					</div>
				))}
			</div>

			{/* Center Area (Deck, Discard, Info) */}
			<div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center gap-4 z-10">
				<GameInfoPanel
					currentPlayerId={gameState.currentPlayerId}
					players={gameState.players}
					turnId={gameState.turnId}
				/>
				<div className="flex gap-4">
					<Deck count={gameState.stockpileSize} />
					<DiscardPile card={gameState.discardTop} />
				</div>
			</div>

			{/* Self Hand and Controls */}
			<div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-full flex flex-col items-center p-2">
				<PlayerHand hand={selfState.revealedHand ?? []} sendMessage={sendMessage} />
				<ActionControls sendMessage={sendMessage} />
			</div>

			{/* Magnified Drawn Card Display */}
			{displayedDrawnCard && (
				<DrawnCardDisplay
					card={displayedDrawnCard}
					onClose={clearDisplayedDrawnCard}
				/>
			)}
		</div>
	);
};

export default GameBoard;