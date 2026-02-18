// src/components/game/PlayerHand.tsx
import React from 'react';
import type { ObfCard, ClientGameAction } from '@/types/game';
import Card from './Card'; // Assuming Card component exists

interface PlayerHandProps {
	hand: ObfCard[];
	sendMessage: (action: ClientGameAction) => void;
	// Add props for selection handling later
}

const PlayerHand: React.FC<PlayerHandProps> = ({ hand }) => {
	// TODO: Implement card selection logic for replace/snap/special actions
	// TODO: Add click handlers

	return (
		<div className="flex justify-center items-end space-x-[-20px] p-2 bg-black/20 rounded mb-2">
			{hand.length === 0 && <div className="text-gray-400 italic h-[100px] flex items-center">Empty Hand</div>}
			{hand.map((card) => (
				<div key={card.id} className="transform transition-transform hover:scale-110 hover:-translate-y-2">
					<Card
						id={card.id}
						rank={card.rank}
						suit={card.suit}
						faceUp={card.known} // Player always sees their own cards face up
						className="w-16 h-auto shadow-md" // Adjust size as needed
						// onClick={() => handleCardClick(card)}
					/>
				</div>
			))}
		</div>
	);
};

export default PlayerHand;