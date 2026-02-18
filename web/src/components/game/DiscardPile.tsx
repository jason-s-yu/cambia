// src/components/game/DiscardPile.tsx
import React from 'react';
import type { ObfCard } from '@/types/game';
import Card from './Card'; // Assuming Card component exists

interface DiscardPileProps {
	card: ObfCard | null | undefined; // Top card, null/undefined if empty
}

const DiscardPile: React.FC<DiscardPileProps> = ({ card }) => {
	return (
		<div className="relative flex flex-col items-center">
			<span className="text-xs text-white mb-1">Discard</span>
			<div className='w-16 h-24'> {/* Fixed size container */}
				{card ? (
					<Card
						id={card.id}
						rank={card.rank}
						suit={card.suit}
						faceUp={true} // Discard top always face up
						className="w-full h-full shadow-lg"
						// onClick={handleDrawDiscard} // Add later via ActionControls
					/>
				) : (
					<div className="w-full h-full bg-black/30 rounded border border-dashed border-white/50 flex items-center justify-center text-white text-xs italic">Empty</div>
				)}
			</div>
		</div>
	);
};

export default DiscardPile;