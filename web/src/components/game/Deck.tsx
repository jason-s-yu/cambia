// src/components/game/Deck.tsx
import React from 'react';
import Card from './Card'; // Assuming Card component exists

interface DeckProps {
	count: number;
}

const Deck: React.FC<DeckProps> = ({ count }) => {
	return (
		<div className="relative flex flex-col items-center">
			<span className="text-xs text-white mb-1">Deck</span>
			<div className='w-16 h-24'> {/* Fixed size container */}
				{count > 0 ? (
					<Card
						id="deck-card"
						faceUp={false}
						className="w-full h-full shadow-lg border border-white/50"
						// onClick={handleDrawStockpile} // Add later via ActionControls
					/>
				) : (
					<div className="w-full h-full bg-black/30 rounded border border-dashed border-white/50 flex items-center justify-center text-white text-xs italic">Empty</div>
				)}
			</div>
			<span className="absolute bottom-[-16px] text-center text-[10px] font-bold text-white bg-black/50 rounded px-1 py-0.5">
				{count}
			</span>
		</div>
	);
};

export default Deck;