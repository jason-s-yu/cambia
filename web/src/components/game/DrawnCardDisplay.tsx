// src/components/game/DrawnCardDisplay.tsx
import React, { useEffect, useState } from 'react';
import type { ObfCard } from '@/types/game';
import Card from './Card';

interface DrawnCardDisplayProps {
	card: ObfCard;
	onClose: () => void; // Callback to signal display can be closed (e.g., after animation)
}

const DrawnCardDisplay: React.FC<DrawnCardDisplayProps> = ({ card, onClose }) => {
	const [isVisible, setIsVisible] = useState(false);

	// Effect for fade-in animation
	useEffect(() => {
		setIsVisible(true); // Trigger fade-in
		// Optional: Auto-close after a delay if needed, or rely on game state changes
		// const timer = setTimeout(() => {
		//   handleClose();
		// }, 3000); // Example: close after 3 seconds
		// return () => clearTimeout(timer);
	}, [card]); // Rerun effect if the card prop changes

	const handleClose = () => {
		setIsVisible(false);
		// Call onClose after the fade-out animation completes
		setTimeout(onClose, 300); // Match transition duration
	};

	// Use a simple overlay for now
	return (
		<div
			className={`fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm transition-opacity duration-300 ${isVisible ? 'opacity-100' : 'opacity-0'}`}
			onClick={handleClose} // Close on overlay click
		>
			<div
				className={`transform transition-transform duration-300 ${isVisible ? 'scale-100' : 'scale-90'}`}
				onClick={(e) => e.stopPropagation()} // Prevent closing when clicking the card itself
			>
				<Card
					id={card.id}
					rank={card.rank}
					suit={card.suit}
					faceUp={card.known}
					className="w-40 h-auto shadow-2xl rounded-lg" // Larger size
				/>
				<p className="text-center text-white text-sm mt-2">
					{card.known ? 'You drew:' : 'Opponent drew a card'}
				</p>
			</div>
		</div>
	);
};

export default DrawnCardDisplay;