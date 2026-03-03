// src/components/game/DrawnCardDisplay.tsx
import React from 'react';
import { motion } from 'framer-motion';
import { springTransition } from './CardAnimations';
import Card from './Card';
import type { ObfCard } from '@/types/game';

interface DrawnCardDisplayProps {
	card: ObfCard;
	onClose: () => void;
}

const DrawnCardDisplay: React.FC<DrawnCardDisplayProps> = ({ card, onClose }) => {
	return (
		<motion.div
			initial={{ opacity: 0 }}
			animate={{ opacity: 1 }}
			exit={{ opacity: 0 }}
			className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
			onClick={onClose}
		>
			<motion.div
				initial={{ scale: 0.8, y: 30 }}
				animate={{ scale: 1, y: 0 }}
				exit={{ scale: 0.8, y: 30 }}
				transition={springTransition}
				className="flex flex-col items-center gap-3"
				onClick={(e) => e.stopPropagation()}
			>
				<p className="text-white text-sm font-medium">
					{card.known ? 'You drew:' : 'Opponent drew a card'}
				</p>
				<div className="w-36">
					<Card
						id={card.id}
						rank={card.rank}
						suit={card.suit}
						faceUp={card.known}
						isInteractive={false}
					/>
				</div>
			</motion.div>
		</motion.div>
	);
};

export default React.memo(DrawnCardDisplay);
