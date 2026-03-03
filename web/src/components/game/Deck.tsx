// src/components/game/Deck.tsx
import React from 'react';
import { motion } from 'framer-motion';
import Card from './Card';

interface DeckProps {
	count: number;
	onClick?: () => void;
	isInteractive?: boolean;
}

const Deck: React.FC<DeckProps> = ({ count, onClick, isInteractive = false }) => {
	return (
		<div className="relative flex flex-col items-center">
			<span className="text-xs text-white mb-1 drop-shadow">Deck</span>
			<motion.div
				className="w-16 h-[89px]"
				whileHover={isInteractive ? { scale: 1.05 } : undefined}
				animate={isInteractive ? { boxShadow: '0 0 14px rgba(255,255,255,0.45)' } : { boxShadow: 'none' }}
				transition={{ type: 'spring', stiffness: 400, damping: 20 }}
			>
				{count > 0 ? (
					<Card
						id="deck-card"
						faceUp={false}
						onClick={onClick}
						isInteractive={isInteractive}
						className="w-full h-full shadow-lg"
					/>
				) : (
					<div className="w-full h-full bg-black/30 rounded-md border border-dashed border-white/50 flex items-center justify-center text-white text-xs italic">
						Empty
					</div>
				)}
			</motion.div>
			{count > 0 && (
				<span className="absolute bottom-[-16px] text-center text-[10px] font-bold text-white bg-black/60 rounded px-1 py-0.5">
					{count}
				</span>
			)}
		</div>
	);
};

export default Deck;
