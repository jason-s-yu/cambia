// src/components/game/DiscardPile.tsx
import React from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import type { ObfCard } from '@/types/game';
import Card from './Card';
import { discardCardVariants } from './CardAnimations';

interface DiscardPileProps {
	card: ObfCard | null | undefined;
	onClick?: () => void;
	isInteractive?: boolean;
}

const DiscardPile: React.FC<DiscardPileProps> = ({ card, onClick, isInteractive = false }) => {
	return (
		<div className="relative flex flex-col items-center">
			<span className="text-xs text-white mb-1 drop-shadow">Discard</span>
			<div className="w-16 h-[89px]">
				<AnimatePresence mode="popLayout">
					{card ? (
						<motion.div
							key={card.id}
							variants={discardCardVariants}
							initial="initial"
							animate="animate"
							exit="exit"
							className="w-full h-full"
						>
							<Card
								id={card.id}
								rank={card.rank}
								suit={card.suit}
								faceUp={true}
								onClick={onClick}
								isInteractive={isInteractive}
								className="w-full h-full shadow-lg"
							/>
						</motion.div>
					) : (
						<motion.div
							key="empty"
							initial={{ opacity: 0 }}
							animate={{ opacity: 1 }}
							exit={{ opacity: 0 }}
							className="w-full h-full bg-black/30 rounded-md border border-dashed border-white/50 flex items-center justify-center text-white text-xs italic"
						>
							Empty
						</motion.div>
					)}
				</AnimatePresence>
			</div>
		</div>
	);
};

export default DiscardPile;
