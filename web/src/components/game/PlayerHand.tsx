// src/components/game/PlayerHand.tsx
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Card from './Card';
import type { ObfCard } from '@/types/game';

interface PlayerHandProps {
	hand: ObfCard[];
	selectedIdx: number | null;
	onCardClick: (card: ObfCard, idx: number) => void;
}

const PlayerHand: React.FC<PlayerHandProps> = ({ hand, selectedIdx, onCardClick }) => {
	return (
		<div className="bg-black/20 rounded-lg p-3 flex items-end">
			<AnimatePresence initial={false}>
				{hand.length === 0 ? (
					<motion.span
						key="empty"
						initial={{ opacity: 0 }}
						animate={{ opacity: 1 }}
						exit={{ opacity: 0 }}
						className="text-white/50 italic text-sm h-[89px] flex items-center"
					>
						Empty Hand
					</motion.span>
				) : (
					hand.map((card, idx) => (
						<motion.div
							key={card.id}
							initial={{ opacity: 0, y: 20 }}
							animate={{ opacity: 1, y: 0 }}
							exit={{ opacity: 0, scale: 0.8 }}
							className={idx === 0 ? 'ml-0' : 'ml-[-12px]'}
						>
							<Card
								id={card.id}
								rank={card.rank}
								suit={card.suit}
								faceUp={card.known}
								layoutId={`hand-card-${card.id}`}
								isSelected={idx === selectedIdx}
								isInteractive
								onClick={() => onCardClick(card, idx)}
							/>
						</motion.div>
					))
				)}
			</AnimatePresence>
		</div>
	);
};

export default React.memo(PlayerHand);
