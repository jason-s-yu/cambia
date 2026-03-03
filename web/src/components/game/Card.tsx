// src/components/game/Card.tsx
import React, { useCallback } from 'react';
import { motion } from 'framer-motion';
import { cardVariants, cardHover, cardTap, springTransition } from './CardAnimations';

export interface CardProps {
	id: string;
	rank?: string;
	suit?: string;
	faceUp: boolean;
	onClick?: () => void;
	className?: string;
	/** Framer Motion layout animation key */
	layoutId?: string;
	/** Whether this card is currently selected */
	isSelected?: boolean;
	/** Whether this card can be interacted with */
	isInteractive?: boolean;
	/** Whether this card is a valid target for the current action */
	isTargetable?: boolean;
	/** Compact sizing for opponent hands */
	compact?: boolean;
}

const spritePath = '/svg-cards.svg';

const getSuitChar = (s?: string): string => {
	switch (s?.toUpperCase()) {
		case 'H': return 'heart';
		case 'D': return 'diamond';
		case 'C': return 'club';
		case 'S': return 'spade';
		case 'R': case 'B': return 'joker';
		default: return '';
	}
};

const getRankChar = (r?: string): string => {
	switch (r?.toUpperCase()) {
		case 'A': return '1';
		case 'T': return '10';
		case 'J': return 'jack';
		case 'Q': return 'queen';
		case 'K': return 'king';
		case 'O': return 'joker';
		default: return r ?? '';
	}
};

const getJokerColor = (s?: string): string => {
	return s?.toUpperCase() === 'R' ? '_red' : '_black';
};

const Card: React.FC<CardProps> = ({
	id, rank, suit, faceUp, onClick, className = '',
	layoutId, isSelected = false, isInteractive = true,
	isTargetable = false, compact = false,
}) => {
	let cardIdInSprite = 'back';
	if (faceUp && rank && suit) {
		if (rank === 'O') {
			cardIdInSprite = `joker${getJokerColor(suit)}`;
		} else {
			const suitName = getSuitChar(suit);
			const rankName = getRankChar(rank);
			if (suitName && rankName) {
				cardIdInSprite = `${suitName}_${rankName}`;
			}
		}
	}

	const href = `${spritePath}#${cardIdInSprite}`;

	const handleClick = useCallback(() => {
		if (isInteractive && onClick) onClick();
	}, [isInteractive, onClick]);

	const sizeClasses = compact
		? 'w-12 h-[67px] sm:w-14 sm:h-[78px]'
		: 'w-16 h-[89px] sm:w-20 sm:h-[111px]';

	const targetRing = isTargetable
		? 'ring-2 ring-blue-400 ring-offset-1 dark:ring-blue-500'
		: '';

	const interactiveCursor = isInteractive ? 'cursor-pointer' : 'cursor-default';

	return (
		<motion.div
			layoutId={layoutId}
			variants={cardVariants}
			animate={isSelected ? 'selected' : 'idle'}
			whileHover={isInteractive ? cardHover : undefined}
			whileTap={isInteractive ? cardTap : undefined}
			transition={springTransition}
			onClick={handleClick}
			className={`${sizeClasses} aspect-[63/88] overflow-hidden rounded-md border border-gray-300 dark:border-gray-600 shadow-sm select-none ${interactiveCursor} ${targetRing} ${className}`}
			title={faceUp ? `${rank} of ${suit}` : 'Card'}
			style={{ touchAction: 'none' }}
		>
			<svg viewBox="0 0 169.075 244.640" className="w-full h-full pointer-events-none">
				<use href={href}></use>
			</svg>
		</motion.div>
	);
};

export default React.memo(Card);
