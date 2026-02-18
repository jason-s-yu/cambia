// src/components/game/Card.tsx
import React from 'react';

interface CardProps {
	id: string;
	rank?: string;
	suit?: string;
	faceUp: boolean;
	onClick?: () => void;
	className?: string;
	// Add other props like `isSelected` if needed later
}

/**
 * Renders a single playing card using SVG sprites.
 */
const Card: React.FC<CardProps> = ({ id, rank, suit, faceUp, onClick, className = '' }) => {
	const spritePath = '/svg-cards.svg'; // Assuming svg-cards.svg is in the public folder

	const getSuitChar = (s?: string): string => {
		switch (s?.toUpperCase()) {
			case 'H': return 'heart';
			case 'D': return 'diamond';
			case 'C': return 'club';
			case 'S': return 'spade';
			case 'R': return 'joker'; // Red Joker
			case 'B': return 'joker'; // Black Joker
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
			case 'O': return 'joker'; // Joker rank
			default: return r ?? ''; // For 2-9
		}
	};

	const getJokerColor = (s?: string): string => {
		return s?.toUpperCase() === 'R' ? '_red' : '_black'; // Default to black joker
	};

	let cardIdInSprite = 'back'; // Default to card back
	if (faceUp && rank && suit) {
		if (rank === 'O') { // Handle Jokers
			cardIdInSprite = `joker${getJokerColor(suit)}`;
		} else {
			const suitName = getSuitChar(suit);
			const rankName = getRankChar(rank);
			if (suitName && rankName) {
				cardIdInSprite = `${suitName}_${rankName}`;
			} else {
				console.warn(`Invalid rank/suit combination: ${rank} ${suit}`);
				cardIdInSprite = 'back'; // Fallback if invalid rank/suit
			}
		}
	}

	const href = `${spritePath}#${cardIdInSprite}`;

	return (
		<div
			className={`aspect-[63/88] overflow-hidden rounded border border-gray-300 dark:border-gray-600 shadow-sm cursor-pointer ${className}`}
			onClick={onClick}
			title={faceUp ? `${rank} of ${suit}` : 'Card'}
		>
			<svg viewBox="0 0 169.075 244.640" className="w-full h-full">
				<use href={href}></use>
			</svg>
		</div>
	);
};

export default Card;