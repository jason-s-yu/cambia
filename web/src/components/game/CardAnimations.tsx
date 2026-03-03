// src/components/game/CardAnimations.tsx
// Shared Framer Motion animation variants and transitions for card components.

import type { Variants, Transition } from 'framer-motion';

// --- Transitions ---

export const springTransition: Transition = {
	type: 'spring',
	stiffness: 500,
	damping: 30,
	mass: 0.5,
};

export const snapTransition: Transition = {
	type: 'spring',
	stiffness: 300,
	damping: 25,
	mass: 0.8,
};

export const flipTransition: Transition = {
	duration: 0.3,
	ease: 'easeInOut',
};

export const fadeTransition: Transition = {
	duration: 0.2,
	ease: 'easeOut',
};

// --- Card Variants ---

export const cardVariants: Variants = {
	idle: {
		y: 0,
		scale: 1,
		zIndex: 0,
		transition: springTransition,
	},
	selected: {
		y: -20,
		scale: 1.05,
		zIndex: 10,
		transition: springTransition,
	},
	dragging: {
		scale: 1.1,
		zIndex: 50,
		boxShadow: '0 20px 40px rgba(0,0,0,0.3)',
	},
};

// Hover/tap for interactive cards
export const cardHover = {
	scale: 1.03,
	y: -6,
	transition: { type: 'spring' as const, stiffness: 400, damping: 20 },
};

export const cardTap = {
	scale: 0.97,
};

// --- Discard Pile Variants ---

export const discardCardVariants: Variants = {
	initial: { opacity: 0, scale: 0.5, y: -40 },
	animate: { opacity: 1, scale: 1, y: 0, transition: springTransition },
	exit: { opacity: 0, scale: 0.8, transition: fadeTransition },
};

// --- Opponent Card Variants ---

export const opponentCardVariants: Variants = {
	idle: { y: 0, scale: 1 },
	targetable: {
		scale: 1.05,
		boxShadow: '0 0 12px rgba(59,130,246,0.5)',
		transition: springTransition,
	},
};

// --- Stagger children (for hand reveal) ---

export const staggerContainer: Variants = {
	hidden: {},
	show: {
		transition: {
			staggerChildren: 0.1,
		},
	},
};

export const staggerChild: Variants = {
	hidden: { rotateY: 180, opacity: 0 },
	show: { rotateY: 0, opacity: 1, transition: flipTransition },
};

// --- Peek glow effect ---

export const peekGlow: Variants = {
	idle: { boxShadow: '0 0 0 rgba(234,179,8,0)' },
	peeking: {
		boxShadow: [
			'0 0 0 rgba(234,179,8,0)',
			'0 0 20px rgba(234,179,8,0.6)',
			'0 0 0 rgba(234,179,8,0)',
		],
		transition: { duration: 1.5, ease: 'easeInOut' },
	},
};

// --- Contextual hint labels ---

export const HINT_LABELS: Record<string, string> = {
	select_snap: 'Select a card to snap',
	select_replace: 'Click a hand slot to replace, or discard pile to discard',
	peek_self: 'Select one of your cards to peek (7/8)',
	peek_other: "Select an opponent's card to peek (9/10)",
	swap_blind: 'Select your card, then an opponent\'s card to swap (J/Q)',
	swap_peek: 'Select your card, then an opponent\'s card to look & swap (K)',
	your_turn: 'Your turn — draw from deck or discard pile',
	waiting: 'Waiting for opponent...',
	cambia_called: 'Cambia called! Last round.',
};
