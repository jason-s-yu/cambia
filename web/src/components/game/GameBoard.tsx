// src/components/game/GameBoard.tsx
import React, { useState, useCallback, useMemo } from 'react';
import { LayoutGroup, AnimatePresence } from 'framer-motion';
import type { ObfGameState, ObfCard, ClientGameAction } from '@/types/game';
import {
	drawStockpileAction,
	drawDiscardPileAction,
	discardAction,
	replaceAction,
	snapAction,
	peekSelfAction,
	peekOtherAction,
	blindSwapAction,
	kingPeekAction,
} from '@/types/game';
import { useGameStore } from '@/stores/gameStore';
import {
	selectPendingAction,
	selectIsSelfTurn,
	selectIsProcessingAction,
	selectDisplayedDrawnCard,
} from '@/stores/gameStore';
import { useAuthStore } from '@/stores/authStore';
import { HINT_LABELS } from './CardAnimations';
import GameTable from './GameTable';
import PlayerHand from './PlayerHand';
import OpponentHand from './OpponentHand';
import Deck from './Deck';
import DiscardPile from './DiscardPile';
import GameInfoPanel from './GameInfoPanel';
import ActionControls from './ActionControls';
import DrawnCardDisplay from './DrawnCardDisplay';

interface GameBoardProps {
	gameState: ObfGameState;
	sendMessage: (msg: ClientGameAction) => void;
}

const GameBoard: React.FC<GameBoardProps> = ({ gameState, sendMessage }) => {
	const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

	const selfId = useAuthStore((state) => state.user?.id);
	const pendingAction = useGameStore(selectPendingAction);
	const isMyTurn = useGameStore(selectIsSelfTurn);
	const isProcessing = useGameStore(selectIsProcessingAction);
	const displayedDrawnCard = useGameStore(selectDisplayedDrawnCard);
	const clearDisplayedDrawnCard = useGameStore((state) => state.clearDisplayedDrawnCard);

	const selfState = gameState.players.find((p) => p.playerId === selfId);
	const opponents = gameState.players.filter((p) => p.playerId !== selfId);
	const specialAction = gameState.specialAction;

	// --- Interaction handlers ---

	const handlePlayerCardClick = useCallback(
		(card: ObfCard, idx: number) => {
			if (isProcessing) return;

			if (pendingAction === 'discard_replace') {
				sendMessage(replaceAction(card.id, idx));
				setSelectedIdx(null);
				return;
			}

			if (pendingAction === 'special_action' && specialAction) {
				const rank = specialAction.cardRank;
				if (rank === '7' || rank === '8') {
					sendMessage(peekSelfAction(card.id, idx));
					setSelectedIdx(null);
					return;
				}
				if ((rank === 'J' || rank === 'Q') && selectedIdx === null) {
					setSelectedIdx(idx);
					return;
				}
				if (rank === 'K' && selectedIdx === null) {
					setSelectedIdx(idx);
					return;
				}
			}

			if (pendingAction === null) {
				setSelectedIdx((prev) => (prev === idx ? null : idx));
			}
		},
		[isProcessing, pendingAction, specialAction, selectedIdx, sendMessage],
	);

	const handleDeckClick = useCallback(() => {
		if (!isMyTurn || pendingAction !== null || isProcessing) return;
		sendMessage(drawStockpileAction());
	}, [isMyTurn, pendingAction, isProcessing, sendMessage]);

	const handleDiscardClick = useCallback(() => {
		if (isProcessing) return;

		if (isMyTurn && pendingAction === null) {
			if (
				(gameState.houseRules as any).allowDrawFromDiscardPile &&
				gameState.discardTop
			) {
				sendMessage(drawDiscardPileAction());
				return;
			}
		}

		if (pendingAction === 'discard_replace') {
			if (selfState?.drawnCard) {
				sendMessage(discardAction(selfState.drawnCard.id));
				setSelectedIdx(null);
				return;
			}
		}

		if (selectedIdx !== null && pendingAction === null) {
			const selectedCard = selfState?.revealedHand?.[selectedIdx];
			if (selectedCard) {
				sendMessage(snapAction(selectedCard.id));
				setSelectedIdx(null);
			}
		}
	}, [isMyTurn, isProcessing, pendingAction, selectedIdx, selfState, gameState, sendMessage]);

	const handleOpponentCardClick = useCallback(
		(playerId: string, idx: number) => {
			if (isProcessing) return;
			const placeholderId = `${playerId}-card-${idx}`;

			if (pendingAction === 'special_action' && specialAction) {
				const rank = specialAction.cardRank;

				if (rank === '9' || rank === 'T') {
					sendMessage(peekOtherAction(placeholderId, idx, playerId));
					setSelectedIdx(null);
					return;
				}

				if ((rank === 'J' || rank === 'Q') && selectedIdx !== null) {
					const myCard = selfState?.revealedHand?.[selectedIdx];
					if (myCard && selfId) {
						sendMessage(blindSwapAction(myCard.id, selectedIdx, selfId, placeholderId, idx, playerId));
						setSelectedIdx(null);
					}
					return;
				}

				if (rank === 'K' && selectedIdx !== null) {
					const myCard = selfState?.revealedHand?.[selectedIdx];
					if (myCard && selfId) {
						sendMessage(kingPeekAction(myCard.id, selectedIdx, selfId, placeholderId, idx, playerId));
						setSelectedIdx(null);
					}
					return;
				}
			}

			if (selectedIdx !== null && pendingAction === null) {
				const allowOpponentSnapping = (gameState.houseRules as any).allowOpponentSnapping ?? true;
				if (allowOpponentSnapping) {
					const selectedCard = selfState?.revealedHand?.[selectedIdx];
					if (selectedCard) {
						sendMessage(snapAction(selectedCard.id));
						setSelectedIdx(null);
					}
				}
			}
		},
		[isProcessing, pendingAction, specialAction, selectedIdx, selfState, selfId, gameState, sendMessage],
	);

	// --- Derived values ---

	const hint = useMemo(() => {
		if (gameState.cambiaCalled) return HINT_LABELS.cambia_called;
		if (!isMyTurn) return HINT_LABELS.waiting;
		if (pendingAction === 'special_action' && specialAction) {
			const rank = specialAction.cardRank;
			if (rank === '7' || rank === '8') return HINT_LABELS.peek_self;
			if (rank === '9' || rank === 'T') return HINT_LABELS.peek_other;
			if (rank === 'J' || rank === 'Q') return HINT_LABELS.swap_blind;
			if (rank === 'K') return HINT_LABELS.swap_peek;
		}
		if (pendingAction === 'discard_replace') return HINT_LABELS.select_replace;
		if (selectedIdx !== null) return HINT_LABELS.select_snap;
		return HINT_LABELS.your_turn;
	}, [isMyTurn, pendingAction, selectedIdx, specialAction, gameState.cambiaCalled]);

	const canCallCambia =
		isMyTurn &&
		pendingAction === null &&
		!isProcessing &&
		!gameState.cambiaCalled &&
		gameState.started &&
		!gameState.gameOver;

	const canSkipSpecial = isMyTurn && pendingAction === 'special_action' && !isProcessing;

	const opponentTargetable =
		pendingAction === 'special_action' || (selectedIdx !== null && pendingAction === null);

	const deckInteractive =
		isMyTurn && pendingAction === null && !isProcessing && gameState.stockpileSize > 0;

	const discardInteractive =
		(isMyTurn &&
			pendingAction === null &&
			!!(gameState.houseRules as any).allowDrawFromDiscardPile &&
			!!gameState.discardTop) ||
		pendingAction === 'discard_replace' ||
		(selectedIdx !== null && pendingAction === null);

	// --- Render ---

	const centerSlot = (
		<div className="flex flex-col items-center gap-3">
			<GameInfoPanel
				currentPlayerId={gameState.currentPlayerId}
				players={gameState.players}
				turnId={gameState.turnId}
			/>
			<div className="flex gap-4">
				<Deck
					count={gameState.stockpileSize}
					onClick={handleDeckClick}
					isInteractive={deckInteractive}
				/>
				<DiscardPile
					card={gameState.discardTop}
					onClick={handleDiscardClick}
					isInteractive={discardInteractive}
				/>
			</div>
		</div>
	);

	const selfSlot = (
		<div className="flex flex-col items-center gap-2 w-full">
			<PlayerHand
				hand={selfState?.revealedHand ?? []}
				selectedIdx={selectedIdx}
				onCardClick={handlePlayerCardClick}
			/>
			<ActionControls
				sendMessage={sendMessage}
				hint={hint}
				canCallCambia={canCallCambia}
				canSkipSpecial={canSkipSpecial}
			/>
		</div>
	);

	return (
		<LayoutGroup>
			<GameTable
				opponents={opponents.map((opp) => (
					<OpponentHand
						key={opp.playerId}
						playerState={opp}
						isTargetable={opponentTargetable}
						onCardClick={handleOpponentCardClick}
					/>
				))}
				centerSlot={centerSlot}
				selfSlot={selfSlot}
			/>
			<AnimatePresence>
				{displayedDrawnCard && (
					<DrawnCardDisplay card={displayedDrawnCard} onClose={clearDisplayedDrawnCard} />
				)}
			</AnimatePresence>
		</LayoutGroup>
	);
};

export default GameBoard;
