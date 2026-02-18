// src/components/game/ActionControls.tsx
import React from 'react';
import type { ClientGameAction } from '@/types/game';
import Button from '@/components/common/Button';
import { useGameStore, selectIsSelfTurn, selectSelfPlayerState, selectPendingAction, selectIsProcessingAction } from '@/stores/gameStore';
import {
	drawStockpileAction,
	drawDiscardPileAction,
	discardAction,
	// Import other action creators as needed
} from '@/types/game';

interface ActionControlsProps {
	sendMessage: (action: ClientGameAction) => void;
}

const ActionControls: React.FC<ActionControlsProps> = ({ sendMessage }) => {
	const isMyTurn = useGameStore(selectIsSelfTurn);
	const selfState = useGameStore(selectSelfPlayerState);
	const pendingAction = useGameStore(selectPendingAction);
	const isProcessing = useGameStore(selectIsProcessingAction);
	const gameState = useGameStore(state => state.gameState); // Need full state for rules

	const canDraw = isMyTurn && pendingAction === null && !isProcessing;
	const canDiscardOrReplace = isMyTurn && pendingAction === 'discard_replace' && !isProcessing;
	// const canDoSpecial = isMyTurn && pendingAction === 'special_action' && !isProcessing;
	const canDrawDiscard = gameState?.houseRules.allowDrawFromDiscardPile ?? false;
	const discardPileNotEmpty = (gameState?.discardSize ?? 0) > 0;

	const handleDrawStockpile = () => {
		if (!canDraw) return;
		sendMessage(drawStockpileAction());
	};

	const handleDrawDiscard = () => {
		if (!canDraw || !canDrawDiscard || !discardPileNotEmpty) return;
		sendMessage(drawDiscardPileAction());
	};

	const handleDiscard = () => {
		if (!canDiscardOrReplace || !selfState?.drawnCard) return;
		sendMessage(discardAction(selfState.drawnCard.id));
	};

	// TODO: Implement handlers for Replace, Snap, Cambia, Special Actions

	return (
		<div className="flex justify-center gap-2 p-2 bg-gray-800/50 dark:bg-black/50 rounded-md mt-2">
			{/* Draw Phase Buttons */}
			{pendingAction === null && isMyTurn && (
				<>
					<Button onClick={handleDrawStockpile} disabled={!canDraw} size="sm">Draw Stock</Button>
					{canDrawDiscard && (
						<Button onClick={handleDrawDiscard} disabled={!canDraw || !discardPileNotEmpty} size="sm">Draw Discard</Button>
					)}
				</>
			)}

			{/* Discard/Replace Phase Buttons */}
			{pendingAction === 'discard_replace' && isMyTurn && (
				<>
					<Button onClick={handleDiscard} disabled={!canDiscardOrReplace} size="sm" variant='primary'>Discard Drawn</Button>
					{/* Add Replace button later - requires card selection */}
					{/* <Button onClick={handleReplace} disabled={!canDiscardOrReplace || !selectedHandCard} size="sm">Replace Selected</Button> */}
				</>
			)}

			{/* TODO: Add Special Action Buttons */}
			{/* {pendingAction === 'special_action' && isMyTurn && ( ... )} */}

			{/* TODO: Add Snap Button (always available?) */}
			{/* <Button onClick={handleSnap} disabled={isProcessing || !selectedHandCard}>Snap Selected</Button> */}

			{/* TODO: Add Cambia Button (available on turn start?) */}
			{/* <Button onClick={handleCallCambia} disabled={!isMyTurn || pendingAction !== null || isProcessing}>Call Cambia</Button> */}
		</div>
	);
};

export default ActionControls;