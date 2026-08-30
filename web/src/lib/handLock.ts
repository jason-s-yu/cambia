// src/lib/handLock.ts
// The LockCallerHand house rule as the client reads it (cambia-1069).
//
// RULES.md 3C: once you call Cambia "your hand is locked and cannot be altered by any player,
// including yourself (snaps, swaps, etc.)". Two server layers enforce that, and this module
// mirrors both:
//
//   - Snaps. service/internal/game/engine_adapter.go handLocked gates the snap path twice
//     (cambia-1043): once on the SNAPPER's seat before either hand is searched, so the caller
//     cannot snap at all, own hand included, and once on the TARGET's seat, so nobody can snap a
//     card out of the caller's hand. Both are refused without a penalty and answered with the
//     public player_snap_fail event, which the table renders as a no-op: the click died silently.
//   - Swap abilities. J/Q blind swap and K look-and-swap never reach an apply at all: the engine
//     drops every BlindSwap/KingLook naming the caller's seat out of the legal action set
//     (engine/legal.go legalAbilitySelect, nplayerLegalAbilitySelect) and fizzles the ability
//     outright when no other seat is left to target (engine/legal.go canUseAbility,
//     engine/abilities.go discardWithAbility).
//
// Peeks are deliberately not locked. 9/T peek_other may name a card in the caller's hand
// (engine/legal.go PendingPeekOther carries no lock branch) because looking alters nothing, so a
// caller of these helpers must gate swaps and snaps on them and leave peek targeting alone.

/** The part of a player's board state the rule reads. */
export interface HandLockPlayerView {
	playerId: string;
	/** Set on the caller's seat in every sync_state (service/internal/game/sync_state.go). */
	hasCalledCambia?: boolean;
}

/** The part of the board state the rule reads. */
export interface HandLockView {
	cambiaCalled: boolean;
	cambiaCallerId?: string | null;
	players: HandLockPlayerView[];
	houseRules: { lockCallerHand?: boolean };
}

/**
 * The seat whose hand LockCallerHand has frozen, or null when no hand is locked.
 *
 * The caller comes from cambiaCallerId, which both the player_cambia event and sync_state carry,
 * and falls back to the seat flagged hasCalledCambia so a client that resynced without the event
 * still finds it. An absent lockCallerHand reads as on: it is the server's default
 * (service/internal/game/rules.go DefaultHouseRules) and the same default the lobby settings
 * panel shows.
 */
export function lockedPlayerId(view: HandLockView): string | null {
	if (!view.cambiaCalled) return null;
	if (!(view.houseRules?.lockCallerHand ?? true)) return null;
	if (view.cambiaCallerId) return view.cambiaCallerId;
	return view.players.find((p) => p.hasCalledCambia)?.playerId ?? null;
}

/** Whether this player's hand is frozen. Mirrors engine_adapter.go handLocked. */
export function isHandLocked(view: HandLockView, playerId: string | null | undefined): boolean {
	if (!playerId) return false;
	return lockedPlayerId(view) === playerId;
}

/**
 * Whether `snapperId` may snap a card out of `ownerId`'s hand, which for an own-hand snap are the
 * same player. Mirrors handleSnapViaEngine's two handLocked gates: a locked snapper is refused
 * whatever they name, and a locked target is refused whoever names it.
 */
export function canSnapCard(view: HandLockView, snapperId: string | null | undefined, ownerId: string | null | undefined): boolean {
	return !isHandLocked(view, snapperId) && !isHandLocked(view, ownerId);
}
