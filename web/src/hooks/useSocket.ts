// src/hooks/useSocket.ts
// Unified WebSocket hook — single connection to /ws/{lobbyId}, subprotocol "cambia".
// Replaces the separate useLobbySocket and useGameSocket hooks.
import { useEffect, useRef, useCallback } from 'react';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import { useGameStore } from '@/stores/gameStore';
import { WS_URL } from '@/lib/runtimeEnv';
import { cardRefsOf, decideResend, tableContext, type OutboundRecord } from '@/lib/resendDecision';
const MAX_RETRIES = 5;
const INITIAL_RETRY_DELAY = 1000;

/**
 * Server envelopes: { seq: number, type: string, payload?: any }
 * Client messages: { last_seq: number, type: string, body?: any }
 */

/**
 * Anything the app sends. Action frames carry their card references at the top level (the hub
 * falls back to the whole frame when no `body` is present, see hub/connection.go), and those
 * references are part of the resend decision, so they are named here rather than hidden behind
 * an opaque body.
 */
interface OutboundMessage {
	type: string;
	body?: unknown;
	special?: string;
	card?: { id?: string };
	card1?: { id?: string };
	card2?: { id?: string };
}

/** Lobby-phase message types routed to lobbyStore */
const LOBBY_TYPES = new Set([
	'lobby_state', 'phase_change', 'chat', 'game_start',
]);

/** Game-phase message types routed to gameStore */
function isGameType(type: string): boolean {
	return type.startsWith('game_') || type.startsWith('player_') || type.startsWith('private_');
}

export function useSocket(lobbyId: string | null | undefined) {
	const ws = useRef<WebSocket | null>(null);
	const reconnectTimeoutId = useRef<number | null>(null);
	const retryCountRef = useRef<number>(0);
	const managedLobbyId = useRef<string | null>(null);
	const isConnecting = useRef<boolean>(false);
	const shouldBeConnected = useRef<boolean>(false);
	const lastSeqRef = useRef<number>(0);
	/** The last frame sent, held until the hub either acts on it or answers it with a repair. */
	const lastSentRef = useRef<{ message: OutboundMessage; record: OutboundRecord } | null>(null);

	const userId = useAuthStore((state) => state.user?.id);

	const lobbyActions = useCurrentLobbyStore();

	/** The board and lobby state a resend decision compares against. */
	const context = useCallback(() => {
		const game = useGameStore.getState();
		return tableContext(
			game.gameState,
			game.pendingAction,
			useCurrentLobbyStore.getState().phase,
			useAuthStore.getState().user?.id
		);
	}, []);

	/** Puts a frame on the wire with the current seq and records it for the repair path. */
	const send = useCallback((message: OutboundMessage, attempt: number): boolean => {
		if (ws.current?.readyState !== WebSocket.OPEN) return false;
		const sentSeq = lastSeqRef.current;
		try {
			ws.current.send(JSON.stringify({ ...message, last_seq: sentSeq }));
		} catch (error) {
			console.error('[useSocket] Failed to send message:', error);
			return false;
		}
		lastSentRef.current = {
			message,
			record: {
				type: message.type,
				cardRefs: cardRefsOf(message),
				sentSeq,
				attempt,
				ctx: context()
			}
		};
		return true;
	}, [context]);

	const connectWebSocket = useCallback((targetLobbyId: string) => {
		if (!targetLobbyId || !userId) {
			isConnecting.current = false;
			shouldBeConnected.current = false;
			if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
				lobbyActions.setLoading(false);
				lobbyActions.setError('Cannot connect: Invalid lobby ID or user not authenticated.');
			}
			return;
		}

		if (managedLobbyId.current === targetLobbyId && (isConnecting.current || ws.current?.readyState === WebSocket.OPEN)) {
			shouldBeConnected.current = true;
			if (ws.current?.readyState === WebSocket.OPEN) {
				lobbyActions.setConnected(true);
				lobbyActions.setLoading(false);
				lobbyActions.clearError();
			}
			return;
		}

		if (reconnectTimeoutId.current) {
			clearTimeout(reconnectTimeoutId.current);
			reconnectTimeoutId.current = null;
		}

		if (ws.current && (managedLobbyId.current !== targetLobbyId || ws.current.readyState === WebSocket.CLOSING)) {
			ws.current.onclose = null;
			ws.current.onerror = null;
			ws.current.onmessage = null;
			ws.current.onopen = null;
			ws.current.close(1000, `Switching to lobby ${targetLobbyId}`);
			ws.current = null;
		}

		if (!WS_URL) {
			lobbyActions.setError('WebSocket URL is not configured.');
			lobbyActions.setLoading(false);
			isConnecting.current = false;
			shouldBeConnected.current = false;
			managedLobbyId.current = null;
			return;
		}

		isConnecting.current = true;
		shouldBeConnected.current = true;
		managedLobbyId.current = targetLobbyId;
		lastSeqRef.current = 0;
		lastSentRef.current = null;

		if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
			lobbyActions.setLoading(true);
			lobbyActions.clearError();
		}

		let socket: WebSocket;
		try {
			socket = new WebSocket(`${WS_URL}/ws/${targetLobbyId}`, 'cambia');
		} catch {
			lobbyActions.setError('Failed to initialize connection.');
			lobbyActions.setLoading(false);
			isConnecting.current = false;
			shouldBeConnected.current = false;
			managedLobbyId.current = null;
			return;
		}
		ws.current = socket;

		socket.onopen = () => {
			if (ws.current !== socket || managedLobbyId.current !== targetLobbyId || !shouldBeConnected.current) {
				socket.close(1000, 'Stale connection opened');
				return;
			}
			retryCountRef.current = 0;
			isConnecting.current = false;
			if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
				lobbyActions.setConnected(true);
				lobbyActions.setLoading(false);
				lobbyActions.clearError();
			}
		};

		socket.onmessage = (event) => {
			if (ws.current !== socket || managedLobbyId.current !== targetLobbyId || !shouldBeConnected.current) return;
			try {
				const envelope = JSON.parse(event.data);
				const { seq, type, payload } = envelope;

				// Update seq tracking
				if (typeof seq === 'number') {
					lastSeqRef.current = Math.max(lastSeqRef.current, seq);
				}

				// Route by message type
				if (type === 'sync_state') {
					// Desync recovery. The repair carries the hub's lobby snapshot and the seq to
					// catch up to; the game snapshot rides its own private_sync_state, and one FIFO
					// writer per connection means every frame this client was behind on has already
					// been applied, so the state read below is current.
					useCurrentLobbyStore.getState().forceSync(payload);
					useGameStore.getState().forceSync(payload);

					// The frame the hub discarded. Without a resend the player's action is simply
					// gone (cambia-891); with a stale one it can mean something else entirely, so
					// decideResend re-checks the intent against the repaired board.
					const sent = lastSentRef.current;
					if (sent) {
						const syncSeq = typeof payload?.seq === 'number' ? payload.seq : lastSeqRef.current;
						const verdict = decideResend(sent.record, context(), syncSeq);
						if (verdict === 'resend') {
							if (!send(sent.message, sent.record.attempt + 1)) lastSentRef.current = null;
						} else {
							lastSentRef.current = null;
							if (verdict === 'notify') useGameStore.getState().noteDroppedAction();
						}
					}
				} else if (type === 'error') {
					// Errors go to both stores
					useCurrentLobbyStore.getState().processLobbyWebSocketMessage(type, payload);
					useGameStore.getState().processGameWebSocketMessage(type, payload);
				} else if (type === 'game_results') {
					// Dual-route (cambia-763 F2): game_results starts with "game_" so isGameType would
					// claim it before LOBBY_TYPES is even consulted, but its lobby_status snapshot is
					// the only place the post-game reset (ReadyStates cleared, InGame false — see
					// api_server.go attachOnGameEnd) reaches the client. LobbyPage's "Back to lobby"
					// button (handleReturnToLobby) flips phase locally with no resync, so without this
					// lobbyStore would keep serving the stale pre-game ready state into the next lobby
					// view. gameStore still needs it too (winner/scores, duplicated from game_end).
					useCurrentLobbyStore.getState().processLobbyWebSocketMessage(type, payload);
					useGameStore.getState().processGameWebSocketMessage(type, payload);
				} else if (LOBBY_TYPES.has(type)) {
					useCurrentLobbyStore.getState().processLobbyWebSocketMessage(type, payload);
				} else if (isGameType(type)) {
					useGameStore.getState().processGameWebSocketMessage(type, payload);
				} else {
					console.warn(`[useSocket] Unknown message type: ${type}`);
				}

				// Handle lobby-not-found fatal error
				if (type === 'error' && payload?.code === 'lobby_not_found') {
					shouldBeConnected.current = false;
					retryCountRef.current = MAX_RETRIES + 1;
					if (ws.current === socket) {
						ws.current.close(1000, 'Lobby not found');
						ws.current = null;
					}
					managedLobbyId.current = null;
					lobbyActions.setError(payload.message || 'Lobby not found.');
					lobbyActions.setLoading(false);
					lobbyActions.setConnected(false);
					lobbyActions.leaveLobby();
				}
			} catch (error) {
				console.error('[useSocket] Failed to parse message:', error);
				lobbyActions.setError('Error processing message from server.');
			}
		};

		socket.onclose = (event) => {
			if (ws.current !== socket && managedLobbyId.current !== targetLobbyId) return;

			ws.current = null;
			isConnecting.current = false;

			const storeLobbyId = useCurrentLobbyStore.getState().currentLobbyId;
			if (storeLobbyId === targetLobbyId) {
				lobbyActions.setConnected(false);
				useGameStore.getState().setConnected(false);
			}

			const wasUnexpected = !event.wasClean && event.code !== 1000;
			const retryAllowed = retryCountRef.current < MAX_RETRIES;

			if (wasUnexpected && retryAllowed && shouldBeConnected.current && managedLobbyId.current === targetLobbyId) {
				const currentRetry = retryCountRef.current++;
				const delay = Math.pow(2, currentRetry) * INITIAL_RETRY_DELAY + Math.random() * 1000;

				if (storeLobbyId === targetLobbyId) {
					lobbyActions.setLoading(true);
					lobbyActions.setError(`Connection lost. Retrying... (Attempt ${retryCountRef.current})`);
				}

				reconnectTimeoutId.current = window.setTimeout(() => {
					if (shouldBeConnected.current && managedLobbyId.current === targetLobbyId) {
						connectWebSocket(targetLobbyId);
					} else {
						if (storeLobbyId === targetLobbyId) lobbyActions.setLoading(false);
						managedLobbyId.current = null;
						retryCountRef.current = 0;
					}
				}, delay);
			} else {
				managedLobbyId.current = null;
				retryCountRef.current = 0;

				if (storeLobbyId === targetLobbyId) {
					lobbyActions.setLoading(false);
					if (wasUnexpected) {
						lobbyActions.setError(retryAllowed ? 'Lost connection. Stopped trying.' : `Lost connection after ${MAX_RETRIES} retries.`);
					} else if (event.code !== 1000 && event.code !== 1005 && event.reason) {
						lobbyActions.setError(`Disconnected: ${event.reason}`);
					}
				}
				if (!retryAllowed || !wasUnexpected) {
					shouldBeConnected.current = false;
				}
			}
		};

		socket.onerror = () => {
			if (ws.current !== socket || !shouldBeConnected.current || managedLobbyId.current !== targetLobbyId) return;
			if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
				lobbyActions.setError('WebSocket connection error.');
			}
		};

	}, [userId, lobbyActions, context, send]);

	// Connect/disconnect based on lobbyId prop
	useEffect(() => {
		if (lobbyId) {
			if (managedLobbyId.current !== lobbyId || (!isConnecting.current && ws.current?.readyState !== WebSocket.OPEN)) {
				shouldBeConnected.current = true;
				retryCountRef.current = 0;
				connectWebSocket(lobbyId);
			} else {
				shouldBeConnected.current = true;
			}
		} else {
			shouldBeConnected.current = false;

			if (reconnectTimeoutId.current) {
				clearTimeout(reconnectTimeoutId.current);
				reconnectTimeoutId.current = null;
			}

			if (ws.current && ws.current.readyState !== WebSocket.CLOSED && ws.current.readyState !== WebSocket.CLOSING) {
				ws.current.onclose = null;
				ws.current.close(1000, 'Lobby ID became null');
				ws.current = null;
			}

			managedLobbyId.current = null;
			isConnecting.current = false;
			retryCountRef.current = 0;

			if (useCurrentLobbyStore.getState().isConnected) {
				lobbyActions.setConnected(false);
				lobbyActions.setLoading(false);
			}
		}

		return () => {
			if (reconnectTimeoutId.current) {
				clearTimeout(reconnectTimeoutId.current);
				reconnectTimeoutId.current = null;
			}
		};
	}, [lobbyId, connectWebSocket, lobbyActions]);

	/** Send a message over the WS. Injects last_seq automatically. */
	const sendMessage = useCallback((message: OutboundMessage) => {
		if (!managedLobbyId.current || ws.current?.readyState !== WebSocket.OPEN) {
			console.warn('[useSocket] sendMessage prevented: not connected.');
			return;
		}
		send(message, 0);
	}, [send]);

	/** Explicitly close the connection. */
	const closeSocket = useCallback(() => {
		shouldBeConnected.current = false;
		lastSentRef.current = null;

		if (reconnectTimeoutId.current) {
			clearTimeout(reconnectTimeoutId.current);
			reconnectTimeoutId.current = null;
		}
		if (ws.current && ws.current.readyState !== WebSocket.CLOSED && ws.current.readyState !== WebSocket.CLOSING) {
			ws.current.onclose = null;
			ws.current.close(1000, 'User initiated disconnect');
			ws.current = null;
		}
		managedLobbyId.current = null;
		isConnecting.current = false;
		retryCountRef.current = 0;

		if (useCurrentLobbyStore.getState().isConnected) {
			lobbyActions.setConnected(false);
			lobbyActions.setLoading(false);
		}
	}, [lobbyActions]);

	const isConnected = useCurrentLobbyStore((s) => s.isConnected);
	const isLoading = useCurrentLobbyStore((s) => s.isLoading);
	const error = useCurrentLobbyStore((s) => s.error);

	return { sendMessage, closeSocket, isConnected, isLoading, error };
}
