// src/hooks/useSocket.ts
// Unified WebSocket hook — single connection to /ws/{lobbyId}, subprotocol "cambia".
// Replaces the separate useLobbySocket and useGameSocket hooks.
import { useEffect, useRef, useCallback } from 'react';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import { useGameStore } from '@/stores/gameStore';

const WS_URL = import.meta.env.VITE_WS_URL;
const MAX_RETRIES = 5;
const INITIAL_RETRY_DELAY = 1000;

/**
 * Server envelopes: { seq: number, type: string, payload?: any }
 * Client messages: { last_seq: number, type: string, body?: any }
 */

/** Lobby-phase message types routed to lobbyStore */
const LOBBY_TYPES = new Set([
	'lobby_state', 'lobby_update', 'phase_change', 'ready_update',
	'lobby_rules_updated', 'chat', 'lobby_countdown_start',
	'lobby_countdown_cancel', 'lobby_invite', 'game_start',
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

	const userId = useAuthStore((state) => state.user?.id);

	const lobbyActions = useCurrentLobbyStore();
	const gameActions = useGameStore();

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

		if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
			lobbyActions.setLoading(true);
			lobbyActions.clearError();
		}

		let socket: WebSocket;
		try {
			socket = new WebSocket(`${WS_URL}/ws/${targetLobbyId}`, 'cambia');
		} catch (error) {
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
					// Desync recovery — force both stores
					useCurrentLobbyStore.getState().forceSync(payload);
					useGameStore.getState().forceSync(payload);
				} else if (type === 'error') {
					// Errors go to both stores
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

	}, [userId, lobbyActions, gameActions]);

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
	const sendMessage = useCallback((message: { type: string; body?: any; [key: string]: any }) => {
		if (!managedLobbyId.current || ws.current?.readyState !== WebSocket.OPEN) {
			console.warn('[useSocket] sendMessage prevented: not connected.');
			return;
		}
		try {
			const wire = {
				...message,
				last_seq: lastSeqRef.current,
			};
			ws.current.send(JSON.stringify(wire));
		} catch (error) {
			console.error('[useSocket] Failed to send message:', error);
		}
	}, []);

	/** Explicitly close the connection. */
	const closeSocket = useCallback(() => {
		shouldBeConnected.current = false;

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
