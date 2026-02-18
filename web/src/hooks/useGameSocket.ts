// src/hooks/useGameSocket.ts
import { useEffect, useRef, useCallback } from 'react';
import { useAuthStore } from '@/stores/authStore';
import { useGameStore } from '@/stores/gameStore';
import type { ClientGameAction, ServerGameEvent } from '@/types/game';

const WS_URL = import.meta.env.VITE_WS_URL;
const MAX_RETRIES = 5;
const INITIAL_RETRY_DELAY = 1000; // ms

/**
 * Manages the WebSocket connection for a specific game instance.
 *
 * Connects when a valid `gameId` prop is provided and disconnects when it's null/undefined or changes.
 * Handles connection logic, message sending/receiving, state updates via gameStore, and reconnection.
 *
 * @param gameId - The ID of the game to connect to, or null/undefined to disconnect.
 * @returns An object containing `sendMessage` and `closeSocket` functions.
 */
export function useGameSocket(gameId: string | null | undefined) {
	const ws = useRef<WebSocket | null>(null);
	const reconnectTimeoutId = useRef<number | null>(null);
	const retryCountRef = useRef<number>(0);
	const managedGameId = useRef<string | null>(null); // Tracks the ID this hook instance manages
	const isConnecting = useRef<boolean>(false);
	const shouldBeConnected = useRef<boolean>(false); // Explicit intent flag

	// Zustand store actions and state
	const {
		setConnected,
		processGameWebSocketMessage,
		setError,
		setLoading,
		clearError,
		clearGameState,
		setGameId,
		setProcessingAction,
		isConnected,
		isLoading,
		error: gameError
	} = useGameStore();

	const user = useAuthStore((state) => state.user); // Get user for logging

	/** Establishes or re-establishes the WebSocket connection */
	const connectWebSocket = useCallback((targetGameId: string) => {
		if (!targetGameId) {
			console.log(`[GameWS Hook ${targetGameId || 'N/A'}] Connect PREVENTED: Invalid targetGameId.`);
			isConnecting.current = false; shouldBeConnected.current = false;
			if (useGameStore.getState().gameId === targetGameId) {
				setLoading(false); setError('Cannot connect: Invalid game ID.');
			}
			return;
		}

		if (managedGameId.current === targetGameId && (isConnecting.current || ws.current?.readyState === WebSocket.OPEN)) {
			shouldBeConnected.current = true;
			if (ws.current?.readyState === WebSocket.OPEN && useGameStore.getState().gameId === targetGameId) {
				setConnected(true); setLoading(false); clearError();
			}
			return;
		}

		if (reconnectTimeoutId.current) clearTimeout(reconnectTimeoutId.current);
		reconnectTimeoutId.current = null;

		if (ws.current && (managedGameId.current !== targetGameId || ws.current.readyState === WebSocket.CLOSING)) {
			console.warn(`[GameWS Hook ${targetGameId}] Closing existing WebSocket (Managed: ${managedGameId.current}, State: ${ws.current.readyState}) for new connection.`);
			ws.current.onclose = null; ws.current.onerror = null; ws.current.onmessage = null; ws.current.onopen = null;
			ws.current.close(1000, `Switching to new game ${targetGameId}`);
			ws.current = null;
		}

		if (!WS_URL) {
			console.error(`[GameWS Hook ${targetGameId}] Cannot connect: VITE_WS_URL is not defined.`);
			setError('WebSocket URL is not configured.'); setLoading(false);
			isConnecting.current = false; shouldBeConnected.current = false; managedGameId.current = null;
			return;
		}

		console.log(`[GameWS Hook ${targetGameId}] Attempting connection (Attempt ${retryCountRef.current + 1})...`);
		isConnecting.current = true; shouldBeConnected.current = true; managedGameId.current = targetGameId;

		// Update store only if it's managing this game ID
		if (useGameStore.getState().gameId === targetGameId) {
			setLoading(true); clearError();
		}

		let socket: WebSocket;
		try {
			socket = new WebSocket(`${WS_URL}/game/ws/${targetGameId}`, 'game');
		} catch (error) {
			console.error(`[GameWS Hook ${targetGameId}] Error creating WebSocket instance:`, error);
			setError('Failed to initialize connection.'); setLoading(false);
			isConnecting.current = false; shouldBeConnected.current = false; managedGameId.current = null;
			return;
		}
		ws.current = socket;

		socket.onopen = () => {
			if (ws.current !== socket || managedGameId.current !== targetGameId || !shouldBeConnected.current) {
				console.warn(`[GameWS Hook ${targetGameId}] onopen for stale instance.`);
				socket.close(1000, 'Stale connection opened'); return;
			}
			console.log(`[GameWS Hook ${targetGameId}] WebSocket connected.`);
			retryCountRef.current = 0; isConnecting.current = false;
			if (useGameStore.getState().gameId === targetGameId) {
				setConnected(true); setLoading(false); clearError();
				// Request initial sync after connection (server should ideally send it automatically, but explicit request can help)
				// sendMessage({ type: 'request_sync' }); // Assuming a message type exists for this
			}
		};

		socket.onmessage = (event) => {
			if (ws.current !== socket || managedGameId.current !== targetGameId || !shouldBeConnected.current) return;
			try {
				const message: ServerGameEvent = JSON.parse(event.data);
				// console.log('[GameWS] Received:', message); // Debug log
				processGameWebSocketMessage(message);

				// Handle specific server errors like game not found
				if (message?.type === 'error' && (message?.code === 'game_not_found' || message?.code === 'game_over')) {
					console.error(`[GameWS Hook ${targetGameId}] Server reported error: ${message.message} (Code: ${message.code}). Closing connection.`);
					shouldBeConnected.current = false; retryCountRef.current = MAX_RETRIES + 1;
					closeSocket(); // Use the hook's close function for cleanup
					if (useGameStore.getState().gameId === targetGameId) {
						setError(message.message || 'Game not found or has ended.');
						setLoading(false); setConnected(false);
						// Optionally navigate away or clear game state more formally
					}
				}
			} catch (error) {
				console.error(`[GameWS Hook ${targetGameId}] Failed parse/handle message:`, error);
				if (useGameStore.getState().gameId === targetGameId) {
					setError('Error processing server message.');
				}
			}
		};

		socket.onclose = (event) => {
			console.warn(`[GameWS Hook ${targetGameId}] Closed. Code: ${event.code}, Reason: "${event.reason || 'N/A'}", Clean: ${event.wasClean}`);
			if (ws.current !== socket && managedGameId.current !== targetGameId) {
				console.log(`[GameWS Hook ${targetGameId}] onclose for stale socket.`); return;
			}

			ws.current = null; isConnecting.current = false;
			const currentStoreId = useGameStore.getState().gameId;

			if (currentStoreId === targetGameId) {
				setConnected(false);
			}

			const wasUnexpected = !event.wasClean && event.code !== 1000;
			const retryAllowed = retryCountRef.current < MAX_RETRIES;

			if (wasUnexpected && retryAllowed && shouldBeConnected.current && managedGameId.current === targetGameId) {
				const currentRetry = retryCountRef.current++;
				const delay = Math.pow(2, currentRetry) * INITIAL_RETRY_DELAY + Math.random() * 1000;
				console.log(`[GameWS Hook ${targetGameId}] Retrying in ${Math.round(delay / 1000)}s (Attempt ${retryCountRef.current})...`);

				if (currentStoreId === targetGameId) {
					setLoading(true); setError(`Connection lost. Retrying... (${retryCountRef.current})`);
				}

				reconnectTimeoutId.current = setTimeout(() => {
					if (shouldBeConnected.current && managedGameId.current === targetGameId) {
						connectWebSocket(targetGameId);
					} else {
						if (currentStoreId === targetGameId) setLoading(false);
						managedGameId.current = null; retryCountRef.current = 0;
					}
				}, delay);
			} else {
				managedGameId.current = null; retryCountRef.current = 0;
				if (currentStoreId === targetGameId) {
					setLoading(false);
					if (wasUnexpected) {
						setError(retryAllowed ? `Lost connection. Stopped trying.` : `Lost connection after ${MAX_RETRIES} retries.`);
					} else if (event.code !== 1000 && event.code !== 1005 && event.reason) {
						setError(`Disconnected: ${event.reason}`);
					}
				}
				if (!retryAllowed || !wasUnexpected) {
					shouldBeConnected.current = false;
				}
			}
		};

		socket.onerror = (event) => {
			if (ws.current !== socket || !shouldBeConnected.current || managedGameId.current !== targetGameId) return;
			console.error(`[GameWS Hook ${targetGameId}] Error:`, event);
			if (useGameStore.getState().gameId === targetGameId) {
				setError('WebSocket connection error.');
			}
		};

	}, [user, setLoading, setError, setConnected, processGameWebSocketMessage, clearError, closeSocket]); // Added closeSocket dependency

	// Effect to manage connection based on gameId prop changes
	useEffect(() => {
		const currentPropId = gameId;
		// console.log(`[GameWS Hook Effect] Running. Prop gameId: '${currentPropId || 'null'}'. Managed: '${managedGameId.current}'. ShouldBe: ${shouldBeConnected.current}`);

		if (currentPropId) {
			setGameId(currentPropId); // Ensure store knows the target game ID
			if (managedGameId.current !== currentPropId || (!isConnecting.current && ws.current?.readyState !== WebSocket.OPEN)) {
				// console.log(`[GameWS Hook Effect ${currentPropId}] Triggering connection...`);
				shouldBeConnected.current = true; retryCountRef.current = 0;
				connectWebSocket(currentPropId);
			} else {
				shouldBeConnected.current = true;
			}
		} else {
			// console.log(`[GameWS Hook Effect] Prop gameId null. Cleaning up.`);
			shouldBeConnected.current = false;
			if (reconnectTimeoutId.current) clearTimeout(reconnectTimeoutId.current);
			reconnectTimeoutId.current = null;

			if (ws.current && managedGameId.current && ws.current.readyState !== WebSocket.CLOSED && ws.current.readyState !== WebSocket.CLOSING) {
				// console.log(`[GameWS Hook Effect] Closing WebSocket for previous managed ID '${managedGameId.current}'.`);
				ws.current.onclose = null;
				ws.current.close(1000, 'Game ID prop became null/undefined');
				ws.current = null;
			}
			managedGameId.current = null; isConnecting.current = false; retryCountRef.current = 0;
			setGameId(null); // Clear store game ID
			clearGameState(); // Clear the game state in the store
			if (isConnected) setConnected(false);
			if (isLoading) setLoading(false);
		}

		// Cleanup function
		return () => {
			// console.log(`[GameWS Hook Effect Cleanup] Running for propId '${currentPropId || 'N/A'}'.`);
			if (reconnectTimeoutId.current) clearTimeout(reconnectTimeoutId.current);
			reconnectTimeoutId.current = null;
		};
	}, [gameId, connectWebSocket, setConnected, setLoading, isConnected, isLoading, clearGameState, setGameId]);

	/** Sends a JSON message over the active WebSocket connection */
	const sendMessage = useCallback((message: ClientGameAction) => {
		const currentManagedId = managedGameId.current;
		if (!currentManagedId || ws.current?.readyState !== WebSocket.OPEN) {
			console.warn(`sendMessage prevented: Not connected to game '${currentManagedId}'. State: ${ws.current?.readyState}`);
			setError('Not connected to game server.'); // Provide feedback
			return;
		}
		if (useGameStore.getState().gameId !== currentManagedId) {
			console.warn(`sendMessage prevented: Store ID (${useGameStore.getState().gameId}) mismatch with managed game ID (${currentManagedId}).`);
			return;
		}

		try {
			// console.log('[GameWS] Sending:', message); // Debug log
			ws.current.send(JSON.stringify(message));
			setProcessingAction(true); // Indicate client is waiting for server response
		} catch (error) {
			console.error(`[GameWS Hook ${currentManagedId}] Failed to send message:`, error);
			setError('Failed to send action to server.');
			setProcessingAction(false);
		}
	}, [setError, setProcessingAction]); // Dependencies: only functions from store

	/** Explicitly closes the WebSocket connection */
	const closeSocket = useCallback(() => { // Renamed from closeWebSocket to closeSocket for consistency
		console.log(`[GameWS Hook closeSocket] Explicitly closing socket for managed ID ${managedGameId.current}`);
		shouldBeConnected.current = false;
		if (reconnectTimeoutId.current) clearTimeout(reconnectTimeoutId.current);
		reconnectTimeoutId.current = null;
		if (ws.current && ws.current.readyState !== WebSocket.CLOSED && ws.current.readyState !== WebSocket.CLOSING) {
			ws.current.onclose = null;
			ws.current.close(1000, 'User initiated disconnect');
			ws.current = null;
		}
		managedGameId.current = null; isConnecting.current = false; retryCountRef.current = 0;
		if (useGameStore.getState().isConnected) setConnected(false);
		if (useGameStore.getState().isLoading) setLoading(false);
		// Do NOT clear game state here, let component/page handle that decision
	}, [setConnected, setLoading]); // Dependencies: store actions

	return { sendMessage, closeSocket, isConnected, isLoading, error: gameError };
}