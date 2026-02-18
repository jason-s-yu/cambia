// src/hooks/useLobbySocket.ts
import { useEffect, useRef, useCallback } from 'react';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';

const WS_URL = import.meta.env.VITE_WS_URL;
const MAX_RETRIES = 5;
const INITIAL_RETRY_DELAY = 1000; // ms

/**
 * Manages the WebSocket connection for a specific lobby.
 *
 * Connects when a valid `lobbyId` prop is provided and disconnects when it's null/undefined or changes.
 * Handles connection logic, message sending/receiving, state updates via Zustand, and automatic reconnection attempts.
 *
 * @param lobbyId - The ID of the lobby to connect to, or null/undefined to disconnect.
 * @returns An object containing `sendMessage` and `closeSocket` functions.
 */
export function useLobbySocket(lobbyId: string | null | undefined) {
	const ws = useRef<WebSocket | null>(null);
	const reconnectTimeoutId = useRef<number | null>(null);
	const retryCountRef = useRef<number>(0);
	const managedLobbyId = useRef<string | null>(null); // Tracks the ID this hook instance is currently managing
	const isConnecting = useRef<boolean>(false);
	const shouldBeConnected = useRef<boolean>(false); // Explicit intent flag

	const userId = useAuthStore((state) => state.user?.id);
	// Actions and state selectors from the Zustand store for lobby state management
	const {
		setConnected,
		processLobbyWebSocketMessage,
		setError,
		setLoading,
		clearError,
		leaveLobby, // Action from store to clear client-side state
	} = useCurrentLobbyStore();

	/**
	 * Establishes or re-establishes the WebSocket connection to the target lobby.
	 * Handles validation, closing existing connections, event listeners, and error handling.
	 */
	const connectWebSocket = useCallback((targetLobbyId: string) => {
		if (!targetLobbyId || !userId) {
			console.log(`[WS Hook ${targetLobbyId || 'N/A'}] Connect PREVENTED: Invalid targetLobbyId or missing userId.`);
			isConnecting.current = false;
			shouldBeConnected.current = false;
			// Only update store if it thinks it's connecting to this ID
			if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
				setLoading(false);
				setError('Cannot connect: Invalid lobby ID or user not authenticated.');
			}
			return;
		}

		if (managedLobbyId.current === targetLobbyId && (isConnecting.current || ws.current?.readyState === WebSocket.OPEN)) {
			shouldBeConnected.current = true;
			if (ws.current?.readyState === WebSocket.OPEN && useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
				setConnected(true); setLoading(false); clearError();
			}
			return;
		}

		if (reconnectTimeoutId.current) {
			clearTimeout(reconnectTimeoutId.current);
			reconnectTimeoutId.current = null;
		}

		if (ws.current && (managedLobbyId.current !== targetLobbyId || ws.current.readyState === WebSocket.CLOSING)) {
			console.warn(`[WS Hook ${targetLobbyId}] Closing existing WebSocket (Managed: ${managedLobbyId.current}, State: ${ws.current.readyState}) before new connection attempt.`);
			ws.current.onclose = null;
			ws.current.onerror = null;
			ws.current.onmessage = null;
			ws.current.onopen = null;
			ws.current.close(1000, `Switching to new lobby ${targetLobbyId}`);
			ws.current = null;
		}

		if (!WS_URL) {
			console.error(`[WS Hook ${targetLobbyId}] Cannot connect: VITE_WS_URL is not defined.`);
			setError('WebSocket URL is not configured.'); setLoading(false);
			isConnecting.current = false; shouldBeConnected.current = false;
			managedLobbyId.current = null;
			return;
		}

		console.log(`[WS Hook ${targetLobbyId}] Attempting connection (Attempt ${retryCountRef.current + 1})...`);
		isConnecting.current = true;
		shouldBeConnected.current = true;
		managedLobbyId.current = targetLobbyId;

		if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
			setLoading(true); clearError();
		} else {
			console.warn(`[WS Hook ${targetLobbyId}] Attempting connection, but store ID is '${useCurrentLobbyStore.getState().currentLobbyId}'. Hook managing ${managedLobbyId.current}`);
		}

		let socket: WebSocket;
		try {
			socket = new WebSocket(`${WS_URL}/lobby/ws/${targetLobbyId}`, 'lobby');
		} catch (error) {
			console.error(`[WS Hook ${targetLobbyId}] Error creating WebSocket instance:`, error);
			setError('Failed to initialize connection.'); setLoading(false);
			isConnecting.current = false; shouldBeConnected.current = false;
			managedLobbyId.current = null;
			return;
		}
		ws.current = socket;

		socket.onopen = () => {
			if (ws.current !== socket || managedLobbyId.current !== targetLobbyId || !shouldBeConnected.current) {
				console.warn(`[WS Hook ${targetLobbyId}] onopen received for a stale/irrelevant WebSocket instance. Closing.`);
				socket.close(1000, 'Stale connection opened');
				return;
			}
			console.log(`[WS Hook ${targetLobbyId}] WebSocket connected.`);
			retryCountRef.current = 0;
			isConnecting.current = false;
			if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
				setConnected(true); setLoading(false); clearError();
			}
		};

		socket.onmessage = (event) => {
			if (ws.current !== socket || managedLobbyId.current !== targetLobbyId || !shouldBeConnected.current) return;
			try {
				const message = JSON.parse(event.data);
				processLobbyWebSocketMessage(message);

				// Handle lobby not found error directly to stop retries
				if (message?.type === 'error' && message?.code === 'lobby_not_found' && message?.lobby_id === targetLobbyId) {
					console.error(`[WS Hook ${targetLobbyId}] Server reported lobby not found. Stopping connection attempts.`);
					shouldBeConnected.current = false;
					retryCountRef.current = MAX_RETRIES + 1;
					if (ws.current === socket) {
						ws.current.close(1000, 'Lobby not found reported by server');
						ws.current = null;
					}
					managedLobbyId.current = null;
					if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
						setError(message.message || `Lobby ${targetLobbyId.substring(0, 8)} not found.`);
						setLoading(false); setConnected(false);
						leaveLobby(); // Trigger store cleanup
					}
				}
			} catch (error) {
				console.error(`[WS Hook ${targetLobbyId}] Failed to parse message or handle update:`, error);
				if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
					setError('Error processing message from server.');
				}
			}
		};

		socket.onclose = (event) => {
			console.warn(`[WS Hook ${targetLobbyId}] WebSocket closed. Code: ${event.code}, Reason: "${event.reason || 'No reason given'}", Clean: ${event.wasClean}`);
			if (ws.current !== socket && managedLobbyId.current !== targetLobbyId) {
				console.log(`[WS Hook ${targetLobbyId}] onclose received for a stale socket instance. Ignoring closure effects.`);
				return;
			}

			ws.current = null;
			isConnecting.current = false;

			const storeLobbyId = useCurrentLobbyStore.getState().currentLobbyId;
			if (storeLobbyId === targetLobbyId) {
				setConnected(false);
			}

			const wasUnexpected = !event.wasClean && event.code !== 1000;
			const retryAllowed = retryCountRef.current < MAX_RETRIES;

			if (wasUnexpected && retryAllowed && shouldBeConnected.current && managedLobbyId.current === targetLobbyId) {
				const currentRetry = retryCountRef.current++;
				const delay = Math.pow(2, currentRetry) * INITIAL_RETRY_DELAY + Math.random() * 1000;
				console.log(`[WS Hook ${targetLobbyId}] Unexpected disconnect. Retrying in ${Math.round(delay / 1000)}s (Attempt ${retryCountRef.current})...`);

				if (storeLobbyId === targetLobbyId) {
					setLoading(true);
					setError(`Connection lost. Retrying... (Attempt ${retryCountRef.current})`);
				}

				reconnectTimeoutId.current = setTimeout(() => {
					if (shouldBeConnected.current && managedLobbyId.current === targetLobbyId) {
						connectWebSocket(targetLobbyId);
					} else {
						console.log(`[WS Hook ${targetLobbyId}] Retry aborted, connection no longer intended.`);
						if (storeLobbyId === targetLobbyId) setLoading(false);
						managedLobbyId.current = null;
						retryCountRef.current = 0;
					}
				}, delay);
			} else {
				managedLobbyId.current = null;
				retryCountRef.current = 0;

				if (storeLobbyId === targetLobbyId) {
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
			if (ws.current !== socket || !shouldBeConnected.current || managedLobbyId.current !== targetLobbyId) return;
			console.error(`[WS Hook ${targetLobbyId}] WebSocket error occurred:`, event);
			if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
				setError('WebSocket connection error.');
			}
		};

	}, [userId, setLoading, setError, setConnected, processLobbyWebSocketMessage, clearError, leaveLobby]);

	// Effect to manage connection/disconnection based on lobbyId prop changes
	useEffect(() => {
		const currentPropId = lobbyId;
		console.log(`[WS Hook Effect] Running. Prop lobbyId: '${currentPropId || 'null/undefined'}'. Managed lobbyId: '${managedLobbyId.current}', ShouldBeConnected: ${shouldBeConnected.current}`);

		if (currentPropId) {
			if (managedLobbyId.current !== currentPropId || (!isConnecting.current && ws.current?.readyState !== WebSocket.OPEN)) {
				console.log(`[WS Hook Effect ${currentPropId}] Prop ID changed or not connected. Triggering connection...`);
				shouldBeConnected.current = true;
				retryCountRef.current = 0;
				connectWebSocket(currentPropId);
			} else {
				shouldBeConnected.current = true;
			}
		} else {
			console.log(`[WS Hook Effect] Prop lobbyId is invalid/null. Cleaning up.`);
			shouldBeConnected.current = false;

			if (reconnectTimeoutId.current) {
				clearTimeout(reconnectTimeoutId.current);
				reconnectTimeoutId.current = null;
			}

			if (ws.current && managedLobbyId.current && ws.current.readyState !== WebSocket.CLOSED && ws.current.readyState !== WebSocket.CLOSING) {
				console.log(`[WS Hook Effect] Closing WebSocket for previous managed ID '${managedLobbyId.current}'.`);
				ws.current.onclose = null;
				ws.current.close(1000, 'Lobby ID prop became null/undefined');
				ws.current = null;
			} else if (isConnecting.current && managedLobbyId.current) {
				console.log(`[WS Hook Effect] Connection attempt for ${managedLobbyId.current} in progress, setting shouldBeConnected to false.`);
			}

			managedLobbyId.current = null;
			isConnecting.current = false;
			retryCountRef.current = 0;

			if (useCurrentLobbyStore.getState().isConnected) {
				setConnected(false);
				setLoading(false);
			}
		}

		// Cleanup function for the effect
		return () => {
			console.log(`[WS Hook Effect Cleanup] Running for propId '${currentPropId || 'N/A'}'.`);
			if (reconnectTimeoutId.current) {
				clearTimeout(reconnectTimeoutId.current);
				reconnectTimeoutId.current = null;
				console.log(`[WS Hook Effect Cleanup] Cleared reconnect timeout.`);
			}
			// Note: Explicit socket closure on unmount is handled by ensuring `shouldBeConnected` is false
			// if the component unmounts while `lobbyId` is null, or by the main effect logic if `lobbyId` changes.
		};
	}, [lobbyId, connectWebSocket, setConnected, setLoading, clearError]);

	/**
	 * Sends a JSON message over the active WebSocket connection.
	 * Validates connection state and store consistency before sending.
	 */
	const sendMessage = useCallback((message: object) => {
		const currentManagedId = managedLobbyId.current;

		if (!currentManagedId || ws.current?.readyState !== WebSocket.OPEN) {
			console.warn(`sendMessage prevented: Not connected or managing lobby '${currentManagedId}'. WS State: ${ws.current?.readyState}`);
			return;
		}
		if (useCurrentLobbyStore.getState().currentLobbyId !== currentManagedId) {
			console.warn(`sendMessage prevented: Store ID (${useCurrentLobbyStore.getState().currentLobbyId}) mismatch with managed ID (${currentManagedId}).`);
			return;
		}

		try {
			ws.current.send(JSON.stringify(message));
		} catch (error) {
			console.error(`[WS Hook ${currentManagedId}] Failed to send message:`, error);
		}
	}, []); // No dependencies that change frequently

	/**
	 * Explicitly closes the WebSocket connection and cleans up associated state.
	 */
	const closeSocket = useCallback(() => {
		console.log(`[WS Hook closeSocket] Explicitly closing socket for managed ID ${managedLobbyId.current}`);
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
			setConnected(false);
			setLoading(false);
		}
	}, [setConnected, setLoading]);

	return { sendMessage, closeSocket };
}