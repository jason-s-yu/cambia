// src/hooks/useSocket.ts
// Unified WebSocket hook - single connection to /ws/{lobbyId}, subprotocol "cambia".
// Replaces the separate useLobbySocket and useGameSocket hooks.
import { useEffect, useRef, useCallback } from 'react';
import { useAuthStore } from '@/stores/authStore';
import { useCurrentLobbyStore } from '@/stores/lobbyStore';
import { useGameStore } from '@/stores/gameStore';
import { WS_URL } from '@/lib/runtimeEnv';
import { wsProtocols } from '@/lib/tabSession';
import { useTabSessionEpoch } from '@/hooks/useTabSession';
import { ackOutbound, cardRefsOf, isLobbyFrame, recordOutbound, resolveOutbox, tableContext, type OutboxEntry } from '@/lib/resendDecision';
import { NIL as NIL_UUID } from 'uuid';
const MAX_RETRIES = 5;
const INITIAL_RETRY_DELAY = 1000;

/** The shape the hub parses the path segment as: `/ws/{lobbyId}` runs uuid.Parse on it and
 *  answers 400 for anything else (service/internal/handlers/ws.go HubWSHandler). */
const LOBBY_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/**
 * Whether an id is worth opening a socket for. A blank, malformed or nil id dials a URL the hub
 * refuses outright, and that refusal is not transient, so the retry path redialed it until the
 * page went away (cambia-1126 item 1). The nil UUID is what the stores hold for "no id yet", so
 * it names no lobby either. Checked before the socket is constructed rather than after the
 * handshake fails, because a dial that cannot succeed should never be made.
 */
function isDialableLobbyId(id: string | null | undefined): id is string {
	return typeof id === 'string' && LOBBY_ID_PATTERN.test(id) && id !== NIL_UUID;
}

/**
 * Drops this hook's hold on a socket and closes it. The handlers come off first, so nothing the
 * socket does afterwards reaches the hook.
 *
 * A handshake still in flight is aborted here, not waited out, even though close() in CONNECTING
 * is what makes a browser log "WebSocket is closed before the connection is established" (note
 * cambia-1174, note cambia-985 R4). Waiting for the open event would silence the log, but opening
 * the socket is what joins the lobby: the hub calls MarkJoinedUnsafe on connect and says in as
 * many words that a client which has just left must close before releasing membership, or its own
 * reconnect hands the membership straight back (service/internal/handlers/ws.go step 7). A late
 * handshake landing after POST /lobby/{id}/leave would do exactly that, so a console line is the
 * cheaper of the two.
 *
 * That log line was never really about this function. It appeared on every Leave because
 * closeSocket's own store writes re-ran the connect effect and opened a second socket to the
 * lobby just left, which the redirect then closed mid-handshake; the same identity churn behind
 * the retry cap (cambia-1236). With the effect no longer re-running, no second socket is opened
 * and there is nothing mid-handshake to close.
 */
function releaseSocket(socket: WebSocket, reason: string): void {
	socket.onopen = null;
	socket.onmessage = null;
	socket.onerror = null;
	socket.onclose = null;
	if (socket.readyState === WebSocket.CLOSING || socket.readyState === WebSocket.CLOSED) return;
	socket.close(1000, reason);
}

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
	/** Slot-addressed frames carry idx (and, for a two-sided ability, the owner); the resend
	 *  decision needs both, since the server resolves those frames by index alone. */
	card?: { id?: string; idx?: number; user?: { id?: string } };
	card1?: { id?: string; idx?: number; user?: { id?: string } };
	card2?: { id?: string; idx?: number; user?: { id?: string } };
}

/** Lobby-phase message types routed to lobbyStore */
const LOBBY_TYPES = new Set([
	// search_status and match_found are the matchmaking pair: the hub emits both, lobbyStore has
	// always handled both, and until cambia-933 neither was routed here, so a found match reached
	// the client as an "Unknown message type" warning and the search never resolved.
	'lobby_state', 'phase_change', 'chat', 'game_start', 'search_status', 'match_found',
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
	/** The lobby whose retry budget is spent, and what spent it. A cap is only a cap if exhausting
	 *  it is remembered: the connect effect dials whenever the socket is not open, so a counter on
	 *  its own was reset and spent again on every re-run (cambia-1236). Cleared by a successful
	 *  open, an explicit close, a different lobby, or a tab identity change: the four things that
	 *  make a fresh dial worth trying again.
	 *
	 *  The reason is carried because a fifth thing makes one worth trying again, and only for one
	 *  of the two: a tab coming back to the foreground. `retries` says the network was unusable a
	 *  moment ago, which a return is evidence against; `refused` says the hub does not have this
	 *  lobby, which returning to the tab does not change (cambia-1521). */
	const gaveUp = useRef<{ lobbyId: string; reason: 'retries' | 'refused' } | null>(null);
	const lastSeqRef = useRef<number>(0);
	/** Frames the hub has neither applied nor answered. A repair window can swallow more than
	 *  one, so this is a queue rather than a slot (cambia-913 F2), and the events an accepted
	 *  frame produces take it back out (cambia-913 R1). Reasoning: resendDecision.ts. */
	const outboxRef = useRef<OutboxEntry<OutboundMessage>[]>([]);

	const userId = useAuthStore((state) => state.user?.id);
	/** Bumped when this tab is pinned or unpinned; the connect effect redials on a change. */
	const sessionEpoch = useTabSessionEpoch();
	const lastEpoch = useRef<number>(sessionEpoch);

	// Selected one action at a time, never a bare `useCurrentLobbyStore()`. The bare call returns
	// the whole state object, whose identity changes on every write to the store, which re-created
	// connectWebSocket and re-ran the connect effect on each one; the effect body then reset the
	// retry counter, so the cap below was never reached and a dead server was dialed 521 times in
	// one session (cambia-1236). These functions are defined once by create() and never replaced,
	// so each selector returns a reference that survives every write.
	const setConnected = useCurrentLobbyStore((s) => s.setConnected);
	const setLoading = useCurrentLobbyStore((s) => s.setLoading);
	const setError = useCurrentLobbyStore((s) => s.setError);
	const clearError = useCurrentLobbyStore((s) => s.clearError);
	const leaveLobby = useCurrentLobbyStore((s) => s.leaveLobby);

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
		outboxRef.current = recordOutbound(outboxRef.current, {
			message,
			record: {
				type: message.type,
				cardRefs: cardRefsOf(message),
				sentSeq,
				attempt,
				sentAt: Date.now(),
				ctx: context()
			}
		});
		return true;
	}, [context]);

	const connectWebSocket = useCallback((targetLobbyId: string) => {
		if (!isDialableLobbyId(targetLobbyId) || !userId) {
			isConnecting.current = false;
			shouldBeConnected.current = false;
			if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
				setLoading(false);
				setError('Cannot connect: Invalid lobby ID or user not authenticated.');
			}
			return;
		}

		if (managedLobbyId.current === targetLobbyId && (isConnecting.current || ws.current?.readyState === WebSocket.OPEN)) {
			shouldBeConnected.current = true;
			if (ws.current?.readyState === WebSocket.OPEN) {
				setConnected(true);
				setLoading(false);
				clearError();
			}
			return;
		}

		if (reconnectTimeoutId.current) {
			clearTimeout(reconnectTimeoutId.current);
			reconnectTimeoutId.current = null;
		}

		if (ws.current && (managedLobbyId.current !== targetLobbyId || ws.current.readyState === WebSocket.CLOSING)) {
			releaseSocket(ws.current, `Switching to lobby ${targetLobbyId}`);
			ws.current = null;
		}

		if (!WS_URL) {
			setError('WebSocket URL is not configured.');
			setLoading(false);
			isConnecting.current = false;
			shouldBeConnected.current = false;
			managedLobbyId.current = null;
			return;
		}

		isConnecting.current = true;
		shouldBeConnected.current = true;
		managedLobbyId.current = targetLobbyId;
		lastSeqRef.current = 0;
		outboxRef.current = [];

		if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
			setLoading(true);
			clearError();
		}

		let socket: WebSocket;
		try {
			// Browsers cannot set headers on a handshake, so a pinned tab's token
			// rides the subprotocol list as a second entry (cambia-1149). The
			// server always selects "cambia"; the token entry is read off
			// Sec-WebSocket-Protocol and never selected.
			socket = new WebSocket(`${WS_URL}/ws/${targetLobbyId}`, wsProtocols());
		} catch {
			setError('Failed to initialize connection.');
			setLoading(false);
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
			gaveUp.current = null;
			isConnecting.current = false;
			if (useCurrentLobbyStore.getState().currentLobbyId === targetLobbyId) {
				setConnected(true);
				setLoading(false);
				clearError();
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

				// An accepted frame is answered by the events applying it produces, never by a
				// reply naming it, so those events are its acknowledgement and they take it out
				// of the outbox. Without this a later repair would judge frames the hub had
				// already applied and call them dropped (cambia-913 R1). The hub dispatches on
				// one goroutine and each connection is FIFO, so an accepted frame's events reach
				// this client ahead of any repair produced after them.
				if (outboxRef.current.length > 0) {
					outboxRef.current = ackOutbound(outboxRef.current, type, payload, useAuthStore.getState().user?.id ?? null);
				}

				// Route by message type
				if (type === 'sync_state') {
					// Desync recovery. The repair carries the hub's lobby snapshot and the seq to
					// catch up to; the game snapshot rides its own private_sync_state, and one FIFO
					// writer per connection means every frame this client was behind on has already
					// been applied, so the state read below is current.
					useCurrentLobbyStore.getState().forceSync(payload);
					useGameStore.getState().forceSync(payload);

					// The frames the hub discarded. Without a resend the player's action is simply
					// gone (cambia-891); with a stale one it can mean something else entirely, so
					// every outstanding frame is re-checked against the repaired board. A window
					// can hold more than one frame and each gets its own answer (cambia-913 F2).
					if (outboxRef.current.length > 0) {
						const syncSeq = typeof payload?.seq === 'number' ? payload.seq : lastSeqRef.current;
						const { pending, resend, notify } = resolveOutbox(outboxRef.current, context(), syncSeq);
						outboxRef.current = pending;
						// A lost lobby frame is told about in the lobby, a lost action on the table:
						// the player is looking at one of the two (cambia-913 F4).
						for (const rec of notify) {
							if (isLobbyFrame(rec)) useCurrentLobbyStore.getState().noteDroppedAction();
							else useGameStore.getState().noteDroppedAction();
						}
						// send() puts each one back in the outbox at its new seq.
						for (const entry of resend) send(entry.message, entry.record.attempt + 1);
					}
				} else if (type === 'error') {
					// Errors go to both stores
					useCurrentLobbyStore.getState().processLobbyWebSocketMessage(type, payload);
					useGameStore.getState().processGameWebSocketMessage(type, payload);
				} else if (type === 'game_results') {
					// Dual-route (cambia-763 F2): game_results starts with "game_" so isGameType would
					// claim it before LOBBY_TYPES is even consulted, but its lobby_status snapshot is
					// how the post-game reset (ReadyStates cleared, InGame false - see api_server.go
					// attachOnGameEnd) reaches the client at the moment the game ends, ahead of the
					// lobby_state the hub broadcasts when the results screen closes (cambia-1238).
					// gameStore still needs it too (winner/scores, duplicated from game_end).
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
					// A lobby the hub says does not exist is not a transient failure, so this is the last
					// dial for that id until the hook is pointed somewhere else. Coming back to the tab
					// does not make it exist either, which is why the reason is recorded.
					gaveUp.current = { lobbyId: targetLobbyId, reason: 'refused' };
					retryCountRef.current = 0;
					if (ws.current === socket) {
						releaseSocket(socket, 'Lobby not found');
						ws.current = null;
					}
					managedLobbyId.current = null;
					setError(payload.message || 'Lobby not found.');
					setLoading(false);
					setConnected(false);
					leaveLobby();
				}
			} catch (error) {
				console.error('[useSocket] Failed to parse message:', error);
				setError('Error processing message from server.');
			}
		};

		socket.onclose = (event) => {
			if (ws.current !== socket && managedLobbyId.current !== targetLobbyId) return;

			ws.current = null;
			isConnecting.current = false;

			const storeLobbyId = useCurrentLobbyStore.getState().currentLobbyId;
			if (storeLobbyId === targetLobbyId) {
				setConnected(false);
				useGameStore.getState().setConnected(false);
			}

			const wasUnexpected = !event.wasClean && event.code !== 1000;
			const retryAllowed = retryCountRef.current < MAX_RETRIES;

			if (wasUnexpected && retryAllowed && shouldBeConnected.current && managedLobbyId.current === targetLobbyId) {
				const currentRetry = retryCountRef.current++;
				const delay = Math.pow(2, currentRetry) * INITIAL_RETRY_DELAY + Math.random() * 1000;

				if (storeLobbyId === targetLobbyId) {
					setLoading(true);
					setError(`Connection lost. Retrying... (Attempt ${retryCountRef.current})`);
				}

				reconnectTimeoutId.current = window.setTimeout(() => {
					if (shouldBeConnected.current && managedLobbyId.current === targetLobbyId) {
						connectWebSocket(targetLobbyId);
					} else {
						if (storeLobbyId === targetLobbyId) setLoading(false);
						managedLobbyId.current = null;
						retryCountRef.current = 0;
					}
				}, delay);
			} else {
				managedLobbyId.current = null;
				retryCountRef.current = 0;
				// Budget spent on a drop that was worth retrying: stop dialing this lobby. Without the
				// latch the connect effect dials again the next time it runs, which is how the cap went
				// unenforced (cambia-1236 AC2).
				if (wasUnexpected && !retryAllowed) gaveUp.current = { lobbyId: targetLobbyId, reason: 'retries' };

				if (storeLobbyId === targetLobbyId) {
					setLoading(false);
					if (wasUnexpected) {
						setError(retryAllowed ? 'Lost connection. Stopped trying.' : `Lost connection after ${MAX_RETRIES} retries.`);
					} else if (event.code !== 1000 && event.code !== 1005 && event.reason) {
						setError(`Disconnected: ${event.reason}`);
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
				setError('WebSocket connection error.');
			}
		};

	}, [userId, setConnected, setLoading, setError, clearError, leaveLobby, context, send]);

	// Connect/disconnect based on lobbyId prop
	useEffect(() => {
		// A pin or unpin changes who this tab is, and the identity a connection
		// handshaked with is fixed for its lifetime, so the open socket is dropped
		// here and the block below dials again with the new protocol list
		// (cambia-1149). Deliberate, so it closes 1000 and the retry path leaves
		// it alone; managedLobbyId is cleared so the reconnect is not read as an
		// already-connected no-op.
		if (lastEpoch.current !== sessionEpoch) {
			lastEpoch.current = sessionEpoch;
			if (ws.current) {
				releaseSocket(ws.current, 'Tab identity changed');
				ws.current = null;
			}
			if (reconnectTimeoutId.current) {
				clearTimeout(reconnectTimeoutId.current);
				reconnectTimeoutId.current = null;
			}
			managedLobbyId.current = null;
			isConnecting.current = false;
			retryCountRef.current = 0;
			gaveUp.current = null;
		}

		// A lobby the hook is no longer pointed at has no claim on the retry budget.
		if (gaveUp.current !== null && gaveUp.current.lobbyId !== lobbyId) {
			gaveUp.current = null;
			retryCountRef.current = 0;
		}

		if (isDialableLobbyId(lobbyId)) {
			// Nothing here resets retryCountRef: the counter belongs to the socket's own open event and
			// to nothing else, or the cap cannot hold across an effect re-run (cambia-1236 AC2).
			const backoffPending = reconnectTimeoutId.current !== null && managedLobbyId.current === lobbyId;
			const needsDial = managedLobbyId.current !== lobbyId || (!isConnecting.current && ws.current?.readyState !== WebSocket.OPEN);
			if (gaveUp.current?.lobbyId === lobbyId) {
				// Budget spent. The onclose branch has already written the copy the table reads.
				shouldBeConnected.current = false;
			} else if (needsDial && !backoffPending) {
				shouldBeConnected.current = true;
				connectWebSocket(lobbyId);
			} else {
				shouldBeConnected.current = true;
			}
		} else if (lobbyId) {
			// Named, but not an id the hub could serve. Refusing it here is what keeps it out of the
			// retry path entirely (cambia-1236 AC3).
			shouldBeConnected.current = false;
			managedLobbyId.current = null;
			isConnecting.current = false;
			retryCountRef.current = 0;
			if (reconnectTimeoutId.current) {
				clearTimeout(reconnectTimeoutId.current);
				reconnectTimeoutId.current = null;
			}
			if (ws.current) {
				releaseSocket(ws.current, 'Invalid lobby ID');
				ws.current = null;
			}
			if (useCurrentLobbyStore.getState().currentLobbyId === lobbyId) {
				setLoading(false);
				setError('Cannot connect: Invalid lobby ID.');
			}
		} else {
			shouldBeConnected.current = false;

			if (reconnectTimeoutId.current) {
				clearTimeout(reconnectTimeoutId.current);
				reconnectTimeoutId.current = null;
			}

			if (ws.current) {
				releaseSocket(ws.current, 'Lobby ID became null');
				ws.current = null;
			}

			managedLobbyId.current = null;
			isConnecting.current = false;
			retryCountRef.current = 0;
			gaveUp.current = null;

			if (useCurrentLobbyStore.getState().isConnected) {
				setConnected(false);
				setLoading(false);
			}
		}

		return () => {
			if (reconnectTimeoutId.current) {
				clearTimeout(reconnectTimeoutId.current);
				reconnectTimeoutId.current = null;
			}
		};
	}, [lobbyId, connectWebSocket, setConnected, setLoading, setError, sessionEpoch]);

	/**
	 * Reconnect when the tab comes back (cambia-1521).
	 *
	 * The mechanism chosen for the backgrounded-tab forfeit is client-side: the server's reconnect
	 * grace stays where it is (DisconnectGraceSec 90 under ForfeitOnDisconnect, service/internal/
	 * game/rules.go DefaultHouseRules), and the client's job is to get back inside it. The grace is
	 * a rule about how long a table waits for an absent player, not a browser allowance, so
	 * stretching it to cover tab throttling would make every genuine walk-off cost the table a
	 * longer stall (MATCHMAKING.md 8).
	 *
	 * What the client was missing is the return itself. Nothing above reacts to the page's
	 * lifecycle, and every recovery path it does have runs on window.setTimeout, which a hidden tab
	 * throttles: Chrome clamps a background page to roughly one timer wake-up a minute once it has
	 * been hidden for five, so a backoff scheduled for 1s can land after the 60s grace has already
	 * closed and the seat has been forfeited while the client still shows "Retrying". A tab
	 * restored from the back/forward cache never ran the backoff at all: its socket was closed on
	 * the way in and its timers were frozen behind it.
	 *
	 * No client keepalive is added with it. The connection is already pinged from the server every
	 * 30s with a 5s deadline (service/internal/hub/connection.go WritePump), which is what notices
	 * a socket that died while the tab slept, and an application-level ping would buy nothing while
	 * the socket is open and cannot run at all once it is not.
	 */
	useEffect(() => {
		if (!isDialableLobbyId(lobbyId)) return;

		const attemptResume = () => {
			// An open socket needs nothing, and a dial already in flight is the attempt.
			if (isConnecting.current || ws.current?.readyState === WebSocket.OPEN) return;

			if (gaveUp.current?.lobbyId === lobbyId) {
				// A lobby the hub refused by name is refused just as hard on return.
				if (gaveUp.current.reason !== 'retries') return;
				// A spent retry budget is a statement about the network a moment ago, and a return to
				// the foreground is evidence against it, so it buys one fresh budget. This is the only
				// reset that is not the socket's own open event, and it holds the cap it was added to
				// (cambia-1236 AC2): a store write, a re-render or an effect re-run cannot produce a
				// page-lifecycle event, so the dial count stays bounded by the user's own tab switches
				// rather than running free.
				gaveUp.current = null;
				retryCountRef.current = 0;
			} else if (!shouldBeConnected.current) {
				// Nothing wants this socket: an explicit closeSocket, or a clean close from the hub.
				return;
			}

			// A backoff still waiting is this attempt, made now rather than whenever the throttle
			// next allows it, so it costs no extra budget: the retry it belongs to was counted when
			// it was scheduled.
			if (reconnectTimeoutId.current !== null) {
				clearTimeout(reconnectTimeoutId.current);
				reconnectTimeoutId.current = null;
			}
			shouldBeConnected.current = true;
			connectWebSocket(lobbyId);
		};

		// visibilitychange also fires on the way out, which is not a return.
		const onVisibilityChange = () => {
			if (document.visibilityState === 'visible') attemptResume();
		};
		// pageshow covers the back/forward cache, where the page resumes without ever having been
		// hidden. `resume` is the Page Lifecycle counterpart to `freeze`: a tab thawed while still
		// in the background can run again, and saving the seat does not wait on the user looking.
		document.addEventListener('visibilitychange', onVisibilityChange);
		window.addEventListener('pageshow', attemptResume);
		document.addEventListener('resume', attemptResume);
		return () => {
			document.removeEventListener('visibilitychange', onVisibilityChange);
			window.removeEventListener('pageshow', attemptResume);
			document.removeEventListener('resume', attemptResume);
		};
	}, [lobbyId, connectWebSocket]);

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
		outboxRef.current = [];

		if (reconnectTimeoutId.current) {
			clearTimeout(reconnectTimeoutId.current);
			reconnectTimeoutId.current = null;
		}
		if (ws.current) {
			releaseSocket(ws.current, 'User initiated disconnect');
			ws.current = null;
		}
		managedLobbyId.current = null;
		isConnecting.current = false;
		retryCountRef.current = 0;
		gaveUp.current = null;

		if (useCurrentLobbyStore.getState().isConnected) {
			setConnected(false);
			setLoading(false);
		}
	}, [setConnected, setLoading]);

	/**
	 * Dials the lobby again after an explicit close. The leave sequence closes the socket before
	 * it asks the server to release the membership, because a socket left open to reconnect hands
	 * the membership straight back; when the server then refuses the leave the player is still at
	 * the table, and without a way back the close has taken their connection for nothing
	 * (cambia-1520).
	 *
	 * Not the connect effect's job: closeSocket clears managedLobbyId and shouldBeConnected, and
	 * nothing in the effect's dependencies changes on a refusal, so it does not re-run. The retry
	 * budget is reset with the dial - this is a fresh decision to be connected, not a continuation
	 * of the run that was closed (cambia-1236's cap belongs to a socket that dropped by itself).
	 */
	const reopenSocket = useCallback(() => {
		if (!isDialableLobbyId(lobbyId)) return;
		retryCountRef.current = 0;
		gaveUp.current = null;
		shouldBeConnected.current = true;
		connectWebSocket(lobbyId);
	}, [lobbyId, connectWebSocket]);

	const isConnected = useCurrentLobbyStore((s) => s.isConnected);
	const isLoading = useCurrentLobbyStore((s) => s.isLoading);
	const error = useCurrentLobbyStore((s) => s.error);

	return { sendMessage, closeSocket, reopenSocket, isConnected, isLoading, error };
}
