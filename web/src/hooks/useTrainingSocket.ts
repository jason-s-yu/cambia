// src/hooks/useTrainingSocket.ts
import { useEffect, useRef, useState, useCallback } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';

const MAX_RECONNECT_DELAY = 30_000;
const INITIAL_DELAY = 1000;

export function useTrainingSocket(runName: string | undefined) {
	const [connected, setConnected] = useState(false);
	const ws = useRef<WebSocket | null>(null);
	const reconnectTimeout = useRef<number | null>(null);
	const retryCount = useRef(0);
	const shouldConnect = useRef(false);
	const activeRunName = useRef<string | undefined>(undefined);

	const clearLogBuffer = useTrainingStore((s) => s.clearLogBuffer);

	const connect = useCallback((name: string) => {
		if (!name) return;

		// Determine WS URL from current page location
		const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
		const host = window.location.host;
		const url = `${proto}//${host}/ws/training/${name}/logs`;

		let socket: WebSocket;
		try {
			socket = new WebSocket(url);
		} catch {
			return;
		}
		ws.current = socket;

		socket.onopen = () => {
			if (ws.current !== socket || !shouldConnect.current) {
				socket.close(1000, 'Stale');
				return;
			}
			retryCount.current = 0;
			setConnected(true);
		};

		socket.onmessage = (event) => {
			if (ws.current !== socket || !shouldConnect.current) return;
			try {
				const msg = JSON.parse(event.data);
				if (msg.type === 'log_line' && msg.data?.line != null) {
					useTrainingStore.getState().appendLogLine(msg.data.line);
				} else if (msg.type === 'log_backfill' && Array.isArray(msg.data?.lines)) {
					useTrainingStore.getState().appendLogBackfill(msg.data.lines);
				}
			} catch {
				// Ignore parse errors
			}
		};

		socket.onclose = (event) => {
			if (ws.current !== socket) return;
			ws.current = null;
			setConnected(false);

			const wasUnexpected = !event.wasClean && event.code !== 1000;
			if (wasUnexpected && shouldConnect.current && activeRunName.current === name) {
				const delay = Math.min(
					INITIAL_DELAY * Math.pow(2, retryCount.current),
					MAX_RECONNECT_DELAY
				);
				retryCount.current++;
				reconnectTimeout.current = window.setTimeout(() => {
					if (shouldConnect.current && activeRunName.current === name) {
						connect(name);
					}
				}, delay);
			}
		};

		socket.onerror = () => {
			// onclose will fire after this
		};
	}, []); // eslint-disable-line react-hooks/exhaustive-deps

	useEffect(() => {
		// Cleanup previous connection on runName change
		if (reconnectTimeout.current) {
			clearTimeout(reconnectTimeout.current);
			reconnectTimeout.current = null;
		}
		if (ws.current) {
			ws.current.onclose = null;
			ws.current.close(1000, 'Run changed');
			ws.current = null;
		}
		setConnected(false);
		retryCount.current = 0;
		clearLogBuffer();

		if (runName) {
			shouldConnect.current = true;
			activeRunName.current = runName;
			connect(runName);
		} else {
			shouldConnect.current = false;
			activeRunName.current = undefined;
		}

		return () => {
			shouldConnect.current = false;
			if (reconnectTimeout.current) {
				clearTimeout(reconnectTimeout.current);
				reconnectTimeout.current = null;
			}
			if (ws.current) {
				ws.current.onclose = null;
				ws.current.close(1000, 'Unmount');
				ws.current = null;
			}
		};
	}, [runName, connect, clearLogBuffer]);

	return { connected };
}
