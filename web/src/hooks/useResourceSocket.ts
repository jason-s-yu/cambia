// src/hooks/useResourceSocket.ts
import { useEffect, useRef, useState, useCallback } from 'react';
import type { ResourceSnapshot } from '@/types/training';

const MAX_RECONNECT_DELAY = 30_000;
const INITIAL_DELAY = 1000;
// Mirrors the server-side ring size (60 samples) so the local history window
// stays consistent with the backfill the server sends on connect.
const MAX_HISTORY = 60;

export interface UseResourceSocketResult {
	connected: boolean;
	snapshot: ResourceSnapshot | null;
	history: ResourceSnapshot[];
}

/**
 * Connects to the single global resource stream (/ws/training/resources),
 * consuming resource_backfill (initial ring) then resource_sample (live)
 * frames. Reconnects with exponential backoff, mirroring useTrainingSocket.
 * State (snapshot + history) is held locally, not in the training store.
 */
export function useResourceSocket(): UseResourceSocketResult {
	const [connected, setConnected] = useState(false);
	const [snapshot, setSnapshot] = useState<ResourceSnapshot | null>(null);
	const [history, setHistory] = useState<ResourceSnapshot[]>([]);

	const ws = useRef<WebSocket | null>(null);
	const reconnectTimeout = useRef<number | null>(null);
	const retryCount = useRef(0);
	const shouldConnect = useRef(false);

	const connect = useCallback(() => {
		const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
		const host = window.location.host;
		const url = `${proto}//${host}/ws/training/resources`;

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
				if (msg.type === 'resource_backfill' && Array.isArray(msg.data?.samples)) {
					const samples = msg.data.samples as ResourceSnapshot[];
					setHistory(samples.slice(-MAX_HISTORY));
					if (samples.length > 0) setSnapshot(samples[samples.length - 1]);
				} else if (msg.type === 'resource_sample' && msg.data) {
					const sample = msg.data as ResourceSnapshot;
					setSnapshot(sample);
					setHistory((prev) => [...prev, sample].slice(-MAX_HISTORY));
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
			if (wasUnexpected && shouldConnect.current) {
				const delay = Math.min(
					INITIAL_DELAY * Math.pow(2, retryCount.current),
					MAX_RECONNECT_DELAY
				);
				retryCount.current++;
				reconnectTimeout.current = window.setTimeout(() => {
					if (shouldConnect.current) connect();
				}, delay);
			}
		};

		socket.onerror = () => {
			// onclose fires after this
		};
	}, []);

	useEffect(() => {
		shouldConnect.current = true;
		connect();

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
	}, [connect]);

	return { connected, snapshot, history };
}
