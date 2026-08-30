// src/hooks/useTabSession.ts
import { useSyncExternalStore } from 'react';
import { getTabSessionEpoch, getTabLabel, getTabNotice, isPinned, subscribeTabSession } from '@/lib/tabSession';

/**
 * A counter that changes whenever this tab is pinned or unpinned
 * (cambia-1149). Sockets take it as an effect dependency: the identity a
 * connection handshaked with is fixed for the life of that connection, so a
 * pin has to close it and dial again with the new subprotocol list.
 */
export function useTabSessionEpoch(): number {
  return useSyncExternalStore(subscribeTabSession, getTabSessionEpoch, getTabSessionEpoch);
}

/** What this tab is pinned to, re-read on every pin or unpin. */
export function useTabSession(): { epoch: number; pinned: boolean; label: string | null; notice: string | null } {
  const epoch = useTabSessionEpoch();
  return { epoch, pinned: isPinned(), label: getTabLabel(), notice: getTabNotice() };
}
