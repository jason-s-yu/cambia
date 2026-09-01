// src/lib/estimateMinutes.ts
// Rough estimated match length in minutes for a matchmaking queue card, since the matchmaking
// queues endpoint does not return one directly. Grounded on the default 15s turn timer: a
// head-to-head game runs ~2-3 minutes, an FFA game (>2 players) ~4-5.
//
// Not derived from QueueConfig.Rounds: a matchmade lobby plays exactly one game regardless of
// the queue's round count, since Circuit.Enabled only turns on through a host update_rules edit
// and no queue path sets it (service/internal/lobby/lobby.go). QueueConfig.Rounds names a match
// length nothing plays yet, so the estimate is for the one game a queue actually creates
// (cambia-1518, same root as roundCounter.ts's cambia-1126 item 2).
//
// Structural rather than an imported QueueInfo, so a caller with just the two facts (no full
// matchmakingService import) can still get an estimate.

/** The queue facts the estimate is decided from. */
export interface EstimateMinutesInput {
  players: number;
}

export function estimateMinutes(queue: EstimateMinutesInput): number {
  return queue.players > 2 ? 5 : 3;
}
