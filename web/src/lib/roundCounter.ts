// src/lib/roundCounter.ts
// Whether the "Round X/Y" readout has a circuit behind it (cambia-1126 item 2).
//
// The counter used to render off match state alone, which is a round count with nothing playing
// it. Two facts have to meet for a match to run more than one round, and matchmaking supplies
// only one of them:
//
//   - TotalRounds reaches the client from SearchLobbyHandler, which sets it on the hub off the
//     queue config and does nothing else with it (service/internal/handlers/lobby.go).
//   - A circuit is what actually plays those rounds, and lobby Circuit.Enabled turns on only
//     through a host update_rules edit (service/internal/lobby/lobby.go). No queue sets it, and
//     a preset cannot express a round count either (service/internal/lobby/presets.go).
//
// So a matchmade h2h_rapid lobby carries totalRounds 8, has no circuit, and plays one game. The
// felt read ROUND 0/8 for the whole of it, with no round_start ever emitted to move the numerator
// off zero (cambia-466 owns the round lifecycle).
//
// The counter is therefore shown only where a circuit exists, and hidden everywhere else rather
// than made to agree with itself while counting nothing. It comes back on its own the day a
// circuit lobby runs its rounds, since that is exactly the state it reads.
//
// Types are structural rather than imported, for the reason lobbyPreset.ts states: node --test
// loads this module directly (web/scripts/test-round-counter.mjs), stripping types and resolving
// no bundler aliases.

/** The lobby and match facts the counter is decided from. */
export interface RoundCounterInput {
  /** lobbyDetails.circuit.enabled, absent on a lobby snapshot that carried no circuit at all. */
  circuitEnabled?: boolean;
  /** matchState.totalRounds. A match with no circuit reports the queue's figure regardless. */
  totalRounds?: number;
  /** matchState.currentRound, which only round_start moves. */
  currentRound?: number;
}

/**
 * The "Round X/Y" label, or null where there is no circuit to count.
 *
 * A single-round circuit reads no counter either: "Round 1/1" is the whole match, and saying so
 * on every surface is the same noise the h2h_quickplay counter was (cambia-933).
 */
export function roundCounterLabel(input: RoundCounterInput | null | undefined): string | null {
  if (!input || !input.circuitEnabled) return null;
  const total = input.totalRounds ?? 0;
  if (total <= 1) return null;
  return `Round ${input.currentRound ?? 0}/${total}`;
}
