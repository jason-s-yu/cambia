// src/components/lobby/circuitScoringCopy.ts
// Copy for the circuit scoring explainer opened from the rule sheet
// (cambia-1100). Kept out of the component so scripts/test-circuit-copy.mjs can
// read it without a DOM, and so the numbers below stay pinned to the server:
// CIRCUIT_DEFAULTS mirrors NewLobbyWithDefaults in service/internal/lobby/lobby.go
// and the test fails if the two drift apart.
//
// Sourced from RULES.md T1-T6, engine/circuit.go (round count, cumulative
// scoring, standings order), engine/scoring.go (ComputeAggressionSubsidy) and
// service/internal/game/game.go (win bonus and false Cambia penalty as
// round-result adjustments, cambia-1009; the circuit branch on a disconnect).
// Nothing here is invented, and the rules the server does not run yet are left
// out rather than described as if it did: nothing reads CircuitRules.TargetScore
// or CircuitRules.FreezeUserOnDisconnect, and nothing calls the engine's
// RecordMissedRound, so the target is named as a stored value only, the freeze
// flag is absent and the 41 point missed round is not claimed.

/** Circuit rule defaults a new lobby is created with (service/internal/lobby/lobby.go). */
export const CIRCUIT_DEFAULTS = {
  targetScore: 100,
  winBonus: -1,
  falseCambiaPenalty: 1
};

/** Signed form for a score adjustment, so a positive penalty reads as one. */
function signed(n: number): string {
  return n > 0 ? `+${n}` : String(n);
}

export interface CircuitCopySection {
  /** Stable id: the explainer's topics are asserted by id in the copy test. */
  id: string;
  heading: string;
  body: string[];
}

export const CIRCUIT_SCORING_TITLE = 'Circuit scoring';

export const CIRCUIT_SCORING_COPY: CircuitCopySection[] = [
  {
    id: 'rounds',
    heading: 'Rounds add up',
    body: [
      'A circuit lobby plays a run of rounds instead of a single game. Each round every player scores their hand, that score is added to their running total, and the lowest total takes the circuit.',
      'The number of rounds comes from the circuit format: 8 for quick, 12 for standard, 20 for championship, and 12 when the lobby names no format. The count has to divide evenly by the number of players, so the deal rotates a whole turn.'
    ]
  },
  {
    id: 'end',
    heading: 'How a circuit ends',
    body: [
      'The circuit ends once its rounds are played. Final standings rank on the lowest cumulative score, and ties break on the lowest pre-subsidy total, then the head to head round record, then the best single round.',
      'A ranked circuit rates its players once, off those final standings. The rounds inside it do not move ratings.',
      `Target score (default ${CIRCUIT_DEFAULTS.targetScore}) is stored with the lobby, but no circuit ends on it: the round count does.`
    ]
  },
  {
    id: 'adjustments',
    heading: 'Win bonus and false Cambia penalty',
    body: [
      `Win bonus (default ${signed(CIRCUIT_DEFAULTS.winBonus)}) lands on each winner's score in that round's result. False Cambia penalty (default ${signed(CIRCUIT_DEFAULTS.falseCambiaPenalty)}) lands on the caller's score when the player who called Cambia does not win the round.`,
      "Both stop at the round result. Circuit totals count the raw hand scores plus a fixed subsidy for the round's top finishers: -3 and 0 heads up, -5, -2, 0 and 0 with three or four players, and -5, -2, -1 and 0 with five or more. The Cambia caller wins ties for that subsidy."
    ]
  },
  {
    id: 'disconnects',
    heading: 'Disconnects',
    body: [
      'A drop in a circuit round does not forfeit the seat. The round plays on around it, the seat is held for 60 seconds, and reconnecting hands the player their cards back.',
      'Forfeit on disconnect and the reconnect grace on the sheet above are single game rules. A circuit round runs that window of its own instead.'
    ]
  },
  {
    id: 'toggle',
    heading: 'The toggle',
    body: [
      'Circuit scoring is a per-lobby setting. On, the target score, win bonus and false Cambia penalty fields open on the sheet and the lobby scores its games as one circuit. Off, every game stands alone and none of this applies.',
      "Circuit rounds are dealt under the tournament rule set: draw from discard on, replace abilities on, the caller's hand unlocked, four cards, two jokers, one deck, a 46 turn cap and Cambia legal from the first round. Of the deal and play settings above, only the snap penalty and the turn clock carry into a circuit round."
    ]
  }
];
