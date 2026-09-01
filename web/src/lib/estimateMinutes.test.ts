// src/lib/estimateMinutes.test.ts
// Coverage for the queue card's duration estimate (cambia-1518). It used to multiply a per-round
// figure by QueueConfig.Rounds, advertising a duration for a match length nothing plays: a
// matchmade lobby runs no circuit (Circuit.Enabled only turns on through a host update_rules
// edit, service/internal/lobby/lobby.go) and plays exactly one game regardless of the queue's
// round count. The estimate no longer takes rounds as an input at all.
import { describe, expect, it } from 'vitest';
import { estimateMinutes } from './estimateMinutes';

describe('estimateMinutes', () => {
  it('estimates a single head-to-head game', () => {
    expect(estimateMinutes({ players: 2 })).toBe(3);
  });

  it('estimates a single FFA game longer than a head-to-head one', () => {
    const ffa = estimateMinutes({ players: 4 });
    const h2h = estimateMinutes({ players: 2 });
    expect(ffa).toBe(5);
    expect(ffa).toBeGreaterThan(h2h);
  });
});
