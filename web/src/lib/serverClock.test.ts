// src/lib/serverClock.test.ts
// The sign convention every countdown on the felt depends on (cambia-1241). gameStore stores
// `serverClockOffsetMs = serverNow - Date.now()`, so a server-clock deadline is read on this
// client's clock by SUBTRACTING it. Adding it instead is the failure this file exists to catch:
// it leaves twice the skew in the answer rather than none, and both signs look plausible in
// isolation because a client with no skew at all cannot tell them apart.
import { describe, expect, it } from 'vitest';
import { msUntil, secondsUntil } from './serverClock';

describe('msUntil', () => {
  it('returns the raw distance when the two clocks agree', () => {
    const now = 1_000_000;
    expect(msUntil(now + 15_000, 0, now)).toBe(15_000);
  });

  it('cancels the skew of a client clock running slow', () => {
    // The client reads 1_000_000 at the instant the server reads 1_004_000: offset +4000. The
    // server placed the deadline fifteen seconds past its own reading of now.
    const clientNow = 1_000_000;
    const serverNow = clientNow + 4_000;
    expect(msUntil(serverNow + 15_000, serverNow - clientNow, clientNow)).toBe(15_000);
  });

  it('cancels the skew of a client clock running fast', () => {
    const clientNow = 1_000_000;
    const serverNow = clientNow - 7_000; // offset -7000
    expect(msUntil(serverNow + 15_000, serverNow - clientNow, clientNow)).toBe(15_000);
  });

  it('goes negative once the deadline has passed', () => {
    const now = 1_000_000;
    expect(msUntil(now - 2_000, 0, now)).toBe(-2_000);
  });
});

describe('secondsUntil', () => {
  it('is null without a deadline', () => {
    expect(secondsUntil(null, 0, 1_000_000)).toBeNull();
    expect(secondsUntil(undefined, 0, 1_000_000)).toBeNull();
  });

  it('rounds a part-second remainder up, so a live window never reads zero', () => {
    const now = 1_000_000;
    expect(secondsUntil(now + 200, 0, now)).toBe(1);
    expect(secondsUntil(now + 44_100, 0, now)).toBe(45);
  });

  it('clamps a passed deadline to zero rather than counting up', () => {
    const now = 1_000_000;
    expect(secondsUntil(now - 30_000, 0, now)).toBe(0);
  });

  it('reads a skewed client\'s window at the length the server set', () => {
    const clientNow = 1_000_000;
    const serverNow = clientNow + 30_000; // this client's clock is half a minute slow
    expect(secondsUntil(serverNow + 45_000, serverNow - clientNow, clientNow)).toBe(45);
  });
});
