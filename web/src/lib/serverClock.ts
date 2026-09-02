// src/lib/serverClock.ts
// Reading a server-clock deadline on this client's clock (cambia-488, cambia-1241).
//
// Every deadline the service sends - a turn timer, a snap fill, a reconnect window - is an epoch
// stamp taken from the SERVER's clock, and the frame that carries it also carries `serverNow`, the
// server's reading of the moment it was sent. gameStore stores the difference as
// `serverClockOffsetMs = serverNow - Date.now()`: how far ahead of this client's clock the
// server's clock runs.
//
// Converting a server stamp to this clock therefore SUBTRACTS that offset. A client whose clock
// runs four seconds slow reports offset +4000, and a deadline the server placed fifteen seconds
// out reads as `deadline - 4000` here, which is fifteen seconds from now on this clock. Adding the
// offset instead does not cancel the skew, it doubles it: the countdown then runs long by twice
// the error in either direction, which is exactly the failure the correction exists to prevent.

/** Milliseconds until a server-clock deadline, measured on this client's clock. May be negative. */
export function msUntil(deadlineMs: number, clockOffsetMs = 0, nowMs: number = Date.now()): number {
  return deadlineMs - clockOffsetMs - nowMs;
}

/**
 * Whole seconds until a server-clock deadline, never below zero, or null when there is no
 * deadline. Rounded up, so a window with any time left reads as at least one second rather than
 * announcing zero for its final fraction.
 */
export function secondsUntil(
  deadlineMs: number | null | undefined,
  clockOffsetMs = 0,
  nowMs: number = Date.now()
): number | null {
  if (deadlineMs == null) return null;
  return Math.max(0, Math.ceil(msUntil(deadlineMs, clockOffsetMs, nowMs) / 1000));
}
