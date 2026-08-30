// src/components/game/handLayout.ts
// Where a hand slot sits on screen (cambia-1095).
//
// A hand is dealt as a square, two cards to a row, filled from the row nearest its owner. The
// engine peeks slots 0 and 1 at the deal (engine/game.go, RULES.md section 2), so those two slots
// ARE the near row and have to be drawn as it. Our own hand is drawn at the bottom of the felt, so
// its near row is the bottom one and slot order runs UP the grid; an opponent sits across the
// table, so their near row is the top one and plain row-major order already draws it right.
//
// Penalty draws grow a hand past four (cambia-820), so the row count comes from the live hand size
// and the extra slots extend the square away from the owner.

export const HAND_COLUMNS = 2;

export interface HandSlotPlacement {
  /** 1-based CSS grid line, counted from the top of the grid. */
  gridRow: number;
  /** 1-based CSS grid line, counted from the left. */
  gridColumn: number;
}

/** Rows a hand of `handSize` cards takes at two cards to a row. */
export const handRowCount = (handSize: number): number => Math.max(1, Math.ceil(handSize / HAND_COLUMNS));

/**
 * Grid placement for one slot of the player's OWN hand: slots 0 and 1 on the bottom row, later
 * slots stacked above them, columns left to right within a row. `handSize` is the number of slots
 * drawn, which is the live hand size including the padding backs.
 */
export const ownHandPlacement = (slot: number, handSize: number): HandSlotPlacement => ({
  gridRow: handRowCount(Math.max(handSize, slot + 1)) - Math.floor(slot / HAND_COLUMNS),
  gridColumn: (slot % HAND_COLUMNS) + 1
});
