// src/components/ds/eyebrow.ts
// The one eyebrow style: --text-2xs uppercase, bold, cap-tracked, tertiary
// text. Field labels, table headers and the felt pile labels all drew their own
// copy of it and drifted; wordSpacing is why it is shared. At this size with
// 0.08em letter-spacing the inter-word gap collapses into the letter gaps and a
// two-word eyebrow reads as one run (SNAPPENALTY, JOKERSPERDECK), so the space
// needs widening back out (cambia-876, DL-3 review F11).
//
// It is the bottom rank of the ladder, not a heading: panel titles and rule
// group titles left it in cambia-1097, since a sheet whose sections and whose
// fields were both set in it had no scannable structure at all.
import type React from 'react';

export const EYEBROW: React.CSSProperties = {
  fontSize: 'var(--text-2xs)',
  fontWeight: 'var(--weight-bold)',
  letterSpacing: 'var(--tracking-caps)',
  wordSpacing: '0.12em',
  textTransform: 'uppercase',
  color: 'var(--text-tertiary)'
};
