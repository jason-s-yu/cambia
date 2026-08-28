// src/components/ds/eyebrow.ts
// The one eyebrow style: 11px uppercase, bold, cap-tracked, tertiary text.
// Panels, field labels, table headers and the felt pile labels all drew their
// own copy of it and drifted; wordSpacing is why it is shared. At 11px with
// 0.08em letter-spacing the inter-word gap collapses into the letter gaps and a
// two-word eyebrow reads as one run (RULESHEET, LOBBYCHAT), so the space needs
// widening back out (cambia-876, DL-3 review F11).
import type React from 'react';

export const EYEBROW: React.CSSProperties = {
  fontSize: 'var(--text-2xs)',
  fontWeight: 'var(--weight-bold)',
  letterSpacing: 'var(--tracking-caps)',
  wordSpacing: '0.12em',
  textTransform: 'uppercase',
  color: 'var(--text-tertiary)'
};
