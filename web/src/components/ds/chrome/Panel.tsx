import React from 'react';

/**
 * Panel title (cambia-1097). Was the shared EYEBROW: uppercase, cap-tracked and
 * --text-2xs, the same style the field labels inside the panel use, so the card
 * had no heading a scan could land on and a rule sheet read as one undivided
 * run of small caps. It is a heading now: sentence case, --ds-text-xl, primary
 * text, over a hairline that closes the header band and opens the body. It is
 * the outermost rank on a page, so it sits a step above the --ds-text-lg
 * headings a panel's own sections use.
 */
const PANEL_TITLE: React.CSSProperties = {
  fontSize: 'var(--ds-text-xl)',
  fontWeight: 'var(--weight-bold)',
  letterSpacing: 'var(--ds-tracking-tight)',
  lineHeight: 'var(--ds-leading-tight)',
  color: 'var(--text-primary)'
};

export interface PanelProps {
  /** Section heading. Omit for a bare card. */
  title?: string;
  /** Right-aligned header slot (e.g. a Button or legend). */
  action?: React.ReactNode;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}

/**
 * Flat card surface with an optional heading row and action slot.
 * Separation comes from the ground/surface step plus a 1px border, not from
 * a shadow.
 */
const Panel: React.FC<PanelProps> = ({ title, action, children, style }) => {
  return (
    <section
      style={{
        background: 'var(--surface-1)',
        border: '1px solid var(--border-default)',
        borderRadius: 'var(--ds-radius-lg)',
        padding: 'var(--space-4) var(--space-5)',
        ...style
      }}
    >
      {(title || action) && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12,
            minHeight: 'var(--control-h-sm)',
            marginBottom: 'var(--space-4)',
            paddingBottom: 'var(--space-3)',
            borderBottom: '1px solid var(--border-default)'
          }}
        >
          <h3 style={{ margin: 0, ...PANEL_TITLE }}>{title}</h3>
          {action}
        </div>
      )}
      {children}
    </section>
  );
};

export default Panel;
