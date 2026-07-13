import React from 'react';

export interface PanelProps {
  /** Uppercase eyebrow heading. Omit for a bare card. */
  title?: string;
  /** Right-aligned header slot (e.g. a Button or legend). */
  action?: React.ReactNode;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}

/**
 * Card surface with an optional uppercase title row and action slot.
 *
 * Adapted from design-system-import/ui_kits/platform/shared.jsx Panel for the
 * additive DS preview (cambia-438). Uses the collision-prefixed --ds- radius
 * token so it coexists with Tailwind's default scale.
 */
const Panel: React.FC<PanelProps> = ({ title, action, children, style }) => {
  return (
    <section
      style={{
        background: 'var(--surface-card)',
        border: '1.5px solid var(--border-default)',
        borderRadius: 'var(--ds-radius-lg)',
        boxShadow: 'var(--shadow-card)',
        padding: '16px 18px',
        ...style
      }}
    >
      {(title || action) && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <h3
            style={{
              margin: 0,
              fontSize: 'var(--text-2xs)',
              fontWeight: 'var(--weight-black)',
              letterSpacing: 'var(--tracking-caps)',
              textTransform: 'uppercase',
              color: 'var(--text-tertiary)'
            }}
          >
            {title}
          </h3>
          {action}
        </div>
      )}
      {children}
    </section>
  );
};

export default Panel;
