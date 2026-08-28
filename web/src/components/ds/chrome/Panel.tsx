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
 * Flat card surface with an optional uppercase title row and action slot.
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
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 'var(--space-3)' }}>
          <h3
            style={{
              margin: 0,
              fontSize: 'var(--text-2xs)',
              fontWeight: 'var(--weight-bold)',
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
