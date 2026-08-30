// src/components/lobby/DsCircuitInfoModal.tsx
// The circuit scoring explainer, opened from the (i) beside the rule sheet's
// circuit row (cambia-1100). Content only: the dialog contract (Escape, scrim
// click, focus into the panel and back to the opener, Tab kept inside) is
// ds/core/Modal's, the same dialog the New lobby form on the dashboard uses.
// Copy lives in circuitScoringCopy.ts so it can be tested without a DOM.
import React from 'react';
import Modal from '@/components/ds/core/Modal';
import Button from '@/components/ds/core/Button';
import { CIRCUIT_SCORING_COPY, CIRCUIT_SCORING_TITLE } from './circuitScoringCopy';

export interface DsCircuitInfoModalProps {
  open: boolean;
  onClose: () => void;
}

const SECTION_HEADING: React.CSSProperties = {
  display: 'block',
  fontWeight: 'var(--weight-bold)',
  fontSize: 'var(--text-md)',
  color: 'var(--text-primary)'
};

const DsCircuitInfoModal: React.FC<DsCircuitInfoModalProps> = ({ open, onClose }) => (
  <Modal
    open={open}
    title={CIRCUIT_SCORING_TITLE}
    onClose={onClose}
    width={520}
    footer={<Button variant='secondary' onClick={onClose}>Close</Button>}
  >
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
      {CIRCUIT_SCORING_COPY.map((section) => (
        <section key={section.id} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
          <span style={SECTION_HEADING}>{section.heading}</span>
          {section.body.map((paragraph) => (
            <p key={paragraph.slice(0, 32)} style={{ margin: 0, lineHeight: 1.5 }}>{paragraph}</p>
          ))}
        </section>
      ))}
    </div>
  </Modal>
);

export default DsCircuitInfoModal;
