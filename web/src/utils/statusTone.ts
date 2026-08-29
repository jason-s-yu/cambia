// src/utils/statusTone.ts
import type { BadgeProps } from '@/components/ds/core/Badge';

/**
 * Maps a run/process status string to the ds Badge's semantic tone. Shared by
 * the training dashboard's ProcessStatus badge (StatusBadge.tsx) and the
 * Training Runs table's legacy run.status fallback (TrainingPage.tsx), since
 * both status vocabularies resolve to the same five tones. Unrecognized
 * values fall back to neutral.
 */
export function statusTone(status: string): NonNullable<BadgeProps['tone']> {
  switch (status) {
    case 'running':
    case 'completed':
      return 'success';
    case 'crashed':
    case 'failed':
      return 'danger';
    case 'starting':
    case 'stopping':
      return 'warning';
    case 'queued':
      return 'info';
    default:
      return 'neutral';
  }
}
