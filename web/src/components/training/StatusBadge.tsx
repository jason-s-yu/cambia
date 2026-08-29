// src/components/training/StatusBadge.tsx
import React from 'react';
import Badge from '@/components/ds/core/Badge';
import { statusTone } from '@/utils/statusTone';
import type { ProcessStatus } from '@/types/training';

interface StatusBadgeProps {
	status: ProcessStatus;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => (
	<Badge tone={statusTone(status)} dot>{status}</Badge>
);

export default StatusBadge;
