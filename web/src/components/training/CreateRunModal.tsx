// src/components/training/CreateRunModal.tsx
import React, { useEffect, useState } from 'react';
import Modal from '@/components/common/Modal';
import Button from '@/components/common/Button';
import { useTrainingStore } from '@/stores/trainingStore';
import type { CreateRunRequest } from '@/types/training';

interface CreateRunModalProps {
	isOpen: boolean;
	onClose: () => void;
	onCreated?: (name: string) => void;
}

interface OverrideForm {
	iterations: string;
	device: string;
	seed: string;
}

const DEFAULT_OVERRIDES: OverrideForm = { iterations: '', device: 'cpu', seed: '' };

function buildOverrides(form: OverrideForm): Record<string, string | number> {
	const overrides: Record<string, string | number> = {};
	if (form.iterations.trim() !== '') {
		const n = Number(form.iterations);
		if (!Number.isNaN(n)) overrides['prt_cfr.iterations'] = n;
	}
	if (form.device.trim() !== '') {
		overrides['prt_cfr.device'] = form.device.trim();
	}
	if (form.seed.trim() !== '') {
		const n = Number(form.seed);
		if (!Number.isNaN(n)) overrides['prt_cfr.seed'] = n;
	}
	return overrides;
}

function buildYamlPreview(
	name: string,
	template: string,
	overrides: Record<string, string | number>,
): string {
	const lines = [`name: ${name || '<unnamed>'}`, `template: ${template || '<none selected>'}`];
	const keys = Object.keys(overrides);
	if (keys.length > 0) {
		lines.push('overrides:');
		for (const k of keys) lines.push(`  ${k}: ${overrides[k]}`);
	}
	return lines.join('\n');
}

const CreateRunModal: React.FC<CreateRunModalProps> = ({ isOpen, onClose, onCreated }) => {
	const { createRun, fetchTemplates, templates } = useTrainingStore();

	const [name, setName] = useState('');
	const [template, setTemplate] = useState('');
	const [overrideForm, setOverrideForm] = useState<OverrideForm>(DEFAULT_OVERRIDES);
	const [showPreview, setShowPreview] = useState(false);
	const [isSubmitting, setIsSubmitting] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		if (isOpen) fetchTemplates();
	}, [isOpen, fetchTemplates]);

	useEffect(() => {
		if (!template && templates.length > 0) setTemplate(templates[0]);
	}, [templates, template]);

	useEffect(() => {
		if (!isOpen) {
			setName('');
			setTemplate('');
			setOverrideForm(DEFAULT_OVERRIDES);
			setShowPreview(false);
			setError(null);
		}
	}, [isOpen]);

	const handleClose = () => {
		if (isSubmitting) return;
		onClose();
	};

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();
		if (!name.trim() || !template) {
			setError('Name and template are required.');
			return;
		}
		const trimmedName = name.trim();
		setIsSubmitting(true);
		setError(null);
		try {
			const overrides = buildOverrides(overrideForm);
			const req: CreateRunRequest = {
				name: trimmedName,
				template,
				...(Object.keys(overrides).length > 0 ? { overrides } : {}),
			};
			// createRun (and the other process actions) resolve void and swallow
			// their own errors into store state rather than rejecting, so success
			// is detected by whether a run with this name is newly present
			// afterward (not merely present -- a name collision means it was
			// already there beforehand and should still surface as an error).
			const before = new Set(useTrainingStore.getState().runs.map((r) => r.name));
			await createRun(req);
			const created = useTrainingStore
				.getState()
				.runs.some((r) => r.name === trimmedName && !before.has(r.name));
			if (created) {
				onCreated?.(trimmedName);
				onClose();
			} else {
				setError('Failed to create run. Check the run name (must be unique) and template.');
			}
		} finally {
			setIsSubmitting(false);
		}
	};

	const overrides = buildOverrides(overrideForm);
	const preview = buildYamlPreview(name, template, overrides);

	return (
		<Modal isOpen={isOpen} onClose={handleClose} title="New Training Run">
			<form onSubmit={handleSubmit} className="space-y-4">
				{error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}

				<div>
					<label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
						Run name
					</label>
					<input
						type="text"
						value={name}
						onChange={(e) => setName(e.target.value)}
						disabled={isSubmitting}
						placeholder="my-prtcfr-run"
						className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
					/>
				</div>

				<div>
					<label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
						Template
					</label>
					<select
						value={template}
						onChange={(e) => setTemplate(e.target.value)}
						disabled={isSubmitting || templates.length === 0}
						className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
					>
						{templates.length === 0 && <option value="">No templates found</option>}
						{templates.map((t) => (
							<option key={t} value={t}>
								{t}
							</option>
						))}
					</select>
				</div>

				<div className="grid grid-cols-3 gap-3">
					<div>
						<label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
							Iterations
						</label>
						<input
							type="number"
							min={1}
							value={overrideForm.iterations}
							onChange={(e) => setOverrideForm((f) => ({ ...f, iterations: e.target.value }))}
							disabled={isSubmitting}
							placeholder="1000"
							className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
						/>
					</div>
					<div>
						<label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
							Device
						</label>
						<select
							value={overrideForm.device}
							onChange={(e) => setOverrideForm((f) => ({ ...f, device: e.target.value }))}
							disabled={isSubmitting}
							className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
						>
							<option value="cpu">cpu</option>
							<option value="cuda">cuda</option>
						</select>
					</div>
					<div>
						<label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
							Seed
						</label>
						<input
							type="number"
							value={overrideForm.seed}
							onChange={(e) => setOverrideForm((f) => ({ ...f, seed: e.target.value }))}
							disabled={isSubmitting}
							placeholder="42"
							className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
						/>
					</div>
				</div>

				<div>
					<button
						type="button"
						onClick={() => setShowPreview((v) => !v)}
						className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
					>
						{showPreview ? 'Hide' : 'Show'} YAML preview
					</button>
					{showPreview && (
						<textarea
							readOnly
							value={preview}
							rows={Math.min(10, preview.split('\n').length + 1)}
							className="mt-2 block w-full font-mono text-xs px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-700 dark:text-gray-300"
						/>
					)}
				</div>

				<div className="flex justify-end space-x-3 pt-2">
					<Button variant="secondary" type="button" onClick={handleClose} disabled={isSubmitting}>
						Cancel
					</Button>
					<Button variant="primary" type="submit" isLoading={isSubmitting} disabled={isSubmitting}>
						{isSubmitting ? 'Creating...' : 'Create run'}
					</Button>
				</div>
			</form>
		</Modal>
	);
};

export default CreateRunModal;
