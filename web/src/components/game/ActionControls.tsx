// src/components/game/ActionControls.tsx
import React from 'react';
import { motion } from 'framer-motion';
import { fadeTransition } from './CardAnimations';
import { callCambiaAction, skipSpecialAction } from '@/types/game';
import type { ClientGameAction } from '@/types/game';

interface ActionControlsProps {
	sendMessage: (action: ClientGameAction) => void;
	hint: string;
	canCallCambia: boolean;
	canSkipSpecial: boolean;
}

const ActionControls: React.FC<ActionControlsProps> = ({
	sendMessage,
	hint,
	canCallCambia,
	canSkipSpecial,
}) => {
	return (
		<div className="flex flex-col items-center gap-2 w-full">
			<div className="w-full overflow-hidden h-7">
				<motion.p
					key={hint}
					initial={{ opacity: 0, y: 8 }}
					animate={{ opacity: 1, y: 0 }}
					exit={{ opacity: 0, y: -8 }}
					transition={fadeTransition}
					className="text-center text-white/70 text-sm"
				>
					{hint}
				</motion.p>
			</div>

			{(canCallCambia || canSkipSpecial) && (
				<div className="flex justify-center gap-3">
					{canCallCambia && (
						<button
							onClick={() => sendMessage(callCambiaAction())}
							className="px-4 py-1.5 rounded-md bg-white/10 hover:bg-white/20 text-white text-sm font-medium transition-colors"
						>
							Call Cambia
						</button>
					)}
					{canSkipSpecial && (
						<button
							onClick={() => sendMessage(skipSpecialAction())}
							className="px-4 py-1.5 rounded-md bg-white/10 hover:bg-white/20 text-white text-sm font-medium transition-colors"
						>
							Skip
						</button>
					)}
				</div>
			)}
		</div>
	);
};

export default React.memo(ActionControls);
