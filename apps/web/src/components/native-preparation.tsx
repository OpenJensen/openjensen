'use client';

import type { ReactNode } from 'react';
import { WorkbenchDisclosure } from './workbench-disclosure';
import './simulation-workspace.css';

/** First-run controls stay visible; a saved result takes priority over another setup. */
export function NativePreparation({ title, hasResult, children }: { title: string; hasResult: boolean; children: ReactNode }) {
  return hasResult
    ? <WorkbenchDisclosure title={title}>{children}</WorkbenchDisclosure>
    : <div className="native-preparation-content">{children}</div>;
}
