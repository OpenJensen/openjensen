'use client';

import { engineRuntime } from '@/lib/workflow-entry';
import { WorkbenchDisclosure } from '@/components/workbench-disclosure';
import EngineWorkflowSection from './engine-workflow';
import { useWorkspace } from '@/components/workspace-context';

export default function EvaluationSection() {
  const { options, engineConfigured, simulation, jobs, replayConfigured, replayHistory, simulationConfigured, simulationHistory, navigateStage, chooseRun } = useWorkspace();
  return <>
    <section className="workflow-context" aria-label="Evaluation purpose"><WorkbenchDisclosure title="Evaluation details"><p>Engine checks measure loading, finite actions and runtime performance. LIBERO measures closed-loop task success with a configured benchmark runtime. Observation replay and experimental Isaac rollouts do not establish task success; scored ACT / Isaac evaluation is not configured.</p></WorkbenchDisclosure>
      {options.isPending ? <p role="status">Checking evaluation targets…</p> : options.isError ? <p role="alert">Evaluation targets could not be loaded. Availability is unknown.</p> : <p role="status">{engineConfigured ? 'Engine evaluation is configured.' : 'No engine evaluation target is configured.'} {options.data?.runtimes.some(item => engineRuntime(item, 'Evaluate') && item.simulation) ? 'A LIBERO target is configured; its policy and protocol still require validation.' : 'No LIBERO evaluation target is configured.'}</p>}
      {simulation.isError && <p role="alert">Isaac profile availability could not be loaded.</p>}
      {jobs.isError && <p role="alert">Saved workflow history could not be refreshed; previously received records may be stale.</p>}
      <div className="native-result-actions">
        {(replayConfigured || replayHistory) && <button className="text-link" onClick={() => { navigateStage(5); chooseRun('replay', 'handoff'); }}>Open observation replay</button>}
        {(simulationConfigured || simulationHistory) && <button className="text-link" onClick={() => { navigateStage(5); chooseRun('native', 'handoff'); }}>Open native Isaac Run</button>}
      </div>
    </section>
    <EngineWorkflowSection />
  </>;
}
