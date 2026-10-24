'use client';

import { engineRuntime } from '@/lib/workflow-entry';
import { WorkbenchDisclosure } from '@/components/workbench-disclosure';
import EngineWorkflowSection from './engine-workflow';
import { useWorkspace } from '@/components/workspace-context';

export default function EvaluationSection() {
  const {
    options,
    engineConfigured,
    simulation,
    jobs,
    replayConfigured,
    replayHistory,
    simulationConfigured,
    simulationHistory,
    navigateStage,
    chooseRun,
  } = useWorkspace();
  return (
    <>
      <section className="workflow-context" aria-label="Evaluation purpose">
        <WorkbenchDisclosure title="Evaluation details">
          <p>
            Engine checks measure loading, action validity and speed. LIBERO measures task success. Replay and Isaac
            rollouts do not measure task success.
          </p>
        </WorkbenchDisclosure>
        {options.isPending ? (
          <p role="status">Checking evaluation targets…</p>
        ) : options.isError ? (
          <p role="alert">Evaluation targets could not be loaded.</p>
        ) : (
          <p role="status">
            {engineConfigured ? 'Engine evaluation is configured.' : 'No engine evaluation target is configured.'}{' '}
            {options.data?.runtimes.some((item) => engineRuntime(item, 'Evaluate') && item.simulation)
              ? 'A LIBERO target is configured.'
              : 'No LIBERO evaluation target is configured.'}
          </p>
        )}
        {simulation.isError && <p role="alert">Isaac profile availability could not be loaded.</p>}
        {jobs.isError && <p role="alert">Workflow history could not be refreshed; it may be out of date.</p>}
        <div className="native-result-actions">
          {(replayConfigured || replayHistory) && (
            <button
              className="text-link"
              onClick={() => {
                navigateStage('simulation');
                chooseRun('replay', 'handoff');
              }}
            >
              Open observation replay
            </button>
          )}
          {(simulationConfigured || simulationHistory) && (
            <button
              className="text-link"
              onClick={() => {
                navigateStage('simulation');
                chooseRun('native', 'handoff');
              }}
            >
              Open native Isaac Run
            </button>
          )}
        </div>
      </section>
      <EngineWorkflowSection />
    </>
  );
}
