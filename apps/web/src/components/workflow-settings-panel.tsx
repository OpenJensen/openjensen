'use client';

import { useEffect } from 'react';
import { LocalComputeSettingsPanel } from '@/components/local-compute-settings';
import { useWorkspace } from '@/components/workspace-context';
import { WorkflowDiagnostics } from './workflow-diagnostics';
import { WorkflowSettingsForm } from './workflow-settings-form';
import type { SettingsTab } from './workflow-shared';
import type { WorkflowState } from './workflow-state';
import './cloud-connections.css';

export function WorkflowSettingsPanel({
  workflow,
  tab,
  onTabChange: setTab,
  onOpenQuantize,
}: {
  workflow: WorkflowState;
  tab: SettingsTab;
  onTabChange: (tab: SettingsTab) => void;
  onOpenQuantize: () => void;
}) {
  const { language, preferences, preferencesBlocker, ready, setPreferences, update } = workflow;
  const { modules, navigateStage } = useWorkspace();
  const trainingInstalled = modules.data?.some((item) => item.id === 'training' && item.installed) ?? false;
  const redirect = tab === 'modules' ? 'modules' : tab === 'providers' || tab === 'cloud-apis' ? 'cloud-apis' : null;
  useEffect(() => {
    if (redirect) navigateStage(redirect);
  }, [redirect, navigateStage]);
  if (redirect) return <p role="status">Opening {redirect === 'modules' ? 'Modules' : 'Cloud & APIs'}…</p>;
  return (
    <section className={`panel ${tab === 'compute' ? 'compute-settings-panel' : 'workflow-panel'}`}>
      <div className="section-tabs settings-tabs">
        <button className={`section-tab ${tab === 'compute' ? 'active' : ''}`} onClick={() => setTab('compute')}>
          Local compute
        </button>
        <button className={`section-tab ${tab === 'settings' ? 'active' : ''}`} onClick={() => setTab('settings')}>
          Workflow settings
        </button>
        <button
          className={`section-tab ${tab === 'diagnostics' ? 'active' : ''}`}
          onClick={() => setTab('diagnostics')}
        >
          Diagnostics
        </button>
      </div>
      {tab === 'settings' && preferencesBlocker && (
        <p className="warning-box" role="status">
          {preferencesBlocker}
        </p>
      )}
      {tab === 'compute' ? (
        trainingInstalled ? (
          <LocalComputeSettingsPanel />
        ) : (
          <div className="cloud-settings">
            <h2>Local compute</h2>
            <p>Install Training from Modules to configure local runs.</p>
            <button className="secondary-button" onClick={() => navigateStage('modules')}>
              Open Modules
            </button>
          </div>
        )
      ) : tab === 'settings' ? (
        <WorkflowSettingsForm
          preferences={preferences}
          ready={ready}
          language={language}
          update={update}
          onSuite={(suite) =>
            setPreferences((old) => ({
              ...old,
              suite,
              mode: suite === 'libero_spatial' ? 'libero' : old.mode,
              steps: suite === 'libero_spatial' ? 280 : 500,
            }))
          }
        />
      ) : (
        <WorkflowDiagnostics workflow={workflow} onTabChange={setTab} onOpenQuantize={onOpenQuantize} />
      )}
    </section>
  );
}
