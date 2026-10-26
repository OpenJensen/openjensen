'use client';

import { CloudConnectionsPanel } from '@/components/cloud-connections';
import { ModulesPanel } from '@/components/modules-panel';
import { ProviderSettings } from './provider-settings';
import { WorkflowDiagnostics } from './workflow-diagnostics';
import { WorkflowSettingsForm } from './workflow-settings-form';
import type { SettingsTab } from './workflow-shared';
import type { WorkflowState } from './workflow-state';

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
  return (
    <section
      className={`panel ${tab === 'compute' || tab === 'modules' ? 'compute-settings-panel' : 'workflow-panel'}`}
    >
      <div className="section-tabs settings-tabs">
        <button className={`section-tab ${tab === 'compute' ? 'active' : ''}`} onClick={() => setTab('compute')}>
          Compute
        </button>
        <button className={`section-tab ${tab === 'modules' ? 'active' : ''}`} onClick={() => setTab('modules')}>
          Modules
        </button>
        <button className={`section-tab ${tab === 'providers' ? 'active' : ''}`} onClick={() => setTab('providers')}>
          AI providers
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
        <CloudConnectionsPanel />
      ) : tab === 'modules' ? (
        <ModulesPanel />
      ) : tab === 'providers' ? (
        <ProviderSettings />
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
