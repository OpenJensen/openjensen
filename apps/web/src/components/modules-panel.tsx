'use client';

import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Icon } from '@/components/icon';
import { StatusBadge } from '@/components/ui/status-badge';
import { useWorkspace } from '@/components/workspace-context';
import { api, type InstallableProfile, type WorkspaceModule } from '@/lib/api';
import { publicPath } from '@/lib/base-path';
import { moduleStage, workspaceRoutes } from '@/lib/workspace-routes';
import './cloud-connections.css';
import './modules-panel.css';

const moduleGroups = ['Robot Lab', 'Data', 'Train', 'Test', 'Workspace', 'Cloud & APIs'] as const;
const purpose: Record<WorkspaceModule['id'], string> = {
  'robot-lab': 'Connect the arm, identify motors and calibrate',
  teaching: 'Goal-directed AI teleoperation and recording',
  datasets: 'Import, inspect and organize datasets',
  augmentation: 'Generate dataset variations',
  training: 'Fine-tune policies on your datasets',
  distillation: 'Distill policies into smaller models',
  quantization: 'Compress policies and export model formats',
  evaluation: 'Evaluate policy quality and compare results',
  simulation: 'Run policies in configured simulation environments',
  'decision-lab': 'Local advisory scoring',
  models: 'Browse, import and manage policy models',
  'cloud-runs': 'Cloud run snapshots',
  'google-cloud': 'Google Cloud compute and storage connections',
  openai: 'OpenAI API access for connected features',
  gemini: 'Gemini API access for connected features',
  huggingface: 'Hugging Face model and dataset access',
};
const runtimeModules: Record<InstallableProfile, WorkspaceModule['id']> = {
  datasets: 'datasets',
  robot: 'robot-lab',
  decision: 'decision-lab',
};
const runtimeNotes: Record<InstallableProfile, string> = {
  datasets: 'Parquet, HDF5 and image readers',
  robot: 'Install on the computer with the USB devices. Pairing is separate.',
  decision: 'CPU runtime. Model files and license acceptance are separate.',
};

export function ModulesPanel() {
  const { modules, navigateStage } = useWorkspace();
  const client = useQueryClient();
  const hasModule = (id: WorkspaceModule['id']) =>
    modules.data?.some((item) => item.id === id && item.installed) ?? false;
  const hasDependencies = ['datasets', 'robot-lab', 'decision-lab', 'training', 'simulation', 'teaching'].some((id) =>
    hasModule(id as WorkspaceModule['id']),
  );
  const [message, setMessage] = useState('');
  const [retry, setRetry] = useState<{ profile: InstallableProfile; id: string }>();
  const packages = useQuery({
    queryKey: ['module-installations'],
    queryFn: api.moduleInstallations,
    retry: false,
    refetchInterval: (query) => (query.state.data?.operation?.status === 'running' ? 1500 : false),
  });
  const training = useQuery({
    queryKey: ['local-setup'],
    queryFn: api.localSetupStatus,
    retry: false,
    refetchInterval: (query) => (query.state.data?.status === 'running' ? 2000 : false),
  });
  const refresh = () => {
    void client.invalidateQueries({ queryKey: ['module-installations'] });
    void client.invalidateQueries({ queryKey: ['modules'] });
    void client.invalidateQueries({ queryKey: ['local-setup'] });
  };
  const visibility = useMutation({
    mutationFn: ({ id, enabled }: { id: WorkspaceModule['id']; enabled: boolean }) => api.setModuleEnabled(id, enabled),
    onSuccess: (_, value) => {
      setMessage(value.enabled ? 'Module shown in the sidebar.' : 'Module hidden. Data and dependencies are kept.');
      refresh();
    },
    onError: (error) => setMessage(error.message),
  });
  const register = useMutation({
    mutationFn: api.installWorkspaceModule,
    onSuccess: () => {
      setMessage('Module installed.');
      refresh();
    },
    onError: (error) => setMessage(error.message),
  });
  const uninstall = useMutation({
    mutationFn: api.uninstallWorkspaceModule,
    onSuccess: () => {
      setMessage('Module uninstalled. Your data and runtime dependencies are kept.');
      refresh();
    },
    onError: (error) => setMessage(error.message),
  });
  const install = useMutation({
    mutationFn: ({ profile, id }: { profile: InstallableProfile; id: string }) => api.installModule(profile, id),
    onSuccess: () => {
      setRetry(undefined);
      setMessage('');
      refresh();
    },
    onError: (error) => {
      setMessage(error.message);
      refresh();
    },
  });
  const cancel = useMutation({
    mutationFn: api.cancelModuleInstallation,
    onSuccess: refresh,
    onError: (error) => setMessage(error.message),
  });
  const setupTraining = useMutation({
    mutationFn: api.startLocalSetup,
    onSuccess: (value) => {
      client.setQueryData(['local-setup'], value);
      setMessage('');
    },
    onError: (error) => setMessage(error.message),
  });
  const cancelTraining = useMutation({
    mutationFn: api.cancelLocalSetup,
    onSuccess: (value) => client.setQueryData(['local-setup'], value),
    onError: (error) => setMessage(error.message),
  });
  const operation = packages.data?.operation;
  useEffect(() => {
    if (operation && operation.status !== 'running') {
      void client.invalidateQueries({ queryKey: ['modules'] });
      void client.invalidateQueries({ queryKey: ['capabilities'] });
    }
  }, [operation, client]);
  useEffect(() => {
    if (training.data?.status === 'succeeded') {
      void client.invalidateQueries({ queryKey: ['modules'] });
      void client.invalidateQueries({ queryKey: ['capabilities'] });
      void client.invalidateQueries({ queryKey: ['compute-settings'] });
    }
  }, [training.data?.status, training.data?.id, client]);
  const busy =
    operation?.status === 'running' ||
    training.data?.status === 'running' ||
    install.isPending ||
    setupTraining.isPending;
  const start = (profile: InstallableProfile) => {
    // Retain the request id after an uncertain response; retry cannot start a duplicate.
    const value =
      retry?.profile === profile && !(operation?.id === retry.id && operation.status !== 'running')
        ? retry
        : { profile, id: crypto.randomUUID().replaceAll('-', '') };
    setRetry(value);
    install.mutate(value);
  };

  return (
    <section className="cloud-settings module-manager" aria-labelledby="modules-heading">
      <div className="module-section-heading">
        <h2 id="modules-heading">Modules</h2>
        <button className="secondary-button" onClick={refresh} disabled={packages.isFetching}>
          Refresh
        </button>
      </div>
      {message && (
        <p className="module-notice" role="status">
          {message}
        </p>
      )}
      {modules.isPending && <p role="status">Loading modules…</p>}
      {modules.error && <p role="alert">{modules.error.message}</p>}
      <p className="module-introduction">
        Install the features you want in this workspace. Datasets and Models are included by default. Hiding changes the
        sidebar; uninstalling removes a feature while keeping your data and runtime dependencies.
      </p>
      <div className="module-groups" aria-label="Workspace modules">
        {moduleGroups.map((group) => {
          const items = modules.data?.filter((item) => item.group === group) ?? [];
          if (!items.length) return null;
          return (
            <section
              className="module-group"
              key={group}
              aria-labelledby={`module-group-${group.replaceAll(/[^a-z0-9]/gi, '-').toLowerCase()}`}
            >
              <h3 id={`module-group-${group.replaceAll(/[^a-z0-9]/gi, '-').toLowerCase()}`}>{group}</h3>
              <ul className="module-cards" aria-label={`${group} modules`}>
                {items.map((item) => (
                  <li key={item.id}>
                    <div className="module-title">
                      <Icon name={workspaceRoutes[moduleStage(item.id)].icon} size={18} />
                      <h4>{item.name}</h4>
                      <StatusBadge tone={item.installed ? 'success' : 'neutral'}>
                        {item.installed
                          ? item.enabled === false
                            ? 'Installed · Hidden'
                            : 'Installed'
                          : 'Not installed'}
                      </StatusBadge>
                    </div>
                    <p>{purpose[item.id]}</p>
                    {item.default_installed && <span className="module-default">Included by default</span>}
                    <div className="module-actions">
                      {item.installed ? (
                        <>
                          <button
                            className="secondary-button"
                            disabled={item.enabled === false && moduleStage(item.id) !== 'cloud-apis'}
                            onClick={() => navigateStage(moduleStage(item.id))}
                            aria-label={`Open ${item.name}`}
                          >
                            Open
                          </button>
                          {!item.managed && (
                            <>
                              {moduleStage(item.id) !== 'cloud-apis' && (
                                <button
                                  className="module-text-button"
                                  disabled={visibility.isPending || uninstall.isPending}
                                  onClick={() => visibility.mutate({ id: item.id, enabled: item.enabled === false })}
                                  aria-label={`${item.enabled === false ? 'Show' : 'Hide'} ${item.name}`}
                                >
                                  {item.enabled === false ? 'Show' : 'Hide'}
                                </button>
                              )}
                              <button
                                className="module-text-button"
                                disabled={uninstall.isPending || visibility.isPending}
                                onClick={() => uninstall.mutate(item.id)}
                                aria-label={`Uninstall ${item.name}`}
                              >
                                Uninstall
                              </button>
                            </>
                          )}
                        </>
                      ) : (
                        <button
                          className="primary-button"
                          disabled={register.isPending || item.managed || !item.available}
                          onClick={() => register.mutate(item.id)}
                          aria-label={`Install module ${item.name}`}
                        >
                          Install module
                        </button>
                      )}
                      {item.managed && <span className="module-muted">Managed by server</span>}
                    </div>
                    {item.installed && !item.configured && (
                      <details>
                        <summary>Setup required</summary>
                        <p>{item.setup}</p>
                        {group === 'Cloud & APIs' || item.id === 'augmentation' ? (
                          <button className="module-text-button" onClick={() => navigateStage('cloud-apis')}>
                            Open Cloud &amp; APIs
                          </button>
                        ) : (
                          <a href={publicPath(item.setup_url || '/guide/')} target="_blank" rel="noreferrer">
                            Setup guide
                          </a>
                        )}
                      </details>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
      </div>
      {hasDependencies && (
        <>
          <div className="module-section-heading module-packages-heading">
            <h2>Runtime dependencies</h2>
            {packages.data && <span className="module-muted">Server: {packages.data.host}</span>}
          </div>
          <p className="module-introduction">
            Shared packages used by installed features. Installing a module does not download models or start a job.
          </p>
          {packages.isPending && <p role="status">Checking dependencies…</p>}
          {packages.error && <p role="alert">{packages.error.message}</p>}
          {packages.data?.error && <p role="alert">{packages.data.error}</p>}
          {operation && (
            <div
              className={`module-install-status module-install-${operation.status}`}
              role={operation.status === 'failed' ? 'alert' : 'status'}
            >
              <strong>
                {operation.status === 'running'
                  ? operation.stage
                  : operation.status === 'succeeded'
                    ? 'Installed'
                    : operation.status === 'failed'
                      ? 'Installation failed'
                      : 'Installation stopped'}
              </strong>
              <p>{operation.message}</p>
              {operation.status === 'running' && (
                <button className="secondary-button" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
                  Cancel installation
                </button>
              )}
            </div>
          )}
          <ul className="module-runtime-list" aria-label="Runtime dependencies">
            {packages.data?.profiles
              .filter((item) => hasModule(runtimeModules[item.id]))
              .map((item) => (
                <li key={item.id}>
                  <div>
                    <h3>{item.name}</h3>
                    <p>
                      {item.location === 'companion'
                        ? `Installed on ${item.host || 'the paired computer'}.`
                        : runtimeNotes[item.id]}
                    </p>
                    {item.location === 'companion' && <p>{item.connected ? 'Companion connected.' : item.message}</p>}
                    {!item.ready && (
                      <details>
                        <summary>Requirements</summary>
                        <p>
                          {item.message} Requires {item.minimum_free_gib} GiB free.
                        </p>
                      </details>
                    )}
                  </div>
                  {item.ready ? (
                    <StatusBadge
                      tone={item.location === 'companion' && item.connected === false ? 'warning' : 'success'}
                    >
                      {item.location === 'companion' && item.connected === false ? 'Installed · Offline' : 'Installed'}
                    </StatusBadge>
                  ) : (
                    <button
                      className="secondary-button"
                      disabled={busy || !item.installable}
                      onClick={() => start(item.id)}
                      aria-label={`Install ${item.name}`}
                    >
                      {operation?.profile === item.id && operation.status === 'running' ? 'Installing…' : 'Install'}
                    </button>
                  )}
                </li>
              ))}
            {hasModule('training') && (
              <li>
                <div>
                  <h3>Training</h3>
                  <p>Local SmolVLA worker · Linux NVIDIA GPU</p>
                  {training.data && training.data.status !== 'idle' && training.data.status !== 'succeeded' && (
                    <p role="status">
                      {training.data.status === 'running' ? training.data.stage : training.data.message}
                    </p>
                  )}
                  {training.error && <p role="alert">{training.error.message}</p>}
                </div>
                {training.data?.status === 'running' ? (
                  <button
                    className="secondary-button"
                    disabled={cancelTraining.isPending}
                    onClick={() => cancelTraining.mutate()}
                  >
                    Cancel training setup
                  </button>
                ) : training.data?.status === 'succeeded' ? (
                  <StatusBadge tone="success">Installed</StatusBadge>
                ) : (
                  <button
                    className="secondary-button"
                    disabled={busy || training.isPending || training.isError}
                    onClick={() => setupTraining.mutate()}
                    aria-label="Install Training"
                  >
                    Install
                  </button>
                )}
              </li>
            )}
            {(hasModule('simulation') || hasModule('teaching')) && (
              <li>
                <div>
                  <h3>Simulation &amp; teaching</h3>
                  <p>External Isaac runtime and scene configuration</p>
                </div>
                <a
                  className="module-text-button"
                  href={publicPath('/guide/#teaching')}
                  target="_blank"
                  rel="noreferrer"
                >
                  Setup guide
                </a>
              </li>
            )}
          </ul>
        </>
      )}
    </section>
  );
}
