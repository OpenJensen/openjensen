'use client';

import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Icon } from '@/components/icon';
import { StatusBadge } from '@/components/ui/status-badge';
import { useWorkspace } from '@/components/workspace-context';
import { api, type InstallableProfile, type WorkspaceModule } from '@/lib/api';
import { publicPath } from '@/lib/base-path';
import { workspaceRoutes } from '@/lib/workspace-routes';
import './cloud-connections.css';
import './modules-panel.css';

const purpose: Record<WorkspaceModule['id'], string> = {
  'robot-lab': 'Arm setup, calibration and recording',
  teaching: 'Simulator teaching and demonstration capture',
  augmentation: 'Generate dataset variations',
  'decision-lab': 'Local advisory scoring',
  'cloud-runs': 'Cloud run snapshots',
};
const runtimeNotes: Record<InstallableProfile, string> = {
  datasets: 'Parquet, HDF5 and image readers',
  robot: 'Install on the computer with the USB devices. Pairing is separate.',
  decision: 'CPU runtime. Model files and license acceptance are separate.',
};

export function ModulesPanel() {
  const { modules, navigateStage, setSettingsTab } = useWorkspace();
  const client = useQueryClient();
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
      setMessage(value.enabled ? 'Module added to the sidebar.' : 'Module hidden. Dependencies and data are kept.');
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
      <ul className="module-cards" aria-label="Workspace modules">
        {modules.data?.map((item) => (
          <li key={item.id}>
            <div className="module-title">
              <Icon name={workspaceRoutes[item.id].icon} size={18} />
              <h3>{item.name}</h3>
              <StatusBadge tone={item.installed ? 'success' : 'neutral'}>
                {item.installed ? 'Added' : item.available ? 'Available' : 'Not installed'}
              </StatusBadge>
            </div>
            <p>{purpose[item.id]}</p>
            <div className="module-actions">
              {item.installed ? (
                <>
                  <button className="secondary-button" onClick={() => navigateStage(item.id)}>
                    Open
                  </button>
                  {!item.managed && (
                    <button
                      className="module-text-button"
                      disabled={visibility.isPending}
                      onClick={() => visibility.mutate({ id: item.id, enabled: false })}
                      aria-label={`Hide ${item.name}`}
                    >
                      Hide
                    </button>
                  )}
                </>
              ) : item.available ? (
                <button
                  className="primary-button"
                  disabled={visibility.isPending || item.managed}
                  onClick={() => visibility.mutate({ id: item.id, enabled: true })}
                  aria-label={`Add ${item.name}`}
                >
                  Add module
                </button>
              ) : item.id === 'augmentation' ? (
                <button className="secondary-button" onClick={() => setSettingsTab('compute')}>
                  Connect provider
                </button>
              ) : (
                <a
                  className="module-text-button"
                  href={publicPath(item.setup_url || '/guide/')}
                  target="_blank"
                  rel="noreferrer"
                >
                  Set up {item.name}
                </a>
              )}
              {item.managed && <span className="module-muted">Managed by server</span>}
            </div>
            {!item.configured && (
              <details>
                <summary>Setup required</summary>
                <p>{item.setup}</p>
                {item.id === 'augmentation' ? (
                  <button className="module-text-button" onClick={() => setSettingsTab('compute')}>
                    Connect provider
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
      <div className="module-section-heading module-packages-heading">
        <h2>Runtime packages</h2>
        {packages.data && <span className="module-muted">Server: {packages.data.host}</span>}
      </div>
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
      <ul className="module-runtime-list" aria-label="Runtime packages">
        {packages.data?.profiles.map((item) => (
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
              <StatusBadge tone={item.location === 'companion' && item.connected === false ? 'warning' : 'success'}>
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
        <li>
          <div>
            <h3>Training</h3>
            <p>Local SmolVLA worker · Linux NVIDIA GPU</p>
            {training.data && training.data.status !== 'idle' && training.data.status !== 'succeeded' && (
              <p role="status">{training.data.status === 'running' ? training.data.stage : training.data.message}</p>
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
        <li>
          <div>
            <h3>Simulation &amp; teaching</h3>
            <p>External Isaac runtime and scene configuration</p>
          </div>
          <a className="module-text-button" href={publicPath('/guide/#teaching')} target="_blank" rel="noreferrer">
            Setup guide
          </a>
        </li>
      </ul>
    </section>
  );
}
