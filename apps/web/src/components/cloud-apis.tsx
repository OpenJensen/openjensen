'use client';

import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type ApiConnection, type ApiConnections, type ApiProviderId } from '@/lib/api';
import { CloudConnectionsPanel } from '@/components/cloud-connections';
import { HuggingFaceSettingsPanel } from '@/components/hf-settings';
import { useWorkspace } from '@/components/workspace-context';
import { Icon } from '@/components/icon';
import { ProviderSettings } from '@/components/provider-settings';
import './cloud-connections.css';
import './cloud-apis.css';

function ApiKeyPanel({ provider }: { provider: ApiProviderId }) {
  const client = useQueryClient();
  const connections = useQuery({
    queryKey: ['api-connections'],
    queryFn: api.apiConnections,
    retry: false,
  });
  const [profilesOpen, setProfilesOpen] = useState(false);
  const [key, setKey] = useState('');
  const [pending, setPending] = useState<'save' | 'remove' | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const status = connections.data?.providers.find((item) => item.id === provider);
  const name = provider === 'openai' ? 'OpenAI' : 'Gemini Omni';
  function saved(value: ApiConnection, message: string) {
    client.setQueryData<ApiConnections>(['api-connections'], (previous) => ({
      providers: [...(previous?.providers ?? []).filter((item) => item.id !== provider), value],
    }));
    setKey('');
    setNotice(message);
    void client.invalidateQueries({ queryKey: ['augmentation-options'] });
    void client.invalidateQueries({ queryKey: ['agent-providers'] });
    void client.invalidateQueries({ queryKey: ['modules'] });
  }
  // Credentials stay in component state only, outside the query/mutation cache.
  async function save() {
    setPending('save');
    setError('');
    setNotice('');
    try {
      saved(await api.saveApiConnection(provider, key.trim()), `${name} API key saved.`);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Could not save the API key.');
    } finally {
      setPending(null);
    }
  }
  async function remove() {
    setPending('remove');
    setError('');
    setNotice('');
    try {
      saved(await api.removeApiConnection(provider), `${name} saved API key removed.`);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Could not remove the API key.');
    } finally {
      setPending(null);
    }
  }
  return (
    <section className="cloud-api-card" aria-labelledby={`api-${provider}-title`}>
      <header className="cloud-api-heading">
        <span className="cloud-api-mark" aria-hidden="true">
          <Icon name={provider === 'openai' ? 'robot' : 'spark'} size={20} />
        </span>
        <h2 id={`api-${provider}-title`}>{name}</h2>
        <span className="cloud-status">
          {connections.isPending ? 'Loading…' : status?.configured ? 'Configured' : 'Not configured'}
        </span>
      </header>
      <p className="cloud-api-description">
        {provider === 'openai' ? 'API access for AI teleoperation.' : 'API access for video augmentation.'}
      </p>
      {status?.configured && (
        <p className="cloud-auth-note">
          {status.source === 'environment'
            ? 'Configured in the server environment.'
            : status.source === 'profile'
              ? 'Configured through a saved provider profile.'
              : 'API key saved privately on this server.'}
        </p>
      )}
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void save();
        }}
      >
        <label>
          {name} API key
          <input
            type="password"
            value={key}
            autoComplete="off"
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            maxLength={4096}
            placeholder={status?.configured ? 'Enter a replacement key' : 'Enter an API key'}
            disabled={pending !== null || connections.isPending || connections.isError}
            onChange={(event) => {
              setKey(event.target.value);
              setError('');
              setNotice('');
            }}
          />
        </label>
        <div className="cloud-provider-actions">
          <button
            type="submit"
            className="secondary-button"
            disabled={!key.trim() || pending !== null || connections.isPending || connections.isError}
          >
            {pending === 'save' ? 'Saving…' : status?.configured ? 'Replace API key' : 'Save API key'}
          </button>
          {status?.source === 'saved' && (
            <button type="button" className="text-button" disabled={pending !== null} onClick={() => void remove()}>
              {pending === 'remove' ? 'Removing…' : 'Remove saved key'}
            </button>
          )}
        </div>
        <p className="cloud-auth-note">Saving a key does not contact the provider or start a paid request.</p>
      </form>
      {connections.error && (
        <p className="cloud-connection-error" role="alert">
          {connections.error.message}
        </p>
      )}
      {status?.message && <p className="cloud-auth-note">{status.message}</p>}
      {error && (
        <p className="cloud-connection-error" role="alert">
          {error}
        </p>
      )}
      {notice && (
        <p className="cloud-feedback" role="status">
          {notice}
        </p>
      )}
      {provider === 'openai' && (
        <details className="cloud-api-disclosure" onToggle={(event) => setProfilesOpen(event.currentTarget.open)}>
          <summary>Advanced OpenAI profiles</summary>
          {profilesOpen && <ProviderSettings />}
        </details>
      )}
    </section>
  );
}

export function CloudApisPanel() {
  const { modules, navigateStage } = useWorkspace();
  const installed = (id: string) => modules.data?.some((item) => item.id === id && item.installed) ?? false;
  const any = ['google-cloud', 'openai', 'gemini', 'huggingface'].some(installed);
  if (modules.isPending) return <p role="status">Loading installed integrations…</p>;
  if (modules.isError)
    return (
      <div className="cloud-settings">
        <p role="alert">{modules.error.message}</p>
        <button className="secondary-button" onClick={() => void modules.refetch()}>
          Retry integrations
        </button>
      </div>
    );
  return (
    <div className="cloud-apis">
      {!any ? (
        <div className="cloud-settings cloud-integrations-empty">
          <h2>Connect your services</h2>
          <p>Add integrations from Modules to configure cloud resources and API keys.</p>
          <button className="primary-button" onClick={() => navigateStage('modules')}>
            Open Modules
          </button>
        </div>
      ) : (
        <>
          {installed('google-cloud') && <CloudConnectionsPanel />}
          {installed('openai') && <ApiKeyPanel provider="openai" />}
          {installed('gemini') && <ApiKeyPanel provider="gemini" />}
          {installed('huggingface') && <HuggingFaceSettingsPanel />}
          <button className="text-button" onClick={() => navigateStage('modules')}>
            Manage integrations
          </button>
        </>
      )}
    </div>
  );
}
