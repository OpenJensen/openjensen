'use client';

import { useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { agentApi, type AgentProvider, type AgentProviderRequest } from '@/lib/agent-api';
import { agentModelName } from '@/lib/agent-models';
import './workbench-form.css';
import './agent-control.css';

export function ProviderSettings() {
  const client = useQueryClient();
  const providers = useQuery({ queryKey: ['agent-providers'], queryFn: agentApi.providers, retry: false });
  const [id, setId] = useState('openai');
  const [provider, setProvider] = useState<'mock' | 'openai_responses'>('openai_responses');
  const [model, setModel] = useState('gpt-6-astra');
  const [customModel, setCustomModel] = useState(false);
  const models = useQuery({
    queryKey: ['openai-models'],
    queryFn: agentApi.openaiModels,
    enabled: false,
    retry: false,
  });
  const [reasoning, setReasoning] = useState('medium');
  const [busy, setBusy] = useState(false);
  const [checkConsent, setCheckConsent] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const working = useRef(false);

  async function run(action: () => Promise<unknown>, done: string) {
    if (working.current) return;
    working.current = true;
    setBusy(true);
    setMessage('');
    setError('');
    try {
      await action();
      await providers.refetch();
      void client.invalidateQueries({ queryKey: ['api-connections'] });
      void client.invalidateQueries({ queryKey: ['modules'] });
      setMessage(done);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Provider settings could not be verified.');
    } finally {
      working.current = false;
      setBusy(false);
    }
  }
  function edit(profile: AgentProvider) {
    if (profile.provider === 'local_mcp') return;
    setId(profile.id);
    setProvider(profile.provider);
    setModel(profile.model);
    setCustomModel(false);
    setReasoning(profile.reasoning_effort);
    setCheckConsent(false);
    setMessage('');
    setError('');
  }
  const validId = /^[A-Za-z0-9_-]{1,64}$/.test(id);
  const modelOptions = Array.from(new Set([model, ...(models.data ?? [])])).filter(Boolean);
  return (
    <section className="agent-provider-settings" aria-label="Agent provider settings">
      <div>
        <h2>AI providers</h2>
        <p>
          Agent Control uses a saved provider profile. A saved key and a successful capability check are separate
          states.
        </p>
      </div>
      <form
        className="workbench-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (!validId || !model.trim() || busy || !providers.isSuccess) return;
          const body = {
            provider,
            model: model.trim(),
            reasoning_effort: reasoning,
          } as AgentProviderRequest;
          setCheckConsent(false);
          void run(
            () => agentApi.saveProvider(id, body),
            'Profile saved. Run a capability check before using a cloud provider.',
          );
        }}
      >
        <div className="workbench-field-grid">
          <label className="workbench-field">
            Profile ID
            <input
              value={id}
              maxLength={64}
              pattern="[A-Za-z0-9_-]+"
              required
              disabled={busy}
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
              onChange={(event) => setId(event.target.value)}
            />
          </label>
          <label className="workbench-field">
            Provider transport
            <select
              value={provider}
              disabled={busy}
              onChange={(event) => {
                const value = event.target.value as 'mock' | 'openai_responses';
                setProvider(value);
                setModel(value === 'mock' ? 'deterministic-mock' : 'gpt-6-astra');
                setCustomModel(false);
              }}
            >
              <option value="openai_responses">OpenAI Responses</option>
              <option value="mock">Deterministic mock</option>
            </select>
          </label>
          {provider === 'openai_responses' && !customModel ? (
            <label className="workbench-field">
              OpenAI model
              <select
                value={model}
                disabled={busy}
                onChange={(event) => {
                  if (event.target.value === '__custom__') setCustomModel(true);
                  else setModel(event.target.value);
                }}
              >
                {modelOptions.map((identity) => (
                  <option key={identity} value={identity}>
                    {agentModelName(identity)} · {identity}
                  </option>
                ))}
                <option value="__custom__">Enter a model ID…</option>
              </select>
            </label>
          ) : (
            <label className="workbench-field">
              Model ID
              <input
                value={model}
                maxLength={128}
                required
                disabled={busy || provider === 'mock'}
                spellCheck={false}
                onChange={(event) => setModel(event.target.value)}
              />
            </label>
          )}
          <label className="workbench-field">
            Reasoning effort
            <select value={reasoning} disabled={busy} onChange={(event) => setReasoning(event.target.value)}>
              {['low', 'medium', 'high', 'xhigh', 'max'].map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
        </div>
        {provider === 'openai_responses' && (
          <>
            <p>Uses the OpenAI API key configured above. Model availability does not verify vision or tool support.</p>
            <div className="workbench-actions">
              <button
                type="button"
                className="secondary-button"
                disabled={busy || models.isFetching}
                onClick={() => void models.refetch()}
              >
                {models.isFetching ? 'Loading OpenAI models…' : 'Load available OpenAI models'}
              </button>
              {customModel && (
                <button type="button" className="text-link" disabled={busy} onClick={() => setCustomModel(false)}>
                  Choose a model by name
                </button>
              )}
            </div>
            <p>Loading model names makes no inference request and does not send camera images.</p>
            {models.isError && <p role="alert">{models.error.message}</p>}
            {models.isSuccess && models.data.length === 0 && (
              <p role="status">No models were returned. Enter a model ID or refresh the list.</p>
            )}
          </>
        )}
        <div className="workbench-actions">
          <button
            type="submit"
            className="primary-button"
            disabled={!validId || !model.trim() || busy || !providers.isSuccess}
          >
            Save provider profile
          </button>
          <button
            type="button"
            className="secondary-button"
            disabled={busy || providers.isFetching}
            onClick={() => void providers.refetch()}
          >
            Refresh providers
          </button>
        </div>
      </form>
      <label className="workbench-check">
        <input
          type="checkbox"
          checked={checkConsent}
          disabled={busy}
          onChange={(event) => setCheckConsent(event.target.checked)}
        />
        Allow a cloud capability check to send a small test image and tool request to the selected provider. Provider
        charges may apply.
      </label>
      {providers.isPending && <p role="status">Checking saved profiles…</p>}
      {providers.error && (
        <p role="alert">Provider profiles are unavailable. Refresh before saving or checking a profile.</p>
      )}
      {providers.data?.map((profile) => (
        <section key={profile.id} className="agent-provider-card" aria-label={`Provider ${profile.id}`}>
          <h3>
            {agentModelName(profile.model)} · {profile.id}
          </h3>
          <p>
            {profile.saved ? 'Saved' : 'Built-in'} ·{' '}
            {profile.provider === 'local_mcp'
              ? 'Connect in AI teleoperation'
              : profile.checked
                ? 'Capability check passed'
                : 'Capabilities unverified'}{' '}
            · {profile.provider}
          </p>
          <p>{profile.message}</p>
          <p>Capabilities: {profile.capabilities.length ? profile.capabilities.join(', ') : 'Not verified'}</p>
          {profile.provider !== 'local_mcp' && (
            <div className="workbench-actions">
              <button type="button" className="secondary-button" disabled={busy} onClick={() => edit(profile)}>
                Edit {profile.id}
              </button>
              <button
                type="button"
                className="secondary-button"
                disabled={busy || (profile.provider !== 'mock' && !checkConsent)}
                onClick={() => {
                  setCheckConsent(false);
                  void run(() => agentApi.checkProvider(profile.id), `Capability check completed for ${profile.id}.`);
                }}
              >
                Check {profile.id}
              </button>
              {profile.saved && (
                <button
                  type="button"
                  className="text-link"
                  disabled={busy}
                  onClick={() => void run(() => agentApi.deleteProvider(profile.id), `Profile ${profile.id} removed.`)}
                >
                  Remove {profile.id}
                </button>
              )}
            </div>
          )}
        </section>
      ))}
      {busy && <p role="status">Provider request in progress…</p>}
      {message && <p role="status">{message}</p>}
      {error && <p role="alert">{error}</p>}
    </section>
  );
}
