'use client';

import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type HuggingFaceConnectionStatus } from '@/lib/api';
import { Icon } from '@/components/icon';
import './cloud-apis.css';

export function HuggingFaceSettingsPanel() {
  const client = useQueryClient();
  const connection = useQuery({
    queryKey: ['huggingface-connection'],
    queryFn: api.huggingFaceStatus,
  });
  const [token, setToken] = useState('');
  const [pending, setPending] = useState<'save' | 'remove' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<string | null>(null);

  function saved(status: HuggingFaceConnectionStatus, message: string) {
    client.setQueryData(['huggingface-connection'], status);
    setToken('');
    setFeedback(message);
  }

  // Keep the entered token out of React Query's mutation cache and browser storage.
  async function save() {
    setPending('save');
    setError(null);
    setFeedback(null);
    try {
      saved(await api.saveHuggingFaceToken(token.trim()), 'Hugging Face token saved.');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Could not save the token.');
    } finally {
      setPending(null);
    }
  }

  async function remove() {
    setPending('remove');
    setError(null);
    setFeedback(null);
    try {
      saved(await api.removeHuggingFaceToken(), 'Hugging Face token removed.');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Could not remove the token.');
    } finally {
      setPending(null);
    }
  }

  const status = connection.data;
  return (
    <section className="cloud-api-card" aria-labelledby="hf-settings-title">
      <header className="cloud-api-heading">
        <span className="cloud-api-mark" aria-hidden="true">
          <Icon name="folder" size={20} />
        </span>
        <h2 id="hf-settings-title">Hugging Face</h2>
        <span className="cloud-status">
          {connection.isPending ? 'Loading…' : status?.configured ? 'Configured' : 'Not configured'}
        </span>
      </header>
      <p className="cloud-api-description">For private or gated models.</p>
      {status?.configured && (
        <p className="cloud-api-account">
          <strong>{status.username}</strong>
          <span>{status.token_hint}</span>
        </p>
      )}
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void save();
        }}
      >
        <label>
          Hugging Face access token
          <input
            type="password"
            value={token}
            maxLength={512}
            autoComplete="off"
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            placeholder={status?.configured ? 'Enter a replacement token' : 'hf_…'}
            disabled={pending !== null}
            onChange={(event) => {
              setToken(event.target.value);
              setError(null);
              setFeedback(null);
            }}
          />
        </label>
        <div className="cloud-provider-actions">
          <button type="submit" className="secondary-button" disabled={!token.trim() || pending !== null}>
            {pending === 'save' ? 'Verifying…' : status?.configured ? 'Replace token' : 'Save token'}
          </button>
          {status?.configured && (
            <button type="button" className="text-button" disabled={pending !== null} onClick={() => void remove()}>
              {pending === 'remove' ? 'Removing…' : 'Remove token'}
            </button>
          )}
        </div>
        <p className="cloud-auth-note">
          Stored privately on this server.{' '}
          <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">
            Create a token <Icon name="external" size={11} />
          </a>
        </p>
        {status?.message && !status.configured && (
          <p className="cloud-connection-error" role="alert">
            {status.message}
          </p>
        )}
        {connection.error && (
          <p className="cloud-connection-error" role="alert">
            {connection.error.message}
          </p>
        )}
        {error && (
          <p className="cloud-connection-error" role="alert">
            {error}
          </p>
        )}
        {feedback && (
          <p className="cloud-feedback" role="status">
            {feedback}
          </p>
        )}
      </form>
    </section>
  );
}
