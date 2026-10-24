'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, ApiError, type ApplicationUpdateStatus } from '@/lib/api';
import './application-updates.css';

const active = new Set(['queued', 'building', 'backing_up', 'validating', 'restarting']);
const pendingKey = 'openjensen-pending-update';

export function ApplicationUpdates() {
  const client = useQueryClient();
  const dialog = useRef<HTMLDialogElement>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const status = useQuery({
    queryKey: ['application-updates'],
    queryFn: api.applicationUpdates,
    refetchInterval: (query) => (active.has(query.state.data?.operation?.phase ?? '') || pending ? 2000 : 30_000),
    retry: false,
  });
  const check = useQuery({
    queryKey: ['application-update-check'],
    queryFn: async () => {
      const value = await api.checkApplicationUpdates();
      client.setQueryData(['application-updates'], value);
      return value;
    },
    enabled: Boolean(status.data) && !active.has(status.data?.operation?.phase ?? '') && !pending,
    staleTime: 600_000,
    refetchInterval: 600_000,
    retry: false,
  });
  const value = status.data;
  const operation = value?.operation;
  const busy = confirming || active.has(operation?.phase ?? '') || Boolean(pending);
  const available = value?.state === 'available';
  const short = (commit?: string | null) => commit?.slice(0, 8) ?? 'Unknown';

  useEffect(() => {
    setPending(sessionStorage.getItem(pendingKey));
  }, []);

  useEffect(() => {
    if (open && !dialog.current?.open) dialog.current?.showModal();
    if (!open && dialog.current?.open) dialog.current.close();
  }, [open]);

  useEffect(() => {
    if (!pending || !operation || operation.id !== pending) return;
    if (active.has(operation.phase)) setError(null);
    if (operation.phase === 'complete' && value?.installed.commit === operation.target_commit) {
      sessionStorage.removeItem(pendingKey);
      sessionStorage.removeItem(pendingKey + '-target');
      window.location.reload();
    } else if (['failed', 'rolled_back', 'recovery_required'].includes(operation.phase)) {
      sessionStorage.removeItem(pendingKey);
      sessionStorage.removeItem(pendingKey + '-target');
      setPending(null);
    }
  }, [operation, pending, value?.installed.commit]);

  function updateCache(next: ApplicationUpdateStatus) {
    client.setQueryData(['application-updates'], next);
  }

  async function refresh() {
    setRefreshing(true);
    setError(null);
    try {
      updateCache(await api.checkApplicationUpdates(true));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Could not check updates.');
    } finally {
      setRefreshing(false);
    }
  }

  async function apply(retry = false) {
    const target = retry ? sessionStorage.getItem(pendingKey + '-target') : value?.upstream_commit;
    if (!target) return;
    const operationId =
      retry && pending
        ? pending
        : Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, '0')).join('');
    sessionStorage.setItem(pendingKey, operationId);
    sessionStorage.setItem(pendingKey + '-target', target);
    setPending(operationId);
    setConfirming(true);
    setError(null);
    try {
      const next = await api.applyApplicationUpdate(target, operationId);
      updateCache(next);
    } catch (failure) {
      if (failure instanceof ApiError && failure.status < 500) {
        sessionStorage.removeItem(pendingKey);
        sessionStorage.removeItem(pendingKey + '-target');
        setPending(null);
        setError(failure.message);
      } else {
        setError('Connection interrupted. Checking whether this update was accepted.');
      }
      void status.refetch();
    } finally {
      setConfirming(false);
    }
  }

  // Old installations and restricted gateways may not expose this API yet.
  if (!value && !open) return null;
  return (
    <>
      <button
        type="button"
        className={`application-version${available ? ' has-update' : ''}`}
        aria-label="Version and updates"
        onClick={() => setOpen(true)}
      >
        {busy
          ? pending && operation?.id !== pending && !confirming
            ? 'Checking update…'
            : 'Updating…'
          : available
            ? 'Update available'
            : `v${value?.installed.version ?? '…'}`}
      </button>
      <dialog
        ref={dialog}
        className="application-update-dialog"
        aria-labelledby="application-update-title"
        onCancel={() => setOpen(false)}
        onClose={() => setOpen(false)}
      >
        <header>
          <h2 id="application-update-title">OpenJensen updates</h2>
          <button type="button" className="text-button" aria-label="Close updates" onClick={() => setOpen(false)}>
            ×
          </button>
        </header>
        <dl className="application-version-facts">
          <div>
            <dt>Installed</dt>
            <dd>
              v{value?.installed.version} · <code>{short(value?.installed.commit)}</code>
            </dd>
          </div>
          <div>
            <dt>Server</dt>
            <dd>{value?.installed.host}</dd>
          </div>
          <div>
            <dt>Upstream main</dt>
            <dd>
              {value?.upstream_commit ? (
                <a
                  href={`https://github.com/OpenJensen/openjensen/commit/${value.upstream_commit}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  {short(value.upstream_commit)}
                </a>
              ) : (
                'Not checked'
              )}
            </dd>
          </div>
          <div>
            <dt>Published release</dt>
            <dd>
              {value?.latest_release ? (
                <a
                  href={`https://github.com/OpenJensen/openjensen/releases/tag/${value.latest_release}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  {value.latest_release}
                </a>
              ) : (
                'Unknown'
              )}
            </dd>
          </div>
        </dl>
        <p role="status">
          {busy && status.isError
            ? 'Server restarting or disconnected. Reconnecting…'
            : operation && (busy || ['failed', 'rolled_back', 'recovery_required'].includes(operation.phase))
              ? operation.message
              : check.isFetching || refreshing
                ? 'Checking upstream main…'
                : value?.message}
        </p>
        {!value?.automatic_update && (
          <p>
            {value?.setup_message}{' '}
            <a
              href="https://github.com/OpenJensen/openjensen/blob/main/docs/upgrading.md"
              target="_blank"
              rel="noreferrer"
            >
              Update setup
            </a>
          </p>
        )}
        {available && value?.automatic_update && !busy && (
          <p>Update this server for everyone using it? The workspace will be backed up and the server will restart.</p>
        )}
        {error && (
          <p className="error-notice" role="alert">
            {error}
          </p>
        )}
        <footer>
          {pending && operation?.id !== pending && !confirming && (
            <button type="button" className="secondary-button" onClick={() => void apply(true)}>
              Retry request
            </button>
          )}
          <button
            type="button"
            className="secondary-button"
            disabled={busy || check.isFetching || refreshing}
            onClick={() => void refresh()}
          >
            Check for updates
          </button>
          <button type="button" className="secondary-button" onClick={() => setOpen(false)}>
            {busy ? 'Close' : 'Cancel'}
          </button>
          {available && value?.automatic_update && (
            <button type="button" className="primary-button" disabled={busy} onClick={() => void apply()}>
              {confirming ? 'Starting…' : busy ? 'Updating…' : 'Update & restart'}
            </button>
          )}
        </footer>
      </dialog>
    </>
  );
}
