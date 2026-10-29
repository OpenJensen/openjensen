'use client';

import { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiOrigin } from '@/lib/api';
import { workJobs, WorkJobHttpError, isWorkActive, type WorkJob, type WorkSubmission } from '@/lib/work-jobs';
import { record, sameJson } from '@/lib/policy-job-mutation';
import { publicPath } from '@/lib/base-path';
import { publicMessage } from '@/lib/public-display';
import { WorkflowIntegration } from './workflow-integration';
import { WorkbenchDisclosure } from './workbench-disclosure';
import './workbench-form.css';
import './decision-panel.css';

type Status = { configured: boolean; available: boolean; busy: boolean; message: string };
type Result = {
  selected_id: string;
  request_sha256: string;
  revision: string;
  scores: { id: string; logit: number; relative_weight: number; tokens: number }[];
  timing_ms: { load: number; score: number };
};
const endpoint = `${apiOrigin}/api/v1/decision`;

async function request<T>(path: string, signal: AbortSignal, body?: object): Promise<T> {
  const response = await fetch(endpoint + path, {
    signal,
    method: body ? 'POST' : 'GET',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    cache: 'no-store',
  });
  if (!response.ok)
    throw new Error(
      response.status === 409
        ? 'The local scorer is busy. Wait and try again manually.'
        : 'No score was accepted. Check the operator configuration and shorten text to fit the 512-token limit.',
    );
  return response.json();
}

export function DecisionPanel({ projectId, preferredJobId }: { projectId: string; preferredJobId?: string }) {
  const client = useQueryClient();
  const [jobId, setJobId] = useState(preferredJobId ?? '');
  const [recovery, setRecovery] = useState<{ key: string; body: WorkSubmission } | null>(null);
  const [journalError, setJournalError] = useState('');
  const [hydrated, setHydrated] = useState(false);
  const endpointScope = new URL(apiOrigin || '/', 'http://same-origin.invalid');
  const storageKey = `openjensen:decision-submission:v1:${encodeURIComponent(endpointScope.origin + endpointScope.pathname)}:${encodeURIComponent(projectId)}`;
  const receiptKey = `${storageKey}:receipt`;
  const mounted = useRef(true);
  const generation = useRef(0);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      generation.current += 1;
    };
  }, []);
  useEffect(() => {
    if (preferredJobId) {
      generation.current += 1;
      setJobId(preferredJobId);
    }
  }, [preferredJobId]);
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(storageKey);
      if (raw) {
        const saved: unknown = JSON.parse(raw);
        if (
          !record(saved) ||
          typeof saved.key !== 'string' ||
          !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(saved.key) ||
          !record(saved.body) ||
          saved.body.operation !== 'decision.score' ||
          !record(saved.body.inputs)
        )
          throw new Error('Saved submission is unreadable. Review recorded jobs before another submission.');
        const inputs = saved.body.inputs;
        if (
          inputs.schema_version !== 1 ||
          typeof inputs.state !== 'string' ||
          typeof inputs.instructions !== 'string' ||
          !Array.isArray(inputs.criteria) ||
          inputs.criteria.length < 2 ||
          inputs.criteria.length > 8 ||
          !inputs.criteria.every((item) => record(item) && typeof item.id === 'string' && typeof item.text === 'string')
        )
          throw new Error('Saved submission is unreadable. Review recorded jobs before another submission.');
        setRecovery(saved as { key: string; body: WorkSubmission });
      } else setRecovery(null);
      const prior = sessionStorage.getItem(receiptKey);
      if (!preferredJobId && prior?.startsWith('core:') && prior.length < 128) setJobId(prior);
    } catch (cause) {
      setJournalError(cause instanceof Error ? cause.message : 'Submission recovery storage is unavailable.');
    } finally {
      setHydrated(true);
    }
  }, [storageKey, receiptKey, preferredJobId]);
  const job = useQuery({
    queryKey: ['decision-job', projectId, jobId],
    enabled: !!projectId && !!jobId,
    queryFn: ({ signal }) => workJobs.get(projectId, jobId, signal),
    retry: false,
    refetchInterval: (query) => (query.state.data && !isWorkActive(query.state.data) ? false : 1000),
  });
  const status = useQuery({
    queryKey: ['decision-status'],
    queryFn: ({ signal }) => request<Status>('/status', signal),
    retry: false,
    staleTime: 0,
  });
  const [state, setState] = useState('');
  const [instructions, setInstructions] = useState(
    'Compare these criteria with the state. Return advisory scores only.',
  );
  const [criteria, setCriteria] = useState(['', '']);
  const [result, setResult] = useState<{ receipt: Result; criteria: string[] } | null>(null);
  const [error, setError] = useState('');
  const [submitting, setPending] = useState(false);
  const pending =
    submitting || !!(job.data?.project_id === projectId && job.data.id === jobId && isWorkActive(job.data));
  const live = (version: number) => mounted.current && generation.current === version;
  const criterionFields = useRef<(HTMLTextAreaElement | null)[]>([]);
  const focusCriterion = useRef<number | null>(null);
  useEffect(() => {
    if (focusCriterion.current !== null) {
      criterionFields.current[focusCriterion.current]?.focus();
      focusCriterion.current = null;
    }
  }, [criteria.length]);
  useEffect(() => {
    if (
      !job.data ||
      job.data.project_id !== projectId ||
      job.data.id !== jobId ||
      job.data.operation !== 'decision.score'
    )
      return;
    if (job.data.status === 'succeeded' && job.data.result) {
      const value: unknown = job.data.result;
      const inputs = job.data.inputs.criteria;
      if (
        !record(value) ||
        typeof value.selected_id !== 'string' ||
        typeof value.request_sha256 !== 'string' ||
        typeof value.revision !== 'string' ||
        !Array.isArray(inputs) ||
        !inputs.every((item) => record(item) && typeof item.id === 'string' && typeof item.text === 'string') ||
        !Array.isArray(value.scores) ||
        value.scores.length !== inputs.length ||
        !value.scores.every(
          (item, index) =>
            record(item) &&
            item.id === (inputs[index] as { id: string }).id &&
            typeof item.logit === 'number' &&
            Number.isFinite(item.logit) &&
            typeof item.relative_weight === 'number' &&
            Number.isFinite(item.relative_weight) &&
            item.relative_weight >= 0 &&
            item.relative_weight <= 1 &&
            Number.isSafeInteger(item.tokens),
        ) ||
        !record(value.timing_ms) ||
        typeof value.timing_ms.load !== 'number' ||
        typeof value.timing_ms.score !== 'number'
      ) {
        setError('The saved scoring result could not be verified. Inspect the job diagnostics.');
        return;
      }
      setResult({
        receipt: value as unknown as Result,
        criteria: inputs.map((item) => (item as { text: string }).text),
      });
    } else if (['failed', 'cancelled', 'interrupted'].includes(job.data.status))
      setError(job.data.message ?? `Job ${job.data.status}.`);
  }, [job.data, projectId, jobId]);
  const valid =
    state.trim().length > 0 &&
    instructions.trim().length > 0 &&
    criteria.every((text) => text.trim()) &&
    new Set(criteria.map((text) => text.trim())).size === criteria.length;
  const ready = status.isSuccess && status.data.configured && status.data.available;
  function edited() {
    setJobId('');
    setResult(null);
    setError('');
  }
  function addCriterion() {
    if (pending || criteria.length >= 8) return;
    focusCriterion.current = criteria.length;
    edited();
    setCriteria([...criteria, '']);
  }
  function removeCriterion(index: number) {
    if (pending || criteria.length <= 2) return;
    focusCriterion.current = Math.min(index, criteria.length - 2);
    edited();
    setCriteria(criteria.filter((_, position) => position !== index));
  }
  async function score() {
    if (!projectId || !hydrated || journalError || pending || (!recovery && (!ready || !valid))) return;
    const version = generation.current;
    setPending(true);
    setError('');
    setResult(null);
    const saved = recovery ?? {
      key: crypto.randomUUID(),
      body: {
        operation: 'decision.score' as const,
        inputs: {
          schema_version: 1,
          state,
          instructions,
          criteria: criteria.map((text, index) => ({ id: `criterion-${index + 1}`, text })),
        },
      },
    };
    try {
      // Save the exact body and key before admission. Recovery never uses edited form fields.
      const prior = sessionStorage.getItem(storageKey);
      if (prior && !sameJson(JSON.parse(prior), saved))
        throw new Error('Saved submission changed. Review recorded jobs before continuing.');
      sessionStorage.setItem(storageKey, JSON.stringify(saved));
      setRecovery(saved);
      const accepted = await workJobs.submit(projectId, saved.body, saved.key);
      if (
        accepted.project_id !== projectId ||
        accepted.operation !== 'decision.score' ||
        !accepted.id.startsWith('core:') ||
        !['schema_version', 'state', 'instructions', 'criteria'].every((key) =>
          sameJson(accepted.inputs[key], saved.body.inputs[key]),
        )
      )
        throw new Error(
          'The submission response could not be verified. Recover the saved request before another submission.',
        );
      client.setQueryData<WorkJob>(['decision-job', projectId, accepted.id], accepted);
      void client.invalidateQueries({ queryKey: ['work-jobs', projectId] });
      if (!live(version)) return;
      setJobId(accepted.id);
      sessionStorage.setItem(receiptKey, accepted.id);
      sessionStorage.removeItem(storageKey);
      setRecovery(null);
    } catch (cause) {
      if (!live(version)) return;
      if (!recovery && cause instanceof WorkJobHttpError && [400, 422].includes(cause.status)) {
        try {
          sessionStorage.removeItem(storageKey);
          setRecovery(null);
        } catch {
          setJournalError('Submission recovery storage is unavailable.');
        }
      }
      setError(
        cause instanceof Error
          ? cause.message
          : 'Submission could not be verified. Recover the saved request before another submission.',
      );
    } finally {
      if (live(version)) {
        setPending(false);
        void status.refetch();
      }
    }
  }
  async function cancelJob() {
    if (!jobId || submitting) return;
    const version = generation.current;
    setPending(true);
    try {
      const cancelled = await workJobs.cancel(projectId, jobId);
      client.setQueryData(['decision-job', projectId, jobId], cancelled);
      if (live(version)) setError(cancelled.message ?? 'Cancellation requested.');
    } catch (cause) {
      if (live(version)) setError(cause instanceof Error ? cause.message : 'Cancellation could not be verified.');
    } finally {
      if (live(version)) setPending(false);
    }
  }
  const statusLabel = status.isPending
    ? 'Checking scorer…'
    : status.isError
      ? 'Status unavailable'
      : status.data?.busy
        ? 'Scorer busy'
        : ready
          ? 'Ready'
          : status.data?.configured
            ? 'Scorer unavailable'
            : 'Setup required';
  return (
    <section className="panel decision-panel" aria-label="Decision advisory">
      <WorkflowIntegration
        label="Advisory model"
        name="Muose-50M"
        meta="Local CPU"
        icon="chart"
        status="Experimental · advisory only"
      />
      <div className="decision-toolbar">
        <p className={ready ? 'decision-ready' : undefined} role="status">
          {statusLabel}
        </p>
        <div className="workbench-actions">
          {!ready && !status.isPending && !status.data?.busy && (
            <a className="text-link" href={publicPath('/guide/#decision-lab')}>
              Setup guide
            </a>
          )}
          <button
            type="button"
            className="text-button"
            aria-label="Refresh scorer status"
            disabled={pending || status.isFetching}
            onClick={() => void status.refetch()}
          >
            Refresh
          </button>
        </div>
      </div>
      <form
        className="workbench-form"
        onSubmit={(event) => {
          event.preventDefault();
          void score();
        }}
      >
        <fieldset disabled={pending || !!recovery || !!journalError}>
          <legend className="visually-hidden">Comparison</legend>
          <div className="workbench-field">
            <label htmlFor="decision-state">State to compare</label>
            <textarea
              id="decision-state"
              rows={3}
              maxLength={8000}
              placeholder="Describe the current situation"
              value={state}
              onChange={(event) => {
                edited();
                setState(event.target.value);
              }}
            />
          </div>
          <div className="workbench-field">
            <label htmlFor="decision-instructions">Comparison instructions</label>
            <textarea
              id="decision-instructions"
              rows={3}
              maxLength={2000}
              value={instructions}
              onChange={(event) => {
                edited();
                setInstructions(event.target.value);
              }}
            />
          </div>
          <div className="decision-criteria" role="group" aria-labelledby="decision-criteria-title">
            <h3 id="decision-criteria-title">Criteria</h3>
            <div className="workbench-field-grid">
              {criteria.map((text, index) => (
                <div className="workbench-field" key={index}>
                  <div className="decision-field-heading">
                    <label htmlFor={`decision-criterion-${index}`}>Criterion {index + 1}</label>
                    {criteria.length > 2 && (
                      <button
                        type="button"
                        className="text-button"
                        aria-label={`Remove criterion ${index + 1}`}
                        onClick={() => removeCriterion(index)}
                      >
                        Remove
                      </button>
                    )}
                  </div>
                  <textarea
                    id={`decision-criterion-${index}`}
                    ref={(element) => {
                      criterionFields.current[index] = element;
                    }}
                    rows={3}
                    maxLength={2000}
                    value={text}
                    onChange={(event) => {
                      edited();
                      setCriteria(criteria.map((item, position) => (position === index ? event.target.value : item)));
                    }}
                  />
                </div>
              ))}
            </div>
            <button
              type="button"
              className="secondary-button decision-add"
              disabled={criteria.length >= 8}
              onClick={addCriterion}
            >
              Add criterion
            </button>
          </div>
        </fieldset>
        <div className="workbench-actions decision-footer">
          <button
            type="submit"
            className="primary-button"
            disabled={!projectId || !hydrated || !!journalError || pending || (!recovery && (!ready || !valid))}
          >
            {pending ? 'Scoring…' : recovery ? 'Recover submission' : 'Start job'}
          </button>
          {pending && (
            <button type="button" className="secondary-button" onClick={() => void cancelJob()} disabled={!jobId}>
              Stop scoring
            </button>
          )}
        </div>
      </form>
      <div className="decision-model-details">
        <WorkbenchDisclosure title="Model details">
          {status.data?.message && <p>{publicMessage(status.data.message)}</p>}
          <p>
            Muose-50M runs on the application host CPU. Each score is saved as a job. Advisory results never authorize
            tools or robot actions.
          </p>
          <p>Not evaluated on robotics tasks. Weights are relative, not probabilities of correctness.</p>
          <p>Model: muose/Muose-50M-Decision · CC-BY-NC-SA-4.0 (noncommercial). An operator must accept the license.</p>
          <p>
            Use 2–8 unique criteria. The state, instructions and each criterion must fit in 512 tokens; longer text is
            rejected, not truncated.
          </p>
        </WorkbenchDisclosure>
      </div>
      {!projectId && <p>Select a project to save scoring jobs.</p>}
      {recovery && (
        <p role="status">
          An unresolved scoring request is saved. Recover it to verify the original outcome before starting another job.
        </p>
      )}
      {journalError && <p role="alert">{journalError}</p>}
      {job.error && <p role="alert">{job.error.message}</p>}
      {error && (
        <p role="alert" className="error-notice">
          {error}
        </p>
      )}
      {result && (
        <section className="decision-result" aria-label="Advisory score result">
          <h3>Advisory result</h3>
          <p>
            Highest relative score:{' '}
            {result.criteria[result.receipt.scores.findIndex((item) => item.id === result.receipt.selected_id)]} · no
            action was taken.
          </p>
          <ol>
            {result.receipt.scores.map((row, index) => (
              <li key={row.id}>
                <strong>{result.criteria[index]}</strong>: {(row.relative_weight * 100).toFixed(2)}% relative weight
                (uncalibrated); raw score {row.logit.toFixed(4)}; {row.tokens} tokens.
              </li>
            ))}
          </ol>
          <p>
            Observed CPU load: {result.receipt.timing_ms.load.toFixed(1)} ms · scoring:{' '}
            {result.receipt.timing_ms.score.toFixed(1)} ms
          </p>
          <details>
            <summary>Scoring details</summary>
            <p style={{ overflowWrap: 'anywhere' }}>
              Model revision: {result.receipt.revision}
              <br />
              Request SHA256: {result.receipt.request_sha256}
            </p>
          </details>
        </section>
      )}
    </section>
  );
}
