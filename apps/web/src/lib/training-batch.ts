'use client';

import { useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiOrigin, type Job, type PolicyRequest } from './api';
import { PolicyJobHttpError, policyJobRequest, record } from './policy-job-mutation';
import { trainingReceipt } from './training-submission';

type Entry = { key: string; body: PolicyRequest; state: 'ready' | 'pending' | 'unknown' | 'accepted' | 'rejected'; job?: Job; missing?: boolean };
type Journal = { version: 1; project: string; entries: Entry[] };
type State = { hydrated: boolean; available: boolean; busy: boolean; journal: Journal | null; error: string };
const initial: State = { hydrated: false, available: false, busy: false, journal: null, error: '' };

/** Each model has its own immutable request/key. Reload only reconciles on explicit action. */
export function useTrainingBatch(project: string) {
  const client = useQueryClient();
  const storageKey = `firebird:training-batch:v1:${encodeURIComponent(apiOrigin || 'same-origin')}:${encodeURIComponent(project)}`;
  const queryKey = ['training-batch', storageKey];
  const state = useQuery<State>({ queryKey, queryFn: async () => initial, initialData: initial, enabled: false, gcTime: Infinity }).data;
  const current = () => client.getQueryData<State>(queryKey) ?? initial;
  const put = (patch: Partial<State>) => client.setQueryData(queryKey, { ...current(), ...patch });
  function save(journal: Journal | null) {
    const before = current().journal;
    const stored = sessionStorage.getItem(storageKey);
    if (stored !== (before ? JSON.stringify(before) : null)) throw new Error('The saved training plan changed. Check recorded jobs before continuing.');
    if (journal) sessionStorage.setItem(storageKey, JSON.stringify(journal));
    else sessionStorage.removeItem(storageKey);
    put({ journal });
  }
  useEffect(() => {
    if (current().hydrated) return;
    try {
      const raw = sessionStorage.getItem(storageKey);
      let journal: Journal | null = null;
      if (raw !== null) {
        if (raw.length > 2_000_000) throw new Error('Saved training plan is too large.');
        const parsed: unknown = JSON.parse(raw);
        if (!record(parsed) || parsed.version !== 1 || parsed.project !== project || !Array.isArray(parsed.entries) || parsed.entries.length < 2 || parsed.entries.length > 16) throw new Error('Saved training plan is unreadable. Inspect recorded jobs.');
        const keys = new Set<string>();
        for (const entry of parsed.entries) {
          if (!record(entry) || typeof entry.key !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(entry.key) || keys.has(entry.key) || !record(entry.body) || entry.body.operation !== 'policy.finetune' || typeof entry.body.runtime_id !== 'string' || !record(entry.body.training) || !['ready', 'pending', 'unknown', 'accepted', 'rejected'].includes(String(entry.state))) throw new Error('Saved training plan is unreadable. Inspect recorded jobs.');
          keys.add(entry.key);
          if (entry.state === 'accepted') trainingReceipt(entry.job, project, entry.body);
        }
        journal = parsed as unknown as Journal;
      }
      put({ hydrated: true, available: true, journal });
    } catch (error) { put({ hydrated: true, available: false, error: error instanceof Error ? error.message : 'Recovery storage is unavailable.' }); }
    // A shared cache fences old mounts and never automatically replays a POST.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project, storageKey, client]);
  function retain(job: Job) {
    client.setQueryData<Job[]>(['jobs', project], previous => [job, ...(previous ?? []).filter(item => item.id !== job.id)]);
  }
  async function execute(post: boolean) {
    if (current().busy || !current().available || !current().journal) return [];
    put({ busy: true, error: '' });
    try {
      for (let index = 0; index < current().journal!.entries.length; index++) {
        let journal = current().journal!;
        let entry = journal.entries[index];
        if (entry.state === 'accepted') { retain(entry.job!); continue; }
        if (entry.state === 'rejected') continue;
        if (entry.state === 'ready' && !post) continue;
        let value: unknown;
        try {
          value = await policyJobRequest(`/projects/${encodeURIComponent(project)}/submissions/${entry.key}?operation=policy.finetune`, undefined, { idempotencyKey: entry.key });
        } catch (error) {
          if (!(error instanceof PolicyJobHttpError) || error.status !== 404) throw error;
          if (!post) {
            journal = { ...journal, entries: journal.entries.map((item, row) => row === index ? { ...item, state: 'unknown', missing: true } : item) };
            save(journal);
            continue;
          }
          const initialPost = entry.state === 'ready';
          journal = { ...journal, entries: journal.entries.map((item, row) => row === index ? { ...item, state: 'pending', missing: false } : item) };
          save(journal); // Durable before any job acceptance can happen.
          try {
            value = await policyJobRequest(`/projects/${encodeURIComponent(project)}/policy-jobs`, entry.body, { idempotencyKey: entry.key, allowInitialRejection: initialPost });
          } catch (fault) {
            journal = current().journal!;
            save({ ...journal, entries: journal.entries.map((item, row) => row === index ? { ...item, state: fault instanceof PolicyJobHttpError && fault.initialRejection ? 'rejected' : 'unknown' } : item) });
            throw fault;
          }
        }
        const job = trainingReceipt(value, project, entry.body);
        retain(job); // Preserve a verified ACK even if the next storage write fails.
        journal = current().journal!;
        save({ ...journal, entries: journal.entries.map((item, row) => row === index ? { ...item, state: 'accepted', job, missing: false } : item) });
      }
      const jobs = current().journal!.entries.flatMap(item => item.job ? [item.job] : []);
      if (current().journal!.entries.every(item => item.state === 'accepted')) save(null);
      return jobs;
    } catch (error) {
      put({ error: error instanceof Error ? error.message : 'Check the saved training plan.' });
      return current().journal?.entries.flatMap(item => item.job ? [item.job] : []) ?? [];
    } finally { put({ busy: false }); void client.invalidateQueries({ queryKey: ['jobs', project] }); }
  }
  async function start(bodies: PolicyRequest[]) {
    if (!current().hydrated || !current().available || current().busy || current().journal) throw new Error('Resolve the saved training plan first.');
    put({ busy: true, error: '' });
    try {
      const preflight = await policyJobRequest(`/projects/${encodeURIComponent(project)}/training-plan`, bodies);
      if (!record(preflight) || preflight.valid !== true || preflight.jobs !== bodies.length || preflight.gpus_per_job !== 1) throw new Error('The application did not validate every independent GPU job.');
      const entries: Entry[] = JSON.parse(JSON.stringify(bodies)).map((body: PolicyRequest) => ({ key: crypto.randomUUID(), body, state: 'ready' }));
      save({ version: 1, project, entries });
    } finally { put({ busy: false }); }
    return execute(true);
  }
  const canContinue = !!state.journal && state.journal.entries.every(entry => ['ready', 'accepted', 'rejected'].includes(entry.state) || entry.missing === true);
  return { ...state, start, check: () => execute(false), continue: () => canContinue ? execute(true) : Promise.resolve([]), canContinue,
    dismiss: () => { if (!current().busy && current().journal?.entries.every(item => ['accepted', 'rejected', 'ready'].includes(item.state))) save(null); } };
}
