"use client";

import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, artifactDownloadUrl, isActive, type Job, type PolicyArtifact, type PolicyOptions } from "@/lib/api";
import { isCloudArtifact } from "@/lib/checkpoints";
import { actExportContext, startActExport, storedActExportAttempt, storeActExportReceipt, storedActExportReceipt } from "@/lib/act-export";
import { storeAttempt, type PolicyJobAttempt } from "@/lib/policy-job-attempt";
import { UncertainPolicyJob } from "@/lib/policy-job-mutation";

class ExportJournalUnavailable extends Error {
  constructor() { super("Browser session storage is unavailable or its export receipt is unreadable. Restore it and reload, then inspect recorded export jobs before submitting again."); }
}

export function ActExportControl({ projectId, checkpoint, runtimes, jobs, artifacts, active }: {
  projectId: string;
  checkpoint: PolicyArtifact;
  runtimes: PolicyOptions["runtimes"];
  jobs: Job[];
  artifacts: PolicyArtifact[];
  active: boolean;
}) {
  const client = useQueryClient();
  const [selectedRuntime, setSelectedRuntime] = useState("");
  const [error, setError] = useState("");
  const [readyContext, setReadyContext] = useState("");
  const [reviewedAttempt, setReviewedAttempt] = useState<PolicyJobAttempt>(null);
  const mounted = useRef(true), busy = useRef(false);
  const context = JSON.stringify([projectId, checkpoint.id]);
  const currentContext = useRef(context); currentContext.current = context;
  const attemptKey = ["act-export-attempt", projectId];
  const acceptedKey = ["act-export-accepted", projectId, checkpoint.id];
  const storageKey = ["act-export-storage-error", projectId];
  const attempt = useQuery<PolicyJobAttempt>({ queryKey: attemptKey, queryFn: async () => null, enabled: false, initialData: null, gcTime: Infinity });
  const accepted = useQuery<Job | null>({ queryKey: acceptedKey, queryFn: async () => null, enabled: false, initialData: null, gcTime: Infinity });
  const storageError = useQuery<string | null>({ queryKey: storageKey, queryFn: async () => null, enabled: false, initialData: null, gcTime: Infinity });
  const history = useQuery({ queryKey: ["jobs", projectId], queryFn: () => api.jobs(projectId), enabled: active && !!projectId, retry: false, refetchInterval: active ? 5_000 : false });
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  useEffect(() => {
    setError(""); setReviewedAttempt(null);
    try {
      if (!client.getQueryData<PolicyJobAttempt>(["act-export-attempt", projectId])) client.setQueryData(["act-export-attempt", projectId], storedActExportAttempt(projectId));
      if (!client.getQueryData<Job | null>(["act-export-accepted", projectId, checkpoint.id])) client.setQueryData(["act-export-accepted", projectId, checkpoint.id], storedActExportReceipt(projectId, checkpoint.id));
      setReadyContext(context);
    } catch { client.setQueryData(["act-export-storage-error", projectId], new ExportJournalUnavailable().message); }
  }, [client, projectId, checkpoint.id, context]);
  const available = runtimes.filter(item => item.act_export && item.enabled !== false &&
    item.execution !== "skypilot" && item.provider === "local");
  const runtime = available.find(item => item.id === selectedRuntime) ?? available[0];
  const observed = history.data ?? jobs;
  const related = observed.filter(item => item.project_id === projectId && item.kind === "policy.export" && "artifact_id" in item.request && item.request.artifact_id === checkpoint.id);
  // A stale/failed history read cannot erase a verified acknowledgment, including after remount.
  const retained = accepted.data?.project_id === projectId ? accepted.data : null;
  const recorded = retained && related.find(item => item.id === retained.id);
  const receipt = recorded ?? retained;
  const pending = related.find(isActive) ?? (receipt && isActive(receipt) ? receipt : null);
  const recoveryPending = observed.find(item => item.project_id === projectId && item.kind === "policy.export" && isActive(item));
  const latest = receipt ?? related[0];
  const exports = artifacts.filter(item => item.project_id === projectId && item.format === "inference_export" && item.parent_ids?.some(id =>
    id === checkpoint.id || artifacts.some(parent => parent.project_id === projectId && parent.id === id && parent.parent_ids?.includes(checkpoint.id))));
  const cloud = isCloudArtifact(checkpoint);
  const sourceJob = observed.find(item => item.project_id === projectId && item.id === checkpoint.job_id);
  const dataset = checkpoint.metadata?.dataset as { source?: string } | undefined;
  const issue = !projectId || checkpoint.project_id !== projectId ? "Select a ready project to export." : cloud && (sourceJob?.status !== "succeeded" || checkpoint.metadata?.reload_verified !== true)
    ? "Wait for the completed, reload-verified ACT checkpoint. Intermediate saves cannot be exported."
    : checkpoint.format !== "training_checkpoint" || checkpoint.metadata?.training_backend !== "lerobot" || checkpoint.metadata?.method !== "full"
      ? "This export requires a complete native ACT training checkpoint."
      : dataset?.source !== "huggingface" ? "ACT export from local dataset snapshots is not supported yet."
      : !runtime ? "Ask the app operator to configure the local ACT export worker." : null;
  const journalReady = readyContext === context && !storageError.data;
  const submitting = journalReady && attempt.data?.state === "pending";
  const ready = active && journalReady && !issue && !pending && !attempt.data && history.isSuccess && !history.isError;
  function storageFailed() {
    const failure = new ExportJournalUnavailable();
    client.setQueryData(storageKey, failure.message);
    // Preserve the disk's pending evidence, but do not leave a remounted view saying Submitting.
    const previous = client.getQueryData<PolicyJobAttempt>(attemptKey);
    if (previous?.state === "pending") client.setQueryData<PolicyJobAttempt>(attemptKey, (): PolicyJobAttempt => ({ state: "uncertain", message: `${previous.message} ${failure.message}` }));
    return failure;
  }
  function saveAttempt(value: PolicyJobAttempt) {
    try { storeAttempt("policy.export", projectId, value); }
    catch { throw storageFailed(); }
    client.setQueryData<PolicyJobAttempt>(attemptKey, value);
  }
  async function submit() {
    const cachedReceipt = client.getQueryData<Job | null>(acceptedKey);
    if (busy.current || !ready || !runtime || client.getQueryData<PolicyJobAttempt>(attemptKey) || client.getQueryData<string | null>(storageKey) || (cachedReceipt && !related.some(item => item.id === cachedReceipt.id && !isActive(item)) && isActive(cachedReceipt))) return;
    const expected = { project: projectId, artifact: checkpoint.id, runtime: runtime.id };
    busy.current = true; setError(""); setReviewedAttempt(null);
    try {
      saveAttempt({ state: "pending", message: actExportContext(expected) });
      const job = await startActExport(expected);
      // Acceptance is retained before any fallible storage cleanup or history refresh, even unmounted.
      client.setQueryData<Job | null>(acceptedKey, job);
      try { storeActExportReceipt(job); } catch { throw storageFailed(); }
      saveAttempt(null);
      void client.invalidateQueries({ queryKey: ["jobs", projectId] });
      void client.invalidateQueries({ queryKey: ["artifacts", projectId] });
    } catch (cause) {
      let message = cause instanceof Error ? cause.message : "The export request failed.";
      try {
        if (cause instanceof ExportJournalUnavailable) { /* Preserve recovery state; never retry failed cleanup. */ }
        else if (cause instanceof UncertainPolicyJob) { message = `${actExportContext(expected)} ${message}`; saveAttempt({ state: "uncertain", message }); }
        else saveAttempt(null);
      } catch (failure) { message = failure instanceof Error ? failure.message : message; }
      if (mounted.current && currentContext.current === context) setError(message);
    } finally { busy.current = false; }
  }
  async function refresh() {
    const before = client.getQueryData<PolicyJobAttempt>(attemptKey);
    const response = await history.refetch();
    void client.invalidateQueries({ queryKey: ["artifacts", projectId] });
    if (mounted.current && currentContext.current === context && before?.state === "uncertain" && before === client.getQueryData<PolicyJobAttempt>(attemptKey) && !response.isError) setReviewedAttempt(before);
  }
  function acknowledge() {
    if (!active || !journalReady || pending || recoveryPending || history.isError || !reviewedAttempt || reviewedAttempt !== client.getQueryData<PolicyJobAttempt>(attemptKey)) return;
    try { saveAttempt(null); setReviewedAttempt(null); setError(""); } catch (cause) { setError((cause as Error).message); }
  }
  return <section aria-label="ACT inference export">
    {cloud && <p className="training-monitor-note">This action downloads the completed checkpoint to this computer (up to 4 GiB), verifies a separate local copy, then exports on CPU. It does not start a cloud GPU. The original cloud checkpoint is preserved.</p>}
    <p className="training-monitor-note">Remove the training-only VAE and keep FP32 weights. Each export checks complete synthetic action chunks in fresh CPU processes. This does not establish robot task success, calibration, GPU fit, or faster inference.</p>
    {available.length > 1 && <label>Export computer<select value={runtime?.id ?? ""} disabled={!ready} onChange={event => setSelectedRuntime(event.target.value)}>{available.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>}
    <button type="button" className="primary-button" disabled={!ready} onClick={() => void submit()}>{submitting || pending ? "Exporting ACT policy…" : cloud ? "Download checkpoint and export" : "Export ACT inference package"}</button>
    <button type="button" className="text-link" disabled={!active || history.isFetching} onClick={() => void refresh()}>Refresh export jobs</button>
    {issue && <p role="status" className="training-monitor-note">{issue}</p>}
    {latest && <section aria-label="ACT export receipt"><p role="status">Export {latest.status} · {latest.id}. {pending ? pending.stage === "downloading" ? "Downloading and verifying the checkpoint." : "Preparing or verifying the CPU export." : "Recorded export job."}</p></section>}
    {latest?.status === "failed" && <p role="alert">Export failed: {latest.error ?? "See the recorded job for details."}</p>}
    {latest && ["cancelled", "interrupted"].includes(latest.status) && <p role="status">Export {latest.status}. No verified package is ready from this attempt.</p>}
    {history.isError && <p role="alert">Export job updates are unavailable. Previously received status may be stale.</p>}
    {(storageError.data || error) && <p role="alert">{storageError.data || error}</p>}
    {submitting && <p role="status">{attempt.data?.message}</p>}
    {attempt.data?.state === "uncertain" && <section aria-label="ACT export recovery" className="warning-box"><p>{attempt.data.message} Further submissions are paused.</p>{recoveryPending && <p role="status">Recorded export {recoveryPending.id} for checkpoint {"artifact_id" in recoveryPending.request ? recoveryPending.request.artifact_id : "unknown"} is {recoveryPending.status}. Wait for active project exports to finish before clearing this recovery record.</p>}<button className="secondary-button" disabled={!active || !journalReady || !!pending || !!recoveryPending || history.isError || reviewedAttempt !== attempt.data} onClick={acknowledge}>I checked the export jobs; allow a new request</button></section>}
    {exports.map(item => <p key={item.id}><a className="text-link" href={artifactDownloadUrl(projectId, item.id)}>Download ACT inference package</a><span className="training-monitor-note"> · Inference only; keep the original checkpoint for training.</span></p>)}
  </section>;
}
