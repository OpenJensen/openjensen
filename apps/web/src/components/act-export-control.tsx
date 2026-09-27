"use client";

import { publicDemo } from "@/lib/public-demo";
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, artifactDownloadUrl, isActive, type Job, type PolicyArtifact, type PolicyOptions } from "@/lib/api";
import { isCloudArtifact } from "@/lib/checkpoints";

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
  const available = runtimes.filter(item => item.act_export && item.enabled !== false &&
    item.execution !== "skypilot" && item.provider === "local");
  const runtime = available.find(item => item.id === selectedRuntime) ?? available[0];
  const related = jobs.filter(item => item.kind === "policy.export" && "artifact_id" in item.request && item.request.artifact_id === checkpoint.id);
  const pending = related.find(isActive);
  const latest = related[0];
  const exports = artifacts.filter(item => item.format === "inference_export" && item.parent_ids?.some(id =>
    id === checkpoint.id || artifacts.some(parent => parent.id === id && parent.parent_ids?.includes(checkpoint.id))));
  const cloud = isCloudArtifact(checkpoint);
  const sourceJob = jobs.find(item => item.id === checkpoint.job_id);
  const dataset = checkpoint.metadata?.dataset as { source?: string } | undefined;
  const issue = !projectId ? "Select a ready project to export." : cloud && (sourceJob?.status !== "succeeded" || checkpoint.metadata?.reload_verified !== true)
    ? "Wait for the completed, reload-verified ACT checkpoint. Intermediate saves cannot be exported."
    : checkpoint.format !== "training_checkpoint" || checkpoint.metadata?.training_backend !== "lerobot" || checkpoint.metadata?.method !== "full"
      ? "This export requires a complete native ACT training checkpoint."
      : dataset?.source !== "huggingface" ? "ACT export from local dataset snapshots is not supported yet."
      : !runtime ? "Ask the app operator to configure the local ACT export worker." : null;
  const mutation = useMutation({
    mutationFn: () => {
      if (issue || !active || !runtime || pending) throw new Error(issue ?? "An export is already running or this view is inactive.");
      return api.policyJob(projectId, { operation: "policy.export", runtime_id: runtime.id, artifact_id: checkpoint.id, training_method: "full", timeout_seconds: 600 });
    },
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["jobs", projectId] });
      void client.invalidateQueries({ queryKey: ["artifacts", projectId] });
    },
  });
  return <section aria-label="ACT inference export">
    {cloud && <p className="training-monitor-note">This action downloads the completed checkpoint to this computer (up to 4 GiB), verifies a separate local copy, then exports on CPU. It does not start a cloud GPU. The original cloud checkpoint is preserved.</p>}
    <p className="training-monitor-note">Remove the training-only VAE and keep FP32 weights. Each export checks complete synthetic action chunks in fresh CPU processes. This does not establish robot task success, calibration, GPU fit, or faster inference.</p>
    {available.length > 1 && <label>Export computer<select value={runtime?.id ?? ""} disabled={publicDemo || !active || mutation.isPending || !!pending} onChange={event => setSelectedRuntime(event.target.value)}>{available.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>}
    <button type="button" className="primary-button" disabled={publicDemo || !active || !!issue || mutation.isPending || !!pending} onClick={() => mutation.mutate()}>{mutation.isPending || pending ? "Exporting ACT policy…" : cloud ? "Download checkpoint and export" : "Export ACT inference package"}</button>
    {issue && <p role="status" className="training-monitor-note">{issue}</p>}
    {pending && <p role="status">Export {pending.status}. {pending.stage === "downloading" ? "Downloading and verifying the checkpoint." : "Preparing or verifying the CPU export."}</p>}
    {latest?.status === "failed" && <p role="alert">Export failed: {latest.error ?? "See the recorded job for details."}</p>}
    {latest && ["cancelled", "interrupted"].includes(latest.status) && <p role="status">Export {latest.status}. No verified package is ready from this attempt.</p>}
    {mutation.error && <p role="alert">{mutation.error.message}</p>}
    {!publicDemo && exports.map(item => <p key={item.id}><a className="text-link" href={artifactDownloadUrl(projectId, item.id)}>Download ACT inference package</a><span className="training-monitor-note"> · Inference only; keep the original checkpoint for training.</span></p>)}
  </section>;
}
