"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type ComputeSettings, type LocalComputeSettings, type LocalWorkerDiscovery, type PolicyOptions } from "@/lib/api";
import { publicPath } from "@/lib/base-path";
import "./local-compute-settings.css";

export function LocalComputeSettingsPanel() {
  const client = useQueryClient();
  const options = useQuery({ queryKey: ["policy-options"], queryFn: api.policyOptions });
  const [draft, setDraft] = useState<LocalComputeSettings | null>(null);
  const [discovery, setDiscovery] = useState<LocalWorkerDiscovery | null>(null);
  const [workerNotice, setWorkerNotice] = useState("");
  const saved = options.data?.compute?.local;
  const current = draft ?? saved ?? { enabled: false, label: "Local machine" };
  const workers = (options.data?.runtimes ?? []).filter(
    (worker) => (worker.provider ?? "local") === "local" && worker.training && worker.device === "cuda",
  );
  function updateCompute(result: ComputeSettings) {
    client.setQueryData<PolicyOptions>(["policy-options"], (previous) => previous
      ? { ...previous, compute: { ...previous.compute, local: result.local, ...(result.gcp ? { gcp: result.gcp } : {}) }, runtimes: result.runtimes }
      : previous);
    void client.invalidateQueries({ queryKey: ["policy-options"] });
    void client.invalidateQueries({ queryKey: ["compute-settings"] });
  }
  const check = useMutation({
    mutationFn: api.checkLocalWorkers,
    onSuccess: setDiscovery,
  });
  const add = useMutation({
    mutationFn: api.addLocalWorker,
    onSuccess: (result) => {
      setDiscovery(result.discovery);
      updateCompute(result.compute);
      setWorkerNotice(`${result.runtime.label} added.${result.compute.local.enabled ? "" : " Enable local runs when you’re ready."}`);
    },
  });
  const save = useMutation({
    mutationFn: () => api.saveComputeSettings({ ...current, label: current.label.trim() }),
    onSuccess: (result) => {
      updateCompute(result);
      setDraft(null);
    },
  });
  const dirty = !!saved && (current.enabled !== saved.enabled || current.label.trim() !== saved.label);
  const busy = check.isPending || add.isPending;
  const registeredIds = new Set(discovery?.candidates.map((candidate) => candidate.runtime_id).filter(Boolean));
  const otherWorkers = workers.filter((worker) => !registeredIds.has(worker.id));
  const needsSetup = discovery && (discovery.status !== "ready" || discovery.candidates.some((candidate) => candidate.status === "setup_required"));
  const issues = [...new Set(discovery?.issues ?? [])].filter((issue) => issue !== discovery?.message);
  const gpuDetails = (gpu: { gpu_name?: string | null; gpu_memory_mib?: number | null }) => [
    gpu.gpu_name,
    gpu.gpu_memory_mib ? `${Number((gpu.gpu_memory_mib / 1024).toFixed(1))} GB` : null,
  ].filter(Boolean).join(" · ");
  return (
    <section className="local-compute-settings" aria-labelledby="local-compute-title">
      <div className="local-compute-heading">
        <div><h2 id="local-compute-title">Local runs</h2></div>
        <span className="cloud-status">{saved ? (current.enabled ? "Enabled" : "Disabled") : "Loading…"}</span>
      </div>
      <div className="local-worker-discovery" aria-busy={busy}>
        <div className="local-discovery-actions">
          <button className="secondary-button" type="button" disabled={busy || save.isPending}
            onClick={() => { add.reset(); setWorkerNotice(""); check.mutate(); }}>
            {check.isPending ? "Checking…" : discovery ? "Check again" : check.isError ? "Retry check" : "Check this machine"}
          </button>
          {discovery && <span className="local-discovery-host" title={`${discovery.host.platform} · ${discovery.host.architecture}`}>App host · {discovery.host.name}</span>}
        </div>
        {check.isPending && <p className="local-compute-note" role="status">Checking the machine running this app…</p>}
        {check.error && <p className="cloud-connection-error" role="alert">{check.error.message}</p>}
        {discovery && <div className="local-discovery-results">
          {discovery.message && (discovery.status !== "ready" || !discovery.candidates.length) && <p className="local-compute-note">{discovery.message}</p>}
          {discovery.candidates.map((candidate) => <div className="local-worker-card" key={candidate.id}>
            <div className="local-worker-details">
              <strong>{candidate.label}</strong>
              {gpuDetails(candidate) && <span>{gpuDetails(candidate)}</span>}
              {candidate.reason && <p>{candidate.reason}</p>}
            </div>
            {candidate.status === "ready" ? <button className="secondary-button" type="button"
              aria-label={`Add ${candidate.label}`} disabled={busy || save.isPending || check.isError}
              onClick={() => { setWorkerNotice(""); add.mutate(candidate.id); }}>
              {add.isPending && add.variables === candidate.id ? "Adding…" : "Add worker"}
            </button> : <span className="cloud-status">{candidate.status === "registered" ? "Added" : "Setup needed"}</span>}
          </div>)}
          {!!issues.length && <ul className="local-discovery-issues">{issues.map((issue) => <li key={issue}>{issue}</li>)}</ul>}
          {needsSetup && <a className="text-link" href={publicPath('/guide/#settings-diagnostics')}>Setup guide</a>}
        </div>}
        {otherWorkers.map((worker) => <div className="local-worker-card" key={worker.id}>
          <div className="local-worker-details"><strong>{worker.label}</strong><span>{gpuDetails(worker)}</span></div>
          <span className="cloud-status">Configured</span>
        </div>)}
        {add.error && <p className="cloud-connection-error" role="alert">{add.error.message}</p>}
        {workerNotice && <p className="cloud-feedback" role="status">{workerNotice}</p>}
      </div>
      <form onSubmit={(event) => { event.preventDefault(); save.mutate(); }}>
        <label className="local-compute-toggle">
          <input type="checkbox" checked={current.enabled} disabled={!saved || save.isPending || busy}
            onChange={(event) => { save.reset(); setDraft({ ...current, enabled: event.target.checked }); }} />
          Enable local runs
        </label>
        <div className="local-compute-fields">
          <label>Machine label
            <input value={current.label} maxLength={80} required disabled={!saved || save.isPending || busy}
              placeholder="My RTX 3070 workstation"
              onChange={(event) => { save.reset(); setDraft({ ...current, label: event.target.value }); }} />
          </label>
          <button className="secondary-button" type="submit" disabled={!dirty || !current.label.trim() || save.isPending || busy}>
            {save.isPending ? "Saving…" : "Save local settings"}
          </button>
        </div>
        {dirty && <p className="local-compute-note">Unsaved changes</p>}
        {options.error && <p className="cloud-connection-error" role="alert">{options.error.message}</p>}
        {save.error && <p className="cloud-connection-error" role="alert">{save.error.message}</p>}
        {save.isSuccess && !dirty && <p className="cloud-feedback" role="status">Local settings saved.</p>}
      </form>
    </section>
  );
}
