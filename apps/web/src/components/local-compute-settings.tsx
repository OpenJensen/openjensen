"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type LocalComputeSettings, type PolicyOptions } from "@/lib/api";
import { Icon } from "@/components/icon";
import "./local-compute-settings.css";

export function LocalComputeSettingsPanel() {
  const client = useQueryClient();
  const options = useQuery({ queryKey: ["policy-options"], queryFn: api.policyOptions });
  const [draft, setDraft] = useState<LocalComputeSettings | null>(null);
  const saved = options.data?.compute?.local;
  const current = draft ?? saved ?? { enabled: true, label: "Local machine" };
  const workers = (options.data?.runtimes ?? []).filter(
    (worker) => (worker.provider ?? "local") === "local" && worker.training && worker.device === "cuda",
  );
  const save = useMutation({
    mutationFn: () => api.saveComputeSettings({ ...current, label: current.label.trim() }),
    onSuccess: (result) => {
      client.setQueryData<PolicyOptions>(["policy-options"], (previous) => previous
        ? { ...previous, compute: { ...previous.compute, local: result.local, ...(result.gcp ? { gcp: result.gcp } : {}) }, runtimes: result.runtimes }
        : previous);
      setDraft(null);
      void client.invalidateQueries({ queryKey: ["policy-options"] });
      void client.invalidateQueries({ queryKey: ["compute-settings"] });
    },
  });
  const dirty = !!saved && (current.enabled !== saved.enabled || current.label.trim() !== saved.label);
  return (
    <section className="local-compute-settings" aria-labelledby="local-compute-title">
      <div className="local-compute-heading">
        <Icon name="layers" size={22} />
        <div><h2 id="local-compute-title">Local runs</h2></div>
        <span className="cloud-status">{saved ? (current.enabled ? "Enabled" : "Disabled") : "Loading…"}</span>
      </div>
      <form onSubmit={(event) => { event.preventDefault(); save.mutate(); }}>
        <label className="local-compute-toggle">
          <input type="checkbox" checked={current.enabled} disabled={!saved || save.isPending}
            onChange={(event) => { save.reset(); setDraft({ ...current, enabled: event.target.checked }); }} />
          Enable local runs
        </label>
        <div className="local-compute-fields">
          <label>Machine label
            <input value={current.label} maxLength={80} required disabled={!saved || save.isPending}
              placeholder="My RTX 3070 workstation"
              onChange={(event) => { save.reset(); setDraft({ ...current, label: event.target.value }); }} />
          </label>
          <button className="secondary-button" type="submit" disabled={!dirty || !current.label.trim() || save.isPending}>
            {save.isPending ? "Saving…" : "Save local settings"}
          </button>
        </div>
        <p className="local-compute-note">{workers.length
          ? `${workers.length} configured GPU worker${workers.length === 1 ? "" : "s"}`
          : "No local GPU worker. See Local worker setup below."}</p>
        {dirty && <p className="local-compute-note">Unsaved changes</p>}
        {options.error && <p className="cloud-connection-error" role="alert">{options.error.message}</p>}
        {save.error && <p className="cloud-connection-error" role="alert">{save.error.message}</p>}
        {save.isSuccess && !dirty && <p className="cloud-feedback" role="status">Local settings saved.</p>}
      </form>
    </section>
  );
}
