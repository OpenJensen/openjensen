"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type GcpComputeSettings } from "@/lib/api";
import { GpuPicker } from "@/components/gpu-picker";
import "./cloud-gpu-settings.css";

export function CloudGpuSettingsPanel() {
  const client = useQueryClient();
  const settings = useQuery({ queryKey: ["compute-settings"], queryFn: api.computeSettings, retry: false });
  const [draft, setDraft] = useState<GcpComputeSettings | null>(null);
  const saved = settings.data?.gcp;
  const current = draft ?? saved ?? { enabled: true, default_gpu: "A100", disk_size_gb: 200, idle_minutes: 10 };
  const choices = settings.data?.gpu_options ?? [];
  const dirty = !!saved && JSON.stringify(current) !== JSON.stringify(saved);
  const save = useMutation({
    mutationFn: () => api.saveCloudCompute(current),
    onSuccess: (result) => {
      client.setQueryData(["compute-settings"], result);
      void client.invalidateQueries({ queryKey: ["policy-options"] });
      setDraft(null);
    },
  });
  function update<K extends keyof GcpComputeSettings>(key: K, value: GcpComputeSettings[K]) {
    setDraft({ ...current, [key]: value });
    save.reset();
  }
  return (
    <details className="cloud-gpu-settings">
      <summary>Training preferences <span>{current.default_gpu}</span></summary>
      <div className="cloud-gpu-settings-body">
        {settings.isPending && <p role="status" className="cloud-auth-note">Loading…</p>}
        {settings.error && <div className="cloud-connection-error" role="alert">{settings.error.message}<button className="text-button" onClick={() => void settings.refetch()}>Retry</button></div>}
        {saved && <form onSubmit={(event) => { event.preventDefault(); save.mutate(); }}>
          <GpuPicker label="Default GPU" value={current.default_gpu} disabled={save.isPending}
            onChange={(value) => update("default_gpu", value)}
            choices={choices.filter((gpu) => gpu.supported).map((gpu) => ({ id: gpu.id, label: gpu.id, memory: `${gpu.gpu_memory_mib / 1024} GB` }))} />
          <details className="cloud-gpu-advanced">
            <summary>Advanced</summary>
            <label className="local-compute-toggle"><input type="checkbox" checked={current.enabled} disabled={save.isPending} onChange={(event) => update("enabled", event.target.checked)} />Allow cloud training</label>
            <div className="cloud-gpu-fields">
              <label>Disk size (GB)<input type="number" min={100} max={2000} step={50} value={current.disk_size_gb} disabled={save.isPending} onChange={(event) => update("disk_size_gb", Number(event.target.value))} /></label>
              <label>Idle shutdown (minutes)<input type="number" min={1} max={60} value={current.idle_minutes} disabled={save.isPending} onChange={(event) => update("idle_minutes", Number(event.target.value))} /></label>
            </div>
          </details>
          <button type="submit" className="secondary-button" disabled={!dirty || save.isPending || !Number.isInteger(current.disk_size_gb) || current.disk_size_gb < 100 || current.disk_size_gb > 2000 || !Number.isInteger(current.idle_minutes) || current.idle_minutes < 1 || current.idle_minutes > 60}>
            {save.isPending ? "Saving…" : "Save preferences"}
          </button>
          {save.isSuccess && !dirty && <p className="cloud-feedback" role="status">Preferences saved.</p>}
          {save.error && <p className="cloud-connection-error" role="alert">{save.error.message}</p>}
        </form>}
      </div>
    </details>
  );
}
