"use client";

import { useEffect, useState } from "react";
import { CloudConnectionsPanel } from "@/components/cloud-connections";
import { conciseRunError, observedProgress, runLabel, runSummary } from "@/lib/run-summary";
import { checkpointLabel, isCloudArtifact, quantizationIssue, sortCheckpoints } from "@/lib/checkpoints";
import "./workflow-panel.css";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BenchmarkReference } from "@/components/benchmark-reference";
import {
  api,
  artifactDownloadUrl,
  isActive,
  type Job,
  type LifecycleResult,
  type PolicyRequest,
} from "@/lib/api";

type Preferences = {
  mode: "engine" | "libero";
  suite: "libero_object" | "libero_spatial";
  taskIds: string;
  parityProfile: string;
  parityRmse: string;
  parityMaxError: string;
  repetitions: number;
  steps: number;
  warmup: number;
  task: number;
  searchStates: string;
  finalStates: string;
  trainingSteps: number;
  camera: string;
  select: boolean;
  minSuccess: number;
  maxLatency: number;
  maxMemory: number;
  precision: "recommended" | "Q4_0" | "Q8_0";
  vision: boolean;
  compareQ4: boolean;
};
const initial: Preferences = {
  mode: "engine",
  suite: "libero_object",
  taskIds: "0,1,2,3,4,5,6,7,8,9",
  parityProfile: "",
  parityRmse: "",
  parityMaxError: "",
  repetitions: 10,
  steps: 500,
  warmup: 3,
  task: 0,
  searchStates: "0,1",
  finalStates: "2,3",
  trainingSteps: 20000,
  camera: "observation.images.front",
  select: false,
  minSuccess: 100,
  maxLatency: 1000,
  maxMemory: 8192,
  precision: "recommended",
  vision: false,
  compareQ4: false,
};
function states(text: string) {
  const values = text.split(",").map((x) => Number(x.trim()));
  if (!text.trim() || values.some((x) => !Number.isInteger(x) || x < 0))
    throw new Error("Enter comma-separated, non-negative state IDs.");
  return values;
}
function resultOf(job?: Job): LifecycleResult | undefined {
  return job?.result && "artifacts" in job.result ? job.result : undefined;
}

export function WorkflowPanel({
  projectId,
  stage,
  tab,
  onTabChange: setTab,
  onOpenQuantize,
  preferredArtifactId,
  onViewTraining,
}: {
  projectId: string;
  stage: string;
  tab: "compute" | "settings" | "diagnostics";
  onTabChange: (tab: "compute" | "settings" | "diagnostics") => void;
  onOpenQuantize: () => void;
  preferredArtifactId?: string;
  onViewTraining?: () => void;
}) {
  const client = useQueryClient();
  const [preferences, setPreferences] = useState(initial);
  const [loadedProjectId, setLoadedProjectId] = useState<string | null>(null);
  const ready = !!projectId && loadedProjectId === projectId;
  const [runtimeId, setRuntimeId] = useState("");
  const [input, setInput] = useState(preferredArtifactId ?? (stage === "settings" ? "" : "latest"));
  const [datasetId, setDatasetId] = useState("");
  const [method, setMethod] = useState("lora");
  const [resumeId, setResumeId] = useState("");
  const [selectedJobId, setSelectedJobId] = useState("");
  const [detailsJobId, setDetailsJobId] = useState<string | null>(null);
  const options = useQuery({
    queryKey: ["policy-options"],
    queryFn: api.policyOptions,
  });
  const jobs = useQuery({
    queryKey: ["jobs", projectId],
    queryFn: () => api.jobs(projectId),
    enabled: !!projectId,
    refetchInterval: (query) =>
      query.state.data?.some(isActive) ? 1000 : 5000,
  });
  const artifacts = useQuery({
    queryKey: ["artifacts", projectId],
    queryFn: () => api.artifacts(projectId),
    enabled: !!projectId,
    refetchInterval: 3000,
  });
  useEffect(() => {
    if (!projectId) return;
    try {
      const restored: Preferences = {
        ...initial,
        ...JSON.parse(
          localStorage.getItem(`firebird.workflow.${projectId}`) ?? "{}",
        ),
      };
      // Restore saved Spatial preferences from before the protocol was restricted.
      setPreferences(restored.suite === "libero_spatial"
        ? { ...restored, mode: "libero", steps: 280 } : restored);
    } catch {
      setPreferences(initial);
    }
    setLoadedProjectId(projectId);
  }, [projectId]);
  useEffect(() => {
    if (ready)
      try {
        localStorage.setItem(
          `firebird.workflow.${projectId}`,
          JSON.stringify(preferences),
        );
      } catch {
        /* Settings still work for this session. */
      }
  }, [projectId, preferences, ready]);
  const policyJobs = (jobs.data ?? [])
    .filter((x) => x.kind.startsWith("policy."))
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  const stageJobs = stage === "settings" ? policyJobs : policyJobs.filter(job =>
    stage === "Quantize" ? ["policy.quantize", "policy.workflow"].includes(job.kind)
      : stage === "Evaluate" ? job.kind === "policy.evaluate"
        : stage === "Fine-tune" ? job.kind === "policy.finetune" : job.kind === "policy.run");
  const selected =
    stageJobs.find((x) => x.id === selectedJobId) ?? stageJobs[0];
  const events = useQuery({
    queryKey: ["job-events", selected?.id],
    queryFn: () => api.events(selected!.id),
    enabled: !!selected,
    refetchInterval: selected && isActive(selected) ? 1000 : false,
  });
  const data = resultOf(selected);
  const progress = observedProgress(events.data ?? []);
  const datasets = (jobs.data ?? []).filter(
    (x) => x.kind === "dataset.inspect" && x.status === "succeeded",
  );
  const inputs = (artifacts.data ?? []).filter((x) =>
    stage === "Quantize"
      ? ["training_checkpoint", "native_checkpoint"].includes(x.format) ||
        (x.format === "gguf" && x.metadata?.precision === "float")
      : stage === "settings"
        ? x.format === "gguf"
        : ["gguf", "deployment_package"].includes(x.format),
  );
  const checkpoints = sortCheckpoints((artifacts.data ?? []).filter(
    (x) => x.format === "training_checkpoint" || x.format === "native_checkpoint",
  ), policyJobs);
  const selectedInput = input === "latest" ? (stage === "Quantize" ? checkpoints[0]?.id : inputs[0]?.id) ?? "" : input;
  const inputArtifact = inputs.find(item => item.id === selectedInput);
  const inputIssue = stage === "Quantize" ? quantizationIssue(inputArtifact) : null;
  const cloudCheckpoint = !!inputArtifact && isCloudArtifact(inputArtifact);
  const needsNativeExecution = stage === "Evaluate" || stage === "Run" || stage === "settings";
  const runtimes = (options.data?.runtimes ?? []).filter(item => item.enabled !== false && !item.export_only &&
    (!needsNativeExecution || item.execution !== "skypilot") &&
    (!needsNativeExecution || stage === "settings" || preferences.mode !== "libero" || item.simulation));
  const runtime = runtimes.find(item => item.id === runtimeId) ??
    (cloudCheckpoint ? runtimes.find(item => item.provider === "gcp" || item.execution === "skypilot") : undefined) ?? runtimes[0];
  const defaultLanguage = options.data?.quantization_defaults[runtime?.device ?? "cpu"].language ?? "Q8_0";
  const nativeQuantization = stage === "Quantize" && runtime?.execution !== "skypilot";
  const language =
    preferences.precision === "recommended"
      ? defaultLanguage
      : preferences.precision;
  const mutation = useMutation({
    mutationFn: async () => {
      if (!ready || !runtime)
        throw new Error("Select a project and a configured execution target.");
      if (inputIssue) throw new Error(inputIssue);
      const evaluation: PolicyRequest["evaluation"] = stage === "Quantize" && !nativeQuantization ? undefined : {
        mode: preferences.mode,
        suite: preferences.suite,
        ...(preferences.suite === "libero_spatial"
          ? { task_ids: states(preferences.taskIds) }
          : {}),
        repetitions: preferences.repetitions,
        warmups: preferences.warmup,
        task_id: preferences.task,
        initial_states: states(preferences.searchStates),
        final_states: states(preferences.finalStates),
        steps: preferences.steps,
        seed: 42,
      };
      if (evaluation && preferences.suite === "libero_spatial") {
        if (!preferences.parityProfile.trim() || !preferences.parityRmse.trim() || !preferences.parityMaxError.trim())
          throw new Error("Set your approved floating-policy parity profile and tolerances in Settings & diagnostics.");
        const rmse = Number(preferences.parityRmse);
        const maximum = Number(preferences.parityMaxError);
        if (!Number.isFinite(rmse) || !Number.isFinite(maximum) || rmse < 0 || maximum < 0)
          throw new Error("Parity tolerances must be finite, non-negative numbers.");
        evaluation.parity_limits = {
          profile: preferences.parityProfile.trim(), max_rmse: rmse, max_abs_error: maximum,
        };
      }
      const body: PolicyRequest = {
        operation:
          stage === "Fine-tune"
            ? "policy.finetune"
            : stage === "Quantize"
              ? nativeQuantization ? "policy.workflow" : "policy.quantize"
              : stage === "Evaluate" || stage === "settings"
                ? "policy.evaluate"
                : "policy.run",
        runtime_id: runtime.id,
        evaluation,
        training_method: method,
        timeout_seconds: 7200,
        precision: { language, vision: preferences.vision ? "Q8_0" : null },
      };
      if (stage === "Fine-tune") {
        body.dataset_job_id = datasetId || datasets[0]?.id;
        if (!body.dataset_job_id)
          throw new Error("Inspect a compatible dataset first.");
        body.training_method = method;
        body.training = {
          steps: preferences.trainingSteps,
          warmup_steps: Math.min(50, preferences.trainingSteps - 1),
          camera_key: preferences.camera,
        };
        if (resumeId) {
          const checkpoint = checkpoints.find((x) => x.id === resumeId);
          const prior = policyJobs.find(
            (x) => x.id === (checkpoint?.job_id ?? resumeId.slice(4)),
          );
          if (!prior || !("training_method" in prior.request))
            throw new Error("The original training recipe is unavailable.");
          body.training_method = prior.request.training_method;
          body.dataset_job_id = prior.request.dataset_job_id;
          body.training = null;
          if (resumeId.startsWith("job:"))
            body.resume_job_id = resumeId.slice(4);
          else body.artifact_id = resumeId;
        }
      } else {
        if (!selectedInput) throw new Error("Choose an input policy.");
        if (selectedInput.startsWith("source:")) body.source_id = selectedInput.slice(7);
        else body.artifact_id = selectedInput;
        if (nativeQuantization) {
          body.candidates = [{ language, vision: preferences.vision ? "Q8_0" : null }];
          if (preferences.compareQ4) body.candidates.push({ language: language === "Q4_0" ? "Q8_0" : "Q4_0", vision: null });
          if (preferences.select && preferences.mode === "libero") body.limits = {
            min_success_rate: preferences.minSuccess / 100, max_success_drop: 0,
            max_p95_ms: preferences.maxLatency, max_peak_device_mib: preferences.maxMemory,
          };
        }
      }
      return api.policyJob(projectId, body);
    },
    onSuccess: (job) => {
      setSelectedJobId(job.id);
      client.setQueryData<Job[]>(["jobs", projectId], previous => [job, ...(previous ?? []).filter(item => item.id !== job.id)]);
      void client.invalidateQueries({ queryKey: ["jobs", projectId] });
    },
  });
  const cancel = useMutation({
    mutationFn: () => api.cancel(selected!.id),
    onSuccess: () =>
      client.invalidateQueries({ queryKey: ["jobs", projectId] }),
  });
  const submittedJob = mutation.data ? policyJobs.find(job => job.id === mutation.data.id) ?? mutation.data : undefined;
  const starting = mutation.isPending || !!submittedJob && isActive(submittedJob);
  function update<K extends keyof Preferences>(key: K, value: Preferences[K]) {
    if (ready) setPreferences((old) => ({ ...old, [key]: value }));
  }
  const errors = [
    options.error,
    jobs.error,
    artifacts.error,
    mutation.error,
    cancel.error,
  ].filter(Boolean);
  const preferencesBlocker = !projectId
    ? "Select or create a project to edit workflow settings."
    : !ready ? "Loading workflow settings…" : null;
  const diagnosticBlocker = options.isPending
    ? "Checking execution targets…"
    : options.isError
      ? "Execution targets could not be loaded. Check the application connection."
      : !projectId
        ? "Select or create a project to save your diagnostic results."
        : !ready
          ? "Loading workflow settings…"
          : !runtime
          ? "No execution target is configured. Set up a native worker on the application host first."
          : artifacts.isPending || jobs.isPending
            ? "Loading project policies and runs…"
            : artifacts.isError || jobs.isError
              ? "Project policies or runs could not be loaded. Check the application connection."
              : policyJobs.some(isActive)
                ? "Wait for the active policy job to finish, or cancel it below."
                : !inputs.length
                  ? "This project has no GGUF policy yet. Create one in Quantize, then return here."
                  : !inputs.some((artifact) => artifact.id === selectedInput)
                    ? "Choose a project policy to evaluate."
                    : preferences.mode === "libero" && !runtime.simulation
                      ? "This target has no LIBERO simulator. Choose engine checks or a simulator-enabled target."
                      : null;
  if (stage === "settings")
    return (
      <section className={`panel ${tab === "compute" ? "compute-settings-panel" : "workflow-panel"}`}>
        <div className="section-tabs settings-tabs">
          <button className={`section-tab ${tab === "compute" ? "active" : ""}`} onClick={() => setTab("compute")}>Compute</button>
          <button
            className={`section-tab ${tab === "settings" ? "active" : ""}`}
            onClick={() => setTab("settings")}
          >
            Workflow settings
          </button>
          <button
            className={`section-tab ${tab === "diagnostics" ? "active" : ""}`}
            onClick={() => setTab("diagnostics")}
          >
            Diagnostics
          </button>
        </div>
        {tab === "settings" && preferencesBlocker && <p className="warning-box" role="status">{preferencesBlocker}</p>}
        {tab === "compute" ? <CloudConnectionsPanel /> : tab === "settings" ? (
          <fieldset className="workflow-fields workflow-controls" disabled={!ready}>
            <legend className="visually-hidden">Project workflow settings</legend>
            <h2>Compression defaults</h2>
            <p className="muted">Start with LM Q8 on CPU and CUDA, preserving vision precision. Q4 is experimental: a prior RTX 3070 pilot lost task success. Every policy still needs evaluation on its execution target.</p>
            <label>
              Quantization recipe
              <select
                value={preferences.precision}
                onChange={(e) =>
                  update(
                    "precision",
                    e.target.value as Preferences["precision"],
                  )
                }
              >
                <option value="recommended">
                  Start with LM Q8; validate on your target
                </option>
                <option value="Q4_0">LM Q4 (experimental)</option>
                <option value="Q8_0">LM Q8</option>
              </select>
            </label>
            <label className="workflow-check">
              <input
                type="checkbox"
                checked={preferences.compareQ4}
                onChange={(e) => update("compareQ4", e.target.checked)}
              />
              Compare Q8 and Q4 (experimental)
            </label>
            <p className="muted">Candidate comparisons use a configured native evaluation worker. Cloud quantization produces the precision selected for that job.</p>
            <label className="workflow-check">
              <input
                type="checkbox"
                checked={preferences.vision}
                onChange={(e) => update("vision", e.target.checked)}
              />
              Also quantize vision to Q8 (experimental)
            </label>
            <h2>Evaluation</h2>
            <label>
              Task suite
              <select
                value={preferences.suite}
                onChange={(e) => {
                  const suite = e.target.value as Preferences["suite"];
                  if (ready) setPreferences((old) => ({ ...old, suite,
                    mode: suite === "libero_spatial" ? "libero" : old.mode,
                    steps: suite === "libero_spatial" ? 280 : 500 }));
                }}
              >
                <option value="libero_object">LIBERO Object (legacy policy)</option>
                <option value="libero_spatial">LIBERO Spatial (pinned SmolVLA policy)</option>
              </select>
            </label>
            {preferences.suite === "libero_spatial" && (
              <>
                <label>
                  Spatial task IDs
                  <input value={preferences.taskIds} onChange={(e) => update("taskIds", e.target.value)} />
                </label>
                <p className="muted">Use unique task IDs 0–9. Every task uses the same paired search and final states. Hardware acceptance is still required.</p>
                <label>
                  Approved parity profile
                  <input value={preferences.parityProfile} onChange={(e) => update("parityProfile", e.target.value)} placeholder="Name your reviewed tolerance profile" />
                </label>
                <label>
                  Maximum action RMSE
                  <input type="number" min="0" step="any" value={preferences.parityRmse} onChange={(e) => update("parityRmse", e.target.value)} />
                </label>
                <label>
                  Maximum absolute action error
                  <input type="number" min="0" step="any" value={preferences.parityMaxError} onChange={(e) => update("parityMaxError", e.target.value)} />
                </label>
                <p className="muted">Declare reviewed tolerances before running. Native and floating C++ actions must pass this check before compression starts; no universal tolerance is assumed.</p>
              </>
            )}
            <label>
              Protocol
              <select
                value={preferences.mode}
                disabled={preferences.suite === "libero_spatial"}
                onChange={(e) =>
                  update("mode", e.target.value as Preferences["mode"])
                }
              >
                <option value="engine">Engine diagnostics</option>
                <option value="libero">Paired LIBERO episodes</option>
              </select>
            </label>
            {preferences.mode === "engine" && <p className="muted">Measures latency and loading; task success requires LIBERO.</p>}
            <label>
              Timed predictions
              <input
                type="number"
                min="2"
                max="1000"
                value={preferences.repetitions}
                onChange={(e) => update("repetitions", Number(e.target.value))}
              />
            </label>
            <label>
              Warmup predictions
              <input
                type="number"
                min="1"
                max="100"
                value={preferences.warmup}
                onChange={(e) => update("warmup", Number(e.target.value))}
              />
            </label>
            {preferences.mode === "libero" && (
              <>
                {preferences.suite === "libero_object" && (
                  <label>
                    LIBERO object task
                    <input type="number" min="0" max="9" value={preferences.task}
                      onChange={(e) => update("task", Number(e.target.value))} />
                  </label>
                )}
                <label>
                  Search initial states
                  <input
                    value={preferences.searchStates}
                    onChange={(e) => update("searchStates", e.target.value)}
                  />
                </label>
                <label>
                  Unused final initial states
                  <input
                    value={preferences.finalStates}
                    onChange={(e) => update("finalStates", e.target.value)}
                  />
                </label>
                <label>
                  Episode step limit
                  <input
                    type="number"
                    min="1"
                    max="500"
                    value={preferences.steps}
                    disabled={preferences.suite === "libero_spatial"}
                    onChange={(e) => update("steps", Number(e.target.value))}
                  />
                </label>
                <label className="workflow-check">
                  <input
                    type="checkbox"
                    checked={preferences.select}
                    onChange={(e) => update("select", e.target.checked)}
                  />
                  Select the smallest candidate that meets my limits
                </label>
                {preferences.select && (
                  <>
                    <label>
                      Minimum task success (%)
                      <input
                        type="number"
                        min="0"
                        max="100"
                        value={preferences.minSuccess}
                        onChange={(e) =>
                          update("minSuccess", Number(e.target.value))
                        }
                      />
                    </label>
                    <label>
                      Maximum p95 latency (ms)
                      <input
                        type="number"
                        min="1"
                        value={preferences.maxLatency}
                        onChange={(e) =>
                          update("maxLatency", Number(e.target.value))
                        }
                      />
                    </label>
                    <label>
                      Maximum measured target memory (MiB)
                      <input
                        type="number"
                        min="1"
                        value={preferences.maxMemory}
                        onChange={(e) =>
                          update("maxMemory", Number(e.target.value))
                        }
                      />
                    </label>
                    <p className="muted">Candidates must match reference success and pass unused states. GPU memory includes other processes.</p>
                  </>
                )}
              </>
            )}
            <h2>Fine-tuning</h2>
            <label>
              Optimizer steps
              <input
                type="number"
                min="2"
                value={preferences.trainingSteps}
                onChange={(e) =>
                  update("trainingSteps", Number(e.target.value))
                }
              />
            </label>
          </fieldset>
        ) : (
          <>
            <h2>Run diagnostics</h2>
            <p className="muted">Check a policy on your configured target. Results are saved to this project.</p>
            <form className="workflow-fields diagnostic-launcher" onSubmit={(event) => {
              event.preventDefault();
              if (!diagnosticBlocker && !mutation.isPending) mutation.mutate();
            }}>
              <label>Diagnostic execution target
                <select value={runtime?.id ?? ""} disabled={!ready} onChange={(event) => setRuntimeId(event.target.value)}>
                  <option value="" disabled>Select a target</option>
                  {runtimes.map((target) => <option key={target.id} value={target.id}>{target.label}</option>)}
                </select>
              </label>
              <label>Diagnostic policy
                <select value={input} disabled={!ready} onChange={(event) => setInput(event.target.value)}>
                  <option value="">Choose a project policy</option>
                  {inputs.map((artifact) => <option key={artifact.id} value={artifact.id}>{artifact.label} · {artifact.id.slice(0, 8)}</option>)}
                </select>
              </label>
              <label>Diagnostic mode
                <select value={preferences.mode} disabled={!ready || preferences.suite === "libero_spatial"} onChange={(event) => update("mode", event.target.value as Preferences["mode"])}>
                  <option value="engine">Engine checks</option>
                  <option value="libero" disabled={!runtime?.simulation}>LIBERO task evaluation</option>
                </select>
              </label>
              <p className="muted">{preferences.mode === "engine"
                ? `Checks loading, finite actions, prediction latency and memory with ${preferences.warmup} warmups and ${preferences.repetitions} repetitions. Task success is not measured.`
                : preferences.suite === "libero_spatial"
                  ? `Evaluates LIBERO Spatial tasks ${preferences.taskIds} with initial states ${preferences.searchStates}, seed 42 and the full 280-step horizon. The policy must include pinned Spatial assets and approved parity limits.`
                  : `Evaluates LIBERO Object task ${preferences.task} with initial states ${preferences.searchStates}, seed 42 and up to ${preferences.steps} steps. The policy must declare LIBERO Object compatibility.`}</p>
              <button className="text-link" type="button" onClick={() => setTab("settings")}>Edit diagnostic settings</button>
              {diagnosticBlocker && <p className="warning-box" id="diagnostic-readiness" role="status">{diagnosticBlocker}</p>}
              {!runtime && !options.isPending && <p className="muted"><a className="text-link" href="https://github.com/sobhanb-eth/firebird-hackathon-codebase/blob/main/docs/policy-workflow.md#configure-the-execution-host" target="_blank" rel="noreferrer">Worker setup instructions ↗</a></p>}
              {!!runtime && !inputs.length && !artifacts.isPending && <button className="secondary-button" type="button" onClick={onOpenQuantize}>Open Quantize</button>}
              <button className="primary-button" type="submit" disabled={!!diagnosticBlocker || mutation.isPending} aria-describedby={diagnosticBlocker ? "diagnostic-readiness" : undefined}>
                {mutation.isPending ? "Starting…" : "Start diagnostics"}
              </button>
            </form>
            {errors.map((error, index) => <p className="error-notice" role="alert" key={index}>{error?.message}</p>)}
            <h2 className="diagnostic-runs-heading">Diagnostic runs</h2>
            {policyJobs.length ? (
              <>
                <label>
                  Run
                  <select
                    value={selected?.id ?? ""}
                    onChange={(e) => setSelectedJobId(e.target.value)}
                  >
                    {policyJobs.map((job) => (
                      <option key={job.id} value={job.id}>
                        {runLabel(job)} · {job.status} ·{" "}
                        {new Date(job.created_at).toLocaleString()}
                      </option>
                    ))}
                  </select>
                </label>
                {selected && <div className="workflow-run-summary" role="status">
                  <strong>{runSummary(selected)}</strong>
                  {progress.step !== null && <span>Last reported step: {progress.step.toLocaleString()}</span>}
                  {progress.checkpoint !== null && <span>Latest checkpoint: step {progress.checkpoint.toLocaleString()}</span>}
                </div>}
                {selected?.error && <p className="error-notice" role="alert">{conciseRunError(selected.error)}</p>}
                {selected && <p className={`status status-${selected.status}`}>{selected.status}</p>}
                {selected && isActive(selected) && <button className="secondary-button" disabled={cancel.isPending} onClick={() => cancel.mutate()}>Cancel run</button>}
                {!!data?.reports?.some(
                  (report) => typeof report.p95_ms === "number",
                ) && (
                  <div className="table-scroll">
                    <table className="feature-table workflow-metrics">
                      <caption>Recorded policy measurements</caption>
                      <thead>
                        <tr>
                          <th>Stage</th>
                          <th>p50 (ms)</th>
                          <th>p95 (ms)</th>
                          <th>Peak target memory (MiB)</th>
                          <th>Task success</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.reports
                          .filter((report) => typeof report.p95_ms === "number")
                          .map((report, index) => (
                            <tr key={index}>
                              <th>
                                {String(report.stage).replaceAll("-", " ")}
                              </th>
                              <td>
                                {typeof report.p50_ms === "number"
                                  ? report.p50_ms.toFixed(1)
                                  : "—"}
                              </td>
                              <td>
                                {typeof report.p95_ms === "number"
                                  ? report.p95_ms.toFixed(1)
                                  : "—"}
                              </td>
                              <td>
                                {typeof report.peak_device_mib === "number"
                                  ? report.peak_device_mib.toFixed(1)
                                  : "—"}
                              </td>
                              <td>
                                {typeof report.success_rate === "number"
                                  ? `${(report.success_rate * 100).toFixed(0)}% · ${report.complete_episodes} episodes`
                                  : "—"}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {data?.reports?.map((report, index) => (
                  <details className="provenance" key={index}>
                    <summary>
                      {String(report.stage)} ·{" "}
                      {String(report.scope ?? report.operation)}
                    </summary>
                    <pre className="workflow-json">
                      {JSON.stringify(report, null, 2)}
                    </pre>
                  </details>
                ))}
                <details className="provenance">
                  <summary>Recorded recipe and artifact lineage</summary>
                  <pre className="workflow-json">
                    {JSON.stringify(
                      {
                        request: selected?.request,
                        artifacts: data?.artifacts,
                      },
                      null,
                      2,
                    )}
                  </pre>
                </details>
                <details className="provenance workflow-activity-details" key={selected?.id} onToggle={event => setDetailsJobId(event.currentTarget.open ? selected?.id ?? null : null)}>
                  <summary>{selected && runLabel(selected) === "Training" ? "Training details" : "Run details"}<span>Recorded activity and errors</span></summary>
                  {detailsJobId === selected?.id && <>
                  {selected?.error && <pre className="workflow-json">{selected.error}</pre>}
                  <ol className="workflow-events" aria-label="Detailed run activity">
                    {events.data?.map(event => <li key={event.sequence}>
                      <time>{new Date(event.timestamp).toLocaleString()}</time>
                      <span>{event.message}</span>
                    </li>)}
                  </ol>
                  {events.error && <p className="error-notice" role="alert">{events.error.message}</p>}
                  </>}
                </details>
              </>
            ) : (
              <p>No policy runs in this project yet.</p>
            )}
            <BenchmarkReference />
          </>
        )}
      </section>
    );
  return (
    <div className="content-grid">
      <section className="panel workflow-panel">
        <h2>
          {stage === "Fine-tune"
            ? "Fine-tune a policy"
            : stage === "Quantize"
              ? "Quantize policy"
              : stage === "Evaluate"
                ? "Evaluate policy"
                : "Run policy"}
        </h2>
        {options.data && !runtimes.length && (
          needsNativeExecution ? <div className="workflow-storage-note" role="status">
            <p>{preferences.mode === "libero" ? "No native simulation worker is configured." : stage === "Evaluate" ? "No native evaluation worker is configured." : "No native policy runner is configured."} Cloud GPUs currently support training and quantization.</p>
            <p>Training and validation loss remain available in Fine-tune.</p>
            {onViewTraining && <button type="button" className="secondary-button" onClick={onViewTraining}>View training metrics</button>}
          </div> : <p className="warning-box">Connect an execution worker to start a run.</p>
        )}
        {preferencesBlocker && <p className="warning-box" role="status">{preferencesBlocker}</p>}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (ready && !mutation.isPending) mutation.mutate();
          }}
        >
          <fieldset className="workflow-fields workflow-controls" disabled={!ready}>
            <legend className="visually-hidden">Project policy job</legend>
          <label>
            Execution target
            <select
              aria-label="Execution target"
              disabled={!runtimes.length}
              value={runtime?.id ?? ""}
              onChange={(e) => setRuntimeId(e.target.value)}
            >
              <option value="" disabled>
                Select a target
              </option>
              {runtimes.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.label}
                </option>
              ))}
            </select>
          </label>
          {stage === "Fine-tune" ? (
            <>
              <label>
                Fine-tuning method
                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                >
                  {options.data?.training_methods.map((x) => (
                    <option key={x.id} value={x.id}>
                      {x.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Inspected dataset
                <select
                  value={datasetId || datasets[0]?.id || ""}
                  onChange={(e) => setDatasetId(e.target.value)}
                >
                  <option value="" disabled>
                    Select an inspection
                  </option>
                  {datasets.map((x) => (
                    <option key={x.id} value={x.id}>
                      {"repo_id" in x.request ? x.request.repo_id : x.id} ·{" "}
                      {new Date(x.created_at).toLocaleDateString()}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Resume checkpoint
                <select
                  value={resumeId}
                  onChange={(e) => setResumeId(e.target.value)}
                >
                  <option value="">Start a new training run</option>
                  {checkpoints.map((x) => (
                    <option key={x.id} value={x.id}>
                      {x.label}
                    </option>
                  ))}
                  {policyJobs
                    .filter(
                      (x) =>
                        x.kind === "policy.finetune" &&
                        ["failed", "interrupted", "cancelled"].includes(
                          x.status,
                        ),
                    )
                    .map((x) => (
                      <option key={x.id} value={`job:${x.id}`}>
                        Last saved checkpoint · {x.id.slice(0, 8)}
                      </option>
                    ))}
                </select>
              </label>
            </>
          ) : (
            <>
              <label>
                {stage === "Quantize" ? "Checkpoint or policy" : "Input policy"}
                <select
                  aria-label={stage === "Quantize" ? "Checkpoint or policy" : "Input policy"}
                  value={input}
                  onChange={(e) => { setInput(e.target.value); mutation.reset(); }}
                  required
                >
                  <option value="">Choose a policy</option>
                  {stage === "Quantize" && checkpoints.length > 0 && <option value="latest">Latest checkpoint · {checkpointLabel(checkpoints[0], policyJobs, options.data?.training_models)}</option>}
                  {stage !== "Quantize" && inputs.length > 0 && <option value="latest">Latest policy · {inputs[0].label}</option>}
                  {stage === "Quantize" && checkpoints.length > 0 && <optgroup label="Trained checkpoints">{checkpoints.map(item => <option key={item.id} value={item.id}>{checkpointLabel(item, policyJobs, options.data?.training_models)}</option>)}</optgroup>}
                  {stage === "Quantize" && !!options.data?.sources.length && <optgroup label="Base policies">{options.data.sources.map((x) => <option key={x.id} value={`source:${x.id}`}>{x.label}</option>)}</optgroup>}
                  {inputs.filter(item => stage !== "Quantize" || !["training_checkpoint", "native_checkpoint"].includes(item.format)).map((x) => (
                    <option key={x.id} value={x.id}>
                      {x.label} · {x.id.slice(0, 8)}
                    </option>
                  ))}
                </select>
              </label>
              {stage === "Quantize" && <>
                <p className="workflow-input-help">Choose the latest saved checkpoint or any earlier step. Quantization creates a separate compressed policy.</p>
                {cloudCheckpoint && <p className="workflow-storage-note">Stored on Google Cloud. The cloud worker reads this checkpoint directly; weights stay off your computer.</p>}
                {inputIssue && <p className="workflow-storage-note" role="status">{inputIssue} You can still download this checkpoint from its training run.</p>}
                {!checkpoints.length && !artifacts.isPending && <p className="muted">Trained checkpoints will appear here after they are saved. You can also start from a configured base policy.</p>}
                <label>Quantization precision<select aria-label="Quantization precision" value={preferences.precision} onChange={event => update("precision", event.target.value as Preferences["precision"])}>
                  <option value="recommended">Recommended · 8-bit (Q8)</option>
                  <option value="Q4_0">4-bit (Q4) · experimental</option>
                  <option value="Q8_0">8-bit (Q8) · higher precision</option>
                </select></label>
                <p className="workflow-input-help">Compresses the language model. Vision stays at its original precision{preferences.vision ? " unless enabled below" : ""}.</p>
                <details className="workflow-advanced"><summary>Advanced quantization</summary><label className="workflow-check"><input type="checkbox" checked={preferences.vision} onChange={event => update("vision", event.target.checked)} />Also quantize vision to Q8 (experimental)</label></details>
              </>}
            </>
          )}
          <button
            className="primary-button"
            type="submit"
            disabled={
              !ready ||
              !runtime ||
              starting ||
              (stage !== "Fine-tune" && !selectedInput) ||
              !!inputIssue ||
              (stage === "Fine-tune" && !runtime.training)
            }
          >
            {mutation.isPending
              ? "Starting…"
              : starting ? stage === "Quantize" ? "Quantizing…" : "Running…"
              : stage === "Fine-tune"
                ? "Start fine-tuning"
                : stage === "Quantize"
                  ? nativeQuantization ? "Run quantization workflow" : "Start quantization"
                  : stage === "Evaluate"
                    ? "Start evaluation"
                    : "Reload and run"}
          </button>
          </fieldset>
        </form>
        {errors.map((error, index) => (
          <p className="error-notice" role="alert" key={index}>
            {error?.message}
          </p>
        ))}
      </section>
      <section className="panel workflow-panel">
        <h2>Activity</h2>
        {selected ? (
          <>
            <label>
              Run
              <select
                value={selected.id}
                onChange={(e) => setSelectedJobId(e.target.value)}
              >
                {stageJobs.map((job) => (
                  <option key={job.id} value={job.id}>
                    {runLabel(job)} · {job.status} · {new Date(job.created_at).toLocaleString()}
                  </option>
                ))}
              </select>
            </label>
            <p className={`status status-${selected.status}`}>
              {selected.status}
            </p>
            <p role="status">{runSummary(selected)}</p>
            {isActive(selected) && <progress className="workflow-progress" aria-label="Run in progress" />}
            {!!events.data?.length && <ol className="workflow-live-events" aria-label="Recent run activity">{events.data.slice(-5).map(event => <li key={event.sequence}><time>{new Date(event.timestamp).toLocaleTimeString()}</time><span>{event.message}</span></li>)}</ol>}
            {events.error && <p className="error-notice" role="alert">Activity is unavailable: {events.error.message}</p>}
            {isActive(selected) && (
              <button
                className="secondary-button"
                disabled={cancel.isPending}
                onClick={() => cancel.mutate()}
              >
                Cancel run
              </button>
            )}
            {selected.error && (
              <p className="error-notice" role="alert">
                {selected.error}
              </p>
            )}
            {data && (
              <>
                <p>
                  {data.decision === "validated"
                    ? "A candidate passed the requested checks and fresh reload."
                    : data.decision === "no_feasible_candidate"
                      ? "No candidate passed all acceptance checks."
                      : data.decision === "diagnostics_only"
                        ? "Diagnostics complete. Task-quality approval is still required."
                        : selected.kind === "policy.quantize" ? "Quantization complete. Your compressed policy is ready to download." : "Operation completed."}
                </p>
                <ul>
                  {data.artifacts?.map((x) => (
                    <li key={x.id}>
                      {x.label}{x.file_bytes > 0 ? ` · ${(x.file_bytes / 1024 / 1024).toLocaleString(undefined, { maximumFractionDigits: 1 })} MiB` : ""} ·{" "}
                      <a
                        className="text-link"
                        href={artifactDownloadUrl(projectId, x.id)}
                      >
                        Download {x.label}
                      </a>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </>
        ) : (
          <p>{stage === "Quantize" ? "Start quantization to follow conversion, compression and verification here." : "No runs yet."}</p>
        )}
      </section>
    </div>
  );
}
