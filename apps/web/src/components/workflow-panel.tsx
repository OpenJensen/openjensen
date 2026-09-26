"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
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
};
const initial: Preferences = {
  mode: "engine",
  repetitions: 10,
  steps: 500,
  warmup: 3,
  task: 0,
  searchStates: "0,1",
  finalStates: "2,3",
  trainingSteps: 1000,
  camera: "observation.images.front",
  select: false,
  minSuccess: 100,
  maxLatency: 1000,
  maxMemory: 8192,
  precision: "recommended",
  vision: false,
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
}: {
  projectId: string;
  stage: string;
}) {
  const client = useQueryClient();
  const [preferences, setPreferences] = useState(initial);
  const [loaded, setLoaded] = useState(false);
  const [runtimeId, setRuntimeId] = useState("");
  const [input, setInput] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [method, setMethod] = useState("lora");
  const [resumeId, setResumeId] = useState("");
  const [tab, setTab] = useState<"settings" | "diagnostics">("settings");
  const [selectedJobId, setSelectedJobId] = useState("");
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
    setLoaded(false);
    try {
      setPreferences({
        ...initial,
        ...JSON.parse(
          localStorage.getItem(`firebird.workflow.${projectId}`) ?? "{}",
        ),
      });
    } catch {
      setPreferences(initial);
    }
    setLoaded(true);
  }, [projectId]);
  useEffect(() => {
    if (loaded)
      try {
        localStorage.setItem(
          `firebird.workflow.${projectId}`,
          JSON.stringify(preferences),
        );
      } catch {
        /* Settings still work for this session. */
      }
  }, [projectId, preferences, loaded]);
  const runtime =
    options.data?.runtimes.find((x) => x.id === runtimeId) ??
    options.data?.runtimes[0];
  const policyJobs = (jobs.data ?? [])
    .filter((x) => x.kind !== "dataset.inspect")
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selected =
    policyJobs.find((x) => x.id === selectedJobId) ?? policyJobs[0];
  const events = useQuery({
    queryKey: ["job-events", selected?.id],
    queryFn: () => api.events(selected!.id),
    enabled: !!selected,
    refetchInterval: selected && isActive(selected) ? 1000 : false,
  });
  const data = resultOf(selected);
  const datasets = (jobs.data ?? []).filter(
    (x) => x.kind === "dataset.inspect" && x.status === "succeeded",
  );
  const inputs = (artifacts.data ?? []).filter((x) =>
    stage === "Quantize"
      ? ["training_checkpoint", "native_checkpoint"].includes(x.format) ||
        (x.format === "gguf" && x.metadata?.precision === "float")
      : ["gguf", "deployment_package"].includes(x.format),
  );
  const checkpoints = (artifacts.data ?? []).filter(
    (x) => x.format === "training_checkpoint",
  );
  const defaultLanguage = runtime?.device === "cuda" ? "Q4_0" : "Q8_0";
  const language =
    preferences.precision === "recommended"
      ? defaultLanguage
      : preferences.precision;
  const mutation = useMutation({
    mutationFn: async () => {
      if (!projectId || !runtime)
        throw new Error("Select a project and a configured execution target.");
      const evaluation: NonNullable<PolicyRequest["evaluation"]> = {
        mode: preferences.mode,
        repetitions: preferences.repetitions,
        warmups: preferences.warmup,
        task_id: preferences.task,
        initial_states: states(preferences.searchStates),
        final_states: states(preferences.finalStates),
        steps: preferences.steps,
        seed: 42,
      };
      const body: PolicyRequest = {
        operation:
          stage === "Fine-tune"
            ? "policy.finetune"
            : stage === "Quantize"
              ? "policy.workflow"
              : stage === "Evaluate"
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
        if (!input) throw new Error("Choose an input policy.");
        if (input.startsWith("source:")) body.source_id = input.slice(7);
        else body.artifact_id = input;
        if (stage === "Quantize") {
          body.candidates = [
            { language, vision: preferences.vision ? "Q8_0" : null },
            { language: language === "Q4_0" ? "Q8_0" : "Q4_0", vision: null },
          ];
          if (preferences.select && preferences.mode === "libero")
            body.limits = {
              min_success_rate: preferences.minSuccess / 100,
              max_success_drop: 0,
              max_p95_ms: preferences.maxLatency,
              max_peak_device_mib: preferences.maxMemory,
            };
        }
      }
      return api.policyJob(projectId, body);
    },
    onSuccess: (job) => {
      setSelectedJobId(job.id);
      void client.invalidateQueries({ queryKey: ["jobs", projectId] });
    },
  });
  const cancel = useMutation({
    mutationFn: () => api.cancel(selected!.id),
    onSuccess: () =>
      client.invalidateQueries({ queryKey: ["jobs", projectId] }),
  });
  function update<K extends keyof Preferences>(key: K, value: Preferences[K]) {
    setPreferences((old) => ({ ...old, [key]: value }));
  }
  const errors = [
    options.error,
    jobs.error,
    artifacts.error,
    mutation.error,
    cancel.error,
  ].filter(Boolean);
  if (stage === "settings")
    return (
      <section className="panel workflow-panel">
        <div className="section-tabs">
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
        {tab === "settings" ? (
          <div className="workflow-fields">
            <h2>Compression defaults</h2>
            <p className="muted">
              Recommended uses LM Q4 on CUDA and LM Q8 on CPU, preserving vision
              precision. These are starting recipes from small pilots; each
              policy still needs evaluation.
            </p>
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
                  Recommended for the execution target
                </option>
                <option value="Q4_0">LM Q4</option>
                <option value="Q8_0">LM Q8</option>
              </select>
            </label>
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
              Protocol
              <select
                value={preferences.mode}
                onChange={(e) =>
                  update("mode", e.target.value as Preferences["mode"])
                }
              >
                <option value="engine">Engine diagnostics</option>
                <option value="libero">Paired LIBERO episodes</option>
              </select>
            </label>
            <p className="muted">
              Engine diagnostics check loading, finite actions and timing.
              Task-quality selection requires compatible LIBERO episodes and
              your acceptance limits.
            </p>
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
                <label>
                  LIBERO object task
                  <input
                    type="number"
                    min="0"
                    max="9"
                    value={preferences.task}
                    onChange={(e) => update("task", Number(e.target.value))}
                  />
                </label>
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
                    <p className="muted">
                      The candidate must also match the floating reference’s
                      success rate and pass the unused final states. GPU
                      measurements include other device users.
                    </p>
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
            <label>
              Dataset camera feature
              <input
                value={preferences.camera}
                onChange={(e) => update("camera", e.target.value)}
              />
            </label>
            <p className="muted">
              Settings are saved for this project on this browser. Each
              submitted job preserves its exact recipe.
            </p>
          </div>
        ) : (
          <>
            <h2>Run diagnostics</h2>
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
                        {job.kind} · {job.status} ·{" "}
                        {new Date(job.created_at).toLocaleString()}
                      </option>
                    ))}
                  </select>
                </label>
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
                                  : "Unavailable"}
                              </td>
                              <td>
                                {typeof report.success_rate === "number"
                                  ? `${(report.success_rate * 100).toFixed(0)}% · ${report.complete_episodes} episodes`
                                  : "Not measured"}
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
                <ol className="workflow-events">
                  {events.data?.map((event) => (
                    <li key={event.sequence}>
                      {event.stage}: {event.message}
                    </li>
                  ))}
                </ol>
                {selected?.error && (
                  <p className="error-notice" role="alert">
                    {selected.error}
                  </p>
                )}
              </>
            ) : (
              <p>No policy runs in this project yet.</p>
            )}
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
              ? "Create a smaller policy"
              : stage === "Evaluate"
                ? "Evaluate a policy"
                : "Reload and run a policy"}
        </h2>
        {!options.data?.runtimes.length && (
          <p className="warning-box">
            No native execution target is configured on this application host.
            Configure a worker environment to enable policy jobs.
          </p>
        )}
        <form
          className="workflow-fields"
          onSubmit={(e) => {
            e.preventDefault();
            mutation.mutate();
          }}
        >
          <label>
            Execution target
            <select
              value={runtime?.id ?? ""}
              onChange={(e) => setRuntimeId(e.target.value)}
            >
              <option value="" disabled>
                Select a target
              </option>
              {options.data?.runtimes.map((x) => (
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
              <p className="muted">
                {
                  options.data?.training_methods.find((x) => x.id === method)
                    ?.description
                }
              </p>
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
              {resumeId && (
                <p className="muted">
                  Resume preserves the original method, dataset and training
                  recipe.
                </p>
              )}
            </>
          ) : (
            <>
              <label>
                Input policy
                <select
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  required
                >
                  <option value="">Choose a policy</option>
                  {stage === "Quantize" &&
                    options.data?.sources.map((x) => (
                      <option key={x.id} value={`source:${x.id}`}>
                        {x.label}
                      </option>
                    ))}
                  {inputs.map((x) => (
                    <option key={x.id} value={x.id}>
                      {x.label} · {x.id.slice(0, 8)}
                    </option>
                  ))}
                </select>
              </label>
              {stage === "Quantize" && (
                <p className="form-note">
                  {preferences.precision === "recommended"
                    ? "Recommended"
                    : "Custom"}{" "}
                  compression is selected. The workflow checks a floating
                  reference, creates candidates and evaluates their actual
                  outputs.
                </p>
              )}
              {stage === "Run" && (
                <p className="form-note">
                  A fresh process checks the exact policy again. With LIBERO
                  enabled in settings, it also runs the configured simulation
                  episodes.
                </p>
              )}
            </>
          )}
          <button
            className="primary-button"
            type="submit"
            disabled={
              !projectId ||
              !runtime ||
              mutation.isPending ||
              (stage === "Fine-tune" && !runtime.training)
            }
          >
            {mutation.isPending
              ? "Starting…"
              : stage === "Fine-tune"
                ? "Start fine-tuning"
                : stage === "Quantize"
                  ? "Run quantization workflow"
                  : stage === "Evaluate"
                    ? "Start evaluation"
                    : "Reload and run"}
          </button>
          <p className="muted">
            Advanced controls and benchmark metrics are in Settings &
            diagnostics.
          </p>
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
                {policyJobs.map((job) => (
                  <option key={job.id} value={job.id}>
                    {job.kind} · {job.status}
                  </option>
                ))}
              </select>
            </label>
            <p className={`status status-${selected.status}`}>
              {selected.status}
            </p>
            <p>{selected.stage?.replaceAll("-", " ") ?? "Waiting to start"}</p>
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
                        : "Operation completed."}
                </p>
                <ul>
                  {data.artifacts?.map((x) => (
                    <li key={x.id}>
                      {x.label} ·{" "}
                      <a
                        className="text-link"
                        href={artifactDownloadUrl(projectId, x.id)}
                      >
                        Download
                      </a>
                    </li>
                  ))}
                </ul>
              </>
            )}
            <p className="muted">
              Detailed measurements and evidence are available in Diagnostics.
            </p>
          </>
        ) : (
          <p>Your policy runs will appear here.</p>
        )}
      </section>
    </div>
  );
}
