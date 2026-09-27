"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  isActive,
  isDatasetJob,
  type DatasetJob,
  type DatasetProfile,
  type Job,
  type PolicyRequest,
} from "@/lib/api";
import { datasetStarters } from "@/lib/dataset-starters";
import { trainingModels, type TrainingModel } from "@/lib/training-models";
import { checkpointStep, trainingRunModelLabel } from "@/lib/checkpoints";
import { CameraPlayer } from "@/components/dataset-explorer";
import { Icon } from "@/components/icon";
import { GpuPicker } from "@/components/gpu-picker";
import { TrainingMonitor } from "@/components/training-monitor";
import { JobHistory, type JobHistoryEntry } from "@/components/job-history";
import "./training-panel.css";

const steps = ["Dataset", "Model", "Compute"];
type Recipe = {
  trainingSteps: number;
  batchSize: number;
  checkpointCount: number;
  checkpointIntervalOverride: number | null;
  seed: number;
  learningRate: number;
  gradientAccumulation: number;
  validationFraction: number;
  evalEvery: number;
};
// KiteML SmolVLA form defaults, verified in its signed-in UI on 2026-09-26.
// Its automatic save cadence targets about five checkpoints across the run.
const defaults: Recipe = {
  trainingSteps: 20000,
  batchSize: 64,
  checkpointCount: 5,
  checkpointIntervalOverride: null,
  seed: 42,
  learningRate: 0.0001,
  gradientAccumulation: 1,
  validationFraction: 0.2,
  evalEvery: 100,
};
type InspectedDataset = DatasetJob & { result: DatasetProfile };

function positiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value > 0;
}

function savedRecipe(saved: Record<string, unknown>): Recipe {
  const version = typeof saved.trainingDefaultsVersion === "number" ? saved.trainingDefaultsVersion : 0;
  const trainingSteps = positiveInteger(saved.trainingSteps) && !(version < 2 && saved.trainingSteps === 1000)
    ? saved.trainingSteps : defaults.trainingSteps;
  const batchSize = positiveInteger(saved.batchSize) && !(version < 2 && saved.batchSize === 1)
    ? saved.batchSize : defaults.batchSize;
  // The old five-step value shipped as a default. Other interval choices remain
  // exact until the user edits the new checkpoint-count control.
  const interval = version < 3
    ? positiveInteger(saved.checkpointEvery) && saved.checkpointEvery !== 5 ? saved.checkpointEvery : null
    : positiveInteger(saved.checkpointIntervalOverride) ? saved.checkpointIntervalOverride : null;
  const checkpointCount = interval ? Math.ceil(trainingSteps / interval)
    : version >= 3 && positiveInteger(saved.checkpointCount) ? saved.checkpointCount : defaults.checkpointCount;
  return {
    trainingSteps, batchSize, checkpointCount, checkpointIntervalOverride: interval,
    seed: typeof saved.seed === "number" && Number.isInteger(saved.seed) && saved.seed >= 0 ? saved.seed : defaults.seed,
    learningRate: typeof saved.learningRate === "number" && Number.isFinite(saved.learningRate) && saved.learningRate > 0 ? saved.learningRate : defaults.learningRate,
    gradientAccumulation: positiveInteger(saved.gradientAccumulation) ? saved.gradientAccumulation : defaults.gradientAccumulation,
    validationFraction: typeof saved.validationFraction === "number" && saved.validationFraction > 0 && saved.validationFraction < 1 ? saved.validationFraction : defaults.validationFraction,
    evalEvery: positiveInteger(saved.evalEvery) ? saved.evalEvery : defaults.evalEvery,
  };
}

function cameras(profile?: DatasetProfile) {
  return Object.entries(profile?.features ?? {})
    .filter(
      ([, feature]) =>
        feature &&
        typeof feature === "object" &&
        "dtype" in feature &&
        ["image", "video"].includes(String(feature.dtype)),
    )
    .map(([key]) => key);
}
function datasetIssue(profile: DatasetProfile) {
  if (profile.source === "local" && !profile.snapshot)
    return "Prepare an immutable training copy when importing this local dataset.";
  if (profile.source === "huggingface" && !/^[0-9a-f]{40}$/i.test(profile.revision))
    return "Inspect an immutable dataset revision before training.";
  if (profile.total_episodes < 2)
    return "At least two episodes are needed for training and validation.";
  if (!cameras(profile).length) return "No image or video camera was found.";
  for (const key of ["observation.state" in profile.features ? "observation.state" : "states", "action"]) {
    const feature = profile.features[key];
    const shape =
      feature && typeof feature === "object" && "shape" in feature
        ? feature.shape
        : undefined;
    if (
      !Array.isArray(shape) ||
      shape.length !== 1 ||
      typeof shape[0] !== "number" ||
      !Number.isInteger(shape[0]) ||
      shape[0] < 1
    )
      return `Training needs a non-empty ${key} vector.`;
  }
  return null;
}
function number(value: number) {
  return value.toLocaleString();
}
// Mirrors the reviewed adapters in native_profiles.py and psi_profile.py. The
// server remains authoritative; camera/dimension limits are model-specific.
const dimensionLimits: Record<string, number> = {
  smolvla: 32, psi0: 36, eo1: 32, evo1: 24, gr00t_n17: 132,
  pi0: 32, pi05: 32, pi0_fast: 32, wall_x: 20, xvla: 20,
};
function modelDatasetIssue(model: TrainingModel | undefined, profile: DatasetProfile | undefined, cameraKeys: string[]) {
  if (!model || !profile) return null;
  if (profile.source === "local" && model.backend !== "lerobot")
    return "Local snapshots currently support native LeRobot models, including ACT.";
  if (model.id === "psi0" && profile.format !== "lerobot_v2")
    return "Psi-Zero currently needs a LeRobot v2 dataset. Choose a v2 inspection or another model.";
  if (model.backend === "lerobot" && profile.format !== "lerobot_v3")
    return `${model.label} currently needs a LeRobot v3 dataset. Choose a v3 inspection or another model.`;
  if (model.required_cameras && cameraKeys.length !== model.required_cameras)
    return `${model.label} requires ${model.required_cameras} selected camera${model.required_cameras === 1 ? "" : "s"}. Change the camera selection in Dataset.`;
  if (model.id === "evo1" && cameraKeys.length > 3)
    return "EVO-1 supports at most three selected cameras. Change the camera selection in Dataset.";
  if (model.id !== "psi0" && !("observation.state" in profile.features))
    return `${model.label} requires an observation.state feature.`;
  const maximum = dimensionLimits[model.id];
  if (maximum) {
    const state = "observation.state" in profile.features ? "observation.state" : "states";
    for (const key of [state, "action"]) {
      const feature = profile.features[key] as { shape?: number[] } | undefined;
      if ((feature?.shape?.[0] ?? 0) > maximum)
        return `${model.label} supports at most ${maximum} ${key} dimensions.`;
    }
  }
  return null;
}
function modelStatusLabel(model: TrainingModel, supported: boolean): string | null {
  switch (model.status) {
    case "connect_account": return "Connect Google Cloud";
    case "setup_required": return "Setup required";
    case "coming_soon": return "Coming soon";
    default: return supported ? null : "Compute unavailable";
  }
}

export function TrainingPanel({
  projectId,
  preferredDatasetId,
  onChooseDataset,
  onDiagnostics,
  onComputeSettings,
  onQuantize,
  startNew,
  showJobsRequest,
  preferredRunId,
  active = true,
}: {
  projectId: string;
  preferredDatasetId?: string;
  onChooseDataset: () => void;
  onDiagnostics: () => void;
  onComputeSettings: () => void;
  onQuantize?: (artifactId: string) => void;
  startNew?: { id: number; datasetId?: string };
  showJobsRequest?: number;
  preferredRunId?: string;
  active?: boolean;
}) {
  const client = useQueryClient();
  const [view, setView] = useState<"jobs" | "new" | "run">(startNew ? "new" : "jobs");
  const [step, setStep] = useState(0);
  const [datasetId, setDatasetId] = useState(preferredDatasetId ?? "");
  const [cameraSelections, setCameraSelections] = useState<
    Record<string, string[]>
  >({});
  const [previewEpisode, setPreviewEpisode] = useState<{
    datasetId: string;
    index: number;
  } | null>(null);
  const [modelId, setModelId] = useState("");
  const [modelBatchSizes, setModelBatchSizes] = useState<Record<string, number>>({});
  const [runtimeId, setRuntimeId] = useState("");
  const [method, setMethod] = useState("");
  const [resumeId, setResumeId] = useState("");
  const [recipe, setRecipe] = useState(defaults);
  const [loaded, setLoaded] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState("");
  const headingRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (!active) return;
    window.scrollTo({ top: 0, behavior: "instant" });
    headingRef.current?.focus({ preventScroll: true });
  }, [step, active, view]);
  const options = useQuery({
    queryKey: ["policy-options"],
    queryFn: api.policyOptions,
    enabled: active,
    refetchInterval: active ? 15000 : false,
  });
  const jobs = useQuery({
    queryKey: ["jobs", projectId],
    queryFn: () => api.jobs(projectId),
    enabled: active && !!projectId,
    refetchInterval: (query) =>
      active ? (query.state.data?.some(isActive) ? 1000 : 5000) : false,
  });
  const artifacts = useQuery({
    queryKey: ["artifacts", projectId],
    queryFn: () => api.artifacts(projectId),
    enabled: active && !!projectId,
    refetchInterval: active ? 5000 : false,
  });
  useEffect(() => {
    if (!projectId) { setLoaded(false); return; }
    try {
      const saved = JSON.parse(
        localStorage.getItem(`firebird.workflow.${projectId}`) ?? "{}",
      );
      const restored = savedRecipe(saved);
      setRecipe(restored);
      setModelId(typeof saved.trainingModelId === "string" ? saved.trainingModelId : "");
      const batches = saved.modelBatchSizes && typeof saved.modelBatchSizes === "object"
        ? Object.fromEntries(Object.entries(saved.modelBatchSizes).filter((entry): entry is [string, number] => positiveInteger(entry[1]))) : {};
      setModelBatchSizes({ smolvla: restored.batchSize, ...batches });
    } catch {
      setRecipe(defaults);
    }
    setLoaded(true);
  }, [projectId]);
  useEffect(() => {
    if (!loaded || !projectId) return;
    try {
      const saved = JSON.parse(
        localStorage.getItem(`firebird.workflow.${projectId}`) ?? "{}",
      );
      localStorage.setItem(
        `firebird.workflow.${projectId}`,
        JSON.stringify({ ...saved, ...recipe, modelBatchSizes, trainingModelId: modelId, checkpointEvery: undefined, trainingDefaultsVersion: 3 }),
      );
    } catch {
      /* This session remains usable without browser storage. */
    }
  }, [projectId, recipe, modelId, modelBatchSizes, loaded]);
  function update<K extends keyof Recipe>(key: K, value: Recipe[K]) {
    setRecipe((old) => ({ ...old, [key]: value }));
    if (key === "batchSize" && typeof value === "number") setModelBatchSizes(previous => ({ ...previous, [modelId || "smolvla"]: value }));
  }

  const trainingRuns = (jobs.data ?? [])
    .filter((job) => job.kind === "policy.finetune")
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  // Completed artifacts have reached the recipe's final step and cannot resume.
  const resumableRuns = trainingRuns.filter((job) =>
    ["failed", "interrupted", "cancelled"].includes(job.status),
  );
  const checkpoints = (artifacts.data ?? []).filter(
    (item) =>
      item.format === "training_checkpoint" &&
      resumableRuns.some((job) => job.id === item.job_id),
  );
  const resumeOptions = [
    ...checkpoints.map((item) => ({
      id: item.id,
      label: item.label,
      jobId: item.job_id,
    })),
    ...resumableRuns
      .filter((job) => !checkpoints.some((item) => item.job_id === job.id))
      .map((job) => ({
        id: `job:${job.id}`,
        label: `Last saved checkpoint · ${job.id.slice(0, 8)}`,
        jobId: job.id,
      })),
  ];
  const resume = resumeOptions.find((item) => item.id === resumeId);
  const prior = trainingRuns.find((job) => job.id === resume?.jobId);
  const priorRequest =
    prior?.request && "training_method" in prior.request
      ? prior.request
      : undefined;
  const allInspections = (jobs.data ?? [])
    .filter(isDatasetJob)
    .filter(
      (job): job is InspectedDataset =>
        job.status === "succeeded" && !!job.result,
    )
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  // Keep the selected inspection's exact ID, while avoiding repeated cards for the same snapshot.
  const unique = new Map<string, InspectedDataset>();
  for (const job of allInspections) {
    const key = `${job.result.source}:${job.result.repo_id ?? job.id}:${job.result.revision}`;
    if (
      !unique.has(key) ||
      job.id === (resumeId ? priorRequest?.dataset_job_id : datasetId)
    )
      unique.set(key, job);
  }
  const datasets = [...unique.values()];
  const dataset =
    datasets.find((job) => job.id === datasetId) ??
    datasets.find((job) => !datasetIssue(job.result));
  const cameraOptions = cameras(dataset?.result);
  const selectedCameras =
    dataset && cameraSelections[dataset.id] !== undefined
      ? cameraOptions.filter((key) =>
          cameraSelections[dataset.id].includes(key),
        )
      : cameraOptions;
  function toggleCamera(key: string) {
    if (!dataset) return;
    setCameraSelections((previous) => ({
      ...previous,
      [dataset.id]: selectedCameras.includes(key)
        ? selectedCameras.filter((value) => value !== key)
        : cameraOptions.filter(
            (value) => value === key || selectedCameras.includes(value),
          ),
    }));
  }
  const mergedModels = new Map(trainingModels.map((item) => [item.id, item]));
  for (const item of options.data?.training_models ?? [])
    mergedModels.set(item.id, {
      ...mergedModels.get(item.id),
      ...item,
      description: mergedModels.get(item.id)?.description ?? item.description,
    } as TrainingModel);
  const models = [...mergedModels.values()];
  const runtimes = options.data?.runtimes ?? [];
  const cloudGpuMemory: Record<string, number> = { L4: 24, T4: 16, A100: 40 };
  const runtimeMemory = (item: (typeof runtimes)[number]) =>
    item.gpu_memory_mib ? item.gpu_memory_mib / 1024 : cloudGpuMemory[item.accelerator ?? ""] ?? 0;
  const hasAdapter = (runtime: (typeof runtimes)[number], item: TrainingModel) =>
    !!item.model_revision && runtime.training && runtime.device === "cuda" &&
    (!item.minimum_gpu_memory_gb || runtimeMemory(runtime) >= item.minimum_gpu_memory_gb) &&
    (runtime.training_model_ids ?? ["smolvla"]).includes(item.id) &&
    // The bundled on-demand SmolVLA adapter remains usable before preflight.
    // Older API catalogs describe that state with an empty runtime_ids list.
    (item.id === "smolvla" || !item.runtime_ids || item.runtime_ids.includes(runtime.id));
  const modelSupported = (item: TrainingModel) =>
    (item.id === "smolvla" && !!item.model_revision) || runtimes.some(runtime => hasAdapter(runtime, item));
  let originalTraining = priorRequest?.training;
  let ancestorRequest = priorRequest;
  const visitedRuns = new Set<string>();
  while (!originalTraining && ancestorRequest) {
    const linkedJobId =
      ancestorRequest.resume_job_id ??
      checkpoints.find((item) => item.id === ancestorRequest?.artifact_id)
        ?.job_id;
    if (!linkedJobId || visitedRuns.has(linkedJobId)) break;
    visitedRuns.add(linkedJobId);
    const linkedRequest = trainingRuns.find(
      (job) => job.id === linkedJobId,
    )?.request;
    ancestorRequest =
      linkedRequest && "training_method" in linkedRequest
        ? linkedRequest
        : undefined;
    originalTraining = ancestorRequest?.training;
  }
  const originalModel = models.find(
    (item) => item.model_id === originalTraining?.model_id,
  );
  const model = resumeId
    ? (originalModel ??
      (!originalTraining?.model_id
        ? models.find((item) => item.id === "smolvla")
        : undefined))
    : (models.find((item) => item.id === modelId) ?? models[0]);
  const availableMethods = (options.data?.training_methods ?? []).filter(
    (item) => model?.methods.includes(item.id),
  );
  const trainingMethod =
    availableMethods.find((item) => item.id === method)?.id ??
    availableMethods.find(
      (item) => item.id === options.data?.default_training_method,
    )?.id ??
    availableMethods[0]?.id;
  const supportsModel = (item: (typeof runtimes)[number]) =>
    item.enabled !== false &&
    !!model && hasAdapter(item, model);
  const gpuChoices = ["L4", "T4", "A100"].filter(gpu => !model?.minimum_gpu_memory_gb ||
    (runtimes.find(item => item.execution === "skypilot" && item.accelerator === gpu)?.gpu_memory_mib
      ? runtimeMemory(runtimes.find(item => item.execution === "skypilot" && item.accelerator === gpu)!)
      : cloudGpuMemory[gpu]) >= model.minimum_gpu_memory_gb);
  const defaultGpu = options.data?.compute?.gcp?.default_gpu ?? "A100";
  const localRuntimes = options.data?.compute?.local.enabled
    ? runtimes.filter((item) => (item.provider ?? "local") === "local" && item.training && item.device === "cuda" && item.enabled !== false &&
      (!model?.minimum_gpu_memory_gb || runtimeMemory(item) >= model.minimum_gpu_memory_gb))
    : [];
  const selectedGpu = gpuChoices.includes(runtimeId) || localRuntimes.some(item => item.id === runtimeId) ? runtimeId
    : gpuChoices.includes(defaultGpu) ? defaultGpu : gpuChoices[0] ?? localRuntimes[0]?.id ?? "";
  const chosenRuntime = gpuChoices.includes(selectedGpu)
    ? runtimes.find((item) => item.execution === "skypilot" && item.accelerator === selectedGpu)
    : localRuntimes.find((item) => item.id === selectedGpu);
  const runtime = chosenRuntime && supportsModel(chosenRuntime) ? chosenRuntime : undefined;
  const needsAccount = !chosenRuntime;
  const priorDataset = allInspections.find(
    (job) => job.id === priorRequest?.dataset_job_id,
  );
  const activeDataset = resumeId ? priorDataset : dataset;
  const activeMethod = resumeId
    ? priorRequest?.training_method
    : trainingMethod;
  const displayedModelId = model?.id;
  const stepComplete = [
    !!activeDataset && !datasetIssue(activeDataset.result),
    !!model && !!activeMethod,
    !!runtime,
  ];
  const activeCameraKeys = resumeId
    ? Array.isArray(originalTraining?.camera_keys)
      ? originalTraining.camera_keys.filter(
          (key): key is string => typeof key === "string",
        )
      : [String(originalTraining?.camera_key ?? "observation.images.front")]
    : selectedCameras;
  const modelIssue = !resumeId ? modelDatasetIssue(model, activeDataset?.result, activeCameraKeys) : null;
  const episodes = useQuery({
    queryKey: ["training-episodes", activeDataset?.id],
    queryFn: () => api.episodes(activeDataset!.id, 0, 6),
    enabled:
      active && view === "new" && step === 0 &&
      !!activeDataset &&
      activeDataset.result.source === "huggingface",
    retry: false,
    staleTime: 60000,
  });
  const episodeIndex =
    previewEpisode?.datasetId === activeDataset?.id
      ? previewEpisode?.index
      : episodes.data?.episodes[0]?.episode_index;
  const preview = useQuery({
    queryKey: ["training-preview", activeDataset?.id, episodeIndex],
    queryFn: () => api.episode(activeDataset!.id, episodeIndex!),
    enabled: active && view === "new" && step === 0 && !!activeDataset && episodeIndex !== undefined,
    retry: false,
    staleTime: 60000,
  });
  const checkpointCount = recipe.checkpointIntervalOverride
    ? Math.ceil(recipe.trainingSteps / recipe.checkpointIntervalOverride) : recipe.checkpointCount;
  const checkpointInterval = recipe.checkpointIntervalOverride ?? Math.ceil(recipe.trainingSteps / recipe.checkpointCount);
  const recipeValid =
    Number.isInteger(recipe.trainingSteps) &&
    recipe.trainingSteps >= 2 &&
    Number.isInteger(recipe.batchSize) &&
    recipe.batchSize >= 1 &&
    positiveInteger(recipe.checkpointCount) &&
    positiveInteger(checkpointInterval) &&
    Number.isInteger(recipe.seed) && recipe.seed >= 0 && recipe.seed <= 2147483647 &&
    Number.isFinite(recipe.learningRate) && recipe.learningRate > 0 &&
    positiveInteger(recipe.gradientAccumulation) &&
    Number.isFinite(recipe.validationFraction) && recipe.validationFraction > 0 && recipe.validationFraction < 1 &&
    positiveInteger(recipe.evalEvery);
  const blocked = !projectId
    ? "Select a project."
    : !loaded || options.isPending || jobs.isPending
      ? "Loading…"
      : options.isError || jobs.isError
        ? "Couldn’t load training options."
        : resumeId && !priorRequest
          ? "The original training recipe is unavailable."
          : !activeDataset
            ? "Select a dataset."
            : !resumeId && datasetIssue(activeDataset.result)
              ? datasetIssue(activeDataset.result)
              : !resumeId && !selectedCameras.length
                ? "Select at least one camera."
                : !model || !activeMethod
                  ? "Select a model and method."
                  : modelIssue
                    ? modelIssue
                  : !runtime
                    ? "Choose an available GPU."
                    : !resumeId && !recipeValid
                      ? "Check training settings."
                      : null;
  const mutation = useMutation({
    mutationFn: async () => {
      if (blocked || !runtime || !activeDataset || !model || !activeMethod)
        throw new Error(blocked ?? "Complete the training setup first.");
      const body: PolicyRequest = {
        operation: "policy.finetune",
        runtime_id: runtime.id,
        dataset_job_id: activeDataset.id,
        training_method: activeMethod,
        timeout_seconds: 86400,
      };
      if (resumeId) {
        body.training = null;
        if (resumeId.startsWith("job:")) body.resume_job_id = resumeId.slice(4);
        else body.artifact_id = resumeId;
      } else {
        body.training = {
          model_id: model.model_id,
          model_revision: model.model_revision,
          ...(model.checkpoint_subdirectory
            ? { checkpoint_subdirectory: model.checkpoint_subdirectory }
            : {}),
          camera_key: selectedCameras[0],
          camera_keys: selectedCameras,
          steps: recipe.trainingSteps,
          warmup_steps: Math.min(50, recipe.trainingSteps - 1),
          batch_size: recipe.batchSize,
          learning_rate: recipe.learningRate,
          gradient_accumulation_steps: recipe.gradientAccumulation,
          seed: recipe.seed,
          validation_fraction: recipe.validationFraction,
          eval_every: recipe.evalEvery,
          save_every: checkpointInterval,
        };
      }
      return api.policyJob(projectId, body);
    },
    onSuccess: (job) => {
      setSelectedRunId(job.id);
      setView("run");
      client.setQueryData<Job[]>(["jobs", projectId], (previous) => [
        job,
        ...(previous ?? []).filter((item) => item.id !== job.id),
      ]);
      void client.invalidateQueries({ queryKey: ["jobs", projectId] });
    },
  });
  const selectedRun =
    trainingRuns.find((job) => job.id === selectedRunId) ??
    (mutation.data?.id === selectedRunId ? mutation.data : undefined);
  const cancel = useMutation({
    mutationFn: (jobId: string) => api.cancel(jobId),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["jobs", projectId] });
    },
  });
  const submittedRun = mutation.data
    ? trainingRuns.find((job) => job.id === mutation.data.id) ?? mutation.data
    : undefined;
  const runStarting = !!submittedRun && isActive(submittedRun);
  const busy = mutation.isPending || runStarting;
  const startLabel = mutation.isPending ? "Starting…"
    : runStarting ? (submittedRun?.stage === "preparing" || submittedRun?.status === "queued" ? "Preparing GPU…" : "Training…")
    : resumeId ? "Resume fine-tuning" : "Start fine-tuning";
  const summaryModel = resumeId
    ? String(model?.label ?? originalTraining?.model_id ?? "Original model")
    : (model?.label ?? "Choose a model");

  function openNew(dataset?: string) {
    if (dataset) setDatasetId(dataset);
    setResumeId("");
    mutation.reset();
    setView("new");
  }
  useEffect(() => {
    if (startNew) {
      setStep(0);
      openNew(startNew.datasetId ?? preferredDatasetId);
    }
    // An entry action is identified by its id; ordinary dataset refreshes must
    // not pull someone out of an existing job or reset a creation draft.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [startNew?.id]);
  useEffect(() => {
    if (!startNew) setView("jobs");
    // Sidebar navigation changes the view while preserving the creation draft.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showJobsRequest]);

  useEffect(() => {
    if (preferredRunId) { setSelectedRunId(preferredRunId); setView("run"); }
  }, [preferredRunId]);

  const historyEntries: JobHistoryEntry[] = trainingRuns.map(job => {
    const request = "training_method" in job.request ? job.request : undefined;
    const recordedModel = trainingRunModelLabel(job, models, artifacts.data ?? [], trainingRuns);
    const dataset = allInspections.find(item => item.id === request?.dataset_job_id);
    const targetSteps = positiveInteger(request?.training?.steps) ? request.training.steps : null;
    const savedSteps = (artifacts.data ?? []).filter(item => item.job_id === job.id)
      .map(checkpointStep).filter((value): value is number => value !== null);
    const reportedSteps = job.result && "reports" in job.result
      ? (job.result.reports ?? []).map(report => report.steps).filter(positiveInteger) : [];
    const completedSteps = reportedSteps.length ? Math.max(...reportedSteps)
      : job.status === "succeeded" ? targetSteps : null;
    const progress = completedSteps !== null ? `${number(completedSteps)} steps completed`
      : savedSteps.length ? `Saved step ${number(Math.max(...savedSteps))}${targetSteps ? ` / ${number(targetSteps)}` : ""}`
        : targetSteps ? `${number(targetSteps)} steps planned` : undefined;
    return {
      id: job.id,
      title: [recordedModel ?? "Fine-tuning", request?.training_method.toUpperCase()].filter(Boolean).join(" · "),
      subtitle: dataset?.result.repo_id ?? (dataset ? "Local dataset" : "Dataset not recorded"),
      status: job.status, createdAt: job.created_at, progress,
    };
  });

  return (
    <>
      {view === "jobs" && <JobHistory title="Fine-tuning jobs" newLabel="Start a new fine-tuning"
        entries={historyEntries} onNew={() => openNew()}
        onSelect={id => { setSelectedRunId(id); setView("run"); }}
        loading={!!projectId && jobs.isPending} error={jobs.error}
        emptyMessage="No fine-tuning jobs yet." disabled={!projectId || mutation.isPending} />}
      {view === "new" && <>
      <div className="training-view-toolbar">
        <button type="button" className="text-button training-back" onClick={() => setView("jobs")}><Icon name="arrow" size={14} />Back to jobs</button>
        <h2>New fine-tuning</h2>
      </div>
      <form
        className="training-workspace"
        onSubmit={(event) => {
          event.preventDefault();
          if (step === 2 && !blocked && !busy) mutation.mutate();
        }}
      >
        <nav className="training-nav" aria-label="Training setup">
          {steps.map((label, index) => (
            <button
              type="button"
              key={label}
              aria-label={label}
              aria-current={step === index ? "step" : undefined}
              onClick={() => setStep(index)}
            >
              <span>
                {index < step && stepComplete[index] ? (
                  <Icon name="check" size={13} />
                ) : (
                  index + 1
                )}
              </span>
              {label}
            </button>
          ))}
        </nav>

        {step === 0 && (
          <section aria-labelledby="training-data-title">
            <div className="training-heading">
              <h2 id="training-data-title" ref={headingRef} tabIndex={-1}>
                Dataset
              </h2>
              <button
                type="button"
                className="text-button"
                onClick={onChooseDataset}
              >
                <Icon name="plus" size={14} /> Import
              </button>
            </div>
            {jobs.isPending && projectId ? (
              <p role="status">Loading datasets…</p>
            ) : datasets.length ? (
              <fieldset className="training-datasets">
                <legend className="visually-hidden">Inspected dataset</legend>
                {datasets.map((job) => {
                  const profile = job.result;
                  const starter = datasetStarters.find(
                    (item) =>
                      item.repoId === profile.repo_id &&
                      item.revision === profile.revision,
                  );
                  const issue = datasetIssue(profile);
                  return (
                    <label
                      className="training-dataset"
                      key={job.id}
                      title={issue ?? profile.repo_id ?? "Local dataset"}
                    >
                      <input
                        type="radio"
                        name="training-dataset"
                        aria-label={profile.repo_id ?? "Local dataset"}
                        checked={activeDataset?.id === job.id}
                        disabled={!!issue || !!resumeId || busy}
                        onChange={() => {
                          setDatasetId(job.id);
                          mutation.reset();
                        }}
                      />
                      {starter ? (
                        <img src={starter.poster} alt="" />
                      ) : (
                        <span className="training-dataset-icon">
                          <Icon name="database" size={18} />
                        </span>
                      )}
                      <span>
                        <strong>
                          {starter?.title ??
                            profile.repo_id?.split("/").at(-1) ??
                            "Local dataset"}
                        </strong>
                        <small>
                          {issue ??
                            `${profile.total_episodes} episodes · ${cameras(profile).length} ${cameras(profile).length === 1 ? "camera" : "cameras"}`}
                        </small>
                      </span>
                    </label>
                  );
                })}
              </fieldset>
            ) : (
              <div className="training-empty">
                <p>No inspected datasets.</p>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={onChooseDataset}
                >
                  Import dataset <Icon name="arrow" size={15} />
                </button>
              </div>
            )}
            {activeDataset && (
              <>
                <div className="training-data-facts">
                  <span>
                    {number(activeDataset.result.total_episodes)} episodes
                  </span>
                  <span>
                    {number(activeDataset.result.total_frames)} frames
                  </span>
                  <span>{activeDataset.result.fps} fps</span>
                  <span>
                    {cameras(activeDataset.result).length}{" "}
                    {cameras(activeDataset.result).length === 1
                      ? "camera"
                      : "cameras"}
                  </span>
                  <span>{activeDataset.result.robot_type}</span>
                </div>
                <div className="training-heading training-camera-heading">
                  <h3>Cameras</h3>
                  {episodes.data && episodes.data.episodes.length > 1 && (
                    <label className="training-episode-picker">
                      <span className="visually-hidden">Preview episode</span>
                      <select
                        value={episodeIndex ?? ""}
                        onChange={(event) =>
                          setPreviewEpisode({
                            datasetId: activeDataset.id,
                            index: Number(event.target.value),
                          })
                        }
                      >
                        {episodes.data.episodes.map((item) => (
                          <option
                            key={item.episode_index}
                            value={item.episode_index}
                          >
                            Episode {item.episode_index}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                </div>
                {preview.data?.cameras.length ? (
                  <CameraPlayer
                    key={`${activeDataset.id}-${episodeIndex}`}
                    preview={preview.data}
                    active={step === 0}
                    selection={{
                      keys: activeCameraKeys,
                      onToggle: toggleCamera,
                      disabled: !!resumeId || busy,
                    }}
                  />
                ) : (
                  <fieldset className="training-camera-fallback">
                    <legend className="visually-hidden">
                      Training cameras
                    </legend>
                    {cameras(activeDataset.result).map((key) => (
                      <label key={key}>
                        <span className="training-camera-placeholder">
                          <Icon name="play" size={22} />
                          <small>
                            {episodes.isFetching || preview.isFetching
                              ? "Loading preview…"
                              : "Preview unavailable"}
                          </small>
                        </span>
                        <span>
                          <input
                            type="checkbox"
                            aria-label={key}
                            checked={activeCameraKeys.includes(key)}
                            disabled={!!resumeId || busy}
                            onChange={() => toggleCamera(key)}
                          />
                          {key.replace(/^observation\.images\./, "")}
                        </span>
                      </label>
                    ))}
                  </fieldset>
                )}
                {!!preview.data?.cameras.length &&
                  cameras(activeDataset.result)
                    .filter(
                      (key) =>
                        !preview.data.cameras.some((item) => item.key === key),
                    )
                    .map((key) => (
                      <label key={key} className="training-extra-camera">
                        <input
                          type="checkbox"
                          aria-label={key}
                          checked={activeCameraKeys.includes(key)}
                          disabled={!!resumeId || busy}
                          onChange={() => toggleCamera(key)}
                        />
                        {key.replace(/^observation\.images\./, "")}
                        <small>Preview unavailable</small>
                      </label>
                    ))}
                {(episodes.isError || preview.isError) && (
                  <button
                    type="button"
                    className="text-button"
                    onClick={() => {
                      void episodes.refetch();
                      if (episodeIndex !== undefined) void preview.refetch();
                    }}
                  >
                    Retry preview
                  </button>
                )}
              </>
            )}
            <div className="training-footer">
              <span>
                {activeDataset && !activeCameraKeys.length
                  ? "Select at least one camera."
                  : ""}
              </span>
              <button
                type="button"
                className="primary-button"
                disabled={
                  !activeDataset ||
                  !activeCameraKeys.length ||
                  (!resumeId && !!datasetIssue(activeDataset.result))
                }
                onClick={() => setStep(1)}
              >
                Next <Icon name="arrow" size={15} />
              </button>
            </div>
          </section>
        )}

        {step === 1 && (
          <section aria-labelledby="training-model-title">
            <div className="training-heading">
              <h2 id="training-model-title" ref={headingRef} tabIndex={-1}>
                Model
              </h2>
            </div>
            <fieldset className="training-model-grid">
              <legend className="visually-hidden">Base model</legend>
              {models.map((item) => {
                const statusLabel = modelStatusLabel(item, modelSupported(item));
                const memory = item.minimum_gpu_memory_gb ?? item.suggested_gpu_memory_gb;
                return (
                <label
                  key={item.id}
                  className={`training-model-tile model-${item.id}`}
                  title={item.model_id}
                >
                  <input
                    type="radio"
                    name="training-model"
                    aria-label={item.label}
                    checked={displayedModelId === item.id}
                    disabled={!!resumeId || busy || !modelSupported(item)}
                    onChange={() => {
                      setModelBatchSizes(previous => ({ ...previous, [displayedModelId ?? "smolvla"]: recipe.batchSize }));
                      setRecipe(previous => ({ ...previous, batchSize: modelBatchSizes[item.id] ?? (item.id === "smolvla" ? defaults.batchSize : item.id === "psi0" ? 2 : 4) }));
                      setModelId(item.id);
                      setRuntimeId("");
                      mutation.reset();
                    }}
                  />
                  <span className="training-model-mark">
                    {(
                      {
                        smolvla: "SMOL",
                        openvla_oft: "OFT",
                        openvla: "VLA",
                        pi0: "π₀",
                        pi05: "π₀.₅",
                        gr00t_n17: "GR00T",
                      } as Record<string, string>
                    )[item.id] ?? item.label.slice(0, 4)}
                  </span>
                  <span className="training-model-copy">
                    <span>
                      <strong>{item.label}</strong>
                      {statusLabel && (
                        <small className="training-model-status">
                          {statusLabel}
                        </small>
                      )}
                    </span>
                    <small title={item.description}>{item.description}</small>
                    <span className="training-model-memory">
                      {typeof memory === "number" && Number.isFinite(memory) && memory > 0
                        ? <>GPU budget: <strong>{number(memory)} GB+</strong></>
                        : <strong>GPU budget not verified</strong>}
                    </span>
                  </span>
                </label>
                );
              })}
            </fieldset>
            <p className="training-memory-help">GPU memory budget for training; usage varies with method and batch size.</p>
            <fieldset className="training-methods">
              <legend>Method</legend>
              {availableMethods.map((item) => (
                <label key={item.id}>
                  <input
                    type="radio"
                    name="training-method"
                    aria-label={item.label}
                    checked={activeMethod === item.id}
                    disabled={!!resumeId || busy}
                    onChange={() => setMethod(item.id)}
                  />
                  {item.label}
                </label>
              ))}
            </fieldset>
            {model?.id === "psi0" && <p className="training-control-note">Psi-Zero trains its action expert while keeping the vision-language backbone frozen. It currently supports one camera and LeRobot v2 datasets.</p>}
            {modelIssue && <p className="error-notice" role="alert">{modelIssue}</p>}
            <div className="training-footer">
              <button
                type="button"
                className="text-button"
                onClick={() => setStep(0)}
              >
                Back
              </button>
              <button
                type="button"
                className="primary-button"
                disabled={!model || !activeMethod || !activeDataset || !!modelIssue}
                onClick={() => setStep(2)}
              >
                Next <Icon name="arrow" size={15} />
              </button>
            </div>
          </section>
        )}

        {step === 2 && (
          <section aria-labelledby="training-compute-title">
            <div className="training-heading">
              <h2 id="training-compute-title" ref={headingRef} tabIndex={-1}>
                Compute
              </h2>

            </div>
            <div className="training-review">
              <span>
                {activeDataset?.result.repo_id?.split("/").at(-1) ??
                  "No dataset"}
              </span>
              <span>
                {summaryModel} · {activeMethod?.toUpperCase()}
              </span>
              <span>{activeCameraKeys.length} {activeCameraKeys.length === 1 ? "camera" : "cameras"}</span>
            </div>
            <GpuPicker
              value={selectedGpu}
              onChange={setRuntimeId}
              disabled={busy}
              choices={[
                ...gpuChoices.map((gpu) => ({ id: gpu, label: gpu, memory: ({ L4: "24 GB", T4: "16 GB", A100: "40 GB" } as Record<string, string>)[gpu] })),
                ...localRuntimes.map((item) => ({ id: item.id, label: item.label,
                  memory: item.gpu_memory_mib ? `${number(item.gpu_memory_mib / 1024)} GB` : undefined })),
              ]}
            />
            {!!model?.minimum_gpu_memory_gb && <p className="training-control-note">{model.label} requires at least {model.minimum_gpu_memory_gb} GB of GPU memory. Only compatible GPUs are shown.</p>}
            <details className="training-disclosure">
              <summary>
                Training settings
                <span>
                  {resumeId
                    ? "Original recipe"
                    : `${number(recipe.trainingSteps)} steps · batch ${recipe.batchSize}`}
                </span>
              </summary>
              {resumeId ? (
                <p>Original recipe preserved.</p>
              ) : (
                <div className="training-fields">
                  <label>
                    Steps
                    <input
                      type="number"
                      min="2"
                      step="1"
                      value={recipe.trainingSteps}
                      disabled={busy}
                      onChange={(event) =>
                        update("trainingSteps", Number(event.target.value))
                      }
                    />
                  </label>
                  <label>
                    Batch size
                    <input
                      type="number"
                      min="1"
                      step="1"
                      value={recipe.batchSize}
                      disabled={busy}
                      onChange={(event) =>
                        update("batchSize", Number(event.target.value))
                      }
                    />
                  </label>
                  <label>
                    Learning rate
                    <input type="number" min="0.000000001" step="any" value={recipe.learningRate} disabled={busy}
                      onChange={event => update("learningRate", Number(event.target.value))} />
                  </label>
                  <label>
                    Gradient accumulation
                    <input type="number" min="1" step="1" value={recipe.gradientAccumulation} disabled={busy}
                      onChange={event => update("gradientAccumulation", Number(event.target.value))} />
                    <small>{positiveInteger(recipe.batchSize) && positiveInteger(recipe.gradientAccumulation) ? `Effective batch size: ${number(recipe.batchSize * recipe.gradientAccumulation)}` : "Microbatches per optimizer step"}</small>
                  </label>
                  <label>
                    Random seed
                    <input type="number" min="0" max="2147483647" step="1" value={recipe.seed} disabled={busy}
                      onChange={event => update("seed", Number(event.target.value))} />
                    <small>Controls episode split and training randomness.</small>
                  </label>
                  <label>
                    Validation fraction
                    <input type="number" min="0.01" max="0.99" step="any" value={recipe.validationFraction} disabled={busy}
                      onChange={event => update("validationFraction", Number(event.target.value))} />
                    <small>Separate episodes reserved for validation.</small>
                  </label>
                  <label>
                    Validate every (steps)
                    <input type="number" min="1" step="1" value={recipe.evalEvery} disabled={busy}
                      onChange={event => update("evalEvery", Number(event.target.value))} />
                  </label>
                  <label>
                  Checkpoints
                    <input type="number" min="1" step="1" value={checkpointCount} disabled={busy}
                      aria-label="Checkpoints"
                      aria-describedby="training-checkpoint-help"
                      onChange={(event) => setRecipe(previous => ({ ...previous, checkpointCount: Number(event.target.value), checkpointIntervalOverride: null }))} />
                    {positiveInteger(checkpointInterval) && <small id="training-checkpoint-help">
                      {recipe.checkpointIntervalOverride
                        ? `Saved preference: every ${number(checkpointInterval)} steps. Change this count to replace it.`
                        : `Every ${number(checkpointInterval)} steps. Final checkpoint included.`}
                      {runtime?.execution === "skypilot" ? " Saved on Google Cloud; only checkpoint details are kept on the application host." : " Saved on the selected compute storage."}
                    </small>}
                  </label>
                </div>
              )}
            </details>
            <div className="training-footer">
              <button
                type="button"
                className="text-button"
                onClick={() => setStep(1)}
              >
                Back
              </button>
              <span className="training-launch">
                {needsAccount
                  ? <button type="button" className="text-button" onClick={onComputeSettings}>Connect account</button>
                  : <span role="status">{runStarting ? "" : blocked}</span>}
                <button
                  type="submit"
                  className="primary-button"
                  disabled={!!blocked || busy}
                >
                  {startLabel}
                  <Icon name="arrow" size={15} />
                </button>
              </span>
            </div>
          </section>
        )}

        {!!resumeOptions.length && (
          <details className="training-disclosure training-resume">
            <summary>Resume a previous run</summary>
            <fieldset className="training-resume-options">
              <legend className="visually-hidden">Resume checkpoint</legend>
              <label>
                <input
                  type="radio"
                  name="resume-checkpoint"
                  checked={!resumeId}
                  disabled={busy}
                  onChange={() => setResumeId("")}
                />
                Start a new run
              </label>
              {resumeOptions.map((item) => (
                <label key={item.id}>
                  <input
                    type="radio"
                    name="resume-checkpoint"
                    aria-label={item.label}
                    checked={resumeId === item.id}
                    disabled={busy}
                    onChange={() => {
                      setResumeId(item.id);
                      setStep(2);
                      mutation.reset();
                    }}
                  />
                  {item.label}
                </label>
              ))}
            </fieldset>
          </details>
        )}
        {[options.error, jobs.error, artifacts.error, mutation.error]
          .filter(Boolean)
          .map((error, index) => (
            <p className="error-notice" role="alert" key={index}>
              {error?.message}
            </p>
          ))}
        {(options.isError || jobs.isError) && (
          <button
            type="button"
            className="secondary-button"
            onClick={() => {
              void options.refetch();
              void jobs.refetch();
            }}
          >
            Retry
          </button>
        )}
      </form>
      </>}

      {view === "run" && (
        <section
          className="training-activity training-job-detail"
          aria-label="Fine-tuning job"
        >
          <button type="button" className="text-button training-back" onClick={() => setView("jobs")}><Icon name="arrow" size={14} />Back to jobs</button>
          <div className="training-heading">
            <h2 ref={headingRef} tabIndex={-1}>{selectedRun ? trainingRunModelLabel(selectedRun, models, artifacts.data ?? [], trainingRuns) ?? "Fine-tuning job" : "Fine-tuning job"}</h2>
            <button
              type="button"
              className="text-button"
              onClick={onDiagnostics}
            >
              Diagnostics <Icon name="arrow" size={14} />
            </button>
          </div>
          {selectedRun && (
            <TrainingMonitor
              key={selectedRun.id}
              run={selectedRun}
              projectId={projectId}
              active={active}
              artifacts={(artifacts.data ?? []).filter(item => item.job_id === selectedRun.id)}
              modelCatalog={models}
              exportRuntimes={options.isSuccess ? options.data.runtimes : []}
              exportJobs={jobs.data ?? []}
              exportArtifacts={artifacts.data ?? []}
              onCancel={() => cancel.mutate(selectedRun.id)}
              cancelling={cancel.isPending && cancel.variables === selectedRun.id}
              cancelError={cancel.variables === selectedRun.id ? cancel.error : null}
              onQuantize={onQuantize}
              onResume={resumeOptions.some(item => item.jobId === selectedRun.id) ? () => {
                setResumeId(resumeOptions.find(item => item.jobId === selectedRun.id)!.id);
                setStep(2);
                mutation.reset();
                setView("new");
                window.scrollTo({ top: 0, behavior: "smooth" });
              } : undefined}
            />
          )}
          {!selectedRun && <p className="muted">This job is no longer available.</p>}
        </section>
      )}
    </>
  );
}
