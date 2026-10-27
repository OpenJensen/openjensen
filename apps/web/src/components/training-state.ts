'use client';

import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api, isActive, isDatasetJob, type Job, type PolicyRequest } from '@/lib/api';
import { datasetLibrary } from '@/lib/dataset-library';
import { projectHistoryView } from '@/lib/project-history';
import { readPublicPreference, writePublicPreference } from '@/lib/public-preferences';
import { gradientAccumulationAvailable, trainingModels, type TrainingModel } from '@/lib/training-models';
import { checkpointStep, trainingRunModelLabel } from '@/lib/checkpoints';
import { type JobHistoryEntry } from '@/components/job-history';
import { trainingReceipt } from '@/lib/training-submission';
import { submissionOf, useDurableSubmission } from '@/lib/durable-submission';
import { type PolicyJobAttempt } from '@/lib/policy-job-attempt';
import {
  defaultTemporal,
  restoreTemporal,
  temporalFamily,
  temporalIssue,
  temporalRecipe,
  type TemporalDrafts,
} from '@/lib/training-temporal';
import { useTrainingBatch } from '@/lib/training-batch';
import { combinationCameraMapping, datasetCombinationIssue } from '@/lib/dataset-combination';
import { defaultRecipe, type Recipe } from '@/lib/training-recipe';
import {
  cameras,
  datasetIssue,
  modelDatasetIssue,
  number,
  positiveInteger,
  savedRecipe,
  validTrainingHistory,
  type InspectedDataset,
} from './training-shared';

// Fine-tune page state, queries and actions; TrainingPanel and its step and job views render from this.
export function useTrainingPanel({
  projectId,
  preferredDatasetId,
  startNew,
  showJobsRequest,
  preferredRunId,
  active,
}: {
  projectId: string;
  preferredDatasetId?: string;
  startNew?: { id: number; datasetId?: string };
  showJobsRequest?: number;
  preferredRunId?: string;
  active: boolean;
}) {
  const client = useQueryClient();
  const modules = useQuery({ queryKey: ['modules'], queryFn: api.modules, enabled: active, staleTime: 60_000 });
  const cloudInstalled = modules.data?.some((item) => item.id === 'google-cloud' && item.installed) ?? false;
  const [view, setView] = useState<'jobs' | 'new' | 'run'>(startNew ? 'new' : 'jobs');
  const [step, setStep] = useState(0);
  const [datasetId, setDatasetId] = useState(startNew?.datasetId ?? preferredDatasetId ?? '');
  const [extraDatasetIds, setExtraDatasetIds] = useState<string[]>([]);
  const [cameraMappings, setCameraMappings] = useState<Record<string, Record<string, string>>>({});
  const [modelIds, setModelIds] = useState<string[]>([]);
  const batch = useTrainingBatch(projectId);
  const [cameraSelections, setCameraSelections] = useState<Record<string, string[]>>({});
  const [previewEpisode, setPreviewEpisode] = useState<{
    datasetId: string;
    index: number;
  } | null>(null);
  const [modelId, setModelId] = useState('');
  const [modelBatchSizes, setModelBatchSizes] = useState<Record<string, number>>({});
  const [runtimeId, setRuntimeId] = useState('');
  const [method, setMethod] = useState('');
  const [resumeId, setResumeId] = useState('');
  const [recipe, setRecipe] = useState(defaultRecipe);
  const [temporalDrafts, setTemporalDrafts] = useState<TemporalDrafts>({});
  const [loaded, setLoaded] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState('');
  const submission = useDurableSubmission({ project: projectId, operation: 'policy.finetune' });
  const attempt = submission.attempt;
  const keyedAttempt = submissionOf(attempt);
  const [historyReviewed, setHistoryReviewed] = useState<PolicyJobAttempt>(null);
  const mounted = useRef(true),
    selectionGeneration = useRef(0),
    currentAttempt = useRef(attempt);
  currentAttempt.current = attempt;
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  function chooseRun(id: string) {
    selectionGeneration.current += 1;
    setSelectedRunId(id);
    setView('run');
  }
  function showHistory() {
    selectionGeneration.current += 1;
    setView('jobs');
  }
  function retainJob(job: Job, generation: number) {
    client.setQueryData<Job[]>(['jobs', projectId], (previous) => [
      job,
      ...(previous ?? []).filter((item) => item.id !== job.id),
    ]);
    if (mounted.current && generation === selectionGeneration.current) chooseRun(job.id);
  }
  const validateReceipt = (value: unknown, body: unknown) => trainingReceipt(value, projectId, body);
  async function recoverSubmission(retry = false) {
    const generation = selectionGeneration.current;
    const job = await (retry ? submission.retry(validateReceipt) : submission.reconcile(validateReceipt));
    if (job) retainJob(job, generation);
  }
  const headingRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    if (!active) return;
    window.scrollTo({ top: 0, behavior: 'instant' });
    headingRef.current?.focus({ preventScroll: true });
  }, [step, active, view]);
  const options = useQuery({
    queryKey: ['policy-options'],
    queryFn: api.policyOptions,
    enabled: active,
    refetchInterval: active ? 15000 : false,
  });
  const jobQuery = useQuery({
    queryKey: ['jobs', projectId],
    queryFn: async () => {
      const history = await api.jobs(projectId);
      if (!validTrainingHistory(history, projectId))
        throw new Error('Training history is malformed or contains another project. Recovery remains paused.');
      return history;
    },
    enabled: active && !!projectId,
    retry: false,
    refetchInterval: (query) =>
      active
        ? validTrainingHistory(query.state.data, projectId) && query.state.data.some(isActive)
          ? 1000
          : 5000
        : false,
  });
  const jobs = projectHistoryView(jobQuery, projectId);
  const library = useQuery({
    queryKey: ['datasets', projectId],
    queryFn: ({ signal }) => datasetLibrary.list(projectId, signal),
    enabled: active && !!projectId,
    retry: false,
    refetchInterval: active ? 5000 : false,
  });
  const artifacts = useQuery({
    queryKey: ['artifacts', projectId],
    queryFn: () => api.artifacts(projectId),
    enabled: active && !!projectId,
    refetchInterval: active ? 5000 : false,
  });
  async function reviewHistory() {
    const expected = attempt;
    if (!expected || expected.state !== 'uncertain' || submissionOf(expected)) return;
    // Cancel an older in-flight read so it cannot count as this explicit review.
    await client.cancelQueries({ queryKey: ['jobs', projectId], exact: true });
    const result = await jobs.refetch();
    if (
      mounted.current &&
      currentAttempt.current === expected &&
      result.isSuccess &&
      !result.isError &&
      validTrainingHistory(result.data, projectId)
    )
      setHistoryReviewed(expected);
  }
  useEffect(() => {
    if (!projectId) {
      setLoaded(false);
      return;
    }
    try {
      const saved = JSON.parse(readPublicPreference(localStorage, `workflow.${projectId}`) ?? '{}');
      const restored = savedRecipe(saved);
      setRecipe(restored);
      setTemporalDrafts(restoreTemporal(saved.trainingTemporalByModel));
      setModelId(typeof saved.trainingModelId === 'string' ? saved.trainingModelId : '');
      const batches =
        saved.modelBatchSizes && typeof saved.modelBatchSizes === 'object'
          ? Object.fromEntries(
              Object.entries(saved.modelBatchSizes).filter((entry): entry is [string, number] =>
                positiveInteger(entry[1]),
              ),
            )
          : {};
      setModelBatchSizes({ smolvla: restored.batchSize, ...batches });
    } catch {
      setRecipe(defaultRecipe);
      setTemporalDrafts({});
    }
    setLoaded(true);
  }, [projectId]);
  useEffect(() => {
    if (!loaded || !projectId) return;
    try {
      const saved = JSON.parse(readPublicPreference(localStorage, `workflow.${projectId}`) ?? '{}');
      writePublicPreference(
        localStorage,
        `workflow.${projectId}`,
        JSON.stringify({
          ...saved,
          ...recipe,
          trainingTemporalByModel: temporalDrafts,
          modelBatchSizes,
          trainingModelId: modelId,
          checkpointEvery: undefined,
          trainingDefaultsVersion: 3,
        }),
      );
    } catch {
      /* This session remains usable without browser storage. */
    }
  }, [projectId, recipe, temporalDrafts, modelId, modelBatchSizes, loaded]);
  function update<K extends keyof Recipe>(key: K, value: Recipe[K]) {
    setRecipe((old) => ({ ...old, [key]: value }));
    if (key === 'batchSize' && typeof value === 'number' && modelId)
      setModelBatchSizes((previous) => ({ ...previous, [modelId]: value }));
  }

  const trainingRuns = [
    ...(jobs.data ?? []),
    ...(submission.receipt && !(jobs.data ?? []).some((job) => job.id === submission.receipt?.id)
      ? [submission.receipt]
      : []),
  ]
    .filter((job) => job.project_id === projectId && job.kind === 'policy.finetune')
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  // Completed artifacts have reached the recipe's final step and cannot resume.
  const resumableRuns = trainingRuns.filter((job) => ['failed', 'interrupted', 'cancelled'].includes(job.status));
  const checkpoints = (artifacts.data ?? []).filter(
    (item) => item.format === 'training_checkpoint' && resumableRuns.some((job) => job.id === item.job_id),
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
  const priorRequest = prior?.request && 'training_method' in prior.request ? prior.request : undefined;
  const allInspections = (jobs.data ?? [])
    .filter(isDatasetJob)
    .filter(
      (job): job is InspectedDataset => job.project_id === projectId && job.status === 'succeeded' && !!job.result,
    )
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  // Keep the selected inspection's exact ID, while avoiding repeated cards for the same snapshot.
  const unique = new Map<string, InspectedDataset>();
  for (const job of allInspections) {
    const key = `${job.result.source}:${job.result.repo_id ?? job.id}:${job.result.revision}`;
    if (
      !unique.has(key) ||
      (resumeId ? [priorRequest?.dataset_job_id] : [datasetId, ...extraDatasetIds]).includes(job.id)
    )
      unique.set(key, job);
  }
  const datasets = [...unique.values()].sort((a, b) => b.created_at.localeCompare(a.created_at));
  // Explicit handoffs and manual choices retain their exact identity when history
  // is delayed or changes. A fresh form may default only without a requested ID.
  const dataset = datasetId
    ? datasets.find((job) => job.id === datasetId)
    : datasets.find((job) => !datasetIssue(job.result));
  const cameraOptions = cameras(dataset?.result);
  const selectedCameras =
    dataset && cameraSelections[dataset.id] !== undefined
      ? cameraOptions.filter((key) => cameraSelections[dataset.id].includes(key))
      : cameraOptions;
  function toggleCamera(key: string) {
    if (!dataset) return;
    setCameraSelections((previous) => ({
      ...previous,
      [dataset.id]: selectedCameras.includes(key)
        ? selectedCameras.filter((value) => value !== key)
        : cameraOptions.filter((value) => value === key || selectedCameras.includes(value)),
    }));
  }
  const selectedDatasets = dataset
    ? [
        dataset,
        ...extraDatasetIds
          .map((id) => datasets.find((item) => item.id === id))
          .filter((item): item is InspectedDataset => !!item && item.id !== dataset.id),
      ]
    : [];
  const mappings = Object.fromEntries(
    selectedDatasets.map((item, index) => [
      item.id,
      index === 0
        ? Object.fromEntries(selectedCameras.map((key) => [key, key]))
        : Object.fromEntries(
            selectedCameras.map((key) => [
              key,
              cameraMappings[item.id]?.[key] ??
                combinationCameraMapping(dataset!.result, item.result, selectedCameras)[key] ??
                '',
            ]),
          ),
    ]),
  );
  const mixtureIssue = resumeId
    ? null
    : extraDatasetIds.some((id) => !datasets.some((item) => item.id === id))
      ? 'A selected dataset is unavailable. Refresh or select it again explicitly.'
      : selectedDatasets.length > 8
        ? 'Select up to eight datasets.'
        : (selectedDatasets
            .slice(1)
            .map((item) => datasetCombinationIssue(dataset!.result, item.result, selectedCameras, mappings[item.id]))
            .find(Boolean) ?? null);
  const mergedModels = new Map(trainingModels.map((item) => [item.id, item]));
  for (const item of options.data?.training_models ?? [])
    mergedModels.set(item.id, {
      ...mergedModels.get(item.id),
      ...item,
      description: mergedModels.get(item.id)?.description ?? item.description,
    } as TrainingModel);
  const models = [...mergedModels.values()];
  const runtimes = (options.data?.runtimes ?? []).filter(
    (item) => cloudInstalled || (item.provider ?? 'local') === 'local',
  );
  const cloudGpuMemory: Record<string, number> = { L4: 24, T4: 16, A100: 40, 'A100-80GB': 80 };
  const runtimeMemory = (item: (typeof runtimes)[number]) =>
    item.gpu_memory_mib ? item.gpu_memory_mib / 1024 : (cloudGpuMemory[item.accelerator ?? ''] ?? 0);
  const hasAdapter = (runtime: (typeof runtimes)[number], item: TrainingModel) =>
    item.status !== 'coming_soon' &&
    !!item.model_revision &&
    runtime.training &&
    runtime.device === 'cuda' &&
    (!item.minimum_gpu_memory_gb || runtimeMemory(runtime) >= item.minimum_gpu_memory_gb) &&
    (runtime.training_model_ids ?? ['smolvla']).includes(item.id) &&
    // The bundled on-demand SmolVLA adapter remains usable before preflight.
    // Older API catalogs describe that state with an empty runtime_ids list.
    (item.id === 'smolvla' || !item.runtime_ids || item.runtime_ids.includes(runtime.id));
  const modelSupported = (item: TrainingModel) =>
    item.status !== 'coming_soon' &&
    ((item.id === 'smolvla' && !!item.model_revision) || runtimes.some((runtime) => hasAdapter(runtime, item)));
  let originalTraining = priorRequest?.training;
  let ancestorRequest = priorRequest;
  const visitedRuns = new Set<string>();
  while (!originalTraining && ancestorRequest) {
    const linkedJobId =
      ancestorRequest.resume_job_id ?? checkpoints.find((item) => item.id === ancestorRequest?.artifact_id)?.job_id;
    if (!linkedJobId || visitedRuns.has(linkedJobId)) break;
    visitedRuns.add(linkedJobId);
    const linkedRequest = trainingRuns.find((job) => job.id === linkedJobId)?.request;
    ancestorRequest = linkedRequest && 'training_method' in linkedRequest ? linkedRequest : undefined;
    originalTraining = ancestorRequest?.training;
  }
  const originalModel = models.find((item) => item.model_id === originalTraining?.model_id);
  const model = resumeId
    ? (originalModel ?? (!originalTraining?.model_id ? models.find((item) => item.id === 'smolvla') : undefined))
    : models.find((item) => item.id === modelId);
  const selectedModels = resumeId
    ? model
      ? [model]
      : []
    : models.filter((item) => (modelIds.length ? modelIds : model ? [model.id] : []).includes(item.id));
  const methodFor = (item: TrainingModel) =>
    item.methods.includes(method)
      ? method
      : item.methods.includes(options.data?.default_training_method ?? '')
        ? options.data!.default_training_method
        : item.methods[0];
  const batchFor = (item: TrainingModel) =>
    item.id === model?.id
      ? recipe.batchSize
      : (modelBatchSizes[item.id] ?? (item.id === 'smolvla' ? 64 : item.id === 'psi0' ? 2 : 4));
  const timingFamily = temporalFamily(model?.id);
  const timing = timingFamily ? (temporalDrafts[timingFamily] ?? defaultTemporal(timingFamily)) : null;
  const timingIssue = !resumeId && timing ? temporalIssue(timing) : null;
  const effectiveTiming = timingFamily && timing ? (timing.enabled ? timing : defaultTemporal(timingFamily)) : null;
  function updateTiming(change: Partial<NonNullable<typeof timing>>) {
    if (timingFamily && timing)
      setTemporalDrafts((previous) => ({ ...previous, [timingFamily]: { ...timing, ...change } }));
  }
  const availableMethods = (options.data?.training_methods ?? []).filter((item) => model?.methods.includes(item.id));
  const trainingMethod =
    availableMethods.find((item) => item.id === method)?.id ??
    availableMethods.find((item) => item.id === options.data?.default_training_method)?.id ??
    availableMethods[0]?.id;
  const supportsModel = (item: (typeof runtimes)[number]) =>
    selectedModels.every(
      (model) =>
        model.id !== 'smolvla' ||
        methodFor(model) !== 'full' ||
        (model.native_full_runtime_ids ?? []).includes(item.id),
    ) &&
    item.enabled !== false &&
    selectedModels.length > 0 &&
    selectedModels.every((model) => hasAdapter(item, model));
  const gpuChoices = (cloudInstalled ? ['L4', 'T4', 'A100', 'A100-80GB'] : []).filter((gpu) =>
    selectedModels.every(
      (item) =>
        cloudGpuMemory[gpu] >=
        (item.id === 'smolvla' && methodFor(item) === 'full' ? 24 : (item.minimum_gpu_memory_gb ?? 0)),
    ),
  );
  const defaultGpu = options.data?.compute?.gcp?.default_gpu ?? 'A100';
  const localRuntimes =
    selectedModels.length <= 1 && options.data?.compute?.local.enabled
      ? runtimes.filter(
          (item) =>
            (item.provider ?? 'local') === 'local' &&
            item.training &&
            item.device === 'cuda' &&
            item.enabled !== false &&
            (!model?.minimum_gpu_memory_gb || runtimeMemory(item) >= model.minimum_gpu_memory_gb),
        )
      : [];
  const selectedGpu =
    gpuChoices.includes(runtimeId) || localRuntimes.some((item) => item.id === runtimeId)
      ? runtimeId
      : gpuChoices.includes(defaultGpu)
        ? defaultGpu
        : (gpuChoices[0] ?? localRuntimes[0]?.id ?? '');
  const chosenRuntime = gpuChoices.includes(selectedGpu)
    ? runtimes.find((item) => item.execution === 'skypilot' && item.accelerator === selectedGpu)
    : localRuntimes.find((item) => item.id === selectedGpu);
  const runtime = chosenRuntime && supportsModel(chosenRuntime) ? chosenRuntime : undefined;
  const needsAccount = cloudInstalled && !chosenRuntime;
  const priorDataset = allInspections.find((job) => job.id === priorRequest?.dataset_job_id);
  const activeDataset = resumeId ? priorDataset : dataset;
  const activeMethod = resumeId ? priorRequest?.training_method : trainingMethod;
  const accumulationSupported = gradientAccumulationAvailable(model, runtime?.id, activeMethod);
  const accumulationIssue =
    !resumeId && recipe.gradientAccumulation !== 1 && !accumulationSupported
      ? `Accumulation ${recipe.gradientAccumulation} is not supported by the selected model, method and compute target. Choose a supported configuration or explicitly use accumulation 1.`
      : null;
  const trainingBudget = `${number(recipe.trainingSteps)} ${accumulationSupported ? 'optimizer updates' : 'steps'} · batch ${recipe.batchSize}`;
  const displayedModelId = model?.id;
  const stepComplete = [!!activeDataset && !datasetIssue(activeDataset.result), !!model && !!activeMethod, !!runtime];
  const activeCameraKeys = resumeId
    ? Array.isArray(originalTraining?.camera_keys)
      ? originalTraining.camera_keys.filter((key): key is string => typeof key === 'string')
      : [String(originalTraining?.camera_key ?? 'observation.images.front')]
    : selectedCameras;
  const modelIssue = !resumeId
    ? (selectedModels
        .map(
          (item) =>
            modelDatasetIssue(
              item.id === 'smolvla' && methodFor(item) === 'full' ? { ...item, backend: 'lerobot' } : item,
              activeDataset?.result,
              activeCameraKeys,
            ) ||
            (selectedDatasets.length > 1 && item.backend === 'psi0'
              ? 'Psi-Zero does not support combined datasets.'
              : null),
        )
        .find(Boolean) ?? null)
    : null;
  const batchIssue =
    selectedModels.length > 1 &&
    selectedModels.some(
      (item) => recipe.gradientAccumulation !== 1 && !gradientAccumulationAvailable(item, runtime?.id, methodFor(item)),
    )
      ? 'At least one selected model requires gradient accumulation 1.'
      : null;
  const episodes = useQuery({
    queryKey: ['training-episodes', activeDataset?.id],
    queryFn: () => api.episodes(activeDataset!.id, 0, 6),
    enabled: active && view === 'new' && step === 0 && !!activeDataset && activeDataset.result.source === 'huggingface',
    retry: false,
    staleTime: 60000,
  });
  const episodeIndex =
    previewEpisode?.datasetId === activeDataset?.id ? previewEpisode?.index : episodes.data?.episodes[0]?.episode_index;
  const preview = useQuery({
    queryKey: ['training-preview', activeDataset?.id, episodeIndex],
    queryFn: () => api.episode(activeDataset!.id, episodeIndex!),
    enabled: active && view === 'new' && step === 0 && !!activeDataset && episodeIndex !== undefined,
    retry: false,
    staleTime: 60000,
  });
  const checkpointCount = recipe.checkpointIntervalOverride
    ? Math.ceil(recipe.trainingSteps / recipe.checkpointIntervalOverride)
    : recipe.checkpointCount;
  const checkpointInterval =
    recipe.checkpointIntervalOverride ?? Math.ceil(recipe.trainingSteps / recipe.checkpointCount);
  const recipeValid =
    Number.isInteger(recipe.trainingSteps) &&
    recipe.trainingSteps >= 2 &&
    Number.isInteger(recipe.batchSize) &&
    recipe.batchSize >= 1 &&
    positiveInteger(recipe.checkpointCount) &&
    positiveInteger(checkpointInterval) &&
    Number.isInteger(recipe.seed) &&
    recipe.seed >= 0 &&
    recipe.seed <= 2147483647 &&
    Number.isFinite(recipe.learningRate) &&
    recipe.learningRate > 0 &&
    positiveInteger(recipe.gradientAccumulation) &&
    Number.isFinite(recipe.validationFraction) &&
    recipe.validationFraction > 0 &&
    recipe.validationFraction < 1 &&
    positiveInteger(recipe.evalEvery);
  const blocked = !projectId
    ? 'Select a project.'
    : !submission.hydrated
      ? 'Loading saved request…'
      : !submission.available
        ? 'Browser recovery storage is unavailable. Restore it before starting a job.'
        : attempt
          ? 'Inspect the earlier training request before submitting another job.'
          : !batch.hydrated || !batch.available || batch.journal
            ? 'Resolve the saved multi-model plan before starting another job.'
            : !resumeId && selectedModels.some((item) => item.status === 'coming_soon')
              ? 'A selected model is coming soon. Choose an available model before training.'
              : !resumeId && modelIds.some((id) => !models.some((item) => item.id === id))
                ? 'A selected model is unavailable. Select your models again.'
                : !loaded || options.isPending || jobs.isPending
                  ? 'Loading…'
                  : options.isError || jobs.isError
                    ? 'Couldn’t load training options.'
                    : resumeId && !priorRequest
                      ? 'The original training recipe is unavailable.'
                      : !activeDataset
                        ? 'Select a dataset.'
                        : !resumeId && datasetIssue(activeDataset.result)
                          ? datasetIssue(activeDataset.result)
                          : !resumeId && !selectedCameras.length
                            ? 'Select at least one camera.'
                            : !model || !activeMethod
                              ? 'Select a model and method.'
                              : mixtureIssue || modelIssue || batchIssue
                                ? mixtureIssue || modelIssue || batchIssue
                                : !runtime
                                  ? 'Choose an available GPU.'
                                  : accumulationIssue
                                    ? accumulationIssue
                                    : timingIssue
                                      ? timingIssue
                                      : !resumeId && !recipeValid
                                        ? 'Check training settings.'
                                        : null;
  const mutation = useMutation({
    retry: false,
    mutationFn: async () => {
      if (blocked || !runtime || !activeDataset || !model || !activeMethod)
        throw new Error(blocked ?? 'Complete the training setup first.');
      const body: PolicyRequest = {
        operation: 'policy.finetune',
        runtime_id: runtime.id,
        dataset_job_id: activeDataset.id,
        training_method: activeMethod,
        timeout_seconds: 86400,
      };
      if (resumeId) {
        body.training = null;
        if (resumeId.startsWith('job:')) body.resume_job_id = resumeId.slice(4);
        else body.artifact_id = resumeId;
      } else {
        body.training = {
          model_id: model.model_id,
          model_revision: model.model_revision,
          ...(model.checkpoint_subdirectory ? { checkpoint_subdirectory: model.checkpoint_subdirectory } : {}),
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
          ...temporalRecipe(timingFamily, timing),
        };
      }
      if (!resumeId && selectedDatasets.length > 1) {
        body.dataset_job_ids = selectedDatasets.map((item) => item.id);
        body.dataset_camera_mappings = mappings;
      }
      const generation = selectionGeneration.current;
      setHistoryReviewed(null);
      if (!resumeId && selectedModels.length > 1) {
        const shared = Object.fromEntries(
          Object.entries(body.training!).filter(
            ([key]) =>
              ![
                'prediction_horizon',
                'execution_horizon',
                'observation_history',
                'frame_stride',
                'chunk_size',
                'checkpoint_subdirectory',
              ].includes(key),
          ),
        );
        const bodies = selectedModels.map((item) => ({
          ...body,
          training_method: methodFor(item),
          training: {
            ...shared,
            model_id: item.model_id,
            model_revision: item.model_revision,
            ...(item.checkpoint_subdirectory ? { checkpoint_subdirectory: item.checkpoint_subdirectory } : {}),
            batch_size: batchFor(item),
            ...temporalRecipe(
              temporalFamily(item.id),
              temporalFamily(item.id)
                ? (temporalDrafts[temporalFamily(item.id)!] ?? defaultTemporal(temporalFamily(item.id)!))
                : null,
            ),
          },
        }));
        const accepted = await batch.start(bodies);
        if (mounted.current && generation === selectionGeneration.current) showHistory();
        return accepted[0] ?? null;
      }
      const job = await submission.submit(body, validateReceipt);
      // The shared hook retains the verified ACK before fallible storage cleanup.
      // A late completion may add history, but never replace a manual selection.
      if (job) retainJob(job, generation);
      return job;
    },
  });
  const selectedRun =
    trainingRuns.find((job) => job.id === selectedRunId) ??
    (mutation.data?.id === selectedRunId ? mutation.data : undefined);
  const cancel = useMutation({
    mutationFn: (jobId: string) => api.cancel(jobId),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['jobs', projectId] });
    },
  });
  const submittedRun = mutation.data
    ? (trainingRuns.find((job) => job.id === mutation.data?.id) ?? mutation.data)
    : undefined;
  const runStarting = !!submittedRun && isActive(submittedRun);
  const busy = batch.busy || !!batch.journal || mutation.isPending || submission.busy || runStarting || !!attempt;
  const startLabel = mutation.isPending
    ? 'Starting…'
    : runStarting
      ? submittedRun?.stage === 'preparing' || submittedRun?.status === 'queued'
        ? 'Preparing GPU…'
        : 'Training…'
      : resumeId
        ? 'Resume fine-tuning'
        : selectedModels.length > 1
          ? `Start ${selectedModels.length} training jobs`
          : 'Start fine-tuning';
  const summaryModel = resumeId
    ? String(model?.label ?? originalTraining?.model_id ?? 'Original model')
    : (model?.label ?? 'Choose a model');

  function openNew(dataset?: string) {
    selectionGeneration.current += 1;
    if (dataset) {
      setDatasetId(dataset);
      setExtraDatasetIds([]);
    }
    setResumeId('');
    mutation.reset();
    setView('new');
  }
  function resumeRun(runId: string, checkpointId?: string) {
    selectionGeneration.current += 1;
    const checkpoint = checkpointId
      ? resumeOptions.find((item) => item.id === checkpointId && item.jobId === runId)
      : resumeOptions.find((item) => item.jobId === runId);
    if (!checkpoint) return;
    setResumeId(checkpoint.id);
    setStep(2);
    mutation.reset();
    setView('new');
    window.scrollTo({ top: 0, behavior: 'smooth' });
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
    if (!startNew) showHistory();
    // Sidebar navigation changes the view while preserving the creation draft.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showJobsRequest]);

  useEffect(() => {
    if (preferredRunId) chooseRun(preferredRunId);
  }, [preferredRunId]);

  const historyEntries: JobHistoryEntry[] = trainingRuns.map((job) => {
    const request = 'training_method' in job.request ? job.request : undefined;
    const recordedModel = trainingRunModelLabel(job, models, artifacts.data ?? [], trainingRuns);
    const dataset = allInspections.find((item) => item.id === request?.dataset_job_id);
    const targetSteps = positiveInteger(request?.training?.steps) ? request.training.steps : null;
    const savedSteps = (artifacts.data ?? [])
      .filter((item) => item.job_id === job.id)
      .map(checkpointStep)
      .filter((value): value is number => value !== null);
    const reportedSteps =
      job.result && 'reports' in job.result
        ? (job.result.reports ?? []).map((report) => report.steps).filter(positiveInteger)
        : [];
    const completedSteps = reportedSteps.length
      ? Math.max(...reportedSteps)
      : job.status === 'succeeded'
        ? targetSteps
        : null;
    const progress =
      completedSteps !== null
        ? `${number(completedSteps)} steps completed`
        : savedSteps.length
          ? `Saved step ${number(Math.max(...savedSteps))}${targetSteps ? ` / ${number(targetSteps)}` : ''}`
          : targetSteps
            ? `${number(targetSteps)} steps planned`
            : undefined;
    return {
      id: job.id,
      title: [recordedModel ?? 'Fine-tuning', request?.training_method.toUpperCase()].filter(Boolean).join(' · '),
      subtitle: dataset?.result.repo_id ?? (dataset ? 'Local dataset' : 'Dataset not recorded'),
      status: job.status,
      createdAt: job.created_at,
      progress,
    };
  });

  return {
    accumulationIssue,
    accumulationSupported,
    active,
    activeCameraKeys,
    activeDataset,
    activeMethod,
    artifacts,
    attempt,
    availableMethods,
    batch,
    batchFor,
    blocked,
    busy,
    cameraSelections,
    cancel,
    checkpointCount,
    checkpointInterval,
    chooseRun,
    cloudGpuMemory,
    cloudInstalled,
    dataset,
    datasetId,
    datasets,
    displayedModelId,
    effectiveTiming,
    episodeIndex,
    episodes,
    gpuChoices,
    headingRef,
    historyEntries,
    historyReviewed,
    jobs,
    keyedAttempt,
    library,
    localRuntimes,
    mappings,
    methodFor,
    mixtureIssue,
    model,
    modelBatchSizes,
    modelIssue,
    models,
    modelSupported,
    mutation,
    needsAccount,
    openNew,
    options,
    originalTraining,
    preview,
    projectId,
    recipe,
    recoverSubmission,
    resumeId,
    resumeOptions,
    resumeRun,
    reviewHistory,
    runStarting,
    runtime,
    selectedCameras,
    selectedDatasets,
    selectedGpu,
    selectedModels,
    selectedRun,
    setCameraMappings,
    setDatasetId,
    setExtraDatasetIds,
    setHistoryReviewed,
    setMethod,
    setModelBatchSizes,
    setModelId,
    setModelIds,
    setPreviewEpisode,
    setRecipe,
    setRuntimeId,
    setStep,
    setTemporalDrafts,
    showHistory,
    startLabel,
    step,
    stepComplete,
    submission,
    summaryModel,
    temporalDrafts,
    timing,
    timingIssue,
    toggleCamera,
    trainingBudget,
    trainingRuns,
    update,
    updateTiming,
    view,
  };
}

export type TrainingPanelState = ReturnType<typeof useTrainingPanel>;
