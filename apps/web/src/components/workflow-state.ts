'use client';

import { useEffect, useState } from 'react';
import { nativeQuantizationOf, nativeQuantizationOnly } from '@/lib/native-quantization';
import { replayJob } from '@/lib/native-replay';
import { isSimulationJob } from '@/lib/native-simulation';
import { observedProgress } from '@/lib/run-summary';
import {
  checkpointStep,
  isCloudArtifact,
  quantizationIssue,
  sortCheckpoints,
  trainingRunModelLabel,
} from '@/lib/checkpoints';
import { initialPreferences, type Preferences } from '@/lib/workflow-preferences';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { readPublicPreference, writePublicPreference } from '@/lib/public-preferences';
import { api, isActive, type Job, type LifecycleResult, type PolicyRequest, type PolicyArtifact } from '@/lib/api';
import { policyRequest, precisionName } from './workflow-shared';

function states(text: string) {
  const values = text.split(',').map((x) => Number(x.trim()));
  if (!text.trim() || values.some((x) => !Number.isInteger(x) || x < 0))
    throw new Error('Enter comma-separated, non-negative state IDs.');
  return values;
}
function resultOf(job?: Job): LifecycleResult | undefined {
  return job?.result && 'artifacts' in job.result ? job.result : undefined;
}

// Owns WorkflowPanel's preferences, job and policy queries, start and cancel mutations, and view navigation.
export function useWorkflowPanel({
  projectId,
  startNew,
  stage,
  preferredArtifactId,
  preferredJobId,
  modelRunId,
  onBackToJobs,
  onBackToModels,
}: {
  projectId: string;
  startNew?: boolean;
  stage: string;
  preferredArtifactId?: string;
  preferredJobId?: string;
  modelRunId?: string;
  onBackToJobs?: () => void;
  onBackToModels?: () => void;
}) {
  const client = useQueryClient();
  const [preferences, setPreferences] = useState(initialPreferences);
  const [loadedProjectId, setLoadedProjectId] = useState<string | null>(null);
  const ready = !!projectId && loadedProjectId === projectId;
  const [runtimeId, setRuntimeId] = useState('');
  const [input, setInput] = useState(preferredArtifactId ?? '');
  const [datasetId, setDatasetId] = useState('');
  const [method, setMethod] = useState('lora');
  const [resumeId, setResumeId] = useState('');
  const [selectedJobId, setSelectedJobId] = useState(preferredJobId ?? '');
  const [detailsJobId, setDetailsJobId] = useState<string | null>(null);
  const [view, setView] = useState<'history' | 'new' | 'detail'>(
    preferredJobId ? 'detail' : preferredArtifactId || startNew ? 'new' : 'history',
  );
  const options = useQuery({
    queryKey: ['policy-options'],
    queryFn: api.policyOptions,
    retry: false,
  });
  const jobs = useQuery({
    queryKey: ['jobs', projectId],
    queryFn: () => api.jobs(projectId),
    enabled: !!projectId,
    retry: false,
    refetchInterval: (query) => (query.state.data?.some(isActive) ? 1000 : 5000),
  });
  const artifacts = useQuery({
    queryKey: ['artifacts', projectId],
    queryFn: () => api.artifacts(projectId),
    enabled: !!projectId,
    refetchInterval: 3000,
  });
  useEffect(() => {
    if (!projectId) return;
    try {
      const restored: Preferences = {
        ...initialPreferences,
        ...JSON.parse(readPublicPreference(localStorage, `workflow.${projectId}`) ?? '{}'),
      };
      // Restore saved Spatial preferences from before the protocol was restricted.
      setPreferences(restored.suite === 'libero_spatial' ? { ...restored, mode: 'libero', steps: 280 } : restored);
    } catch {
      setPreferences(initialPreferences);
    }
    setLoadedProjectId(projectId);
  }, [projectId]);
  useEffect(() => {
    if (ready)
      try {
        writePublicPreference(localStorage, `workflow.${projectId}`, JSON.stringify(preferences));
      } catch {
        /* Settings still work for this session. */
      }
  }, [projectId, preferences, ready]);
  const policyJobs = (jobs.data ?? [])
    .filter(
      (x) =>
        x.project_id === projectId &&
        x.kind.startsWith('policy.') &&
        !isSimulationJob(x) &&
        !replayJob(x) &&
        !nativeQuantizationOf(x),
    )
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  const stageJobs =
    stage === 'settings'
      ? policyJobs
      : policyJobs.filter((job) =>
          stage === 'Quantize'
            ? ['policy.quantize', 'policy.workflow'].includes(job.kind)
            : stage === 'Evaluate'
              ? job.kind === 'policy.evaluate'
              : stage === 'Fine-tune'
                ? job.kind === 'policy.finetune'
                : job.kind === 'policy.run',
        );
  const selected = selectedJobId ? stageJobs.find((x) => x.id === selectedJobId) : stageJobs[0];
  const events = useQuery({
    queryKey: ['job-events', selected?.id],
    queryFn: () => api.events(selected!.id),
    enabled: !!selected && (stage === 'settings' || view === 'detail'),
    refetchInterval: selected && (stage === 'settings' || view === 'detail') && isActive(selected) ? 1000 : false,
  });
  const data = resultOf(selected);
  const progress = observedProgress(events.data ?? []);
  const datasets = (jobs.data ?? []).filter((x) => x.kind === 'dataset.inspect' && x.status === 'succeeded');
  const inputs = (artifacts.data ?? [])
    .filter((x) => x.project_id === projectId)
    .filter((x) =>
      stage === 'Quantize'
        ? ['training_checkpoint', 'native_checkpoint'].includes(x.format) ||
          (x.format === 'gguf' && x.metadata?.precision === 'float')
        : stage === 'settings'
          ? x.format === 'gguf'
          : ['gguf', 'deployment_package'].includes(x.format),
    );
  const checkpoints = sortCheckpoints(
    (artifacts.data ?? [])
      .filter((x) => x.project_id === projectId)
      .filter((x) => x.format === 'training_checkpoint' || x.format === 'native_checkpoint'),
    policyJobs,
  );
  const selectedInput = input;
  const inputArtifact = inputs.find((item) => item.id === selectedInput);
  const inputIssue = stage === 'Quantize' ? quantizationIssue(inputArtifact) : null;
  const cloudCheckpoint = !!inputArtifact && isCloudArtifact(inputArtifact);
  const needsNativeExecution = stage === 'settings';
  const runtimes = (options.data?.runtimes ?? []).filter(
    (item) =>
      item.enabled !== false &&
      !item.training_only &&
      !item.export_only &&
      !item.native_distillation_only &&
      !item.native_replay_only &&
      !nativeQuantizationOnly(item) &&
      (!needsNativeExecution || item.execution !== 'skypilot') &&
      (stage !== 'Evaluate' || item.engine_evaluation !== false) &&
      (stage !== 'Run' || item.run !== false) &&
      (item.execution !== 'skypilot' || stage !== 'Evaluate' || item.engine_evaluation === true) &&
      (item.execution !== 'skypilot' || stage !== 'Run' || item.run === true),
  );
  const runtime =
    runtimes.find((item) => item.id === runtimeId) ??
    (cloudCheckpoint ? runtimes.find((item) => item.provider === 'gcp' || item.execution === 'skypilot') : undefined) ??
    runtimes[0];
  const defaultLanguage = options.data?.quantization_defaults[runtime?.device ?? 'cpu'].language ?? 'Q8_0';
  const nativeQuantization = stage === 'Quantize' && runtime?.execution !== 'skypilot';
  const cloudEngine = (stage === 'Evaluate' || stage === 'Run') && runtime?.execution === 'skypilot';
  const evaluationMode = cloudEngine ? 'engine' : preferences.mode;
  const evaluationSuite = cloudEngine ? 'libero_object' : preferences.suite;
  const executionIssue =
    cloudCheckpoint && runtime && runtime.execution !== 'skypilot'
      ? 'This policy is stored on Google Cloud. Choose a cloud target.'
      : cloudEngine && inputArtifact?.metadata?.architecture && inputArtifact.metadata.architecture !== 'smolvla'
        ? 'Cloud engine checks currently support SmolVLA GGUF policies.'
        : (stage === 'Evaluate' || stage === 'Run') &&
            !cloudEngine &&
            preferences.mode === 'libero' &&
            runtime &&
            !runtime.simulation
          ? 'This target has no simulator. Choose engine checks or a simulation target.'
          : null;
  const language = preferences.precision === 'recommended' ? defaultLanguage : preferences.precision;
  const mutation = useMutation({
    mutationFn: async () => {
      if (!ready || !runtime) throw new Error('Select a project and a configured execution target.');
      if (inputIssue) throw new Error(inputIssue);
      if (executionIssue) throw new Error(executionIssue);
      const evaluation: PolicyRequest['evaluation'] =
        stage === 'Quantize' && !nativeQuantization
          ? undefined
          : {
              mode: evaluationMode,
              suite: evaluationSuite,
              ...(evaluationSuite === 'libero_spatial' ? { task_ids: states(preferences.taskIds) } : {}),
              repetitions: preferences.repetitions,
              warmups: preferences.warmup,
              task_id: preferences.task,
              initial_states: states(preferences.searchStates),
              final_states: states(preferences.finalStates),
              steps: preferences.steps,
              seed: 42,
            };
      if (evaluation && evaluationSuite === 'libero_spatial') {
        if (!preferences.parityProfile.trim() || !preferences.parityRmse.trim() || !preferences.parityMaxError.trim())
          throw new Error('Set your approved floating-policy parity profile and tolerances in Settings.');
        const rmse = Number(preferences.parityRmse);
        const maximum = Number(preferences.parityMaxError);
        if (!Number.isFinite(rmse) || !Number.isFinite(maximum) || rmse < 0 || maximum < 0)
          throw new Error('Parity tolerances must be finite, non-negative numbers.');
        evaluation.parity_limits = {
          profile: preferences.parityProfile.trim(),
          max_rmse: rmse,
          max_abs_error: maximum,
        };
      }
      const body: PolicyRequest = {
        operation:
          stage === 'Fine-tune'
            ? 'policy.finetune'
            : stage === 'Quantize'
              ? nativeQuantization
                ? 'policy.workflow'
                : 'policy.quantize'
              : stage === 'Evaluate' || stage === 'settings'
                ? 'policy.evaluate'
                : 'policy.run',
        runtime_id: runtime.id,
        evaluation,
        training_method: method,
        timeout_seconds: 7200,
        precision: { language, vision: preferences.vision ? 'Q8_0' : null },
      };
      if (stage === 'Fine-tune') {
        body.dataset_job_id = datasetId || datasets[0]?.id;
        if (!body.dataset_job_id) throw new Error('Inspect a compatible dataset first.');
        body.training_method = method;
        body.training = {
          steps: preferences.trainingSteps,
          warmup_steps: Math.min(50, preferences.trainingSteps - 1),
          camera_key: preferences.camera,
        };
        if (resumeId) {
          const checkpoint = checkpoints.find((x) => x.id === resumeId);
          const prior = policyJobs.find((x) => x.id === (checkpoint?.job_id ?? resumeId.slice(4)));
          if (!prior || !('training_method' in prior.request))
            throw new Error('The original training recipe is unavailable.');
          body.training_method = prior.request.training_method;
          body.dataset_job_id = prior.request.dataset_job_id;
          body.training = null;
          if (resumeId.startsWith('job:')) body.resume_job_id = resumeId.slice(4);
          else body.artifact_id = resumeId;
        }
      } else {
        if (!inputArtifact || inputArtifact.project_id !== projectId)
          throw new Error('Choose one of this project’s saved models.');
        body.artifact_id = inputArtifact.id;
        if (nativeQuantization) {
          body.candidates = [{ language, vision: preferences.vision ? 'Q8_0' : null }];
          if (preferences.compareQ4)
            body.candidates.push({ language: language === 'Q4_0' ? 'Q8_0' : 'Q4_0', vision: null });
          if (preferences.select && preferences.mode === 'libero')
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
      setView('detail');
      client.setQueryData<Job[]>(['jobs', projectId], (previous) => [
        job,
        ...(previous ?? []).filter((item) => item.id !== job.id),
      ]);
      void client.invalidateQueries({ queryKey: ['jobs', projectId] });
    },
  });
  const cancel = useMutation({
    mutationFn: () => api.cancel(selected!.id),
    onSuccess: () => client.invalidateQueries({ queryKey: ['jobs', projectId] }),
  });
  const submittedJob = mutation.data
    ? (policyJobs.find((job) => job.id === mutation.data.id) ?? mutation.data)
    : undefined;
  const starting = mutation.isPending || (!!submittedJob && isActive(submittedJob));
  function update<K extends keyof Preferences>(key: K, value: Preferences[K]) {
    if (ready) setPreferences((old) => ({ ...old, [key]: value }));
  }
  const errors = [options.error, jobs.error, artifacts.error, mutation.error, cancel.error].filter(Boolean);
  const preferencesBlocker = !projectId
    ? 'Select or create a project to edit workflow settings.'
    : !ready
      ? 'Loading workflow settings…'
      : null;
  const diagnosticBlocker = options.isPending
    ? 'Checking execution targets…'
    : options.isError
      ? 'Execution targets could not be loaded. Check the application connection.'
      : !projectId
        ? 'Select or create a project to save your diagnostic results.'
        : !ready
          ? 'Loading workflow settings…'
          : !runtime
            ? 'No execution target is configured. Set up a native worker on the application host first.'
            : artifacts.isPending || jobs.isPending
              ? 'Loading project policies and runs…'
              : artifacts.isError || jobs.isError
                ? 'Project policies or runs could not be loaded. Check the application connection.'
                : policyJobs.some(isActive)
                  ? 'Wait for the active policy job to finish, or cancel it below.'
                  : !inputs.length
                    ? 'This project has no GGUF policy yet. Create one in Quantize, then return here.'
                    : !inputs.some((artifact) => artifact.id === selectedInput)
                      ? 'Choose a project policy to evaluate.'
                      : preferences.mode === 'libero' && !runtime.simulation
                        ? 'This target has no LIBERO simulator. Choose engine checks or a simulator-enabled target.'
                        : null;
  const historyTitle =
    stage === 'Quantize'
      ? 'Quantization jobs'
      : stage === 'Evaluate'
        ? 'Evaluation jobs'
        : stage === 'Fine-tune'
          ? 'Training jobs'
          : 'Run jobs';
  const newLabel =
    stage === 'Quantize'
      ? 'New quantization'
      : stage === 'Evaluate'
        ? 'New evaluation'
        : stage === 'Fine-tune'
          ? 'New training'
          : 'New run';
  const allArtifacts = [...(artifacts.data ?? []), ...(data?.artifacts ?? [])];
  const sourceFor = (job: Job): PolicyArtifact | undefined =>
    allArtifacts.find((artifact) => artifact.id === policyRequest(job)?.artifact_id);
  const modelFor = (job: Job): string =>
    trainingRunModelLabel(job, options.data?.training_models, allArtifacts, policyJobs) ?? 'Policy';
  const checkpointFor = (job: Job): string => {
    const source = sourceFor(job);
    const step = source ? checkpointStep(source) : null;
    if (step !== null) return `Step ${step.toLocaleString()}`;
    const request = policyRequest(job);
    const configured = options.data?.sources.find((item) => item.id === request?.source_id);
    return source?.label ?? configured?.label ?? (request?.artifact_id ? 'Saved policy' : '—');
  };
  const precisionFor = (job: Job): string => {
    const request = policyRequest(job);
    if (job.kind === 'policy.workflow' && request?.candidates?.length) {
      return request.candidates.map(precisionName).join(' / ');
    }
    return precisionName(job.kind === 'policy.quantize' ? request?.precision : sourceFor(job)?.metadata?.precision);
  };
  const openNew = () => {
    mutation.reset();
    cancel.reset();
    setInput('');
    setSelectedJobId('');
    setView('new');
  };
  const openJob = (id: string) => {
    mutation.reset();
    cancel.reset();
    setSelectedJobId(id);
    setView('detail');
  };
  const backToHistory = () => {
    mutation.reset();
    cancel.reset();
    if (view === 'new' && modelRunId && onBackToModels) onBackToModels();
    else if (onBackToJobs) onBackToJobs();
    else setView('history');
  };
  return {
    artifacts,
    backToHistory,
    cancel,
    checkpointFor,
    checkpoints,
    cloudEngine,
    data,
    datasetId,
    datasets,
    detailsJobId,
    diagnosticBlocker,
    errors,
    evaluationMode,
    events,
    executionIssue,
    historyTitle,
    input,
    inputArtifact,
    inputIssue,
    inputs,
    jobs,
    language,
    method,
    modelFor,
    modelRunId,
    mutation,
    nativeQuantization,
    newLabel,
    openJob,
    openNew,
    options,
    policyJobs,
    precisionFor,
    preferences,
    preferencesBlocker,
    progress,
    projectId,
    ready,
    resumeId,
    runtime,
    runtimes,
    selected,
    selectedInput,
    setDatasetId,
    setDetailsJobId,
    setInput,
    setMethod,
    setPreferences,
    setResumeId,
    setRuntimeId,
    setSelectedJobId,
    stage,
    stageJobs,
    starting,
    update,
    view,
  };
}

export type WorkflowState = ReturnType<typeof useWorkflowPanel>;
