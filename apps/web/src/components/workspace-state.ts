'use client';

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef, useState } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { api, isActive, isDatasetJob, type PolicyArtifact, type Project } from '@/lib/api';
import { quantizeModeFor, type ModelAction } from '@/lib/model-library';
import { engineRuntime, initialQuantizeEntry, initialRunEntry, runJobMode, type Entry, type ProjectEntry, type QuantizeMode, type RunMode } from '@/lib/workflow-entry';
import { storedSimulationAttempt } from '@/lib/native-simulation-recovery';
import { storedAttempt, type PolicyJobAttempt } from '@/lib/policy-job-attempt';
import { replayRuntime } from '@/lib/native-replay';
import { simulationOptions } from '@/lib/native-simulation';
import type { SimulationHandoff } from '@/lib/native-simulation-handoff';
import type { DistillationModel } from '@/components/distillation-panel';
import type { ManagedPublishedCapture } from '@/lib/managed-teaching';
import type { LibraryDataset } from '@/lib/dataset-library';
import { workspaceRoutes, workspaceStage } from '@/lib/workspace-routes';

export function useWorkbenchState() {
  const queryClient = useQueryClient();
  const [projectId, setProjectId] = useState('');
  const [selectedJobId, setSelectedJobId] = useState('');
  const router = useRouter();
  const pathname = usePathname();
  const activeStage = workspaceStage(pathname);
  // Fence delayed preflight reads as soon as navigation is requested, before
  // Next finishes fetching and committing the destination's route payload.
  const routeSelection = useRef({ pathname, stage: activeStage });
  if (routeSelection.current.pathname !== pathname) routeSelection.current = { pathname, stage: activeStage };
  const isSectionCurrent = (index: number) => routeSelection.current.stage === index;
  function openStage(index: number) {
    const route = workspaceRoutes[index];
    if (route) {
      routeSelection.current.stage = index;
      router.push(route.path, { scroll: false });
    }
  }
  const [entries, setEntries] = useState<Record<string, ProjectEntry>>({});
  const [distillationModels, setDistillationModels] = useState<Record<string, DistillationModel | undefined>>({});
  const [entryError, setEntryError] = useState<{ projectId: string; stage: number; message: string } | null>(null);
  const [openSimulation, setOpenSimulation] = useState<{ projectId: string; id: string } | null>(null);
  const [quantizeArtifact, setQuantizeArtifact] = useState<{ projectId: string; artifactId: string } | null>(null);
  const [distillTeachers, setDistillTeachers] = useState<Record<string, string | undefined>>({});
  const [replayArtifact, setReplayArtifact] = useState<{ projectId: string; artifactId: string } | null>(null);
  const [simulationArtifact, setSimulationArtifact] = useState<SimulationHandoff | null>(null);
  const [modelInput, setModelInput] = useState<{ projectId: string; artifactId: string } | null>(null);
  const [importModel, setImportModel] = useState(false);
  const [teachingMode, setTeachingMode] = useState<'manual' | 'managed'>('manual');
  const [teachingCapture, setTeachingCapture] = useState<ManagedPublishedCapture | null>(null);
  const [trainingNavigation, setTrainingNavigation] = useState(0);
  const [openTrainingRun, setOpenTrainingRun] = useState<{ projectId: string; id: string; artifactId?: string } | null>(null);
  const [openQuantizationRun, setOpenQuantizationRun] = useState<{ projectId: string; id: string } | null>(null);
  const [workflowNavigation, setWorkflowNavigation] = useState(0);
  const [startTraining, setStartTraining] = useState<{ id: number; datasetId?: string }>();
  const [settingsTab, setSettingsTab] = useState<'compute' | 'settings' | 'diagnostics'>('compute');
  const [datasetView, setDatasetView] = useState<'sources' | 'inspection' | 'labels'>('sources');
  const [selectedLibrary, setSelectedLibrary] = useState<LibraryDataset | null>(null);
  const [importLibrary, setImportLibrary] = useState<LibraryDataset | null>(null);
  const [intakeSelection, setIntakeSelection] = useState(0);
  const health = useQuery({ queryKey: ['health'], queryFn: api.health, refetchInterval: 15_000, retry: false });
  const projects = useQuery({ queryKey: ['projects'], queryFn: api.projects, retry: false });
  const capabilities = useQuery({ queryKey: ['capabilities'], queryFn: api.capabilities, retry: false });
  useEffect(() => {
    if (!health.isSuccess || health.isError) return;
    // A successful health poll can be the first request after API startup or
    // recovery. Retry the views that may have failed while it was unavailable.
    void queryClient.invalidateQueries({ queryKey: ['projects'] });
    void queryClient.invalidateQueries({ queryKey: ['capabilities'] });
  }, [health.isSuccess, health.isError, queryClient]);
  const jobs = useQuery({
    queryKey: ['jobs', projectId],
    queryFn: () => api.jobs(projectId),
    enabled: !!projectId,
    refetchInterval: query => query.state.data?.some(isActive) ? 1_000 : 5_000,
    retry: false,
  });
  useEffect(() => {
    if (!projects.data || projects.data.some(project => project.id === projectId)) return;
    let stored = '';
    try { stored = localStorage.getItem('firebird.project') ?? ''; } catch { /* Storage may be unavailable. */ }
    const selected = projects.data.find(project => project.id === stored) ?? projects.data[0];
    setProjectId(selected?.id ?? '');
  }, [projects.data, projectId]);
  function navigateStage(index: number) {
    setOpenQuantizationRun(null);
    setModelInput(null); setImportModel(false);
    setOpenTrainingRun(null);
    setOpenSimulation(null);
    setQuantizeArtifact(null);
    setReplayArtifact(null);
    setSimulationArtifact(null);
    setTeachingCapture(null);
    setWorkflowNavigation(value => value + 1);
    if (index === 1) {
      setStartTraining(undefined);
      setTrainingNavigation(value => value + 1);
    }
    openStage(index);
  }
  function startTrainingOnDataset(datasetId: string) {
    const id = trainingNavigation + 1;
    setTrainingNavigation(id);
    setStartTraining({ id, datasetId });
    openStage(1);
  }
  function selectProject(id: string) {
    setOpenQuantizationRun(null);
    setModelInput(null); setImportModel(false);
    setOpenTrainingRun(null);
    setOpenSimulation(null);
    setQuantizeArtifact(null);
    setReplayArtifact(null);
    setSimulationArtifact(null);
    setTeachingCapture(null); setTeachingMode('manual');
    setStartTraining(undefined);
    setProjectId(id);
    setSelectedJobId('');
    setDatasetView('sources');
    setSelectedLibrary(null); setImportLibrary(null);
    try { localStorage.setItem('firebird.project', id); } catch { /* Session selection still works. */ }
  }
  const projectMutation = useMutation({
    mutationFn: (name: string) => api.createProject(name),
    onSuccess: (project) => {
      queryClient.setQueryData<Project[]>(['projects'], previous => [...(previous ?? []), project]);
      selectProject(project.id);
    },
  });
  const project = projects.data?.find(item => item.id === projectId);
  function openDataset(entry: LibraryDataset) {
    selectProject(entry.project_id);
    navigateStage(0);
    if (entry.job_id && entry.id.startsWith('inspection:')) { setSelectedJobId(entry.job_id); setDatasetView('inspection'); }
    else if (entry.status === 'ready' || entry.status === 'converting') { setSelectedLibrary(entry); setDatasetView('labels'); }
    else { setImportLibrary(entry); setIntakeSelection(value=>value+1); setDatasetView('sources'); }
  }
  // Only confirmed membership may enable project-scoped workflow controls.
  const workflowProjectId = projects.isSuccess ? project?.id ?? '' : '';
  function openModelWorkflow(artifact: PolicyArtifact, action: ModelAction) {
    if (!projects.isSuccess || !projects.data.some(item => item.id === artifact.project_id)) return;
    selectProject(artifact.project_id);
    const target = action === 'distill' ? 2 : action === 'quantize' ? 3 : action === 'evaluate' ? 4 : 5;
    navigateStage(target);
    setModelInput({ projectId: artifact.project_id, artifactId: artifact.id });
    setWorkflowNavigation(value => value + 1);
    if (action === 'distill') {
      setDistillTeachers(previous => ({ ...previous, [artifact.project_id]: artifact.id }));
      setDistillationModels(previous => ({ ...previous, [artifact.project_id]: 'act' }));
    } else if (action === 'quantize') {
      const mode = quantizeModeFor(artifact);
      if (!mode) return;
      setQuantizeArtifact({ projectId: artifact.project_id, artifactId: artifact.id });
      setEntries(previous => ({ ...previous, [artifact.project_id]: { ...previous[artifact.project_id], quantize: { mode, origin: 'handoff' } } }));
    } else if (action === 'replay' || action === 'simulate') {
      if (action === 'replay') setReplayArtifact({ projectId: artifact.project_id, artifactId: artifact.id });
      setEntries(previous => ({ ...previous, [artifact.project_id]: { ...previous[artifact.project_id], run: { mode: action === 'replay' ? 'replay' : 'native', origin: 'handoff' } } }));
    }
  }
  const teachingContext = useRef({ project: workflowProjectId, stage: activeStage, mode: teachingMode });
  teachingContext.current = { project: workflowProjectId, stage: activeStage, mode: teachingMode };
  function reviewTeachingCapture(capture: ManagedPublishedCapture) {
    const context = teachingContext.current;
    if (context.project !== capture.project_id || context.stage !== 9 || context.mode !== 'managed') return;
    setTeachingCapture(capture);
  }
  const choosingWorkflow = [3, 4, 5].includes(activeStage);
  const options = useQuery({ queryKey: ['policy-options'], queryFn: api.policyOptions, enabled: choosingWorkflow, retry: false, refetchInterval: choosingWorkflow ? 10_000 : false });
  const simulation = useQuery({ queryKey: ['simulation-options'], queryFn: simulationOptions, enabled: activeStage === 4 || activeStage === 5, retry: false, refetchInterval: activeStage === 4 || activeStage === 5 ? 10_000 : false });
  const context = workflowProjectId ? entries[workflowProjectId] : undefined;
  const quantizeMode = context?.quantize?.mode;
  const runMode = context?.run?.mode;
  function chooseQuantize(mode: QuantizeMode, origin: Entry<QuantizeMode>['origin'] = 'manual', jobId?: string) {
    if (!workflowProjectId) return;
    setEntries(previous => ({ ...previous, [workflowProjectId]: { ...previous[workflowProjectId], quantize: { mode, origin, jobId: jobId ?? (origin === 'manual' && previous[workflowProjectId]?.quantize?.mode === mode ? previous[workflowProjectId].quantize?.jobId : undefined) } } }));
    setEntryError(null);
  }
  function chooseRun(mode: RunMode, origin: Entry<RunMode>['origin'] = 'manual', jobId?: string) {
    if (!workflowProjectId) return;
    setEntries(previous => ({ ...previous, [workflowProjectId]: { ...previous[workflowProjectId], run: { mode, origin, jobId: jobId ?? (origin === 'manual' && previous[workflowProjectId]?.run?.mode === mode ? previous[workflowProjectId].run?.jobId : undefined) } } }));
    setEntryError(null);
  }
  useEffect(() => {
    const stage = activeStage === 3 ? 'quantize' : activeStage === 5 ? 'run' : null;
    if (!stage || !workflowProjectId || context?.[stage]) return;
    // An unfinished request has priority over restoring owned history.
    // Read only: the child owns its journal and explicit recovery acknowledgement.
    let recovery = false, simulationRecovery = false;
    try {
      const operation = stage === 'quantize' ? 'policy.quantize' : 'policy.run.replay';
      const cacheKey = [stage === 'quantize' ? 'native-quantization-attempt' : 'native-replay-attempt', workflowProjectId];
      recovery = !!queryClient.getQueryData<PolicyJobAttempt>(cacheKey) || !!storedAttempt(operation, workflowProjectId);
      if (stage === 'run') simulationRecovery = !!queryClient.getQueryData<PolicyJobAttempt>(['native-simulation-attempt', workflowProjectId]) || !!storedSimulationAttempt(workflowProjectId);
      if (recovery && simulationRecovery) { setEntryError({ projectId: workflowProjectId, stage: activeStage, message: 'Both observation replay and native simulation have unresolved requests. Choose either workflow to inspect its recovery record; neither request was retried.' }); return; }
    } catch {
      setEntryError({ projectId: workflowProjectId, stage: activeStage, message: 'Saved request recovery could not be read. Choose a workflow to inspect its history.' });
      return;
    }
    if (!recovery && !simulationRecovery && (!jobs.isSuccess || jobs.isError)) return;
    const next = stage === 'quantize'
      ? recovery ? { mode: 'native' as const, origin: 'recovery' as const } : initialQuantizeEntry(workflowProjectId, jobs.data!)
      : simulationRecovery ? { mode: 'native' as const, origin: 'recovery' as const } : recovery ? { mode: 'replay' as const, origin: 'recovery' as const } : initialRunEntry(workflowProjectId, jobs.data!);
    if (next) setEntries(previous => previous[workflowProjectId]?.[stage] ? previous : { ...previous, [workflowProjectId]: { ...previous[workflowProjectId], [stage]: next } });
    setEntryError(null);
  }, [activeStage, workflowProjectId, context, jobs.data, jobs.isSuccess, jobs.isError, queryClient]);
  const ownedJobs = (jobs.data ?? []).filter(job => job.project_id === workflowProjectId);
  const replayHistory = ownedJobs.some(job => runJobMode(job, workflowProjectId) === 'replay');
  const simulationHistory = ownedJobs.some(job => runJobMode(job, workflowProjectId) === 'native');
  const replayConfigured = options.isSuccess && !options.isError && options.data.runtimes.some(replayRuntime);
  const simulationConfigured = simulation.isSuccess && !simulation.isError && simulation.data.profiles.length > 0;
  const engineConfigured = options.isSuccess && !options.isError && options.data.runtimes.some(item => engineRuntime(item, activeStage === 4 ? 'Evaluate' : activeStage === 3 ? 'Quantize' : 'Run'));
  const entryPending = (activeStage === 3 && !quantizeMode) || (activeStage === 5 && !runMode);
  const entryFailed = jobs.isError || options.isError || (activeStage === 5 && simulation.isError);
  const recoveryError = entryError?.projectId === workflowProjectId && entryError.stage === activeStage ? entryError.message : null;
  const sortedJobs = [...(jobs.data ?? [])].filter(job => job.project_id === workflowProjectId).filter(isDatasetJob).sort((a, b) => b.created_at.localeCompare(a.created_at));
  const selectedJob = selectedJobId ? sortedJobs.find(job => job.id === selectedJobId) : sortedJobs[0];
  const connected = health.isSuccess && !health.isError;

  const stage = workspaceRoutes[activeStage];

  return {
    projectId, selectedJobId, activeStage, distillationModels, openSimulation, quantizeArtifact,
    distillTeachers, replayArtifact, simulationArtifact, modelInput, importModel, teachingMode,
    teachingCapture, trainingNavigation, openTrainingRun, openQuantizationRun, workflowNavigation, startTraining,
    settingsTab, datasetView, selectedLibrary, importLibrary, intakeSelection, health,
    projects, capabilities, jobs, project, projectMutation, workflowProjectId,
    context, quantizeMode, runMode, options, simulation, ownedJobs,
    replayHistory, simulationHistory, replayConfigured, simulationConfigured, engineConfigured, entryPending,
    entryFailed, recoveryError, sortedJobs, selectedJob, connected, stage,
    setSelectedJobId, setDistillationModels, setOpenSimulation, setQuantizeArtifact, setDistillTeachers, setReplayArtifact,
    setSimulationArtifact, setImportModel, setTeachingMode, setTeachingCapture, setOpenTrainingRun, setOpenQuantizationRun,
    setWorkflowNavigation, setStartTraining, setSettingsTab, setDatasetView, setImportLibrary, setIntakeSelection,
    openStage, navigateStage, startTrainingOnDataset, selectProject, openDataset, openModelWorkflow,
    reviewTeachingCapture, chooseQuantize, chooseRun, isSectionCurrent,
  };
}
