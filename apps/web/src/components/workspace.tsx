'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect, useState, type ReactNode } from 'react';
import Link from 'next/link';
import dynamic from 'next/dynamic';
import { useWorkbenchState } from '@/components/workspace-state';
import { WorkspaceContext } from '@/components/workspace-context';
import { Icon } from '@/components/icon';
import { WorkspaceShell } from '@/components/workspace-shell';
import { ApplicationUpdates } from '@/components/application-updates';
import { ProjectMenu } from '@/components/project-menu';
import { JobComputeSelectionNotice } from '@/components/job-compute-selection';
import { WorkJobs } from '@/components/work-jobs';
import { ErrorNotice } from '@/components/workspace-ui';
import { publicPath } from '@/lib/base-path';
import { guideSectionId } from '@/lib/guide-section';
import { isModule, workspaceRoutes, navigationGroups } from '@/lib/workspace-routes';

const DatasetsSection = dynamic(() => import('@/components/workspace-sections/datasets'));
const TrainingSection = dynamic(() => import('@/components/workspace-sections/training'));
const AugmentationSection = dynamic(() => import('@/components/workspace-sections/augmentation'));
const RealToSimSection = dynamic(() => import('@/components/workspace-sections/real-to-sim'));
const DecisionLabSection = dynamic(() => import('@/components/workspace-sections/decision-lab'));

function Workbench({ children }: { children: ReactNode }) {
  const state = useWorkbenchState();
  const {
    stage,
    activeStage,
    project,
    projects,
    projectId,
    selectProject,
    projectMutation,
    setSettingsTab,
    navigateStage,
    modules,
    moduleInstalled,
    health,
    capabilities,
    connected,
    workflowProjectId,
    jobs,
    entryPending,
    entryFailed,
    recoveryError,
    options,
    simulation,
    moduleJobView,
    workNavigationError,
    setModuleJobView,
    newWorkJob,
    openWorkJob,
  } = state;
  const workloadModule = [
    'training',
    'distillation',
    'quantization',
    'evaluation',
    'simulation',
    'augmentation',
    'real-to-sim',
    'datasets',
    'models',
    'teaching',
    'decision-lab',
  ].includes(activeStage);
  const hasJobHistory = ['training', 'distillation', 'quantization', 'evaluation', 'datasets', 'models'].includes(
    activeStage,
  );
  const showingJobs = workloadModule && !hasJobHistory && moduleJobView === 'jobs';
  const retainedComposer =
    ['augmentation', 'real-to-sim', 'decision-lab'].includes(activeStage) && !showingJobs ? activeStage : null;
  const [lastComposer, setLastComposer] = useState(retainedComposer);
  useEffect(() => {
    if (retainedComposer) setLastComposer(retainedComposer);
  }, [retainedComposer]);
  const composerStage = retainedComposer ?? lastComposer;
  const configurationDetour = ['settings', 'cloud-apis', 'modules'].includes(activeStage);
  const [datasetsVisited, setDatasetsVisited] = useState(activeStage === 'datasets');
  useEffect(() => {
    if (activeStage === 'datasets') setDatasetsVisited(true);
  }, [activeStage]);

  return (
    <WorkspaceContext.Provider value={state}>
      <WorkspaceShell
        headerActions={<ApplicationUpdates />}
        showGuideShortcut={false}
        onHomeNavigate={() => navigateStage('dashboard')}
        breadcrumb={
          <>
            <Icon name={stage.icon} size={18} />
            <strong>{stage.name}</strong>
            <span className="breadcrumb-divider">/</span>
            <span className="breadcrumb-project">{project?.name ?? 'No project selected'}</span>
          </>
        }
        navigation={
          <>
            <section className="projects-section" aria-labelledby="projects-heading">
              <div className="sidebar-section-label">
                <h2 id="projects-heading">Project</h2>
              </div>
              {projects.isPending && (
                <p className="sidebar-note" role="status">
                  Loading projects…
                </p>
              )}
              <ErrorNotice error={projects.error} />
              {projects.isError && (
                <button className="text-button" onClick={() => void projects.refetch()} disabled={projects.isFetching}>
                  Retry projects
                </button>
              )}
              <ProjectMenu
                projects={projects.data ?? []}
                value={projectId}
                disabled={projects.isPending}
                onSelect={selectProject}
                onCreate={(name) => projectMutation.mutateAsync(name)}
              />
            </section>
            <nav
              className="stage-navigation grouped-navigation"
              aria-label="Policy lifecycle"
              aria-busy={modules.isPending}
              onFocusCapture={(event) => {
                const navigation = event.currentTarget;
                const link = event.target.closest<HTMLElement>('.stage-button');
                if (!link || navigation.scrollWidth <= navigation.clientWidth) return;
                const reveal = () => {
                  if (!link.isConnected || document.activeElement !== link) return;
                  const viewport = navigation.getBoundingClientRect();
                  const bounds = link.getBoundingClientRect();
                  const left = viewport.left + navigation.clientLeft + 6;
                  const right = viewport.left + navigation.clientLeft + navigation.clientWidth - 6;
                  const delta =
                    bounds.left < left ? bounds.left - left : bounds.right > right ? bounds.right - right : 0;
                  if (delta) navigation.scrollBy({ left: delta, behavior: 'instant' });
                };
                reveal();
                // Reconcile the browser's own focus scroll without moving the page.
                requestAnimationFrame(reveal);
              }}
            >
              {navigationGroups.map((group) => {
                const items = group.items.filter((id) => !isModule(id) || moduleInstalled(id));
                return (
                  items.length > 0 && (
                    <div
                      className="navigation-group"
                      key={group.name ?? 'home'}
                      role={group.name ? 'group' : undefined}
                      aria-label={group.name}
                    >
                      {group.name && <p className="sidebar-section-label">{group.name}</p>}
                      <ul className="stage-list">
                        {items.map((id) => {
                          const item = workspaceRoutes[id];
                          return (
                            <li key={id}>
                              <Link
                                href={item.path}
                                prefetch
                                className={`stage-button${activeStage === id ? ' selected' : ''}`}
                                onNavigate={(event) => {
                                  event.preventDefault();
                                  if (id === 'settings') setSettingsTab('compute');
                                  navigateStage(id);
                                }}
                                aria-current={activeStage === id ? 'page' : undefined}
                              >
                                <Icon name={item.icon} size={18} />
                                <span>{item.name}</span>
                              </Link>
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )
                );
              })}
            </nav>
          </>
        }
      >
        <div className={`page-heading${activeStage === 'training' ? ' training-page-heading' : ''}`}>
          <h1>{stage.name}</h1>
          <div className="page-actions">
            <a className="page-guide" href={publicPath(`/guide/#${guideSectionId(stage.name)}`)}>
              <Icon name="book" size={16} />
              Guide
            </a>
          </div>
        </div>
        {!connected && !health.isPending && (
          <div className="connection-notice">
            <ErrorNotice error={health.error} />
            <button
              className="text-button"
              onClick={() => {
                void health.refetch();
                void projects.refetch();
                void capabilities.refetch();
              }}
            >
              Retry connection
            </button>
          </div>
        )}
        {capabilities.error && connected && (
          <div className="connection-notice">
            <ErrorNotice error={capabilities.error} />
            <button
              className="text-button"
              onClick={() => void capabilities.refetch()}
              disabled={capabilities.isFetching}
            >
              Retry capabilities
            </button>
          </div>
        )}
        {workNavigationError && (
          <p className="error-notice" role="alert">
            {workNavigationError}
          </p>
        )}
        {workloadModule &&
          !hasJobHistory &&
          !showingJobs &&
          !(activeStage === 'simulation' && state.runMode === 'engine') && (
            <div className="workbench-actions work-task-navigation">
              <button className="text-button" onClick={() => setModuleJobView('jobs')}>
                <Icon name="arrow" size={14} /> Back to jobs
              </button>
            </div>
          )}
        {showingJobs &&
          activeStage === 'simulation' &&
          (state.context?.run?.origin === 'recovery' || state.recoveryError) && (
            <section className="panel" aria-label="Pending run request">
              <p>A previous run request needs review before another request in that workflow.</p>
              <button className="secondary-button" onClick={() => setModuleJobView('workspace')}>
                Review pending run
              </button>
            </section>
          )}
        {showingJobs && moduleInstalled(activeStage as Parameters<typeof moduleInstalled>[0]) && (
          <WorkJobs
            key={`${workflowProjectId}:${activeStage}`}
            projectId={workflowProjectId}
            module={activeStage}
            onNew={newWorkJob}
            onOpen={openWorkJob}
          />
        )}
        {workloadModule && !showingJobs && (
          <JobComputeSelectionNotice
            key={`${workflowProjectId}:${activeStage}:${state.requestedJobOperation?.id ?? ''}`}
            projectId={workflowProjectId}
            module={activeStage}
            operation={state.requestedJobOperation?.id}
            onSettings={() => {
              setSettingsTab('compute');
              state.openStage('settings');
            }}
          />
        )}
        {composerStage === 'augmentation' &&
          moduleInstalled('augmentation') &&
          (activeStage === 'augmentation' || configurationDetour) && (
            <div hidden={activeStage !== 'augmentation' || showingJobs}>
              <AugmentationSection />
            </div>
          )}
        {composerStage === 'real-to-sim' &&
          moduleInstalled('real-to-sim') &&
          (activeStage === 'real-to-sim' || configurationDetour) && (
            <div hidden={activeStage !== 'real-to-sim' || showingJobs}>
              <RealToSimSection />
            </div>
          )}
        {composerStage === 'decision-lab' &&
          moduleInstalled('decision-lab') &&
          (activeStage === 'decision-lab' || configurationDetour) && (
            <div hidden={activeStage !== 'decision-lab' || showingJobs}>
              <DecisionLabSection />
            </div>
          )}
        {/* Retain intake drafts and pending uploads when another section opens. */}
        {moduleInstalled('datasets') && (datasetsVisited || activeStage === 'datasets') && (
          <div hidden={showingJobs}>
            <DatasetsSection />
          </div>
        )}
        {/* Configuration detours retain the exact unfinished training recipe. */}
        {moduleInstalled('training') && ['training', 'settings', 'cloud-apis', 'modules'].includes(activeStage) && (
          <div hidden={showingJobs}>
            <TrainingSection />
          </div>
        )}
        {(!isModule(activeStage) || moduleInstalled(activeStage)) &&
          !showingJobs &&
          entryPending &&
          (!workflowProjectId || jobs.isPending || entryFailed || recoveryError) && (
            <section className="panel" aria-label="Workflow selection status">
              <p role={entryFailed || recoveryError ? 'alert' : 'status'}>
                {!workflowProjectId
                  ? 'Select a project to see its workflow history.'
                  : (recoveryError ??
                    (entryFailed
                      ? 'Workflow availability or history could not be loaded. Choose a mode to inspect it, or retry these reads.'
                      : 'Loading this project’s workflow history and configured workers…'))}
              </p>
              {entryFailed && (
                <button
                  className="secondary-button"
                  onClick={() => {
                    void jobs.refetch();
                    void options.refetch();
                    if (activeStage === 'simulation') void simulation.refetch();
                  }}
                >
                  Retry workflow context
                </button>
              )}
            </section>
          )}
        {isModule(activeStage) && !moduleInstalled(activeStage) ? (
          <section className="panel" aria-label="Module setup">
            {modules.isPending ? (
              <p role="status">Checking module availability…</p>
            ) : modules.isError ? (
              <>
                <ErrorNotice error={modules.error} />
                <button className="secondary-button" onClick={() => void modules.refetch()}>
                  Retry modules
                </button>
              </>
            ) : (
              <>
                <p>Install or show this feature in Modules to open it.</p>
                <button
                  className="secondary-button"
                  onClick={() => {
                    navigateStage('modules');
                  }}
                >
                  Manage modules
                </button>
              </>
            )}
          </section>
        ) : showingJobs ? null : state.isSectionCurrent(activeStage) ? (
          children
        ) : (
          <p role="status">Opening section…</p>
        )}
      </WorkspaceShell>
    </WorkspaceContext.Provider>
  );
}

export function Workspace({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () => new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: true, staleTime: 5_000 } } }),
  );
  return (
    <QueryClientProvider client={queryClient}>
      <Workbench>{children}</Workbench>
    </QueryClientProvider>
  );
}
