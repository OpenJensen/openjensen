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
import { ErrorNotice } from '@/components/workspace-ui';
import { publicPath } from '@/lib/base-path';
import { guideSectionId } from '@/lib/guide-section';
import { isModule, workspaceRoutes, navigationGroups } from '@/lib/workspace-routes';

const DatasetsSection = dynamic(() => import('@/components/workspace-sections/datasets'));
const TrainingSection = dynamic(() => import('@/components/workspace-sections/training'));

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
  } = state;
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
            >
              {navigationGroups.map((group) => {
                // Uninstalled modules stay reachable by URL; only an open one is listed.
                const items = group.items.filter((id) => !isModule(id) || moduleInstalled(id) || activeStage === id);
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
                                prefetch={false}
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
        {/* Retain intake drafts and pending uploads when another section opens. */}
        {(datasetsVisited || activeStage === 'datasets') && <DatasetsSection />}
        {/* A compute-settings detour retains the exact unfinished training recipe. */}
        {(activeStage === 'training' || activeStage === 'settings') && <TrainingSection />}
        {entryPending && (!workflowProjectId || jobs.isPending || entryFailed || recoveryError) && (
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
        {state.isSectionCurrent(activeStage) ? children : <p role="status">Opening workspace section…</p>}
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
