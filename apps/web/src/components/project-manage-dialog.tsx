'use client';

import { useEffect, useId, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, type Project } from '@/lib/api';
import { Icon } from '@/components/icon';

type Props = {
  projects: Project[];
  onChanged: (select?: string) => Promise<unknown> | void;
  onClose: () => void;
};

function counts(datasets: number, jobs: number) {
  const part = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`;
  return `${part(datasets, 'dataset')} · ${part(jobs, 'job')}`;
}

/** Remove projects from this app, and move in projects from other workspaces on this computer. */
export function ManageProjectsDialog({ projects, onChanged, onClose }: Props) {
  const id = useId();
  const dialog = useRef<HTMLDialogElement>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [working, setWorking] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const nearby = useQuery({ queryKey: ['projects', 'nearby'], queryFn: api.nearbyProjects, retry: false });
  useEffect(() => {
    dialog.current?.showModal();
  }, []);

  async function run(key: string, action: () => Promise<string>, select?: () => string | undefined) {
    setWorking(key);
    setError('');
    setNotice('');
    try {
      const message = await action();
      setConfirming(null);
      await onChanged(select?.());
      await nearby.refetch();
      setNotice(message);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'That did not work. Try again.');
    } finally {
      setWorking(null);
    }
  }
  const remove = (project: Project) =>
    run(`remove:${project.id}`, async () => {
      await api.removeProject(project.id);
      return `Removed ${project.name}.`;
    });
  const move = (workspace: string, project: { id: string; name: string }) => {
    let moved = '';
    return run(
      `move:${project.id}`,
      async () => {
        moved = (await api.moveProject(workspace, project.id)).id;
        return `Moved ${project.name} here.`;
      },
      () => moved,
    );
  };

  return (
    <dialog
      ref={dialog}
      className="project-create-dialog project-manage-dialog"
      aria-labelledby={`${id}-title`}
      onCancel={(event) => {
        if (working) event.preventDefault();
      }}
      onClose={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget && !working) dialog.current?.close();
      }}
    >
      <div className="project-manage-body">
        <header>
          <span className="project-dialog-mark">
            <Icon name="folder" size={20} />
          </span>
          <h2 id={`${id}-title`}>Projects</h2>
          <button
            type="button"
            className="project-dialog-close"
            aria-label="Close projects"
            disabled={!!working}
            onClick={() => dialog.current?.close()}
          >
            ×
          </button>
        </header>

        <section aria-labelledby={`${id}-here`}>
          <h3 id={`${id}-here`}>In this app</h3>
          {!projects.length && <p className="project-manage-note">No projects yet.</p>}
          <ul className="project-manage-list">
            {projects.map((project) => (
              <li key={project.id}>
                <span className="project-manage-name">{project.name}</span>
                {confirming === project.id ? (
                  <span className="project-manage-actions">
                    <span className="project-manage-warning">Datasets, jobs and recordings are deleted.</span>
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={!!working}
                      onClick={() => setConfirming(null)}
                    >
                      Keep
                    </button>
                    <button
                      type="button"
                      className="danger-button"
                      disabled={!!working}
                      onClick={() => void remove(project)}
                    >
                      {working === `remove:${project.id}` ? 'Removing…' : `Remove ${project.name}`}
                    </button>
                  </span>
                ) : (
                  <button
                    type="button"
                    className="text-button"
                    disabled={!!working}
                    aria-label={`Remove ${project.name}`}
                    onClick={() => setConfirming(project.id)}
                  >
                    Remove
                  </button>
                )}
              </li>
            ))}
          </ul>
        </section>

        <section aria-labelledby={`${id}-nearby`}>
          <h3 id={`${id}-nearby`}>On this computer</h3>
          {nearby.isPending && <p className="project-manage-note">Looking for other workspaces…</p>}
          {nearby.isError && <p className="project-create-error">{nearby.error.message}</p>}
          {nearby.data?.length === 0 && (
            <p className="project-manage-note">No other OpenJensen workspaces on this computer.</p>
          )}
          {nearby.data?.map((workspace) => (
            <div key={workspace.path} className="project-manage-workspace">
              <p className="project-manage-path">{workspace.label}</p>
              {workspace.message && <p className="project-manage-note">{workspace.message}</p>}
              <ul className="project-manage-list">
                {workspace.projects.map((project) => (
                  <li key={project.id}>
                    <span className="project-manage-name">
                      {project.name}
                      <small>{counts(project.datasets, project.jobs)}</small>
                    </span>
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={!workspace.available || project.here || !!working}
                      aria-label={project.here ? `${project.name} is already here` : `Move ${project.name} here`}
                      onClick={() => void move(workspace.path, project)}
                    >
                      {project.here ? 'Already here' : working === `move:${project.id}` ? 'Moving…' : 'Move here'}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </section>

        {error && (
          <p className="project-create-error" role="alert">
            {error}
          </p>
        )}
        {notice && (
          <p className="project-create-note" role="status">
            {notice}
          </p>
        )}
      </div>
    </dialog>
  );
}
