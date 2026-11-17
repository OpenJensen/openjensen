'use client';

import { useEffect, useId, useRef, useState } from 'react';
import type { Project } from '@/lib/api';
import { Icon } from '@/components/icon';
import { ManageProjectsDialog } from '@/components/project-manage-dialog';
import './project-menu.css';

type ProjectMenuProps = {
  projects: Project[];
  value: string;
  disabled?: boolean;
  onSelect: (id: string) => void;
  onCreate: (name: string) => Promise<unknown>;
  onChanged: (select?: string) => Promise<unknown> | void;
};

function CreateProjectDialog({ onCreate, onClose }: Pick<ProjectMenuProps, 'onCreate'> & { onClose: () => void }) {
  const id = useId();
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const submitting = useRef(false);
  const [name, setName] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    dialog.current?.showModal();
    input.current?.focus();
  }, []);

  async function submit() {
    if (!name.trim() || submitting.current) return;
    submitting.current = true;
    setPending(true);
    setError('');
    try {
      await onCreate(name.trim());
      dialog.current?.close();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Could not create the project. Try again.');
    } finally {
      submitting.current = false;
      setPending(false);
    }
  }

  return (
    <dialog
      ref={dialog}
      className="project-create-dialog"
      aria-labelledby={`${id}-title`}
      onCancel={(event) => {
        if (submitting.current) event.preventDefault();
      }}
      onClose={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget && !submitting.current) dialog.current?.close();
      }}
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
      >
        <header>
          <span className="project-dialog-mark">
            <Icon name="folder" size={20} />
          </span>
          <h2 id={`${id}-title`}>Create project</h2>
          <button
            type="button"
            className="project-dialog-close"
            aria-label="Close project dialog"
            disabled={pending}
            onClick={() => dialog.current?.close()}
          >
            ×
          </button>
        </header>
        <label htmlFor={`${id}-name`}>Project name</label>
        <input
          ref={input}
          id={`${id}-name`}
          name="name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          required
          maxLength={100}
          placeholder="Project name"
          disabled={pending}
        />
        {error && (
          <p className="project-create-error" role="alert">
            {error}
          </p>
        )}
        {pending && (
          <p className="project-create-note" role="status">
            Creating project…
          </p>
        )}
        <footer>
          <button type="button" className="secondary-button" disabled={pending} onClick={() => dialog.current?.close()}>
            Cancel
          </button>
          <button type="submit" className="primary-button" disabled={!name.trim() || pending}>
            {pending ? 'Creating…' : 'Create project'}
          </button>
        </footer>
      </form>
    </dialog>
  );
}

type ProjectView = 'closed' | 'choose' | 'create' | 'manage';

/** Every project as a card, the open one marked. One click opens a project. */
function ChooseProjectDialog({
  projects,
  value,
  onSelect,
  onClose,
}: Pick<ProjectMenuProps, 'projects' | 'value' | 'onSelect'> & { onClose: (next: ProjectView) => void }) {
  const id = useId();
  const dialog = useRef<HTMLDialogElement>(null);
  const next = useRef<ProjectView>('closed');
  useEffect(() => {
    dialog.current?.showModal();
    const current = dialog.current?.querySelector<HTMLElement>('[aria-checked="true"]');
    (current ?? dialog.current?.querySelector<HTMLElement>('.project-choice'))?.focus();
  }, []);
  function finish(view: ProjectView) {
    next.current = view;
    dialog.current?.close();
  }

  return (
    <dialog
      ref={dialog}
      className="project-create-dialog project-choose-dialog"
      aria-labelledby={`${id}-title`}
      onClose={() => onClose(next.current)}
      onClick={(event) => {
        if (event.target === event.currentTarget) dialog.current?.close();
      }}
    >
      <div className="project-choose-body">
        <header>
          <h2 id={`${id}-title`}>Projects</h2>
          <button
            type="button"
            className="project-dialog-close"
            aria-label="Close projects"
            onClick={() => dialog.current?.close()}
          >
            ×
          </button>
        </header>
        {!projects.length && <p className="project-choose-empty">No projects yet.</p>}
        <div className="project-choose-grid">
          {projects.length > 0 && (
            <div className="project-choose-list" role="radiogroup" aria-label="Open project">
              {projects.map((project) => (
                <button
                  key={project.id}
                  type="button"
                  role="radio"
                  aria-checked={project.id === value}
                  data-project-id={project.id}
                  className="project-choice"
                  onClick={() => {
                    if (project.id !== value) onSelect(project.id);
                    finish('closed');
                  }}
                >
                  <span className="project-choice-mark">
                    <Icon name={project.id === value ? 'check' : 'folder'} size={20} />
                  </span>
                  <span className="project-choice-name">{project.name}</span>
                </button>
              ))}
            </div>
          )}
          <button type="button" className="project-choice project-choice-new" onClick={() => finish('create')}>
            <span className="project-choice-mark">
              <Icon name="plus" size={20} />
            </span>
            <span className="project-choice-name">New project</span>
          </button>
        </div>
        <footer>
          <button type="button" className="text-button project-choose-manage" onClick={() => finish('manage')}>
            Remove or bring in projects
          </button>
        </footer>
      </div>
    </dialog>
  );
}

/** The open project, shown as a card at the top of the sidebar. */
export function ProjectMenu({ projects, value, onSelect, onCreate, onChanged, disabled = false }: ProjectMenuProps) {
  const id = useId();
  const trigger = useRef<HTMLButtonElement>(null);
  const [view, setView] = useState<ProjectView>('closed');
  const selected = projects.find((project) => project.id === value);
  const restoreFocus = () => trigger.current?.focus({ preventScroll: true });

  return (
    <div className="project-menu">
      <button
        ref={trigger}
        type="button"
        className="project-card"
        aria-label="Current project"
        aria-describedby={`${id}-value`}
        data-project-id={value}
        aria-haspopup="dialog"
        aria-expanded={view === 'choose' && !disabled}
        disabled={disabled}
        onClick={() => setView('choose')}
      >
        <span className="project-card-mark">
          <Icon name="folder" size={20} />
        </span>
        <span className="project-card-text">
          <span id={`${id}-value`} className="project-card-name">
            {disabled ? 'Loading projects…' : (selected?.name ?? 'No project yet')}
          </span>
          {!disabled && <span className="project-card-action">{selected ? 'Switch project' : 'Create a project'}</span>}
        </span>
      </button>
      {view === 'choose' && !disabled && (
        <ChooseProjectDialog
          projects={projects}
          value={value}
          onSelect={onSelect}
          onClose={(next) => {
            setView(next);
            if (next === 'closed') restoreFocus();
          }}
        />
      )}
      {view === 'manage' && (
        <ManageProjectsDialog
          projects={projects}
          onChanged={onChanged}
          onClose={() => {
            setView('closed');
            restoreFocus();
          }}
        />
      )}
      {view === 'create' && (
        <CreateProjectDialog
          onCreate={onCreate}
          onClose={() => {
            setView('closed');
            restoreFocus();
          }}
        />
      )}
    </div>
  );
}
