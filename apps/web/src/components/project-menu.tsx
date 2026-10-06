"use client";

import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";
import { createPortal } from "react-dom";
import type { Project } from "@/lib/api";
import { Icon } from "@/components/icon";
import "./project-menu.css";

type ProjectMenuProps = {
  projects: Project[];
  value: string;
  disabled?: boolean;
  onSelect: (id: string) => void;
  onCreate: (name: string) => Promise<unknown>;
};

function CreateProjectDialog({ onCreate, onClose }: Pick<ProjectMenuProps, "onCreate"> & { onClose: () => void }) {
  const id = useId();
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const submitting = useRef(false);
  const [name, setName] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    dialog.current?.showModal();
    input.current?.focus();
  }, []);

  async function submit() {
    if (!name.trim() || submitting.current) return;
    submitting.current = true;
    setPending(true);
    setError("");
    try {
      await onCreate(name.trim());
      dialog.current?.close();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not create the project. Try again.");
    } finally {
      submitting.current = false;
      setPending(false);
    }
  }

  return <dialog ref={dialog} className="project-create-dialog" aria-labelledby={`${id}-title`}
    onCancel={event => { if (submitting.current) event.preventDefault(); }} onClose={onClose}
    onClick={event => { if (event.target === event.currentTarget && !submitting.current) dialog.current?.close(); }}>
    <form onSubmit={event => { event.preventDefault(); void submit(); }}>
      <header><span className="project-dialog-mark"><Icon name="folder" size={20} /></span><h2 id={`${id}-title`}>Create project</h2><button type="button" className="project-dialog-close" aria-label="Close project dialog" disabled={pending} onClick={() => dialog.current?.close()}>×</button></header>
      <label htmlFor={`${id}-name`}>Project name</label>
      <input ref={input} id={`${id}-name`} name="name" value={name} onChange={event => setName(event.target.value)} required maxLength={100} placeholder="Project name" disabled={pending} />
      {error && <p className="project-create-error" role="alert">{error}</p>}
      {pending && <p className="project-create-note" role="status">Creating project…</p>}
      <footer><button type="button" className="secondary-button" disabled={pending} onClick={() => dialog.current?.close()}>Cancel</button><button type="submit" className="primary-button" disabled={!name.trim() || pending}>{pending ? "Creating…" : "Create project"}</button></footer>
    </form>
  </dialog>;
}

export function ProjectMenu({ projects, value, onSelect, onCreate, disabled = false }: ProjectMenuProps) {
  const id = useId();
  const trigger = useRef<HTMLButtonElement>(null);
  const menu = useRef<HTMLDivElement>(null);
  const items = useRef<Array<HTMLButtonElement | null>>([]);
  const search = useRef({ text: "", time: 0 });
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [creating, setCreating] = useState(false);
  const [placement, setPlacement] = useState({ left: 16, top: 0, width: 260, height: 320 });
  const selected = projects.find(project => project.id === value);
  const expanded = open && !disabled && !creating;

  function close(restoreFocus = false) {
    setOpen(false);
    if (restoreFocus) trigger.current?.focus({ preventScroll: true });
  }
  function show(index = Math.max(0, projects.findIndex(project => project.id === value))) {
    if (disabled) return;
    search.current = { text: "", time: 0 };
    setActive(index);
    setOpen(true);
  }
  function choose(index: number) {
    if (index === projects.length) {
      close();
      setCreating(true);
    } else if (projects[index]) {
      onSelect(projects[index].id);
      close(true);
    }
  }
  function keyDown(event: KeyboardEvent) {
    const last = projects.length;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      setActive((active + (event.key === "ArrowDown" ? 1 : last)) % (last + 1));
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      setActive(event.key === "Home" ? 0 : last);
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      close(true);
    } else if (event.key === "Tab") {
      // Return to the trigger before the browser moves to the next sidebar control.
      close(true);
    } else if (event.key.length === 1 && event.key !== " " && !event.altKey && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      const now = Date.now();
      const text = (now - search.current.time < 600 ? search.current.text : "") + event.key.toLowerCase();
      const query = [...text].every(character => character === text[0]) ? text[0] : text;
      const labels = [...projects.map(project => project.name), "Create project"];
      const match = Array.from({ length: labels.length }, (_, offset) => (active + offset + 1) % labels.length)
        .find(index => labels[index].toLowerCase().startsWith(query));
      if (match !== undefined) setActive(match);
      search.current = { text, time: now };
    }
  }

  useLayoutEffect(() => {
    if (!expanded) return;
    function position() {
      const bounds = trigger.current?.getBoundingClientRect();
      if (!bounds) return;
      const width = Math.min(Math.max(bounds.width, 260), window.innerWidth - 32);
      const below = window.innerHeight - bounds.bottom - 22;
      const above = bounds.top - 22;
      const desired = Math.min(320, (projects.length + 1) * 44 + 22);
      const upwards = below < desired && above > below;
      const height = Math.max(44, Math.min(desired, upwards ? above : below));
      setPlacement({ left: Math.max(16, Math.min(bounds.left, window.innerWidth - width - 16)),
        top: upwards ? bounds.top - height - 6 : bounds.bottom + 6, width, height });
    }
    position();
    window.addEventListener("resize", position);
    window.addEventListener("scroll", position, true);
    return () => {
      window.removeEventListener("resize", position);
      window.removeEventListener("scroll", position, true);
    };
  }, [expanded, projects.length]);

  useEffect(() => {
    if (expanded) {
      const index = Math.min(active, projects.length);
      if (index !== active) setActive(index);
      items.current[index]?.focus({ preventScroll: true });
      items.current[index]?.scrollIntoView({ block: "nearest" });
    }
  }, [expanded, active, projects.length]);
  useEffect(() => { if (disabled) setOpen(false); }, [disabled]);
  useEffect(() => {
    if (!expanded) return;
    function outside(event: Event) {
      if (event.target instanceof Node && !trigger.current?.contains(event.target) && !menu.current?.contains(event.target)) close();
    }
    document.addEventListener("pointerdown", outside);
    document.addEventListener("focusin", outside);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("focusin", outside);
    };
  }, [expanded]);

  return <div className="project-menu">
    <button ref={trigger} type="button" className="project-menu-trigger" aria-label="Current project"
      aria-describedby={`${id}-value`}
      data-project-id={value} aria-haspopup="menu" aria-expanded={expanded} aria-controls={expanded ? `${id}-menu` : undefined}
      disabled={disabled} onClick={() => expanded ? close() : show()}
      onKeyDown={event => {
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          show(event.key === "ArrowUp" ? projects.length : undefined);
        } else if (event.key === "Escape") close();
      }}>
      <Icon name="folder" size={16} /><span id={`${id}-value`}>{disabled ? "Loading projects…" : selected?.name ?? "Select or create a project"}</span>
      <svg className="project-menu-chevron" width="15" height="15" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m5 7.5 5 5 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
    </button>
    {expanded && createPortal(<div ref={menu} id={`${id}-menu`} role="menu" aria-label="Projects" className="project-menu-popover"
      style={{ left: placement.left, top: placement.top, width: placement.width, maxHeight: placement.height }} onKeyDown={keyDown}>
      {!projects.length && <p className="project-menu-empty">No projects yet.</p>}
      {projects.map((project, index) => <button key={project.id} ref={element => { items.current[index] = element; }} type="button" role="menuitemradio"
        aria-checked={project.id === value} data-project-id={project.id} tabIndex={active === index ? 0 : -1} className="project-menu-item"
        onFocus={() => setActive(index)} onClick={() => choose(index)}>
        <Icon name="folder" size={16} /><span>{project.name}</span><span className="project-menu-check">{project.id === value && <Icon name="check" size={16} />}</span>
      </button>)}
      <div className="project-menu-divider" role="separator" />
      <button ref={element => { items.current[projects.length] = element; }} type="button" role="menuitem" tabIndex={active === projects.length ? 0 : -1}
        className="project-menu-item project-menu-create" onFocus={() => setActive(projects.length)} onClick={() => choose(projects.length)}>
        <Icon name="plus" size={16} /><span>Create project…</span>
      </button>
    </div>, document.body)}
    {creating && <CreateProjectDialog onCreate={onCreate} onClose={() => { setCreating(false); trigger.current?.focus({ preventScroll: true }); }} />}
  </div>;
}
