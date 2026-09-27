"use client";

import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";
import "./gpu-picker.css";

export type GpuChoice = {
  id: string;
  label: string;
  memory?: string;
  description?: string;
};

type GpuPickerProps = {
  value: string;
  onChange: (value: string) => void;
  choices: GpuChoice[];
  disabled?: boolean;
  label?: string;
};

function GpuMark() {
  return (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="5" y="5" width="14" height="14" rx="3" stroke="currentColor" strokeWidth="1.5" />
      <path d="M9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <rect x="9" y="9" width="6" height="6" rx="1.5" fill="currentColor" opacity=".65" />
    </svg>
  );
}

/** Select-only combobox: focus stays on the trigger while its active option changes. */
export function GpuPicker({ value, onChange, choices, disabled = false, label = "GPU" }: GpuPickerProps) {
  const id = useId();
  const labelId = `${id}-label`;
  const listId = `${id}-options`;
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const options = useRef<Array<HTMLLIElement | null>>([]);
  const search = useRef({ text: "", time: 0 });
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const [placement, setPlacement] = useState({ above: false, height: 320 });
  const selectedIndex = choices.findIndex(choice => choice.id === value);
  const selected = choices[selectedIndex];
  const expanded = open && !disabled && choices.length > 0;
  const active = Math.min(Math.max(activeIndex, 0), choices.length - 1);
  const tone = (choice?: GpuChoice) => ["L4", "T4", "A100"].includes(choice?.id ?? "") ? choice!.id.toLowerCase() : "local";

  function show(index = selectedIndex >= 0 ? selectedIndex : 0) {
    if (disabled || !choices.length) return;
    setActiveIndex(index);
    setOpen(true);
    search.current = { text: "", time: 0 };
  }

  function close(restoreFocus = false) {
    setOpen(false);
    search.current = { text: "", time: 0 };
    if (restoreFocus) trigger.current?.focus({ preventScroll: true });
  }

  function select(index: number) {
    const choice = choices[index];
    if (disabled || !choice) return;
    onChange(choice.id);
    close(true);
  }

  function keyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (disabled || !choices.length) return;
    switch (event.key) {
      case "ArrowDown":
      case "ArrowUp":
        event.preventDefault();
        if (!expanded) show();
        else setActiveIndex(Math.max(0, Math.min(choices.length - 1, active + (event.key === "ArrowDown" ? 1 : -1))));
        return;
      case "Home":
      case "End":
        event.preventDefault();
        show(event.key === "Home" ? 0 : choices.length - 1);
        return;
      case "Enter":
      case " ":
        event.preventDefault();
        if (expanded) select(active);
        else show();
        return;
      case "Escape":
        if (expanded) {
          event.preventDefault();
          event.stopPropagation();
          close(true);
        }
        return;
      case "Tab":
        close();
        return;
    }
    if (event.key.length !== 1 || event.altKey || event.ctrlKey || event.metaKey) return;
    event.preventDefault();
    const now = Date.now();
    const text = (now - search.current.time < 600 ? search.current.text : "") + event.key.toLowerCase();
    const query = [...text].every(character => character === text[0]) ? text[0] : text;
    const start = expanded ? active : Math.max(selectedIndex, 0);
    const match = Array.from({ length: choices.length }, (_, offset) => (start + offset + 1) % choices.length)
      .find(index => choices[index].label.toLowerCase().startsWith(query));
    if (!expanded) show(match ?? Math.max(selectedIndex, 0));
    else if (match !== undefined) setActiveIndex(match);
    search.current = { text, time: now };
  }

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  useEffect(() => {
    if (!expanded) return;
    function outside(event: PointerEvent) {
      if (event.target instanceof Node && !root.current?.contains(event.target)) setOpen(false);
    }
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [expanded]);

  useLayoutEffect(() => {
    if (!expanded) return;
    function position() {
      const bounds = trigger.current?.getBoundingClientRect();
      if (!bounds) return;
      const bottomRoom = window.innerHeight - bounds.bottom - 16;
      const topRoom = bounds.top - 16;
      const desired = Math.min(320, choices.length * 76 + 16);
      const above = bottomRoom < desired && topRoom > bottomRoom;
      const height = Math.max(76, Math.min(320, above ? topRoom : bottomRoom));
      setPlacement(previous => previous.above === above && previous.height === height ? previous : { above, height });
    }
    position();
    window.addEventListener("resize", position);
    window.addEventListener("scroll", position, true);
    return () => {
      window.removeEventListener("resize", position);
      window.removeEventListener("scroll", position, true);
    };
  }, [expanded, choices.length]);

  useEffect(() => {
    if (expanded) options.current[active]?.scrollIntoView({ block: "nearest" });
  }, [expanded, active]);

  return (
    <div className="gpu-picker" ref={root} onBlur={event => {
      if (!event.currentTarget.contains(event.relatedTarget)) close();
    }}>
      <label className="gpu-picker-label" id={labelId} htmlFor={`${id}-trigger`}>{label}</label>
      <div className="gpu-picker-control">
        <button
          ref={trigger}
          id={`${id}-trigger`}
          type="button"
          role="combobox"
          className="gpu-picker-trigger"
          value={value}
          data-gpu={tone(selected)}
          aria-labelledby={labelId}
          aria-haspopup="listbox"
          aria-expanded={expanded}
          aria-controls={expanded ? listId : undefined}
          aria-activedescendant={expanded ? `${id}-option-${active}` : undefined}
          disabled={disabled || !choices.length}
          onClick={() => expanded ? close() : show()}
          onKeyDown={keyDown}
        >
          <span className="gpu-picker-mark"><GpuMark /></span>
          <span className="gpu-picker-value">{selected?.label ?? (value || "Choose a GPU")}</span>
          {selected?.memory && <span className="gpu-picker-memory">{selected.memory}</span>}
          <svg className="gpu-picker-chevron" width="16" height="16" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m5 7.5 5 5 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>
        </button>
        {expanded && (
          <ul
            id={listId}
            role="listbox"
            aria-labelledby={labelId}
            className={`gpu-picker-menu${placement.above ? " gpu-picker-menu-above" : ""}`}
            style={{ maxHeight: placement.height }}
          >
            {choices.map((choice, index) => (
              <li
                key={choice.id}
                ref={element => { options.current[index] = element; }}
                id={`${id}-option-${index}`}
                role="option"
                aria-label={choice.label}
                aria-describedby={choice.memory || choice.description ? `${id}-details-${index}` : undefined}
                aria-selected={choice.id === value}
                data-gpu={tone(choice)}
                className={`gpu-picker-option${index === active ? " is-active" : ""}`}
                onPointerMove={() => setActiveIndex(index)}
                onPointerDown={event => event.preventDefault()}
                onClick={() => select(index)}
              >
                <span className="gpu-picker-mark"><GpuMark /></span>
                <span className="gpu-picker-copy">
                  <strong>{choice.label}</strong>
                  {choice.description && <small>{choice.description}</small>}
                </span>
                {choice.memory && <span className="gpu-picker-memory">{choice.memory}</span>}
                {(choice.memory || choice.description) && <span className="visually-hidden" id={`${id}-details-${index}`}>{[choice.memory, choice.description].filter(Boolean).join(". ")}</span>}
                <span className="gpu-picker-selected" aria-hidden="true">
                  {choice.id === value && <svg width="15" height="15" viewBox="0 0 20 20" fill="none"><path d="m4.5 10 3.5 3.5 7.5-7.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg>}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
