"use client";

import { useId, type ComponentProps, type ReactNode } from "react";
import { Icon } from "@/components/icon";
import "./workflow-choice-grid.css";

export type WorkflowChoice = {
  value: string;
  label: string;
  description?: string;
  meta?: string;
  icon?: ComponentProps<typeof Icon>["name"];
  disabled?: boolean;
};

export function WorkflowChoiceGrid({ name, label, value, options, onChange, disabled = false, description, emptyMessage, help }: {
  name: string;
  label: string;
  value: string;
  options: WorkflowChoice[];
  onChange: (value: string) => void;
  disabled?: boolean;
  description?: string;
  emptyMessage?: string;
  help?: ReactNode;
}) {
  const id = useId();
  return <fieldset className="workflow-choice-grid" disabled={disabled} aria-label={help ? label : undefined} aria-describedby={description ? `${id}-help` : undefined}>
    <legend>{label}{help}</legend>
    {description && <p id={`${id}-help`} className="workflow-choice-help">{description}</p>}
    <div className="workflow-choice-options">
      {options.map((option, index) => <label className="workflow-choice-card" key={option.value}>
        <input type="radio" name={`${id}-${name}`} value={option.value} aria-label={option.label}
          aria-describedby={option.description || option.meta ? `${id}-option-${index}` : undefined}
          checked={value === option.value} disabled={option.disabled}
          onChange={() => onChange(option.value)} />
        {option.icon && <span className="workflow-choice-icon"><Icon name={option.icon} size={21} /></span>}
        <span className="workflow-choice-copy">
          <strong>{option.label}</strong>
          {(option.description || option.meta) && <span className="workflow-choice-detail" id={`${id}-option-${index}`}>
            {option.description && <span>{option.description}</span>}
            {option.meta && <small>{option.meta}</small>}
          </span>}
        </span>
        <span className="workflow-choice-indicator" aria-hidden="true"><Icon name="check" size={12} /></span>
      </label>)}
    </div>
    {!options.length && <p className="workflow-choice-empty" role="status">{emptyMessage ?? "No options available yet."}</p>}
  </fieldset>;
}
