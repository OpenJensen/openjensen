'use client';

import { useId } from 'react';
import './gpu-picker.css';

export type GpuChoice = {
  id: string;
  label: string;
  memory?: string;
  description?: string;
  accelerator?: string;
};

type GpuPickerProps = {
  value: string;
  onChange: (value: string) => void;
  choices: GpuChoice[];
  disabled?: boolean;
  label?: string;
  /** The accelerator or choice id the current recipe fits best; its card says so. */
  suggested?: string | null;
};

function GpuMark() {
  return (
    <svg width="21" height="21" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="5" y="5" width="14" height="14" rx="3" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <rect x="9" y="9" width="6" height="6" rx="1.5" fill="currentColor" opacity=".65" />
    </svg>
  );
}

/** Every machine a job can run on, shown at once: one card each, with its memory. */
export function GpuPicker({ value, onChange, choices, disabled = false, label = 'GPU', suggested }: GpuPickerProps) {
  const id = useId();
  const labelId = `${id}-label`;
  const tone = (choice: GpuChoice) => {
    const hardware = choice.accelerator ?? choice.id;
    return hardware === 'A100-80GB'
      ? 'a100'
      : ['L4', 'T4', 'A100'].includes(hardware)
        ? hardware.toLowerCase()
        : 'local';
  };
  return (
    <div className="gpu-picker">
      <span className="gpu-picker-label" id={labelId}>
        {label}
      </span>
      <div className="gpu-picker-cards" role="radiogroup" aria-labelledby={labelId} data-value={value}>
        {choices.map((choice, index) => {
          const recommended = !!suggested && (choice.accelerator ?? choice.id) === suggested;
          const details = [choice.memory, choice.description, recommended ? 'Suggested' : null].filter(Boolean);
          return (
            <button
              key={choice.id}
              type="button"
              role="radio"
              className="gpu-picker-card"
              aria-checked={choice.id === value}
              aria-label={choice.label}
              aria-describedby={details.length ? `${id}-details-${index}` : undefined}
              data-value={choice.id}
              data-gpu={tone(choice)}
              disabled={disabled}
              onClick={() => {
                if (choice.id !== value) onChange(choice.id);
              }}
            >
              <span className="gpu-picker-mark">
                <GpuMark />
              </span>
              <span className="gpu-picker-copy">
                <strong>{choice.label}</strong>
                {choice.description && <small>{choice.description}</small>}
              </span>
              {recommended && <span className="gpu-picker-suggested">Suggested</span>}
              {choice.memory && <span className="gpu-picker-memory">{choice.memory}</span>}
              {details.length > 0 && (
                <span className="visually-hidden" id={`${id}-details-${index}`}>
                  {details.join('. ')}
                </span>
              )}
            </button>
          );
        })}
        {!choices.length && (
          <p className="gpu-picker-empty">{label === 'Compute' ? 'No compute available' : 'No GPU available'}</p>
        )}
      </div>
    </div>
  );
}
