'use client';

import type { ReactNode } from 'react';

/**
 * One choice among a few, shown all at once instead of hidden in a dropdown.
 * `Segmented` suits short labels (2° / 5°, Auto / 10 fps); `ChoiceCards` suits
 * things with a name and a state (an arm, a camera, a trained policy).
 */
export type Choice<T extends string | number> = {
  value: T;
  label: ReactNode;
  /** A second line on a card: what this choice is or why it cannot be picked. */
  detail?: ReactNode;
  /** A short state shown at the card's edge, such as "Connected". */
  badge?: ReactNode;
  disabled?: boolean;
  /** Accessible name when the label is not plain text. */
  name?: string;
};

type GroupProps<T extends string | number> = {
  /** Accessible name of the group; say what is being chosen. */
  label: string;
  value: T | null | undefined;
  options: Choice<T>[];
  onChange: (value: T) => void;
  disabled?: boolean;
  className?: string;
};

export function Segmented<T extends string | number>({
  label,
  value,
  options,
  onChange,
  disabled,
  className,
}: GroupProps<T>) {
  return (
    <div className={`segmented${className ? ` ${className}` : ''}`} role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <button
          key={String(option.value)}
          type="button"
          role="radio"
          data-value={String(option.value)}
          aria-checked={option.value === value}
          aria-label={option.name}
          disabled={disabled || option.disabled}
          onClick={() => {
            if (option.value !== value) onChange(option.value);
          }}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function ChoiceCards<T extends string | number>({
  label,
  value,
  options,
  onChange,
  disabled,
  className,
}: GroupProps<T>) {
  return (
    <div className={`choice-cards${className ? ` ${className}` : ''}`} role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <button
          key={String(option.value)}
          type="button"
          role="radio"
          data-value={String(option.value)}
          className="choice-card"
          aria-checked={option.value === value}
          aria-label={option.name}
          disabled={disabled || option.disabled}
          onClick={() => {
            if (option.value !== value) onChange(option.value);
          }}
        >
          <span className="choice-card-mark" aria-hidden="true" />
          <span className="choice-card-text">
            <strong>{option.label}</strong>
            {option.detail && <small>{option.detail}</small>}
          </span>
          {option.badge && <span className="choice-card-badge">{option.badge}</span>}
        </button>
      ))}
    </div>
  );
}
