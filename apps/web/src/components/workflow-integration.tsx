import type { ComponentProps } from 'react';
import { Icon } from './icon';
import './workflow-integration.css';

/** Names the configured implementation without adding a pretend provider choice. */
export function WorkflowIntegration({ label, name, meta, status, icon }: {
  label: string;
  name: string;
  meta?: string;
  status?: string;
  icon: ComponentProps<typeof Icon>['name'];
}) {
  return <div className="workflow-integration" role="group" aria-label={label}>
    <span className="workflow-integration-label">{label}</span>
    <div className="workflow-integration-card">
      <span className="workflow-integration-icon"><Icon name={icon} size={22} /></span>
      <span className="workflow-integration-copy"><strong>{name}</strong>{meta && <small>{meta}</small>}</span>
      {status && <span className="workflow-integration-status">{status}</span>}
    </div>
  </div>;
}
