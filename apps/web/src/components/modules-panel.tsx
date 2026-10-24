'use client';

import { Icon } from '@/components/icon';
import { StatusBadge } from '@/components/ui/status-badge';
import { useWorkspace } from '@/components/workspace-context';
import { workspaceRoutes } from '@/lib/workspace-routes';
import './cloud-connections.css';

export function ModulesPanel() {
  const { modules } = useWorkspace();
  return (
    <section className="cloud-settings" aria-labelledby="modules-heading">
      <h2 id="modules-heading">Modules</h2>
      {modules.isPending && (
        <p className="cloud-loading" role="status">
          Loading modules…
        </p>
      )}
      {modules.error && (
        <p className="cloud-connection-error" role="alert">
          {modules.error.message}
        </p>
      )}
      <ul className="cloud-worker-list modules-list">
        {modules.data?.map((item) => (
          <li key={item.id}>
            <Icon name={workspaceRoutes[item.id].icon} size={18} />
            <strong>{item.name}</strong>
            {!item.installed && <span>{item.setup}</span>}
            <StatusBadge tone={item.installed ? 'success' : 'neutral'}>
              {item.installed ? 'Installed' : 'Not installed'}
            </StatusBadge>
          </li>
        ))}
      </ul>
    </section>
  );
}
