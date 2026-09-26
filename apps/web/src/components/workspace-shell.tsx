import type { ReactNode } from 'react';
import { Icon } from '@/components/icon';
import { ThemeToggle } from '@/components/theme-toggle';
import { apiReferenceUrl } from '@/lib/api';

type WorkspaceShellProps = {
  navigation: ReactNode;
  children: ReactNode;
  breadcrumb: ReactNode;
  sidebarLabel?: string;
  skipLabel?: string;
  contentClassName?: string;
  sidebarFooter?: ReactNode;
};

// Both the workspace and reference use this shell so navigation, branding,
// accessibility, and appearance controls evolve together.
export function WorkspaceShell({
  navigation,
  children,
  breadcrumb,
  sidebarLabel = 'Workspace navigation',
  skipLabel = 'Skip to workspace',
  contentClassName,
  sidebarFooter,
}: WorkspaceShellProps) {
  return <div className="workspace">
    <a href="#main" className="skip-link">{skipLabel}</a>
    <aside className="sidebar" aria-label={sidebarLabel}>
      <a className="brand" href="/" aria-label="Firebird workspace home"><span className="brand-mark"><Icon name="layers" size={21} /></span><span>Firebird<span className="brand-subtitle">Robotics workspace</span></span></a>
      <div className="sidebar-content">{navigation}</div>
      <div className="sidebar-bottom">
        {sidebarFooter}
        <a className="sidebar-link" href={apiReferenceUrl}><Icon name="book" size={18} /> API reference</a>
        <div className="workspace-identity"><span className="workspace-avatar">F</span><div><strong>Local workspace</strong><span>Firebird · v0.1</span></div></div>
      </div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><div className="breadcrumb">{breadcrumb}</div><div className="topbar-actions"><a className="mobile-api-link" href={apiReferenceUrl} aria-label="API reference"><Icon name="book" size={18} /></a><ThemeToggle /></div></header>
      <main id="main" className={`main-content${contentClassName ? ` ${contentClassName}` : ''}`} tabIndex={-1}>
        {children}
      </main>
    </div>
  </div>;
}
