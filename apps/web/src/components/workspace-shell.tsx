import type { ReactNode } from 'react';
import Link from 'next/link';
import './workspace-shell.css';
import { Icon } from '@/components/icon';
import { ThemeToggle } from '@/components/theme-toggle';
import { apiReferenceUrl } from '@/lib/api';
import { publicPath } from '@/lib/base-path';

type WorkspaceShellProps = {
  navigation: ReactNode;
  children: ReactNode;
  breadcrumb: ReactNode;
  sidebarLabel?: string;
  skipLabel?: string;
  contentClassName?: string;
  sidebarFooter?: ReactNode;
  showGuideShortcut?: boolean;
  onHomeNavigate?: () => void;
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
  showGuideShortcut = true,
  onHomeNavigate,
}: WorkspaceShellProps) {
  return <div className="workspace">
    <a href="#main" className="skip-link">{skipLabel}</a>
    <aside className="sidebar" aria-label={sidebarLabel}>
      <Link className="brand" href="/dashboard/" prefetch={false} onNavigate={onHomeNavigate ? event => { event.preventDefault(); onHomeNavigate(); } : undefined} aria-label="Open Jensen workspace home"><span className="brand-mark"><Icon name="layers" size={21} /></span><span className="brand-copy">Open Jensen</span></Link>
      <div className="sidebar-content">{navigation}</div>
      <div className="sidebar-bottom">
        {sidebarFooter}
        <a className="sidebar-link" href={publicPath('/guide/')}><Icon name="book" size={18} /> Guide</a>
        <a className="sidebar-link" href={apiReferenceUrl}><Icon name="book" size={18} /> API reference</a>
        <div className="workspace-identity"><span className="workspace-avatar">OJ</span><div><strong>Local workspace</strong></div></div>
      </div>
    </aside>
    <div className="main-shell">
      <header className="topbar"><div className="breadcrumb">{breadcrumb}</div><div className="topbar-actions">{showGuideShortcut && <a className="mobile-guide-link" href={publicPath('/guide/')} aria-label="Workspace guide"><Icon name="book" size={18} /></a>}<a className="mobile-api-link" href={apiReferenceUrl} aria-label="API reference"><Icon name="book" size={18} /></a><ThemeToggle /></div></header>
      <main id="main" className={`main-content${contentClassName ? ` ${contentClassName}` : ''}`} tabIndex={-1}>
        {children}
      </main>
    </div>
  </div>;
}
