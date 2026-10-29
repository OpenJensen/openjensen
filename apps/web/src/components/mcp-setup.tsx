'use client';

import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Icon } from '@/components/icon';
import { publicPath } from '@/lib/base-path';
import './mcp-setup.css';

function McpSetupDialog({ onClose }: { onClose: () => void }) {
  const id = useId();
  const dialog = useRef<HTMLDialogElement>(null);
  const connection = document.getElementById('local-agent-connection');

  useEffect(() => {
    dialog.current?.showModal();
  }, []);

  function showPanel(event: React.MouseEvent<HTMLAnchorElement | HTMLButtonElement>) {
    const target =
      document.getElementById('local-agent-connection') ?? document.getElementById('agent-control-heading');
    if (!target) return;
    event.preventDefault();
    dialog.current?.close();
    target.scrollIntoView({ block: 'start' });
    target.focus({ preventScroll: true });
  }

  return (
    <dialog
      ref={dialog}
      className="mcp-setup-dialog"
      aria-labelledby={id}
      onClose={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) dialog.current?.close();
      }}
    >
      <div className="mcp-setup-content">
        <header>
          <h2 id={id}>MCP setup</h2>
          <button
            type="button"
            className="mcp-setup-close"
            aria-label="Close MCP setup"
            onClick={() => dialog.current?.close()}
          >
            ×
          </button>
        </header>
        <p>Connect a local Codex, Claude or another MCP client to a supervised AI teleoperation session.</p>
        <ol>
          <li>
            Open AI teleoperation, create a task, and select <strong>Local agent (MCP)</strong> as the agent provider.
          </li>
          <li>
            Select your saved arm and camera, complete physical preflight, then choose <strong>Arm AI session</strong>.
          </li>
          <li>
            In <strong>Local agent connection</strong>, enter the application’s loopback URL reachable from your agent.
            An existing local SSH forward also works.
          </li>
          <li>
            Choose <strong>Create local agent connection</strong>, then copy the Codex or Claude configuration into your
            client’s MCP settings. Use an installed <code>uvx</code>, or select an installed <code>jensen</code> CLI.
          </li>
          <li>
            Connect the MCP server in your client, choose <strong>Start task</strong> here, and ask the agent to inspect
            the pending request.
          </li>
        </ol>
        <p>
          Review and accept each proposed action here. Motion uses the same calibrated controller, limits and gripper
          handling as regular teleoperation.
        </p>
        <p>
          The connection credential belongs to this session. Stop or Revoke invalidates it. Your agent controls its
          model, costs and data sharing.
        </p>
        <footer>
          {connection ? (
            <button type="button" className="primary-button" onClick={showPanel}>
              Show session connection
            </button>
          ) : (
            <a className="primary-button" href={publicPath('/ai-teleoperation/')} onClick={showPanel}>
              Open AI teleoperation
            </a>
          )}
          <button type="button" className="secondary-button" onClick={() => dialog.current?.close()}>
            Close
          </button>
        </footer>
      </div>
    </dialog>
  );
}

export function McpSetup({ compact = false }: { compact?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        className={compact ? 'mobile-mcp-setup' : 'sidebar-link mcp-setup-trigger'}
        aria-label="MCP setup"
        aria-haspopup="dialog"
        onClick={() => setOpen(true)}
      >
        <Icon name="branch" size={compact ? 18 : 16} />
        {!compact && 'MCP setup'}
      </button>
      {open && createPortal(<McpSetupDialog onClose={() => setOpen(false)} />, document.body)}
    </>
  );
}
