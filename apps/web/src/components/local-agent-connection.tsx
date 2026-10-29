'use client';

import { useEffect, useState } from 'react';
import { agentApi, type AgentSession } from '@/lib/agent-api';
import { apiOrigin } from '@/lib/api';

export function LocalAgentConnection({
  session,
  owned,
  onSession,
}: {
  session: AgentSession;
  owned: boolean;
  onSession: (session: AgentSession) => void;
}) {
  const [token, setToken] = useState('');
  const [sourceCommit, setSourceCommit] = useState<string | null>(null);
  const [launcher, setLauncher] = useState<'uvx' | 'jensen'>('uvx');
  const [executable, setExecutable] = useState('uvx');
  const [apiUrl, setApiUrl] = useState('http://127.0.0.1:8000');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  useEffect(() => {
    const url = new URL(apiOrigin || window.location.origin);
    if (['127.0.0.1', 'localhost', '[::1]'].includes(url.hostname)) setApiUrl(url.origin);
    setToken('');
  }, [session.id]);
  useEffect(() => {
    if (!session.control_armed) setToken('');
  }, [session.control_armed]);
  const args = ['agent-mcp', '--api-url', apiUrl, '--session-id', session.id];
  const launchArgs =
    launcher === 'uvx' && sourceCommit
      ? [
          '--python',
          '3.14',
          '--from',
          'git+https://github.com/OpenJensen/openjensen.git@' + sourceCommit + '#subdirectory=packages/core',
          'jensen',
          ...args,
        ]
      : args;
  const ready = launcher === 'jensen' || !!sourceCommit;
  const claude = JSON.stringify(
    {
      mcpServers: {
        openjensen: {
          command: executable,
          args: launchArgs,
          env: { OPENJENSEN_AGENT_TOKEN: token },
        },
      },
    },
    null,
    2,
  );
  const codex = [
    '[mcp_servers.openjensen]',
    'command = ' + JSON.stringify(executable),
    'args = [' + launchArgs.map((arg) => JSON.stringify(arg)).join(', ') + ']',
    '',
    '[mcp_servers.openjensen.env]',
    'OPENJENSEN_AGENT_TOKEN = ' + JSON.stringify(token),
  ].join('\n');

  async function connect() {
    setBusy(true);
    setError('');
    setMessage('');
    try {
      const access = await agentApi.localAccess(session.id);
      if (access.session_id !== session.id) throw new Error('Connection belongs to another session.');
      setToken(access.access_token);
      setSourceCommit(access.source_commit ?? null);
      setMessage('Connection created for this session. Copy a configuration, then start the task.');
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Local agent connection failed.');
    } finally {
      setBusy(false);
    }
  }
  async function disconnect() {
    setBusy(true);
    setError('');
    setMessage('');
    try {
      onSession(await agentApi.localRevoke(session.id));
      setToken('');
      setMessage('Local agent access revoked and planning paused. The operator page still owns the local hold.');
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Local agent disconnection was not acknowledged.');
    } finally {
      setBusy(false);
    }
  }
  async function copy(value: string, name: string) {
    try {
      await navigator.clipboard.writeText(value);
      setMessage(name + ' configuration copied.');
    } catch {
      setError('Clipboard unavailable. Expand the configuration and copy it manually.');
    }
  }
  const pending = ['observing', 'planning', 'executing', 'stopping'].includes(session.status);
  return (
    <section
      id="local-agent-connection"
      className="agent-provider-card"
      aria-label="Local agent connection"
      tabIndex={-1}
    >
      <h3>Local agent (MCP)</h3>
      <p>
        Connect your Codex or Claude through OpenJensen’s MCP adapter. It can read this session’s state and admitted
        images, propose bounded actions, and Stop. Review and accept each action here. With gripper position targets
        enabled, use propose_gripper_position with base_link coordinates in metres; OpenJensen solves the next bounded
        waypoint.
      </p>
      <p>
        Motion uses OpenJensen’s calibrated controller and existing gripper handling. The local agent owns its model,
        costs and data sharing; model usage is not measured here. OpenJensen bounds planning turns, motion and time.
      </p>
      <label className="workbench-field">
        Agent’s local application URL
        <input value={apiUrl} disabled={busy} onChange={(event) => setApiUrl(event.target.value)} />
      </label>
      <p>Use a loopback URL reachable from the agent, including an existing local SSH forward.</p>
      <label className="workbench-field">
        MCP launcher
        <select
          value={launcher}
          disabled={busy}
          onChange={(event) => {
            const next = event.target.value as 'uvx' | 'jensen';
            setLauncher(next);
            setExecutable(next);
          }}
        >
          <option value="uvx">uvx · load the matching OpenJensen version</option>
          <option value="jensen">Installed jensen CLI</option>
        </select>
      </label>
      <label className="workbench-field">
        MCP launcher executable
        <input
          value={executable}
          disabled={busy}
          placeholder="uvx or its full path"
          onChange={(event) => setExecutable(event.target.value)}
        />
      </label>
      <p>
        The uvx launcher downloads the pinned CLI and Python 3.14 on first use. Use the executable’s full path if your
        agent cannot find it.
      </p>
      <div className="workbench-actions">
        <button
          type="button"
          className="secondary-button"
          disabled={busy || !owned || pending}
          onClick={() => void connect()}
        >
          Create local agent connection
        </button>
        <button type="button" className="secondary-button" disabled={busy || !owned} onClick={() => void disconnect()}>
          Revoke local agent access
        </button>
      </div>
      {token && (
        <>
          <div className="workbench-actions">
            <button
              type="button"
              className="secondary-button"
              disabled={!ready || !executable.trim()}
              onClick={() => void copy(codex, 'Codex')}
            >
              Copy Codex configuration
            </button>
            <button
              type="button"
              className="secondary-button"
              disabled={!ready || !executable.trim()}
              onClick={() => void copy(claude, 'Claude')}
            >
              Copy Claude configuration
            </button>
          </div>
          <p>The configuration contains a private session credential. Stop or Revoke invalidates it.</p>
          {!ready && <p role="status">This build has no source revision. Select an installed jensen CLI.</p>}
          <details>
            <summary>Show connection configuration</summary>
            <label className="workbench-field">
              Codex config.toml entry
              <textarea readOnly rows={8} value={codex} />
            </label>
            <label className="workbench-field">
              Claude MCP JSON
              <textarea readOnly rows={12} value={claude} />
            </label>
          </details>
        </>
      )}
      <p>
        After connecting the MCP server, click Start task and ask your agent to inspect OpenJensen’s pending request.
      </p>
      {message && <p role="status">{message}</p>}
      {error && <p role="alert">{error}</p>}
    </section>
  );
}
