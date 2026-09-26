"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  api,
  type CloudConnection,
  type CloudConnections,
  type CloudProvider,
} from "@/lib/api";
import { Icon } from "@/components/icon";
import { CloudGpuSettingsPanel } from "@/components/cloud-gpu-settings";
import { HuggingFaceSettingsPanel } from "@/components/hf-settings";
import { LocalComputeSettingsPanel } from "@/components/local-compute-settings";
import "./cloud-connections.css";

const statusLabels: Record<CloudConnection["status"], string> = {
  disconnected: "Not connected",
  connected: "Connected",
  unverified: "Verify connection",
  setup_required: "Sign-in required",
  error: "Connection error",
};
const providerDocs = "https://docs.cloud.google.com/sdk/docs/install";
function ProviderMark({ provider }: { provider: CloudProvider }) {
  return (
    <span
      className={`cloud-provider-mark cloud-provider-${provider}`}
      aria-hidden="true"
    >
      <svg width="28" height="28" viewBox="0 0 24 24">
        <path
          fill="#4285f4"
          d="M21.6 12.2c0-.7-.1-1.4-.2-2.1H12v4h5.4a4.7 4.7 0 0 1-2 3v2.5h3.2c1.9-1.8 3-4.3 3-7.4Z"
        />
        <path
          fill="#34a853"
          d="M12 22c2.7 0 5-1 6.6-2.4L15.4 17a6 6 0 0 1-9-3.2H3.1v2.6A10 10 0 0 0 12 22Z"
        />
        <path
          fill="#fbbc05"
          d="M6.4 13.8a6 6 0 0 1 0-3.6V7.6H3.1a10 10 0 0 0 0 8.8l3.3-2.6Z"
        />
        <path
          fill="#ea4335"
          d="M12 6c1.5 0 2.8.5 3.9 1.5l2.9-2.8A10 10 0 0 0 3.1 7.6l3.3 2.6A6 6 0 0 1 12 6Z"
        />
      </svg>
    </span>
  );
}

function ConnectionDialog({
  connection,
  onClose,
  onConnected,
}: {
  connection: CloudConnection;
  onClose: () => void;
  onConnected: (connection: CloudConnection) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [projectId, setProjectId] = useState(
    connection.config && "project_id" in connection.config
      ? connection.config.project_id
      : "",
  );
  const [region, setRegion] = useState(
    connection.config?.region ?? "us-central1",
  );
  const [attempt, setAttempt] = useState<CloudConnection | null>(null);
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    dialog.current?.showModal();
    dialog.current?.querySelector<HTMLInputElement>("input")?.focus();
  }, []);
  const verify = useMutation({
    mutationFn: () =>
      api.connectCloud(connection.provider, {
        project_id: projectId.trim(),
        region: region.trim(),
      }),
    onSuccess: (result) => {
      if (result.status === "connected") {
        onConnected(result);
        dialog.current?.close();
      } else setAttempt(result);
    },
  });
  const commands = attempt?.setup_commands ?? ["gcloud auth login"];
  return (
    <dialog
      ref={dialog}
      className="cloud-dialog"
      aria-labelledby="cloud-dialog-title"
      onCancel={(event) => {
        if (verify.isPending) event.preventDefault();
      }}
      onClose={onClose}
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          setAttempt(null);
          verify.mutate();
        }}
      >
        <header>
          <ProviderMark provider={connection.provider} />
          <h2 id="cloud-dialog-title">Connect {connection.name}</h2>
          <button
            type="button"
            className="cloud-dialog-close"
            aria-label="Close connection dialog"
            disabled={verify.isPending}
            onClick={() => dialog.current?.close()}
          >
            ×
          </button>
        </header>
        <div className="cloud-dialog-fields">
          <label>
            Project ID
            <input
              autoFocus
              value={projectId}
              onChange={(event) => setProjectId(event.target.value)}
              required
              pattern="[a-z][a-z0-9\-]{4,28}[a-z0-9]"
              placeholder="my-robotics-project"
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
              disabled={verify.isPending}
            />
          </label>
          <label>
            Region
            <input
              value={region}
              onChange={(event) => setRegion(event.target.value)}
              required
              list={`cloud-regions-${connection.provider}`}
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
              disabled={verify.isPending}
            />
          </label>
          <datalist id={`cloud-regions-${connection.provider}`}>
            {[
              "us-central1",
              "us-east1",
              "us-west1",
              "europe-west1",
              "europe-west4",
              "asia-east1",
            ].map((item) => (
              <option key={item} value={item} />
            ))}
          </datalist>
        </div>
        <p className="cloud-auth-note">
          Uses your existing Google Cloud CLI sign-in on this host.
        </p>
        {attempt && (
          <p className="cloud-connection-error" role="alert">
            {attempt.message ?? "Connection could not be verified."}
          </p>
        )}
        {verify.error && (
          <p className="cloud-connection-error" role="alert">
            {verify.error.message}
          </p>
        )}
        <details
          className="cloud-signin"
          key={attempt?.status ?? "initial"}
          open={attempt?.status === "setup_required"}
        >
          <summary>Sign-in setup</summary>
          <div className="cloud-command-list">
            {commands.map((command) => (
              <code key={command}>{command}</code>
            ))}
          </div>
          <div className="cloud-signin-actions">
            <a href={providerDocs} target="_blank" rel="noreferrer">
              Install CLI <Icon name="external" size={12} />
            </a>
            <button
              type="button"
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(commands.join("\n"));
                  setCopied(true);
                } catch {
                  setCopied(false);
                }
              }}
            >
              {copied ? "Copied" : "Copy commands"}
            </button>
          </div>
        </details>
        <footer>
          <button
            type="button"
            className="secondary-button"
            disabled={verify.isPending}
            onClick={() => dialog.current?.close()}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="primary-button"
            disabled={verify.isPending || !projectId.trim() || !region.trim()}
          >
            {verify.isPending ? "Verifying…" : "Verify & save"}
          </button>
        </footer>
      </form>
    </dialog>
  );
}

export function CloudConnectionsPanel() {
  const client = useQueryClient();
  const [editing, setEditing] = useState<CloudProvider | null>(null);
  const [notice, setNotice] = useState("");
  const providers = useQuery({
    queryKey: ["cloud-connections"],
    queryFn: api.cloudConnections,
    retry: false,
  });
  const options = useQuery({
    queryKey: ["policy-options"],
    queryFn: api.policyOptions,
  });
  function update(result: CloudConnection) {
    client.setQueryData<CloudConnections>(
      ["cloud-connections"],
      (previous) => ({
        providers: (previous?.providers ?? []).map((item) =>
          item.provider === result.provider ? result : item,
        ),
      }),
    );
    void client.invalidateQueries({ queryKey: ["cloud-connections"] });
    void client.invalidateQueries({ queryKey: ["compute-settings"] });
    void client.invalidateQueries({ queryKey: ["policy-options"] });
    void client.invalidateQueries({ queryKey: ["augmentation-options"] });
  }
  const recheck = useMutation({
    mutationFn: api.recheckCloud,
    onSuccess: (result) => {
      update(result);
      setNotice(
        result.status === "connected"
          ? `${result.name} verified.`
          : (result.message ?? "Connection could not be verified."),
      );
    },
  });
  const disconnect = useMutation({
    mutationFn: api.disconnectCloud,
    onSuccess: (result) => {
      update(result);
      setNotice(`${result.name} disconnected.`);
    },
  });
  const current = providers.data?.providers.find(
    (item) => item.provider === editing,
  );
  const workers = (options.data?.runtimes ?? []).filter(
    (item) => item.training && item.device === "cuda" && item.execution !== "skypilot",
  );
  return (
    <div className="cloud-settings">
      <h2>Cloud providers</h2>
      <p className="cloud-auth-note">Google Cloud is available now. More providers are planned.</p>
      {providers.isPending && (
        <p className="cloud-loading" role="status">
          Loading providers…
        </p>
      )}
      {providers.error && (
        <div className="cloud-connection-error" role="alert">
          <p>
            {providers.error.message.includes("404")
              ? "Restart the app backend to enable cloud connections."
              : providers.error.message}
          </p>
          <button
            type="button"
            className="text-button"
            onClick={() => void providers.refetch()}
          >
            Retry
          </button>
        </div>
      )}
      <div className="cloud-provider-grid">
        {providers.data?.providers
          .filter((item) => item.provider === "gcp")
          .map((item) => {
            const busy =
              (recheck.isPending && recheck.variables === item.provider) ||
              (disconnect.isPending && disconnect.variables === item.provider);
            const scope = item.config?.project_id;
            return (
              <section
                className="cloud-provider-card"
                key={item.provider}
                aria-label={item.name}
              >
                <div className="cloud-provider-top">
                  <ProviderMark provider={item.provider} />
                  <h3>{item.name}</h3>
                  <span className={`cloud-status cloud-status-${item.status}`}>
                    {statusLabels[item.status]}
                  </span>
                </div>
                {item.config && (
                  <dl className="cloud-provider-details">
                    <div>
                      <dt>Project</dt>
                      <dd>{scope}</dd>
                    </div>
                    <div>
                      <dt>Region</dt>
                      <dd>{item.config.region}</dd>
                    </div>
                    {item.identity && (
                      <div>
                        <dt>Account</dt>
                        <dd>{item.identity.account ?? "—"}</dd>
                      </div>
                    )}
                  </dl>
                )}
                {item.message && (
                  <p className="cloud-card-message">{item.message}</p>
                )}
                <div className="cloud-provider-actions">
                  {item.config ? (
                    <>
                      <button
                        type="button"
                        className="secondary-button"
                        disabled={busy}
                        onClick={() => recheck.mutate(item.provider)}
                      >
                        {recheck.isPending &&
                        recheck.variables === item.provider
                          ? "Checking…"
                          : "Recheck"}
                      </button>
                      <button
                        type="button"
                        className="text-button"
                        disabled={busy}
                        onClick={() => setEditing(item.provider)}
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        className="text-button cloud-disconnect"
                        disabled={busy}
                        onClick={() => disconnect.mutate(item.provider)}
                      >
                        {disconnect.isPending &&
                        disconnect.variables === item.provider
                          ? "Disconnecting…"
                          : "Disconnect"}
                      </button>
                    </>
                  ) : (
                    <button
                      type="button"
                      className="secondary-button"
                      disabled={busy}
                      onClick={() => setEditing(item.provider)}
                    >
                      <Icon name="plus" size={14} />
                      Connect
                    </button>
                  )}
                  {item.checked_at && (
                    <time
                      dateTime={item.checked_at}
                      title={new Date(item.checked_at).toLocaleString()}
                    >
                      Checked{" "}
                      {new Date(item.checked_at).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </time>
                  )}
                </div>
              </section>
            );
          })}
      </div>
      {(recheck.error || disconnect.error) && (
        <p className="cloud-connection-error" role="alert">
          {recheck.error?.message ?? disconnect.error?.message}
        </p>
      )}
      {notice && (
        <p className="cloud-feedback" role="status">
          {notice}
        </p>
      )}
      <HuggingFaceSettingsPanel />
      <CloudGpuSettingsPanel />
      <LocalComputeSettingsPanel />
      <section className="cloud-workers" aria-labelledby="cloud-workers-title">
        <div className="cloud-workers-heading">
          <h2 id="cloud-workers-title">Existing GPU workers</h2>
          {workers.length > 0 && <span>{workers.length}</span>}
        </div>
        {workers.length ? (
          <div className="cloud-worker-list">
            {workers.map((worker) => (
              <div key={worker.id}>
                <Icon name="layers" size={19} />
                <strong>{worker.label}</strong>
                <span>
                  {worker.gpu_name ?? "NVIDIA GPU"}
                  {worker.gpu_memory_mib
                    ? ` · ${worker.gpu_memory_mib / 1024} GiB`
                    : ""}
                </span>
                <span className="cloud-status">Configured</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="cloud-auth-note">No GPU workers configured.</p>
        )}
        <details className="cloud-worker-setup">
          <summary>Local worker setup</summary>
          <p>
            For a GPU already installed on this machine, add a training worker in{" "}
            <code>FIREBIRD_RUNTIME_CONFIG</code> to make its GPU available in
            training.
          </p>
          <p>
            Setup: <code>docs/policy-workflow.md</code>
          </p>
        </details>
      </section>
      {current && (
        <ConnectionDialog
          key={current.provider}
          connection={current}
          onClose={() => setEditing(null)}
          onConnected={(result) => {
            update(result);
            setNotice(`${result.name} connected.`);
          }}
        />
      )}
    </div>
  );
}
