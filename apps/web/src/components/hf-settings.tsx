"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type HuggingFaceConnectionStatus } from "@/lib/api";
import { Icon } from "@/components/icon";
import "./hf-settings.css";

export function HuggingFaceSettingsPanel() {
  const client = useQueryClient();
  const connection = useQuery({
    queryKey: ["huggingface-connection"],
    queryFn: api.huggingFaceStatus,
  });
  const [token, setToken] = useState("");
  const [pending, setPending] = useState<"save" | "remove" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<string | null>(null);

  function saved(status: HuggingFaceConnectionStatus, message: string) {
    client.setQueryData(["huggingface-connection"], status);
    setToken("");
    setFeedback(message);
  }

  // Keep the entered token out of React Query's mutation cache and browser storage.
  async function save() {
    setPending("save");
    setError(null);
    setFeedback(null);
    try {
      saved(await api.saveHuggingFaceToken(token.trim()), "Hugging Face token saved.");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not save the token.");
    } finally {
      setPending(null);
    }
  }

  async function remove() {
    setPending("remove");
    setError(null);
    setFeedback(null);
    try {
      saved(await api.removeHuggingFaceToken(), "Hugging Face token removed.");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not remove the token.");
    } finally {
      setPending(null);
    }
  }

  const status = connection.data;
  return (
    <section className="hf-settings" aria-labelledby="hf-settings-title">
      <header className="hf-settings-heading">
        <Icon name="folder" size={22} />
        <div>
          <h2 id="hf-settings-title">Hugging Face</h2>
          <p>Access tokens for model downloads</p>
        </div>
        <span className="hf-settings-badge">{status?.configured ? "Token saved" : "Optional"}</span>
      </header>
      <p className="hf-settings-note">
        Public models download without a token. Add one for gated or private models your account can access.
      </p>
      {status?.configured && (
        <p className="hf-settings-account">
          <strong>{status.username}</strong><span>{status.token_hint}</span>
        </p>
      )}
      <form onSubmit={(event) => { event.preventDefault(); void save(); }}>
        <div className="hf-settings-fields">
          <label>
            Hugging Face access token
            <input
              type="password"
              value={token}
              maxLength={512}
              autoComplete="off"
              autoCapitalize="none"
              autoCorrect="off"
              spellCheck={false}
              placeholder={status?.configured ? "Enter a replacement token" : "hf_…"}
              disabled={pending !== null}
              onChange={(event) => { setToken(event.target.value); setError(null); setFeedback(null); }}
            />
          </label>
          <button type="submit" className="secondary-button" disabled={!token.trim() || pending !== null}>
            {pending === "save" ? "Verifying…" : status?.configured ? "Replace token" : "Save token"}
          </button>
          {status?.configured && (
            <button type="button" className="text-button" disabled={pending !== null} onClick={() => void remove()}>
              {pending === "remove" ? "Removing…" : "Remove token"}
            </button>
          )}
        </div>
        <p className="hf-settings-note">
          Stored privately on this application server. The saved token is never shown here.
          {" "}<a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer">Create a token <Icon name="external" size={11} /></a>
        </p>
        {status?.message && !status.configured && <p className="hf-settings-error" role="alert">{status.message}</p>}
        {connection.error && <p className="hf-settings-error" role="alert">{connection.error.message}</p>}
        {error && <p className="hf-settings-error" role="alert">{error}</p>}
        {feedback && <p className="hf-settings-feedback" role="status">{feedback}</p>}
      </form>
    </section>
  );
}
