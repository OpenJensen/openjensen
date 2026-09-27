"use client";

import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiOrigin } from "@/lib/api";
import "./workbench-form.css";
import "./teaching-panel.css";

type Context = { session_id: string; episode_id: string | null; revision: number };
type Readiness = { broker_reachable: boolean; configured: boolean; busy: boolean; credential_source?: string; configuration_revision?: string | null };
type Saved = { saved: boolean; revision: string | null; voice_model: string | null; message: string };
type Receipt = { context: Context; requested_model: string; returned_model: string; elapsed_seconds: number; reported_cost_usd: number | null; frame: { step: number; rgb_sha256: string } | null };
type Result = { request_id: string; kind: "decision" | "perception"; proposal: { choice?: string; confidence?: number; probabilities?: Record<string, number>; summary?: string; uncertain?: boolean; receipt: Receipt }; choices: { id: string; description: string }[] };
type Attempt = { id: string; session: string; controller: AbortController };
function contextKey(context: Context | null) { return context ? JSON.stringify([context.session_id, context.episode_id, context.revision]) : "none"; }
const endpoint = `${apiOrigin}/api/v1/teaching/intelligence`;

export async function intelligenceRequest<T>(path: string, method = "GET", body?: object, controller = new AbortController()): Promise<T> {
  const timeout = setTimeout(() => controller.abort(), method === "POST" && path !== "/cancel" ? 20000 : 8000);
  try {
    const response = await fetch(endpoint + path, { method, body: body ? JSON.stringify(body) : undefined, headers: body ? { "Content-Type": "application/json" } : undefined, cache: "no-store", signal: controller.signal });
    const value = await response.json();
    if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "Intelligence request unavailable.");
    return value;
  } catch (error) {
    if (controller.signal.aborted) throw new Error("Request stopped or timed out. Completion and billing are unverified; no automatic retry was made.");
    throw error;
  } finally { clearTimeout(timeout); }
}

function cancel(attempt: Attempt) {
  attempt.controller.abort();
  return intelligenceRequest("/cancel", "POST", { request_id: attempt.id, session_id: attempt.session });
}

export function TeachingIntelligence({ context, online }: { context: Context | null; online: boolean }) {
  const [key, setKey] = useState("");
  const [model, setModel] = useState("");
  const [saving, setSaving] = useState(false);
  const savingRef = useRef(false);
  const [configurationMessage, setConfigurationMessage] = useState("");
  const [question, setQuestion] = useState("");
  const [consent, setConsent] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const active = useRef<Attempt | null>(null);
  const alive = useRef(true);
  const readiness = useQuery({ queryKey: ["teaching-intelligence-status"], queryFn: () => intelligenceRequest<Readiness>("/status"), refetchInterval: 3000, retry: false });
  const saved = useQuery({ queryKey: ["teaching-intelligence-settings"], queryFn: () => intelligenceRequest<Saved>("/settings"), retry: false });
  const identity = contextKey(context);
  const currentIdentity = useRef(identity); currentIdentity.current = identity;
  const currentOnline = useRef(online); currentOnline.current = online;
  useEffect(() => {
    setConsent(false);
    if (active.current) {
      const attempt = active.current; active.current = null;
      void cancel(attempt).catch(() => {});
      setPending(false); setError("Teaching context changed or disconnected. The request was cancelled; completion and billing remain unverified.");
    }
  }, [identity, online]);
  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; const attempt = active.current; active.current = null; if (attempt) void cancel(attempt).catch(() => {}); };
  }, []);
  async function save(remove = false) {
    if (savingRef.current) return;
    savingRef.current = true; setSaving(true); setConfigurationMessage("");
    const body = { openrouter_api_key: key, voice_model: model.trim() }; setKey("");
    try {
      const reply = await intelligenceRequest<Saved>("/settings", remove ? "DELETE" : "PUT", remove ? undefined : body);
      if (alive.current) { setConfigurationMessage(reply.message); await saved.refetch(); await readiness.refetch(); }
    } catch (problem) { if (alive.current) setConfigurationMessage(problem instanceof Error ? problem.message : "Could not update private settings."); }
    finally { savingRef.current = false; if (alive.current) setSaving(false); }
  }
  async function ask(kind: "decision" | "perception") {
    if (!context || !online || !consent || !question.trim() || active.current || !readiness.data?.configured || readiness.isError) return;
    const attempt: Attempt = { id: crypto.randomUUID(), session: context.session_id, controller: new AbortController() };
    active.current = attempt; setPending(true); setError(""); setResult(null);
    const boundIdentity = identity;
    try {
      const reply = await intelligenceRequest<Result>(`/${kind}`, "POST", { request_id: attempt.id, session_id: context.session_id, episode_id: context.episode_id, expected_revision: context.revision, prompt: question.trim(), consent: true }, attempt.controller);
      if (!alive.current || active.current !== attempt) return;
      if (!currentOnline.current || currentIdentity.current !== boundIdentity || reply.request_id !== attempt.id || reply.kind !== kind || contextKey(reply.proposal.receipt.context) !== boundIdentity) throw new Error("Teaching context changed; proposal withheld.");
      setResult(reply);
    } catch (problem) {
      if (alive.current && active.current === attempt) {
        setError(problem instanceof Error ? problem.message : "No confirmed result; no automatic retry was made.");
        void cancel(attempt).catch(() => {});
      }
    } finally { if (active.current === attempt) { active.current = null; if (alive.current) { setPending(false); setConsent(false); } } }
  }
  async function stop() {
    const attempt = active.current; if (!attempt) return;
    active.current = null; setPending(false); setConsent(false);
    setError("Cancellation requested. Completion and provider billing remain unverified.");
    try { await cancel(attempt); } catch { if (alive.current) setError("Cancellation could not be confirmed. The worker deadline remains in effect; billing is unverified."); }
  }
  const ready = readiness.isSuccess && readiness.data?.broker_reachable && readiness.data.configured && !readiness.data.busy;
  const old = !!result && (!online || contextKey(result.proposal.receipt.context) !== identity);
  return <section aria-labelledby="teaching-intelligence-title" className="panel teaching-advice">
    <div className="teaching-advice-heading"><h3 id="teaching-intelligence-title">Advice</h3><span className="metadata-badge">{ready ? "Configured · access unverified" : "Not ready"}</span></div>
    {!ready && <p role="status">{readiness.isPending ? "Checking intelligence broker…" : readiness.data?.busy ? "The intelligence broker is busy." : "Connect the advice broker and configure OpenRouter."}</p>}
    <details className="teaching-services"><summary>Configure optional services</summary><div className="teaching-services-body">
      <p>Save a private OpenRouter key on this application host. Saving does not contact a provider or configure LiveKit, Google Cloud, or a running voice agent.</p>
      <form className="workbench-form" onSubmit={event => { event.preventDefault(); void save(); }}>
        <div className="workbench-field-grid">
          <div className="workbench-field"><label htmlFor="teaching-openrouter-key">OpenRouter API key</label><input id="teaching-openrouter-key" type="password" autoComplete="off" value={key} maxLength={4096} onChange={event => setKey(event.target.value)} disabled={saving} /></div>
          <div className="workbench-field"><label htmlFor="teaching-voice-model">Voice chat model</label><input id="teaching-voice-model" placeholder="provider/model" value={model} maxLength={192} onChange={event => setModel(event.target.value)} disabled={saving} /></div>
        </div>
        <p className="field-help">Choose an explicit OpenRouter chat model for voice. Jev is a decision model, not the voice chat model.</p>
        <div className="workbench-actions"><button className="secondary-button" disabled={saving || key.length < 16 || !model.includes("/")}>Save private settings</button>
          {saved.data?.saved && <button type="button" className="secondary-button" disabled={saving} onClick={() => void save(true)}>Remove saved settings</button>}
        </div>
      </form>
      <p>{saved.data?.saved ? `Saved voice model: ${saved.data.voice_model}. Provider access is unverified.` : "No app-managed credentials saved."}</p>
      {configurationMessage && <p role="status">{configurationMessage}</p>}
      {saved.isError && <p role="alert">Saved settings could not be checked.</p>}
      <p className="field-help">The operator must explicitly point the separate broker and voice agent at <code>teaching-intelligence.json</code> in the application data directory using <code>FIREBIRD_TEACHING_INTELLIGENCE_FILE</code>. Existing environment credentials take precedence. Restart the voice agent after changing its model.</p>
      {saved.data?.saved && readiness.data?.configuration_revision !== saved.data.revision && <p role="status">The broker has not confirmed loading this saved revision. It may use its own environment or require configuration.</p>}
    </div></details>
    <div className="workbench-form teaching-advice-form">
      <div className="workbench-field"><label htmlFor="teaching-advice-question">Question for intelligence</label><input id="teaching-advice-question" value={question} maxLength={512} onChange={event => setQuestion(event.target.value)} disabled={pending} placeholder="What should I review before recording?" /></div>
      <label className="workbench-check"><input type="checkbox" checked={consent} onChange={event => setConsent(event.target.checked)} disabled={!online || !ready || pending} /><span>I consent to sending this instruction and session context to OpenRouter; camera questions also send one image. Provider charges may apply.</span></label>
      <div className="workbench-actions">
      <button className="secondary-button" disabled={!online || !ready || !consent || !question.trim() || pending} onClick={() => void ask("decision")}>Ask Jev for a review suggestion</button>
      <button className="secondary-button" disabled={!online || !ready || !consent || !question.trim() || pending} onClick={() => void ask("perception")}>Ask Mk1.5 about the camera</button>
      {pending && <button className="secondary-button" onClick={() => void stop()}>Cancel intelligence request</button>}
      </div>
    </div>
    {pending && <p role="status">Waiting for a bounded advisory response…</p>}{error && <p role="alert">{error}</p>}
    {result && <article aria-label="Intelligence suggestion"><h4>{old ? "Historical suggestion · context changed" : "Captured suggestion · generated advice"}</h4><p>{result.kind === "decision" ? result.choices.find(item => item.id === result.proposal.choice)?.description : result.proposal.summary}</p><p className="field-help">{result.proposal.receipt.returned_model} · revision {result.proposal.receipt.context.revision}{result.proposal.receipt.frame ? ` · captured frame ${result.proposal.receipt.frame.step}` : ""}. This is not a live observation, calibrated confidence, or task-success measurement.</p><details><summary>Request receipt</summary><dl className="dataset-facts"><div><dt>Request</dt><dd>{result.request_id}</dd></div><div><dt>Response time</dt><dd>{result.proposal.receipt.elapsed_seconds.toFixed(2)} s</dd></div><div><dt>Provider-reported cost</dt><dd>{result.proposal.receipt.reported_cost_usd === null ? "Not reported" : `$${result.proposal.receipt.reported_cost_usd}`}</dd></div></dl></details></article>}
  </section>;
}
