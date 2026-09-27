"use client";

import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiOrigin } from "@/lib/api";
import { basePath } from "@/lib/base-path";

type Status = { configured: boolean; available: boolean; busy: boolean; message: string };
type Result = { selected_id: string; request_sha256: string; revision: string; scores: { id: string; logit: number; relative_weight: number; tokens: number }[]; timing_ms: { load: number; score: number } };
const endpoint = `${apiOrigin}${basePath}/api/v1/decision`;

async function request<T>(path: string, signal: AbortSignal, body?: object): Promise<T> {
  const response = await fetch(endpoint + path, { signal, method: body ? "POST" : "GET", headers: body ? { "Content-Type": "application/json" } : undefined, body: body ? JSON.stringify(body) : undefined, cache: "no-store" });
  if (!response.ok) throw new Error(response.status === 409 ? "The local scorer is busy. Wait and try again manually." : "No score was accepted. Check the operator configuration and shorten text to fit the 512-token limit.");
  return response.json();
}

export function DecisionPanel() {
  const status = useQuery({ queryKey: ["decision-status"], queryFn: ({ signal }) => request<Status>("/status", signal), retry: false, staleTime: 0 });
  const [state, setState] = useState("");
  const [instructions, setInstructions] = useState("Compare these criteria with the state. Return advisory scores only.");
  const [criteria, setCriteria] = useState(["", ""]);
  const [result, setResult] = useState<{ receipt: Result; criteria: string[] } | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const active = useRef<AbortController | null>(null);
  useEffect(() => () => active.current?.abort(), []);
  const valid = state.trim().length > 0 && instructions.trim().length > 0 && criteria.every(text => text.trim()) && new Set(criteria.map(text => text.trim())).size === criteria.length;
  const ready = status.isSuccess && status.data.configured && status.data.available;
  function edited() { setResult(null); setError(""); }
  async function score() {
    if (!ready || !valid || active.current) return;
    const controller = new AbortController(); active.current = controller;
    const timer = setTimeout(() => controller.abort(), 50_000);
    setPending(true); setError(""); setResult(null);
    const submitted = [...criteria];
    try {
      const receipt = await request<Result>("/score", controller.signal, { schema_version: 1, state, instructions, criteria: submitted.map((text, index) => ({ id: `criterion-${index + 1}`, text })) });
      if (!controller.signal.aborted) setResult({ receipt, criteria: submitted });
    } catch (cause) {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Scoring failed.");
      else setError("Scoring stopped. No result was accepted; there is no automatic retry.");
    } finally {
      clearTimeout(timer); active.current = null; setPending(false);
      void status.refetch();
    }
  }
  return <section className="panel" aria-labelledby="decision-title">
    <h2 id="decision-title">Local decision advisory</h2>
    <p>Compare text criteria using Muose-50M on the application host CPU. Scoring never starts a job, calls a tool, or moves a robot.</p>
    <p className="warning-box"><strong>Experimental: only 3 of 6 workflow examples were correct</strong> in a small hand-authored fixture (5 of 6 banking examples). This is not a robotics benchmark. Relative weights are uncalibrated, not probabilities of correctness.</p>
    <p>Model: muose/Muose-50M-Decision · Noncommercial CC-BY-NC-SA-4.0; the operator must explicitly accept the license.</p>
    <p role="status">{status.isPending ? "Checking local configuration…" : status.isError ? "Local scorer status is unavailable." : status.data?.message}</p>
    <button type="button" className="secondary-button" disabled={pending || status.isFetching} onClick={() => void status.refetch()}>Refresh scorer status</button>
    <form onSubmit={event => { event.preventDefault(); void score(); }}>
      <fieldset disabled={pending}>
        <legend>Manual text experiment</legend>
        <label htmlFor="decision-state">State to compare</label>
        <textarea id="decision-state" rows={3} maxLength={8000} value={state} onChange={event => { edited(); setState(event.target.value); }} />
        <label htmlFor="decision-instructions">Comparison instructions</label>
        <textarea id="decision-instructions" rows={2} maxLength={2000} value={instructions} onChange={event => { edited(); setInstructions(event.target.value); }} />
        {criteria.map((text, index) => <div key={index}>
          <label htmlFor={`decision-criterion-${index}`}>Criterion {index + 1}</label>
          <textarea id={`decision-criterion-${index}`} rows={2} maxLength={2000} value={text} onChange={event => { edited(); setCriteria(criteria.map((item, position) => position === index ? event.target.value : item)); }} />
          {criteria.length > 2 && <button type="button" className="secondary-button" aria-label={`Remove criterion ${index + 1}`} onClick={() => { edited(); setCriteria(criteria.filter((_, position) => position !== index)); }}>Remove</button>}
        </div>)}
        <button type="button" className="secondary-button" disabled={criteria.length >= 8} onClick={() => { edited(); setCriteria([...criteria, ""]); }}>Add criterion</button>
        <p>Supply 2–8 unique criteria. Each state + instructions + criterion must fit 512 model tokens; excessive text is rejected, never truncated.</p>
      </fieldset>
      <button type="submit" disabled={!ready || !valid || pending}>{pending ? "Scoring locally…" : "Score criteria"}</button>
      {pending && <button type="button" className="secondary-button" onClick={() => active.current?.abort()}>Stop scoring</button>}
    </form>
    {error && <p role="alert" className="error-notice">{error}</p>}
    {result && <section aria-label="Advisory score result">
      <h3>Advisory result</h3>
      <p>Highest relative score: {result.criteria[result.receipt.scores.findIndex(item => item.id === result.receipt.selected_id)]}. Review it yourself; no action was taken.</p>
      <ol>{result.receipt.scores.map((row, index) => <li key={row.id}><strong>{result.criteria[index]}</strong>: {(row.relative_weight * 100).toFixed(2)}% relative weight (uncalibrated); raw score {row.logit.toFixed(4)}; {row.tokens} tokens.</li>)}</ol>
      <p>CPU load {result.receipt.timing_ms.load.toFixed(1)} ms; scoring {result.receipt.timing_ms.score.toFixed(1)} ms. These are observations, not a latency guarantee.</p>
      <details><summary>Verified receipt identity</summary><p style={{ overflowWrap: "anywhere" }}>Model revision: {result.receipt.revision}<br />Request SHA256: {result.receipt.request_sha256}</p></details>
    </section>}
  </section>;
}
