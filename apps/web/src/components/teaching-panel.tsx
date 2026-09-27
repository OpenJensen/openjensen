"use client";

import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { TeachingIntelligence } from "./teaching-intelligence";
import { TeachingPreview } from "./teaching-preview";
import { apiOrigin } from "@/lib/api";
import type { Room } from "livekit-client";

type State = { mode: string; episode_id: string | null; revision: number; session_id: string | null; instruction: string; outcome: string; steps: number; sim_time: number; joints: string[]; state_rad: number[] | null; fault: string | null };
type Connection = { connected: boolean; state: State | null; message: string | null; voice_configured: boolean };
type Receipt = { command_id: string; status: "queued" | "executing" | "acknowledged" | "rejected"; error?: string };
const endpoint = `${apiOrigin}/api/v1/teaching`;
async function request<T>(path: string, body?: object): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), path === "/voice/join" ? 15000 : 8000);
  try {
    const response = await fetch(endpoint + path, { method: body ? "POST" : "GET", headers: body ? { "Content-Type": "application/json" } : undefined, body: body ? JSON.stringify(body) : undefined, cache: "no-store", signal: controller.signal });
    const value = await response.json();
    if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "Teaching request failed.");
    return value;
  } catch (error) {
    if (controller.signal.aborted) throw new Error(body ? "No response arrived. The request outcome is unverified; inspect the executor before retrying." : "Teaching connection timed out.");
    throw error;
  } finally { clearTimeout(timeout); }
}

export function TeachingPanel() {
  const [instruction, setInstruction] = useState("");
  const [joint, setJoint] = useState("");
  const [receiptId, setReceiptId] = useState("");
  const [receiptExpired, setReceiptExpired] = useState(false);
  const voiceSession = useRef<string | null>(null);
  const [voiceState, setVoiceState] = useState("Disconnected");
  const [voiceError, setVoiceError] = useState("");
  const [agentPresent, setAgentPresent] = useState(false);
  const voiceReadiness = useQuery({ queryKey: ["teaching-voice-status"], queryFn: () => request<{ broker_reachable: boolean; configuration_present: boolean; dependencies_present: boolean; message: string }>("/voice/status"), refetchInterval: 3000, retry: false });
  const room = useRef<Room | null>(null);
  const joining = useRef(false);
  const generation = useRef(0);
  const mounted = useRef(true);
  const media = useRef<HTMLDivElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connection = useQuery({ queryKey: ["teaching-state"], queryFn: () => request<Connection>("/state"), refetchInterval: 1000, retry: false });
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => { const tick = setInterval(() => setNow(Date.now()), 500); return () => clearInterval(tick); }, []);
  const state = connection.data?.state;
  const lastState = useRef<State | null>(null);
  if (state?.session_id) lastState.current = state;
  const previewState = state?.session_id ? state : lastState.current;
  const online = connection.isSuccess && connection.data?.connected && !!state?.session_id && now - connection.dataUpdatedAt <= 5000;
  const receipt = useQuery({ queryKey: ["teaching-receipt", receiptId], queryFn: () => request<Receipt>(`/commands/${receiptId}`), enabled: !!receiptId && !receiptExpired, refetchInterval: query => !receiptExpired && !query.state.error && ["queued", "executing"].includes(query.state.data?.status ?? "queued") ? 250 : false, retry: false });
  const command = useMutation({
    mutationFn: async ({ operation, args = {} }: { operation: string; args?: object }) => {
      const current = await request<Connection>("/state");
      if (!current.connected || !current.state) throw new Error("Teaching executor is disconnected.");
      if (!state?.session_id || current.state.session_id !== state.session_id) { void connection.refetch(); throw new Error("The simulator session changed. Review its state before issuing another command."); }
      return request<Receipt>("/commands", { command_id: crypto.randomUUID(), session_id: current.state.session_id, episode_id: current.state.episode_id, expected_revision: current.state.revision, operation, arguments: args });
    },
    onSuccess: result => { setReceiptExpired(false); setReceiptId(result.command_id); void connection.refetch(); },
  });
  useEffect(() => {
    if (!receiptId || receipt.data?.status === "acknowledged" || receipt.data?.status === "rejected") return;
    const deadline = setTimeout(() => setReceiptExpired(true), 10000);
    return () => clearTimeout(deadline);
  }, [receiptId, receipt.data?.status]);
  useEffect(() => {
    if (!voiceSession.current || (online && state?.session_id === voiceSession.current)) return;
    generation.current += 1; joining.current = false; voiceSession.current = null;
    if (timer.current) clearTimeout(timer.current);
    const previous = room.current; room.current = null;
    void previous?.disconnect(); media.current?.replaceChildren(); setVoiceState("Disconnected"); setAgentPresent(false);
  }, [online, state?.session_id]);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false; generation.current += 1; joining.current = false;
      if (timer.current) clearTimeout(timer.current);
      void room.current?.disconnect(); room.current = null;
    };
  }, []);
  async function leave() {
    generation.current += 1; joining.current = false; voiceSession.current = null;
    if (timer.current) clearTimeout(timer.current);
    const current = room.current; room.current = null;
    await current?.disconnect(); media.current?.replaceChildren(); setVoiceState("Disconnected"); setAgentPresent(false);
  }
  async function join() {
    if (!state?.session_id || room.current || joining.current) return;
    joining.current = true; voiceSession.current = state.session_id;
    const attempt = ++generation.current;
    const live = () => mounted.current && generation.current === attempt;
    setVoiceError(""); setVoiceState("Connecting");
    let current: Room | null = null;
    try {
      const access = await request<{ url: string; token: string; expires_in_seconds: number }>("/voice/join", { session_id: state.session_id });
      const { Room, RoomEvent, ParticipantKind } = await import("livekit-client");
      if (!live()) return;
      current = new Room({ adaptiveStream: true, dynacast: true });
      room.current = current;
      const updateAgent = () => { if (live() && current) setAgentPresent([...current.remoteParticipants.values()].some(participant => participant.kind === ParticipantKind.AGENT)); };
      current.on(RoomEvent.ParticipantConnected, updateAgent);
      current.on(RoomEvent.ParticipantDisconnected, updateAgent);
      current.on(RoomEvent.ParticipantAttributesChanged, updateAgent);
      current.on(RoomEvent.TrackSubscribed, track => {
        if (!live()) return;
        const element = track.attach();
        element.setAttribute("aria-label", track.kind === "video" ? "Live simulator camera" : "Voice agent audio");
        element.style.maxWidth = "100%";
        media.current?.append(element);
      });
      current.on(RoomEvent.TrackUnsubscribed, track => track.detach().forEach(element => element.remove()));
      current.on(RoomEvent.Disconnected, () => { if (room.current === current) { generation.current += 1; joining.current = false; if (timer.current) clearTimeout(timer.current); timer.current = null; voiceSession.current = null; if (mounted.current) { setVoiceState("Disconnected"); setAgentPresent(false); setVoiceError("Voice connection closed. Reconnect to continue."); } media.current?.replaceChildren(); room.current = null; } });
      await current.connect(access.url, access.token);
      if (!live()) { await current.disconnect(); return; }
      await current.localParticipant.setMicrophoneEnabled(true);
      if (!live()) { await current.disconnect(); return; }
      updateAgent(); setVoiceState("Microphone connected");
      timer.current = setTimeout(() => void leave(), Math.min(access.expires_in_seconds, 300) * 1000);
    } catch {
      await current?.disconnect();
      if (!live()) return;
      if (room.current === current) room.current = null;
      voiceSession.current = null;
      setVoiceState("Disconnected"); setAgentPresent(false); setVoiceError("Voice could not connect. Check microphone permission and the server's LiveKit/OpenRouter settings.");
    } finally { if (generation.current === attempt) joining.current = false; }
  }
  return <section className="panel" aria-labelledby="teaching-title">
    <div className="panel-title"><h2 id="teaching-title">Teach in simulation</h2><span className="metadata-badge">{online ? "Executor reachable" : "Executor disconnected"}</span></div>
    <p>Record what the simulator actually does: camera, state, applied action, timing and corrections. Voice supplies instructions; recorded actions supply training data.</p>
    {!online && <p role="status">{connection.data?.message ?? "Checking teaching executor…"} The existing rollout monitor does not provide teaching control.</p>}
    <details className="teaching-setup"><summary>How to connect a simulator</summary><p>Teaching needs a running OPEN JENSEN teaching executor, its private control token and an operator-configured connection on the application host. Google Cloud readiness and previous rollout logs do not create this connection.</p><ol><li>Start the teaching executor with your reviewed scene and recording settings.</li><li>Have the application operator configure its loopback connection and private control token.</li><li>Wait for Executor reachable and a fresh camera observation before recording. Voice is optional and requires its separate configured teaching room.</li></ol><p>This page does not start a VM or simulator. See <code>workers/teaching/README.md</code> in the installed repository for the operator setup instructions.</p></details>
    <dl className="cloud-run-facts"><div><dt>Executor connection</dt><dd>{online ? "Reachable" : connection.isPending ? "Checking" : "Not connected"}</dd></div><div><dt>Voice service</dt><dd>{voiceReadiness.isSuccess && voiceReadiness.data?.broker_reachable && voiceReadiness.data.configuration_present && voiceReadiness.data.dependencies_present ? "Configured; access unverified" : "Not configured or broker unavailable"}</dd></div><div><dt>Last response from application</dt><dd>{connection.dataUpdatedAt ? new Date(connection.dataUpdatedAt).toLocaleTimeString() : "Not received"}</dd></div></dl>
    {connection.error && <p role="alert">Teaching connection could not be checked.</p>}
    {state?.fault && <p role="alert">The executor stopped after a fault. Inspect the worker before continuing.</p>}
    {previewState?.session_id && <TeachingPreview key={previewState.session_id} online={!!online} context={{ session_id: previewState.session_id, revision: previewState.revision, active_episode_id: previewState.episode_id, mode: previewState.mode }} />}
    {!previewState?.session_id && <p className="warning-box">Simulator preview unavailable until an executor is connected. No camera or microphone has been enabled.</p>}
    <form onSubmit={event => { event.preventDefault(); command.mutate({ operation: "task", args: { instruction: instruction.trim() } }); }}>
      <label className="field-label" htmlFor="teaching-task">Task instruction</label>
      <input id="teaching-task" value={instruction} maxLength={256} onChange={event => setInstruction(event.target.value)} placeholder="Move the gripper above the object" disabled={!online} />
      <button type="submit" className="secondary-button" disabled={!online || !instruction.trim() || command.isPending}>Set instruction</button>
    </form>
    <div className="teaching-actions" style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBlock: 16 }}>
      <button className="primary-button" disabled={!online || !state.instruction || state.mode === "running" || command.isPending} onClick={() => command.mutate({ operation: "start" })}>Start recording</button>
      <button className="secondary-button" disabled={!online || state.mode !== "running"} onClick={() => command.mutate({ operation: "pause" })}>Pause</button>
      <button className="secondary-button" disabled={!online || state.mode === "idle"} onClick={() => command.mutate({ operation: "finish" })}>Finish episode</button>
      <button className="secondary-button" disabled={!online || command.isPending} onClick={() => command.mutate({ operation: "reset" })}>Reset scene</button>
      <button className="secondary-button" disabled={!online || state.mode === "idle" || command.isPending} onClick={() => command.mutate({ operation: "mark_failure", args: { reason: "Operator marked this episode as failed" } })}>Mark failure</button>
    </div>
    <label className="field-label">Joint correction<select aria-label="Joint correction" value={joint || state?.joints?.[0] || ""} onChange={event => setJoint(event.target.value)} disabled={!online}>{state?.joints?.map(name => <option key={name}>{name}</option>)}</select></label>
    <div style={{ display: "flex", gap: 8 }}>
      {[-.05, .05].map(delta => <button key={delta} className="secondary-button" disabled={!online || state.mode !== "running" || command.isPending} onClick={() => command.mutate({ operation: "correct", args: { joint: joint || state?.joints[0], delta_rad: delta } })}>{delta > 0 ? "+" : "−"} 0.05 rad</button>)}
    </div>
    {command.error && <p role="alert">{command.error.message}</p>}
    {receipt.data && <p role="status">Command: {receipt.data.status}{receipt.data.error ? ` · ${receipt.data.error}` : ""}</p>}
    {receiptExpired && <p role="alert">No final acknowledgement arrived. Execution is unverified; inspect the executor before retrying.</p>}
    {receipt.error && <p role="alert">Command acknowledgement unavailable. Execution is not confirmed.</p>}
    {state && <dl className="dataset-facts"><div><dt>Session</dt><dd>{state.session_id}</dd></div><div><dt>Executor mode</dt><dd>{state.mode}</dd></div><div><dt>Recorded steps</dt><dd>{state.steps}</dd></div><div><dt>Simulation time</dt><dd>{state.sim_time?.toFixed(2)} s</dd></div><div><dt>Outcome</dt><dd>{state.outcome || "Unverified"}</dd></div></dl>}
    <TeachingIntelligence context={state?.session_id ? { session_id: state.session_id, episode_id: state.episode_id, revision: state.revision } : null} online={!!online} />
    <hr />
    <h3>Optional voice connection</h3>
    <p>Uses an isolated LiveKit teaching room and OpenRouter. Hosted speech/model services require their configured accounts. Joining enables your microphone for up to five minutes.</p>
    <button className="primary-button" disabled={!online || !(voiceReadiness.isSuccess && voiceReadiness.data?.broker_reachable && voiceReadiness.data.configuration_present && voiceReadiness.data.dependencies_present) || voiceState !== "Disconnected"} onClick={() => void join()}>Connect voice</button>{voiceState !== "Disconnected" && <button className="secondary-button" onClick={() => void leave()}>Disconnect microphone</button>}
    <p role="status">{voiceState} · Voice agent {agentPresent ? "present in room" : "not observed"}</p>{voiceError && <p role="alert">{voiceError}</p>}
    <div ref={media} aria-label="Teaching room media" />
    <p className="field-help">Finalized demonstrations can be imported as an immutable training copy. Task success and physical robot calibration remain separate checks.</p>
  </section>;
}
