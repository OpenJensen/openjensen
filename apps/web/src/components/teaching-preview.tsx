"use client";

import { useEffect, useRef, useState } from "react";
import { apiOrigin } from "@/lib/api";

type Context = { session_id: string; revision: number; active_episode_id: string | null; mode: string };
type Frame = Context & {
  schema_version: number; available: boolean; current_context: Context; capture_id: string;
  episode_id: string; step: number; sim_time: number; camera_key: string; camera_prim: string;
  width: number; height: number; rgb_base64: string; rgb_sha256: string; relay_age_ns: number;
  joints: string[]; state_rad: number[]; units: string;
};
const maximumAge = 5000;
const maximumBody = 16 * 1024 * 1024;
const matches = (left: Context, right: Context) => left.session_id === right.session_id && left.revision === right.revision && left.active_episode_id === right.active_episode_id && left.mode === right.mode;

async function readFrame(signal: AbortSignal, session: string): Promise<Frame> {
  const response = await fetch(`${apiOrigin}/api/v1/teaching/frame?session_id=${encodeURIComponent(session)}`, { signal, cache: "no-store" });
  if (!response.ok || !response.body || Number(response.headers.get("content-length")) > maximumBody) throw new Error("Preview unavailable");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      length += chunk.value.length;
      if (length > maximumBody) throw new Error("Preview response exceeded its limit");
      chunks.push(chunk.value);
    }
  } finally { await reader.cancel(); }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)) as Frame;
}

async function pixels(frame: Frame, expected: Context): Promise<Uint8ClampedArray<ArrayBuffer>> {
  if (frame.schema_version !== 1 || frame.available !== true || !frame.current_context || !matches(frame, expected) || !matches(frame.current_context, expected) || !["idle", "running", "paused"].includes(frame.mode)) throw new Error("Observation context changed");
  if (!/^\d{1,19}$/.test(frame.capture_id) || !Number.isSafeInteger(frame.relay_age_ns) || frame.relay_age_ns < 0 || frame.relay_age_ns > maximumAge * 1e6) throw new Error("Observation timing is invalid");
  if (!Number.isInteger(frame.width) || !Number.isInteger(frame.height) || frame.width < 1 || frame.height < 1 || frame.width > 1920 || frame.height > 1920) throw new Error("Invalid preview size");
  if (frame.camera_key !== "observation.images.front" || !/^\/[A-Za-z_][A-Za-z_0-9]*(?:\/[A-Za-z_][A-Za-z_0-9]*)*$/.test(frame.camera_prim) || frame.units !== "rad" || !Array.isArray(frame.joints) || !Array.isArray(frame.state_rad) || frame.joints.length < 1 || frame.joints.length > 32 || new Set(frame.joints).size !== frame.joints.length || frame.state_rad.length !== frame.joints.length || !frame.state_rad.every(Number.isFinite)) throw new Error("Invalid preview metadata");
  const size = frame.width * frame.height * 3;
  if (typeof frame.rgb_base64 !== "string" || frame.rgb_base64.length !== 4 * Math.ceil(size / 3)) throw new Error("Invalid preview pixels");
  const raw = Uint8Array.from(atob(frame.rgb_base64), char => char.charCodeAt(0));
  if (raw.length !== size) throw new Error("Invalid preview pixels");
  const digest = await crypto.subtle.digest("SHA-256", raw);
  const hash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, "0")).join("");
  if (hash !== frame.rgb_sha256) throw new Error("Preview hash differs");
  const rgba = new Uint8ClampedArray(frame.width * frame.height * 4);
  for (let source = 0, target = 0; source < raw.length; source += 3, target += 4) { rgba[target] = raw[source]; rgba[target + 1] = raw[source + 1]; rgba[target + 2] = raw[source + 2]; rgba[target + 3] = 255; }
  return rgba;
}

export function TeachingPreview({ context, online }: { context: Context; online: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const expected = useRef(context);
  expected.current = context;
  const accepted = useRef<{ id: bigint; expires: number; fingerprint: string } | null>(null);
  const retired = useRef<bigint>(-1n);
  const [observation, setObservation] = useState<Frame | null>(null);
  const [fresh, setFresh] = useState(false);
  const [message, setMessage] = useState("Waiting for a fresh simulator observation.");
  const invalidate = () => {
    if (accepted.current) retired.current = accepted.current.id > retired.current ? accepted.current.id : retired.current;
    setFresh(false);
  };
  useEffect(() => {
    // A known context change retires old pixels even if a later server response
    // replays them. HTTP recovery alone cannot revive an invalidated capture.
    invalidate();
    setMessage(online ? "Waiting for a fresh simulator observation." : "Preview disconnected. Previous pixels are not live.");
    if (!online) return;
    let disposed = false;
    let next: ReturnType<typeof setTimeout>;
    let controller: AbortController | undefined;
    const expire = setInterval(() => {
      if (accepted.current && performance.now() >= accepted.current.expires) {
        invalidate(); setMessage("Observation is stale. Waiting for new pixels.");
      }
    }, 200);
    async function poll() {
      controller = new AbortController();
      const deadline = setTimeout(() => controller?.abort(), 6000);
      const started = performance.now();
      let observedId: bigint | null = null;
      try {
        const frame = await readFrame(controller.signal, context.session_id);
        if (typeof frame.capture_id === "string" && /^\d{1,19}$/.test(frame.capture_id)) observedId = BigInt(frame.capture_id);
        const rgba = await pixels(frame, context);
        const received = performance.now();
        if (disposed) return;
        if (!matches(expected.current, context)) throw new Error("Observation context changed");
        const id = BigInt(frame.capture_id);
        const age = frame.relay_age_ns / 1e6 + received - started;
        if (age > maximumAge || id <= retired.current || (accepted.current && id < accepted.current.id)) throw new Error("Observation is stale or superseded");
        const fingerprint = JSON.stringify([frame.revision, frame.active_episode_id, frame.mode, frame.episode_id, frame.step, frame.sim_time, frame.width, frame.height, frame.rgb_sha256, frame.camera_prim, frame.joints, frame.state_rad]);
        if (accepted.current?.id === id && accepted.current.fingerprint !== fingerprint) throw new Error("Capture identity was reused");
        const expires = Math.min(received + maximumAge - age, accepted.current?.id === id ? accepted.current.expires : Infinity);
        if (expires <= received) throw new Error("Observation is stale");
        const drawing = canvas.current?.getContext("2d");
        if (!canvas.current || !drawing) throw new Error("Preview rendering unavailable");
        canvas.current.width = frame.width; canvas.current.height = frame.height;
        drawing.putImageData(new ImageData(rgba, frame.width, frame.height), 0, 0);
        accepted.current = { id, expires, fingerprint };
        setObservation(frame); setFresh(true); setMessage("Fresh simulator observation");
      } catch {
        if (!disposed) {
          if (observedId !== null && observedId > retired.current) retired.current = observedId;
          invalidate(); setMessage("Preview unavailable, stale or changed. Waiting for a new observation."); }
      } finally {
        clearTimeout(deadline);
        if (!disposed) next = setTimeout(() => void poll(), 750);
      }
    }
    void poll();
    return () => { disposed = true; clearTimeout(next); clearInterval(expire); controller?.abort(); };
    // Session changes remount this component; other context changes retire pixels.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [online, context.session_id, context.revision, context.active_episode_id, context.mode]);
  const visibleFresh = fresh && online && observation && matches(observation, context);
  return <section className={`teaching-preview${visibleFresh ? "" : " teaching-preview-stale"}`} aria-label="Simulator preview">
    <h3>Simulator camera and joints</h3>
    <p>Read-only observations, independent of voice. A frame does not establish task success or calibration.</p>
    <p role="status">{visibleFresh ? message : observation ? "Previous observation — not live. " + message : message}</p>
    <canvas ref={canvas} role="img" aria-label={visibleFresh ? "Fresh simulator camera" : "Simulator camera unavailable or stale"} hidden={!observation} />
    {observation && <><p className="field-help">{observation.camera_key} · Frame {observation.step} · {observation.sim_time.toFixed(2)} simulator seconds</p><dl className="cloud-run-facts">{observation.joints.map((name, index) => <div key={name}><dt>{name}</dt><dd>{observation.state_rad[index].toFixed(4)} rad</dd></div>)}</dl></>}
  </section>;
}
