"use strict";
const button = document.querySelector("#connect");
const heading = document.querySelector("#heading");
const message = document.querySelector("#message");
const ownedMessage = document.querySelector("#owned-message");
const confirmed = document.querySelector("#confirm");
const start = document.querySelector("#start");
const stop = document.querySelector("#stop");
const restart = document.querySelector("#restart");
const openOwned = document.querySelector("#open-owned");
let current;
const pending = new Set();
const invoke = (command, args) => window.__TAURI__.core.invoke(command, args);
function render(status) {
  current = status;
  const busy = pending.size > 0;
  const active = pending.has("start_owned") || ["starting", "owned", "stopping"].includes(status.mode);
  ownedMessage.textContent = status.message;
  confirmed.disabled = busy || active || !status.payload_available || status.cleanup_unknown;
  start.disabled = confirmed.disabled || !confirmed.checked;
  stop.disabled = pending.has("stop_owned") || pending.has("restart_owned") || pending.has("probe_backend") || (!active && status.mode !== "failed");
  restart.disabled = busy || status.mode !== "owned" || status.cleanup_unknown;
  button.disabled = busy || active || status.mode === "failed" && status.address !== null;
  openOwned.hidden = status.mode !== "owned" || !status.address;
  openOwned.disabled = busy;
}
async function refresh() { render(await invoke("desktop_status")); }
confirmed.addEventListener("change", () => current && render(current));
openOwned.addEventListener("click", async () => {
  try {
    await refresh();
    if (current.mode === "owned" && current.address) window.location.replace(current.address);
  } catch { ownedMessage.textContent = "The desktop could not confirm the owned backend status."; }
});
async function change(command, args) {
  if (pending.has(command)) return;
  pending.add(command);
  if (current) render(current);
  ownedMessage.textContent = command === "start_owned" ? "Starting the dedicated workspace…" : "Waiting for owned backend cleanup…";
  let error;
  try { render(await invoke(command, args)); }
  catch (failure) { error = typeof failure === "string" ? failure : "The desktop operation failed. Return to the connection screen to inspect its status."; }
  finally {
    pending.delete(command);
    try { await refresh(); } catch { if (current) render(current); }
    if (error) ownedMessage.textContent = error;
  }
}
start.addEventListener("click", () => change("start_owned", { confirmed: confirmed.checked }));
stop.addEventListener("click", () => change("stop_owned"));
restart.addEventListener("click", () => change("restart_owned"));
button.addEventListener("click", async () => {
  if (pending.size) return;
  pending.add("probe_backend");
  if (current) render(current);
  heading.textContent = "Checking your application…";
  message.textContent = "Verifying the local API and web interface.";
  try {
    const result = await invoke("probe_backend");
    if (result.status === "ready") {
      heading.textContent = "Connected";
      message.textContent = `Opening OPEN JENSEN ${result.version} at ${result.address}`;
      window.location.replace(result.address);
      return;
    }
    heading.textContent = "Application unavailable";
    message.textContent = result.message;
  } catch {
    heading.textContent = "Connection check failed";
    message.textContent = "The desktop could not check the application. Retry from this connection screen.";
  } finally {
    pending.delete("probe_backend");
    try { await refresh(); } catch { button.disabled = false; }
  }
});
refresh().catch(() => { ownedMessage.textContent = "Desktop ownership status is unavailable. Startup remains disabled."; });
