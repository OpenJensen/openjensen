//! Fixed bundled child ownership. HTTP pages never receive these commands.
use crate::connection::{Backend, Probe, is_connection_page, probe};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, mpsc};
use std::thread;
use std::time::{Duration, Instant};

const MAX_LINE: usize = 4096;
const MAX_STDERR: usize = 64 * 1024;
const UNAVAILABLE: &str = "This desktop has no verified bundled Python payload. Connect to an existing application instead.";
// A separately reviewed freezer build must supply an immutable pin before this can be Some.
// No environment, command argument, web input or discovered file can enable production startup.
const PRODUCTION_PAYLOAD: Option<PayloadPin> = None;

#[derive(Clone, Copy)]
struct PayloadPin {
    executable_sha256: &'static str,
    executable_bytes: u64,
    resources_sha256: &'static str,
    build_id: &'static str,
}

#[derive(Clone)]
struct Payload {
    executable: PathBuf,
    resources: PathBuf,
    resources_sha256: String,
    build_id: String,
    pin: PayloadPin,
    #[cfg(test)]
    fixture: Option<PathBuf>,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // A poisoned owner must still be accessible for cleanup. Never discard its Child handle.
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn read_regular(path: &Path, maximum: u64) -> Result<Vec<u8>, String> {
    let before = fs::symlink_metadata(path).map_err(|_| "Required bundled file is missing.")?;
    if !before.is_file() || before.len() > maximum {
        return Err("Expected a bounded regular file, without a symlink.".into());
    }
    let file = File::open(path).map_err(|_| "Cannot open the required file.")?;
    let after = file
        .metadata()
        .map_err(|_| "Cannot inspect the opened file.")?;
    if !after.is_file() || after.len() != before.len() {
        return Err("File changed while opening it.".into());
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "Cannot read the required file.")?;
    if bytes.len() as u64 != before.len() {
        return Err("File changed or exceeded its byte limit.".into());
    }
    Ok(bytes)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourceManifest {
    schema_version: u32,
    build_id: String,
    app_version: String,
    files: std::collections::BTreeMap<String, serde_json::Value>,
}

impl Payload {
    fn bundled(root: &Path) -> Result<Self, String> {
        let pin = PRODUCTION_PAYLOAD.ok_or(UNAVAILABLE)?;
        Self::from_pin(root, pin)
    }

    fn from_pin(root: &Path, pin: PayloadPin) -> Result<Self, String> {
        let root = root
            .canonicalize()
            .map_err(|_| "Bundled resource directory is missing.")?;
        let folder = root.join("sidecar/firebird-sidecar");
        let executable = folder.join(if cfg!(windows) {
            "firebird-sidecar.exe"
        } else {
            "firebird-sidecar"
        });
        let resources = folder.join("_internal/sidecar-resources");
        for path in [&folder, &resources, &executable] {
            if fs::symlink_metadata(path)
                .map_err(|_| "Bundled payload is incomplete.")?
                .file_type()
                .is_symlink()
                || !path
                    .canonicalize()
                    .map_err(|_| "Bundled payload is incomplete.")?
                    .starts_with(&root)
            {
                return Err(
                    "Bundled payload must stay inside its fixed resource directory.".into(),
                );
            }
        }
        let payload = Self {
            executable,
            resources,
            resources_sha256: pin.resources_sha256.into(),
            build_id: pin.build_id.into(),
            pin,
            #[cfg(test)]
            fixture: None,
        };
        payload.verify()?;
        Ok(payload)
    }

    fn verify(&self) -> Result<(), String> {
        #[cfg(test)]
        if self.fixture.is_some() {
            return Ok(());
        }
        // Stream the fixed executable hash; a future pin also fixes its maximum size.
        let info =
            fs::symlink_metadata(&self.executable).map_err(|_| "Bundled executable is missing.")?;
        if !info.is_file() || info.len() != self.pin.executable_bytes {
            return Err("Bundled executable identity changed.".into());
        }
        let mut source = File::open(&self.executable)
            .map_err(|_| "Cannot read bundled executable.")?
            .take(self.pin.executable_bytes + 1);
        let mut digest = Sha256::new();
        let mut seen = 0;
        let mut buffer = [0_u8; 65536];
        loop {
            let count = source
                .read(&mut buffer)
                .map_err(|_| "Cannot hash bundled executable.")?;
            if count == 0 {
                break;
            }
            seen += count as u64;
            digest.update(&buffer[..count]);
        }
        if seen != self.pin.executable_bytes
            || hex(&digest.finalize()) != self.pin.executable_sha256
        {
            return Err("Bundled executable identity changed.".into());
        }
        let raw = read_regular(&self.resources.join("resources.json"), 4 * 1024 * 1024)?;
        if hex(&Sha256::digest(&raw)) != self.resources_sha256 {
            return Err("Bundled static manifest identity changed.".into());
        }
        let manifest: ResourceManifest =
            serde_json::from_slice(&raw).map_err(|_| "Invalid bundled manifest.")?;
        if manifest.schema_version != 1
            || manifest.build_id != self.build_id
            || manifest.app_version != env!("CARGO_PKG_VERSION")
            || manifest.files.is_empty()
            || manifest.files.len() > 10000
            || !manifest.files.contains_key("index.html")
        {
            return Err("Bundled manifest does not match this desktop build.".into());
        }
        // The pinned child verifies every static payload file before ready. This native check
        // authenticates the immutable manifest, not arbitrary unpinned onedir dependencies.
        Ok(())
    }

    fn command(&self, workspace: &Path) -> Command {
        let mut command = Command::new(&self.executable);
        #[cfg(test)]
        if let Some(fixture) = &self.fixture {
            command.arg(fixture);
        }
        command
            .args(["serve", "--resources"])
            .arg(&self.resources)
            .arg("--data-dir")
            .arg(workspace);
        command.env_clear();
        // The initial owned mode has no external workers, credentials or provider configuration.
        // Only platform runtime essentials survive; HOME/PATH/PYTHONPATH are deliberately absent.
        for name in ["SYSTEMROOT", "WINDIR", "TMP", "TEMP", "TMPDIR"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command
            .current_dir(workspace)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct WorkspaceOwner {
    schema_version: u32,
    owner: String,
    build_id: String,
    resources_sha256: String,
}

fn directory(path: &Path) -> Result<(), String> {
    let meta = fs::symlink_metadata(path).map_err(|_| "Workspace directory is unavailable.")?;
    if !meta.is_dir() || meta.file_type().is_symlink() {
        return Err("Workspace must be a real directory.".into());
    }
    Ok(())
}

fn workspace(parent: &Path, payload: &Payload) -> Result<PathBuf, String> {
    if !parent.is_absolute() {
        return Err("Desktop data directory must be absolute.".into());
    }
    // Only the OS-selected app directory is accepted. Reject symlink ancestors before creating.
    for ancestor in parent.ancestors() {
        if ancestor.exists() {
            directory(ancestor)?;
        } else if ancestor.symlink_metadata().is_ok() {
            return Err("Workspace path contains a link.".into());
        }
    }
    fs::create_dir_all(parent)
        .map_err(|_| "Cannot create the dedicated desktop data directory.")?;
    let root = parent.join("desktop-v1");
    let expected = WorkspaceOwner {
        schema_version: 1,
        owner: "openjensen-desktop".into(),
        build_id: payload.build_id.clone(),
        resources_sha256: payload.resources_sha256.clone(),
    };
    match fs::create_dir(&root) {
        Ok(()) => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&root, fs::Permissions::from_mode(0o700))
                    .map_err(|_| "Cannot protect the new workspace.")?;
            }
            let marker = root.join("desktop-owner.json");
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options.open(marker).map_err(
                |_| "Cannot publish desktop workspace ownership; partial directory preserved.",
            )?;
            file.write_all(
                &serde_json::to_vec(&expected).map_err(|_| "Cannot encode workspace ownership.")?,
            )
            .map_err(|_| "Cannot write workspace ownership.")?;
            file.sync_all()
                .map_err(|_| "Cannot flush workspace ownership.")?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            directory(&root)?;
            let owner: WorkspaceOwner = serde_json::from_slice(&read_regular(
                &root.join("desktop-owner.json"),
                MAX_LINE as u64,
            )?)
            .map_err(|_| "Existing workspace has invalid ownership.")?;
            if owner != expected {
                return Err("Existing workspace belongs to another build. Automatic adoption or migration is disabled.".into());
            }
        }
        Err(_) => return Err("Cannot create the dedicated desktop workspace.".into()),
    }
    root.canonicalize()
        .map_err(|_| "Cannot resolve the dedicated desktop workspace.".into())
}

fn workspace_id(path: &Path) -> Result<String, String> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Ok(hex(&Sha256::digest(path.as_os_str().as_bytes())))
    }
    #[cfg(not(unix))]
    {
        path.to_str()
            .map(|path| hex(&Sha256::digest(path.as_bytes())))
            .ok_or_else(|| "Workspace path is not representable in the private protocol.".into())
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
enum Event {
    Ready {
        schema_version: u32,
        nonce: String,
        host: String,
        port: u16,
        build_id: String,
        app_version: String,
        resources_sha256: String,
        workspace_id: String,
    },
    Stopped {
        schema_version: u32,
        nonce: String,
        reason: String,
    },
}

fn read_event(reader: &mut impl BufRead) -> Result<Option<Event>, String> {
    let mut raw = Vec::new();
    let count = reader
        .take((MAX_LINE + 2) as u64)
        .read_until(b'\n', &mut raw)
        .map_err(|_| "Private child output could not be read.")?;
    if count == 0 {
        return Ok(None);
    }
    if raw.pop() != Some(b'\n') || raw.len() > MAX_LINE {
        return Err("Private child output is oversized or incomplete.".into());
    }
    serde_json::from_slice(&raw)
        .map(Some)
        .map_err(|_| "Private child output has an invalid schema.".into())
}

#[derive(Clone, Copy)]
struct Limits {
    startup: Duration,
    shutdown: Duration,
    reap: Duration,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            startup: Duration::from_secs(90),
            shutdown: Duration::from_secs(30),
            reap: Duration::from_secs(5),
        }
    }
}

struct OwnedChild {
    process: Child,
    input: Option<ChildStdin>,
    events: mpsc::Receiver<Result<Event, String>>,
    sender: Option<mpsc::SyncSender<Result<Event, String>>>,
    stderr: Arc<Mutex<VecDeque<u8>>>,
    nonce: String,
    stopped: bool,
    protocol_failed: bool,
}

impl OwnedChild {
    fn spawn(payload: &Payload, workspace: &Path, nonce: String) -> Result<Self, String> {
        let mut process = payload
            .command(workspace)
            .spawn()
            .map_err(|_| "Bundled backend could not be started.")?;
        let input = process.stdin.take();
        let (sender, events) = mpsc::sync_channel(4);
        Ok(Self {
            process,
            input,
            events,
            sender: Some(sender),
            stderr: Arc::new(Mutex::new(VecDeque::with_capacity(MAX_STDERR))),
            nonce,
            stopped: false,
            protocol_failed: false,
        })
    }

    fn readers(&mut self) -> Result<(), String> {
        let output = self
            .process
            .stdout
            .take()
            .ok_or("Child stdout is unavailable.")?;
        let sender = self
            .sender
            .take()
            .ok_or("Child output reader already started.")?;
        thread::Builder::new()
            .name("desktop-control".into())
            .spawn(move || {
                let mut reader = BufReader::new(output);
                // Exactly ready then stopped; never accumulate an unbounded event history.
                for _ in 0..3 {
                    match read_event(&mut reader) {
                        Ok(Some(event)) => {
                            if sender.send(Ok(event)).is_err() {
                                return;
                            }
                        }
                        Ok(None) => return,
                        Err(error) => {
                            let _ = sender.send(Err(error));
                            return;
                        }
                    }
                }
            })
            .map_err(|_| "Cannot start the private output reader.")?;
        let mut error = self
            .process
            .stderr
            .take()
            .ok_or("Child stderr is unavailable.")?;
        let ring = Arc::clone(&self.stderr);
        thread::Builder::new()
            .name("desktop-diagnostics".into())
            .spawn(move || {
                let mut buffer = [0_u8; 8192];
                while let Ok(count) = error.read(&mut buffer) {
                    if count == 0 {
                        break;
                    }
                    let mut tail = lock(&ring);
                    for byte in &buffer[..count] {
                        if tail.len() == MAX_STDERR {
                            tail.pop_front();
                        }
                        tail.push_back(*byte);
                    }
                }
            })
            .map_err(|_| "Cannot start the private diagnostic reader.")?;
        Ok(())
    }

    fn send(&mut self, command: &str) -> Result<(), String> {
        let message = serde_json::json!({"schema_version":1,"command":command,"nonce":self.nonce});
        let mut bytes =
            serde_json::to_vec(&message).map_err(|_| "Cannot encode private control.")?;
        bytes.push(b'\n');
        self.input
            .as_mut()
            .ok_or("Private control pipe is closed.")?
            .write_all(&bytes)
            .map_err(|_| "Private control pipe could not be written.".into())
    }

    fn stopped_events(&mut self) {
        while let Ok(event) = self.events.try_recv() {
            match event {
                Ok(Event::Stopped {
                    schema_version: 1,
                    nonce,
                    reason,
                }) if nonce == self.nonce
                    && !self.stopped
                    && matches!(reason.as_str(), "shutdown" | "parent_eof") =>
                {
                    self.stopped = true
                }
                _ => self.protocol_failed = true,
            }
        }
    }

    fn stop(&mut self, limits: Limits) -> Cleanup {
        let _ = self.send("shutdown");
        self.input.take(); // Parent EOF is also an explicit graceful shutdown signal.
        let deadline = Instant::now() + limits.shutdown;
        loop {
            self.stopped_events();
            match self.process.try_wait() {
                Ok(Some(status)) => {
                    // The pipe reader may be a scheduling tick behind an already exited child.
                    while !self.stopped && !self.protocol_failed && Instant::now() < deadline {
                        match self.events.recv_timeout(Duration::from_millis(10)) {
                            Ok(event) => match event {
                                Ok(Event::Stopped {
                                    schema_version: 1,
                                    nonce,
                                    reason,
                                }) if nonce == self.nonce
                                    && matches!(reason.as_str(), "shutdown" | "parent_eof") =>
                                {
                                    self.stopped = true
                                }
                                _ => self.protocol_failed = true,
                            },
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                            Err(mpsc::RecvTimeoutError::Timeout) => {}
                        }
                    }
                    self.stopped_events();
                    return Cleanup {
                        reaped: true,
                        graceful: status.success() && self.stopped && !self.protocol_failed,
                    };
                }
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
                _ => break,
            }
        }
        let _ = self.process.kill(); // Only this live Child handle; never a saved PID or process group.
        let deadline = Instant::now() + limits.reap;
        loop {
            match self.process.try_wait() {
                Ok(Some(_)) => {
                    return Cleanup {
                        reaped: true,
                        graceful: false,
                    };
                }
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
                _ => {
                    return Cleanup {
                        reaped: false,
                        graceful: false,
                    };
                }
            }
        }
    }
}

struct Cleanup {
    reaped: bool,
    graceful: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct DesktopStatus {
    pub mode: &'static str,
    pub address: Option<String>,
    pub generation: u64,
    pub payload_available: bool,
    pub cleanup_unknown: bool,
    pub message: String,
}

pub struct Controller {
    child: Mutex<Option<OwnedChild>>,
    view: Mutex<DesktopStatus>,
    payload: Result<Payload, String>,
    workspace_parent: PathBuf,
    attached: Result<Backend, String>,
    cancelled: AtomicU64,
    exiting: AtomicBool,
    limits: Limits,
}

impl Controller {
    pub fn new(resources: &Path, data: PathBuf, attached: Result<Backend, String>) -> Self {
        Self::with_payload(
            Payload::bundled(resources),
            data.join("workspaces"),
            attached,
        )
    }

    fn with_payload(
        payload: Result<Payload, String>,
        workspace_parent: PathBuf,
        attached: Result<Backend, String>,
    ) -> Self {
        let available = payload.is_ok();
        Self {
            child: Mutex::new(None),
            view: Mutex::new(DesktopStatus {
                mode: "idle",
                address: None,
                generation: 0,
                payload_available: available,
                cleanup_unknown: false,
                message: if available {
                    "Choose a separate desktop workspace or connect to your existing application."
                        .into()
                } else {
                    UNAVAILABLE.into()
                },
            }),
            payload,
            workspace_parent,
            attached,
            cancelled: AtomicU64::new(0),
            exiting: AtomicBool::new(false),
            limits: Limits::default(),
        }
    }

    fn update(
        &self,
        mode: &'static str,
        address: Option<String>,
        unknown: bool,
        message: impl Into<String>,
    ) {
        let mut view = lock(&self.view);
        view.mode = mode;
        view.address = address;
        view.cleanup_unknown = unknown;
        view.message = message.into();
    }

    pub fn status(&self) -> DesktopStatus {
        // Never block the UI behind startup/shutdown. Observe unexpected child exits on the next
        // status/navigation request; do not relaunch jobs or adopt a server still using its port.
        if let Ok(mut slot) = self.child.try_lock()
            && let Some(child) = slot.as_mut()
        {
            child.stopped_events();
            if child.protocol_failed || child.stopped {
                self.update("failed", None, true, "The owned backend private protocol ended or failed. Stop it to resolve process ownership; automatic restart is disabled.");
            }
            match child.process.try_wait() {
                Ok(Some(_)) => {
                    *slot = None;
                    self.update("failed", None, true, "The owned backend exited unexpectedly. Job and nested-worker cleanup is unverified; automatic restart is disabled.");
                }
                Err(_) => self.update(
                    "failed",
                    None,
                    true,
                    "Cannot inspect the owned process. Cleanup remains unknown.",
                ),
                Ok(None) => {}
            }
        }
        lock(&self.view).clone()
    }

    pub fn permits_navigation(&self, url: &tauri::Url) -> bool {
        is_connection_page(url)
            || self
                .status()
                .address
                .and_then(|value| tauri::Url::parse(&value).ok())
                .is_some_and(|allowed| Backend { url: allowed }.permits_navigation(url))
    }

    pub fn attach(&self) -> Probe {
        let slot = lock(&self.child);
        if slot.is_some() || self.exiting.load(Ordering::SeqCst) {
            return Probe::unavailable(
                "Stop the owned backend before connecting to an external application.",
            );
        }
        let backend = match &self.attached {
            Ok(backend) => backend,
            Err(error) => return Probe::unavailable(error),
        };
        let result = probe(backend, Duration::from_secs(2));
        let unknown = lock(&self.view).cleanup_unknown;
        if result.status == "ready" {
            self.update(
                "attached",
                result.address.clone(),
                unknown,
                if unknown {
                    "Connected to an external application, which remains running on close. Previous owned cleanup remains unknown; another owned start is blocked."
                } else {
                    "Connected to an external application. Desktop close will leave it running."
                },
            );
        } else {
            self.update("idle", None, unknown, &result.message);
        }
        result
    }

    pub fn start(&self, confirmed: bool) -> Result<DesktopStatus, String> {
        if !confirmed {
            return Err(
                "Explicit confirmation is required to create a dedicated desktop workspace.".into(),
            );
        }
        let epoch = self.cancelled.load(Ordering::SeqCst);
        let mut slot = lock(&self.child);
        self.start_locked(&mut slot, epoch)
    }

    fn start_locked(
        &self,
        slot: &mut Option<OwnedChild>,
        epoch: u64,
    ) -> Result<DesktopStatus, String> {
        if self.exiting.load(Ordering::SeqCst) || epoch != self.cancelled.load(Ordering::SeqCst) {
            return Err("Startup was cancelled.".into());
        }
        if slot.is_some() {
            return Err("An owned backend is already active or still awaiting cleanup.".into());
        }
        if lock(&self.view).cleanup_unknown {
            return Err("Previous cleanup is unverified. Close the desktop and inspect the preserved workspace before starting another backend.".into());
        }
        let payload = self.payload.as_ref().map_err(Clone::clone)?;
        payload.verify()?;
        let data = workspace(&self.workspace_parent, payload)?;
        let mut random = [0_u8; 16];
        getrandom::fill(&mut random).map_err(|_| "Cannot create a private startup nonce.")?;
        let nonce = hex(&random);
        let expected_workspace = workspace_id(&data)?;
        if self.exiting.load(Ordering::SeqCst) || epoch != self.cancelled.load(Ordering::SeqCst) {
            return Err("Startup was cancelled before process creation.".into());
        }
        lock(&self.view).generation += 1;
        self.update(
            "starting",
            None,
            false,
            "Starting the dedicated desktop workspace…",
        );
        let deadline = Instant::now() + self.limits.startup;
        let started = OwnedChild::spawn(payload, &data, nonce.clone());
        match started {
            Ok(child) => *slot = Some(child),
            Err(error) => {
                self.update("failed", None, false, &error);
                return Err(error);
            }
        }
        // Register the Child before any operation that can fail, including reader creation.
        let result = (|| {
            let child = slot.as_mut().ok_or("Owned child registration failed.")?;
            child.readers()?;
            child.send("start")?;
            loop {
                if self.exiting.load(Ordering::SeqCst)
                    || epoch != self.cancelled.load(Ordering::SeqCst)
                {
                    return Err("Startup was cancelled.".into());
                }
                if Instant::now() >= deadline {
                    return Err("Backend startup exceeded its 90-second supervisor budget.".into());
                }
                match child.events.recv_timeout(Duration::from_millis(10)) {
                    Ok(Ok(Event::Ready { schema_version: 1, nonce: received, host, port, build_id, app_version, resources_sha256, workspace_id }))
                        if received == nonce && host == "127.0.0.1" && port != 0 && build_id == payload.build_id && app_version == env!("CARGO_PKG_VERSION") && resources_sha256 == payload.resources_sha256 && workspace_id == expected_workspace => {
                            if child.process.try_wait().map_err(|_| "Cannot inspect started backend.")?.is_some() { return Err("Backend exited during readiness.".into()); }
                            return Backend::from_port(Some(&port.to_string()));
                        }
                    Ok(_) => return Err("Owned backend readiness identity did not match the private startup request.".into()),
                    Err(mpsc::RecvTimeoutError::Disconnected) => return Err("Backend closed its private output before readiness.".into()),
                    Err(mpsc::RecvTimeoutError::Timeout) => if child.process.try_wait().map_err(|_| "Cannot inspect starting backend.")?.is_some() { return Err("Backend exited before readiness.".into()); },
                }
            }
        })();
        match result {
            Ok(backend) => {
                self.update("owned", Some(backend.url.to_string()), false, "The dedicated desktop backend is ready. Closing this desktop requests its shutdown.");
                Ok(lock(&self.view).clone())
            }
            Err(error) => {
                let cleanup = slot.as_mut().expect("registered child").stop(self.limits);
                if cleanup.reaped {
                    *slot = None;
                }
                self.update(
                    "failed",
                    None,
                    !cleanup.graceful,
                    format!(
                        "{error} {}",
                        if cleanup.graceful {
                            "The child was stopped and reaped."
                        } else {
                            "Job or nested-worker cleanup remains unverified."
                        }
                    ),
                );
                Err(lock(&self.view).message.clone())
            }
        }
    }

    pub fn stop(&self) -> DesktopStatus {
        self.cancelled.fetch_add(1, Ordering::SeqCst);
        let mut slot = lock(&self.child);
        self.stop_locked(&mut slot);
        lock(&self.view).clone()
    }

    fn stop_locked(&self, slot: &mut Option<OwnedChild>) {
        let Some(child) = slot.as_mut() else {
            return;
        }; // Never touches an attached server.
        self.update(
            "stopping",
            None,
            false,
            "Stopping the owned backend and waiting for cleanup…",
        );
        let cleanup = child.stop(self.limits);
        if cleanup.reaped {
            *slot = None;
        }
        if cleanup.graceful {
            self.update(
                "idle",
                None,
                false,
                "The owned backend confirmed shutdown and was reaped. Workspace data is preserved.",
            );
        } else {
            self.update("failed", None, true, if cleanup.reaped { "The owned process was reaped without a verified graceful shutdown. Nested-worker or remote cleanup is unknown; automatic restart is disabled." } else { "The owned process could not be reaped within the supervisor budget. Cleanup is unknown and another start is blocked." });
        }
    }

    pub fn restart(&self) -> Result<DesktopStatus, String> {
        let epoch = self.cancelled.fetch_add(1, Ordering::SeqCst) + 1;
        let mut slot = lock(&self.child);
        if slot.is_none() || lock(&self.view).mode != "owned" {
            return Err("Restart is available only for this desktop's ready owned backend.".into());
        }
        self.stop_locked(&mut slot);
        self.start_locked(&mut slot, epoch)
    }

    pub fn begin_exit(&self) -> bool {
        !self.exiting.swap(true, Ordering::SeqCst)
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        // Also cover local error/unwind paths. Never abandon a known Child merely because
        // its UI/controller was dropped; this remains bounded and never touches attached servers.
        let slot = self
            .child
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(child) = slot.as_mut() {
            let cleanup = child.stop(self.limits);
            if !cleanup.graceful {
                eprintln!("Desktop owner dropped with unverified nested-worker cleanup.");
            }
        }
    }
}

#[cfg(test)]
#[path = "owned_backend_tests.rs"]
mod tests;
