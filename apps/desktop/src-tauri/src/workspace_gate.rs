//! Fixed offline-handoff startup gate. No webpage paths or new IPC.
use super::*;
use std::collections::BTreeSet;

const JOURNAL: &str = ".desktop-handoff";
const JOURNAL_LIMIT: u64 = 4 * 1024 * 1024;

pub(super) struct Gate {
    _lock: File,
}

impl Gate {
    pub(super) fn acquire(parent: &Path) -> Result<Self, String> {
        prepare_workspace_parent(parent)?;
        let path = parent.join(".desktop-maintenance.lock");
        if let Ok(meta) = fs::symlink_metadata(&path)
            && !meta.is_file()
        {
            return Err("Maintenance lock must be a regular file.".into());
        }
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options
            .open(&path)
            .map_err(|_| "Cannot open maintenance lock.")?;
        let opened = file
            .metadata()
            .map_err(|_| "Cannot inspect maintenance lock.")?;
        let named = fs::symlink_metadata(&path).map_err(|_| "Maintenance lock disappeared.")?;
        if !opened.is_file() || !named.is_file() {
            return Err("Maintenance lock must remain a regular file.".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if opened.ino() != named.ino() || opened.dev() != named.dev() {
                return Err("Maintenance lock changed while opening.".into());
            }
        }
        file.try_lock()
            .map_err(|_| "Offline maintenance already owns this workspace.")?;
        Ok(Self { _lock: file })
    }

    pub(super) fn before_start(&self, workspace: &Path, payload: &Payload) -> Result<(), String> {
        let journal = workspace
            .parent()
            .ok_or("Workspace parent unavailable.")?
            .join(JOURNAL);
        match fs::symlink_metadata(&journal) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(_) => return Err("Cannot inspect the handoff journal; startup is blocked.".into()),
            Ok(_) => {}
        }
        directory(&journal)?;
        let names = fs::read_dir(&journal)
            .map_err(|_| "Cannot inspect handoff journal.")?
            .take(10)
            .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
            .collect::<Result<BTreeSet<_>, _>>()
            .map_err(|_| "Cannot inspect handoff entries.")?;
        let mut expected: BTreeSet<String> =
            ["prepared.json", "commit-intent.json", "committed.json"]
                .map(String::from)
                .into();
        let reversed = names.contains("reversed.json");
        if reversed {
            expected.extend(["reverse-intent.json".into(), "reversed.json".into()]);
        }
        if names.contains("start-intent.json") {
            expected.insert("start-intent.json".into());
        }
        if names != expected {
            return Err(
                "Desktop handoff is incomplete or unrecognized. Inspect it offline before startup."
                    .into(),
            );
        }
        let raw = read_regular(&journal.join("prepared.json"), JOURNAL_LIMIT)?;
        let prepared: Prepared =
            serde_json::from_slice(&raw).map_err(|_| "Invalid prepared handoff.")?;
        let digest = hex(&Sha256::digest(&raw));
        let path_hash = workspace_id(workspace)?;
        if prepared.schema_version != 1
            || prepared.operation != "desktop.workspace.handoff"
            || prepared.workspace != workspace.to_string_lossy()
            || prepared.workspace_path_sha256 != path_hash
            || prepared.transaction_id.len() != 32
            || !prepared
                .transaction_id
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err("Prepared handoff identity differs.".into());
        }
        for name in if reversed {
            vec!["commit-intent.json", "reverse-intent.json"]
        } else {
            vec!["commit-intent.json"]
        } {
            let intent: Intent =
                serde_json::from_slice(&read_regular(&journal.join(name), MAX_LINE as u64)?)
                    .map_err(|_| "Invalid handoff intent.")?;
            if intent.schema_version != 1 || intent.prepared_sha256 != digest {
                return Err("Handoff intent identity differs.".into());
            }
        }
        let committed_raw = read_regular(&journal.join("committed.json"), MAX_LINE as u64)?;
        let committed: Completion =
            serde_json::from_slice(&committed_raw).map_err(|_| "Invalid committed handoff.")?;
        committed.check(&digest, &path_hash, &prepared.new_payload.marker)?;
        let (target, terminal_raw) = if reversed {
            let raw = read_regular(&journal.join("reversed.json"), MAX_LINE as u64)?;
            let completion: Completion =
                serde_json::from_slice(&raw).map_err(|_| "Invalid reversed handoff.")?;
            completion.check(&digest, &path_hash, &prepared.old_payload.marker)?;
            (&prepared.old_payload.marker, raw)
        } else {
            (&prepared.new_payload.marker, committed_raw)
        };
        let expected_owner = expected_owner(payload);
        if target != &expected_owner {
            return Err("Handoff belongs to another compiled payload.".into());
        }
        let intent = StartIntent {
            schema_version: 1,
            transaction_id: prepared.transaction_id,
            prepared_sha256: digest,
            completion_sha256: hex(&Sha256::digest(&terminal_raw)),
            workspace_path_sha256: path_hash,
            marker: expected_owner,
        };
        let path = journal.join("start-intent.json");
        if names.contains("start-intent.json") {
            let saved: StartIntent = serde_json::from_slice(&read_regular(&path, MAX_LINE as u64)?)
                .map_err(|_| "Invalid saved start intent.")?;
            if saved != intent {
                return Err("Saved start intent differs.".into());
            }
        } else {
            // Deliberately retain any partial record on error. Startup and reversal then refuse.
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let mut file = options
                .open(&path)
                .map_err(|_| "Cannot exclusively publish start intent.")?;
            file.write_all(
                &serde_json::to_vec(&intent).map_err(|_| "Cannot encode start intent.")?,
            )
            .map_err(|_| "Cannot write start intent.")?;
            file.sync_all().map_err(|_| "Cannot flush start intent.")?;
            File::open(&journal)
                .and_then(|file| file.sync_all())
                .map_err(|_| "Cannot flush start-intent directory.")?;
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Prepared {
    schema_version: u32,
    operation: String,
    transaction_id: String,
    workspace: String,
    workspace_path_sha256: String,
    old_payload: PayloadRecord,
    new_payload: PayloadRecord,
    #[serde(rename = "old_marker_hex")]
    _old_marker_hex: String,
    #[serde(rename = "files")]
    _files: serde_json::Value,
    #[serde(rename = "backup")]
    _backup: String,
    #[serde(rename = "backup_receipt_sha256")]
    _backup_receipt_sha256: String,
    #[serde(rename = "backup_inventory")]
    _backup_inventory: serde_json::Value,
    #[serde(rename = "limits")]
    _limits: serde_json::Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PayloadRecord {
    marker: WorkspaceOwner,
    #[serde(rename = "paths")]
    _paths: serde_json::Value,
    #[serde(rename = "manifest_sha256")]
    _manifest_sha256: String,
    #[serde(rename = "acceptance_sha256")]
    _acceptance_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Intent {
    schema_version: u32,
    prepared_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Completion {
    schema_version: u32,
    prepared_sha256: String,
    workspace_path_sha256: String,
    marker: WorkspaceOwner,
    inventory_sha256: String,
}
impl Completion {
    fn check(
        &self,
        prepared: &str,
        workspace: &str,
        marker: &WorkspaceOwner,
    ) -> Result<(), String> {
        if self.schema_version != 1
            || self.prepared_sha256 != prepared
            || self.workspace_path_sha256 != workspace
            || self.marker != *marker
            || self.inventory_sha256.len() != 64
            || !self
                .inventory_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err("Handoff completion identity differs.".into());
        }
        Ok(())
    }
}
#[derive(Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct StartIntent {
    schema_version: u32,
    transaction_id: String,
    prepared_sha256: String,
    completion_sha256: String,
    workspace_path_sha256: String,
    marker: WorkspaceOwner,
}
