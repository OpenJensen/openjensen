//! Authenticate the complete fixed onedir payload before any owned child starts.
//! The installed bundle is a trusted same-user filesystem, not a hostile-writer sandbox.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, Metadata};
use std::io::Read;
use std::path::{Path, PathBuf};

const MAX_ENTRIES: usize = 20_000;
const MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_MANIFEST: u64 = 8 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) struct ManifestPin {
    pub sha256: &'static str,
    pub bytes: u64,
    pub identity: &'static str,
    pub entries: usize,
    pub file_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema_version: u32,
    entries: BTreeMap<String, Entry>,
    entry_count: usize,
    file_bytes: u64,
    identity_sha256: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Entry {
    File {
        bytes: u64,
        sha256: String,
        mode: u32,
        macho_cpu_types: Option<Vec<u32>>,
    },
    Directory {
        mode: u32,
    },
    Symlink {
        target: String,
        mode: u32,
    },
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn relative(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 4096
        && !value.contains(['\\', '\0'])
        && value
            .split('/')
            .all(|part| !matches!(part, "" | "." | ".."))
}

#[cfg(unix)]
fn mode(info: &Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    info.permissions().mode() & 0o7777
}
#[cfg(not(unix))]
fn mode(_info: &Metadata) -> u32 {
    0
}

#[cfg(unix)]
fn same_file(left: &Metadata, right: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.mode() == right.mode()
        && left.len() == right.len()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
        && left.ctime() == right.ctime()
        && left.ctime_nsec() == right.ctime_nsec()
}
#[cfg(not(unix))]
fn same_file(left: &Metadata, right: &Metadata) -> bool {
    left.len() == right.len()
        && left.is_file() == right.is_file()
        && left.modified().ok() == right.modified().ok()
}

fn hash_file(path: &Path, bytes: u64, expected: &str) -> Result<(), String> {
    let before = fs::symlink_metadata(path).map_err(|_| "Missing payload file.")?;
    if !before.is_file() || before.len() != bytes {
        return Err("Payload file type/size changed.".into());
    }
    let mut file = File::open(path).map_err(|_| "Cannot open payload file.")?;
    let opened = file
        .metadata()
        .map_err(|_| "Cannot inspect payload file.")?;
    if !same_file(&before, &opened) {
        return Err("Payload file changed before hashing.".into());
    }
    let mut hasher = Sha256::new();
    let mut seen = 0_u64;
    let mut buffer = [0_u8; 65536];
    loop {
        let maximum = buffer.len().min((bytes + 1 - seen) as usize);
        let count = file
            .read(&mut buffer[..maximum])
            .map_err(|_| "Cannot hash payload file.")?;
        if count == 0 {
            break;
        }
        seen += count as u64;
        if seen > bytes {
            return Err("Payload file grew while hashing.".into());
        }
        hasher.update(&buffer[..count]);
    }
    let after = file
        .metadata()
        .map_err(|_| "Cannot inspect hashed payload file.")?;
    if seen != bytes || !same_file(&opened, &after) || hex(&hasher.finalize()) != expected {
        return Err("Payload file identity changed.".into());
    }
    Ok(())
}

pub(crate) fn verify(root: &Path, manifest_path: &Path, pin: ManifestPin) -> Result<(), String> {
    if !cfg!(unix) {
        return Err("This payload inventory requires Unix permissions.".into());
    }
    if pin.bytes == 0
        || pin.bytes > MAX_MANIFEST
        || pin.entries == 0
        || pin.entries > MAX_ENTRIES
        || pin.file_bytes > MAX_BYTES
        || !digest(pin.sha256)
        || !digest(pin.identity)
    {
        return Err("Invalid compiled payload inventory pin.".into());
    }
    hash_file(manifest_path, pin.bytes, pin.sha256)?;
    let mut raw = Vec::new();
    File::open(manifest_path)
        .map_err(|_| "Cannot open payload inventory.")?
        .take(pin.bytes + 1)
        .read_to_end(&mut raw)
        .map_err(|_| "Cannot read payload inventory.")?;
    if raw.len() as u64 != pin.bytes || hex(&Sha256::digest(&raw)) != pin.sha256 {
        return Err("Payload inventory changed while reading.".into());
    }
    let manifest: Manifest =
        serde_json::from_slice(&raw).map_err(|_| "Invalid payload inventory schema.")?;
    if manifest.schema_version != 1
        || manifest.entry_count != pin.entries
        || manifest.entries.len() != pin.entries
        || manifest.file_bytes != pin.file_bytes
        || manifest.identity_sha256 != pin.identity
    {
        return Err("Payload inventory does not match this native build.".into());
    }
    let mut total = 0_u64;
    for (name, entry) in &manifest.entries {
        if !relative(name) {
            return Err("Invalid payload inventory path.".into());
        }
        let permissions = match entry {
            Entry::File {
                bytes,
                sha256,
                mode,
                macho_cpu_types,
            } => {
                total = total
                    .checked_add(*bytes)
                    .filter(|value| *value <= MAX_BYTES)
                    .ok_or("Payload byte limit exceeded.")?;
                if !digest(sha256)
                    || macho_cpu_types
                        .as_ref()
                        .is_some_and(|values| values != &[0x0100_000c])
                {
                    return Err("Invalid file digest or non-ARM64 payload.".into());
                }
                *mode
            }
            Entry::Directory { mode } => *mode,
            Entry::Symlink { target, mode } => {
                if target.is_empty()
                    || target.len() > 4096
                    || target.contains('\0')
                    || Path::new(target).is_absolute()
                {
                    return Err("Invalid payload symlink target.".into());
                }
                *mode
            }
        };
        if permissions > 0o7777 {
            return Err("Invalid payload permission bits.".into());
        }
    }
    if total != pin.file_bytes {
        return Err("Payload byte sum does not match its pin.".into());
    }
    let root_info = fs::symlink_metadata(root).map_err(|_| "Payload root is missing.")?;
    if !root.is_absolute() || !root_info.is_dir() || root_info.file_type().is_symlink() {
        return Err("Payload root must be an absolute real directory.".into());
    }
    let root = root
        .canonicalize()
        .map_err(|_| "Cannot resolve payload root.")?;
    let mut stack = vec![root.clone()];
    let mut seen = 0_usize;
    while let Some(directory) = stack.pop() {
        for item in fs::read_dir(directory).map_err(|_| "Cannot enumerate payload.")? {
            let path: PathBuf = item.map_err(|_| "Cannot inspect payload entry.")?.path();
            seen += 1;
            if seen > pin.entries {
                return Err("Extra payload entry.".into());
            }
            let name = path
                .strip_prefix(&root)
                .ok()
                .and_then(Path::to_str)
                .ok_or("Invalid payload path encoding.")?;
            let entry = manifest
                .entries
                .get(name)
                .ok_or("Unrecorded payload entry.")?;
            let info = fs::symlink_metadata(&path).map_err(|_| "Cannot inspect payload entry.")?;
            let expected_mode = match entry {
                Entry::File {
                    bytes,
                    sha256,
                    mode,
                    ..
                } => {
                    hash_file(&path, *bytes, sha256)?;
                    *mode
                }
                Entry::Directory { mode } => {
                    if !info.is_dir() {
                        return Err("Payload directory type changed.".into());
                    }
                    stack.push(path);
                    *mode
                }
                Entry::Symlink { target, mode } => {
                    if !info.file_type().is_symlink()
                        || fs::read_link(&path)
                            .ok()
                            .as_ref()
                            .map(|value| value.as_os_str())
                            != Some(std::ffi::OsStr::new(target))
                    {
                        return Err("Payload symlink identity changed.".into());
                    }
                    let resolved = path
                        .canonicalize()
                        .map_err(|_| "Dangling or cyclic payload symlink.")?;
                    if !resolved.starts_with(&root) || !(resolved.is_file() || resolved.is_dir()) {
                        return Err("Payload symlink escapes or targets a special file.".into());
                    }
                    *mode
                }
            };
            if mode(&info) != expected_mode {
                return Err("Payload entry permissions changed.".into());
            }
        }
    }
    if seen != pin.entries {
        return Err("Missing payload entries.".into());
    }
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::os::unix::fs::{PermissionsExt, symlink};

    struct Fixture {
        folder: PathBuf,
        root: PathBuf,
        manifest: PathBuf,
        value: Value,
    }
    impl Fixture {
        fn new() -> Self {
            let mut random = [0_u8; 16];
            getrandom::fill(&mut random).unwrap();
            let folder = std::env::temp_dir().join(format!("payload-inventory-{}", hex(&random)));
            let root = folder.join("payload");
            fs::create_dir_all(root.join("_internal")).unwrap();
            fs::write(root.join("_internal/module"), b"payload library").unwrap();
            let value = json!({"schema_version":1,"entry_count":2,"file_bytes":15,
                "identity_sha256":"a".repeat(64),"entries":{
                "_internal":{"kind":"directory","mode":mode(&fs::metadata(root.join("_internal")).unwrap())},
                "_internal/module":{"kind":"file","mode":mode(&fs::metadata(root.join("_internal/module")).unwrap()),
                 "bytes":15,"sha256":hex(&Sha256::digest(b"payload library")),"macho_cpu_types":null}}});
            Self {
                manifest: folder.join("manifest.json"),
                folder,
                root,
                value,
            }
        }
        fn pin(&self) -> ManifestPin {
            let raw = serde_json::to_vec(&self.value).unwrap();
            fs::write(&self.manifest, &raw).unwrap();
            ManifestPin {
                sha256: Box::leak(hex(&Sha256::digest(&raw)).into_boxed_str()),
                bytes: raw.len() as u64,
                identity: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                entries: 2,
                file_bytes: 15,
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.folder).unwrap();
        }
    }

    #[test]
    fn full_manifest_accepts_unchanged_and_rejects_library_tampering() {
        let f = Fixture::new();
        let pin = f.pin();
        verify(&f.root, &f.manifest, pin).unwrap();
        fs::write(f.root.join("_internal/module"), b"changed library").unwrap();
        assert!(verify(&f.root, &f.manifest, pin).is_err());
    }
    #[test]
    fn additions_removals_modes_and_file_types_are_part_of_identity() {
        for case in 0..4 {
            let f = Fixture::new();
            let pin = f.pin();
            let file = f.root.join("_internal/module");
            match case {
                0 => fs::write(f.root.join("extra"), b"x").unwrap(),
                1 => fs::remove_file(file).unwrap(),
                2 => fs::set_permissions(file, fs::Permissions::from_mode(0o777)).unwrap(),
                _ => {
                    fs::remove_file(&file).unwrap();
                    symlink("../_internal", file).unwrap();
                }
            }
            assert!(verify(&f.root, &f.manifest, pin).is_err(), "case {case}");
        }
    }
    #[test]
    fn strict_manifest_rejects_types_unknown_fields_paths_and_bounds() {
        for (pointer, bad) in [
            ("/schema_version", json!(true)),
            ("/entry_count", json!(2.0)),
            ("/file_bytes", json!(16)),
            ("/surprise", json!(1)),
            ("/entries/_internal~1module/bytes", json!(true)),
            ("/entries/_internal~1module/mode", json!(65535)),
            ("/entries/_internal~1module/sha256", json!("bad")),
            ("/entries/_internal~1module/macho_cpu_types", json!([7])),
            ("/entries/_internal~1module/extra", json!(1)),
        ] {
            let mut f = Fixture::new();
            if pointer == "/surprise" {
                f.value["surprise"] = bad;
            } else if pointer.ends_with("/extra") {
                f.value["entries"]["_internal/module"]["extra"] = bad;
            } else {
                *f.value.pointer_mut(pointer).unwrap() = bad;
            }
            assert!(verify(&f.root, &f.manifest, f.pin()).is_err(), "{pointer}");
        }
        for path in [
            "/absolute",
            "../escape",
            "_internal//module",
            "_internal/./module",
            "back\\slash",
        ] {
            let mut f = Fixture::new();
            let entry = f.value["entries"]
                .as_object_mut()
                .unwrap()
                .remove("_internal/module")
                .unwrap();
            f.value["entries"][path] = entry;
            assert!(verify(&f.root, &f.manifest, f.pin()).is_err());
        }
        let f = Fixture::new();
        let mut pin = f.pin();
        pin.entries = MAX_ENTRIES + 1;
        assert!(verify(&f.root, &f.manifest, pin).is_err());
    }
    #[test]
    fn manifest_bytes_and_root_symlinks_are_not_adopted() {
        let f = Fixture::new();
        let pin = f.pin();
        fs::write(&f.manifest, b"{}").unwrap();
        assert!(verify(&f.root, &f.manifest, pin).is_err());
        let pin = f.pin();
        let link = f.folder.join("linked");
        symlink(&f.root, &link).unwrap();
        assert!(verify(&link, &f.manifest, pin).is_err());
    }
    #[test]
    fn literal_safe_symlinks_are_verified_but_escaping_dangling_and_cycles_fail() {
        for target in ["module", "./module", "missing", "loop", "../../outside"] {
            let mut f = Fixture::new();
            fs::write(f.folder.join("outside"), b"outside").unwrap();
            let link = f.root.join("_internal/loop");
            symlink(target, &link).unwrap();
            f.value["entries"]["_internal/loop"] = json!({"kind":"symlink","target":target,"mode":mode(&fs::symlink_metadata(&link).unwrap())});
            f.value["entry_count"] = json!(3);
            let mut pin = f.pin();
            pin.entries = 3;
            let outcome = verify(&f.root, &f.manifest, pin);
            assert_eq!(
                outcome.is_ok(),
                matches!(target, "module" | "./module"),
                "{target}: {outcome:?}"
            );
            if target == "./module" {
                fs::remove_file(&link).unwrap();
                symlink("module", &link).unwrap();
                assert!(verify(&f.root, &f.manifest, pin).is_err());
            }
        }
    }
}
