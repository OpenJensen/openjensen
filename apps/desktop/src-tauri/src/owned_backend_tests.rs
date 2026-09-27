use super::*;
use std::io::Cursor;

struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let mut random = [0_u8; 16];
        getrandom::fill(&mut random).unwrap();
        let root = std::env::temp_dir().join(format!("openjensen-owner-test-{}", hex(&random)));
        fs::create_dir(&root).unwrap();
        Self(root.canonicalize().unwrap())
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn payload() -> Payload {
    let pin = PayloadPin {
        manifest: crate::payload_manifest::ManifestPin {
            sha256: "unused",
            bytes: 0,
            identity: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            entries: 0,
            file_bytes: 0,
        },
        executable_sha256: "unused",
        executable_bytes: 0,
        resources_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        build_id: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    };
    Payload {
        folder: PathBuf::from("/not-used-by-test"),
        manifest: PathBuf::from("/not-used-by-test"),
        executable: PathBuf::from("/bin/sh"),
        resources: PathBuf::from("/not-used-by-test"),
        resources_sha256: pin.resources_sha256.into(),
        build_id: pin.build_id.into(),
        pin,
        fixture: Some(PathBuf::new()),
    }
}

fn event() -> serde_json::Value {
    serde_json::json!({"schema_version":1,"event":"ready","nonce":"0123456789abcdef0123456789abcdef","host":"127.0.0.1","port":23456,"build_id":"b".repeat(40),"app_version":"0.1.0","resources_sha256":"a".repeat(64),"workspace_id":"c".repeat(64)})
}

#[test]
fn ready_protocol_types_size_unknown_fields_duplicates_and_incomplete_lines() {
    let good = event();
    assert!(matches!(
        read_event(&mut Cursor::new(format!("{good}\n"))).unwrap(),
        Some(Event::Ready { port: 23456, .. })
    ));
    for (field, bad) in [
        ("schema_version", serde_json::json!(true)),
        ("port", serde_json::json!(23456.0)),
        ("port", serde_json::json!(-1)),
        ("nonce", serde_json::json!(null)),
        ("surprise", serde_json::json!(1)),
    ] {
        let mut value = good.clone();
        value[field] = bad;
        assert!(
            read_event(&mut Cursor::new(format!("{value}\n"))).is_err(),
            "{field}: {value}"
        );
    }
    for raw in [format!("{good}"), "x".repeat(MAX_LINE+1)+"\n", "{\"event\":\"stopped\",\"schema_version\":1,\"nonce\":\"a\",\"nonce\":\"b\",\"reason\":\"shutdown\"}\n".into()] {
        assert!(read_event(&mut Cursor::new(raw)).is_err());
    }
    assert!(
        read_event(&mut Cursor::new(Vec::<u8>::new()))
            .unwrap()
            .is_none()
    );
}

#[test]
#[cfg(not(feature = "local-payload-experiment"))]
fn production_start_is_disabled_even_when_directories_exist() {
    let scratch = Scratch::new();
    let controller = Controller::new(&scratch.0, scratch.0.join("data"), Backend::from_port(None));
    assert!(!controller.status().payload_available);
    assert!(controller.start(true).unwrap_err().contains("no verified"));
    assert!(!scratch.0.join("data").exists());
    assert!(
        controller
            .start(false)
            .unwrap_err()
            .contains("confirmation")
    );
}

#[test]
fn workspace_requires_explicit_owner_and_exact_build_without_adoption() {
    let scratch = Scratch::new();
    let payload = payload();
    let parent = scratch.0.join("workspaces");
    let root = workspace(&parent, &payload).unwrap();
    fs::write(root.join("project-evidence"), "preserve").unwrap();
    assert_eq!(workspace(&parent, &payload).unwrap(), root);
    let mut changed_payload = payload.clone();
    changed_payload.pin.manifest.identity =
        "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    assert!(workspace(&parent, &changed_payload).is_err());
    let mut other = payload.clone();
    other.build_id = "d".repeat(40);
    assert!(
        workspace(&parent, &other)
            .unwrap_err()
            .contains("another build")
    );
    fs::remove_file(root.join("desktop-owner.json")).unwrap();
    assert!(workspace(&parent, &payload).is_err());
    assert_eq!(
        fs::read_to_string(root.join("project-evidence")).unwrap(),
        "preserve"
    );
}

#[cfg(unix)]
#[test]
fn workspace_rejects_link_ancestors_and_existing_link_marker() {
    use std::os::unix::fs::symlink;
    let scratch = Scratch::new();
    let payload = payload();
    let root = workspace(&scratch.0.join("workspaces"), &payload).unwrap();
    fs::rename(root.join("desktop-owner.json"), root.join("saved.json")).unwrap();
    symlink(root.join("saved.json"), root.join("desktop-owner.json")).unwrap();
    assert!(workspace(&scratch.0.join("workspaces"), &payload).is_err());
    symlink(&scratch.0, scratch.0.join("alias")).unwrap();
    assert!(workspace(&scratch.0.join("alias/new"), &payload).is_err());
}

#[test]
fn private_commands_are_fixed_and_environment_is_explicit() {
    let payload = payload();
    let command = payload.command(Path::new("/workspace"));
    let args: Vec<_> = command
        .get_args()
        .map(|value| value.to_string_lossy().to_string())
        .collect();
    assert_eq!(
        &args[1..],
        [
            "serve",
            "--resources",
            "/not-used-by-test",
            "--data-dir",
            "/workspace"
        ]
    );
    let allowed = ["SYSTEMROOT", "WINDIR", "TMP", "TEMP", "TMPDIR"];
    for (name, _) in command.get_envs() {
        assert!(allowed.contains(&name.to_str().unwrap()));
    }
    assert_eq!(command.get_current_dir(), Some(Path::new("/workspace")));
}

#[cfg(unix)]
fn fixture(scratch: &Scratch, behavior: &str) -> Controller {
    let mut payload = payload();
    let parent = scratch.0.join("workspaces");
    let data = workspace(&parent, &payload).unwrap();
    let mut ready = event();
    ready["nonce"] = serde_json::json!("%s");
    ready["workspace_id"] = serde_json::json!(workspace_id(&data).unwrap());
    let ready = ready.to_string();
    let script = scratch.0.join("fixture.sh");
    let stopped = r#"printf '{"schema_version":1,"event":"stopped","nonce":"%s","reason":"shutdown"}\n' "$nonce""#;
    let good = format!(
        r#"printf '{ready}\n' "$nonce"
IFS= read -r stop
{stopped}"#
    );
    let body = match behavior {
        "good" => good,
        "wrong_nonce" => format!(r#"printf '{ready}\n' 'wrong'
IFS= read -r stop
exit 1"#),
        "wrong_build" => format!(r#"printf '{}\n' "$nonce"
IFS= read -r stop
exit 1"#, ready.replace(&"b".repeat(40), &"c".repeat(40))),
        "wrong_workspace" => format!(r#"printf '{}\n' "$nonce"
IFS= read -r stop
exit 1"#, ready.replace(&workspace_id(&data).unwrap(), &"d".repeat(64))),
        "zero_port" => format!(r#"printf '{}\n' "$nonce"
IFS= read -r stop
exit 1"#, ready.replace("23456", "0")),
        "flood" => format!(r#"i=0; while [ $i -lt 12000 ]; do printf 'diagnostic\n' >&2; i=$((i+1)); done
{good}"#),
        "stall_start" => "IFS= read -r stop\nexit 0".into(),
        "ignore_stop" => format!(r#"printf '{ready}\n' "$nonce"
IFS= read -r stop
while :; do :; done"#),
        "exit_after_ready" => format!(r#"printf '{ready}\n' "$nonce"
exit 1"#),
        "oversize" => r#"i=0; while [ $i -lt 5000 ]; do printf 'x'; i=$((i+1)); done; printf '\n'; IFS= read -r stop; exit 1"#.into(),
        _ => panic!("Unknown fixture"),
    };
    fs::write(
        &script,
        format!(
            r#"#!/bin/sh
IFS= read -r request
nonce=${{request#*\"nonce\":\"}}
nonce=${{nonce%%\"*}}
{body}
"#
        ),
    )
    .unwrap();
    payload.fixture = Some(script);
    let mut owner = Controller::with_payload(Ok(payload), parent, Backend::from_port(None));
    owner.limits = Limits {
        startup: Duration::from_secs(2),
        shutdown: Duration::from_millis(300),
        reap: Duration::from_secs(1),
    };
    owner
}

#[cfg(unix)]
#[test]
fn real_child_restart_changes_nonce_and_preserves_workspace_then_reaps() {
    let scratch = Scratch::new();
    let owner = fixture(&scratch, "good");
    let started = owner.start(true).unwrap();
    assert_eq!(started.mode, "owned");
    let nonce = lock(&owner.child).as_ref().unwrap().nonce.clone();
    let data = scratch.0.join("workspaces/desktop-v1");
    fs::write(data.join("keep"), b"immutable").unwrap();
    let next = owner.restart().unwrap();
    assert_eq!(next.generation, started.generation + 1);
    assert_ne!(lock(&owner.child).as_ref().unwrap().nonce, nonce);
    assert_eq!(fs::read(data.join("keep")).unwrap(), b"immutable");
    assert!(owner.permits_navigation(&tauri::Url::parse("http://127.0.0.1:23456/jobs").unwrap()));
    let stopped = owner.stop();
    assert_eq!(stopped.mode, "idle");
    assert!(!stopped.cleanup_unknown);
    assert!(lock(&owner.child).is_none());
    assert!(!owner.permits_navigation(&tauri::Url::parse("http://127.0.0.1:23456/").unwrap()));
}

#[cfg(unix)]
#[test]
fn malformed_identity_and_oversized_output_fail_and_reap() {
    for behavior in [
        "wrong_nonce",
        "wrong_build",
        "wrong_workspace",
        "zero_port",
        "oversize",
    ] {
        let scratch = Scratch::new();
        let owner = fixture(&scratch, behavior);
        assert!(owner.start(true).is_err(), "{behavior}");
        assert!(lock(&owner.child).is_none(), "{behavior}");
        assert!(owner.status().address.is_none());
    }
}

#[cfg(unix)]
#[test]
fn startup_timeout_and_ignored_shutdown_are_bounded_and_not_claimed_graceful() {
    let scratch = Scratch::new();
    let mut owner = fixture(&scratch, "stall_start");
    owner.limits.startup = Duration::from_millis(80);
    assert!(owner.start(true).unwrap_err().contains("supervisor budget"));
    assert!(lock(&owner.child).is_none());
    assert!(owner.status().cleanup_unknown);
    let scratch = Scratch::new();
    let owner = fixture(&scratch, "ignore_stop");
    owner.start(true).unwrap();
    let began = Instant::now();
    let stopped = owner.stop();
    assert!(stopped.cleanup_unknown);
    assert!(lock(&owner.child).is_none());
    assert!(began.elapsed() < Duration::from_secs(2));
    assert!(
        owner
            .start(true)
            .unwrap_err()
            .contains("cleanup is unverified")
    );
}

#[cfg(unix)]
#[test]
fn stderr_flood_is_drained_and_only_bounded_tail_retained() {
    let scratch = Scratch::new();
    let owner = fixture(&scratch, "flood");
    owner.start(true).unwrap();
    let ring = Arc::clone(&lock(&owner.child).as_ref().unwrap().stderr);
    assert!(!owner.stop().cleanup_unknown);
    assert_eq!(lock(&ring).len(), MAX_STDERR);
}

#[cfg(unix)]
#[test]
fn concurrent_starts_spawn_only_once_and_stop_can_cancel_startup() {
    let scratch = Scratch::new();
    let owner = Arc::new(fixture(&scratch, "good"));
    let second = Arc::clone(&owner);
    let first = thread::spawn(move || second.start(true));
    let result = owner.start(true);
    let other = first.join().unwrap();
    assert_ne!(result.is_ok(), other.is_ok());
    assert_eq!(owner.status().generation, 1);
    owner.stop();
    let scratch = Scratch::new();
    let owner = Arc::new(fixture(&scratch, "stall_start"));
    let second = Arc::clone(&owner);
    let task = thread::spawn(move || second.start(true));
    let deadline = Instant::now() + Duration::from_secs(2);
    while owner.status().mode != "starting" {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    owner.stop();
    assert!(task.join().unwrap().unwrap_err().contains("cancelled"));
    assert!(lock(&owner.child).is_none());
}

#[cfg(unix)]
#[test]
fn unexpected_exit_revokes_navigation_and_never_automatically_restarts() {
    let scratch = Scratch::new();
    let owner = fixture(&scratch, "exit_after_ready");
    let _ = owner.start(true);
    let deadline = Instant::now() + Duration::from_secs(2);
    while owner.status().mode == "owned" {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    assert!(owner.status().address.is_none());
    assert!(lock(&owner.child).is_none());
    assert_eq!(owner.status().generation, 1);
}

#[test]
fn attached_state_is_not_stopped_or_restarted_and_exit_blocks_new_start() {
    let scratch = Scratch::new();
    let owner = Controller::new(&scratch.0, scratch.0.join("data"), Backend::from_port(None));
    owner.update(
        "attached",
        Some("http://127.0.0.1:8000/".into()),
        false,
        "external",
    );
    assert_eq!(owner.stop().mode, "attached");
    assert!(owner.restart().unwrap_err().contains("only"));
    assert!(owner.permits_navigation(&tauri::Url::parse("http://127.0.0.1:8000/").unwrap()));
    assert!(!owner.permits_navigation(&tauri::Url::parse("http://127.0.0.1:8001/").unwrap()));
    assert!(owner.begin_exit());
    assert!(!owner.begin_exit());
    assert!(owner.start(true).unwrap_err().contains("cancelled"));
}

#[test]
#[cfg(unix)]
fn payload_pin_verifies_fixed_executable_and_manifest_before_admission() {
    let scratch = Scratch::new();
    let folder = scratch.0.join("sidecar/firebird-sidecar");
    let resources = folder.join("_internal/sidecar-resources");
    fs::create_dir_all(&resources).unwrap();
    let executable = folder.join(if cfg!(windows) {
        "firebird-sidecar.exe"
    } else {
        "firebird-sidecar"
    });
    fs::write(&executable, b"fixed bundled executable").unwrap();
    let manifest = serde_json::to_vec(&serde_json::json!({"schema_version":1,"build_id":"b".repeat(40),"app_version":"0.1.0","files":{"index.html":{"bytes":0,"sha256":"a".repeat(64)}}})).unwrap();
    fs::write(resources.join("resources.json"), &manifest).unwrap();
    use std::os::unix::fs::PermissionsExt;
    let mut entries = serde_json::Map::new();
    for name in [
        "_internal",
        "_internal/sidecar-resources",
        "_internal/sidecar-resources/resources.json",
        "firebird-sidecar",
    ] {
        let path = folder.join(name);
        let info = fs::metadata(&path).unwrap();
        let mode = info.permissions().mode() & 0o7777;
        let entry = if info.is_dir() {
            serde_json::json!({"kind":"directory","mode":mode})
        } else {
            serde_json::json!({"kind":"file","mode":mode,"bytes":info.len(),"sha256":hex(&Sha256::digest(fs::read(&path).unwrap())),"macho_cpu_types":null})
        };
        entries.insert(name.into(), entry);
    }
    let full_manifest=serde_json::to_vec(&serde_json::json!({"schema_version":1,"entries":entries,"entry_count":4,"file_bytes":24+manifest.len(),"identity_sha256":"c".repeat(64)})).unwrap();
    fs::write(
        scratch.0.join("sidecar/payload-manifest.json"),
        &full_manifest,
    )
    .unwrap();
    let pin = PayloadPin {
        manifest: crate::payload_manifest::ManifestPin {
            sha256: Box::leak(hex(&Sha256::digest(&full_manifest)).into_boxed_str()),
            bytes: full_manifest.len() as u64,
            identity: "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            entries: 4,
            file_bytes: 24 + manifest.len() as u64,
        },
        executable_sha256: Box::leak(
            hex(&Sha256::digest(b"fixed bundled executable")).into_boxed_str(),
        ),
        executable_bytes: 24,
        resources_sha256: Box::leak(hex(&Sha256::digest(&manifest)).into_boxed_str()),
        build_id: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    };
    let admitted = Payload::from_pin(&scratch.0, pin).unwrap();
    fs::write(resources.join("resources.json"), b"{}").unwrap();
    assert!(admitted.verify().is_err());
    fs::write(resources.join("resources.json"), manifest).unwrap();
    fs::write(&executable, b"other bundled executable").unwrap();
    assert!(admitted.verify().is_err());
}

#[test]
fn stopping_attached_mode_does_not_affect_the_actual_external_http_server() {
    use std::net::TcpListener;
    let scratch = Scratch::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let backend =
        Backend::from_port(Some(&listener.local_addr().unwrap().port().to_string())).unwrap();
    let task = thread::spawn(move || {
        for n in 0..4 {
            let until = Instant::now() + Duration::from_secs(5);
            let mut socket = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(Instant::now() < until);
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("{error}"),
                }
            };
            socket.set_nonblocking(false).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = [0; 4096];
            let _ = socket.read(&mut request);
            let (kind, body) = if n % 2 == 0 {
                ("application/json", r#"{"status":"ok","version":"0.1.0"}"#)
            } else {
                ("text/html", "<title>OPEN JENSEN</title>")
            };
            write!(socket,"HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
        }
    });
    let owner = Controller::new(&scratch.0, scratch.0.join("data"), Ok(backend));
    assert_eq!(owner.attach().status, "ready");
    assert_eq!(owner.stop().mode, "attached");
    owner.update("failed", None, true, "Previous owned cleanup is unknown");
    assert_eq!(owner.attach().status, "ready");
    assert!(owner.status().cleanup_unknown);
    assert!(
        owner
            .start(true)
            .unwrap_err()
            .contains("cleanup is unverified")
    );
    assert!(owner.restart().is_err());
    task.join().unwrap();
    assert!(!scratch.0.join("data").exists());
}

#[test]
fn compiled_candidate_identifier_cannot_adopt_the_normal_application_namespace() {
    #[cfg(feature = "local-payload-experiment")]
    {
        assert!(validate_identifier("dev.firebird.workbench").is_err());
        assert!(validate_identifier(LOCAL_EXPERIMENT_IDENTIFIER).is_ok());
    }
    #[cfg(not(feature = "local-payload-experiment"))]
    assert!(validate_identifier("dev.firebird.workbench").is_ok());
}
