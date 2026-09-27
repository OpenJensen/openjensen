use std::io::Read;

fn main() {
    println!("cargo:rerun-if-env-changed=FIREBIRD_DESKTOP_EXPERIMENT_PIN");
    if std::env::var_os("CARGO_FEATURE_LOCAL_PAYLOAD_EXPERIMENT").is_some() {
        assert_eq!(
            std::env::var("CARGO_CFG_TARGET_OS").as_deref(),
            Ok("macos"),
            "Local payload experiment is macOS-only"
        );
        assert_eq!(
            std::env::var("CARGO_CFG_TARGET_ARCH").as_deref(),
            Ok("aarch64"),
            "Local payload experiment is ARM64-only"
        );
        let input = std::path::PathBuf::from(
            std::env::var_os("FIREBIRD_DESKTOP_EXPERIMENT_PIN")
                .expect("Explicit generated local experiment pin is required"),
        );
        assert!(input.is_absolute(), "Generated pin path must be absolute");
        let meta = std::fs::symlink_metadata(&input).expect("Generated pin is missing");
        assert!(
            meta.is_file() && meta.len() <= 16384,
            "Generated pin must be a bounded regular file"
        );
        let mut raw = Vec::new();
        std::fs::File::open(&input)
            .expect("Cannot open generated pin")
            .take(16385)
            .read_to_end(&mut raw)
            .expect("Cannot read generated pin");
        assert!(
            raw.len() <= 16384
                && raw
                    .starts_with(b"// Generated local experiment pin. No production activation.\n"),
            "Invalid generated pin header or size"
        );
        println!("cargo:rerun-if-changed={}", input.display());
        let output =
            std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("Cargo OUT_DIR missing"));
        std::fs::write(output.join("local-payload-pin.rs"), raw)
            .expect("Cannot stage generated pin");
    }
    tauri_build::try_build(tauri_build::Attributes::new().app_manifest(
        tauri_build::AppManifest::new().commands(&[
            "probe_backend",
            "desktop_status",
            "start_owned",
            "stop_owned",
            "restart_owned",
        ]),
    ))
    .expect("Failed to build the desktop capability manifest");
}
