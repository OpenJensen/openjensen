fn main() {
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
