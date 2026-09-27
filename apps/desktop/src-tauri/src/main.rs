#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod connection;
mod owned_backend;
mod payload_manifest;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tauri::menu::{MenuBuilder, MenuItem, SubmenuBuilder};
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

use connection::{Backend, Probe, is_connection_page, probe};
use owned_backend::{Controller, DesktopStatus};

#[derive(Clone)]
struct Connection(Arc<Controller>);

fn bundled(window: &tauri::WebviewWindow) -> Result<(), String> {
    if window.url().is_ok_and(|url| is_connection_page(&url)) {
        Ok(())
    } else {
        Err("Only the bundled connection screen may use desktop ownership commands.".into())
    }
}

#[tauri::command]
async fn probe_backend(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, Connection>,
) -> Result<Probe, String> {
    bundled(&window)?;
    let owner = Arc::clone(&state.0);
    tauri::async_runtime::spawn_blocking(move || owner.attach())
        .await
        .map_err(|_| "The local connection check was interrupted.".into())
}

#[tauri::command]
fn desktop_status(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, Connection>,
) -> Result<DesktopStatus, String> {
    bundled(&window)?;
    Ok(state.0.status())
}

#[tauri::command]
async fn start_owned(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, Connection>,
    confirmed: bool,
) -> Result<DesktopStatus, String> {
    bundled(&window)?;
    let owner = Arc::clone(&state.0);
    tauri::async_runtime::spawn_blocking(move || owner.start(confirmed))
        .await
        .map_err(|_| "Owned backend startup was interrupted.".to_string())?
}

#[tauri::command]
async fn stop_owned(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, Connection>,
) -> Result<DesktopStatus, String> {
    bundled(&window)?;
    let owner = Arc::clone(&state.0);
    tauri::async_runtime::spawn_blocking(move || owner.stop())
        .await
        .map_err(|_| "Owned backend shutdown was interrupted.".into())
}

#[tauri::command]
async fn restart_owned(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, Connection>,
) -> Result<DesktopStatus, String> {
    bundled(&window)?;
    let owner = Arc::clone(&state.0);
    tauri::async_runtime::spawn_blocking(move || owner.restart())
        .await
        .map_err(|_| "Owned backend restart was interrupted.".to_string())?
}

fn shutdown(app: &tauri::AppHandle, complete: &Arc<AtomicBool>) {
    let owner = Arc::clone(&app.state::<Connection>().0);
    if !owner.begin_exit() {
        return;
    }
    let app = app.clone();
    let complete = Arc::clone(complete);
    tauri::async_runtime::spawn_blocking(move || {
        let result = owner.stop();
        if result.cleanup_unknown {
            eprintln!("Desktop exit: owned backend or nested-worker cleanup is unverified.");
        }
        complete.store(true, Ordering::SeqCst);
        app.exit(0);
    });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let attached = match std::env::var("FIREBIRD_DESKTOP_PORT") {
        Ok(ref value) => Backend::from_port(Some(value)),
        Err(std::env::VarError::NotPresent) => Backend::from_port(None),
        Err(_) => Err("FIREBIRD_DESKTOP_PORT must contain a valid decimal port.".into()),
    };
    if std::env::args().any(|arg| arg == "--self-check") {
        let result = match &attached {
            Ok(backend) => probe(backend, Duration::from_secs(2)),
            Err(message) => Probe::unavailable(message),
        };
        println!("{}", serde_json::to_string(&result)?);
        if result.status != "ready" {
            std::process::exit(1);
        }
        return Ok(());
    }
    let exit_complete = Arc::new(AtomicBool::new(false));
    let window_exit = Arc::clone(&exit_complete);
    let app = tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            probe_backend,
            desktop_status,
            start_owned,
            stop_owned,
            restart_owned
        ])
        .setup(move |app| {
            owned_backend::validate_identifier(&app.config().identifier)
                .map_err(std::io::Error::other)?;
            let owner = Arc::new(Controller::new(
                &app.path().resource_dir()?,
                app.path().app_local_data_dir()?,
                attached,
            ));
            app.manage(Connection(Arc::clone(&owner)));
            let reconnect = MenuItem::with_id(
                app,
                "connection",
                "Connection",
                true,
                Some("CmdOrCtrl+Shift+R"),
            )?;
            let application = SubmenuBuilder::new(app, "OPEN JENSEN")
                .item(&reconnect)
                .separator()
                .quit()
                .build()?;
            let edit = SubmenuBuilder::new(app, "Edit")
                .undo()
                .redo()
                .separator()
                .cut()
                .copy()
                .paste()
                .select_all()
                .build()?;
            app.set_menu(
                MenuBuilder::new(app)
                    .items(&[&application, &edit])
                    .build()?,
            )?;
            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("OPEN JENSEN")
                .inner_size(1280.0, 840.0)
                .min_inner_size(780.0, 560.0)
                .on_navigation(move |url| owner.permits_navigation(url))
                .on_new_window(|_, _| tauri::webview::NewWindowResponse::Deny)
                .build()?;
            Ok(())
        })
        .on_menu_event(|app, event| {
            if event.id().as_ref() != "connection" {
                return;
            }
            let Some(window) = app.get_webview_window("main") else {
                return;
            };
            let address = if cfg!(target_os = "windows") {
                "http://tauri.localhost/index.html"
            } else {
                "tauri://localhost/index.html"
            };
            if let Ok(url) = tauri::Url::parse(address)
                && let Err(error) = window.navigate(url)
            {
                eprintln!("Cannot return to the desktop connection screen: {error}");
            }
        })
        .on_window_event(move |window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event
                && !window_exit.load(Ordering::SeqCst)
            {
                api.prevent_close();
                shutdown(window.app_handle(), &window_exit);
            }
        })
        .build(tauri::generate_context!())?;
    app.run(move |app, event| {
        if let tauri::RunEvent::ExitRequested { api, .. } = event
            && !exit_complete.load(Ordering::SeqCst)
        {
            api.prevent_exit();
            shutdown(app, &exit_complete);
        }
    });
    Ok(())
}
