#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod connection;

use std::time::Duration;
use tauri::menu::{MenuBuilder, MenuItem, SubmenuBuilder};
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

use connection::{Backend, Probe, is_connection_page, probe};

#[derive(Clone)]
struct Connection(Result<Backend, String>);

#[tauri::command]
async fn probe_backend(
    window: tauri::WebviewWindow,
    state: tauri::State<'_, Connection>,
) -> Result<Probe, String> {
    // The ACL also binds this command to bundled code. No command accepts an address.
    if !window.url().is_ok_and(|url| is_connection_page(&url)) {
        return Err("Only the bundled connection screen may perform readiness checks.".into());
    }
    let backend = match &state.0 {
        Ok(backend) => backend.clone(),
        Err(message) => return Ok(Probe::unavailable(message)),
    };
    tauri::async_runtime::spawn_blocking(move || probe(&backend, Duration::from_secs(2)))
        .await
        .map_err(|_| "The local connection check was interrupted.".into())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let port = std::env::var("FIREBIRD_DESKTOP_PORT");
    let connection = Connection(match port {
        Ok(ref value) => Backend::from_port(Some(value)),
        Err(std::env::VarError::NotPresent) => Backend::from_port(None),
        Err(_) => Err("FIREBIRD_DESKTOP_PORT must contain a valid decimal port.".into()),
    });
    if std::env::args().any(|arg| arg == "--self-check") {
        let result = match &connection.0 {
            Ok(backend) => probe(backend, Duration::from_secs(2)),
            Err(message) => Probe::unavailable(message),
        };
        println!("{}", serde_json::to_string(&result)?);
        if result.status != "ready" {
            std::process::exit(1);
        }
        return Ok(());
    }
    let navigation = connection.clone();
    tauri::Builder::default()
        .manage(connection)
        .invoke_handler(tauri::generate_handler![probe_backend])
        .setup(move |app| {
            let reconnect = MenuItem::with_id(
                app,
                "connection",
                "Connection",
                true,
                Some("CmdOrCtrl+Shift+R"),
            )?;
            let application = SubmenuBuilder::new(app, "Firebird")
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
                .title("Firebird")
                .inner_size(1280.0, 840.0)
                .min_inner_size(780.0, 560.0)
                .on_navigation(move |url| {
                    navigation.0.as_ref().map_or_else(
                        |_| is_connection_page(url),
                        |backend| backend.permits_navigation(url),
                    )
                })
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
            let connection_url = match tauri::Url::parse(address) {
                Ok(url) => url,
                Err(error) => {
                    eprintln!("Invalid internal connection address: {error}");
                    return;
                }
            };
            if let Err(error) = window.navigate(connection_url) {
                eprintln!("Cannot return to the desktop connection screen: {error}");
            }
        })
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::Destroyed) {
                // This process owns only a window. No backend child or cloud job is stopped.
                window.app_handle().exit(0);
            }
        })
        .run(tauri::generate_context!())?;
    Ok(())
}
