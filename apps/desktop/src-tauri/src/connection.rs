//! Read-only readiness checks for one explicitly selected loopback application.
use std::io::Read;
use std::time::Duration;

use reqwest::blocking::{Client, Response};
use serde::{Deserialize, Serialize};
use tauri::Url;

const MAX_HEALTH_BYTES: u64 = 16 * 1024;
const MAX_HTML_BYTES: u64 = 512 * 1024;

#[derive(Clone, Debug)]
pub struct Backend {
    pub url: Url,
}

impl Backend {
    pub fn from_port(value: Option<&str>) -> Result<Self, String> {
        let raw = value.unwrap_or("8000");
        if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
            return Err("FIREBIRD_DESKTOP_PORT must be a decimal port from 1 to 65535.".into());
        }
        let port = raw
            .parse::<u16>()
            .ok()
            .filter(|port| *port != 0)
            .ok_or_else(|| "FIREBIRD_DESKTOP_PORT must be from 1 to 65535.".to_string())?;
        let url = Url::parse(&format!("http://127.0.0.1:{port}/"))
            .map_err(|_| "Could not construct the local application address.".to_string())?;
        Ok(Self { url })
    }

    pub fn permits_navigation(&self, url: &Url) -> bool {
        is_connection_page(url)
            || (url.origin() == self.url.origin()
                && url.username().is_empty()
                && url.password().is_none())
    }
}

pub fn is_connection_page(url: &Url) -> bool {
    let origin = (url.scheme() == "tauri" && url.host_str() == Some("localhost"))
        || (url.scheme() == "http" && url.host_str() == Some("tauri.localhost"));
    origin
        && url.port().is_none()
        && url.username().is_empty()
        && url.password().is_none()
        && matches!(url.path(), "" | "/" | "/index.html")
        && url.query().is_none()
        && url.fragment().is_none()
}

#[derive(Deserialize)]
struct Health {
    status: String,
    version: String,
}

#[derive(Serialize, Debug)]
pub struct Probe {
    pub status: &'static str,
    pub address: Option<String>,
    pub version: Option<String>,
    pub message: String,
}

impl Probe {
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: "unavailable",
            address: None,
            version: None,
            message: message.into(),
        }
    }
}

fn read_bounded(response: Response, limit: u64) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    response
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "The local application response could not be read.".to_string())?;
    if bytes.len() as u64 > limit {
        return Err("The local application response exceeded the readiness-check limit.".into());
    }
    Ok(bytes)
}

fn check(backend: &Backend, timeout: Duration) -> Result<String, String> {
    let client = Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(timeout)
        .connect_timeout(timeout.min(Duration::from_millis(750)))
        .build()
        .map_err(|_| "Could not initialize the local connection check.".to_string())?;
    let health_url = backend
        .url
        .join("api/v1/health")
        .map_err(|_| "Could not construct the health-check address.".to_string())?;
    let response = client.get(health_url).send().map_err(|_| {
        format!(
            "Cannot reach {}. Start Firebird locally, then retry.",
            backend.url
        )
    })?;
    if response.status() != reqwest::StatusCode::OK {
        return Err(
            "The local application health check did not succeed. Redirects are not followed."
                .into(),
        );
    }
    let health: Health = serde_json::from_slice(&read_bounded(response, MAX_HEALTH_BYTES)?)
        .map_err(|_| {
            "The local server did not return a valid Firebird health response.".to_string()
        })?;
    if health.status != "ok" || health.version != env!("CARGO_PKG_VERSION") {
        return Err(format!(
            "This desktop requires Firebird {}. Check the local application version and retry.",
            env!("CARGO_PKG_VERSION")
        ));
    }
    let response = client.get(backend.url.clone()).send().map_err(|_| {
        "The API is reachable, but the web interface could not be loaded.".to_string()
    })?;
    if response.status() != reqwest::StatusCode::OK
        || !response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.split(';').next() == Some("text/html"))
    {
        return Err("The API is reachable, but the built Firebird web interface is missing. Build the web app and restart Firebird, then retry.".into());
    }
    let html = read_bounded(response, MAX_HTML_BYTES)?;
    if !String::from_utf8_lossy(&html).contains("<title>Firebird") {
        return Err("The local server is not serving the expected Firebird web interface.".into());
    }
    Ok(health.version)
}

pub fn probe(backend: &Backend, timeout: Duration) -> Probe {
    match check(backend, timeout) {
        Ok(version) => Probe {
            status: "ready",
            address: Some(backend.url.to_string()),
            version: Some(version),
            message: "The local API and static web interface are ready.".into(),
        },
        Err(message) => Probe::unavailable(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;

    fn accept_bounded(listener: &TcpListener) -> std::net::TcpStream {
        listener.set_nonblocking(true).expect("bounded test accept");
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        loop {
            match listener.accept() {
                Ok((stream, _)) => {
                    stream
                        .set_nonblocking(false)
                        .expect("blocking accepted stream");
                    return stream;
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "Expected local request before test deadline"
                    );
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("Test listener failed: {error}"),
            }
        }
    }

    fn serve(replies: Vec<String>) -> (Backend, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener");
        let backend = Backend::from_port(Some(
            &listener.local_addr().expect("address").port().to_string(),
        ))
        .expect("test backend");
        let task = thread::spawn(move || {
            for reply in replies {
                let mut stream = accept_bounded(&listener);
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .expect("read timeout");
                let mut request = [0; 2048];
                let _ = stream.read(&mut request);
                let _ = stream.write_all(reply.as_bytes());
            }
        });
        (backend, task)
    }

    fn response(status: u16, content_type: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status} Test\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn health() -> String {
        response(
            200,
            "application/json",
            r#"{"status":"ok","version":"0.1.0"}"#,
        )
    }

    #[test]
    fn only_decimal_nonzero_ports_are_configuration() {
        assert_eq!(
            Backend::from_port(None).expect("default").url.as_str(),
            "http://127.0.0.1:8000/"
        );
        for value in [
            "",
            "0",
            "65536",
            "-1",
            "+8000",
            " 8000",
            "8000/",
            "evil.example",
            "8000?token=secret",
        ] {
            assert!(Backend::from_port(Some(value)).is_err(), "{value}");
        }
    }

    #[test]
    fn navigation_stays_in_configured_origin_or_exact_bundled_connection_page() {
        let backend = Backend::from_port(None).expect("backend");
        for value in [
            "http://127.0.0.1:8000/",
            "http://127.0.0.1:8000/docs/?q=a",
            "tauri://localhost/index.html",
            "tauri://localhost",
            "http://tauri.localhost/",
        ] {
            assert!(
                backend.permits_navigation(&Url::parse(value).expect("url")),
                "{value}"
            );
        }
        for value in [
            "https://example.com/",
            "http://127.0.0.1:8001/",
            "http://localhost:8000/",
            "http://127.0.0.1.evil:8000/",
            "http://user@127.0.0.1:8000/",
            "file:///tmp/a",
            "tauri://evil/",
            "tauri://localhost/other",
            "http://tauri.localhost:8000/",
            "tauri://localhost/?next=evil",
        ] {
            assert!(
                !backend.permits_navigation(&Url::parse(value).expect("url")),
                "{value}"
            );
        }
    }

    #[test]
    fn both_api_and_exported_frontend_are_required() {
        let (backend, server) = serve(vec![
            health(),
            response(
                200,
                "text/html; charset=utf-8",
                "<title>Firebird · Dataset workspace</title>",
            ),
        ]);
        let result = probe(&backend, Duration::from_secs(1));
        assert_eq!(result.status, "ready");
        assert_eq!(result.version.as_deref(), Some("0.1.0"));
        server.join().expect("server");
    }

    #[test]
    fn unhealthy_wrong_version_malformed_oversized_and_redirect_are_rejected() {
        for reply in [response(503, "application/json", "{}"), response(200, "application/json", "not json"), response(200, "application/json", r#"{"status":"ok","version":"99.0"}"#), response(200, "application/json", &"x".repeat(16385)), "HTTP/1.1 302 Found\r\nLocation: https://example.com/\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into()] {
            let (backend, server) = serve(vec![reply]);
            assert_eq!(probe(&backend, Duration::from_secs(1)).status, "unavailable");
            server.join().expect("server");
        }
    }

    #[test]
    fn api_only_or_other_web_page_is_not_ready() {
        for reply in [
            response(404, "application/json", "{}"),
            response(200, "text/html", "<title>Another application</title>"),
            response(200, "application/json", "{}"),
            response(200, "text/html", &"x".repeat(524289)),
        ] {
            let (backend, server) = serve(vec![health(), reply]);
            assert_eq!(
                probe(&backend, Duration::from_secs(1)).status,
                "unavailable"
            );
            server.join().expect("server");
        }
    }

    #[test]
    fn refused_connection_is_retryable_and_bounded() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener");
        let backend = Backend::from_port(Some(
            &listener.local_addr().expect("address").port().to_string(),
        ))
        .expect("backend");
        drop(listener);
        assert_eq!(
            probe(&backend, Duration::from_millis(200)).status,
            "unavailable"
        );
    }

    #[test]
    fn stalled_health_response_times_out() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener");
        let backend = Backend::from_port(Some(
            &listener.local_addr().expect("address").port().to_string(),
        ))
        .expect("backend");
        let server = thread::spawn(move || {
            let _stream = accept_bounded(&listener);
            thread::sleep(Duration::from_millis(200));
        });
        let start = std::time::Instant::now();
        let result = probe(&backend, Duration::from_millis(40));
        assert_eq!(result.status, "unavailable");
        assert!(start.elapsed() < Duration::from_secs(1));
        server.join().expect("server");
    }

    #[test]
    fn bundled_capability_has_no_remote_grant_or_system_permissions() {
        let capability: serde_json::Value =
            serde_json::from_str(include_str!("../capabilities/connection-screen.json"))
                .expect("capability JSON");
        assert_eq!(capability["local"], true);
        assert!(capability.get("remote").is_none());
        assert_eq!(
            capability["permissions"],
            serde_json::json!(["allow-probe-backend"])
        );
    }
}
