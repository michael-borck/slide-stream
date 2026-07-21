// SlideStream Desktop — a thin Tauri shell around `slide-stream serve`.
//
// On launch it (1) finds or installs the `uv` Python manager, (2) starts
// `slide-stream serve` in local mode on a free 127.0.0.1 port via `uvx`
// (which creates/caches the Python env on first run), then (3) navigates
// the webview to that URL. Closing the window shuts the server down.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{Emitter, Manager, State};

struct Srv {
    child: Mutex<Option<Child>>,
    port: Mutex<u16>,
    running: AtomicBool,
}

fn home() -> PathBuf {
    PathBuf::from(
        std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_default(),
    )
}

/// Locate `uv`. GUI apps get a minimal PATH (especially on macOS), so probe
/// the standard install locations explicitly.
fn find_uv() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    let mut candidates: Vec<PathBuf> = vec![
        home().join(".local").join("bin").join(exe),
        home().join(".cargo").join("bin").join(exe),
    ];
    if !cfg!(windows) {
        candidates.push(PathBuf::from("/opt/homebrew/bin/uv"));
        candidates.push(PathBuf::from("/usr/local/bin/uv"));
    }
    for c in &candidates {
        if c.is_file() {
            return Some(c.clone());
        }
    }
    // Fall back to PATH.
    let ok = Command::new(exe)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        Some(PathBuf::from(exe))
    } else {
        None
    }
}

fn has_cmd(name: &str) -> bool {
    Command::new(name)
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(8471)
}

fn emit_status(app: &tauri::AppHandle, phase: &str, msg: &str, url: &str) {
    let _ = app.emit(
        "status",
        serde_json::json!({"phase": phase, "msg": msg, "url": url}),
    );
}

/// Best-effort clean shutdown: POST /api/quit, then kill the launcher process.
fn stop_server(state: &Srv) {
    let port = *state.port.lock().unwrap();
    if port != 0 {
        if let Ok(mut s) = TcpStream::connect(("127.0.0.1", port)) {
            let _ = s.write_all(
                format!(
                    "POST /api/quit HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
                .as_bytes(),
            );
            std::thread::sleep(Duration::from_millis(400));
        }
    }
    if let Some(mut child) = state.child.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[tauri::command]
fn install_uv() -> Result<String, String> {
    let status = if cfg!(windows) {
        Command::new("powershell")
            .args([
                "-ExecutionPolicy",
                "ByPass",
                "-Command",
                "irm https://astral.sh/uv/install.ps1 | iex",
            ])
            .status()
    } else {
        Command::new("sh")
            .args(["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
            .status()
    };
    match status {
        Ok(s) if s.success() => Ok("installed".into()),
        Ok(s) => Err(format!("installer exited with {s}")),
        Err(e) => Err(format!("could not run installer: {e}")),
    }
}

#[tauri::command]
fn bootstrap(app: tauri::AppHandle, state: State<'_, Srv>) {
    if state.running.swap(true, Ordering::SeqCst) {
        return; // already starting/started
    }
    let app = app.clone();
    std::thread::spawn(move || {
        let state = app.state::<Srv>();
        emit_status(&app, "checking", "Checking runtime…", "");

        let Some(uv) = find_uv() else {
            state.running.store(false, Ordering::SeqCst);
            emit_status(
                &app,
                "uv-missing",
                "The uv Python manager is not installed.",
                "",
            );
            return;
        };
        if !has_cmd("ffmpeg") {
            emit_status(
                &app,
                "note",
                "ffmpeg was not found — voice cloning and avatars may not work. Install it from ffmpeg.org (or `brew install ffmpeg`).",
                "",
            );
        }

        let port = free_port();
        *state.port.lock().unwrap() = port;
        emit_status(
            &app,
            "starting",
            "Preparing SlideStream (the first run downloads its engine — a few minutes)…",
            "",
        );

        let spawned = Command::new(&uv)
            .args([
                "tool",
                "run",
                "--from",
                "slide-stream[all]>=2.12",
                "slide-stream",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--no-browser",
            ])
            .env("SLIDESTREAM_LOCAL", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let mut child = match spawned {
            Ok(c) => c,
            Err(e) => {
                state.running.store(false, Ordering::SeqCst);
                emit_status(&app, "error", &format!("Could not start uv: {e}"), "");
                return;
            }
        };

        // Stream install/server output to the launcher UI.
        for pipe in [
            child.stdout.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
            child.stderr.take().map(|s| Box::new(s) as Box<dyn std::io::Read + Send>),
        ]
        .into_iter()
        .flatten()
        {
            let app2 = app.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                    let _ = app2.emit("log", line);
                }
            });
        }

        *state.child.lock().unwrap() = Some(child);

        // Wait (generously, for the first-run download) for the port to open.
        let url = format!("http://127.0.0.1:{port}/");
        let deadline = Instant::now() + Duration::from_secs(900);
        loop {
            if TcpStream::connect_timeout(
                &format!("127.0.0.1:{port}").parse().unwrap(),
                Duration::from_millis(400),
            )
            .is_ok()
            {
                emit_status(&app, "ready", "Ready", &url);
                return;
            }
            // Died before it came up?
            if let Some(c) = state.child.lock().unwrap().as_mut() {
                if let Ok(Some(status)) = c.try_wait() {
                    state.running.store(false, Ordering::SeqCst);
                    emit_status(
                        &app,
                        "error",
                        &format!("SlideStream exited early ({status}). See the log above."),
                        "",
                    );
                    return;
                }
            }
            if Instant::now() > deadline {
                state.running.store(false, Ordering::SeqCst);
                emit_status(&app, "error", "Timed out waiting for SlideStream to start.", "");
                return;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
    });
}

fn main() {
    tauri::Builder::default()
        .manage(Srv {
            child: Mutex::new(None),
            port: Mutex::new(0),
            running: AtomicBool::new(false),
        })
        .invoke_handler(tauri::generate_handler![bootstrap, install_uv])
        .build(tauri::generate_context!())
        .expect("error building SlideStream")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                stop_server(&app_handle.state::<Srv>());
            }
        });
}
