//! A signal that lands while the client waits for a response does not make it send the request
//! again: Postgres backends take signals all the time, and a repeat is a second delete, upsert or
//! query on the wire.
#![cfg(unix)]

use std::{
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use pg_pinecone_client::{Client, Config};
use serde_json::json;

const DELAY: Duration = Duration::from_millis(400);

type Seen = Arc<Mutex<Vec<String>>>;

/// Records `METHOD path` of every request; the data plane answers a `DELETE` after [`DELAY`].
fn start() -> (String, Seen) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    let seen = Seen::default();
    let (host, log) = (url.clone(), seen.clone());
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let (host, log) = (host.clone(), log.clone());
            thread::spawn(move || serve(stream, &host, &log));
        }
    });
    (url, seen)
}

fn serve(stream: TcpStream, host: &str, seen: &Seen) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone"));
    let mut writer = stream;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let mut words = line.split_whitespace();
        let (method, path) = (
            words.next().unwrap_or_default(),
            words.next().unwrap_or_default(),
        );
        seen.lock().expect("seen").push(format!("{method} {path}"));
        let mut length = 0usize;
        loop {
            let mut header = String::new();
            if reader.read_line(&mut header).unwrap_or(0) == 0 {
                return;
            }
            if header == "\r\n" {
                break;
            }
            if let Some((name, value)) = header.split_once(':')
                && name.eq_ignore_ascii_case("content-length")
            {
                length = value.trim().parse().unwrap_or(0);
            }
        }
        if reader.read_exact(&mut vec![0u8; length]).is_err() {
            return;
        }
        let body = if method == "DELETE" {
            thread::sleep(DELAY);
            "{}".to_string()
        } else {
            json!({"host": host}).to_string()
        };
        let reply = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        if writer.write_all(reply.as_bytes()).is_err() {
            return;
        }
    }
}

extern "C" fn nothing(_: libc::c_int) {}

/// A handler under `SA_RESTART`, as Postgres installs its own.
fn handle_sigusr1() {
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = nothing as *const () as usize;
        action.sa_flags = libc::SA_RESTART;
        libc::sigemptyset(&mut action.sa_mask);
        assert_eq!(
            libc::sigaction(libc::SIGUSR1, &action, std::ptr::null_mut()),
            0
        );
    }
}

#[test]
fn a_signal_while_waiting_for_the_response_sends_the_request_once() {
    handle_sigusr1();
    let (url, seen) = start();
    let client = Client::new(Config {
        control_plane_url: url,
        api_key: "key".to_string(),
        timeout: Duration::from_secs(5),
        user_agent: None,
    });

    let caller = unsafe { libc::pthread_self() } as usize;
    let signals = thread::spawn(move || {
        for _ in 0..3 {
            thread::sleep(Duration::from_millis(80));
            unsafe { libc::pthread_kill(caller as libc::pthread_t, libc::SIGUSR1) };
        }
    });
    client.delete_namespace("idx", "ns").expect("delete");
    signals.join().expect("signals");

    let seen = seen.lock().expect("seen");
    assert_eq!(*seen, ["GET /indexes/idx", "DELETE /namespaces/ns"]);
}
