//! In-backend wiremock harness.
//!
//! The Postgres backend process is synchronous, so wiremock (async) runs on a
//! dedicated OS thread that owns its own current-thread tokio runtime. The
//! thread blocks on that runtime for its whole lifetime, which is what keeps
//! the mock servers' listener tasks polled and alive while the test drives
//! the extension's functions through SPI on the backend's own thread.

use std::{future::Future, sync::mpsc, thread};

use tokio::{runtime::Builder, sync::oneshot};
use wiremock::MockServer;

pub mod docs;
pub mod vectors;

/// A control-plane + data-plane mock pair, alive for as long as this value lives.
/// Only the control-plane URL is exposed: that's the one the extension talks
/// to directly, and it's what tests point `pinecone.control_plane_url` at. The
/// data-plane URL is only needed inside `setup`, to embed in the mocked control
/// plane's `host` response.
pub struct MockBackend {
    pub control_plane_url: String,
    shutdown: Option<oneshot::Sender<()>>,
    thread: Option<thread::JoinHandle<()>>,
}

impl MockBackend {
    /// Starts both servers on a background thread and runs `setup` (to mount
    /// mocks) before their base URLs are sent back to the caller.
    pub fn start<F, Fut>(setup: F) -> Self
    where
        F: FnOnce(MockServer, MockServer) -> Fut + Send + 'static,
        Fut: Future<Output = (MockServer, MockServer)> + Send + 'static,
    {
        let (addr_tx, addr_rx) = mpsc::channel::<String>();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let thread = thread::spawn(move || {
            let runtime = Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build tokio runtime for mock backend");
            runtime.block_on(async move {
                let control_plane = MockServer::start().await;
                let data_plane = MockServer::start().await;
                let (control_plane, data_plane) = setup(control_plane, data_plane).await;
                addr_tx
                    .send(control_plane.uri())
                    .expect("send mock server address");
                // Keep polling the servers' listener tasks until told to stop.
                let _ = shutdown_rx.await;
                drop((control_plane, data_plane));
            });
        });

        let control_plane_url = addr_rx.recv().expect("receive mock server address");
        Self {
            control_plane_url,
            shutdown: Some(shutdown_tx),
            thread: Some(thread),
        }
    }
}

impl Drop for MockBackend {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
