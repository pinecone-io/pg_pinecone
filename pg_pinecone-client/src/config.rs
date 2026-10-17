use std::{fmt, time::Duration};

/// Client configuration.
#[derive(Clone)]
pub struct Config {
    /// Base URL of the Pinecone control plane, e.g. `https://api.pinecone.io`.
    pub control_plane_url: String,
    pub api_key: String,
    /// Applied to both the control-plane host lookup and the data-plane query.
    pub timeout: Duration,
    /// `User-Agent` sent with every request; `None` uses this crate's default
    /// (see `Client::with_sleep`).
    pub user_agent: Option<String>,
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("control_plane_url", &self.control_plane_url)
            .field("api_key", &"<redacted>")
            .field("timeout", &self.timeout)
            .field("user_agent", &self.user_agent)
            .finish()
    }
}
