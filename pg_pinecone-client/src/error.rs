use thiserror::Error;

/// Client errors. No variant carries the raw API key, so these are safe to log or display.
#[derive(Debug, Error)]
pub enum Error {
    #[error("no API key configured — set Config::api_key before creating a client")]
    MissingApiKey,

    #[error(
        "index '{index}' not found — check the index name and that your API key belongs to the right project"
    )]
    IndexNotFound { index: String },

    #[error(
        "backup '{backup_id}' not found — check the backup id and that your API key belongs to the right project"
    )]
    BackupNotFound { backup_id: String },

    #[error(
        "namespace '{namespace}' not found — nothing has been written to it, or it was already deleted"
    )]
    NamespaceNotFound { namespace: String },

    #[error(
        "authentication failed — check that your API key is valid and has access to this project"
    )]
    Auth,

    #[error("bad request: {message}")]
    BadRequest { message: String },

    #[error("index '{name}' already exists")]
    Conflict { name: String },

    #[error(
        "request timed out after {ms}ms — the server may be overloaded; try increasing Config::timeout"
    )]
    Timeout { ms: u64 },

    #[error("could not reach {url}: {reason}")]
    Transport { url: String, reason: String },

    #[error("index API returned unexpected status {status}: {message}")]
    Api { status: u16, message: String },
}
