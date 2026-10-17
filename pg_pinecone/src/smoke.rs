//! Raw HTTP helper for the live tests. `SmokeIndex` owns a real Pinecone index
//! by name and deletes it (best effort) when dropped, so a live test's index is
//! cleaned up on every exit path — including a panic unwind.

use pg_pinecone_client::{API_VERSION, DOC_API_VERSION};
use ureq::delete;

const CONTROL_PLANE_URL: &str = "https://api.pinecone.io";

/// The key a live test may call Pinecone with, or why it skips: only with `PG_PINECONE_LIVE_TESTS=1` as
/// well as `PINECONE_API_KEY`, so a shell that exports a key never reaches Pinecone from a plain
/// `cargo pgrx test`.
pub fn live_gate(opt_in: Option<&str>, key: Option<String>) -> Result<String, &'static str> {
    if opt_in != Some("1") {
        return Err(
            "SKIPPED: live test; set PG_PINECONE_LIVE_TESTS=1 (and PINECONE_API_KEY) to run it",
        );
    }
    key.ok_or("SKIPPED: PINECONE_API_KEY not set")
}

/// A temporary serverless index, deleted (best effort) when dropped.
pub struct SmokeIndex {
    pub name: String,
    api_key: String,
}

impl SmokeIndex {
    /// Wraps an already-created index by name for cleanup only (its `Drop`
    /// impl), for tests where the code under test — not this helper — creates
    /// the index (e.g. `pinecone.promote`'s live round trip).
    pub fn adopt(api_key: &str, name: &str) -> Self {
        Self {
            name: name.to_string(),
            api_key: api_key.to_string(),
        }
    }
}

impl Drop for SmokeIndex {
    /// Best-effort cleanup on every exit path, including a panic unwind.
    fn drop(&mut self) {
        let _ = delete(format!("{CONTROL_PLANE_URL}/indexes/{}", self.name))
            .header("Api-Key", &self.api_key)
            .header("X-Pinecone-Api-Version", API_VERSION)
            .call();
    }
}

/// A temporary document (full-text) index, deleted (best effort) when dropped. Document indexes
/// live behind their own API version, so the delete says `DOC_API_VERSION`.
pub struct SmokeDocIndex {
    pub name: String,
    api_key: String,
}

impl SmokeDocIndex {
    /// Wraps an index the code under test creates (`pinecone.track_docs`), for cleanup only.
    pub fn adopt(api_key: &str, name: &str) -> Self {
        Self {
            name: name.to_string(),
            api_key: api_key.to_string(),
        }
    }
}

impl Drop for SmokeDocIndex {
    /// Best-effort cleanup on every exit path, including a panic unwind.
    fn drop(&mut self) {
        let _ = delete(format!("{CONTROL_PLANE_URL}/indexes/{}", self.name))
            .header("Api-Key", &self.api_key)
            .header("X-Pinecone-Api-Version", DOC_API_VERSION)
            .call();
    }
}

#[cfg(test)]
mod tests {
    use super::live_gate;

    #[test]
    fn a_live_test_needs_the_opt_in_and_a_key() {
        let key = || Some("pcsk-test".to_string());
        assert_eq!(live_gate(Some("1"), key()), Ok("pcsk-test".to_string()));
        for opt_in in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("yes"),
            Some("11"),
        ] {
            let why = live_gate(opt_in, key()).unwrap_err();
            assert!(
                why.contains("PG_PINECONE_LIVE_TESTS=1"),
                "{opt_in:?}: {why}"
            );
        }
        let why = live_gate(Some("1"), None).unwrap_err();
        assert!(why.contains("PINECONE_API_KEY"), "{why}");
        assert!(live_gate(None, None).is_err());
    }
}
