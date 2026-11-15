//! Implementation of `pinecone.doctor`: one row per install/health check. Runs as the caller and
//! never raises for something it can report; what the caller cannot see is a `warn`, not a `fail`.
//! It reuses the probes `status()` is built from (queue stats, failed partitions, the launcher lock).

use std::net::{Ipv4Addr, Ipv6Addr};

use pg_pinecone_client::Error as ClientError;
use pgrx::{pg_sys, spi::Spi};

use crate::{
    API_KEY, CONTROL_PLANE_URL, WORKER_ENABLED,
    backend::{self, EffectiveCredentials, Limit429, guc_string},
    catalog, docs_outbox, docs_registry, outbox, registry, secret, status,
};

/// `(check_name, status, detail, hint)`.
pub(crate) type DoctorRow = (String, String, String, Option<String>);

fn row(name: &str, status: &str, detail: String, hint: Option<&str>) -> DoctorRow {
    (
        name.to_string(),
        status.to_string(),
        detail,
        hint.map(str::to_string),
    )
}

fn ok(name: &str, detail: impl Into<String>) -> DoctorRow {
    row(name, "ok", detail.into(), None)
}

fn warn(name: &str, detail: impl Into<String>, hint: &str) -> DoctorRow {
    row(name, "warn", detail.into(), Some(hint))
}

fn fail(name: &str, detail: impl Into<String>, hint: &str) -> DoctorRow {
    row(name, "fail", detail.into(), Some(hint))
}

pub(crate) fn doctor(remote: bool) -> Vec<DoctorRow> {
    let mut rows = vec![version_check(
        env!("CARGO_PKG_VERSION"),
        extension_version("pg_pinecone"),
    )];

    // `pg_settings` shows NULL where `current_setting` would raise for a role without the grant.
    let preload = Spi::get_one::<String>(
        "SELECT setting FROM pg_catalog.pg_settings WHERE name = 'shared_preload_libraries'",
    )
    .ok()
    .flatten();
    let preloaded = preload.as_deref().map(is_preloaded);
    rows.push(preload_check(preloaded));
    rows.push(pgvector_check(extension_version("vector")));

    let source = credential_source();
    rows.push(credential_check(&source));
    rows.push(url_check(&source));
    rows.push(reachability(remote, &source));

    let queues = Queues::read();
    rows.push(outbox_check(&queues));
    rows.push(failed_partitions_check(&queues));
    rows.push(worker_check(preloaded, queues.backlog));
    rows
}

fn extension_version(name: &str) -> Option<String> {
    Spi::get_one_with_args::<String>(
        "SELECT extversion FROM pg_catalog.pg_extension WHERE extname = $1",
        &[name.into()],
    )
    .ok()
    .flatten()
}

/// A mismatch is a `warn`, not an error: after a restart the new library serves the old catalog.
fn version_check(library: &str, catalog: Option<String>) -> DoctorRow {
    match catalog {
        Some(catalog) if catalog == library => {
            ok("version", format!("library and catalog {library}"))
        }
        Some(catalog) => warn(
            "version",
            format!("library {library}, catalog {catalog}"),
            "run ALTER EXTENSION pg_pinecone UPDATE right after the restart (runbook: After ALTER EXTENSION UPDATE)",
        ),
        None => warn(
            "version",
            format!("library {library}; the installed catalog version could not be read"),
            "SELECT extversion FROM pg_extension WHERE extname = 'pg_pinecone'",
        ),
    }
}

fn is_preloaded(setting: &str) -> bool {
    setting.split(',').any(|lib| {
        let lib = lib.trim().trim_matches('"');
        lib.strip_suffix(".so").unwrap_or(lib) == "pg_pinecone"
    })
}

fn preload_check(preloaded: Option<bool>) -> DoctorRow {
    match preloaded {
        Some(true) => ok("shared_preload_libraries", "pg_pinecone is preloaded"),
        Some(false) => warn(
            "shared_preload_libraries",
            "pg_pinecone is not preloaded: no background worker, so async tables never drain",
            "add pg_pinecone to shared_preload_libraries in postgresql.conf and restart",
        ),
        None => warn(
            "shared_preload_libraries",
            "the setting is not visible to this role",
            "run as a superuser or a member of pg_read_all_settings, or check postgresql.conf",
        ),
    }
}

fn pgvector_check(version: Option<String>) -> DoctorRow {
    match version {
        Some(version) => ok("pgvector", format!("vector {version}")),
        None => fail(
            "pgvector",
            "the vector extension is not installed in this database",
            "install pgvector (scripts/install-pgvector.sh) and run CREATE EXTENSION vector",
        ),
    }
}

enum Source {
    Session(EffectiveCredentials),
    Stored(EffectiveCredentials),
    Unreadable,
    Missing,
}

impl Source {
    fn credentials(&self) -> Option<&EffectiveCredentials> {
        match self {
            Source::Session(creds) | Source::Stored(creds) => Some(creds),
            Source::Unreadable | Source::Missing => None,
        }
    }
}

/// `backend::effective_credentials` without the raise. A stored key is bound to its stored URL.
fn credential_source() -> Source {
    let guc_key = guc_string(&API_KEY);
    if !guc_key.is_empty() {
        return Source::Session(EffectiveCredentials {
            api_key: guc_key,
            control_plane_url: guc_string(&CONTROL_PLANE_URL),
        });
    }
    if !secret::has_stored_api_key() {
        return Source::Missing;
    }
    let can_read = Spi::get_one::<bool>(
        "SELECT pg_catalog.has_table_privilege(current_user, 'pinecone.secret'::regclass, 'SELECT')",
    )
    .ok()
    .flatten()
    .unwrap_or(false);
    if !can_read {
        return Source::Unreadable;
    }
    let stored = Spi::get_two::<String, String>(
        "SELECT (SELECT api_key FROM pinecone.secret WHERE id), \
                (SELECT control_plane_url FROM pinecone.secret WHERE id)",
    )
    .ok();
    match stored {
        Some((Some(key), url)) if !key.is_empty() => Source::Stored(EffectiveCredentials {
            api_key: key,
            control_plane_url: url.unwrap_or_default(),
        }),
        _ => Source::Missing,
    }
}

fn credential_check(source: &Source) -> DoctorRow {
    match source {
        Source::Session(_) => ok(
            "credentials",
            "session setting pinecone.api_key (it overrides the stored key; the background worker \
             only uses a key stored by pinecone.set_api_key)",
        ),
        Source::Stored(_) => ok("credentials", "key stored in pinecone.secret"),
        Source::Unreadable => warn(
            "credentials",
            "a key is stored, but this role cannot read pinecone.secret",
            "SET pinecone.api_key for the session, or GRANT SELECT ON pinecone.secret TO this role",
        ),
        Source::Missing => warn(
            "credentials",
            "no API key: remote calls and the background worker cannot reach Pinecone",
            "SET pinecone.api_key = '<key>' for the session, or run pinecone.set_api_key('<key>') as an owner",
        ),
    }
}

fn url_check(source: &Source) -> DoctorRow {
    const NAME: &str = "control_plane_url";
    const SET_GUC: &str = "SET pinecone.control_plane_url to an https:// URL (the default is https://api.pinecone.io)";
    let (url, origin, hint) = match source {
        Source::Session(creds) => (creds.control_plane_url.clone(), "", SET_GUC),
        Source::Stored(creds) => (
            creds.control_plane_url.clone(),
            " (stored with the key)",
            "re-run pinecone.set_api_key('<key>', 'https://api.pinecone.io') with an https:// URL",
        ),
        Source::Unreadable | Source::Missing => (
            guc_string(&CONTROL_PLANE_URL),
            " (session setting; the stored URL is not visible)",
            SET_GUC,
        ),
    };
    let shown = redact(&url);
    match url_kind(&url) {
        UrlKind::Https => ok(NAME, format!("{shown}{origin}")),
        UrlKind::LoopbackHttp => warn(
            NAME,
            format!("{shown}{origin} is plain http on loopback"),
            "fine for a local mock; use https:// anywhere else",
        ),
        UrlKind::Insecure => fail(
            NAME,
            format!("{shown}{origin} is not https: the API key would travel unencrypted"),
            hint,
        ),
    }
}

#[derive(Debug, PartialEq)]
enum UrlKind {
    Https,
    LoopbackHttp,
    Insecure,
}

/// The policy of `pg_pinecone_client::require_secure_url` (PR #37); switch to it once both are merged.
fn url_kind(url: &str) -> UrlKind {
    if url
        .chars()
        .any(|c| c.is_control() || c.is_whitespace() || c == '\\')
    {
        return UrlKind::Insecure;
    }
    let Some((scheme, rest)) = url.split_once("://") else {
        return UrlKind::Insecure;
    };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let hostport = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let host = match hostport.strip_prefix('[') {
        Some(v6) => v6.split_once(']').map_or("", |(h, _)| h),
        None => hostport.split(':').next().unwrap_or(""),
    };
    let loopback = host.eq_ignore_ascii_case("localhost")
        || host.parse::<Ipv4Addr>().is_ok_and(|ip| ip.is_loopback())
        || host.parse::<Ipv6Addr>().is_ok_and(|ip| ip.is_loopback());
    if host.is_empty() {
        UrlKind::Insecure
    } else if scheme.eq_ignore_ascii_case("https") {
        UrlKind::Https
    } else if scheme.eq_ignore_ascii_case("http") && loopback {
        UrlKind::LoopbackHttp
    } else {
        UrlKind::Insecure
    }
}

/// Doctor output gets pasted into tickets: drop any `user:password@` from a URL.
fn redact(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_string();
    };
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    match rest[..end].rsplit_once('@') {
        Some((_, host)) => format!("{scheme}://***@{host}{}", &rest[end..]),
        None => url.to_string(),
    }
}

fn reachability(remote: bool, source: &Source) -> DoctorRow {
    if !remote {
        return ok("reachability", "skipped (remote => false)");
    }
    let Some(creds) = source.credentials() else {
        return warn(
            "reachability",
            "not checked: no API key this session can use",
            "fix the credentials check first",
        );
    };
    if url_kind(&creds.control_plane_url) == UrlKind::Insecure {
        return warn(
            "reachability",
            "not checked: the key is only sent over https",
            "fix the control_plane_url check first",
        );
    }
    let client = backend::client_for(creds.api_key.clone(), creds.control_plane_url.clone());
    classify_probe(client.probe(), &redact(&creds.control_plane_url))
}

/// The detail's prefix (`network:`, `auth:`, `plan or permission:`) is what tells the causes apart.
fn classify_probe(result: Result<(), ClientError>, url: &str) -> DoctorRow {
    const NAME: &str = "reachability";
    match result {
        Ok(()) => ok(NAME, format!("Pinecone at {url} accepted the key")),
        Err(ClientError::Auth) => fail(
            NAME,
            "auth: Pinecone refused the key (HTTP 401)",
            "check the key belongs to the right project, then pinecone.set_api_key or SET pinecone.api_key again",
        ),
        Err(ClientError::Api {
            status: 403,
            message,
        }) => fail(
            NAME,
            format!(
                "plan or permission: Pinecone refused the request (HTTP 403): {}",
                clip(&message)
            ),
            "the key is valid but not allowed this call: check its role and the project's plan",
        ),
        Err(ClientError::Api {
            status: 429,
            message,
        }) => row(
            NAME,
            "warn",
            format!("limit: Pinecone answered HTTP 429: {}", clip(&message)),
            backend::classify_429(429, &message).map(Limit429::hint),
        ),
        Err(ClientError::Api { status, message }) => fail(
            NAME,
            format!(
                "unexpected: Pinecone answered HTTP {status}: {}",
                clip(&message)
            ),
            "check pinecone.control_plane_url points at a Pinecone control plane",
        ),
        Err(ClientError::Timeout { ms }) => fail(
            NAME,
            format!("network: no answer from {url} within {ms} ms"),
            "raise pinecone.timeout_ms, or check connectivity and proxy settings from this host",
        ),
        Err(ClientError::Transport { url, reason }) => fail(
            NAME,
            format!("network: could not reach {}: {reason}", redact(&url)),
            "check pinecone.control_plane_url, DNS, firewall and proxy settings; remote => false skips this check",
        ),
        Err(other) => fail(NAME, other.to_string(), "see the runbook"),
    }
}

fn clip(message: &str) -> String {
    const MAX: usize = 200;
    match message.char_indices().nth(MAX) {
        Some((at, _)) => format!("{}...", &message[..at]),
        None => message.to_string(),
    }
}

#[derive(Default)]
struct Queues {
    /// Stats functions this role cannot run for a lane it sees: that lane's `backlog` and `backoff`
    /// are unknown, `parked` is not.
    blind: Vec<&'static str>,
    backlog: i64,
    backoff: Vec<String>,
    parked: Vec<String>,
    failed_partitions: i64,
}

impl Queues {
    fn read() -> Self {
        let mut queues = Queues::default();
        let tables = registry::list_for_reader();
        let stats_allowed = tables.is_empty() || outbox::can_call_outbox_stats();
        let docs = docs_registry::list_visible();
        let doc_stats_allowed = docs.is_empty() || docs_outbox::can_call_doc_outbox_stats();
        if !stats_allowed {
            queues.blind.push("pinecone._outbox_stats(oid)");
        }
        if !doc_stats_allowed {
            queues.blind.push("pinecone._doc_outbox_stats(oid)");
        }

        for row in tables {
            let name = table_label(row.relid);
            if row.queue_parked {
                queues.parked.push(name.clone());
            }
            if stats_allowed {
                let (backlog, _, _, in_backoff) = status::outbox_stats(row.relid, true);
                queues.backlog += backlog.unwrap_or(0);
                if in_backoff && !row.queue_parked {
                    queues.backoff.push(name);
                }
            }
            queues.failed_partitions += i64::from(status::failed_partitions(
                row.relid,
                row.syncs(),
                row.partition.as_ref(),
            ));
        }
        for doc in &docs {
            let name = format!("{} (documents)", table_label(doc.relid));
            if doc.queue_parked {
                queues.parked.push(name.clone());
            }
            if let Some(stats) = docs_outbox::stats_if(doc_stats_allowed, doc.relid) {
                queues.backlog += stats.backlog;
                if stats.in_backoff && !doc.queue_parked {
                    queues.backoff.push(name);
                }
            }
        }
        queues
    }
}

fn table_label(relid: pg_sys::Oid) -> String {
    match catalog::resolve_table_name(relid) {
        Some((schema, table)) => format!("{schema}.{table}"),
        None => format!("<dropped:{relid}>"),
    }
}

fn outbox_check(queues: &Queues) -> DoctorRow {
    if !queues.parked.is_empty() {
        return fail(
            "outbox",
            format!("queue parked: {}", queues.parked.join(", ")),
            "pinecone.status() shows last_error; fix the cause, then pinecone.outbox_retry(tbl), or \
             pinecone.unpark_docs(tbl) for a (documents) queue (runbook: Parked queue)",
        );
    }
    if !queues.backoff.is_empty() {
        return warn(
            "outbox",
            format!(
                "{} queued, retrying with backoff: {}",
                queues.backlog,
                queues.backoff.join(", ")
            ),
            "Pinecone is failing these writes; pinecone.status() shows last_error (runbook: Backlog monitoring)",
        );
    }
    if !queues.blind.is_empty() {
        let functions = queues.blind.join(", ");
        return warn(
            "outbox",
            format!("queue depth not visible: this role cannot run {functions}"),
            &format!("GRANT EXECUTE ON FUNCTION {functions} TO this role"),
        );
    }
    ok("outbox", format!("{} queued", queues.backlog))
}

fn failed_partitions_check(queues: &Queues) -> DoctorRow {
    match queues.failed_partitions {
        0 => ok("failed_partitions", "none"),
        n => fail(
            "failed_partitions",
            format!("{n} partition(s) Pinecone refused for good"),
            "pinecone.failed_partitions(tbl) lists them; fix the cause, then pinecone.retry_failed_partitions(tbl)",
        ),
    }
}

/// A stopped launcher is only an outage while something is queued, so it is a `warn` until then.
fn worker_check(preloaded: Option<bool>, backlog: i64) -> DoctorRow {
    if outbox::launcher_lock_held() {
        return ok("worker", "launcher running");
    }
    let (detail, hint) = if !WORKER_ENABLED.get() {
        (
            "launcher not running: pinecone.worker_enabled is off",
            "ALTER SYSTEM SET pinecone.worker_enabled = on, then SELECT pg_reload_conf()",
        )
    } else if preloaded == Some(false) {
        (
            "launcher not running: pg_pinecone is not preloaded",
            "add pg_pinecone to shared_preload_libraries and restart",
        )
    } else {
        (
            "launcher not running",
            "check the server log for the pg_pinecone launcher (runbook: NO WORKER)",
        )
    };
    if backlog > 0 {
        fail("worker", format!("{detail}; {backlog} queued"), hint)
    } else {
        warn("worker", detail, hint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status_of(row: &DoctorRow) -> &str {
        &row.1
    }

    #[test]
    fn preload_list_is_read_like_postgres_writes_it() {
        assert!(is_preloaded("pg_pinecone"));
        assert!(is_preloaded("pg_stat_statements, pg_pinecone"));
        assert!(is_preloaded("\"pg_pinecone\",auto_explain"));
        assert!(is_preloaded("pg_pinecone.so"));
        assert!(!is_preloaded(""));
        assert!(!is_preloaded("pg_pinecone_other,pinecone"));
    }

    #[test]
    fn a_version_mismatch_or_unreadable_catalog_warns() {
        let same = version_check("0.13.0", Some("0.13.0".into()));
        assert_eq!(status_of(&same), "ok");
        let old = version_check("0.13.0", Some("0.12.1".into()));
        assert_eq!(status_of(&old), "warn");
        assert!(old.2.contains("0.12.1"), "{}", old.2);
        assert_eq!(status_of(&version_check("0.13.0", None)), "warn");
    }

    #[test]
    fn the_probe_separates_network_auth_and_plan() {
        let url = "https://api.pinecone.io";
        let api = |status, message: &str| ClientError::Api {
            status,
            message: message.to_string(),
        };
        let probe = |result| classify_probe(result, url);

        assert_eq!(status_of(&probe(Ok(()))), "ok");
        let auth = probe(Err(ClientError::Auth));
        assert!(auth.1 == "fail" && auth.2.starts_with("auth:"), "{auth:?}");
        let plan = probe(Err(api(403, "upgrade your plan")));
        assert!(
            plan.1 == "fail" && plan.2.starts_with("plan or permission:"),
            "{plan:?}"
        );
        let network = probe(Err(ClientError::Transport {
            url: url.to_string(),
            reason: "refused".to_string(),
        }));
        assert!(
            network.1 == "fail" && network.2.starts_with("network:"),
            "{network:?}"
        );
        let timeout = probe(Err(ClientError::Timeout { ms: 5 }));
        assert!(timeout.2.starts_with("network:"), "{timeout:?}");
        assert_eq!(
            status_of(&probe(Err(api(429, "Pace your requests")))),
            "warn"
        );
        assert_eq!(status_of(&probe(Err(api(500, "boom")))), "fail");
    }

    #[test]
    fn a_long_error_body_is_clipped() {
        assert_eq!(clip("short"), "short");
        assert_eq!(clip(&"x".repeat(500)).len(), 203);
    }

    #[test]
    fn parked_beats_backoff_beats_blind() {
        let mut queues = Queues {
            blind: vec!["pinecone._doc_outbox_stats(oid)"],
            parked: vec!["public.t (documents)".into()],
            backoff: vec!["public.u".into()],
            ..Queues::default()
        };
        let parked = outbox_check(&queues);
        assert_eq!(status_of(&parked), "fail");
        assert!(parked.3.unwrap().contains("unpark_docs"));
        queues.parked.clear();
        assert!(outbox_check(&queues).2.contains("backoff"));
        queues.backoff.clear();
        let blind = outbox_check(&queues);
        assert_eq!(status_of(&blind), "warn");
        assert!(blind.2.contains("_doc_outbox_stats"), "{blind:?}");
        queues.blind.clear();
        assert_eq!(status_of(&outbox_check(&queues)), "ok");
    }

    #[test]
    fn only_https_or_loopback_http_carries_the_key() {
        for url in [
            "https://api.pinecone.io",
            "HTTPS://api.pinecone.io/",
            "https://user:pw@api.pinecone.io",
        ] {
            assert_eq!(url_kind(url), UrlKind::Https, "{url}");
        }
        for url in [
            "http://localhost:5080",
            "http://127.0.0.1:5555",
            "http://[::1]:9000/path",
        ] {
            assert_eq!(url_kind(url), UrlKind::LoopbackHttp, "{url}");
        }
        for url in [
            "http://api.pinecone.io",
            "http://localhost.evil.com",
            "http://127.0.0.1@evil.com",
            "http://evil.com\\@127.0.0.1",
            "ftp://localhost",
            "api.pinecone.io",
            "",
        ] {
            assert_eq!(url_kind(url), UrlKind::Insecure, "{url}");
        }
    }

    #[test]
    fn a_cleartext_url_fails_with_the_fix_for_its_source() {
        let creds = |url: &str| EffectiveCredentials {
            api_key: "k".into(),
            control_plane_url: url.into(),
        };
        let stored = url_check(&Source::Stored(creds("http://u:secret@pinecone.example")));
        assert_eq!(status_of(&stored), "fail");
        assert!(stored.3.as_deref().unwrap().contains("set_api_key"));
        assert!(!stored.2.contains("secret"), "{stored:?}");
        let session = url_check(&Source::Session(creds("http://pinecone.example")));
        assert_eq!(status_of(&session), "fail");
        assert!(
            session
                .3
                .as_deref()
                .unwrap()
                .contains("SET pinecone.control_plane_url")
        );
        let mock = url_check(&Source::Session(creds("http://127.0.0.1:5080")));
        assert_eq!(status_of(&mock), "warn");
        let https = url_check(&Source::Stored(creds("https://api.pinecone.io")));
        assert_eq!(status_of(&https), "ok");
    }

    #[test]
    fn userinfo_is_redacted() {
        assert_eq!(
            redact("https://u:pw@host:1/p?q#f"),
            "https://***@host:1/p?q#f"
        );
        assert_eq!(redact("https://host/a@b"), "https://host/a@b");
        assert_eq!(redact("no scheme"), "no scheme");
    }
}
