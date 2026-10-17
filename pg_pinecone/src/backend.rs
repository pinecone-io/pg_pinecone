//! Backend-local Pinecone client cache, error mapping, and shared argument
//! validation for the extension's `pinecone.*` SQL functions.

use std::{
    collections::HashMap,
    ffi::CString,
    sync::{Arc, Mutex, MutexGuard, OnceLock},
    time::Duration,
};

use pg_pinecone_client::{Client, Config, Error as ClientError};
use pgrx::{
    guc::{GucCheckError, GucSetting},
    prelude::*,
};

use crate::{
    API_KEY, CONTROL_PLANE_URL, MAX_CANDIDATES, SCAN_FACTOR, SOURCE_TAG, TIMEOUT_MS, partition_lane,
};

/// `(control_plane_url, api_key, timeout, user_agent)` — the full `Config`,
/// so any GUC change (including `pinecone.timeout_ms` and
/// `pinecone.source_tag`) takes effect on the next call.
type ClientKey = (String, String, Duration, String);

/// Backend-local cache of `Client`s keyed by `ClientKey`, so the client's
/// per-index host cache survives across calls. A GUC change lands on a
/// different key and so gets a fresh `Client`; this is per-backend, single-threaded,
/// so a plain `Mutex` is enough.
static CLIENTS: OnceLock<Mutex<HashMap<ClientKey, Arc<Client>>>> = OnceLock::new();

/// Bounds the cache across repeated GUC changes; the whole map is dropped when full.
const MAX_CACHED_CLIENTS: usize = 8;

/// Returns the `Client` for the credentials resolved by [`effective_credentials`]
/// (session GUC key+URL if a key is set, else the stored key bound to its
/// *stored* URL). Used by user-invoked remote operations
/// (`pinecone.promote`/`pinecone.demote`/`pinecone.sync`) and by the
/// sync-trigger helpers.
pub(crate) fn client_for_effective_key() -> Arc<Client> {
    let creds = effective_credentials();
    client_for(creds.api_key, creds.control_plane_url)
}

/// Shared cache lookup behind [`client_for_effective_key`]: builds (or reuses)
/// the `Client` for `api_key`/`control_plane_url` plus the current `timeout` GUC.
fn client_for(api_key: String, control_plane_url: String) -> Arc<Client> {
    let timeout = Duration::from_millis(TIMEOUT_MS.get() as u64);
    let user_agent = user_agent_header();

    let key = (
        control_plane_url.clone(),
        api_key.clone(),
        timeout,
        user_agent.clone(),
    );
    let clients = CLIENTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(client) = lock_clients(clients).get(&key) {
        return Arc::clone(client);
    }

    // Constructed outside the lock so a panic in Client::with_sleep can't
    // poison it. `interruptible_sleep` (rather than the default
    // `std::thread::sleep`) is what keeps `upsert_vectors`' retry backoff and
    // `wait_index_ready`'s polling responsive to `pg_cancel_backend`/
    // `statement_timeout` instead of blocking uninterruptibly.
    let client = Arc::new(Client::with_sleep(
        Config {
            control_plane_url,
            api_key,
            timeout,
            user_agent: Some(user_agent),
        },
        interruptible_sleep,
    ));
    let mut clients = lock_clients(clients);
    if clients.len() >= MAX_CACHED_CLIENTS {
        clients.clear();
    }
    clients.insert(key, Arc::clone(&client));
    client
}

/// The exact `User-Agent` every request carries — also what
/// `pinecone.user_agent()` returns. `PG_MAJORVERSION_NUM` is fixed at compile
/// time by the pgrx `pgNN` feature, not the connected server.
pub(crate) fn user_agent_header() -> String {
    format!(
        "pg_pinecone-client-{}; lang=rust; pg={}; client={}; source_tag={}",
        env!("CARGO_PKG_VERSION"),
        pg_sys::PG_MAJORVERSION_NUM,
        pg_pinecone_client::VERSION,
        source_tag(),
    )
}

/// `pg_pinecone`, or `pg_pinecone:<normalised value>` once `pinecone.source_tag` is
/// set — lets a partner packaging (e.g. a managed-Postgres vendor) attribute
/// its share of pg_pinecone's Pinecone traffic alongside pg_pinecone's own.
fn source_tag() -> String {
    match normalise_source_tag(&guc_string(&SOURCE_TAG)) {
        tag if tag.is_empty() => "pg_pinecone".to_string(),
        tag => format!("pg_pinecone:{tag}"),
    }
}

/// Mirrors the Pinecone Python/TypeScript SDKs' own `source_tag` normalisation.
fn normalise_source_tag(value: &str) -> String {
    let cleaned: String = value
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | ' ' | ':'))
        .collect();

    let mut result = String::with_capacity(cleaned.len());
    let mut in_space_run = false;
    for c in cleaned.trim().chars() {
        if c == ' ' {
            in_space_run = true;
            continue;
        }
        if in_space_run {
            result.push('_');
            in_space_run = false;
        }
        result.push(c);
    }
    result
}

/// `pinecone.source_tag`'s GUC check hook: the empty default always passes;
/// any other value must [`normalise_source_tag`] to something non-empty, or
/// this rejects it with a clear error.
///
/// # Safety
/// Called only by Postgres, as the `check_hook` argument to
/// `GucRegistry::define_string_guc_with_hooks`; `newval` is a valid pointer
/// to a NUL-terminated C string for the duration of this call.
#[pg_guard]
pub(crate) unsafe extern "C-unwind" fn check_source_tag(
    newval: *mut *mut std::os::raw::c_char,
    _extra: *mut *mut std::os::raw::c_void,
    _source: pg_sys::GucSource::Type,
) -> bool {
    let value = unsafe { std::ffi::CStr::from_ptr(*newval) }.to_string_lossy();
    if value.is_empty() || !normalise_source_tag(&value).is_empty() {
        return true;
    }
    unsafe {
        GucCheckError::new(format!(
            "pinecone.source_tag must contain at least one lowercase letter, digit, `_`, ` `, \
             or `:` after normalisation; got '{value}'"
        ))
        .apply();
    }
    false
}

/// The credentials a user-invoked remote operation should use, resolved as one
/// unit so a stored key can never be combined with a URL the caller merely
/// happens to have set — see [`effective_credentials`].
pub(crate) struct EffectiveCredentials {
    pub(crate) api_key: String,
    pub(crate) control_plane_url: String,
}

/// Resolves the API key *and* control-plane URL for a user-invoked remote
/// operation, as one unit: the session GUC `pinecone.api_key`, if non-empty,
/// pairs with the session GUC `pinecone.control_plane_url` (the caller
/// brought their own key, so there's nothing of ours to protect); otherwise
/// the key stored in `pinecone.secret` pairs with **its own stored**
/// `control_plane_url` — the session's `pinecone.control_plane_url` GUC is
/// ignored *entirely* in that case.
///
/// This binding is the fix for a stored-key exfiltration path: without it, a
/// writer with no key of their own could `SET pinecone.control_plane_url` to
/// a server they control and have the sync trigger's `SECURITY DEFINER` path
/// resolve the *stored* key but send it to *their* URL. Binding the stored
/// key to the stored URL (see `secret`'s module rustdoc) closes that off for
/// every caller of [`client_for_effective_key`] — user-invoked remote
/// operations and the sync-trigger helpers alike.
///
/// The stored key is read as the calling role (plain `SELECT`, not `SECURITY
/// DEFINER`) — so resolution fails, naming both remedies, unless the caller
/// either sets the GUC or has been granted `SELECT` on `pinecone.secret` by
/// whoever ran `pinecone.set_api_key` (see `secret`'s module rustdoc). The one
/// `SECURITY DEFINER` accessor, `pinecone._sync_trigger`, never goes through
/// this privilege check itself — it runs as its own owner, who always has
/// access.
///
/// Privilege is probed via `has_table_privilege` rather than by attempting the
/// `SELECT` and catching a permission-denied error: the latter is a genuine
/// Postgres `ERROR` (not a plain `Result`), and recovering from one safely
/// mid-transaction needs a subtransaction this call has no reason to pay for.
pub(crate) fn effective_credentials() -> EffectiveCredentials {
    let guc_key = guc_string(&API_KEY);
    if !guc_key.is_empty() {
        return EffectiveCredentials {
            api_key: guc_key,
            control_plane_url: guc_string(&CONTROL_PLANE_URL),
        };
    }

    let has_select = Spi::get_one::<bool>(
        "SELECT pg_catalog.has_table_privilege(current_user, 'pinecone.secret'::regclass, 'SELECT')",
    )
    .expect("pg_pinecone: has_table_privilege(pinecone.secret) query failed")
    .expect("pg_pinecone: has_table_privilege unexpectedly NULL");
    if !has_select {
        no_effective_api_key_error();
    }

    // Scalar subqueries (rather than a bare `SELECT ... FROM pinecone.secret
    // WHERE id`) always yield exactly one row, with NULL columns when
    // `pinecone.secret` is empty — unlike the bare form, which panics with
    // `InvalidPosition` on zero rows (see `registry::index_namespace_owner`'s
    // rustdoc for the same pattern applied there).
    let (key, control_plane_url) = Spi::get_two::<String, String>(
        "SELECT (SELECT api_key FROM pinecone.secret WHERE id), \
                (SELECT control_plane_url FROM pinecone.secret WHERE id)",
    )
    .expect("pg_pinecone: pinecone.secret query failed");
    let key = key.unwrap_or_default();
    if key.is_empty() {
        no_effective_api_key_error();
    }
    EffectiveCredentials {
        api_key: key,
        control_plane_url: control_plane_url.unwrap_or_default(),
    }
}

/// The two-remedy error shared by every [`effective_credentials`] failure —
/// whether because nothing is stored or because the caller can't read it.
fn no_effective_api_key_error() -> ! {
    error!(
        "pg_pinecone: no API key available — either SET pinecone.api_key = '<your key>' for this \
         session, or have an owner/superuser run pinecone.set_api_key('<your key>') and \
         GRANT SELECT ON pinecone.secret TO <your role>"
    )
}

/// Which credential-resolution strategy [`client_for_drain`] should use —
/// threaded through `outbox::drain_step`/`Scheduler` so the same dispatch
/// code path serves both the inline `sync_mode = 'sync'` drain (still the
/// writer's own session) and the background worker (`worker.rs`), which must
/// never share that session's resolution rules (see [`stored_credentials`]).
#[derive(Clone, Copy)]
pub(crate) enum DrainCredentials {
    /// The inline drain `pinecone._sync_trigger` calls for `sync_mode = 'sync'`
    /// tables, in the writer's own session — [`effective_credentials`].
    Inline,
    /// The background worker's drain, running outside any user session —
    /// [`stored_credentials`] only. Constructed by `outbox::Scheduler`, which
    /// `worker.rs`'s per-database worker loop drives.
    Worker,
}

/// Resolves the `Client` `outbox::drain_step` should dispatch through, per
/// [`DrainCredentials`].
pub(crate) fn client_for_drain(mode: DrainCredentials) -> Arc<Client> {
    let creds = match mode {
        DrainCredentials::Inline => effective_credentials(),
        DrainCredentials::Worker => stored_credentials(),
    };
    client_for(creds.api_key, creds.control_plane_url)
}

/// The API key/control-plane-URL pair *stored* in `pinecone.secret` — unlike
/// [`effective_credentials`], never falls back to (or is overridable by) the
/// session GUCs. This is the only credential source the background worker
/// may use: it runs outside any user session, so
/// there is no session GUC to prefer, and honoring one anyway would let a DB
/// owner smuggle an arbitrary `pinecone.control_plane_url`/session key into
/// the worker's trusted context and exfiltrate the stored key to it.
///
/// `outbox::drain_step`'s `Worker` path checks `secret::has_stored_api_key()`
/// itself *before* ever reaching this function, so a
/// database with backlog but no key configured yet backs off cleanly instead
/// of landing here. This function's own `error!` below is only a defensive
/// fallback for the narrow TOCTOU window between that check and this query
/// (e.g. another session clearing the key in between) — it should be
/// unreachable in practice.
pub(crate) fn stored_credentials() -> EffectiveCredentials {
    let (key, control_plane_url) = Spi::get_two::<String, String>(
        "SELECT (SELECT api_key FROM pinecone.secret WHERE id), \
                (SELECT control_plane_url FROM pinecone.secret WHERE id)",
    )
    .expect("pg_pinecone: pinecone.secret query failed (worker credentials)");
    let key = key.unwrap_or_default();
    if key.is_empty() {
        error!(
            "pg_pinecone: no API key stored — the pg_pinecone worker needs pinecone.set_api_key(...) \
             before it can drain any async table"
        );
    }
    EffectiveCredentials {
        api_key: key,
        control_plane_url: control_plane_url.unwrap_or_default(),
    }
}

/// Classifies a [`ClientError`] as transient (worth an exponential-backoff
/// retry) or permanent (counts toward `pinecone.outbox_max_attempts`, and
/// eventually parks the relid). Network/timeout
/// failures and 5xx/429 responses are transient; everything else (auth,
/// not-found, other 4xx validation) is permanent — retrying those forever
/// would just replay the same rejection.
pub(crate) fn is_transient(err: &ClientError) -> bool {
    match err {
        ClientError::Timeout { .. } | ClientError::Transport { .. } => true,
        ClientError::Api { status, .. } => *status >= 500 || *status == 429,
        ClientError::MissingApiKey
        | ClientError::IndexNotFound { .. }
        | ClientError::BackupNotFound { .. }
        | ClientError::NamespaceNotFound { .. }
        | ClientError::Auth
        | ClientError::BadRequest { .. }
        | ClientError::Conflict { .. } => false,
    }
}

/// Whether `err` refuses one namespace for good: a permanent error that says nothing about the
/// credentials, the index or the whole table. A lane gives up on that partition alone; every
/// other error stays the table's. So does a status that every namespace would get alike: a 404
/// (the host no longer holds the index) and a 200 whose body cannot be read.
pub(crate) fn fails_one_partition(err: &ClientError) -> bool {
    match err {
        ClientError::Api { status, .. } => {
            (400..500).contains(status) && !matches!(status, 404 | 429)
        }
        other => {
            !is_transient(other)
                && !matches!(
                    other,
                    ClientError::MissingApiKey
                        | ClientError::Auth
                        | ClientError::IndexNotFound { .. }
                        | ClientError::NamespaceNotFound { .. }
                )
        }
    }
}

/// Which limit a Pinecone 429 reports: the monthly unit cap clears with the billing period or a
/// plan upgrade, a per-second rate limit within moments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Limit429 {
    Quota,
    Rate,
}

impl Limit429 {
    /// SQLSTATE of a call that hit the limit: 53400 for the quota, 58030 for the rate.
    pub(crate) fn sqlstate(self) -> PgSqlErrorCode {
        match self {
            Limit429::Quota => PgSqlErrorCode::ERRCODE_CONFIGURATION_LIMIT_EXCEEDED,
            Limit429::Rate => PgSqlErrorCode::ERRCODE_IO_ERROR,
        }
    }

    pub(crate) fn hint(self) -> &'static str {
        match self {
            Limit429::Quota => {
                "monthly unit limit reached — clears when the billing period resets or the plan \
                 is upgraded; queued writes keep retrying"
            }
            Limit429::Rate => {
                "rate limited — transient, retry in a moment; queued writes retry automatically"
            }
        }
    }
}

/// Tells the two 429s apart by the message Pinecone documents for each: the quota names a "unit
/// limit" for the current month and says to upgrade the plan; the rate limit says "Pace your ...
/// requests". `None` for any other status; an unrecognised 429 counts as a rate limit, the same
/// transient treatment [`is_transient`] gives every 429.
pub(crate) fn classify_429(status: u16, message: &str) -> Option<Limit429> {
    if status != 429 {
        return None;
    }
    let message = message.to_ascii_lowercase();
    let quota = message.contains("unit limit") || message.contains("upgrade your plan");
    Some(if quota {
        Limit429::Quota
    } else {
        Limit429::Rate
    })
}

/// `err` as a queue's `last_error` records it: the client's text, for a 429 which limit it was,
/// and for a refused first write of a namespace what to do about the plan.
pub(crate) fn failure_text(err: &ClientError) -> String {
    let text = match err {
        ClientError::Api { status, message } => match classify_429(*status, message) {
            Some(limit) => format!("{err} — {}", limit.hint()),
            None => err.to_string(),
        },
        other => other.to_string(),
    };
    if text.contains(partition_lane::NEW_NAMESPACE_MARK) {
        format!("{text} — {}", partition_lane::NAMESPACE_LIMIT_HINT)
    } else {
        text
    }
}

/// Chunk size used by [`interruptible_sleep`], so a long wait (`upsert_vectors`
/// retry backoff, `wait_index_ready` polling) never blocks uninterruptibly for
/// its full duration.
const SLEEP_CHUNK: Duration = Duration::from_millis(100);

/// Sleeps for `duration`, in chunks of at most [`SLEEP_CHUNK`], checking for
/// interrupts between each chunk — passed to `Client::with_sleep` so both of
/// the client's sleep call sites stay responsive to `pg_cancel_backend`/
/// `statement_timeout` instead of blocking the whole duration at once.
pub(crate) fn interruptible_sleep(duration: Duration) {
    let mut remaining = duration;
    while remaining > Duration::ZERO {
        let chunk = remaining.min(SLEEP_CHUNK);
        std::thread::sleep(chunk);
        remaining -= chunk;
        pgrx::check_for_interrupts!();
    }
}

/// Pinecone's documented maximum vector-id length, in bytes — enforced both
/// by `pinecone.promote`'s backfill and by the sync-trigger push/delete path,
/// through this shared helper, so the error is identical either way.
pub(crate) const MAX_ID_LEN: usize = 512;

/// Raises a clear Postgres error if `id` exceeds [`MAX_ID_LEN`].
pub(crate) fn require_id_len(id: &str) {
    if id.len() > MAX_ID_LEN {
        error!(
            "pg_pinecone: id '{id}' is {} bytes, exceeding Pinecone's {MAX_ID_LEN}-byte id limit",
            id.len()
        );
    }
}

/// Same message as [`require_id_len`], but as a returned [`ClientError`]
/// rather than a raised Postgres error — for callers (`promote::
/// try_backfill_chunk`, dispatched from the background worker) that need to
/// record/park an oversized id as a permanent-class dispatch failure instead
/// of crashing the process it runs in.
pub(crate) fn id_len_error(id: &str) -> ClientError {
    ClientError::BadRequest {
        message: format!(
            "id '{id}' is {} bytes, exceeding Pinecone's {MAX_ID_LEN}-byte id limit",
            id.len()
        ),
    }
}

fn lock_clients(
    clients: &Mutex<HashMap<ClientKey, Arc<Client>>>,
) -> MutexGuard<'_, HashMap<ClientKey, Arc<Client>>> {
    // Unreachable in practice: only infallible HashMap ops run under the lock.
    clients
        .lock()
        .expect("pg_pinecone client cache lock poisoned")
}

/// Converts a `real[]` argument into a `Vec<f32>`, raising a clear Postgres
/// error (rather than a raw null panic) for a NULL array or NULL elements within
/// it. `func`/`arg` name the SQL function and argument in the message.
pub(crate) fn require_real_vector(
    func: &str,
    arg: &str,
    values: Option<Vec<Option<f32>>>,
) -> Vec<f32> {
    let elements = values.unwrap_or_else(|| {
        error!("{func}: {arg} must not be NULL — pass a real[] vector, e.g. ARRAY[0.1, 0.2, 0.3]")
    });
    elements
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            v.unwrap_or_else(|| {
                error!(
                    "{func}: {arg}[{}] is NULL — all vector elements must be non-NULL",
                    i + 1
                )
            })
        })
        .collect()
}

/// Converts a required `text` argument into an owned `String`, raising a clear
/// Postgres error (rather than a raw null panic) for a NULL value. `func`/`arg`
/// name the SQL function and argument in the message.
pub(crate) fn require_text(func: &str, arg: &str, value: Option<&str>) -> String {
    value
        .unwrap_or_else(|| {
            error!("{func}: {arg} must not be NULL — pass a text value, e.g. 'example text'")
        })
        .to_string()
}

/// Converts a `text[]` argument into a `Vec<String>`, raising a clear Postgres
/// error for a NULL array or NULL elements within it (1-based position, mirrors
/// [`require_real_vector`]).
pub(crate) fn require_text_array(
    func: &str,
    arg: &str,
    values: Option<Vec<Option<String>>>,
) -> Vec<String> {
    let elements = values.unwrap_or_else(|| {
        error!("{func}: {arg} must not be NULL — pass a text[] array, e.g. ARRAY['a', 'b']")
    });
    elements
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            v.unwrap_or_else(|| {
                error!(
                    "{func}: {arg}[{}] is NULL — all {arg} elements must be non-NULL",
                    i + 1
                )
            })
        })
        .collect()
}

/// Converts an `int` argument declared non-negative in SQL (e.g. `top_k`/`k`)
/// into a `u32`, raising a clear Postgres error (rather than a silent wrap) for
/// a negative value. `func`/`arg` name the SQL function and argument in the message.
pub(crate) fn require_non_negative_u32(func: &str, arg: &str, value: i32) -> u32 {
    u32::try_from(value)
        .unwrap_or_else(|_| error!("{func}: {arg} must be non-negative, got {value}"))
}

/// Reads and validates `pinecone.scan_factor`/`pinecone.max_candidates` for a
/// DRN query body (`am::remote_query`, `topk::topk_remote`) — `0.0`/`0` means
/// "omit", already enforced by each GUC's own registered range; the
/// 0.5-4.0 gap for `scan_factor` isn't expressible there (see [`crate::
/// SCAN_FACTOR`]'s rustdoc), so it's checked here instead, at the point of use.
pub(crate) fn drn_query_tuning() -> (Option<f64>, Option<u32>) {
    let scan_factor = SCAN_FACTOR.get();
    let scan_factor = match scan_factor {
        0.0 => None,
        f if (0.5..=4.0).contains(&f) => Some(f),
        f => error!(
            "pg_pinecone: pinecone.scan_factor must be 0 (omit) or between 0.5 and 4.0, got {f}"
        ),
    };
    let max_candidates = match MAX_CANDIDATES.get() {
        0 => None,
        n => Some(n as u32),
    };
    (scan_factor, max_candidates)
}

/// Raises a Postgres ERROR with an explicit SQLSTATE and optional DETAIL/HINT:
/// `error!()` is always XX000, and `ereport!()` takes a DETAIL but no HINT.
pub(crate) fn error_with_code(
    code: PgSqlErrorCode,
    message: impl Into<String>,
    detail: Option<&str>,
    hint: Option<&str>,
) -> ! {
    let mut report =
        pgrx::pg_sys::panic::ErrorReport::new(code, message.into(), pgrx::function_name!());
    if let Some(detail) = detail {
        report = report.set_detail(detail);
    }
    if let Some(hint) = hint {
        report = report.set_hint(hint);
    }
    report.report(pgrx::PgLogLevel::ERROR);
    unreachable!()
}

/// Raises a Postgres ERROR with a HINT (SQLSTATE 21000).
pub(crate) fn error_with_hint(message: String, hint: String) -> ! {
    error_with_code(
        PgSqlErrorCode::ERRCODE_CARDINALITY_VIOLATION,
        message,
        None,
        Some(&hint),
    )
}

/// Reads a `GucSetting<Option<CString>>`-typed GUC (e.g. `pinecone.api_key`)
/// as an owned `String`, treating an unset GUC as empty rather than `None` —
/// every caller here (and `am::ambuild`'s `pinecone.am_cloud`/`am_region`
/// reads) already has its own "empty means use the default/error" handling.
pub(crate) fn guc_string(setting: &GucSetting<Option<CString>>) -> String {
    setting
        .get()
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Maps every client error to a Postgres `ERROR` stating the cause and the fix.
/// Never echoes the API key, since none of these variants carry it.
pub(crate) fn raise(err: ClientError) -> ! {
    match err {
        ClientError::MissingApiKey => error!(
            "pg_pinecone: pinecone.api_key is not set — run: SET pinecone.api_key = '<your key>' \
             (this session), or ALTER ROLE <role> SET pinecone.api_key = '<your key>' \
             (persistent, but readable by all users via pg_db_role_setting)"
        ),
        ClientError::IndexNotFound { index } => error!(
            "pg_pinecone: index '{index}' not found — check the index name and that pinecone.api_key \
             belongs to the right project"
        ),
        ClientError::BackupNotFound { backup_id } => error!(
            "pg_pinecone: backup '{backup_id}' not found — check the backup id and that pinecone.api_key \
             belongs to the right project"
        ),
        ClientError::NamespaceNotFound { namespace } => error!(
            "pg_pinecone: namespace '{namespace}' not found — nothing has been written to it, or it \
             was already deleted"
        ),
        ClientError::Auth => error!(
            "pg_pinecone: authentication failed — check that pinecone.api_key is valid and has access \
             to this project"
        ),
        ClientError::BadRequest { message } => raise_text(
            format!("pg_pinecone: Pinecone rejected the request: {message}"),
            &message,
        ),
        ClientError::Conflict { name } => {
            error!("pg_pinecone: index '{name}' already exists")
        }
        ClientError::Timeout { ms } => error!(
            "pg_pinecone: request timed out after {ms}ms — raise pinecone.timeout_ms if the index is \
             slow to respond"
        ),
        ClientError::Transport { url, reason } => error!(
            "pg_pinecone: could not reach {url}: {reason} — check pinecone.control_plane_url and \
             network connectivity"
        ),
        ClientError::Api { status, message } => raise_text(
            format!("pg_pinecone: Pinecone returned unexpected status {status}: {message}"),
            &message,
        ),
    }
}

/// Raises `text`; one that [`partition_lane::explain_first_write`] marked also gets its hint.
fn raise_text(text: String, source: &str) -> ! {
    if source.contains(partition_lane::NEW_NAMESPACE_MARK) {
        error_with_code(
            PgSqlErrorCode::ERRCODE_INTERNAL_ERROR,
            text,
            None,
            Some(partition_lane::NAMESPACE_LIMIT_HINT),
        )
    }
    error!("{text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    const READ_UNITS: &str = "Request failed. You've reached your read unit limit for the \
        current month limit. To continue reading data, upgrade your plan.";
    const WRITE_UNITS: &str = "Request failed. You've reached your write unit limit for the \
        current month. To continue writing data, upgrade your plan.";
    const WRITE_UNITS_NUMBERED: &str = "429: You've reached your write unit limit for the \
        current month (2000000). To continue writing data, upgrade your plan.";
    const RATE: &str = "Request failed. You've reached the max query requests per second per \
        namespace. Pace your query requests or contact Pinecone Support to request a higher limit.";

    fn api(status: u16, message: &str) -> ClientError {
        ClientError::Api {
            status,
            message: message.to_string(),
        }
    }

    #[test]
    fn the_documented_monthly_messages_are_quota() {
        for message in [READ_UNITS, WRITE_UNITS, WRITE_UNITS_NUMBERED] {
            assert_eq!(
                classify_429(429, message),
                Some(Limit429::Quota),
                "{message}"
            );
        }
    }

    #[test]
    fn the_documented_per_second_message_is_rate() {
        assert_eq!(classify_429(429, RATE), Some(Limit429::Rate));
        assert_eq!(
            classify_429(429, "Pace your upsert requests"),
            Some(Limit429::Rate)
        );
    }

    #[test]
    fn a_json_wrapped_body_classifies_like_its_message() {
        let wrap = |message: &str| {
            format!(
                r#"{{"error":{{"code":"RESOURCE_EXHAUSTED","message":"{message}"}},"status":429}}"#
            )
        };
        assert_eq!(classify_429(429, &wrap(WRITE_UNITS)), Some(Limit429::Quota));
        assert_eq!(classify_429(429, &wrap(RATE)), Some(Limit429::Rate));
    }

    #[test]
    fn an_unrecognised_429_is_treated_as_a_rate_limit() {
        assert_eq!(classify_429(429, ""), Some(Limit429::Rate));
        assert_eq!(classify_429(429, "slow down"), Some(Limit429::Rate));
    }

    #[test]
    fn only_a_429_is_classified() {
        for status in [200, 400, 403, 500, 503] {
            assert_eq!(classify_429(status, WRITE_UNITS), None, "{status}");
        }
    }

    #[test]
    fn each_limit_has_its_sqlstate_and_hint() {
        assert_eq!(
            Limit429::Quota.sqlstate(),
            PgSqlErrorCode::ERRCODE_CONFIGURATION_LIMIT_EXCEEDED
        );
        assert_eq!(Limit429::Rate.sqlstate(), PgSqlErrorCode::ERRCODE_IO_ERROR);
        assert!(Limit429::Quota.hint().contains("billing period"));
        assert!(Limit429::Rate.hint().contains("retry in a moment"));
    }

    #[test]
    fn failure_text_adds_the_hint_to_a_429_only() {
        let quota = failure_text(&api(429, WRITE_UNITS));
        assert!(quota.contains("status 429"), "{quota}");
        assert!(quota.contains(WRITE_UNITS), "{quota}");
        assert!(quota.ends_with(Limit429::Quota.hint()), "{quota}");
        let rate = failure_text(&api(429, RATE));
        assert!(rate.ends_with(Limit429::Rate.hint()), "{rate}");

        let outage = api(503, "unavailable");
        assert_eq!(failure_text(&outage), outage.to_string());
        let timeout = ClientError::Timeout { ms: 5 };
        assert_eq!(failure_text(&timeout), timeout.to_string());
    }

    #[test]
    fn failure_text_adds_the_plan_hint_to_a_marked_first_write() {
        let refused = partition_lane::explain_first_write(
            api(400, "You've reached the maximum number of namespaces"),
            "public.t:801",
        );
        let text = failure_text(&refused);
        assert!(text.contains("public.t:801"), "{text}");
        assert!(
            text.ends_with(partition_lane::NAMESPACE_LIMIT_HINT),
            "{text}"
        );
        let plain = api(400, "Vector dimension 3 does not match");
        assert_eq!(failure_text(&plain), plain.to_string());
    }

    #[test]
    fn a_refusal_of_one_namespace_fails_that_partition_and_nothing_else_does() {
        let bad_request = ClientError::BadRequest {
            message: "bad id".to_string(),
        };
        let one = [
            bad_request,
            api(403, "maximum number of namespaces"),
            api(409, "conflict"),
            ClientError::Conflict {
                name: "x".to_string(),
            },
        ];
        for err in &one {
            assert!(fails_one_partition(err), "{err:?}");
        }

        let table_wide = [
            api(429, RATE),
            api(500, "boom"),
            api(503, "unavailable"),
            api(200, "could not parse response body as JSON: expected value"),
            api(404, "gone"),
            ClientError::Timeout { ms: 5 },
            ClientError::Transport {
                url: "https://host".to_string(),
                reason: "reset".to_string(),
            },
            ClientError::MissingApiKey,
            ClientError::Auth,
            ClientError::IndexNotFound {
                index: "i".to_string(),
            },
            ClientError::NamespaceNotFound {
                namespace: "n".to_string(),
            },
        ];
        for err in &table_wide {
            assert!(!fails_one_partition(err), "{err:?}");
        }
    }
}
