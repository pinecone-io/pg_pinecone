//! The background worker: a static launcher (registered in `_PG_init`, only
//! when `pg_pinecone` is preloaded via `shared_preload_libraries`) that spawns one
//! dynamic per-database worker per eligible database, and the per-database
//! worker itself.
//!
//! Launcher: connects to the `postgres` database (`BackgroundWorker::
//! connect_worker_to_spi(Some("postgres"), None)` — SPI needs a real,
//! initialized database context to function at all; connecting with no
//! database (`None, None`, "shared catalogs only") looks like it should be
//! enough for a plain `pg_database` scan, since that's a shared catalog, but
//! it isn't: SPI's underlying executor/relcache machinery assumes a valid
//! `MyDatabaseId` unconditionally, and crashed outright when this was tried —
//! `postgres` is the one database `initdb` always creates, so it's the
//! anchor every discovery scan runs from regardless of which databases
//! `pinecone.databases`/the scan itself ultimately targets), discovers target
//! databases every [`naptime`], and spawns a [`DynamicBackgroundWorker`] for
//! any that doesn't already have one it's tracking, isn't inside its own
//! post-exit [`RespawnCooldown`] window, or that the per-database advisory
//! lock (`outbox::worker_lock_held_for`) shows already running — that
//! advisory lock, not the launcher's own handle bookkeeping, is the
//! *authoritative* single-worker guard; the handle map is just an
//! optimization to skip pointless respawn attempts.
//!
//! Per-database worker: connects straight to its assigned database, takes the
//! per-database singleton advisory lock (exits immediately if some other
//! backend already holds it), checks `pg_pinecone` is installed (else exits),
//! then loops `outbox::Scheduler::round` with a commit after every round
//! and a `wait_latch(naptime)` in between (none while a backfill, rebuild or
//! recover job has chunks left: its next round starts at once) — re-checking
//! `pinecone.worker_enabled` every round and (every
//! [`EXTENSION_CHECK_EVERY_ROUNDS`] rounds) that `pg_pinecone` is still installed,
//! exiting cleanly the moment either fails — and exiting on its own after
//! [`IDLE_EXIT_ROUNDS`] consecutive idle rounds to free its
//! `max_worker_processes` slot — the launcher respawns one on demand. A SIGTERM ends the round
//! after its current backfill chunk (`outbox::shutdown_requested`) and the worker exits.
//!
//! Version guard ([`VersionGuard`]): every round starts by comparing the
//! installed extension version with this library's compiled one. While they
//! differ (a preloaded library keeps its old code after `ALTER EXTENSION
//! pg_pinecone UPDATE`) the round runs nothing and the worker only naps: old code
//! must not drain op kinds it does not know. A restart loads the new code.

use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use pgrx::{
    PgMemoryContexts,
    bgworkers::{
        BackgroundWorker, BackgroundWorkerBuilder, BackgroundWorkerStatus, DynamicBackgroundWorker,
        SignalWakeFlags,
    },
    prelude::*,
    spi::Spi,
};

use crate::{
    WORKER_DATABASES, WORKER_ENABLED, WORKER_NAPTIME_MS, catalog, docs_backfill, docs_outbox,
    outbox,
};

/// Current `pinecone.worker_naptime_ms` GUC value, as a `Duration` — the
/// launcher's own tick interval and every per-database worker's
/// between-round sleep both read this.
pub(crate) fn naptime() -> Duration {
    Duration::from_millis(WORKER_NAPTIME_MS.get() as u64)
}

/// Current `pinecone.worker_enabled` GUC value.
pub(crate) fn worker_enabled() -> bool {
    WORKER_ENABLED.get()
}

fn databases_csv() -> String {
    WORKER_DATABASES
        .get()
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Parses `pinecone.databases`'s CSV value into trimmed, non-empty database
/// names. Pure, so it's plain-`#[test]`-able without any SPI/Postgres
/// connection.
pub(crate) fn parse_databases_list(csv: &str) -> Vec<String> {
    csv.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Whether `_PG_init` should register the static launcher BGW — only when
/// `pg_pinecone` is actually being preloaded via `shared_preload_libraries`
/// (custom background workers can only be registered from that hook). Takes
/// the preload flag as a parameter, rather than reading
/// `pg_sys::process_shared_preload_libraries_in_progress` itself, so it's
/// plain-`#[test]`-able: a `cargo test` binary never runs inside a Postgres
/// process (that `extern "C"` global isn't something to poke safely there),
/// and the `#[pg_test]` cluster always preloads `pg_pinecone` (so it can't
/// exercise the "false" branch either) — hence testing this as an injectable,
/// pure predicate instead (absence can't be integration-tested).
pub(crate) fn should_register_launcher(preloading: bool) -> bool {
    preloading
}

/// [`discover_databases`]'s negative-result cache: once
/// a full `pg_database` scan (the `pinecone.databases = ''` case) finds no
/// eligible database, skip re-scanning for `naptime * NEGATIVE_SCAN_TTL_ROUNDS`
/// rather than re-running it every single tick. Also the window
/// [`RespawnCooldown`] (below) reuses for its own, separate purpose: both are
/// "don't act again on the same thing for about 10 naptimes" cooldowns.
const NEGATIVE_SCAN_TTL_ROUNDS: u32 = 10;

struct DiscoveryCache {
    empty_since: Option<Instant>,
}

impl DiscoveryCache {
    fn new() -> Self {
        Self { empty_since: None }
    }

    fn should_skip_scan(&self, naptime: Duration) -> bool {
        self.empty_since
            .is_some_and(|since| since.elapsed() < naptime * NEGATIVE_SCAN_TTL_ROUNDS)
    }

    fn record(&mut self, found_any: bool) {
        if found_any {
            self.empty_since = None;
        } else if self.empty_since.is_none() {
            self.empty_since = Some(Instant::now());
        }
    }
}

/// [`Launcher`]'s per-dboid respawn cooldown: once a
/// database's worker exits (idle timeout, crash, disabled GUC, dropped
/// extension — anything `reap_dead` observes), the launcher won't spawn it a
/// fresh one again until `naptime * NEGATIVE_SCAN_TTL_ROUNDS` has passed,
/// even if a later tick re-discovers it with fresh queued rows. Without
/// this, a worker that exits on its own idle timeout
/// (`IDLE_EXIT_ROUNDS`) gets respawned on literally the very next launcher
/// tick — constant spawn/exit churn that defeats the whole point of the
/// idle exit (freeing the `max_worker_processes` slot).
///
/// Trade-off: a write to a `sync_mode = 'async'` table whose worker just
/// exited can now wait up to this same cooldown window before a fresh
/// worker picks it up and drains it, instead of on the very next tick.
/// `sync_mode = 'sync'` tables are unaffected — those drain inline,
/// regardless of whether any worker is running.
struct RespawnCooldown {
    exited_at: HashMap<pg_sys::Oid, Instant>,
}

impl RespawnCooldown {
    fn new() -> Self {
        Self {
            exited_at: HashMap::new(),
        }
    }

    /// Records that `dboid`'s worker just exited — `tick` won't spawn it a
    /// new one until [`Self::should_skip`] stops saying so.
    fn record_exit(&mut self, dboid: pg_sys::Oid) {
        self.exited_at.insert(dboid, Instant::now());
    }

    /// Clears `dboid`'s cooldown once a fresh worker actually spawns for it,
    /// so a *later* exit starts the window over rather than reusing however
    /// much of a stale, already-expired one happened to remain.
    fn record_spawn(&mut self, dboid: pg_sys::Oid) {
        self.exited_at.remove(&dboid);
    }

    fn should_skip(&self, dboid: pg_sys::Oid, naptime: Duration) -> bool {
        self.exited_at
            .get(&dboid)
            .is_some_and(|since| since.elapsed() < naptime * NEGATIVE_SCAN_TTL_ROUNDS)
    }
}

/// The launcher's per-tick target list: the explicit `pinecone.databases` CSV
/// list if set, else every connectable, non-template database (a full
/// `pg_database` scan, subject to [`DiscoveryCache`]'s negative-result TTL).
fn discover_databases(cache: &mut DiscoveryCache, naptime: Duration) -> Vec<pg_sys::Oid> {
    let names = parse_databases_list(&databases_csv());
    if !names.is_empty() {
        return explicit_databases(&names);
    }
    if cache.should_skip_scan(naptime) {
        return Vec::new();
    }
    let found = scan_all_databases();
    cache.record(!found.is_empty());
    found
}

fn scan_all_databases() -> Vec<pg_sys::Oid> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT oid::oid AS oid FROM pg_catalog.pg_database \
                 WHERE datallowconn AND NOT datistemplate",
                None,
                &[],
            )
            .expect("pg_pinecone: launcher pg_database scan failed")
            .map(|row| {
                row.get_by_name::<pg_sys::Oid, _>("oid")
                    .expect("pg_pinecone: launcher pg_database scan row failed")
                    .expect("pg_pinecone: pg_database.oid unexpectedly NULL")
            })
            .collect()
    })
}

fn explicit_databases(names: &[String]) -> Vec<pg_sys::Oid> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT oid::oid AS oid FROM pg_catalog.pg_database \
                 WHERE datname = ANY($1) AND datallowconn",
                None,
                &[names.to_vec().into()],
            )
            .expect("pg_pinecone: launcher explicit database lookup failed")
            .map(|row| {
                row.get_by_name::<pg_sys::Oid, _>("oid")
                    .expect("pg_pinecone: launcher explicit database lookup row failed")
                    .expect("pg_pinecone: pg_database.oid unexpectedly NULL")
            })
            .collect()
    })
}

/// The per-database worker's startup singleton guard:
/// `false` means some backend already holds this database's worker lock —
/// the caller must exit 0 immediately; `true` means this backend now holds
/// it, for the rest of its process lifetime (a session-level advisory lock,
/// released automatically when the worker disconnects).
///
/// Checks `outbox::worker_lock_held()` (a `pg_locks` probe) *before*
/// attempting the real `pg_try_advisory_lock`: Postgres's advisory locks are
/// self-compatible (a session re-requesting a lock it already holds always
/// succeeds instantly), so a bare `pg_try_advisory_lock` retry could never
/// observe "held" from the very session that holds it — which is exactly the
/// scenario the `worker_singleton_lock_guard_exits_when_already_held` test
/// needs to drive deterministically from a single `#[pg_test]` connection. A
/// brand-new worker backend has never touched this lock itself, so in
/// production "held by any backend" and "held by another worker" coincide,
/// and the two checks always agree.
pub(crate) fn try_take_singleton_lock() -> bool {
    if outbox::worker_lock_held() {
        return false;
    }
    Spi::get_one_with_args::<bool>(
        "SELECT pg_catalog.pg_try_advisory_lock($1, $2::int4)",
        &[
            outbox::WORKER_LOCK_MAGIC.into(),
            unsafe { pg_sys::MyDatabaseId }.into(),
        ],
    )
    .expect("pg_pinecone: worker singleton lock attempt failed")
    .expect("pg_pinecone: pg_try_advisory_lock unexpectedly NULL")
}

/// Whether this per-database worker should run at all: `pg_pinecone` must be
/// installed in the connected database (a worker spawned for some other
/// database in the cluster — e.g. a full `pg_database` scan finding one with
/// no `pg_pinecone` at all — just exits quietly). A version mismatch is not
/// checked here: [`VersionGuard`] idles the rounds instead of ending the worker.
fn extension_ready() -> bool {
    catalog::extension_version("pg_pinecone").is_some()
}

/// Only the very same version may run: after `ALTER EXTENSION pg_pinecone UPDATE`
/// a preloaded library keeps its old code until Postgres restarts.
fn versions_match(compiled: &str, installed: &str) -> bool {
    compiled == installed
}

/// The installed version if it differs from `compiled`: what the guard idles
/// on, shared with `pinecone.worker_version()` so the two agree. No row at all
/// is no mismatch; `extension_ready` ends such a worker.
fn skew<'a>(compiled: &str, installed: Option<&'a str>) -> Option<&'a str> {
    installed.filter(|installed| !versions_match(compiled, installed))
}

#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    Run,
    /// Skip the round; carries the WARNING on the first round of a state.
    Skip(Option<String>),
}

/// Remembers the installed version the worker last idled on: one WARNING per
/// change of state, not per round.
pub(crate) struct VersionGuard {
    idle_on: Option<String>,
}

impl VersionGuard {
    pub(crate) fn new() -> Self {
        Self { idle_on: None }
    }

    fn decide(&mut self, compiled: &str, installed: Option<&str>) -> Verdict {
        let Some(installed) = skew(compiled, installed) else {
            self.idle_on = None;
            return Verdict::Run;
        };
        if self.idle_on.as_deref() == Some(installed) {
            return Verdict::Skip(None);
        }
        self.idle_on = Some(installed.to_string());
        Verdict::Skip(Some(format!(
            "pg_pinecone worker {compiled} idle: extension is {installed} — restart Postgres to \
             load the new worker"
        )))
    }
}

/// Runs `work`, the round's first job, unless the guard skips the round
/// (`None`: run nothing else, just nap). Reads the installed version in the
/// caller's transaction, the round's own.
pub(crate) fn guarded<T>(
    guard: &mut VersionGuard,
    compiled: &str,
    work: impl FnOnce() -> T,
) -> Option<T> {
    let installed = catalog::extension_version("pg_pinecone");
    match guard.decide(compiled, installed.as_deref()) {
        Verdict::Run => Some(work()),
        Verdict::Skip(announce) => {
            if let Some(message) = announce {
                warning!("{message}");
            }
            None
        }
    }
}

/// `pinecone.worker_version()`: `(compiled, installed, idle)`.
pub(crate) fn worker_version() -> (String, Option<String>, bool) {
    let compiled = env!("CARGO_PKG_VERSION");
    let installed = catalog::extension_version("pg_pinecone");
    let idle = skew(compiled, installed.as_deref()).is_some();
    (compiled.to_string(), installed, idle)
}

/// Consecutive idle rounds (no eligible relid had anything to drain) a
/// per-database worker tolerates before exiting — frees its
/// `max_worker_processes` slot; the launcher respawns one on demand the next
/// time this database has queued work.
const IDLE_EXIT_ROUNDS: u32 = 30;

/// Registers the static launcher BGW — called from `_PG_init`, guarded by
/// [`should_register_launcher`].
pub(crate) fn register_launcher() {
    BackgroundWorkerBuilder::new("pg_pinecone launcher")
        .set_function("pg_pinecone_launcher_main")
        .set_library("pg_pinecone")
        .enable_spi_access()
        .set_restart_time(Some(Duration::from_secs(5)))
        .load();
}

/// Registers a dynamic per-database worker for `dboid`. `set_restart_time(None)`:
/// a worker that exits (idle timeout, lock already held, extension
/// absent, `SIGTERM`) should just free its slot — the launcher
/// decides afresh, next tick, whether this database still needs one, rather
/// than Postgres immediately restarting the exact same process.
///
/// Runs `load_dynamic()` with `CurrentMemoryContext` switched to
/// `TopMemoryContext`: Postgres's own `RegisterDynamicBackgroundWorker`
/// (which it calls into) `palloc`s the `DynamicBackgroundWorkerHandle` it
/// returns in whatever context is current at the call site. Every caller
/// here runs inside `Launcher::tick`'s own `BackgroundWorker::transaction`
/// scope — a context Postgres destroys the instant that transaction commits
/// — while [`Launcher`] keeps the returned handle in `self.workers` across
/// every later tick (each its own separate transaction). Without switching
/// to a context that outlives any single tick first, the very next
/// `reap_dead()` call dereferences already-freed memory and reliably
/// crashes the whole postmaster (`Assert("handle->slot <
/// max_worker_processes")` in Postgres's own `bgworker.c`) — reproduced by
/// hand against a real cluster with `pinecone.worker_enabled = on`, and by
/// running `worker_bgw_drain_against_mock` (this module's ignored
/// integration test) in isolation, before this fix.
fn spawn_worker(dboid: pg_sys::Oid) -> Result<DynamicBackgroundWorker, ()> {
    unsafe {
        PgMemoryContexts::TopMemoryContext.switch_to(|_cxt| {
            BackgroundWorkerBuilder::new(&format!("pg_pinecone worker (database {dboid})"))
                .set_function("pg_pinecone_worker_main")
                .set_library("pg_pinecone")
                .enable_spi_access()
                .set_restart_time(None)
                .set_argument(Some(pg_sys::Datum::from(dboid)))
                .load_dynamic()
                .map_err(|_| ())
        })
    }
}

/// The launcher's in-memory state: which databases it believes currently
/// have a worker running, plus [`DiscoveryCache`]'s TTL bookkeeping and
/// [`RespawnCooldown`]'s per-dboid post-exit cooldown.
struct Launcher {
    workers: HashMap<pg_sys::Oid, DynamicBackgroundWorker>,
    discovery: DiscoveryCache,
    respawn_cooldown: RespawnCooldown,
}

impl Launcher {
    fn new() -> Self {
        Self {
            workers: HashMap::new(),
            discovery: DiscoveryCache::new(),
            respawn_cooldown: RespawnCooldown::new(),
        }
    }

    /// One tick: reap handles for workers that have exited, then spawn a
    /// fresh worker for any currently-discovered database that doesn't
    /// already have one tracked, isn't inside its own post-exit
    /// [`RespawnCooldown`] window, *and* whose advisory lock isn't already
    /// held (the authoritative check — see this module's rustdoc).
    fn tick(&mut self) {
        self.reap_dead();
        let naptime = naptime();
        for dboid in discover_databases(&mut self.discovery, naptime) {
            if self.workers.contains_key(&dboid) {
                continue;
            }
            if self.respawn_cooldown.should_skip(dboid, naptime) {
                continue;
            }
            if outbox::worker_lock_held_for(dboid) {
                continue;
            }
            if let Ok(handle) = spawn_worker(dboid) {
                self.respawn_cooldown.record_spawn(dboid);
                self.workers.insert(dboid, handle);
            }
        }
    }

    fn reap_dead(&mut self) {
        let respawn_cooldown = &mut self.respawn_cooldown;
        self.workers.retain(|dboid, handle| {
            let alive = !matches!(
                handle.pid(),
                Err(BackgroundWorkerStatus::Stopped | BackgroundWorkerStatus::PostmasterDied)
            );
            if !alive {
                respawn_cooldown.record_exit(*dboid);
            }
            alive
        });
    }
}

// `#[unsafe(no_mangle)]` + `pub`: `BackgroundWorkerBuilder::set_function`
// hands Postgres a plain string it `dlsym`s at spawn time — `#[pg_guard]`
// alone only auto-adds this for `_PG_init`/`_PG_fini`, so every other BGW
// entry point needs it spelled out explicitly, or the symbol gets mangled
// (or dead-code-eliminated, since nothing in Rust itself ever calls it) and
// the worker silently fails to start.
#[unsafe(no_mangle)]
#[pg_guard]
pub extern "C-unwind" fn pg_pinecone_launcher_main(_arg: pg_sys::Datum) {
    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGHUP | SignalWakeFlags::SIGTERM);
    // See this module's rustdoc: SPI needs a real database context, so the
    // launcher anchors to `postgres` (always present) even though its own
    // queries only ever touch shared catalogs.
    BackgroundWorker::connect_worker_to_spi(Some("postgres"), None);

    let mut launcher = Launcher::new();
    // The launcher's cluster-wide liveness signal:
    // `pinecone.flush`/`status()`'s `queue_state` probe this session-scoped
    // advisory lock (`outbox::launcher_lock_held`) to decide whether a
    // queued backlog will *ever* drain — the per-database worker's own lock
    // is useless for that (it idle-exits between backlogs by design). Held
    // in step with `pinecone.worker_enabled` rather than for the whole
    // process lifetime: a disabled launcher spawns nothing, which is exactly
    // the "nothing will drain this" condition those probes exist to detect.
    let mut holds_liveness_lock = false;
    loop {
        if worker_enabled() {
            if !holds_liveness_lock {
                holds_liveness_lock = BackgroundWorker::transaction(std::panic::AssertUnwindSafe(
                    try_take_launcher_liveness_lock,
                ));
            }
            BackgroundWorker::transaction(std::panic::AssertUnwindSafe(|| launcher.tick()));
        } else if holds_liveness_lock {
            BackgroundWorker::transaction(std::panic::AssertUnwindSafe(
                release_launcher_liveness_lock,
            ));
            holds_liveness_lock = false;
        }
        if !BackgroundWorker::wait_latch(Some(naptime())) {
            break;
        }
        if BackgroundWorker::sighup_received() {
            unsafe { pg_sys::ProcessConfigFile(pg_sys::GucContext::PGC_SIGHUP) };
        }
    }
}

/// Takes the launcher's session-scoped liveness lock
/// (`outbox::LAUNCHER_LOCK_MAGIC`, second key fixed at `0` — see its rustdoc
/// for why there's no per-database component). `try_`, not a blocking
/// `pg_advisory_lock`: a test simulating a live launcher may already hold
/// it, and the real launcher must never wedge its whole loop behind that —
/// it just retries next tick. Guarded by the caller's own `holds` flag, never
/// re-acquired while held: session advisory locks stack a hold count per
/// acquisition, and the paired [`release_launcher_liveness_lock`] releases
/// exactly one.
fn try_take_launcher_liveness_lock() -> bool {
    Spi::get_one_with_args::<bool>(
        "SELECT pg_catalog.pg_try_advisory_lock($1, 0)",
        &[outbox::LAUNCHER_LOCK_MAGIC.into()],
    )
    .expect("pg_pinecone: launcher liveness lock attempt failed")
    .expect("pg_pinecone: pg_try_advisory_lock unexpectedly NULL")
}

/// Releases [`try_take_launcher_liveness_lock`]'s lock — the launcher drops
/// it the tick it notices `pinecone.worker_enabled` went off, so
/// `outbox::launcher_lock_held` flips to `false` (and `flush()`/`status()`
/// report `NO WORKER`) without waiting for the launcher process to exit —
/// it never does; it keeps ticking, disabled, until postmaster shutdown.
fn release_launcher_liveness_lock() {
    Spi::run_with_args(
        "SELECT pg_catalog.pg_advisory_unlock($1, 0)",
        &[outbox::LAUNCHER_LOCK_MAGIC.into()],
    )
    .expect("pg_pinecone: launcher liveness lock release failed");
}

/// How many loop rounds a running per-database worker goes between
/// re-checking that `pg_pinecone` is still installed (`extension_ready`) —
/// `pinecone.worker_enabled` is re-checked *every* round instead (a plain GUC
/// read, no SPI/catalog query needed), but `extension_ready` needs one, and a
/// `DROP EXTENSION` is rare enough that a several-round delay in noticing it
/// doesn't matter. (The *version* is compared every round, by
/// [`VersionGuard`].)
const EXTENSION_CHECK_EVERY_ROUNDS: u64 = 30;

/// How long the worker waits after a round: not at all while a job has chunks left (live rows go
/// first in the next round anyway), else `naptime`.
pub(crate) fn pause_after(chunks_left: bool, naptime: Duration) -> Duration {
    if chunks_left { Duration::ZERO } else { naptime }
}

/// Waits `duration` on the latch (a zero wait still serves interrupts, SIGHUP and SIGTERM). `false`:
/// the postmaster died or SIGTERM arrived.
fn pause(duration: Duration) -> bool {
    if !BackgroundWorker::wait_latch(Some(duration)) {
        return false;
    }
    if BackgroundWorker::sighup_received() {
        unsafe { pg_sys::ProcessConfigFile(pg_sys::GucContext::PGC_SIGHUP) };
    }
    true
}

#[unsafe(no_mangle)]
#[pg_guard]
pub extern "C-unwind" fn pg_pinecone_worker_main(arg: pg_sys::Datum) {
    let dboid = pg_sys::Oid::from(arg.value() as u32);
    BackgroundWorker::attach_signal_handlers(SignalWakeFlags::SIGHUP | SignalWakeFlags::SIGTERM);
    BackgroundWorker::connect_worker_to_spi_by_oid(Some(dboid), None);

    let ready = BackgroundWorker::transaction(std::panic::AssertUnwindSafe(|| {
        try_take_singleton_lock() && extension_ready()
    }));
    if !ready {
        return;
    }

    let mut scheduler = outbox::Scheduler::new();
    let mut doc_scheduler = docs_outbox::DocScheduler::new();
    let mut version_guard = VersionGuard::new();
    let mut idle_rounds = 0u32;
    let mut round_num: u64 = 0;
    loop {
        // Re-checked every round: once running, this
        // worker previously never noticed `pinecone.worker_enabled` being
        // turned back off, or `pg_pinecone` being updated out from under it,
        // until its own idle-exit timeout eventually happened to kick in —
        // which a busy database with steady backlog might never reach.
        if !worker_enabled() {
            log!("pg_pinecone worker: pinecone.worker_enabled is now off — exiting");
            break;
        }
        round_num += 1;
        if round_num.is_multiple_of(EXTENSION_CHECK_EVERY_ROUNDS)
            && !BackgroundWorker::transaction(std::panic::AssertUnwindSafe(extension_ready))
        {
            break;
        }

        // Guard first, in the drain's transaction: a skipped round runs nothing else and is no
        // idle round, so the worker stays up and resumes once the versions agree.
        let Some(round) = BackgroundWorker::transaction(std::panic::AssertUnwindSafe(|| {
            guarded(&mut version_guard, env!("CARGO_PKG_VERSION"), || {
                scheduler.round()
            })
        })) else {
            if !pause(naptime()) {
                break;
            }
            continue;
        };
        // SIGTERM: the round ended after its current chunk and committed, so nothing else starts.
        if outbox::shutdown_requested() {
            break;
        }
        // The document lane (`track_docs` tables): its own transactions, one per table's step, so
        // a failing or slow Pinecone documents API never holds back the vector drain above (or
        // another table's lane lock), and the other way round. Async only: documents are never
        // delivered inside the writer's statement.
        let docs_active = docs_outbox::worker_doc_round(&mut doc_scheduler);
        // Then the backfill chunks of those tables: one transaction per chunk (the table lock
        // is held only inside a chunk, every cursor commits at once), for at most
        // `pinecone.backfill_round_ms`. A pending job keeps `docs_active` true (which also
        // spares databases without document tables the extra transaction), so the worker does
        // not idle out between budgets.
        let mut docs_chunks_left = false;
        if docs_active {
            let step = docs_backfill::worker_step(&mut doc_scheduler);
            if step.terminated {
                break;
            }
            docs_chunks_left = step.chunks_left;
        }
        idle_rounds = if round.active || docs_active {
            0
        } else {
            idle_rounds + 1
        };

        // Local-engine maintenance: a separate round, own
        // transaction — `crate::maintain::run_round`'s own no-op-unless-
        // `pinecone.auto_maintain` gate, evaluated every round alongside the
        // outbox drain. Deliberately NOT folded into `any_active`/
        // `idle_rounds`: a database with no outbox backlog but scheduled
        // maintenance still idle-exits on its own ordinary cadence — the
        // launcher's `RespawnCooldown` window (`worker.rs` module rustdoc)
        // just means drift gets re-checked every ~10 naptimes instead of
        // every one while otherwise idle, which is fine for a threshold
        // meant to trip at 20-30% drift, not a tight SLA.
        BackgroundWorker::transaction(std::panic::AssertUnwindSafe(|| {
            crate::maintain::run_round();
        }));

        // Lifecycle: pending `pinecone.demote_index` rebuilds — same
        // "separate round, not folded into idle_rounds" rationale as
        // `maintain::run_round` just above (a demote's REINDEX wait is not a
        // tight SLA either). ONE TRANSACTION PER STEP, deliberately not one
        // `lifecycle::run_round()` call in one transaction: each demote's
        // REINDEX + registry delete + outbox purge + pending-row purge commit
        // FIRST (their own
        // `demote_step` transaction — releasing the rebuild's
        // AccessExclusiveLock before any network wait), and only then does
        // the remote-deletion HTTP call run, in its own follow-up
        // transaction where nothing local remains to abort — an abort after
        // the remote data is gone is structurally impossible, and the AEL
        // never spans HTTP.
        let pending = BackgroundWorker::transaction(std::panic::AssertUnwindSafe(|| {
            crate::lifecycle::pending_demotes()
        }));
        for pd in pending {
            BackgroundWorker::transaction(std::panic::AssertUnwindSafe(|| {
                crate::lifecycle::demote_step(&pd);
            }));
        }
        // Phase 2 is a RETRIABLE queue drained separately —
        // `demote_step` persists each cleanup obligation to
        // `pinecone.pending_cleanups` (in its own committed transaction above),
        // and this loop retries the remote-delete HTTP call, one transaction
        // per row, removing the row only once the resource is confirmed gone.
        // A partition or crash that outlasts phase 1 just leaves the row for a
        // later round, so the superseded remote index is never orphaned.
        let cleanups = BackgroundWorker::transaction(std::panic::AssertUnwindSafe(|| {
            crate::lifecycle::pending_cleanups()
        }));
        for cleanup in cleanups {
            BackgroundWorker::transaction(std::panic::AssertUnwindSafe(|| {
                crate::lifecycle::remote_cleanup(&cleanup);
            }));
        }
        if idle_rounds >= IDLE_EXIT_ROUNDS {
            break;
        }
        if !pause(pause_after(
            round.chunks_left || docs_chunks_left,
            naptime(),
        )) {
            break;
        }
    }
}

// Pure-Rust (no pgrx/pg dependency), unlike this crate's `#[pg_test]`s — runs
// under plain `cargo test`/`cargo pgrx test` alike.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_databases_list_trims_and_skips_empty_entries() {
        assert_eq!(
            parse_databases_list(" alpha, beta ,, gamma"),
            vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
        );
    }

    #[test]
    fn parse_databases_list_empty_string_is_empty() {
        assert!(parse_databases_list("").is_empty());
        assert!(parse_databases_list("   ").is_empty());
    }

    #[test]
    fn the_worker_starts_the_next_round_at_once_while_chunks_are_left() {
        let nap = Duration::from_millis(2000);
        assert_eq!(pause_after(true, nap), Duration::ZERO);
        assert_eq!(
            pause_after(false, nap),
            nap,
            "an idle or blocked queue naps"
        );
    }

    #[test]
    fn gating_fn_only_registers_when_preloading() {
        assert!(should_register_launcher(true));
        assert!(!should_register_launcher(false));
    }

    #[test]
    fn versions_match_only_for_the_very_same_version() {
        assert!(versions_match("0.9.0", "0.9.0"));
        assert!(!versions_match("0.9.0", "0.9.1"));
        assert!(!versions_match("0.9.1", "0.9.0"));
        assert!(!versions_match("0.9.0", "0.9.0-rc1"));
        assert!(!versions_match("0.9.0", ""));
    }

    #[test]
    fn version_guard_warns_once_per_state_and_runs_when_versions_agree() {
        let mut guard = VersionGuard::new();
        assert_eq!(guard.decide("0.9.0", Some("0.9.0")), Verdict::Run);

        // A mismatch announces itself on its first round only.
        assert_eq!(
            guard.decide("0.9.0", Some("0.9.1")),
            Verdict::Skip(Some(
                "pg_pinecone worker 0.9.0 idle: extension is 0.9.1 — restart Postgres to load the \
                 new worker"
                    .to_string()
            ))
        );
        assert_eq!(guard.decide("0.9.0", Some("0.9.1")), Verdict::Skip(None));
        assert_eq!(guard.decide("0.9.0", Some("0.9.1")), Verdict::Skip(None));

        // Another installed version is a new state: announced again.
        assert!(matches!(
            guard.decide("0.9.0", Some("0.9.2")),
            Verdict::Skip(Some(_))
        ));

        // Agreeing again resumes; a later mismatch is a new state too.
        assert_eq!(guard.decide("0.9.0", Some("0.9.0")), Verdict::Run);
        assert!(matches!(
            guard.decide("0.9.0", Some("0.9.1")),
            Verdict::Skip(Some(_))
        ));
    }

    #[test]
    fn version_guard_leaves_a_missing_extension_row_to_the_presence_check() {
        let mut guard = VersionGuard::new();
        assert_eq!(guard.decide("0.9.0", None), Verdict::Run);

        // ...and forgets an idle state it was in.
        assert!(matches!(
            guard.decide("0.9.0", Some("0.9.1")),
            Verdict::Skip(Some(_))
        ));
        assert_eq!(guard.decide("0.9.0", None), Verdict::Run);
        assert!(matches!(
            guard.decide("0.9.0", Some("0.9.1")),
            Verdict::Skip(Some(_))
        ));
    }

    #[test]
    fn discovery_cache_skips_scan_within_ttl_then_resumes() {
        let mut cache = DiscoveryCache::new();
        assert!(!cache.should_skip_scan(Duration::from_millis(10)));

        cache.record(false); // empty scan
        assert!(cache.should_skip_scan(Duration::from_secs(3600)));

        cache.record(true); // a later scan found something
        assert!(!cache.should_skip_scan(Duration::from_secs(3600)));
    }

    #[test]
    fn respawn_cooldown_skips_then_resumes_after_ttl_or_a_fresh_spawn() {
        let mut cooldown = RespawnCooldown::new();
        let dboid = pg_sys::Oid::from(16_384u32);
        assert!(!cooldown.should_skip(dboid, Duration::from_millis(10)));

        cooldown.record_exit(dboid);
        assert!(cooldown.should_skip(dboid, Duration::from_secs(3600)));

        // A short enough naptime means the cooldown window has already
        // elapsed, regardless of the exit having just been recorded.
        assert!(!cooldown.should_skip(dboid, Duration::from_nanos(1)));

        cooldown.record_spawn(dboid);
        assert!(!cooldown.should_skip(dboid, Duration::from_secs(3600)));

        // An unrelated database was never in cooldown to begin with.
        let other = pg_sys::Oid::from(16_385u32);
        assert!(!cooldown.should_skip(other, Duration::from_secs(3600)));
    }
}
