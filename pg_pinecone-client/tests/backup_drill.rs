//! ONE-OFF live drill for the pg_pinecone 0.4.0 disaster-recovery release
//! (`docs/disaster-recovery.md`). NOT part of the regular suite — run
//! explicitly:
//!
//!   PG_PINECONE_LIVE_TESTS=1 PG_PINECONE_DRILL_INDEX_A=... PG_PINECONE_DRILL_BACKUP_NAME=... \
//!       PG_PINECONE_DRILL_INDEX_RESTORED=... cargo test --test backup_drill -- --ignored --nocapture
//!
//! `PINECONE_API_KEY` must already be in the environment (inherited, never
//! passed on argv), and `PG_PINECONE_LIVE_TESTS=1` says the call to Pinecone is meant. Exercises the backup control-plane surface `pg_pinecone-client`
//! ships (`create_backup`/`describe_backup`/`create_index_from_backup`) that
//! isn't yet SQL-exposed in pg_pinecone 0.4.0 (see README's "Backup/stats client
//! surfaces" note) — this is the only way to drill it before a real SQL gate
//! exists. A permanent release asset (like `scripts/vacuum-drill.sh`):
//! `#[ignore]`d, so the regular suite never runs it, driven only by
//! `scripts/disaster-recovery-drill.sh`'s BACKUP leg.
//!
//! Prints `DRILL_<KEY>=<value>` lines the calling script
//! (`scripts/disaster-recovery-drill.sh`) greps for. On a plan that doesn't
//! support backups, `create_backup` fails and this test panics with
//! `DRILL_BACKUP_CREATE_ERROR=<the API's own message>` — the caller
//! distinguishes plan-gated-SKIPPED from a genuine FAIL by pattern-matching
//! that message, not by this test's own pass/fail exit code.

use std::{env, time::Duration};

use pg_pinecone_client::{Client, Config, CreateBackupRequest};

fn client() -> Client {
    let api_key = env::var("PINECONE_API_KEY")
        .expect("PINECONE_API_KEY must be set in the environment for this live drill");
    Client::new(Config {
        control_plane_url: "https://api.pinecone.io".to_string(),
        api_key,
        timeout: Duration::from_secs(30),
        user_agent: None,
    })
}

#[test]
#[ignore]
fn live_backup_create_and_restore_drill() {
    assert_eq!(
        env::var("PG_PINECONE_LIVE_TESTS").as_deref(),
        Ok("1"),
        "set PG_PINECONE_LIVE_TESTS=1 (and PINECONE_API_KEY) to run this live test"
    );
    let index =
        env::var("PG_PINECONE_DRILL_INDEX_A").expect("PG_PINECONE_DRILL_INDEX_A must be set");
    let backup_name = env::var("PG_PINECONE_DRILL_BACKUP_NAME")
        .expect("PG_PINECONE_DRILL_BACKUP_NAME must be set");
    let restored_name = env::var("PG_PINECONE_DRILL_INDEX_RESTORED")
        .expect("PG_PINECONE_DRILL_INDEX_RESTORED must be set");
    let client = client();

    let info = client
        .create_backup(&CreateBackupRequest {
            index: index.clone(),
            name: Some(backup_name.clone()),
            description: Some("pg_pinecone 0.4.0 disaster-recovery release-gate drill".to_string()),
        })
        .unwrap_or_else(|e| panic!("DRILL_BACKUP_CREATE_ERROR={e}"));
    println!("DRILL_BACKUP_ID={}", info.backup_id);
    println!("DRILL_BACKUP_STATUS={}", info.status);

    let deadline = std::time::Instant::now() + Duration::from_secs(600);
    let mut last = info;
    while last.status != "Ready" {
        assert!(
            std::time::Instant::now() < deadline,
            "DRILL_BACKUP_TIMEOUT last_status={}",
            last.status
        );
        std::thread::sleep(Duration::from_secs(5));
        last = client
            .describe_backup(&last.backup_id)
            .unwrap_or_else(|e| panic!("DRILL_BACKUP_DESCRIBE_ERROR={e}"));
        println!("DRILL_BACKUP_POLL_STATUS={}", last.status);
    }
    println!("DRILL_BACKUP_READY=true");
    println!(
        "DRILL_BACKUP_RECORD_COUNT={}",
        last.record_count
            .map(|c| c.to_string())
            .unwrap_or_else(|| "unknown".to_string())
    );

    let restored = client
        .create_index_from_backup(&last.backup_id, &restored_name)
        .unwrap_or_else(|e| panic!("DRILL_RESTORE_CREATE_ERROR={e}"));
    println!("DRILL_RESTORED_HOST={}", restored.host);

    let ready = client
        .wait_index_ready(
            &restored_name,
            Duration::from_secs(300),
            Duration::from_secs(3),
        )
        .unwrap_or_else(|e| panic!("DRILL_RESTORE_WAIT_ERROR={e}"));
    println!("DRILL_RESTORED_READY={}", ready.ready);
    println!("DRILL_RESTORED_DIMENSION={}", ready.dimension);
    println!("DRILL_RESTORED_METRIC={}", ready.metric);
}
