//! ONE-OFF live probe: what does `POST /vectors/update`
//! (`Client::update_vector_metadata`) actually return for an id that
//! doesn't exist in the index — a 200 no-op, or a 404? NOT part of the
//! regular suite — run explicitly:
//!
//!   PG_PINECONE_LIVE_TESTS=1 cargo test -p pg_pinecone-client --test update_missing_id_probe -- --ignored --nocapture
//!
//! `PINECONE_API_KEY` must already be in the environment (inherited, never
//! passed on argv), and `PG_PINECONE_LIVE_TESTS=1` says the call to Pinecone is meant. Creates a THROWAWAY index `pg_pinecone-d4-<pid>`, upserts one
//! real vector, then calls `update_vector_metadata` against (a) that real id
//! and (b) a sibling id that was never upserted — printing the raw
//! `Result<(), Error>` for both. Deletes the index at the end and VERIFIES
//! the deletion. Findings feed `pg_pinecone-client/src/types.rs`'s
//! `UpdateVectorMetadataRequest` rustdoc and
//! `sync_trigger::try_update_metadata`'s ghost handling — kept (like
//! `backup_drill.rs`/`fetch_by_metadata_eval.rs`) as a reproducible probe.

use std::{env, time::Duration};

use pg_pinecone_client::{
    Client, Config, CreateIndexRequest, UpdateVectorMetadataRequest, UpsertVectorsRequest, Vector,
};
use serde_json::json;

fn client() -> Client {
    let api_key = env::var("PINECONE_API_KEY")
        .expect("PINECONE_API_KEY must be set in the environment for this live probe");
    Client::new(Config {
        control_plane_url: "https://api.pinecone.io".to_string(),
        api_key,
        timeout: Duration::from_secs(30),
        user_agent: None,
    })
}

#[test]
#[ignore]
fn live_update_metadata_on_missing_id_probe() {
    assert_eq!(
        env::var("PG_PINECONE_LIVE_TESTS").as_deref(),
        Ok("1"),
        "set PG_PINECONE_LIVE_TESTS=1 (and PINECONE_API_KEY) to run this live test"
    );
    let index = format!("pg_pinecone-d4-{}", std::process::id());
    let client = client();
    println!("PROBE_INDEX={index}");

    client
        .create_index(&CreateIndexRequest {
            name: index.clone(),
            dimension: 8,
            metric: "cosine".to_string(),
            spec: json!({"serverless": {"cloud": "aws", "region": "us-east-1"}}),
        })
        .unwrap_or_else(|e| panic!("PROBE_CREATE_ERROR={e}"));
    client
        .wait_index_ready(&index, Duration::from_secs(300), Duration::from_secs(3))
        .unwrap_or_else(|e| panic!("PROBE_READY_ERROR={e}"));

    // One real vector, so the "existing id" leg of the probe has something to
    // hit — proves the merge-semantics half of the open verification note at
    // the same time as the missing-id half.
    client
        .upsert_vectors(&UpsertVectorsRequest {
            index: index.clone(),
            namespace: None,
            vectors: vec![Vector {
                id: "real-1".to_string(),
                values: vec![0.1; 8],
                metadata: Some(json!({"kept": "yes", "will_be_replaced": "old"})),
            }],
        })
        .unwrap_or_else(|e| panic!("PROBE_UPSERT_ERROR={e}"));

    // Serverless is eventually consistent — wait until the upsert is visible.
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    loop {
        let stats = client
            .describe_index_stats(&index)
            .unwrap_or_else(|e| panic!("PROBE_STATS_ERROR={e}"));
        if stats.total_vector_count >= 1 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "PROBE_FRESHNESS_TIMEOUT"
        );
        std::thread::sleep(Duration::from_secs(2));
    }

    // ---- leg (a): update on the REAL id — confirms merge (not replace)
    // semantics: `kept` must survive, `will_be_replaced` must take the new
    // value, and a brand-new key must appear.
    let real_result = client.update_vector_metadata(&UpdateVectorMetadataRequest {
        index: index.clone(),
        namespace: None,
        id: "real-1".to_string(),
        metadata: json!({"will_be_replaced": "new", "added": "yes"}),
    });
    println!("PROBE_REAL_ID_UPDATE_RESULT={real_result:?}");
    real_result.unwrap_or_else(|e| panic!("PROBE_REAL_ID_UPDATE_ERROR={e}"));

    let fetched = client
        .fetch_vectors(&pg_pinecone_client::FetchVectorsRequest {
            index: index.clone(),
            namespace: None,
            ids: vec!["real-1".to_string()],
        })
        .unwrap_or_else(|e| panic!("PROBE_FETCH_AFTER_UPDATE_ERROR={e}"));
    let after = fetched.get("real-1").expect("PROBE_REAL_ID_VANISHED");
    println!("PROBE_REAL_ID_METADATA_AFTER={:?}", after.metadata);
    let meta = after
        .metadata
        .as_ref()
        .expect("PROBE_NO_METADATA_AFTER_UPDATE");
    assert_eq!(
        meta.get("kept").and_then(|v| v.as_str()),
        Some("yes"),
        "MERGE semantics broken: an untouched key must survive a setMetadata update"
    );
    assert_eq!(
        meta.get("will_be_replaced").and_then(|v| v.as_str()),
        Some("new"),
        "a changed key must take the new value"
    );
    assert_eq!(
        meta.get("added").and_then(|v| v.as_str()),
        Some("yes"),
        "a brand-new key must be added"
    );
    println!(
        "PROBE_MERGE_SEMANTICS=confirmed (untouched key kept, changed key replaced, new key added)"
    );

    // ---- leg (b): update on an id that was NEVER upserted — the ghost case
    // the sync trigger's metadata update can hit.
    let missing_result = client.update_vector_metadata(&UpdateVectorMetadataRequest {
        index: index.clone(),
        namespace: None,
        id: "never-upserted-ghost-id".to_string(),
        metadata: json!({"whatever": "value"}),
    });
    println!("PROBE_MISSING_ID_UPDATE_RESULT={missing_result:?}");
    match &missing_result {
        Ok(()) => println!("PROBE_MISSING_ID_UPDATE=200 no-op (Ok(()))"),
        Err(e) => println!("PROBE_MISSING_ID_UPDATE=Err {e:?}"),
    }

    // Critical correctness question: did that 200 silently CREATE a phantom
    // vector (metadata but no real values) rather than truly no-op? Give
    // serverless's eventual consistency a moment, then check.
    std::thread::sleep(Duration::from_secs(5));
    let ghost_fetch = client
        .fetch_vectors(&pg_pinecone_client::FetchVectorsRequest {
            index: index.clone(),
            namespace: None,
            ids: vec!["never-upserted-ghost-id".to_string()],
        })
        .unwrap_or_else(|e| panic!("PROBE_GHOST_FETCH_ERROR={e}"));
    match ghost_fetch.get("never-upserted-ghost-id") {
        None => println!("PROBE_GHOST_PHANTOM_CHECK=absent (true no-op, nothing created)"),
        Some(v) => println!(
            "PROBE_GHOST_PHANTOM_CHECK=EXISTS! values.len()={} metadata={:?} — setMetadata on a \
             missing id CREATED a phantom vector",
            v.values.len(),
            v.metadata
        ),
    }
    let stats_after = client
        .describe_index_stats(&index)
        .unwrap_or_else(|e| panic!("PROBE_STATS_AFTER_ERROR={e}"));
    println!(
        "PROBE_TOTAL_VECTOR_COUNT_AFTER_GHOST_UPDATE={}",
        stats_after.total_vector_count
    );

    // ---- teardown: delete and VERIFY ----
    client
        .delete_index(&index)
        .unwrap_or_else(|e| panic!("PROBE_DELETE_ERROR={e}"));
    let deadline = std::time::Instant::now() + Duration::from_secs(120);
    loop {
        match client.describe_index(&index) {
            Err(pg_pinecone_client::Error::IndexNotFound { .. }) => break,
            Ok(_) => {}
            Err(e) => panic!("PROBE_DELETE_VERIFY_ERROR={e}"),
        }
        assert!(
            std::time::Instant::now() < deadline,
            "PROBE_DELETE_VERIFY_TIMEOUT index={index} still describable"
        );
        std::thread::sleep(Duration::from_secs(2));
    }
    println!("PROBE_INDEX_DELETED=verified");
}
