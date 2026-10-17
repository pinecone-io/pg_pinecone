//! ONE-OFF live measurement of `fetch_by_metadata`. NOT part of the regular
//! suite — run explicitly:
//!
//!   PG_PINECONE_LIVE_TESTS=1 cargo test -p pg_pinecone-client --test fetch_by_metadata_eval -- --ignored --nocapture
//!
//! `PINECONE_API_KEY` must already be in the environment (inherited, never
//! passed on argv), and `PG_PINECONE_LIVE_TESTS=1` says the call to Pinecone is meant. Creates a THROWAWAY index `pg_pinecone-eval-fbm-<pid>`, upserts
//! ~20k small vectors stamped with one synthetic constant metadata key
//! (`pg_pinecone_batch`), then measures:
//!
//! 1. `fetch_by_metadata` (constant-key filter) pages/sec + records/sec,
//!    full pagination walk — including whether it reliably returns ALL
//!    records exactly once (the filter-semantics question).
//! 2. The same data via the `list_vector_ids_page` + `fetch_vectors` path
//!    `pinecone.recover_from_remote` uses today.
//! 3. fp32 byte-fidelity of every value returned by both paths, against the
//!    exact bits upserted (deterministic seeded generator).
//!
//! Deletes the index at the end and VERIFIES the deletion (describe until
//! IndexNotFound). Kept (like `backup_drill.rs`) as a reproducible measurement
//! harness.

use std::{
    collections::HashMap,
    env,
    time::{Duration, Instant},
};

use pg_pinecone_client::{
    Client, Config, CreateIndexRequest, FetchByMetadataRequest, FetchVectorsRequest,
    ListVectorIdsPageRequest, UpsertVectorsRequest, Vector,
};
use serde_json::json;

/// Vector count — ~20k (2 full default-limit pages + a
/// remainder, so the pagination edge is actually exercised).
const N_VECTORS: usize = 20_000;
/// Small vectors: 8 dims keeps a 10k-record page well under the endpoint's
/// documented 4MB response cap while still carrying real fp32 payloads.
const DIMENSION: usize = 8;
/// The synthetic constant metadata key the write path would have to stamp if
/// recover_from_remote adopted this endpoint (~15 bytes on the wire).
const BATCH_KEY: &str = "pg_pinecone_batch";
const BATCH_VALUE: &str = "eval";

fn client() -> Client {
    let api_key = env::var("PINECONE_API_KEY")
        .expect("PINECONE_API_KEY must be set in the environment for this live eval");
    Client::new(Config {
        control_plane_url: "https://api.pinecone.io".to_string(),
        api_key,
        timeout: Duration::from_secs(30),
        user_agent: None,
    })
}

/// Deterministic fp32 generator (splitmix64-derived), so byte-fidelity can be
/// asserted without storing 20k vectors anywhere: regenerate and compare bits.
/// Values are finite, roughly in (-1, 1), and use enough mantissa entropy
/// that a lossy (f32->f64->text->f32 or truncating) round-trip would show.
fn vector_for(i: u64) -> Vec<f32> {
    let mut state = i.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..DIMENSION)
        .map(|_| {
            state ^= state >> 30;
            state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            state ^= state >> 27;
            state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
            state ^= state >> 31;
            // 24 mantissa-significant bits, scaled to (-1, 1).
            ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

fn assert_bit_exact(id: &str, got: &[f32]) {
    let i: u64 = id.parse().expect("eval ids are numeric");
    let want = vector_for(i);
    assert_eq!(
        got.len(),
        want.len(),
        "id {id}: dimension changed in flight"
    );
    for (d, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "id {id} dim {d}: fp32 bits differ (got {g}, want {w}) — NOT byte-fidelity"
        );
    }
}

#[test]
#[ignore]
fn live_fetch_by_metadata_eval() {
    assert_eq!(
        env::var("PG_PINECONE_LIVE_TESTS").as_deref(),
        Ok("1"),
        "set PG_PINECONE_LIVE_TESTS=1 (and PINECONE_API_KEY) to run this live test"
    );
    let index = format!("pg_pinecone-eval-fbm-{}", std::process::id());
    let client = client();
    println!("EVAL_INDEX={index}");

    // ---- setup: create index, upsert 20k stamped vectors ----
    client
        .create_index(&CreateIndexRequest {
            name: index.clone(),
            dimension: DIMENSION as u32,
            metric: "cosine".to_string(),
            spec: json!({"serverless": {"cloud": "aws", "region": "us-east-1"}}),
        })
        .unwrap_or_else(|e| panic!("EVAL_CREATE_ERROR={e}"));
    client
        .wait_index_ready(&index, Duration::from_secs(300), Duration::from_secs(3))
        .unwrap_or_else(|e| panic!("EVAL_READY_ERROR={e}"));

    let upsert_start = Instant::now();
    for chunk_start in (0..N_VECTORS).step_by(1000) {
        let vectors: Vec<Vector> = (chunk_start..(chunk_start + 1000).min(N_VECTORS))
            .map(|i| Vector {
                id: i.to_string(),
                values: vector_for(i as u64),
                metadata: Some(json!({BATCH_KEY: BATCH_VALUE})),
            })
            .collect();
        let n = client
            .upsert_vectors(&UpsertVectorsRequest {
                index: index.clone(),
                namespace: None,
                vectors,
            })
            .unwrap_or_else(|e| panic!("EVAL_UPSERT_ERROR={e}"));
        assert_eq!(n as usize, 1000.min(N_VECTORS - chunk_start));
    }
    println!(
        "EVAL_UPSERT_SECS={:.1}",
        upsert_start.elapsed().as_secs_f64()
    );

    // Serverless indexing is eventually consistent — wait until stats report
    // every vector before measuring anything.
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        let stats = client
            .describe_index_stats(&index)
            .unwrap_or_else(|e| panic!("EVAL_STATS_ERROR={e}"));
        if stats.total_vector_count as usize == N_VECTORS {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "EVAL_FRESHNESS_TIMEOUT count={}",
            stats.total_vector_count
        );
        std::thread::sleep(Duration::from_secs(3));
    }
    println!("EVAL_INDEXED_COUNT={N_VECTORS}");

    // ---- measurement 1: fetch_by_metadata, default (10k) page limit ----
    let filter = json!({BATCH_KEY: {"$eq": BATCH_VALUE}});
    let fbm = walk_fbm(&client, &index, filter.clone(), None);
    println!(
        "EVAL_FBM_10K pages={} records={} secs={:.2} pages_per_sec={:.2} records_per_sec={:.0}",
        fbm.pages,
        fbm.records.len(),
        fbm.secs,
        fbm.pages as f64 / fbm.secs,
        fbm.records.len() as f64 / fbm.secs,
    );
    assert_eq!(
        fbm.records.len(),
        N_VECTORS,
        "constant-key filter must return every record exactly once (dupes collapse in the map; \
         a shortfall means dropped records or broken pagination)"
    );
    assert_eq!(fbm.dupes, 0, "a record must never appear on two pages");
    for (id, v) in &fbm.records {
        assert_bit_exact(id, v);
    }
    println!("EVAL_FBM_10K_FIDELITY=bit-exact ({N_VECTORS} vectors x {DIMENSION} dims)");

    // ---- measurement 1b: fetch_by_metadata at limit=2000 (more pages, so
    // the documented 5 req/s/namespace cap becomes observable) ----
    let fbm2k = walk_fbm(&client, &index, filter.clone(), Some(2000));
    println!(
        "EVAL_FBM_2K pages={} records={} secs={:.2} pages_per_sec={:.2} records_per_sec={:.0}",
        fbm2k.pages,
        fbm2k.records.len(),
        fbm2k.secs,
        fbm2k.pages as f64 / fbm2k.secs,
        fbm2k.records.len() as f64 / fbm2k.secs,
    );
    assert_eq!(fbm2k.records.len(), N_VECTORS);
    assert_eq!(fbm2k.dupes, 0);

    // Filter-semantics negative probe: a value nothing was stamped with.
    let none = walk_fbm(
        &client,
        &index,
        json!({BATCH_KEY: {"$eq": "no-such-batch"}}),
        None,
    );
    assert_eq!(
        none.records.len(),
        0,
        "a non-matching constant-key filter must return nothing"
    );
    println!("EVAL_FBM_NEGATIVE_FILTER=empty (as expected)");

    // ---- measurement 2: the list+fetch path recover_from_remote uses ----
    let lf_start = Instant::now();
    let mut list_pages = 0usize;
    let mut fetch_calls = 0usize;
    let mut lf_records: HashMap<String, Vec<f32>> = HashMap::with_capacity(N_VECTORS);
    let mut token: Option<String> = None;
    loop {
        let page = client
            .list_vector_ids_page(&ListVectorIdsPageRequest {
                index: index.clone(),
                namespace: None,
                pagination_token: token.clone(),
            })
            .unwrap_or_else(|e| panic!("EVAL_LIST_ERROR={e}"));
        list_pages += 1;
        if !page.ids.is_empty() {
            // fetch_vectors chunks internally at 200 ids/call.
            fetch_calls += page.ids.len().div_ceil(200);
            let fetched = client
                .fetch_vectors(&FetchVectorsRequest {
                    index: index.clone(),
                    namespace: None,
                    ids: page.ids,
                })
                .unwrap_or_else(|e| panic!("EVAL_FETCH_ERROR={e}"));
            for (id, v) in fetched {
                lf_records.insert(id, v.values);
            }
        }
        match page.next_token {
            Some(next) => token = Some(next),
            None => break,
        }
    }
    let lf_secs = lf_start.elapsed().as_secs_f64();
    println!(
        "EVAL_LIST_FETCH list_pages={list_pages} fetch_calls={fetch_calls} records={} \
         secs={lf_secs:.2} records_per_sec={:.0}",
        lf_records.len(),
        lf_records.len() as f64 / lf_secs,
    );
    assert_eq!(lf_records.len(), N_VECTORS);
    for (id, v) in &lf_records {
        assert_bit_exact(id, v);
    }
    println!("EVAL_LIST_FETCH_FIDELITY=bit-exact");

    println!(
        "EVAL_SPEEDUP_VS_LIST_FETCH={:.1}x (10k-page walk)",
        lf_secs / fbm.secs
    );

    // ---- teardown: delete and VERIFY ----
    client
        .delete_index(&index)
        .unwrap_or_else(|e| panic!("EVAL_DELETE_ERROR={e}"));
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        match client.describe_index(&index) {
            Err(pg_pinecone_client::Error::IndexNotFound { .. }) => break,
            Ok(_) => {}
            Err(e) => panic!("EVAL_DELETE_VERIFY_ERROR={e}"),
        }
        assert!(
            Instant::now() < deadline,
            "EVAL_DELETE_VERIFY_TIMEOUT index={index} still describable"
        );
        std::thread::sleep(Duration::from_secs(2));
    }
    println!("EVAL_INDEX_DELETED=verified");
}

struct FbmWalk {
    pages: usize,
    records: HashMap<String, Vec<f32>>,
    dupes: usize,
    secs: f64,
}

/// Walks fetch_by_metadata pagination to exhaustion, retrying a page on 429
/// (the endpoint is documented at 5 req/s/namespace; a measurement harness
/// hitting the cap should wait it out, not die — retries count toward wall
/// time, which is the honest throughput number).
fn walk_fbm(
    client: &Client,
    index: &str,
    filter: serde_json::Value,
    limit: Option<u32>,
) -> FbmWalk {
    let start = Instant::now();
    let mut pages = 0usize;
    let mut dupes = 0usize;
    let mut records: HashMap<String, Vec<f32>> = HashMap::new();
    let mut token: Option<String> = None;
    loop {
        let page = match client.fetch_by_metadata(&FetchByMetadataRequest {
            index: index.to_string(),
            namespace: None,
            filter: filter.clone(),
            limit,
            pagination_token: token.clone(),
        }) {
            Ok(page) => page,
            Err(pg_pinecone_client::Error::Api { status: 429, .. }) => {
                std::thread::sleep(Duration::from_millis(500));
                continue;
            }
            Err(e) => panic!("EVAL_FBM_ERROR={e}"),
        };
        pages += 1;
        for (id, v) in page.vectors {
            if records.insert(id, v.values).is_some() {
                dupes += 1;
            }
        }
        match page.next_token {
            Some(next) => token = Some(next),
            None => break,
        }
        assert!(pages < 10_000, "runaway pagination");
    }
    FbmWalk {
        pages,
        records,
        dupes,
        secs: start.elapsed().as_secs_f64(),
    }
}
