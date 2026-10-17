//! Integration tests for `pg_pinecone-client`, driven against mock control-plane and
//! data-plane HTTP servers. `Client` is synchronous; tests call it inside
//! `tokio::task::spawn_blocking` so it can run alongside wiremock's async server.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use pg_pinecone_client::{
    BackupInfo, Client, Config, CreateBackupRequest, CreateDocIndexRequest, CreateIndexRequest,
    DeleteDocsRequest, DeleteVectorsRequest, DocDeleteSelector, DocIndexInfo, DocMatch,
    EmbedRequest, Error, FetchByMetadataPage, FetchByMetadataRequest, FetchVectorsRequest,
    IndexInfo, IndexStats, ListNamespacesPage, ListNamespacesRequest, ListVectorIdsPageRequest,
    ListVectorsRequest, Match, NamespaceInfo, NamespaceStats, QueryRequest, ReadCapacity,
    RerankRequest, RerankResult, SearchDocsRequest, UPSERT_MAX_BYTES, UPSERT_MAX_RECORDS,
    UpdateVectorMetadataRequest, UpsertDocsRequest, UpsertVectorsRequest, Vector,
};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{body_json, header, method, path, query_param, query_param_is_missing},
};

fn config(control_plane_url: impl Into<String>, api_key: impl Into<String>) -> Config {
    Config {
        control_plane_url: control_plane_url.into(),
        api_key: api_key.into(),
        timeout: Duration::from_secs(5),
        user_agent: None,
    }
}

fn query(index: impl Into<String>) -> QueryRequest {
    QueryRequest {
        index: index.into(),
        // Dyadic values are exact in both f32 and f64, so the serialized body
        // matches a `json!`-built expectation bit-for-bit.
        vector: vec![0.5, 0.25, 0.125],
        top_k: 5,
        namespace: None,
        filter: None,
        include_metadata: true,
        scan_factor: None,
        max_candidates: None,
    }
}

async fn query_blocking(client: Arc<Client>, req: QueryRequest) -> Result<Vec<Match>, Error> {
    tokio::task::spawn_blocking(move || client.query(&req))
        .await
        .expect("blocking task panicked")
}

/// Binds an ephemeral port then immediately releases it, yielding a scheme-less
/// `host:port` nothing is listening on (connects fail fast with connection-refused).
fn unreachable_addr() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    drop(listener);
    addr.to_string()
}

/// `send_json` emits `application/json; charset=utf-8`; accept any json content type.
fn content_type_is_json(request: &Request) -> bool {
    request
        .headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("application/json"))
}

fn assert_no_key_leak(err: &Error, api_key: &str) {
    assert!(
        !format!("{err}").contains(api_key),
        "api key leaked in Display: {err}"
    );
    assert!(
        !format!("{err:?}").contains(api_key),
        "api key leaked in Debug: {err:?}"
    );
}

#[tokio::test]
async fn resolves_host_then_queries_data_plane_with_required_headers() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(content_type_is_json)
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "matches": [
                {"id": "vec-1", "score": 0.95, "metadata": {"genre": "drama"}},
                {"id": "vec-2", "score": 0.42}
            ]
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let result = query_blocking(Arc::new(client), query("docs"))
        .await
        .unwrap();

    assert_eq!(
        result,
        vec![
            Match {
                id: "vec-1".to_string(),
                score: 0.95,
                metadata: Some(json!({"genre": "drama"})),
            },
            Match {
                id: "vec-2".to_string(),
                score: 0.42,
                metadata: None,
            },
        ]
    );
}

#[tokio::test]
async fn query_body_omits_namespace_and_filter_when_none() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .and(body_json(json!({
            "vector": [0.5, 0.25, 0.125],
            "topK": 5,
            "includeMetadata": true,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"matches": []})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = query_blocking(Arc::new(client), query("docs")).await;

    assert_eq!(result.unwrap(), Vec::<Match>::new());
}

#[tokio::test]
async fn query_body_includes_namespace_and_filter_when_provided() {
    let filter = json!({"genre": {"$eq": "documentary"}});

    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .and(body_json(json!({
            "vector": [0.5, 0.25, 0.125],
            "topK": 5,
            "includeMetadata": true,
            "namespace": "prod",
            "filter": filter.clone(),
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"matches": []})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let mut req = query("docs");
    req.namespace = Some("prod".to_string());
    req.filter = Some(filter);

    let client = Client::new(config(cp.uri(), "key"));
    let result = query_blocking(Arc::new(client), req).await;

    assert_eq!(result.unwrap(), Vec::<Match>::new());
}

#[tokio::test]
async fn query_body_includes_scan_factor_and_max_candidates_when_provided() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .and(body_json(json!({
            "vector": [0.5, 0.25, 0.125],
            "topK": 5,
            "includeMetadata": true,
            "scan_factor": 1.5,
            "max_candidates": 500,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"matches": []})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let mut req = query("docs");
    req.scan_factor = Some(1.5);
    req.max_candidates = Some(500);

    let client = Client::new(config(cp.uri(), "key"));
    let result = query_blocking(Arc::new(client), req).await;

    assert_eq!(result.unwrap(), Vec::<Match>::new());
}

#[tokio::test]
async fn query_body_omits_metadata_when_include_metadata_is_false() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .and(body_json(json!({
            "vector": [0.5, 0.25, 0.125],
            "topK": 5,
            "includeMetadata": false,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"matches": []})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let mut req = query("docs");
    req.include_metadata = false;

    let client = Client::new(config(cp.uri(), "key"));
    let result = query_blocking(Arc::new(client), req).await;

    assert_eq!(result.unwrap(), Vec::<Match>::new());
}

#[tokio::test]
async fn second_query_to_same_index_skips_control_plane() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"matches": []})))
        .expect(2)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Arc::new(Client::new(config(cp.uri(), "key")));
    query_blocking(Arc::clone(&client), query("docs"))
        .await
        .unwrap();
    query_blocking(Arc::clone(&client), query("docs"))
        .await
        .unwrap();
}

#[tokio::test]
async fn same_client_different_indexes_resolve_separately() {
    let dp_a = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"matches": []})))
        .expect(1)
        .mount(&dp_a)
        .await;

    let dp_b = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"matches": []})))
        .expect(1)
        .mount(&dp_b)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/idx-a"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp_a.uri()})))
        .expect(1)
        .mount(&cp)
        .await;
    Mock::given(method("GET"))
        .and(path("/indexes/idx-b"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp_b.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Arc::new(Client::new(config(cp.uri(), "key")));
    query_blocking(Arc::clone(&client), query("idx-a"))
        .await
        .unwrap();
    query_blocking(Arc::clone(&client), query("idx-b"))
        .await
        .unwrap();
}

#[tokio::test]
async fn separate_clients_do_not_share_host_cache() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"matches": []})))
        .expect(2)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(2)
        .mount(&cp)
        .await;

    let client_a = Arc::new(Client::new(config(cp.uri(), "key")));
    let client_b = Arc::new(Client::new(config(cp.uri(), "key")));

    query_blocking(client_a, query("docs")).await.unwrap();
    query_blocking(client_b, query("docs")).await.unwrap();
}

#[tokio::test]
async fn evicts_stale_host_and_reresolves_once_then_succeeds() {
    let dead_host = format!("http://{}", unreachable_addr());

    let live_dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"matches": []})))
        .expect(2)
        .mount(&live_dp)
        .await;
    let live_host = live_dp.uri();

    let cp = MockServer::start().await;
    let call_count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&call_count);
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(move |_req: &Request| {
            let host = if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                dead_host.clone()
            } else {
                live_host.clone()
            };
            ResponseTemplate::new(200).set_body_json(json!({"host": host}))
        })
        .expect(2)
        .mount(&cp)
        .await;

    let client = Arc::new(Client::new(config(cp.uri(), "key")));

    // First query: dead cached host -> evict -> re-resolve -> success.
    let result = query_blocking(Arc::clone(&client), query("docs")).await;
    assert_eq!(result.unwrap(), Vec::<Match>::new());

    // Second query must reuse the re-resolved host (cp stays at 2 calls).
    let result = query_blocking(Arc::clone(&client), query("docs")).await;
    assert_eq!(result.unwrap(), Vec::<Match>::new());
}

#[tokio::test]
async fn stops_after_one_reresolve_when_dataplane_stays_unreachable() {
    let api_key = "sk-transport-9f8e7d";
    let dead_host_1 = format!("http://{}", unreachable_addr());
    let dead_host_2 = format!("http://{}", unreachable_addr());

    let cp = MockServer::start().await;
    let call_count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&call_count);
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(move |_req: &Request| {
            let host = if counter.fetch_add(1, Ordering::SeqCst) == 0 {
                dead_host_1.clone()
            } else {
                dead_host_2.clone()
            };
            ResponseTemplate::new(200).set_body_json(json!({"host": host}))
        })
        .expect(2)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = query_blocking(Arc::new(client), query("docs")).await;

    let err = result.expect_err("expected transport error");
    assert!(matches!(err, Error::Transport { .. }), "got {err:?}");
    assert_no_key_leak(&err, api_key);
}

#[tokio::test]
async fn scheme_less_resolved_host_defaults_to_https() {
    let dead_host = unreachable_addr();

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dead_host})))
        .expect(2)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = query_blocking(Arc::new(client), query("docs")).await;

    match result {
        Err(Error::Transport { url, .. }) => {
            assert!(url.starts_with("https://"), "url = {url}");
        }
        other => panic!("expected Transport, got {other:?}"),
    }
}

#[tokio::test]
async fn control_plane_404_maps_to_index_not_found() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/ghost"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"code": "NOT_FOUND", "message": "index 'ghost' not found"},
            "status": 404
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = query_blocking(Arc::new(client), query("ghost")).await;

    match result {
        Err(Error::IndexNotFound { index }) => assert_eq!(index, "ghost"),
        other => panic!("expected IndexNotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn control_plane_401_maps_to_auth_error() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "bad-key"));
    let result = query_blocking(Arc::new(client), query("docs")).await;

    assert!(matches!(result, Err(Error::Auth)), "got {result:?}");
}

#[tokio::test]
async fn data_plane_400_maps_to_bad_request_with_message() {
    let api_key = "sk-badreq-5c4d3e";

    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3,
            "message": "Vector dimension 3 does not match the dimension of the index 8"
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = query_blocking(Arc::new(client), query("docs")).await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => {
            assert!(message.contains("Vector dimension 3 does not match"));
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

#[tokio::test]
async fn data_plane_500_maps_to_api_error_without_reresolve() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream connect error"))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = query_blocking(Arc::new(client), query("docs")).await;

    match result {
        Err(Error::Api { status, .. }) => assert_eq!(status, 500),
        other => panic!("expected Api, got {other:?}"),
    }
}

/// A namespace nothing was ever written to answers 404 "Namespace not found" on
/// `/query`. A partitioned table's tenant with no rows yet must read as "no
/// matches", not as an error. The 404 is a data-plane answer, not a transport
/// failure, so the host is neither evicted nor re-resolved.
#[tokio::test]
async fn query_404_namespace_not_found_is_an_empty_result() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .and(body_json(json!({
            "vector": [0.5, 0.25, 0.125],
            "topK": 5,
            "includeMetadata": true,
            "namespace": "public.docs:42",
        })))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "code": 5, "message": "Namespace not found"
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let mut req = query("docs");
    req.namespace = Some("public.docs:42".to_string());
    let client = Client::new(config(cp.uri(), "key"));
    let result = query_blocking(Arc::new(client), req).await;

    assert_eq!(result.unwrap(), Vec::<Match>::new());
}

/// Only 404 changed: every other non-200 `/query` status keeps its mapping.
#[tokio::test]
async fn query_other_error_statuses_keep_their_mapping() {
    for (status, is_auth) in [
        (401u16, true),
        (403, true),
        (409, false),
        (429, false),
        (503, false),
    ] {
        let dp = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/query"))
            .respond_with(ResponseTemplate::new(status).set_body_string("nope"))
            .expect(1)
            .mount(&dp)
            .await;
        let cp = MockServer::start().await;
        mount_host(&cp, "docs", &dp.uri()).await;

        let client = Client::new(config(cp.uri(), "key"));
        let result = query_blocking(Arc::new(client), query("docs")).await;

        match result {
            Err(Error::Auth) if is_auth => {}
            Err(Error::Api { status: got, .. }) if !is_auth => assert_eq!(got, status),
            other => panic!("status {status}: unexpected {other:?}"),
        }
    }
}

#[tokio::test]
async fn missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = query_blocking(Arc::new(client), query("docs")).await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn error_display_never_contains_the_api_key() {
    let api_key = "sk-super-secret-12345";

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"code": "NOT_FOUND", "message": "not found"},
            "status": 404
        })))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = query_blocking(Arc::new(client), query("docs")).await;

    let err = result.expect_err("expected an error");
    assert_no_key_leak(&err, api_key);
}

#[tokio::test]
async fn slow_control_plane_response_times_out() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"host": "http://127.0.0.1:1"}))
                .set_delay(Duration::from_secs(2)),
        )
        .mount(&cp)
        .await;

    let cfg = Config {
        timeout: Duration::from_millis(100),
        ..config(cp.uri(), "key")
    };
    let client = Client::new(cfg);

    let start = Instant::now();
    let result = query_blocking(Arc::new(client), query("docs")).await;
    let elapsed = start.elapsed();

    match result {
        Err(Error::Timeout { ms }) => assert_eq!(ms, 100),
        other => panic!("expected Timeout, got {other:?}"),
    }
    assert!(
        elapsed < Duration::from_secs(1),
        "elapsed={elapsed:?}, expected close to the 100ms configured timeout"
    );
}

// ---- Standard-index vector upsert (upsert_vectors) ----

async fn upsert_vectors_blocking(
    client: Arc<Client>,
    req: UpsertVectorsRequest,
) -> Result<u64, Error> {
    tokio::task::spawn_blocking(move || client.upsert_vectors(&req))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn upsert_vectors_posts_vectors_and_returns_server_count() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/upsert"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(content_type_is_json)
        .and(body_json(json!({
            "vectors": [
                {"id": "v1", "values": [0.5, 0.25, 0.125], "metadata": {"genre": "drama"}},
                {"id": "v2", "values": [0.75, 0.5, 0.25]}
            ],
            "namespace": "prod",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"upsertedCount": 2})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let count = upsert_vectors_blocking(
        Arc::new(client),
        UpsertVectorsRequest {
            index: "docs".to_string(),
            namespace: Some("prod".to_string()),
            vectors: vec![
                Vector {
                    id: "v1".to_string(),
                    values: vec![0.5, 0.25, 0.125],
                    metadata: Some(json!({"genre": "drama"})),
                },
                Vector {
                    id: "v2".to_string(),
                    values: vec![0.75, 0.5, 0.25],
                    metadata: None,
                },
            ],
        },
    )
    .await
    .unwrap();

    assert_eq!(count, 2);
}

#[tokio::test]
async fn upsert_vectors_omits_namespace_and_metadata_when_none() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/upsert"))
        .and(body_json(json!({
            "vectors": [{"id": "v1", "values": [0.5, 0.25, 0.125]}],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"upsertedCount": 1})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let count = upsert_vectors_blocking(
        Arc::new(client),
        UpsertVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
            vectors: vec![Vector {
                id: "v1".to_string(),
                values: vec![0.5, 0.25, 0.125],
                metadata: None,
            }],
        },
    )
    .await
    .unwrap();

    assert_eq!(count, 1);
}

#[tokio::test]
async fn upsert_vectors_empty_batch_is_a_noop_even_without_an_api_key() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let count = upsert_vectors_blocking(
        Arc::new(client),
        UpsertVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
            vectors: vec![],
        },
    )
    .await
    .unwrap();

    assert_eq!(count, 0);
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn upsert_vectors_400_maps_to_bad_request_without_key_leak() {
    let api_key = "sk-upsertvec-1a2b3c";

    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/upsert"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3,
            "message": "Vector dimension 3 does not match the dimension of the index 8"
        })))
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = upsert_vectors_blocking(
        Arc::new(client),
        UpsertVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
            vectors: vec![Vector {
                id: "v1".to_string(),
                values: vec![0.75, 0.5, 0.25],
                metadata: None,
            }],
        },
    )
    .await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(message.contains("does not match")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

// ---- upsert_vectors: request size ----

/// A vector of full-precision floats, what an embedding model returns; deterministic.
fn embedding(id: &str, dims: usize, seed: u32) -> Vector {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    let values = (0..dims)
        .map(|_| {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (x >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect();
    Vector {
        id: id.to_string(),
        values,
        metadata: None,
    }
}

/// `embedding` with metadata of exactly `bytes` bytes of JSON.
fn embedding_with_metadata(id: &str, dims: usize, bytes: usize) -> Vector {
    let mut vector = embedding(id, dims, 7);
    let text = "x".repeat(bytes - r#"{"m":""}"#.len());
    vector.metadata = Some(json!({ "m": text }));
    vector
}

/// Sends `req` to a mock data plane and returns the raw body it received.
async fn upsert_body(req: UpsertVectorsRequest) -> String {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/upsert"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"upsertedCount": 1})))
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, &req.index, &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    upsert_vectors_blocking(Arc::new(client), req)
        .await
        .unwrap();

    let received = dp.received_requests().await.unwrap();
    assert_eq!(received.len(), 1);
    String::from_utf8(received[0].body.clone()).expect("a JSON body is UTF-8")
}

/// A float goes out in the shortest text that reads back as the same f32: `0.1`, not the 17
/// digits of the f64 it widens to.
#[tokio::test]
async fn upsert_vectors_writes_each_float_in_its_shortest_f32_form() {
    let values = vec![0.1_f32, 0.2, 1.0 / 3.0, -0.012_345_679, 123_456.79, 0.0001];
    let body = upsert_body(UpsertVectorsRequest {
        index: "docs".to_string(),
        namespace: None,
        vectors: vec![Vector {
            id: "v1".to_string(),
            values: values.clone(),
            metadata: None,
        }],
    })
    .await;

    assert!(
        body.contains(r#""values":[0.1,0.2,0.33333334,-0.012345679,123456.79,0.0001]"#),
        "{body}"
    );
    let parsed: Value = serde_json::from_str(&body).unwrap();
    let read_back: Vec<f32> = parsed["vectors"][0]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_f64().unwrap() as f32)
        .collect();
    assert_eq!(read_back, values, "the same f32s");
}

/// The body is compact JSON: no indentation, one float per line was a third of every request.
#[tokio::test]
async fn upsert_vectors_sends_compact_json() {
    let body = upsert_body(UpsertVectorsRequest {
        index: "docs".to_string(),
        namespace: Some("prod".to_string()),
        vectors: vec![embedding_with_metadata("v1", 768, 400)],
    })
    .await;

    assert!(!body.contains('\n') && !body.contains("  "), "pretty JSON");
    assert!(
        body.len() < 10_500,
        "{} bytes for one 768-d vector with 400 bytes of metadata (the pretty f64 form took 22,900)",
        body.len()
    );
}

#[tokio::test]
async fn upsert_docs_sends_compact_json() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DOC_UPSERT_PATH))
        .respond_with(ResponseTemplate::new(202).set_body_json(json!({"upserted_count": 1})))
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let doc = json!({"_id": "d1", "body": "text", "vec": [0.5, 0.25]});
    let req = UpsertDocsRequest {
        index: "articles".to_string(),
        namespace: None,
        documents: vec![doc],
    };
    upsert_docs_blocking(Arc::new(client), req).await.unwrap();

    let received = dp.received_requests().await.unwrap();
    let body = String::from_utf8(received[0].body.clone()).unwrap();
    assert!(!body.contains('\n'), "pretty JSON: {body}");
}

/// A record weighs what Pinecone counts against a request's 2 MB: 4 bytes a value, the id and the
/// metadata as JSON, and a little framing. The budget never admits more records than Pinecone's
/// own table of the largest batch (docs.pinecone.io, "Upsert in batches") and stays within 10% of it.
#[test]
fn record_bytes_follows_pinecones_table_of_the_largest_batch() {
    assert_eq!(UPSERT_MAX_RECORDS, 1_000);
    assert_eq!(UPSERT_MAX_BYTES, 1_900_000);
    let fits = |vector: Vector| (UPSERT_MAX_BYTES / vector.record_bytes()).min(UPSERT_MAX_RECORDS);

    let (a, b, c) = (
        fits(embedding_with_metadata("v1", 768, 500)),
        fits(embedding_with_metadata("v1", 1536, 2_000)),
        fits(embedding("v1", 386, 1)),
    );

    assert!(
        (503..=559).contains(&a),
        "768 dims, 500 bytes: {a} (Pinecone: 559)"
    );
    assert!(
        (221..=245).contains(&b),
        "1536 dims, 2,000 bytes: {b} (Pinecone: 245)"
    );
    assert_eq!(
        c, 1_000,
        "386 dims: the 1,000 records bind (Pinecone: 1,000)"
    );
    assert_eq!(fits(embedding("v1", 768, 1)), 611, "768 dims, no metadata");
}

#[test]
fn record_bytes_counts_the_id_and_the_metadata_json() {
    let bare = embedding("v1", 8, 1);
    let long_id = embedding("v1234567890", 8, 1);
    let mut with_metadata = embedding("v1", 8, 1);
    with_metadata.metadata = Some(json!({"genre": "drama", "year": 2019}));
    let json_len = r#"{"genre":"drama","year":2019}"#.len();

    assert_eq!(long_id.record_bytes(), bare.record_bytes() + 9);
    assert_eq!(with_metadata.record_bytes(), bare.record_bytes() + json_len);
    assert_eq!(
        embedding("v1", 9, 1).record_bytes(),
        bare.record_bytes() + 4
    );
}

// ---- Inference (embed / rerank) ----
//
// Both post straight to the control plane's /embed and /rerank — no index-host
// resolution, so these mount their mock on `cp` only.

async fn embed_blocking(client: Arc<Client>, req: EmbedRequest) -> Result<Vec<Vec<f32>>, Error> {
    tokio::task::spawn_blocking(move || client.embed(&req))
        .await
        .expect("blocking task panicked")
}

async fn rerank_blocking(
    client: Arc<Client>,
    req: RerankRequest,
) -> Result<Vec<RerankResult>, Error> {
    tokio::task::spawn_blocking(move || client.rerank(&req))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn embed_posts_dense_request_and_returns_vectors_in_order() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(content_type_is_json)
        .and(body_json(json!({
            "model": "multilingual-e5-large",
            "parameters": {"input_type": "passage"},
            "inputs": [{"text": "apples are red"}, {"text": "bananas are yellow"}],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "multilingual-e5-large",
            "vector_type": "dense",
            "data": [
                {"values": [0.1, 0.2], "vector_type": "dense"},
                {"values": [0.3, 0.4], "vector_type": "dense"}
            ],
            "usage": {"total_tokens": 12}
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let result = embed_blocking(
        Arc::new(client),
        EmbedRequest {
            model: "multilingual-e5-large".to_string(),
            inputs: vec![
                "apples are red".to_string(),
                "bananas are yellow".to_string(),
            ],
            input_type: "passage".to_string(),
        },
    )
    .await
    .unwrap();

    assert_eq!(result, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
}

#[tokio::test]
async fn embed_empty_inputs_is_a_noop_even_without_an_api_key() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = embed_blocking(
        Arc::new(client),
        EmbedRequest {
            model: "multilingual-e5-large".to_string(),
            inputs: vec![],
            input_type: "query".to_string(),
        },
    )
    .await
    .unwrap();

    assert_eq!(result, Vec::<Vec<f32>>::new());
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn embed_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = embed_blocking(
        Arc::new(client),
        EmbedRequest {
            model: "multilingual-e5-large".to_string(),
            inputs: vec!["hi".to_string()],
            input_type: "query".to_string(),
        },
    )
    .await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn embed_chunks_at_96_inputs_per_call_and_concatenates_in_order() {
    let inputs: Vec<String> = (0..200).map(|i| format!("doc-{i}")).collect();

    let cp = MockServer::start().await;
    for (chunk_idx, chunk) in inputs.chunks(96).enumerate() {
        let expected_body = json!({
            "model": "llama-text-embed-v2",
            "parameters": {"input_type": "passage"},
            "inputs": chunk.iter().map(|t| json!({"text": t})).collect::<Vec<_>>(),
        });
        let data: Vec<serde_json::Value> = chunk
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let global_idx = (chunk_idx * 96 + i) as f32;
                json!({"values": [global_idx, global_idx * 2.0]})
            })
            .collect();
        Mock::given(method("POST"))
            .and(path("/embed"))
            .and(body_json(expected_body))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": data})))
            .expect(1)
            .mount(&cp)
            .await;
    }

    let client = Client::new(config(cp.uri(), "key"));
    let result = embed_blocking(
        Arc::new(client),
        EmbedRequest {
            model: "llama-text-embed-v2".to_string(),
            inputs,
            input_type: "passage".to_string(),
        },
    )
    .await
    .unwrap();

    assert_eq!(result.len(), 200);
    for (i, values) in result.iter().enumerate() {
        assert_eq!(values, &vec![i as f32, i as f32 * 2.0], "mismatch at {i}");
    }
}

#[tokio::test]
async fn embed_sparse_response_maps_to_bad_request() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "pinecone-sparse-english-v0",
            "vector_type": "sparse",
            "data": [
                {"sparse_values": [0.4, 0.3], "sparse_indices": [12, 287], "sparse_tokens": ["foo", "bar"], "vector_type": "sparse"}
            ],
            "usage": {"total_tokens": 5}
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = embed_blocking(
        Arc::new(client),
        EmbedRequest {
            model: "pinecone-sparse-english-v0".to_string(),
            inputs: vec!["hi".to_string()],
            input_type: "query".to_string(),
        },
    )
    .await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(
            message.contains("dense"),
            "expected message to mention dense-only support, got {message}"
        ),
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn embed_response_length_mismatch_maps_to_api_error() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{"values": [0.1, 0.2]}]
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = embed_blocking(
        Arc::new(client),
        EmbedRequest {
            model: "multilingual-e5-large".to_string(),
            inputs: vec!["one".to_string(), "two".to_string()],
            input_type: "query".to_string(),
        },
    )
    .await;

    assert!(matches!(result, Err(Error::Api { .. })), "got {result:?}");
}

#[tokio::test]
async fn embed_400_maps_to_bad_request_with_message() {
    let api_key = "sk-embed-1a2b3c";

    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3,
            "message": "model 'bogus' not found"
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = embed_blocking(
        Arc::new(client),
        EmbedRequest {
            model: "bogus".to_string(),
            inputs: vec!["hi".to_string()],
            input_type: "query".to_string(),
        },
    )
    .await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(message.contains("not found")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

#[tokio::test]
async fn embed_401_maps_to_auth_error() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "bad-key"));
    let result = embed_blocking(
        Arc::new(client),
        EmbedRequest {
            model: "multilingual-e5-large".to_string(),
            inputs: vec!["hi".to_string()],
            input_type: "query".to_string(),
        },
    )
    .await;

    assert!(matches!(result, Err(Error::Auth)), "got {result:?}");
}

#[tokio::test]
async fn embed_500_maps_to_api_error() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embed"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream exploded"))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let result = embed_blocking(
        Arc::new(client),
        EmbedRequest {
            model: "multilingual-e5-large".to_string(),
            inputs: vec!["hi".to_string()],
            input_type: "query".to_string(),
        },
    )
    .await;

    match result.expect_err("expected api error") {
        Error::Api { status, message } => {
            assert_eq!(status, 500);
            assert!(message.contains("upstream exploded"));
        }
        other => panic!("expected Api, got {other:?}"),
    }
}

#[tokio::test]
async fn rerank_posts_request_and_returns_results_in_response_order() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(content_type_is_json)
        .and(body_json(json!({
            "model": "bge-reranker-v2-m3",
            "query": "what is a condiment",
            "documents": [{"text": "ketchup"}, {"text": "mustard"}],
            "return_documents": false,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "bge-reranker-v2-m3",
            "data": [
                {"index": 1, "score": 0.93},
                {"index": 0, "score": 0.21}
            ],
            "usage": {"rerank_units": 1}
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let result = rerank_blocking(
        Arc::new(client),
        RerankRequest {
            model: "bge-reranker-v2-m3".to_string(),
            query: "what is a condiment".to_string(),
            documents: vec!["ketchup".to_string(), "mustard".to_string()],
            top_n: None,
        },
    )
    .await
    .unwrap();

    assert_eq!(
        result,
        vec![
            RerankResult {
                index: 1,
                score: 0.93,
            },
            RerankResult {
                index: 0,
                score: 0.21,
            },
        ]
    );
}

#[tokio::test]
async fn rerank_includes_top_n_when_provided() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .and(body_json(json!({
            "model": "bge-reranker-v2-m3",
            "query": "q",
            "documents": [{"text": "a"}],
            "return_documents": false,
            "top_n": 1,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": []})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = rerank_blocking(
        Arc::new(client),
        RerankRequest {
            model: "bge-reranker-v2-m3".to_string(),
            query: "q".to_string(),
            documents: vec!["a".to_string()],
            top_n: Some(1),
        },
    )
    .await;

    assert_eq!(result.unwrap(), Vec::<RerankResult>::new());
}

#[tokio::test]
async fn rerank_empty_documents_is_a_noop_even_without_an_api_key() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = rerank_blocking(
        Arc::new(client),
        RerankRequest {
            model: "bge-reranker-v2-m3".to_string(),
            query: "q".to_string(),
            documents: vec![],
            top_n: None,
        },
    )
    .await
    .unwrap();

    assert_eq!(result, Vec::<RerankResult>::new());
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn rerank_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = rerank_blocking(
        Arc::new(client),
        RerankRequest {
            model: "bge-reranker-v2-m3".to_string(),
            query: "q".to_string(),
            documents: vec!["a".to_string()],
            top_n: None,
        },
    )
    .await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn rerank_400_maps_to_bad_request_with_message() {
    let api_key = "sk-rerank-4a3b2c";

    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3,
            "message": "model 'bogus' not found"
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = rerank_blocking(
        Arc::new(client),
        RerankRequest {
            model: "bogus".to_string(),
            query: "q".to_string(),
            documents: vec!["a".to_string()],
            top_n: None,
        },
    )
    .await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(message.contains("not found")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

#[tokio::test]
async fn rerank_401_maps_to_auth_error() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "bad-key"));
    let result = rerank_blocking(
        Arc::new(client),
        RerankRequest {
            model: "bge-reranker-v2-m3".to_string(),
            query: "q".to_string(),
            documents: vec!["a".to_string()],
            top_n: None,
        },
    )
    .await;

    assert!(matches!(result, Err(Error::Auth)), "got {result:?}");
}

#[tokio::test]
async fn rerank_500_maps_to_api_error() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream exploded"))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let result = rerank_blocking(
        Arc::new(client),
        RerankRequest {
            model: "bge-reranker-v2-m3".to_string(),
            query: "q".to_string(),
            documents: vec!["a".to_string()],
            top_n: None,
        },
    )
    .await;

    match result.expect_err("expected api error") {
        Error::Api { status, message } => {
            assert_eq!(status, 500);
            assert!(message.contains("upstream exploded"));
        }
        other => panic!("expected Api, got {other:?}"),
    }
}

// ---- Index lifecycle (create_index / describe_index / wait_index_ready / delete_index) ----

fn create_index_request(name: impl Into<String>) -> CreateIndexRequest {
    CreateIndexRequest {
        name: name.into(),
        dimension: 1536,
        metric: "cosine".to_string(),
        spec: json!({"serverless": {"cloud": "aws", "region": "us-east-1"}}),
    }
}

async fn create_index_blocking(
    client: Arc<Client>,
    req: CreateIndexRequest,
) -> Result<IndexInfo, Error> {
    tokio::task::spawn_blocking(move || client.create_index(&req))
        .await
        .expect("blocking task panicked")
}

async fn describe_index_blocking(client: Arc<Client>, name: String) -> Result<IndexInfo, Error> {
    tokio::task::spawn_blocking(move || client.describe_index(&name))
        .await
        .expect("blocking task panicked")
}

async fn wait_index_ready_blocking(
    client: Arc<Client>,
    name: String,
    deadline: Duration,
    poll_every: Duration,
) -> Result<IndexInfo, Error> {
    tokio::task::spawn_blocking(move || client.wait_index_ready(&name, deadline, poll_every))
        .await
        .expect("blocking task panicked")
}

async fn delete_index_blocking(client: Arc<Client>, name: String) -> Result<(), Error> {
    tokio::task::spawn_blocking(move || client.delete_index(&name))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn create_index_posts_body_and_returns_normalized_info() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(body_json(json!({
            "name": "docs",
            "vector_type": "dense",
            "dimension": 1536,
            "metric": "cosine",
            "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
            "deletion_protection": "disabled",
        })))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({
            "name": "docs",
            "vector_type": "dense",
            "dimension": 1536,
            "metric": "cosine",
            "host": "docs-abcd1234.svc.pinecone.io",
            "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
            "status": {"ready": false, "state": "Initializing"}
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let info = create_index_blocking(Arc::new(client), create_index_request("docs"))
        .await
        .unwrap();

    assert_eq!(
        info,
        IndexInfo {
            host: "https://docs-abcd1234.svc.pinecone.io".to_string(),
            dimension: 1536,
            metric: "cosine".to_string(),
            vector_type: "dense".to_string(),
            ready: false,
            read_capacity: None,
        }
    );
}

#[tokio::test]
async fn create_index_409_maps_to_conflict() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes"))
        .respond_with(ResponseTemplate::new(409).set_body_json(json!({
            "error": {"code": "ALREADY_EXISTS", "message": "index 'docs' already exists"},
            "status": 409
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = create_index_blocking(Arc::new(client), create_index_request("docs")).await;

    match result {
        Err(Error::Conflict { name }) => assert_eq!(name, "docs"),
        other => panic!("expected Conflict, got {other:?}"),
    }
}

#[tokio::test]
async fn create_index_400_maps_to_bad_request_with_message() {
    let api_key = "sk-createidx-1a2b3c";

    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3,
            "message": "dimension must be between 1 and 20000"
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = create_index_blocking(Arc::new(client), create_index_request("docs")).await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(message.contains("dimension must be between")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

#[tokio::test]
async fn create_index_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = create_index_blocking(Arc::new(client), create_index_request("docs")).await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn describe_index_returns_info_with_vector_type_and_metric() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "docs",
            "vector_type": "dense",
            "dimension": 8,
            "metric": "euclidean",
            "host": "docs-abcd1234.svc.pinecone.io",
            "status": {"ready": true, "state": "Ready"}
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let info = describe_index_blocking(Arc::new(client), "docs".to_string())
        .await
        .unwrap();

    assert_eq!(
        info,
        IndexInfo {
            host: "https://docs-abcd1234.svc.pinecone.io".to_string(),
            dimension: 8,
            metric: "euclidean".to_string(),
            vector_type: "dense".to_string(),
            ready: true,
            read_capacity: None,
        }
    );
}

#[tokio::test]
async fn describe_index_reads_read_capacity_nested_under_spec_serverless() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "dimension": 8,
            "metric": "cosine",
            "host": "docs-abcd1234.svc.pinecone.io",
            "status": {"ready": true},
            "spec": {
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1",
                    "read_capacity": {
                        "mode": "Dedicated",
                        "dedicated": {
                            "node_type": "b1",
                            "scaling": "Manual",
                            "manual": {"shards": 2, "replicas": 1}
                        },
                        "status": {"state": "Ready"}
                    }
                }
            }
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let info = describe_index_blocking(Arc::new(client), "docs".to_string())
        .await
        .unwrap();

    assert_eq!(
        info.read_capacity,
        Some(ReadCapacity {
            mode: "Dedicated".to_string(),
            node_type: Some("b1".to_string()),
            shards: Some(2),
            replicas: Some(1),
            state: Some("Ready".to_string()),
        })
    );
}

#[tokio::test]
async fn describe_index_reads_read_capacity_as_top_level_sibling_of_spec() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "dimension": 8,
            "metric": "cosine",
            "host": "docs-abcd1234.svc.pinecone.io",
            "status": {"ready": true},
            "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
            "read_capacity": {"mode": "OnDemand"}
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let info = describe_index_blocking(Arc::new(client), "docs".to_string())
        .await
        .unwrap();

    assert_eq!(
        info.read_capacity,
        Some(ReadCapacity {
            mode: "OnDemand".to_string(),
            node_type: None,
            shards: None,
            replicas: None,
            state: None,
        })
    );
}

#[tokio::test]
async fn describe_index_read_capacity_is_none_when_absent() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "dimension": 8,
            "metric": "cosine",
            "host": "docs-abcd1234.svc.pinecone.io",
            "status": {"ready": true}
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let info = describe_index_blocking(Arc::new(client), "docs".to_string())
        .await
        .unwrap();

    assert_eq!(info.read_capacity, None);
}

#[tokio::test]
async fn describe_index_404_maps_to_index_not_found() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/ghost"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"code": "NOT_FOUND", "message": "index 'ghost' not found"},
            "status": 404
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = describe_index_blocking(Arc::new(client), "ghost".to_string()).await;

    match result {
        Err(Error::IndexNotFound { index }) => assert_eq!(index, "ghost"),
        other => panic!("expected IndexNotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn describe_index_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = describe_index_blocking(Arc::new(client), "docs".to_string()).await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

/// `describe_index` percent-encodes the index name into the path (shared
/// with `resolve_host`/`delete_index`/`delete_doc_index` via `index_segment`)
/// — a name containing characters that would otherwise redirect or break the
/// path must resolve against the encoded form.
#[tokio::test]
async fn describe_index_percent_encodes_a_name_containing_slash_and_question_mark() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/weird%2Fname%3Fx"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "weird/name?x",
            "vector_type": "dense",
            "dimension": 4,
            "metric": "cosine",
            "host": "weird-abcd1234.svc.pinecone.io",
            "status": {"ready": true, "state": "Ready"}
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let info = describe_index_blocking(Arc::new(client), "weird/name?x".to_string())
        .await
        .unwrap();

    assert_eq!(info.dimension, 4);
}

#[tokio::test]
async fn wait_index_ready_polls_until_ready() {
    let cp = MockServer::start().await;
    let call_count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&call_count);
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(move |_req: &Request| {
            let ready = counter.fetch_add(1, Ordering::SeqCst) >= 2;
            ResponseTemplate::new(200).set_body_json(json!({
                "name": "docs",
                "vector_type": "dense",
                "dimension": 8,
                "metric": "cosine",
                "host": "docs-abcd1234.svc.pinecone.io",
                "status": {"ready": ready, "state": if ready { "Ready" } else { "Initializing" }}
            }))
        })
        .expect(3)
        .mount(&cp)
        .await;

    // No-op sleep: the poll interval doesn't matter, only that polling continues.
    let client = Client::with_sleep(config(cp.uri(), "key"), |_| {});
    let result = wait_index_ready_blocking(
        Arc::new(client),
        "docs".to_string(),
        Duration::from_secs(5),
        Duration::from_millis(10),
    )
    .await;

    assert!(result.unwrap().ready);
}

#[tokio::test]
async fn wait_index_ready_times_out_when_never_ready() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "docs",
            "vector_type": "dense",
            "dimension": 8,
            "metric": "cosine",
            "host": "docs-abcd1234.svc.pinecone.io",
            "status": {"ready": false, "state": "Initializing"}
        })))
        .mount(&cp)
        .await;

    let client = Client::with_sleep(config(cp.uri(), "key"), |_| {});
    let result = wait_index_ready_blocking(
        Arc::new(client),
        "docs".to_string(),
        Duration::from_millis(0),
        Duration::from_millis(10),
    )
    .await;

    match result {
        Err(Error::Timeout { ms }) => assert_eq!(ms, 0),
        other => panic!("expected Timeout, got {other:?}"),
    }
}

async fn wait_doc_index_ready_blocking(
    client: Arc<Client>,
    name: String,
    deadline: Duration,
    poll_every: Duration,
) -> Result<IndexInfo, Error> {
    tokio::task::spawn_blocking(move || client.wait_doc_index_ready(&name, deadline, poll_every))
        .await
        .expect("blocking task panicked")
}

/// [`Client::wait_doc_index_ready`] shares its polling loop with
/// [`Client::wait_index_ready`] (already covered by
/// `wait_index_ready_polls_until_ready`/`_times_out_when_never_ready`); this
/// only needs to prove the one thing that differs — the `2026-07` header.
#[tokio::test]
async fn wait_doc_index_ready_polls_with_2026_07_header_until_ready() {
    let cp = MockServer::start().await;
    let call_count = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&call_count);
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .respond_with(move |_req: &Request| {
            let ready = counter.fetch_add(1, Ordering::SeqCst) >= 1;
            ResponseTemplate::new(200).set_body_json(json!({
                "name": "articles",
                "host": "articles-abcd1234.svc.pinecone.io",
                "status": {"ready": ready, "state": if ready { "Ready" } else { "Initializing" }}
            }))
        })
        .expect(2)
        .mount(&cp)
        .await;

    let client = Client::with_sleep(config(cp.uri(), "key"), |_| {});
    let result = wait_doc_index_ready_blocking(
        Arc::new(client),
        "articles".to_string(),
        Duration::from_secs(5),
        Duration::from_millis(10),
    )
    .await;

    assert!(result.unwrap().ready);
}

#[tokio::test]
async fn delete_index_202_is_ok() {
    let cp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/indexes/docs"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .respond_with(ResponseTemplate::new(202))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    delete_index_blocking(Arc::new(client), "docs".to_string())
        .await
        .unwrap();
}

#[tokio::test]
async fn delete_index_404_maps_to_index_not_found() {
    let cp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/indexes/ghost"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"code": "NOT_FOUND", "message": "index 'ghost' not found"},
            "status": 404
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = delete_index_blocking(Arc::new(client), "ghost".to_string()).await;

    match result {
        Err(Error::IndexNotFound { index }) => assert_eq!(index, "ghost"),
        other => panic!("expected IndexNotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn delete_index_401_maps_to_auth_error() {
    let cp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = delete_index_blocking(Arc::new(client), "docs".to_string()).await;

    assert!(matches!(result, Err(Error::Auth)), "got {result:?}");
}

#[tokio::test]
async fn delete_index_403_maps_to_bad_request_naming_deletion_protection() {
    let cp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({
            "error": {"code": "FORBIDDEN", "message": "deletion_protection is enabled"},
            "status": 403
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = delete_index_blocking(Arc::new(client), "docs".to_string()).await;

    match result {
        Err(Error::BadRequest { message }) => {
            assert!(message.contains("docs"), "message = {message}");
            assert!(
                message.contains("deletion_protection"),
                "message = {message}"
            );
            assert!(
                message.contains("deletion_protection is enabled"),
                "message = {message}"
            );
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn delete_index_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = delete_index_blocking(Arc::new(client), "docs".to_string()).await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

// ---- Vector delete (delete_vectors) ----

async fn delete_vectors_blocking(
    client: Arc<Client>,
    req: DeleteVectorsRequest,
) -> Result<(), Error> {
    tokio::task::spawn_blocking(move || client.delete_vectors(&req))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn delete_vectors_with_ids_includes_namespace() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/delete"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(body_json(json!({
            "ids": ["id-1", "id-2"],
            "namespace": "prod",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    delete_vectors_blocking(
        Arc::new(client),
        DeleteVectorsRequest {
            index: "docs".to_string(),
            namespace: Some("prod".to_string()),
            ids: Some(vec!["id-1".to_string(), "id-2".to_string()]),
            delete_all: false,
            filter: None,
        },
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn delete_vectors_with_delete_all_omits_namespace_when_none() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/delete"))
        .and(body_json(json!({ "deleteAll": true })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    delete_vectors_blocking(
        Arc::new(client),
        DeleteVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
            ids: None,
            delete_all: true,
            filter: None,
        },
    )
    .await
    .unwrap();
}

/// Pinecone answers 404 "Namespace not found" when the namespace has never
/// been written to — the first promote() of any table hits exactly this on its
/// rebuild deleteAll. Deletes are idempotent, so 404 must be success.
#[tokio::test]
async fn delete_vectors_404_namespace_not_found_is_ok() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/delete"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "code": 5, "message": "Namespace not found"
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    delete_vectors_blocking(
        Arc::new(client),
        DeleteVectorsRequest {
            index: "docs".to_string(),
            namespace: Some("never-written".to_string()),
            ids: None,
            delete_all: true,
            filter: None,
        },
    )
    .await
    .expect("deleteAll on a never-written namespace must be an idempotent no-op");
}

#[tokio::test]
async fn delete_vectors_both_ids_and_delete_all_is_bad_request_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = delete_vectors_blocking(
        Arc::new(client),
        DeleteVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
            ids: Some(vec!["id-1".to_string()]),
            delete_all: true,
            filter: None,
        },
    )
    .await;

    assert!(
        matches!(result, Err(Error::BadRequest { .. })),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn delete_vectors_neither_ids_nor_delete_all_is_bad_request_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = delete_vectors_blocking(
        Arc::new(client),
        DeleteVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
            ids: None,
            delete_all: false,
            filter: None,
        },
    )
    .await;

    assert!(
        matches!(result, Err(Error::BadRequest { .. })),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn delete_vectors_400_maps_to_bad_request_without_key_leak() {
    let api_key = "sk-deletevec-9z8y7x";

    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/delete"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3,
            "message": "cannot specify both ids and deleteAll"
        })))
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = delete_vectors_blocking(
        Arc::new(client),
        DeleteVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
            ids: Some(vec!["id-1".to_string()]),
            delete_all: false,
            filter: None,
        },
    )
    .await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(message.contains("cannot specify both")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

// ---- list_vector_ids (`/vectors/list`, paginated) ----

async fn list_vector_ids_blocking(
    client: Arc<Client>,
    req: ListVectorsRequest,
) -> Result<Vec<String>, Error> {
    tokio::task::spawn_blocking(move || client.list_vector_ids(&req))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn list_vector_ids_single_page_returns_ids_in_response_order() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/list"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(query_param("limit", "100"))
        .and(query_param("namespace", "prod"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "vectors": [{"id": "1:1"}, {"id": "1:2"}],
            "namespace": "prod",
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let ids = list_vector_ids_blocking(
        Arc::new(client),
        ListVectorsRequest {
            index: "docs".to_string(),
            namespace: Some("prod".to_string()),
        },
    )
    .await
    .unwrap();
    assert_eq!(ids, vec!["1:1".to_string(), "1:2".to_string()]);
}

#[tokio::test]
async fn list_vector_ids_omits_namespace_query_param_when_none() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/list"))
        .and(query_param_is_missing("namespace"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"vectors": []})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let ids = list_vector_ids_blocking(
        Arc::new(client),
        ListVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
        },
    )
    .await
    .unwrap();
    assert!(ids.is_empty());
}

#[tokio::test]
async fn list_vector_ids_follows_pagination_token_across_pages() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/list"))
        .and(query_param_is_missing("paginationToken"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "vectors": [{"id": "1:1"}],
            "pagination": {"next": "page-2-token"},
        })))
        .expect(1)
        .mount(&dp)
        .await;
    Mock::given(method("GET"))
        .and(path("/vectors/list"))
        .and(query_param("paginationToken", "page-2-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "vectors": [{"id": "1:2"}],
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let ids = list_vector_ids_blocking(
        Arc::new(client),
        ListVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(ids, vec!["1:1".to_string(), "1:2".to_string()]);
}

/// `namespace`/`paginationToken` must be encoded with a QUERY-component
/// escape set, not the path-segment one — `&`/`=`/`+` are meaningful in a
/// query string (parameter/value delimiters, and `+` decodes to a literal
/// space) but not in a path
/// segment, so reusing the path escape set there let any of the three
/// corrupt the query string instead of round-tripping byte-for-byte.
/// `query_param` asserts against the *decoded* value wiremock's own URL
/// parser recovers from the wire, so this only passes if the client escaped
/// all three; before the fix, the raw `&`/`=` would have split the fixture's
/// pagination token into extra, wrongly-named query parameters instead.
#[tokio::test]
async fn list_vector_ids_percent_encodes_reserved_query_characters_in_namespace_and_token() {
    let namespace = "ns+with/plus=and&amp";
    let token = "tok+en/with=plus&amp==";

    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/list"))
        .and(query_param("namespace", namespace))
        .and(query_param_is_missing("paginationToken"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "vectors": [{"id": "1:1"}],
            "pagination": {"next": token},
        })))
        .expect(1)
        .mount(&dp)
        .await;
    Mock::given(method("GET"))
        .and(path("/vectors/list"))
        .and(query_param("namespace", namespace))
        .and(query_param("paginationToken", token))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"vectors": [{"id": "1:2"}]})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let ids = list_vector_ids_blocking(
        Arc::new(client),
        ListVectorsRequest {
            index: "docs".to_string(),
            namespace: Some(namespace.to_string()),
        },
    )
    .await
    .unwrap();
    assert_eq!(ids, vec!["1:1".to_string(), "1:2".to_string()]);
}

/// A namespace that was never written to (or no longer exists) is an
/// idempotent empty result, not an error — same convention as
/// `delete_vectors`'s own 404 handling.
#[tokio::test]
async fn list_vector_ids_404_is_empty() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/list"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "code": 5, "message": "Namespace not found"
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let ids = list_vector_ids_blocking(
        Arc::new(client),
        ListVectorsRequest {
            index: "docs".to_string(),
            namespace: Some("never-written".to_string()),
        },
    )
    .await
    .unwrap();
    assert!(ids.is_empty());
}

#[tokio::test]
async fn list_vector_ids_400_maps_to_bad_request_with_message() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/list"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3, "message": "bad pagination token"
        })))
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = list_vector_ids_blocking(
        Arc::new(client),
        ListVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
        },
    )
    .await;
    match result {
        Err(Error::BadRequest { message }) => assert!(message.contains("bad pagination token")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn list_vector_ids_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;
    let client = Client::new(config(cp.uri(), ""));
    let result = list_vector_ids_blocking(
        Arc::new(client),
        ListVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
        },
    )
    .await;
    assert!(matches!(result, Err(Error::MissingApiKey)), "{result:?}");
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

// ---- upsert_vectors retry on 429/5xx ----

fn one_vector_upsert(index: impl Into<String>) -> UpsertVectorsRequest {
    UpsertVectorsRequest {
        index: index.into(),
        namespace: None,
        vectors: vec![Vector {
            id: "v1".to_string(),
            values: vec![0.5, 0.25, 0.125],
            metadata: None,
        }],
    }
}

/// Mounts a data-plane `/vectors/upsert` mock that responds `first_status` (with
/// optional `Retry-After`) for the first `fail_times` calls, then 200 with
/// `upsertedCount: 1`.
async fn mount_flaky_upsert(
    dp: &MockServer,
    first_status: u16,
    fail_times: usize,
    retry_after: Option<&'static str>,
) {
    let call_count = Arc::new(AtomicUsize::new(0));
    Mock::given(method("POST"))
        .and(path("/vectors/upsert"))
        .respond_with(move |_req: &Request| {
            let n = call_count.fetch_add(1, Ordering::SeqCst);
            if n < fail_times {
                let mut resp = ResponseTemplate::new(first_status);
                if let Some(ra) = retry_after {
                    resp = resp.insert_header("Retry-After", ra);
                }
                resp
            } else {
                ResponseTemplate::new(200).set_body_json(json!({"upsertedCount": 1}))
            }
        })
        .expect((fail_times + 1) as u64)
        .mount(dp)
        .await;
}

async fn mount_host(cp: &MockServer, index: &str, dp_uri: &str) {
    Mock::given(method("GET"))
        .and(path(format!("/indexes/{index}")))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp_uri})))
        .mount(cp)
        .await;
}

#[tokio::test]
async fn upsert_vectors_retries_429_then_succeeds() {
    let dp = MockServer::start().await;
    mount_flaky_upsert(&dp, 429, 1, None).await;

    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::with_sleep(config(cp.uri(), "key"), |_| {});
    let count = upsert_vectors_blocking(Arc::new(client), one_vector_upsert("docs"))
        .await
        .unwrap();

    assert_eq!(count, 1);
    assert_eq!(dp.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn upsert_vectors_retries_503_then_succeeds() {
    let dp = MockServer::start().await;
    mount_flaky_upsert(&dp, 503, 1, None).await;

    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::with_sleep(config(cp.uri(), "key"), |_| {});
    let count = upsert_vectors_blocking(Arc::new(client), one_vector_upsert("docs"))
        .await
        .unwrap();

    assert_eq!(count, 1);
    assert_eq!(dp.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn upsert_vectors_uses_default_backoff_when_no_retry_after_header() {
    let dp = MockServer::start().await;
    mount_flaky_upsert(&dp, 429, 1, None).await;

    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let sleeps = Arc::new(Mutex::new(Vec::<Duration>::new()));
    let sleeps_clone = Arc::clone(&sleeps);
    let client = Client::with_sleep(config(cp.uri(), "key"), move |d| {
        sleeps_clone.lock().unwrap().push(d);
    });
    upsert_vectors_blocking(Arc::new(client), one_vector_upsert("docs"))
        .await
        .unwrap();

    assert_eq!(*sleeps.lock().unwrap(), vec![Duration::from_millis(500)]);
}

#[tokio::test]
async fn upsert_vectors_honors_retry_after_capped_at_10s() {
    let dp = MockServer::start().await;
    mount_flaky_upsert(&dp, 429, 1, Some("999")).await;

    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let sleeps = Arc::new(Mutex::new(Vec::<Duration>::new()));
    let sleeps_clone = Arc::clone(&sleeps);
    let client = Client::with_sleep(config(cp.uri(), "key"), move |d| {
        sleeps_clone.lock().unwrap().push(d);
    });
    let count = upsert_vectors_blocking(Arc::new(client), one_vector_upsert("docs"))
        .await
        .unwrap();

    assert_eq!(count, 1);
    assert_eq!(*sleeps.lock().unwrap(), vec![Duration::from_secs(10)]);
}

#[tokio::test]
async fn upsert_vectors_exhausts_retries_and_returns_mapped_error() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/upsert"))
        .respond_with(ResponseTemplate::new(500).set_body_string("upstream connect error"))
        .expect(3)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::with_sleep(config(cp.uri(), "key"), |_| {});
    let result = upsert_vectors_blocking(Arc::new(client), one_vector_upsert("docs")).await;

    match result {
        Err(Error::Api { status, .. }) => assert_eq!(status, 500),
        other => panic!("expected Api, got {other:?}"),
    }
    assert_eq!(dp.received_requests().await.unwrap().len(), 3);
}

// ---- Document-schema indexes (create_doc_index / delete_doc_index) ----

fn create_doc_index_request(name: impl Into<String>) -> CreateDocIndexRequest {
    CreateDocIndexRequest {
        name: name.into(),
        schema: json!({
            "fields": {"title": {"type": "string", "full_text_search": {"language": "en"}}}
        }),
        cloud: "aws".to_string(),
        region: "us-east-1".to_string(),
    }
}

async fn create_doc_index_blocking(
    client: Arc<Client>,
    req: CreateDocIndexRequest,
) -> Result<(), Error> {
    tokio::task::spawn_blocking(move || client.create_doc_index(&req))
        .await
        .expect("blocking task panicked")
}

async fn delete_doc_index_blocking(client: Arc<Client>, name: String) -> Result<(), Error> {
    tokio::task::spawn_blocking(move || client.delete_doc_index(&name))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn create_doc_index_posts_schema_body_with_2026_07_header() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .and(body_json(json!({
            "name": "articles",
            "deployment": {"deployment_type": "managed", "cloud": "aws", "region": "us-east-1"},
            "schema": {
                "fields": {"title": {"type": "string", "full_text_search": {"language": "en"}}}
            },
            "deletion_protection": "disabled",
        })))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({
            "name": "articles",
            "host": "articles-abcd1234.svc.pinecone.io",
            "schema": {"fields": {}},
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    create_doc_index_blocking(Arc::new(client), create_doc_index_request("articles"))
        .await
        .unwrap();
}

#[tokio::test]
async fn create_doc_index_409_maps_to_conflict() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes"))
        .respond_with(ResponseTemplate::new(409).set_body_json(json!({
            "error": {"code": "ALREADY_EXISTS", "message": "index 'articles' already exists"},
            "status": 409
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result =
        create_doc_index_blocking(Arc::new(client), create_doc_index_request("articles")).await;

    match result {
        Err(Error::Conflict { name }) => assert_eq!(name, "articles"),
        other => panic!("expected Conflict, got {other:?}"),
    }
}

#[tokio::test]
async fn create_doc_index_400_maps_to_bad_request_with_message() {
    let api_key = "sk-createdocidx-1a2b3c";

    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3,
            "message": "plain filterable fields are not allowed in schema"
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result =
        create_doc_index_blocking(Arc::new(client), create_doc_index_request("articles")).await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(message.contains("plain filterable fields")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

#[tokio::test]
async fn create_doc_index_401_maps_to_auth_error() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "bad-key"));
    let result =
        create_doc_index_blocking(Arc::new(client), create_doc_index_request("articles")).await;

    assert!(matches!(result, Err(Error::Auth)), "got {result:?}");
}

#[tokio::test]
async fn create_doc_index_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result =
        create_doc_index_blocking(Arc::new(client), create_doc_index_request("articles")).await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn delete_doc_index_202_is_ok_with_2026_07_header() {
    let cp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/indexes/articles"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .respond_with(ResponseTemplate::new(202))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    delete_doc_index_blocking(Arc::new(client), "articles".to_string())
        .await
        .unwrap();
}

#[tokio::test]
async fn delete_doc_index_404_maps_to_index_not_found() {
    let cp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/indexes/ghost"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"code": "NOT_FOUND", "message": "index 'ghost' not found"},
            "status": 404
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = delete_doc_index_blocking(Arc::new(client), "ghost".to_string()).await;

    match result {
        Err(Error::IndexNotFound { index }) => assert_eq!(index, "ghost"),
        other => panic!("expected IndexNotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn delete_doc_index_401_maps_to_auth_error() {
    let cp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/indexes/articles"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = delete_doc_index_blocking(Arc::new(client), "articles".to_string()).await;

    assert!(matches!(result, Err(Error::Auth)), "got {result:?}");
}

#[tokio::test]
async fn delete_doc_index_403_maps_to_bad_request_naming_deletion_protection() {
    let cp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/indexes/articles"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({
            "error": {"code": "FORBIDDEN", "message": "deletion_protection is enabled"},
            "status": 403
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = delete_doc_index_blocking(Arc::new(client), "articles".to_string()).await;

    match result {
        Err(Error::BadRequest { message }) => {
            assert!(message.contains("articles"), "message = {message}");
            assert!(
                message.contains("deletion_protection"),
                "message = {message}"
            );
            assert!(
                message.contains("deletion_protection is enabled"),
                "message = {message}"
            );
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn delete_doc_index_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = delete_doc_index_blocking(Arc::new(client), "articles".to_string()).await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

// ---- upsert_docs ----

fn one_doc_upsert(index: impl Into<String>) -> UpsertDocsRequest {
    UpsertDocsRequest {
        index: index.into(),
        namespace: None,
        documents: vec![json!({"_id": "doc1", "title": "hello"})],
    }
}

async fn upsert_docs_blocking(client: Arc<Client>, req: UpsertDocsRequest) -> Result<u64, Error> {
    tokio::task::spawn_blocking(move || client.upsert_docs(&req))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn upsert_docs_resolves_host_with_2026_07_header_then_posts_documents() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/namespaces/__default__/documents/upsert"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .and(content_type_is_json)
        .and(body_json(json!({
            "documents": [{"_id": "doc1", "title": "hello"}],
        })))
        .respond_with(ResponseTemplate::new(202).set_body_json(json!({"upserted_count": 1})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let count = upsert_docs_blocking(Arc::new(client), one_doc_upsert("articles"))
        .await
        .unwrap();

    assert_eq!(count, 1);
}

#[tokio::test]
async fn upsert_docs_percent_encodes_namespace_path_segment() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/namespaces/team%20space/documents/upsert"))
        .respond_with(ResponseTemplate::new(202).set_body_json(json!({"upserted_count": 1})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let mut req = one_doc_upsert("articles");
    req.namespace = Some("team space".to_string());
    let count = upsert_docs_blocking(Arc::new(client), req).await.unwrap();

    assert_eq!(count, 1);
}

#[tokio::test]
async fn upsert_docs_empty_batch_is_a_noop_even_without_an_api_key() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let count = upsert_docs_blocking(
        Arc::new(client),
        UpsertDocsRequest {
            index: "articles".to_string(),
            namespace: None,
            documents: vec![],
        },
    )
    .await
    .unwrap();

    assert_eq!(count, 0);
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn upsert_docs_chunks_at_1000_docs_per_call_and_sums_counts() {
    let documents: Vec<Value> = (0..1500)
        .map(|i| json!({"_id": format!("doc{i}"), "title": format!("title {i}")}))
        .collect();

    let dp = MockServer::start().await;
    for chunk in documents.chunks(1000) {
        let expected_body = json!({ "documents": chunk });
        let count = chunk.len() as u64;
        Mock::given(method("POST"))
            .and(path("/namespaces/__default__/documents/upsert"))
            .and(body_json(expected_body))
            .respond_with(
                ResponseTemplate::new(202).set_body_json(json!({"upserted_count": count})),
            )
            .expect(1)
            .mount(&dp)
            .await;
    }

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let count = upsert_docs_blocking(
        Arc::new(client),
        UpsertDocsRequest {
            index: "articles".to_string(),
            namespace: None,
            documents,
        },
    )
    .await
    .unwrap();

    assert_eq!(count, 1500);
    assert_eq!(dp.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn upsert_docs_byte_budget_forces_a_split_below_1000_docs() {
    // Each document serializes to a bit over 200,000 bytes; 10 of them sum to
    // ~2MB, well past the ~1.5MB byte budget but far short of the 1000-doc
    // cap — the byte budget, not the doc-count cap, must force this split.
    let documents: Vec<Value> = (0..10)
        .map(|i| json!({"_id": format!("doc{i}"), "title": "a".repeat(200_000)}))
        .collect();

    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/namespaces/__default__/documents/upsert"))
        .respond_with(|req: &Request| {
            let body: Value = req.body_json().expect("request body is JSON");
            let count = body["documents"]
                .as_array()
                .expect("documents is an array")
                .len() as u64;
            ResponseTemplate::new(202).set_body_json(json!({"upserted_count": count}))
        })
        .expect(2)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let count = upsert_docs_blocking(
        Arc::new(client),
        UpsertDocsRequest {
            index: "articles".to_string(),
            namespace: None,
            documents,
        },
    )
    .await
    .unwrap();

    assert_eq!(count, 10);
    assert_eq!(dp.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn upsert_docs_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = upsert_docs_blocking(Arc::new(client), one_doc_upsert("articles")).await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn upsert_docs_400_maps_to_bad_request_without_key_leak() {
    let api_key = "sk-upsertdocs-9z8y7x";

    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/namespaces/__default__/documents/upsert"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3,
            "message": "document doc1 exceeds max field size"
        })))
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = upsert_docs_blocking(Arc::new(client), one_doc_upsert("articles")).await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(message.contains("exceeds max field size")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

// ---- doc-endpoint retry on 429/5xx (upsert_docs, delete_docs) ----

const DOC_UPSERT_PATH: &str = "/namespaces/__default__/documents/upsert";

/// Answers call `n` with `plan[n]` = (status, `Retry-After`); 2xx carries `ok_body`.
async fn mount_scripted_post(
    dp: &MockServer,
    endpoint: &str,
    plan: Vec<(u16, Option<&'static str>)>,
    ok_body: Value,
) {
    let calls = Arc::new(AtomicUsize::new(0));
    let expected = plan.len() as u64;
    Mock::given(method("POST"))
        .and(path(endpoint))
        .respond_with(move |_req: &Request| {
            let n = calls.fetch_add(1, Ordering::SeqCst);
            let (status, retry_after) = plan.get(n).copied().unwrap_or((500, None));
            let mut resp = ResponseTemplate::new(status);
            if let Some(ra) = retry_after {
                resp = resp.insert_header("Retry-After", ra);
            }
            if status < 300 {
                resp = resp.set_body_json(ok_body.clone());
            }
            resp
        })
        .expect(expected)
        .mount(dp)
        .await;
}

fn client_recording_sleeps(cp: &MockServer) -> (Arc<Client>, Arc<Mutex<Vec<Duration>>>) {
    let sleeps = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&sleeps);
    let client = Client::with_sleep(config(cp.uri(), "key"), move |d| {
        recorder.lock().unwrap().push(d);
    });
    (Arc::new(client), sleeps)
}

#[tokio::test]
async fn upsert_docs_retries_429_then_succeeds() {
    let dp = MockServer::start().await;
    let plan = vec![(429, None), (202, None)];
    mount_scripted_post(&dp, DOC_UPSERT_PATH, plan, json!({"upserted_count": 1})).await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let (client, _) = client_recording_sleeps(&cp);
    let count = upsert_docs_blocking(client, one_doc_upsert("articles"))
        .await
        .unwrap();

    assert_eq!(count, 1);
    assert_eq!(dp.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn upsert_docs_returns_api_error_after_three_503s() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DOC_UPSERT_PATH))
        .respond_with(ResponseTemplate::new(503).set_body_string("service unavailable"))
        .expect(3)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let (client, _) = client_recording_sleeps(&cp);
    let result = upsert_docs_blocking(client, one_doc_upsert("articles")).await;

    match result {
        Err(Error::Api { status, message }) => {
            assert_eq!(status, 503);
            assert!(
                message.contains("service unavailable"),
                "message = {message}"
            );
        }
        other => panic!("expected Api, got {other:?}"),
    }
    assert_eq!(dp.received_requests().await.unwrap().len(), 3);
}

#[tokio::test]
async fn upsert_docs_honors_retry_after_and_caps_it_at_10s() {
    let dp = MockServer::start().await;
    let plan = vec![(429, Some("3")), (503, Some("999")), (202, None)];
    mount_scripted_post(&dp, DOC_UPSERT_PATH, plan, json!({"upserted_count": 1})).await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let (client, sleeps) = client_recording_sleeps(&cp);
    let count = upsert_docs_blocking(client, one_doc_upsert("articles"))
        .await
        .unwrap();

    assert_eq!(count, 1);
    assert_eq!(
        *sleeps.lock().unwrap(),
        vec![Duration::from_secs(3), Duration::from_secs(10)]
    );
}

#[tokio::test]
async fn upsert_docs_backs_off_500ms_then_2s_without_retry_after() {
    let dp = MockServer::start().await;
    let plan = vec![(500, None), (429, None), (202, None)];
    mount_scripted_post(&dp, DOC_UPSERT_PATH, plan, json!({"upserted_count": 1})).await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let (client, sleeps) = client_recording_sleeps(&cp);
    upsert_docs_blocking(client, one_doc_upsert("articles"))
        .await
        .unwrap();

    assert_eq!(
        *sleeps.lock().unwrap(),
        vec![Duration::from_millis(500), Duration::from_secs(2)]
    );
}

#[tokio::test]
async fn upsert_docs_does_not_retry_a_400() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DOC_UPSERT_PATH))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({"message": "bad document"})))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let (client, sleeps) = client_recording_sleeps(&cp);
    let result = upsert_docs_blocking(client, one_doc_upsert("articles")).await;

    assert!(
        matches!(result, Err(Error::BadRequest { .. })),
        "got {result:?}"
    );
    assert!(sleeps.lock().unwrap().is_empty());
}

// ---- search_docs ----

fn search_docs_request(index: impl Into<String>) -> SearchDocsRequest {
    SearchDocsRequest {
        index: index.into(),
        namespace: None,
        include_fields: vec!["title".to_string(), "body".to_string()],
        filter: None,
        score_by: json!([{"type": "text", "fields": ["body"], "query": "machine learning"}]),
        top_k: 10,
    }
}

async fn search_docs_blocking(
    client: Arc<Client>,
    req: SearchDocsRequest,
) -> Result<Vec<DocMatch>, Error> {
    tokio::task::spawn_blocking(move || client.search_docs(&req))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn search_docs_resolves_host_with_2026_07_header_then_splits_id_and_score_from_doc() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/namespaces/__default__/documents/search"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .and(body_json(json!({
            "include_fields": ["title", "body"],
            "score_by": [{"type": "text", "fields": ["body"], "query": "machine learning"}],
            "top_k": 10,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "matches": [
                {"_id": "doc-1", "_score": 0.928, "title": "ML basics", "body": "..."},
                {"_id": "doc-2", "_score": 0.4, "title": "Other"}
            ],
            "namespace": "__default__",
            "usage": {"read_units": 5}
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let result = search_docs_blocking(Arc::new(client), search_docs_request("articles"))
        .await
        .unwrap();

    assert_eq!(
        result,
        vec![
            DocMatch {
                id: "doc-1".to_string(),
                score: 0.928,
                doc: json!({"title": "ML basics", "body": "..."}),
            },
            DocMatch {
                id: "doc-2".to_string(),
                score: 0.4,
                doc: json!({"title": "Other"}),
            },
        ]
    );
}

#[tokio::test]
async fn search_docs_includes_namespace_path_and_filter_when_provided() {
    let filter = json!({"category": {"$eq": "technology"}, "year": {"$gte": 2024}});

    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/namespaces/prod/documents/search"))
        .and(body_json(json!({
            "include_fields": ["title", "body"],
            "filter": filter.clone(),
            "score_by": [{"type": "text", "fields": ["body"], "query": "machine learning"}],
            "top_k": 10,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"matches": []})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let mut req = search_docs_request("articles");
    req.namespace = Some("prod".to_string());
    req.filter = Some(filter);

    let client = Client::new(config(cp.uri(), "key"));
    let result = search_docs_blocking(Arc::new(client), req).await;

    assert_eq!(result.unwrap(), Vec::<DocMatch>::new());
}

#[tokio::test]
async fn search_docs_missing_matches_is_empty() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/namespaces/__default__/documents/search"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"usage": {"read_units": 1}})))
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = search_docs_blocking(Arc::new(client), search_docs_request("articles")).await;

    assert_eq!(result.unwrap(), Vec::<DocMatch>::new());
}

/// Same never-written-namespace rule as `/query`: a document search on a
/// namespace that holds nothing yet is "no matches", not an error.
#[tokio::test]
async fn search_docs_404_namespace_not_found_is_an_empty_result() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/namespaces/public.articles:42/documents/search"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "code": 5, "message": "Namespace not found"
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let mut req = search_docs_request("articles");
    req.namespace = Some("public.articles:42".to_string());
    let client = Client::new(config(cp.uri(), "key"));
    let result = search_docs_blocking(Arc::new(client), req).await;

    assert_eq!(result.unwrap(), Vec::<DocMatch>::new());
}

/// Only 404 changed: every other non-200 search status keeps its mapping.
#[tokio::test]
async fn search_docs_other_error_statuses_keep_their_mapping() {
    for (status, is_auth) in [(401u16, true), (403, true), (409, false), (500, false)] {
        let dp = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/namespaces/__default__/documents/search"))
            .respond_with(ResponseTemplate::new(status).set_body_string("nope"))
            .expect(1)
            .mount(&dp)
            .await;
        let cp = MockServer::start().await;
        mount_host(&cp, "articles", &dp.uri()).await;

        let client = Client::new(config(cp.uri(), "key"));
        let result = search_docs_blocking(Arc::new(client), search_docs_request("articles")).await;

        match result {
            Err(Error::Auth) if is_auth => {}
            Err(Error::Api { status: got, .. }) if !is_auth => assert_eq!(got, status),
            other => panic!("status {status}: unexpected {other:?}"),
        }
    }
}

#[tokio::test]
async fn search_docs_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = search_docs_blocking(Arc::new(client), search_docs_request("articles")).await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn search_docs_400_maps_to_bad_request_without_key_leak() {
    let api_key = "sk-searchdocs-4a3b2c";

    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/namespaces/__default__/documents/search"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3,
            "message": "include_fields must be non-empty"
        })))
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = search_docs_blocking(Arc::new(client), search_docs_request("articles")).await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(message.contains("include_fields")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

// ---- delete_docs ----

const DOC_DELETE_PATH: &str = "/namespaces/__default__/documents/delete";

fn delete_docs_request(index: impl Into<String>, selector: DocDeleteSelector) -> DeleteDocsRequest {
    DeleteDocsRequest {
        index: index.into(),
        namespace: None,
        selector,
    }
}

fn doc_ids(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("doc{i}")).collect()
}

async fn delete_docs_blocking(client: Arc<Client>, req: DeleteDocsRequest) -> Result<(), Error> {
    tokio::task::spawn_blocking(move || client.delete_docs(&req))
        .await
        .expect("blocking task panicked")
}

async fn mount_delete_accepting_all(dp: &MockServer) {
    Mock::given(method("POST"))
        .and(path(DOC_DELETE_PATH))
        .respond_with(ResponseTemplate::new(202).set_body_json(json!({})))
        .mount(dp)
        .await;
}

fn ids_sent_per_request(requests: &[Request]) -> Vec<Vec<String>> {
    requests
        .iter()
        .map(|r| {
            let body: Value = r.body_json().expect("request body is JSON");
            serde_json::from_value(body["ids"].clone()).expect("ids is a string array")
        })
        .collect()
}

#[tokio::test]
async fn delete_docs_ids_resolves_host_with_2026_07_header_then_posts_ids() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DOC_DELETE_PATH))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .and(content_type_is_json)
        .and(body_json(json!({"ids": ["doc1", "doc2"]})))
        .respond_with(ResponseTemplate::new(202).set_body_json(json!({})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let selector = DocDeleteSelector::Ids(vec!["doc1".to_string(), "doc2".to_string()]);
    delete_docs_blocking(Arc::new(client), delete_docs_request("articles", selector))
        .await
        .unwrap();
}

#[tokio::test]
async fn delete_docs_all_posts_delete_all_true() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DOC_DELETE_PATH))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .and(body_json(json!({"delete_all": true})))
        .respond_with(ResponseTemplate::new(202).set_body_json(json!({})))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    delete_docs_blocking(
        Arc::new(client),
        delete_docs_request("articles", DocDeleteSelector::All),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn delete_docs_200_is_ok() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DOC_DELETE_PATH))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    delete_docs_blocking(
        Arc::new(client),
        delete_docs_request("articles", DocDeleteSelector::Ids(doc_ids(1))),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn delete_docs_percent_encodes_namespace_path_segment() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/namespaces/team%20space/documents/delete"))
        .respond_with(ResponseTemplate::new(202).set_body_json(json!({})))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let mut req = delete_docs_request("articles", DocDeleteSelector::All);
    req.namespace = Some("team space".to_string());
    delete_docs_blocking(Arc::new(client), req).await.unwrap();
}

#[tokio::test]
async fn delete_docs_1000_ids_is_a_single_request() {
    let dp = MockServer::start().await;
    mount_delete_accepting_all(&dp).await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let ids = doc_ids(1000);
    let client = Client::new(config(cp.uri(), "key"));
    let req = delete_docs_request("articles", DocDeleteSelector::Ids(ids.clone()));
    delete_docs_blocking(Arc::new(client), req).await.unwrap();

    let sent = ids_sent_per_request(&dp.received_requests().await.unwrap());
    assert_eq!(sent, vec![ids]);
}

#[tokio::test]
async fn delete_docs_1001_ids_splits_into_1000_then_1_in_order() {
    let dp = MockServer::start().await;
    mount_delete_accepting_all(&dp).await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let ids = doc_ids(1001);
    let client = Client::new(config(cp.uri(), "key"));
    let req = delete_docs_request("articles", DocDeleteSelector::Ids(ids.clone()));
    delete_docs_blocking(Arc::new(client), req).await.unwrap();

    let sent = ids_sent_per_request(&dp.received_requests().await.unwrap());
    assert_eq!(sent, vec![ids[..1000].to_vec(), ids[1000..].to_vec()]);
}

#[tokio::test]
async fn delete_docs_stops_at_the_first_failing_chunk() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DOC_DELETE_PATH))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({"message": "bad id"})))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let req = delete_docs_request("articles", DocDeleteSelector::Ids(doc_ids(1001)));
    let result = delete_docs_blocking(Arc::new(client), req).await;

    assert!(
        matches!(result, Err(Error::BadRequest { .. })),
        "got {result:?}"
    );
    assert_eq!(dp.received_requests().await.unwrap().len(), 1);
}

#[tokio::test]
async fn delete_docs_empty_ids_is_a_noop_even_without_an_api_key() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let req = delete_docs_request("articles", DocDeleteSelector::Ids(vec![]));
    delete_docs_blocking(Arc::new(client), req).await.unwrap();

    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

// 404-as-Ok mirrors delete_vectors; not yet observed live on document indexes.
#[tokio::test]
async fn delete_docs_404_namespace_not_found_is_ok() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/namespaces/never-written/documents/delete"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "code": 5, "message": "Namespace not found"
        })))
        .expect(2)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let client = Arc::new(Client::new(config(cp.uri(), "key")));
    for selector in [DocDeleteSelector::All, DocDeleteSelector::Ids(doc_ids(2))] {
        let mut req = delete_docs_request("articles", selector);
        req.namespace = Some("never-written".to_string());
        delete_docs_blocking(Arc::clone(&client), req)
            .await
            .expect("delete on a never-written namespace is an idempotent no-op");
    }
}

#[tokio::test]
async fn delete_docs_unknown_index_is_index_not_found_not_ok() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/ghost"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"code": "NOT_FOUND", "message": "index 'ghost' not found"},
            "status": 404
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = delete_docs_blocking(
        Arc::new(client),
        delete_docs_request("ghost", DocDeleteSelector::All),
    )
    .await;

    match result {
        Err(Error::IndexNotFound { index }) => assert_eq!(index, "ghost"),
        other => panic!("expected IndexNotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn delete_docs_400_and_422_map_to_bad_request_without_key_leak() {
    let api_key = "sk-deletedocs-4d3c2b";
    for status in [400, 422] {
        let dp = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(DOC_DELETE_PATH))
            .respond_with(ResponseTemplate::new(status).set_body_json(json!({
                "code": 3,
                "message": "ids must contain between 1 and 1000 entries"
            })))
            .expect(1)
            .mount(&dp)
            .await;
        let cp = MockServer::start().await;
        mount_host(&cp, "articles", &dp.uri()).await;

        let client = Client::new(config(cp.uri(), api_key));
        let req = delete_docs_request("articles", DocDeleteSelector::Ids(doc_ids(1)));
        let err = delete_docs_blocking(Arc::new(client), req)
            .await
            .expect_err("expected bad-request error");

        match &err {
            Error::BadRequest { message } => {
                assert!(
                    message.contains("between 1 and 1000"),
                    "message = {message}"
                );
            }
            other => panic!("status {status}: expected BadRequest, got {other:?}"),
        }
        assert_no_key_leak(&err, api_key);
    }
}

#[tokio::test]
async fn delete_docs_401_and_403_map_to_auth_error() {
    for status in [401, 403] {
        let dp = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path(DOC_DELETE_PATH))
            .respond_with(ResponseTemplate::new(status))
            .expect(1)
            .mount(&dp)
            .await;
        let cp = MockServer::start().await;
        mount_host(&cp, "articles", &dp.uri()).await;

        let client = Client::new(config(cp.uri(), "bad-key"));
        let result = delete_docs_blocking(
            Arc::new(client),
            delete_docs_request("articles", DocDeleteSelector::All),
        )
        .await;

        assert!(
            matches!(result, Err(Error::Auth)),
            "status {status}: got {result:?}"
        );
    }
}

#[tokio::test]
async fn delete_docs_unexpected_status_maps_to_api_error_without_retry() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DOC_DELETE_PATH))
        .respond_with(ResponseTemplate::new(409).set_body_string("conflicting delete"))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = delete_docs_blocking(
        Arc::new(client),
        delete_docs_request("articles", DocDeleteSelector::All),
    )
    .await;

    match result {
        Err(Error::Api { status, message }) => {
            assert_eq!(status, 409);
            assert!(
                message.contains("conflicting delete"),
                "message = {message}"
            );
        }
        other => panic!("expected Api, got {other:?}"),
    }
}

#[tokio::test]
async fn delete_docs_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = delete_docs_blocking(
        Arc::new(client),
        delete_docs_request("articles", DocDeleteSelector::Ids(doc_ids(1))),
    )
    .await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn delete_docs_retries_429_then_succeeds() {
    let dp = MockServer::start().await;
    let plan = vec![(429, None), (202, None)];
    mount_scripted_post(&dp, DOC_DELETE_PATH, plan, json!({})).await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let (client, _) = client_recording_sleeps(&cp);
    let req = delete_docs_request("articles", DocDeleteSelector::Ids(doc_ids(2)));
    delete_docs_blocking(client, req).await.unwrap();

    assert_eq!(dp.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn delete_docs_returns_api_error_after_three_503s() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DOC_DELETE_PATH))
        .respond_with(ResponseTemplate::new(503).set_body_string("service unavailable"))
        .expect(3)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let (client, _) = client_recording_sleeps(&cp);
    let result = delete_docs_blocking(
        client,
        delete_docs_request("articles", DocDeleteSelector::All),
    )
    .await;

    match result {
        Err(Error::Api { status, message }) => {
            assert_eq!(status, 503);
            assert!(
                message.contains("service unavailable"),
                "message = {message}"
            );
        }
        other => panic!("expected Api, got {other:?}"),
    }
    assert_eq!(dp.received_requests().await.unwrap().len(), 3);
}

#[tokio::test]
async fn delete_docs_429_exhaustion_maps_to_api_429() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path(DOC_DELETE_PATH))
        .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
        .expect(3)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let (client, _) = client_recording_sleeps(&cp);
    let result = delete_docs_blocking(
        client,
        delete_docs_request("articles", DocDeleteSelector::All),
    )
    .await;

    assert!(
        matches!(result, Err(Error::Api { status: 429, .. })),
        "got {result:?}"
    );
}

#[tokio::test]
async fn delete_docs_honors_retry_after_and_caps_it_at_10s() {
    let dp = MockServer::start().await;
    let plan = vec![(429, Some("3")), (503, Some("999")), (202, None)];
    mount_scripted_post(&dp, DOC_DELETE_PATH, plan, json!({})).await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let (client, sleeps) = client_recording_sleeps(&cp);
    delete_docs_blocking(
        client,
        delete_docs_request("articles", DocDeleteSelector::All),
    )
    .await
    .unwrap();

    assert_eq!(
        *sleeps.lock().unwrap(),
        vec![Duration::from_secs(3), Duration::from_secs(10)]
    );
}

#[tokio::test]
async fn delete_docs_backs_off_500ms_then_2s_without_retry_after() {
    let dp = MockServer::start().await;
    let plan = vec![(500, None), (429, None), (202, None)];
    mount_scripted_post(&dp, DOC_DELETE_PATH, plan, json!({})).await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let (client, sleeps) = client_recording_sleeps(&cp);
    delete_docs_blocking(
        client,
        delete_docs_request("articles", DocDeleteSelector::All),
    )
    .await
    .unwrap();

    assert_eq!(
        *sleeps.lock().unwrap(),
        vec![Duration::from_millis(500), Duration::from_secs(2)]
    );
}

// ---- describe_doc_index ----

async fn describe_doc_index_blocking(
    client: Arc<Client>,
    name: String,
) -> Result<DocIndexInfo, Error> {
    tokio::task::spawn_blocking(move || client.describe_doc_index(&name))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn describe_doc_index_sends_2026_07_header_and_maps_the_description() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "articles",
            "host": "articles-abcd1234.svc.pinecone.io",
            "status": {"ready": true, "state": "Ready"},
            "schema": {"fields": {
                "title": {"type": "string", "full_text_search": {"language": "en"}},
                "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}
            }},
            "deletion_protection": "disabled",
            "read_capacity": {"mode": "OnDemand"},
            "tags": {"env": "prod"},
            "a_field_from_the_future": {"x": 1}
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let info = describe_doc_index_blocking(Arc::new(client), "articles".to_string())
        .await
        .unwrap();

    assert_eq!(
        info,
        DocIndexInfo {
            host: "https://articles-abcd1234.svc.pinecone.io".to_string(),
            ready: true,
            schema: json!({"fields": {
                "title": {"type": "string", "full_text_search": {"language": "en"}},
                "embedding": {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}
            }}),
            deletion_protection: Some("disabled".to_string()),
            read_capacity: Some(json!({"mode": "OnDemand"})),
        }
    );
}

#[tokio::test]
async fn describe_doc_index_defaults_missing_optional_fields() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({"host": "articles-abcd1234.svc.pinecone.io"})),
        )
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let info = describe_doc_index_blocking(Arc::new(client), "articles".to_string())
        .await
        .unwrap();

    assert_eq!(
        info,
        DocIndexInfo {
            host: "https://articles-abcd1234.svc.pinecone.io".to_string(),
            ready: false,
            schema: Value::Null,
            deletion_protection: None,
            read_capacity: None,
        }
    );
}

#[tokio::test]
async fn describe_doc_index_reports_an_initializing_index_as_not_ready() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "host": "http://127.0.0.1:5555",
            "status": {"ready": false, "state": "Initializing"},
            "schema": {"fields": {}}
        })))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let info = describe_doc_index_blocking(Arc::new(client), "articles".to_string())
        .await
        .unwrap();

    assert!(!info.ready);
    assert_eq!(info.host, "http://127.0.0.1:5555");
}

#[tokio::test]
async fn describe_doc_index_404_maps_to_index_not_found() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/ghost"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"code": "NOT_FOUND", "message": "index 'ghost' not found"},
            "status": 404
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = describe_doc_index_blocking(Arc::new(client), "ghost".to_string()).await;

    match result {
        Err(Error::IndexNotFound { index }) => assert_eq!(index, "ghost"),
        other => panic!("expected IndexNotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn describe_doc_index_401_and_403_map_to_auth_error() {
    for status in [401, 403] {
        let cp = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/indexes/articles"))
            .respond_with(ResponseTemplate::new(status))
            .expect(1)
            .mount(&cp)
            .await;

        let client = Client::new(config(cp.uri(), "bad-key"));
        let result = describe_doc_index_blocking(Arc::new(client), "articles".to_string()).await;

        assert!(
            matches!(result, Err(Error::Auth)),
            "status {status}: got {result:?}"
        );
    }
}

#[tokio::test]
async fn describe_doc_index_500_maps_to_api_error() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/articles"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = describe_doc_index_blocking(Arc::new(client), "articles".to_string()).await;

    match result {
        Err(Error::Api { status, message }) => {
            assert_eq!(status, 500);
            assert_eq!(message, "boom");
        }
        other => panic!("expected Api, got {other:?}"),
    }
}

#[tokio::test]
async fn describe_doc_index_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = describe_doc_index_blocking(Arc::new(client), "articles".to_string()).await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

// ---- fetch_vectors ----

async fn fetch_vectors_blocking(
    client: Arc<Client>,
    req: FetchVectorsRequest,
) -> Result<std::collections::HashMap<String, Vector>, Error> {
    tokio::task::spawn_blocking(move || client.fetch_vectors(&req))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn fetch_vectors_resolves_host_then_gets_with_repeated_ids_and_namespace() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/fetch"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(query_param("ids", "v1"))
        .and(query_param("ids", "v2"))
        .and(query_param("namespace", "prod"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "vectors": {
                "v1": {"id": "v1", "values": [0.1, 0.2]},
                "v2": {"id": "v2", "values": [0.3, 0.4], "metadata": {"genre": "drama"}}
            },
            "namespace": "prod",
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let result = fetch_vectors_blocking(
        Arc::new(client),
        FetchVectorsRequest {
            index: "docs".to_string(),
            namespace: Some("prod".to_string()),
            ids: vec!["v1".to_string(), "v2".to_string()],
        },
    )
    .await
    .unwrap();

    assert_eq!(result.len(), 2);
    assert_eq!(result["v1"].values, vec![0.1, 0.2]);
    assert_eq!(result["v1"].metadata, None);
    assert_eq!(result["v2"].values, vec![0.3, 0.4]);
    assert_eq!(result["v2"].metadata, Some(json!({"genre": "drama"})));
}

/// An id Pinecone has no vector for is simply absent from the returned map —
/// not an error. Covers both the "missing-id" and "partial-response" cases.
#[tokio::test]
async fn fetch_vectors_missing_id_is_simply_absent_from_the_map() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/fetch"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "vectors": { "v1": {"id": "v1", "values": [0.1]} },
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = fetch_vectors_blocking(
        Arc::new(client),
        FetchVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
            ids: vec!["v1".to_string(), "ghost".to_string()],
        },
    )
    .await
    .unwrap();

    assert_eq!(result.len(), 1);
    assert!(result.contains_key("v1"));
    assert!(
        !result.contains_key("ghost"),
        "an id Pinecone has no vector for must be absent, not an error"
    );
}

/// Pinecone's hard cap is 1000 ids/call; this client chunks at 200 (response-size
/// safety margin) — 250 requested ids must split across two calls and merge.
#[tokio::test]
async fn fetch_vectors_chunks_at_200_ids_per_call_and_merges_results() {
    let ids: Vec<String> = (0..250).map(|i| format!("id-{i}")).collect();

    let dp = MockServer::start().await;
    for (chunk_idx, chunk) in ids.chunks(200).enumerate() {
        let mut vectors_obj = serde_json::Map::new();
        for id in chunk {
            vectors_obj.insert(id.clone(), json!({"id": id, "values": [chunk_idx as f32]}));
        }
        let mut matcher = Mock::given(method("GET")).and(path("/vectors/fetch"));
        for id in chunk {
            matcher = matcher.and(query_param("ids", id.as_str()));
        }
        matcher
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "vectors": Value::Object(vectors_obj),
            })))
            .expect(1)
            .mount(&dp)
            .await;
    }

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = fetch_vectors_blocking(
        Arc::new(client),
        FetchVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
            ids: ids.clone(),
        },
    )
    .await
    .unwrap();

    assert_eq!(result.len(), 250);
    for id in &ids {
        assert!(result.contains_key(id), "missing {id}");
    }
}

#[tokio::test]
async fn fetch_vectors_empty_ids_is_a_noop_even_without_an_api_key() {
    let cp = MockServer::start().await;
    let client = Client::new(config(cp.uri(), ""));
    let result = fetch_vectors_blocking(
        Arc::new(client),
        FetchVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
            ids: vec![],
        },
    )
    .await
    .unwrap();
    assert!(result.is_empty());
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn fetch_vectors_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;
    let client = Client::new(config(cp.uri(), ""));
    let result = fetch_vectors_blocking(
        Arc::new(client),
        FetchVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
            ids: vec!["v1".to_string()],
        },
    )
    .await;
    assert!(matches!(result, Err(Error::MissingApiKey)), "{result:?}");
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn fetch_vectors_400_maps_to_bad_request_without_key_leak() {
    let api_key = "sk-fetchvec-9f8e7d";

    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/fetch"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3, "message": "ids must not be empty"
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = fetch_vectors_blocking(
        Arc::new(client),
        FetchVectorsRequest {
            index: "docs".to_string(),
            namespace: None,
            ids: vec!["v1".to_string()],
        },
    )
    .await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(message.contains("ids must not be empty")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

/// A namespace with nothing fetchable (never written to) is an idempotent
/// empty map — same convention as `list_vector_ids`/`delete_vectors`'s own
/// 404 handling.
#[tokio::test]
async fn fetch_vectors_404_is_empty_map() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/fetch"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "code": 5, "message": "Namespace not found"
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = fetch_vectors_blocking(
        Arc::new(client),
        FetchVectorsRequest {
            index: "docs".to_string(),
            namespace: Some("never-written".to_string()),
            ids: vec!["v1".to_string()],
        },
    )
    .await
    .unwrap();
    assert!(result.is_empty());
}

// ---- describe_index_stats ----

async fn describe_index_stats_blocking(
    client: Arc<Client>,
    index: String,
) -> Result<IndexStats, Error> {
    tokio::task::spawn_blocking(move || client.describe_index_stats(&index))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn describe_index_stats_resolves_host_then_posts_with_2026_04_header() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/describe_index_stats"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(content_type_is_json)
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "totalVectorCount": 42,
            "namespaces": {"prod": {"vectorCount": 30}, "": {"vectorCount": 12}},
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let stats = describe_index_stats_blocking(Arc::new(client), "docs".to_string())
        .await
        .unwrap();

    assert_eq!(stats.total_vector_count, 42);
    assert_eq!(
        stats.namespaces.get("prod"),
        Some(&NamespaceStats { vector_count: 30 })
    );
    assert_eq!(
        stats.namespaces.get(""),
        Some(&NamespaceStats { vector_count: 12 })
    );
}

#[tokio::test]
async fn describe_index_stats_400_maps_to_bad_request_with_message() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/describe_index_stats"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3, "message": "malformed filter"
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = describe_index_stats_blocking(Arc::new(client), "docs".to_string()).await;
    match result {
        Err(Error::BadRequest { message }) => assert!(message.contains("malformed filter")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn describe_index_stats_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;
    let client = Client::new(config(cp.uri(), ""));
    let result = describe_index_stats_blocking(Arc::new(client), "docs".to_string()).await;
    assert!(matches!(result, Err(Error::MissingApiKey)), "{result:?}");
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

// ---- list_vector_ids_page (the resumable-cursor primitive for recover_from_remote) ----

async fn list_vector_ids_page_blocking(
    client: Arc<Client>,
    req: ListVectorIdsPageRequest,
) -> Result<pg_pinecone_client::ListVectorIdsPage, Error> {
    tokio::task::spawn_blocking(move || client.list_vector_ids_page(&req))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn list_vector_ids_page_first_page_returns_ids_and_next_token() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/list"))
        .and(query_param_is_missing("paginationToken"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "vectors": [{"id": "1:1"}, {"id": "1:2"}],
            "pagination": {"next": "page-2-token"},
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let page = list_vector_ids_page_blocking(
        Arc::new(client),
        ListVectorIdsPageRequest {
            index: "docs".to_string(),
            namespace: None,
            pagination_token: None,
        },
    )
    .await
    .unwrap();

    assert_eq!(page.ids, vec!["1:1".to_string(), "1:2".to_string()]);
    assert_eq!(page.next_token.as_deref(), Some("page-2-token"));
}

#[tokio::test]
async fn list_vector_ids_page_follows_a_persisted_token_and_reports_exhaustion() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/list"))
        .and(query_param("paginationToken", "page-2-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "vectors": [{"id": "1:3"}],
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let page = list_vector_ids_page_blocking(
        Arc::new(client),
        ListVectorIdsPageRequest {
            index: "docs".to_string(),
            namespace: None,
            pagination_token: Some("page-2-token".to_string()),
        },
    )
    .await
    .unwrap();

    assert_eq!(page.ids, vec!["1:3".to_string()]);
    assert_eq!(
        page.next_token, None,
        "no further pagination.next means exhausted"
    );
}

#[tokio::test]
async fn list_vector_ids_page_404_is_an_empty_exhausted_page() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/vectors/list"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "code": 5, "message": "Namespace not found"
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let page = list_vector_ids_page_blocking(
        Arc::new(client),
        ListVectorIdsPageRequest {
            index: "docs".to_string(),
            namespace: Some("never-written".to_string()),
            pagination_token: None,
        },
    )
    .await
    .unwrap();

    assert!(page.ids.is_empty());
    assert_eq!(page.next_token, None);
}

#[tokio::test]
async fn list_vector_ids_page_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;
    let client = Client::new(config(cp.uri(), ""));
    let result = list_vector_ids_page_blocking(
        Arc::new(client),
        ListVectorIdsPageRequest {
            index: "docs".to_string(),
            namespace: None,
            pagination_token: None,
        },
    )
    .await;
    assert!(matches!(result, Err(Error::MissingApiKey)), "{result:?}");
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

// ---- update_vector_metadata ----

async fn update_vector_metadata_blocking(
    client: Arc<Client>,
    req: UpdateVectorMetadataRequest,
) -> Result<(), Error> {
    tokio::task::spawn_blocking(move || client.update_vector_metadata(&req))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn update_vector_metadata_posts_set_metadata_body() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/update"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(body_json(json!({
            "id": "v1",
            "setMetadata": {"genre": "comedy"},
            "namespace": "prod",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    update_vector_metadata_blocking(
        Arc::new(client),
        UpdateVectorMetadataRequest {
            index: "docs".to_string(),
            namespace: Some("prod".to_string()),
            id: "v1".to_string(),
            metadata: json!({"genre": "comedy"}),
        },
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn update_vector_metadata_omits_namespace_when_none() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/update"))
        .and(body_json(json!({
            "id": "v1",
            "setMetadata": {"genre": "comedy"},
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    update_vector_metadata_blocking(
        Arc::new(client),
        UpdateVectorMetadataRequest {
            index: "docs".to_string(),
            namespace: None,
            id: "v1".to_string(),
            metadata: json!({"genre": "comedy"}),
        },
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn update_vector_metadata_400_maps_to_bad_request_without_key_leak() {
    let api_key = "sk-updatemeta-2b3c4d";

    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/update"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3, "message": "vector v1 does not exist"
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = update_vector_metadata_blocking(
        Arc::new(client),
        UpdateVectorMetadataRequest {
            index: "docs".to_string(),
            namespace: None,
            id: "v1".to_string(),
            metadata: json!({"genre": "comedy"}),
        },
    )
    .await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(message.contains("does not exist")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

#[tokio::test]
async fn update_vector_metadata_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;
    let client = Client::new(config(cp.uri(), ""));
    let result = update_vector_metadata_blocking(
        Arc::new(client),
        UpdateVectorMetadataRequest {
            index: "docs".to_string(),
            namespace: None,
            id: "v1".to_string(),
            metadata: json!({"genre": "comedy"}),
        },
    )
    .await;
    assert!(matches!(result, Err(Error::MissingApiKey)), "{result:?}");
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

// ---- Backups & backup schedules (control plane, BACKUP_API_VERSION = "2026-07") ----
//
// Per-endpoint API-version split: every mock below asserts
// the `2026-07` header explicitly, while `describe_index`/`create_index`
// elsewhere in this file keep asserting `2026-04` — proving the two
// generations coexist without a global bump.

fn backup_response(backup_id: &str, status: &str) -> Value {
    json!({
        "backup_id": backup_id,
        "name": "nightly",
        "status": status,
        "source_index_name": "docs",
        "dimension": 8,
        "record_count": 100,
        "namespace_count": 2,
        "size_bytes": 4096,
        "created_at": "2026-09-11T00:00:00Z",
    })
}

fn expected_backup_info(backup_id: &str, status: &str) -> BackupInfo {
    BackupInfo {
        backup_id: backup_id.to_string(),
        name: Some("nightly".to_string()),
        status: status.to_string(),
        source_index_name: Some("docs".to_string()),
        dimension: Some(8),
        record_count: Some(100),
        namespace_count: Some(2),
        size_bytes: Some(4096),
        created_at: Some("2026-09-11T00:00:00Z".to_string()),
    }
}

async fn create_backup_blocking(
    client: Arc<Client>,
    req: CreateBackupRequest,
) -> Result<BackupInfo, Error> {
    tokio::task::spawn_blocking(move || client.create_backup(&req))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn create_backup_posts_name_and_description_with_2026_07_header() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes/docs/backups"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .and(body_json(
            json!({"name": "nightly", "description": "pre-migration"}),
        ))
        .respond_with(ResponseTemplate::new(201).set_body_json(backup_response("bkp-1", "Pending")))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let info = create_backup_blocking(
        Arc::new(client),
        CreateBackupRequest {
            index: "docs".to_string(),
            name: Some("nightly".to_string()),
            description: Some("pre-migration".to_string()),
        },
    )
    .await
    .unwrap();

    assert_eq!(info, expected_backup_info("bkp-1", "Pending"));
}

#[tokio::test]
async fn create_backup_omits_name_and_description_when_none() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes/docs/backups"))
        .and(body_json(json!({})))
        .respond_with(ResponseTemplate::new(201).set_body_json(backup_response("bkp-2", "Pending")))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    create_backup_blocking(
        Arc::new(client),
        CreateBackupRequest {
            index: "docs".to_string(),
            name: None,
            description: None,
        },
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn create_backup_404_maps_to_index_not_found() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes/ghost/backups"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"code": "NOT_FOUND", "message": "index 'ghost' not found"},
            "status": 404,
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = create_backup_blocking(
        Arc::new(client),
        CreateBackupRequest {
            index: "ghost".to_string(),
            name: None,
            description: None,
        },
    )
    .await;
    match result {
        Err(Error::IndexNotFound { index }) => assert_eq!(index, "ghost"),
        other => panic!("expected IndexNotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn create_backup_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;
    let client = Client::new(config(cp.uri(), ""));
    let result = create_backup_blocking(
        Arc::new(client),
        CreateBackupRequest {
            index: "docs".to_string(),
            name: None,
            description: None,
        },
    )
    .await;
    assert!(matches!(result, Err(Error::MissingApiKey)), "{result:?}");
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

async fn list_backups_blocking(
    client: Arc<Client>,
    index: String,
) -> Result<Vec<BackupInfo>, Error> {
    tokio::task::spawn_blocking(move || client.list_backups(&index))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn list_backups_is_index_scoped_and_walks_pagination_with_2026_07_header() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs/backups"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .and(query_param_is_missing("paginationToken"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [backup_response("bkp-1", "Ready")],
            "pagination": {"next": "page-2"},
        })))
        .expect(1)
        .mount(&cp)
        .await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs/backups"))
        .and(query_param("paginationToken", "page-2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [backup_response("bkp-2", "Ready")],
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let backups = list_backups_blocking(Arc::new(client), "docs".to_string())
        .await
        .unwrap();

    assert_eq!(
        backups,
        vec![
            expected_backup_info("bkp-1", "Ready"),
            expected_backup_info("bkp-2", "Ready"),
        ]
    );
}

async fn list_project_backups_blocking(client: Arc<Client>) -> Result<Vec<BackupInfo>, Error> {
    tokio::task::spawn_blocking(move || client.list_project_backups())
        .await
        .expect("blocking task panicked")
}

/// Project-scoped: no index in the path, and no index-host resolution — the
/// control plane is hit directly.
#[tokio::test]
async fn list_project_backups_hits_the_project_wide_path_with_2026_07_header() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/backups"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [backup_response("bkp-1", "Ready"), backup_response("bkp-2", "Pending")],
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let backups = list_project_backups_blocking(Arc::new(client))
        .await
        .unwrap();
    assert_eq!(backups.len(), 2);
}

async fn describe_backup_blocking(client: Arc<Client>, id: String) -> Result<BackupInfo, Error> {
    tokio::task::spawn_blocking(move || client.describe_backup(&id))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn describe_backup_returns_info_with_2026_07_header() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/backups/bkp-1"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .respond_with(ResponseTemplate::new(200).set_body_json(backup_response("bkp-1", "Ready")))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let info = describe_backup_blocking(Arc::new(client), "bkp-1".to_string())
        .await
        .unwrap();
    assert_eq!(info, expected_backup_info("bkp-1", "Ready"));
}

#[tokio::test]
async fn describe_backup_404_maps_to_backup_not_found() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/backups/ghost"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": {"code": "NOT_FOUND", "message": "backup not found"},
            "status": 404,
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = describe_backup_blocking(Arc::new(client), "ghost".to_string()).await;
    match result {
        Err(Error::BackupNotFound { backup_id }) => assert_eq!(backup_id, "ghost"),
        other => panic!("expected BackupNotFound, got {other:?}"),
    }
}

async fn delete_backup_blocking(client: Arc<Client>, id: String) -> Result<(), Error> {
    tokio::task::spawn_blocking(move || client.delete_backup(&id))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn delete_backup_202_is_ok_with_2026_07_header() {
    let cp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/backups/bkp-1"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .respond_with(ResponseTemplate::new(202))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    delete_backup_blocking(Arc::new(client), "bkp-1".to_string())
        .await
        .unwrap();
}

#[tokio::test]
async fn delete_backup_404_maps_to_backup_not_found() {
    let cp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/backups/ghost"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = delete_backup_blocking(Arc::new(client), "ghost".to_string()).await;
    match result {
        Err(Error::BackupNotFound { backup_id }) => assert_eq!(backup_id, "ghost"),
        other => panic!("expected BackupNotFound, got {other:?}"),
    }
}

async fn create_index_from_backup_blocking(
    client: Arc<Client>,
    backup_id: String,
    new_index_name: String,
) -> Result<IndexInfo, Error> {
    tokio::task::spawn_blocking(move || {
        client.create_index_from_backup(&backup_id, &new_index_name)
    })
    .await
    .expect("blocking task panicked")
}

#[tokio::test]
async fn create_index_from_backup_posts_new_name_with_2026_07_header() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/backups/bkp-1/create-index"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .and(body_json(json!({"name": "docs-restored"})))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({
            "name": "docs-restored",
            "vector_type": "dense",
            "dimension": 8,
            "metric": "cosine",
            "host": "docs-restored-abcd.svc.pinecone.io",
            "status": {"ready": false, "state": "Initializing"},
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let info = create_index_from_backup_blocking(
        Arc::new(client),
        "bkp-1".to_string(),
        "docs-restored".to_string(),
    )
    .await
    .unwrap();

    assert_eq!(
        info,
        IndexInfo {
            host: "https://docs-restored-abcd.svc.pinecone.io".to_string(),
            dimension: 8,
            metric: "cosine".to_string(),
            vector_type: "dense".to_string(),
            ready: false,
            read_capacity: None,
        }
    );
}

#[tokio::test]
async fn create_index_from_backup_404_maps_to_backup_not_found() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/backups/ghost/create-index"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = create_index_from_backup_blocking(
        Arc::new(client),
        "ghost".to_string(),
        "docs-restored".to_string(),
    )
    .await;
    match result {
        Err(Error::BackupNotFound { backup_id }) => assert_eq!(backup_id, "ghost"),
        other => panic!("expected BackupNotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn create_index_from_backup_409_maps_to_conflict() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/backups/bkp-1/create-index"))
        .respond_with(ResponseTemplate::new(409))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = create_index_from_backup_blocking(
        Arc::new(client),
        "bkp-1".to_string(),
        "docs-restored".to_string(),
    )
    .await;
    match result {
        Err(Error::Conflict { name }) => assert_eq!(name, "docs-restored"),
        other => panic!("expected Conflict, got {other:?}"),
    }
}

// ---- Backup schedules (provisional shape — see Client::create_backup_schedule) ----

#[tokio::test]
async fn create_backup_schedule_posts_schedule_verbatim_with_2026_07_header() {
    let cp = MockServer::start().await;
    let schedule = json!({"schedule": {"cron": "0 0 * * *"}});
    Mock::given(method("POST"))
        .and(path("/indexes/docs/backups/schedule"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .and(body_json(schedule.clone()))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({"schedule_id": "sched-1"})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let index = "docs".to_string();
    let result =
        tokio::task::spawn_blocking(move || client.create_backup_schedule(&index, schedule))
            .await
            .expect("blocking task panicked")
            .unwrap();
    assert_eq!(result["schedule_id"], json!("sched-1"));
}

#[tokio::test]
async fn get_backup_schedule_404_is_none() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs/backups/schedule"))
        .and(header("X-Pinecone-Api-Version", "2026-07"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let index = "docs".to_string();
    let result = tokio::task::spawn_blocking(move || client.get_backup_schedule(&index))
        .await
        .expect("blocking task panicked")
        .unwrap();
    assert_eq!(result, None);
}

// ---- Backup plan-gate 403 ----
//
// Pinecone returns 403 with a plan-gate body on backup endpoints for
// Starter/Builder orgs. Mapping it to Error::Auth masked it as "check your
// API key" (the drill's BACKUP leg hit exactly this); it must surface as
// BadRequest carrying the server's own message so the 0.6 backup gate can
// tell plan-ineligible apart from bad credentials programmatically.

/// The 403 body shape the live drill hit (message text as quoted in the
/// drill findings).
fn plan_gate_403() -> ResponseTemplate {
    ResponseTemplate::new(403).set_body_json(json!({
        "error": {
            "code": "FORBIDDEN",
            "message": "Your organization's plan does not include backups. To use backups, \
                        upgrade to the Standard or Enterprise plan."
        },
        "status": 403,
    }))
}

fn assert_plan_gate_surfaced(result: Result<impl std::fmt::Debug, Error>) {
    match result {
        Err(Error::BadRequest { message }) => {
            assert!(
                message.contains("plan does not include backups"),
                "the server's own plan-gate message must survive: {message}"
            );
            assert!(
                message.to_lowercase().contains("plan"),
                "message = {message}"
            );
        }
        Err(Error::Auth) => panic!(
            "a plan-gate 403 must NOT map to Error::Auth — that masks it as a credentials \
             problem"
        ),
        other => panic!("expected BadRequest carrying the plan-gate body, got {other:?}"),
    }
}

#[tokio::test]
async fn create_backup_403_with_plan_gate_body_is_bad_request_not_auth() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes/docs/backups"))
        .respond_with(plan_gate_403())
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = create_backup_blocking(
        Arc::new(client),
        CreateBackupRequest {
            index: "docs".to_string(),
            name: None,
            description: None,
        },
    )
    .await;
    assert_plan_gate_surfaced(result);
}

#[tokio::test]
async fn list_backups_403_with_plan_gate_body_is_bad_request_not_auth() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs/backups"))
        .respond_with(plan_gate_403())
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = list_backups_blocking(Arc::new(client), "docs".to_string()).await;
    assert_plan_gate_surfaced(result);
}

#[tokio::test]
async fn describe_backup_403_with_plan_gate_body_is_bad_request_not_auth() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/backups/bkp-1"))
        .respond_with(plan_gate_403())
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = describe_backup_blocking(Arc::new(client), "bkp-1".to_string()).await;
    assert_plan_gate_surfaced(result);
}

#[tokio::test]
async fn create_index_from_backup_403_with_plan_gate_body_is_bad_request_not_auth() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/backups/bkp-1/create-index"))
        .respond_with(plan_gate_403())
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = create_index_from_backup_blocking(
        Arc::new(client),
        "bkp-1".to_string(),
        "docs-restored".to_string(),
    )
    .await;
    assert_plan_gate_surfaced(result);
}

/// A bare 401 must still map to Auth on the backup surface — the split is
/// 401 = credentials, 403 = authenticated-but-refused (plan gate).
#[tokio::test]
async fn create_backup_401_still_maps_to_auth() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes/docs/backups"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = create_backup_blocking(
        Arc::new(client),
        CreateBackupRequest {
            index: "docs".to_string(),
            name: None,
            description: None,
        },
    )
    .await;
    assert!(matches!(result, Err(Error::Auth)), "got {result:?}");
}

// ---- fetch_by_metadata ----
//
// Request/response shapes follow this client's sibling data-plane endpoints
// (`/vectors/fetch`'s `vectors` object, `/vectors/list`'s `pagination.next`
// token). Documented constraints: 10k/page, 4MB response cap, 5 rps,
// non-empty filter required.

fn fetch_by_metadata_req(filter: Value) -> FetchByMetadataRequest {
    FetchByMetadataRequest {
        index: "docs".to_string(),
        namespace: None,
        filter,
        limit: None,
        pagination_token: None,
    }
}

async fn fetch_by_metadata_blocking(
    client: Arc<Client>,
    req: FetchByMetadataRequest,
) -> Result<FetchByMetadataPage, Error> {
    tokio::task::spawn_blocking(move || client.fetch_by_metadata(&req))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn fetch_by_metadata_resolves_host_then_posts_filter_with_2026_04_header() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/fetch_by_metadata"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(body_json(json!({
            "filter": {"pg_pinecone_batch": {"$eq": "eval-1"}},
            "limit": 10000,
            "namespace": "ns-a",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "vectors": {
                "v1": {"id": "v1", "values": [0.1, 0.2], "metadata": {"pg_pinecone_batch": "eval-1"}},
            },
            "pagination": {"next": "page-2-token"},
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let req = FetchByMetadataRequest {
        index: "docs".to_string(),
        namespace: Some("ns-a".to_string()),
        filter: json!({"pg_pinecone_batch": {"$eq": "eval-1"}}),
        limit: None,
        pagination_token: None,
    };
    let page = fetch_by_metadata_blocking(Arc::new(client), req)
        .await
        .unwrap();

    assert_eq!(page.next_token, Some("page-2-token".to_string()));
    let v1 = page.vectors.get("v1").expect("v1 present");
    assert_eq!(v1.values, vec![0.1, 0.2]);
    assert_eq!(v1.metadata, Some(json!({"pg_pinecone_batch": "eval-1"})));
}

#[tokio::test]
async fn fetch_by_metadata_sends_pagination_token_when_resuming() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/fetch_by_metadata"))
        .and(body_json(json!({
            "filter": {"pg_pinecone_batch": {"$eq": "eval-1"}},
            "limit": 10000,
            "paginationToken": "page-2-token",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"vectors": {}})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let mut req = fetch_by_metadata_req(json!({"pg_pinecone_batch": {"$eq": "eval-1"}}));
    req.pagination_token = Some("page-2-token".to_string());
    let page = fetch_by_metadata_blocking(Arc::new(client), req)
        .await
        .unwrap();
    assert!(page.vectors.is_empty());
    assert_eq!(
        page.next_token, None,
        "an absent pagination object means no next page"
    );
}

#[tokio::test]
async fn fetch_by_metadata_respects_an_explicit_limit() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/fetch_by_metadata"))
        .and(body_json(json!({
            "filter": {"k": {"$eq": "v"}},
            "limit": 500,
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"vectors": {}})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let mut req = fetch_by_metadata_req(json!({"k": {"$eq": "v"}}));
    req.limit = Some(500);
    fetch_by_metadata_blocking(Arc::new(client), req)
        .await
        .unwrap();
}

#[tokio::test]
async fn fetch_by_metadata_empty_object_filter_is_rejected_without_any_http_call() {
    let cp = MockServer::start().await;
    let client = Client::new(config(cp.uri(), "key"));
    let result =
        fetch_by_metadata_blocking(Arc::new(client), fetch_by_metadata_req(json!({}))).await;
    match result {
        Err(Error::BadRequest { message }) => {
            assert!(message.contains("non-empty filter"), "message = {message}")
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn fetch_by_metadata_null_filter_is_rejected_without_any_http_call() {
    let cp = MockServer::start().await;
    let client = Client::new(config(cp.uri(), "key"));
    let result =
        fetch_by_metadata_blocking(Arc::new(client), fetch_by_metadata_req(Value::Null)).await;
    assert!(
        matches!(result, Err(Error::BadRequest { .. })),
        "{result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn fetch_by_metadata_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;
    let client = Client::new(config(cp.uri(), ""));
    let result = fetch_by_metadata_blocking(
        Arc::new(client),
        fetch_by_metadata_req(json!({"k": {"$eq": "v"}})),
    )
    .await;
    assert!(matches!(result, Err(Error::MissingApiKey)), "{result:?}");
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn fetch_by_metadata_400_maps_to_bad_request_with_message() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/fetch_by_metadata"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3, "message": "filter must be non-empty"
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = fetch_by_metadata_blocking(
        Arc::new(client),
        fetch_by_metadata_req(json!({"k": {"$eq": "v"}})),
    )
    .await;
    match result {
        Err(Error::BadRequest { message }) => {
            assert!(message.contains("filter must be non-empty"), "{message}")
        }
        other => panic!("expected BadRequest, got {other:?}"),
    }
}

#[tokio::test]
async fn fetch_by_metadata_401_maps_to_auth() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/fetch_by_metadata"))
        .respond_with(ResponseTemplate::new(401))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = fetch_by_metadata_blocking(
        Arc::new(client),
        fetch_by_metadata_req(json!({"k": {"$eq": "v"}})),
    )
    .await;
    assert!(matches!(result, Err(Error::Auth)), "{result:?}");
}

/// No namespace at all (never written to) — same idempotent "nothing there"
/// convention every other data-plane 404 in this client uses.
#[tokio::test]
async fn fetch_by_metadata_404_is_empty_page() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/fetch_by_metadata"))
        .respond_with(ResponseTemplate::new(404))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let page = fetch_by_metadata_blocking(
        Arc::new(client),
        fetch_by_metadata_req(json!({"k": {"$eq": "v"}})),
    )
    .await
    .unwrap();
    assert!(page.vectors.is_empty());
    assert_eq!(page.next_token, None);
}

// ---- Namespace operations: list / describe / delete (API_VERSION "2026-04") ----

fn list_namespaces_req(index: impl Into<String>) -> ListNamespacesRequest {
    ListNamespacesRequest {
        index: index.into(),
        prefix: None,
        limit: None,
        pagination_token: None,
    }
}

async fn list_namespaces_blocking(
    client: Arc<Client>,
    req: ListNamespacesRequest,
) -> Result<ListNamespacesPage, Error> {
    tokio::task::spawn_blocking(move || client.list_namespaces(&req))
        .await
        .expect("blocking task panicked")
}

fn ns(name: &str, record_count: u64) -> NamespaceInfo {
    NamespaceInfo {
        name: name.to_string(),
        record_count: Some(record_count),
        ..NamespaceInfo::default()
    }
}

/// `GET /namespaces` with every parameter set: both headers (host resolution
/// and data plane) and the OAS response shape, where `record_count` and
/// `size_bytes` are decimal STRINGS and `pagination.next` is the next token.
#[tokio::test]
async fn list_namespaces_gets_the_page_with_headers_and_query_params() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/namespaces"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(query_param("prefix", "public.docs:"))
        .and(query_param("limit", "50"))
        .and(query_param("paginationToken", "tok-2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "namespaces": [
                {
                    "name": "public.docs:1",
                    "record_count": "20000",
                    "size_bytes": "1048576",
                    "schema": {"fields": {"tenant_id": {"filterable": true}}},
                    "indexed_fields": {"fields": ["tenant_id"]}
                },
                {"name": "public.docs:2", "record_count": "0"}
            ],
            "pagination": {"next": "tok-3"},
            "total_count": 3
        })))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let page = list_namespaces_blocking(
        Arc::new(client),
        ListNamespacesRequest {
            prefix: Some("public.docs:".to_string()),
            limit: Some(50),
            pagination_token: Some("tok-2".to_string()),
            ..list_namespaces_req("docs")
        },
    )
    .await
    .unwrap();

    let first = NamespaceInfo {
        name: "public.docs:1".to_string(),
        record_count: Some(20_000),
        size_bytes: Some(1_048_576),
        indexed_fields: vec!["tenant_id".to_string()],
        schema: Some(json!({"fields": {"tenant_id": {"filterable": true}}})),
    };
    assert_eq!(page.namespaces, vec![first, ns("public.docs:2", 0)]);
    assert_eq!(page.next.as_deref(), Some("tok-3"));
}

#[tokio::test]
async fn list_namespaces_sends_no_query_string_when_no_param_is_set() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/namespaces"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"namespaces": []})))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let page = list_namespaces_blocking(Arc::new(client), list_namespaces_req("docs"))
        .await
        .unwrap();

    assert_eq!(page, ListNamespacesPage::default());
    let requests = dp.received_requests().await.unwrap();
    assert_eq!(requests[0].url.query(), None);
}

/// `prefix` and `paginationToken` are query-string components: `&`, `=` and
/// `+` must be escaped (the path-segment set leaves them alone), while the
/// table-prefix colon stays literal.
#[tokio::test]
async fn list_namespaces_percent_encodes_prefix_and_token_as_query_components() {
    let prefix = "a b&c=d+e/f%g:h";
    let token = "tok+en/with=plus&amp==";

    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/namespaces"))
        .and(query_param("prefix", prefix))
        .and(query_param("paginationToken", token))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"namespaces": []})))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    list_namespaces_blocking(
        Arc::new(client),
        ListNamespacesRequest {
            prefix: Some(prefix.to_string()),
            pagination_token: Some(token.to_string()),
            ..list_namespaces_req("docs")
        },
    )
    .await
    .unwrap();

    let requests = dp.received_requests().await.unwrap();
    let mut parts: Vec<&str> = requests[0].url.query().unwrap().split('&').collect();
    parts.sort();
    assert_eq!(
        parts,
        [
            "paginationToken=tok%2Ben%2Fwith%3Dplus%26amp%3D%3D",
            "prefix=a%20b%26c%3Dd%2Be%2Ff%25g:h",
        ]
    );
}

/// Counts arrive as decimal strings per the OAS but may be plain numbers;
/// anything unparsable, missing or oddly typed becomes `None`/empty instead of
/// failing the whole page, and unknown fields are ignored.
#[tokio::test]
async fn list_namespaces_parses_leniently() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/namespaces"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "namespaces": [
                {"name": "numbers", "record_count": 7, "size_bytes": 9},
                {"name": "strings", "record_count": "12", "size_bytes": "34"},
                {
                    "name": "garbage",
                    "record_count": "n/a",
                    "size_bytes": {"x": 1},
                    "schema": null,
                    "indexed_fields": "oops"
                },
                {"name": "bare"},
                {"name": "extras", "record_count": "1", "future_field": [1, 2]},
                {}
            ],
            "pagination": {}
        })))
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let page = list_namespaces_blocking(Arc::new(client), list_namespaces_req("docs"))
        .await
        .unwrap();

    let named = |name: &str| NamespaceInfo {
        name: name.to_string(),
        ..NamespaceInfo::default()
    };
    let numbers = NamespaceInfo {
        record_count: Some(7),
        size_bytes: Some(9),
        ..named("numbers")
    };
    let strings = NamespaceInfo {
        record_count: Some(12),
        size_bytes: Some(34),
        ..named("strings")
    };
    let expected = vec![
        numbers,
        strings,
        named("garbage"),
        named("bare"),
        ns("extras", 1),
        NamespaceInfo::default(),
    ];
    assert_eq!(page.namespaces, expected);
    assert_eq!(page.next, None);
}

/// No `pagination`, an empty one, or an empty `next` all mean "last page".
#[tokio::test]
async fn list_namespaces_without_a_usable_next_token_is_the_last_page() {
    for body in [
        json!({}),
        json!({"namespaces": []}),
        json!({"namespaces": [], "pagination": null}),
        json!({"namespaces": [], "pagination": {}}),
        json!({"namespaces": [], "pagination": {"next": ""}}),
    ] {
        let dp = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/namespaces"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body.clone()))
            .expect(1)
            .mount(&dp)
            .await;
        let cp = MockServer::start().await;
        mount_host(&cp, "docs", &dp.uri()).await;

        let client = Client::new(config(cp.uri(), "key"));
        let page = list_namespaces_blocking(Arc::new(client), list_namespaces_req("docs"))
            .await
            .unwrap();

        assert_eq!(page, ListNamespacesPage::default(), "body {body}");
    }
}

#[tokio::test]
async fn list_namespaces_400_maps_to_bad_request_without_key_leak() {
    let api_key = "sk-listns-7d6e5f";

    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/namespaces"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3,
            "message": "limit must be between 1 and 100"
        })))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), api_key));
    let req = ListNamespacesRequest {
        limit: Some(0),
        ..list_namespaces_req("docs")
    };
    let result = list_namespaces_blocking(Arc::new(client), req).await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(message.contains("limit must be")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

/// A list has no namespace-scoped 404: like the other data-plane calls,
/// anything but 200/400/401/403 is `Error::Api`.
#[tokio::test]
async fn list_namespaces_maps_other_statuses_like_the_other_data_plane_calls() {
    for (status, is_auth) in [
        (401u16, true),
        (403, true),
        (404, false),
        (429, false),
        (500, false),
    ] {
        let dp = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/namespaces"))
            .respond_with(ResponseTemplate::new(status).set_body_string("nope"))
            .expect(1)
            .mount(&dp)
            .await;
        let cp = MockServer::start().await;
        mount_host(&cp, "docs", &dp.uri()).await;

        let client = Client::new(config(cp.uri(), "key"));
        let result = list_namespaces_blocking(Arc::new(client), list_namespaces_req("docs")).await;

        match result {
            Err(Error::Auth) if is_auth => {}
            Err(Error::Api { status: got, .. }) if !is_auth => assert_eq!(got, status),
            other => panic!("status {status}: unexpected {other:?}"),
        }
    }
}

#[tokio::test]
async fn list_namespaces_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = list_namespaces_blocking(Arc::new(client), list_namespaces_req("docs")).await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

async fn list_all_namespaces_blocking(
    client: Arc<Client>,
    index: &'static str,
    prefix: Option<&'static str>,
) -> Result<Vec<NamespaceInfo>, Error> {
    tokio::task::spawn_blocking(move || client.list_all_namespaces(index, prefix))
        .await
        .expect("blocking task panicked")
}

/// `list_all_namespaces` walks every page: `prefix` and the maximum page size
/// ride on each request, the `pagination.next` token on each one after the
/// first, and the pages concatenate in order.
#[tokio::test]
async fn list_all_namespaces_follows_pagination_repeating_prefix_and_limit() {
    let token2 = "p2+/=&";

    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/namespaces"))
        .and(query_param("prefix", "public.docs:"))
        .and(query_param("limit", "100"))
        .and(query_param_is_missing("paginationToken"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "namespaces": [{"name": "public.docs:1", "record_count": "1"}],
            "pagination": {"next": token2}
        })))
        .expect(1)
        .mount(&dp)
        .await;
    Mock::given(method("GET"))
        .and(path("/namespaces"))
        .and(query_param("prefix", "public.docs:"))
        .and(query_param("limit", "100"))
        .and(query_param("paginationToken", token2))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "namespaces": [{"name": "public.docs:2", "record_count": "2"}],
            "pagination": {"next": "p3"}
        })))
        .expect(1)
        .mount(&dp)
        .await;
    Mock::given(method("GET"))
        .and(path("/namespaces"))
        .and(query_param("prefix", "public.docs:"))
        .and(query_param("limit", "100"))
        .and(query_param("paginationToken", "p3"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "namespaces": [{"name": "public.docs:3", "record_count": "3"}]
        })))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    let all = list_all_namespaces_blocking(Arc::new(client), "docs", Some("public.docs:"))
        .await
        .unwrap();

    let names: Vec<&str> = all.iter().map(|n| n.name.as_str()).collect();
    assert_eq!(names, ["public.docs:1", "public.docs:2", "public.docs:3"]);
}

#[tokio::test]
async fn list_all_namespaces_without_a_prefix_omits_it() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/namespaces"))
        .and(query_param_is_missing("prefix"))
        .and(query_param("limit", "100"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "namespaces": [{"name": "a", "record_count": "1"}]
        })))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let all = list_all_namespaces_blocking(Arc::new(client), "docs", None)
        .await
        .unwrap();

    assert_eq!(all, vec![ns("a", 1)]);
}

/// A server that keeps handing back the token it was just given would
/// otherwise spin this synchronous walk (inside a Postgres backend) up to the
/// page cap.
#[tokio::test]
async fn list_all_namespaces_stops_when_the_next_token_does_not_advance() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/namespaces"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "namespaces": [{"name": "a", "record_count": "1"}],
            "pagination": {"next": "stuck"}
        })))
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = list_all_namespaces_blocking(Arc::new(client), "docs", None).await;

    assert!(matches!(result, Err(Error::Api { .. })), "got {result:?}");
    // The first page, then the one that returned the same token again.
    assert_eq!(dp.received_requests().await.unwrap().len(), 2);
}

async fn describe_namespace_blocking(
    client: Arc<Client>,
    index: &'static str,
    namespace: &'static str,
) -> Result<NamespaceInfo, Error> {
    tokio::task::spawn_blocking(move || client.describe_namespace(index, namespace))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn describe_namespace_gets_the_namespace_with_headers() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/namespaces/public.docs:42"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "name": "public.docs:42",
            "record_count": "20000",
            "indexed_fields": {"fields": ["genre", "year"]},
            "size_bytes": "2048"
        })))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let info = describe_namespace_blocking(Arc::new(client), "docs", "public.docs:42")
        .await
        .unwrap();

    let expected = NamespaceInfo {
        name: "public.docs:42".to_string(),
        record_count: Some(20_000),
        size_bytes: Some(2_048),
        indexed_fields: vec!["genre".to_string(), "year".to_string()],
        schema: None,
    };
    assert_eq!(info, expected);
}

/// The same path-segment escaping every namespaced call uses: `/`, `?`, `#`
/// and `%` must not redirect or truncate the request, `:` stays literal.
#[tokio::test]
async fn describe_namespace_percent_encodes_the_path_segment() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/namespaces/a%20b%2Fc%3Fd%23e%25f:g"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"name": "a b/c?d#e%f:g"})))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let info = describe_namespace_blocking(Arc::new(client), "docs", "a b/c?d#e%f:g")
        .await
        .unwrap();

    assert_eq!(info.name, "a b/c?d#e%f:g");
}

/// 404 is the one status with a meaning of its own on a describe: the
/// namespace does not exist.
#[tokio::test]
async fn describe_namespace_404_maps_to_namespace_not_found() {
    let dp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/namespaces/public.docs:42"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "code": 5, "message": "Namespace not found"
        })))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = describe_namespace_blocking(Arc::new(client), "docs", "public.docs:42").await;

    let err = result.expect_err("expected namespace-not-found");
    match &err {
        Error::NamespaceNotFound { namespace } => assert_eq!(namespace, "public.docs:42"),
        other => panic!("expected NamespaceNotFound, got {other:?}"),
    }
    assert!(err.to_string().contains("public.docs:42"), "{err}");
}

#[tokio::test]
async fn describe_namespace_maps_other_statuses_like_the_other_data_plane_calls() {
    for (status, kind) in [
        (400u16, "bad"),
        (401, "auth"),
        (403, "auth"),
        (429, "api"),
        (500, "api"),
    ] {
        let dp = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/namespaces/public.docs:42"))
            .respond_with(ResponseTemplate::new(status).set_body_string("nope"))
            .expect(1)
            .mount(&dp)
            .await;
        let cp = MockServer::start().await;
        mount_host(&cp, "docs", &dp.uri()).await;

        let client = Client::new(config(cp.uri(), "key"));
        let result = describe_namespace_blocking(Arc::new(client), "docs", "public.docs:42").await;

        match (kind, result) {
            ("bad", Err(Error::BadRequest { message })) => assert_eq!(message, "nope"),
            ("auth", Err(Error::Auth)) => {}
            ("api", Err(Error::Api { status: got, .. })) => assert_eq!(got, status),
            (_, other) => panic!("status {status}: unexpected {other:?}"),
        }
    }
}

/// An empty name would make the request `GET /namespaces/`, a different
/// endpoint (the list); Pinecone's default namespace is `__default__` in paths.
#[tokio::test]
async fn describe_namespace_empty_name_is_bad_request_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = describe_namespace_blocking(Arc::new(client), "docs", "").await;

    assert!(
        matches!(result, Err(Error::BadRequest { .. })),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn describe_namespace_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = describe_namespace_blocking(Arc::new(client), "docs", "public.docs:42").await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

async fn delete_namespace_blocking(
    client: Arc<Client>,
    index: &'static str,
    namespace: &'static str,
) -> Result<(), Error> {
    tokio::task::spawn_blocking(move || client.delete_namespace(index, namespace))
        .await
        .expect("blocking task panicked")
}

#[tokio::test]
async fn delete_namespace_sends_delete_with_headers_and_returns_ok() {
    let dp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/namespaces/public.docs:42"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "test-key"));
    delete_namespace_blocking(Arc::new(client), "docs", "public.docs:42")
        .await
        .unwrap();
}

#[tokio::test]
async fn delete_namespace_percent_encodes_the_path_segment() {
    let dp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/namespaces/a%20b%2Fc%3Fd%23e%25f:g"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    delete_namespace_blocking(Arc::new(client), "docs", "a b/c?d#e%f:g")
        .await
        .unwrap();
}

/// Idempotent like `delete_vectors`: a namespace that is already gone (or was
/// never written to) is success, not an error.
#[tokio::test]
async fn delete_namespace_404_is_ok() {
    let dp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/namespaces/public.docs:42"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "code": 5, "message": "Namespace not found"
        })))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    delete_namespace_blocking(Arc::new(client), "docs", "public.docs:42")
        .await
        .expect("deleting an absent namespace must be an idempotent no-op");
}

#[tokio::test]
async fn delete_namespace_400_maps_to_bad_request_without_key_leak() {
    let api_key = "sk-delns-1a2b3c";

    let dp = MockServer::start().await;
    Mock::given(method("DELETE"))
        .and(path("/namespaces/public.docs:42"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "code": 3,
            "message": "namespace is the target of one or more aliases"
        })))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), api_key));
    let result = delete_namespace_blocking(Arc::new(client), "docs", "public.docs:42").await;

    let err = result.expect_err("expected bad-request error");
    match &err {
        Error::BadRequest { message } => assert!(message.contains("target of one or more")),
        other => panic!("expected BadRequest, got {other:?}"),
    }
    assert_no_key_leak(&err, api_key);
}

#[tokio::test]
async fn delete_namespace_maps_other_statuses_like_the_other_data_plane_calls() {
    for (status, is_auth) in [(401u16, true), (403, true), (429, false), (500, false)] {
        let dp = MockServer::start().await;
        Mock::given(method("DELETE"))
            .and(path("/namespaces/public.docs:42"))
            .respond_with(ResponseTemplate::new(status).set_body_string("nope"))
            .expect(1)
            .mount(&dp)
            .await;
        let cp = MockServer::start().await;
        mount_host(&cp, "docs", &dp.uri()).await;

        let client = Client::new(config(cp.uri(), "key"));
        let result = delete_namespace_blocking(Arc::new(client), "docs", "public.docs:42").await;

        match result {
            Err(Error::Auth) if is_auth => {}
            Err(Error::Api { status: got, .. }) if !is_auth => assert_eq!(got, status),
            other => panic!("status {status}: unexpected {other:?}"),
        }
    }
}

#[tokio::test]
async fn delete_namespace_empty_name_is_bad_request_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = delete_namespace_blocking(Arc::new(client), "docs", "").await;

    assert!(
        matches!(result, Err(Error::BadRequest { .. })),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

#[tokio::test]
async fn delete_namespace_missing_api_key_fails_without_any_http_call() {
    let cp = MockServer::start().await;

    let client = Client::new(config(cp.uri(), ""));
    let result = delete_namespace_blocking(Arc::new(client), "docs", "public.docs:42").await;

    assert!(
        matches!(result, Err(Error::MissingApiKey)),
        "got {result:?}"
    );
    assert_eq!(cp.received_requests().await.unwrap().len(), 0);
}

// ---- Vector delete by metadata filter (`DeleteVectorsRequest::filter`) ----

fn delete_by_filter(namespace: Option<&str>, filter: Value) -> DeleteVectorsRequest {
    DeleteVectorsRequest {
        index: "docs".to_string(),
        namespace: namespace.map(str::to_string),
        ids: None,
        delete_all: false,
        filter: Some(filter),
    }
}

#[tokio::test]
async fn delete_vectors_with_filter_posts_filter_and_namespace() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/delete"))
        .and(header("Api-Key", "test-key"))
        .and(header("X-Pinecone-Api-Version", "2026-04"))
        .and(body_json(json!({
            "filter": {"tenant_id": {"$eq": 42}},
            "namespace": "public.docs",
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "test-key"));
    let req = delete_by_filter(Some("public.docs"), json!({"tenant_id": {"$eq": 42}}));
    delete_vectors_blocking(Arc::new(client), req)
        .await
        .unwrap();
}

#[tokio::test]
async fn delete_vectors_with_filter_omits_namespace_when_none() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/delete"))
        .and(body_json(json!({"filter": {"tenant_id": {"$eq": "acme"}}})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({})))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let req = delete_by_filter(None, json!({"tenant_id": {"$eq": "acme"}}));
    delete_vectors_blocking(Arc::new(client), req)
        .await
        .unwrap();
}

/// Deletes are idempotent: a filter delete on a namespace that was never
/// written to is success, like the ids and delete-all forms.
#[tokio::test]
async fn delete_vectors_filter_404_namespace_not_found_is_ok() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/delete"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "code": 5, "message": "Namespace not found"
        })))
        .expect(1)
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let req = delete_by_filter(Some("never-written"), json!({"tenant_id": {"$eq": 1}}));
    delete_vectors_blocking(Arc::new(client), req)
        .await
        .unwrap();
}

/// Exactly one of `ids` / `delete_all` / `filter`: any pair (or all three) is
/// rejected client-side, before any network call.
#[tokio::test]
async fn delete_vectors_filter_with_ids_or_delete_all_is_bad_request_without_any_http_call() {
    let combos = [
        (Some(vec!["id-1".to_string()]), false),
        (None, true),
        (Some(vec!["id-1".to_string()]), true),
    ];
    for (ids, delete_all) in combos {
        let cp = MockServer::start().await;
        let client = Client::new(config(cp.uri(), "key"));
        let req = DeleteVectorsRequest {
            ids,
            delete_all,
            ..delete_by_filter(None, json!({"tenant_id": {"$eq": 1}}))
        };
        let result = delete_vectors_blocking(Arc::new(client), req).await;

        assert!(
            matches!(result, Err(Error::BadRequest { .. })),
            "got {result:?}"
        );
        assert_eq!(cp.received_requests().await.unwrap().len(), 0);
    }
}

/// Pinecone documents `filter` with `minProperties: 1`; an empty or null one
/// must never reach the wire, where "match everything" would be a disaster.
#[tokio::test]
async fn delete_vectors_empty_or_null_filter_is_bad_request_without_any_http_call() {
    for filter in [json!({}), Value::Null] {
        let cp = MockServer::start().await;
        let client = Client::new(config(cp.uri(), "key"));
        let result =
            delete_vectors_blocking(Arc::new(client), delete_by_filter(None, filter.clone())).await;

        assert!(
            matches!(result, Err(Error::BadRequest { .. })),
            "filter {filter}: got {result:?}"
        );
        assert_eq!(cp.received_requests().await.unwrap().len(), 0);
    }
}

// ---- User-Agent (Pinecone source_tag attribution) ----

const CONFIGURED_USER_AGENT: &str =
    "pg_pinecone-client-9.9.9; lang=rust; pg=17; client=9.9.9; source_tag=pg_pinecone:acme";

fn config_with_user_agent(
    control_plane_url: impl Into<String>,
    api_key: impl Into<String>,
) -> Config {
    Config {
        user_agent: Some(CONFIGURED_USER_AGENT.to_string()),
        ..config(control_plane_url, api_key)
    }
}

#[tokio::test]
async fn control_plane_call_sends_configured_user_agent() {
    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .and(header("User-Agent", CONFIGURED_USER_AGENT))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"host": "docs.svc.pinecone.io"})),
        )
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config_with_user_agent(cp.uri(), "key"));
    describe_index_blocking(Arc::new(client), "docs".to_string())
        .await
        .unwrap();
}

#[tokio::test]
async fn data_plane_query_sends_configured_user_agent() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/query"))
        .and(header("User-Agent", CONFIGURED_USER_AGENT))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"matches": []})))
        .expect(1)
        .mount(&dp)
        .await;

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"host": dp.uri()})))
        .mount(&cp)
        .await;

    let client = Client::new(config_with_user_agent(cp.uri(), "key"));
    query_blocking(Arc::new(client), query("docs"))
        .await
        .unwrap();
}

#[tokio::test]
async fn document_index_call_sends_configured_user_agent() {
    let cp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/indexes"))
        .and(header("User-Agent", CONFIGURED_USER_AGENT))
        .respond_with(ResponseTemplate::new(201).set_body_json(json!({
            "name": "articles",
            "host": "articles-abcd1234.svc.pinecone.io",
            "schema": {"fields": {}},
        })))
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config_with_user_agent(cp.uri(), "key"));
    create_doc_index_blocking(Arc::new(client), create_doc_index_request("articles"))
        .await
        .unwrap();
}

#[tokio::test]
async fn default_user_agent_is_sent_when_not_configured() {
    let default_ua = format!(
        "pg_pinecone-client-{}; lang=rust; source_tag=pg_pinecone",
        env!("CARGO_PKG_VERSION")
    );

    let cp = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/indexes/docs"))
        .and(header("User-Agent", default_ua.as_str()))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(json!({"host": "docs.svc.pinecone.io"})),
        )
        .expect(1)
        .mount(&cp)
        .await;

    let client = Client::new(config(cp.uri(), "key"));
    describe_index_blocking(Arc::new(client), "docs".to_string())
        .await
        .unwrap();
}

// ---- 403 that names the namespace limit ----

const NAMESPACE_LIMIT: &str = "Request failed. You've reached the maximum number of namespaces for \
    this index (100). To add more, upgrade your plan or delete a namespace.";

async fn upsert_after_403(body: &str) -> Result<u64, Error> {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/vectors/upsert"))
        .respond_with(ResponseTemplate::new(403).set_body_string(body.to_string()))
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "docs", &dp.uri()).await;
    let client = Client::new(config(cp.uri(), "key"));
    upsert_vectors_blocking(
        Arc::new(client),
        UpsertVectorsRequest {
            index: "docs".to_string(),
            namespace: Some("t:801".to_string()),
            vectors: vec![Vector {
                id: "v1".to_string(),
                values: vec![0.75, 0.5, 0.25],
                metadata: None,
            }],
        },
    )
    .await
}

/// Pinecone's documented status for an object quota is 403; the message says which quota.
#[tokio::test]
async fn upsert_vectors_403_naming_the_namespace_limit_is_a_refusal_not_an_auth_failure() {
    let json_body = json!({"code": 7, "message": NAMESPACE_LIMIT, "details": []}).to_string();
    for body in [NAMESPACE_LIMIT, json_body.as_str()] {
        match upsert_after_403(body).await {
            Err(Error::Api {
                status: 403,
                message,
            }) => {
                assert!(
                    message.contains("maximum number of namespaces"),
                    "{message}"
                );
            }
            other => panic!("expected the refusal with its message, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn upsert_vectors_403_for_any_other_reason_stays_an_auth_failure() {
    for body in ["", "Forbidden", "Invalid API Key", "namespace is not yours"] {
        let result = upsert_after_403(body).await;
        assert!(matches!(result, Err(Error::Auth)), "{body:?}: {result:?}");
    }
}

#[tokio::test]
async fn upsert_docs_403_naming_the_namespace_limit_is_a_refusal_not_an_auth_failure() {
    let dp = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/namespaces/t:801/documents/upsert"))
        .respond_with(ResponseTemplate::new(403).set_body_json(json!({
            "status": 403,
            "error": {"code": "FORBIDDEN", "message": NAMESPACE_LIMIT}
        })))
        .mount(&dp)
        .await;
    let cp = MockServer::start().await;
    mount_host(&cp, "articles", &dp.uri()).await;

    let client = Client::new(config(cp.uri(), "key"));
    let result = upsert_docs_blocking(
        Arc::new(client),
        UpsertDocsRequest {
            namespace: Some("t:801".to_string()),
            ..one_doc_upsert("articles")
        },
    )
    .await;

    match result {
        Err(Error::Api {
            status: 403,
            message,
        }) => {
            assert!(
                message.contains("maximum number of namespaces"),
                "{message}"
            );
        }
        other => panic!("expected the refusal with its message, got {other:?}"),
    }
}
