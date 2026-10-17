use std::{
    collections::HashMap,
    sync::{Mutex, MutexGuard},
    time::{Duration, Instant},
};

use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{Value, json};
use ureq::{
    Agent, Body,
    http::Response,
    unversioned::{
        resolver::DefaultResolver,
        transport::{Connector, DefaultConnector},
    },
};

use crate::{
    config::Config,
    error::Error,
    transport::ResumeReads,
    types::{
        BackupInfo, CreateBackupRequest, CreateDocIndexRequest, CreateIndexRequest,
        DeleteDocsRequest, DeleteVectorsRequest, DocDeleteSelector, DocIndexInfo, DocMatch,
        EmbedRequest, FetchByMetadataPage, FetchByMetadataRequest, FetchVectorsRequest, IndexInfo,
        IndexStats, ListNamespacesPage, ListNamespacesRequest, ListVectorIdsPage,
        ListVectorIdsPageRequest, ListVectorsRequest, Match, NamespaceInfo, NamespaceStats,
        QueryRequest, ReadCapacity, RerankRequest, RerankResult, SearchDocsRequest,
        UpdateVectorMetadataRequest, UpsertDocsRequest, UpsertVectorsRequest, Vector,
    },
};

/// Path segment used when the caller specifies no namespace; the document
/// endpoints require a namespace in the URL, and this is Pinecone's default one.
const DEFAULT_NAMESPACE: &str = "__default__";

/// Characters escaped when a string (an index name or namespace) is
/// interpolated into a URL path segment. Neither has documented character
/// restrictions, so anything that could break or redirect the path is
/// percent-encoded.
const PATH_SEGMENT: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'%')
    .add(b'/')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'\\')
    .add(b'^')
    .add(b'`')
    .add(b'{')
    .add(b'|')
    .add(b'}');

/// Percent-encodes a namespace for use as a single URL path segment.
fn namespace_segment(namespace: Option<&str>) -> impl std::fmt::Display + '_ {
    utf8_percent_encode(namespace.unwrap_or(DEFAULT_NAMESPACE), PATH_SEGMENT)
}

/// Characters escaped when a string is interpolated as a `key=value` query
/// STRING component (`list_vector_ids_host`'s `namespace`/`paginationToken`)
/// — broader than [`PATH_SEGMENT`]: `&`/`=` delimit query parameters and
/// their values, and `+` decodes to a literal space in this position, none
/// of which [`PATH_SEGMENT`] escapes (it was designed for a URL *path*
/// segment, where those three are unremarkable). Reusing `PATH_SEGMENT` here
/// let a namespace or base64url-ish pagination token containing any of the
/// three corrupt the query string (inject a bogus parameter, mis-split a
/// value, or silently decode a `+` as a space) instead of round-tripping
/// byte-for-byte.
const QUERY_COMPONENT: &AsciiSet = &PATH_SEGMENT.add(b'&').add(b'+').add(b'=');

/// Percent-encodes an index name for use as a single URL path segment —
/// applied everywhere a name is interpolated into a control-plane path
/// (`resolve_host`, [`Client::describe_index`], [`Client::delete_index`],
/// [`Client::delete_doc_index`]). Index names are further restricted by
/// Pinecone itself, but this guards against a name containing a character
/// (e.g. `/` or `?`) that would otherwise redirect or break the request path.
fn index_segment(name: &str) -> impl std::fmt::Display + '_ {
    utf8_percent_encode(name, PATH_SEGMENT)
}

/// The Pinecone API version this client speaks, sent as `X-Pinecone-Api-Version`.
/// Used by every classic vector/inference endpoint.
pub const API_VERSION: &str = "2026-04";

/// The API version required by document-schema (full-text-search) index
/// endpoints — a separate, additive shape on top of the classic control and
/// data planes. Sent as `X-Pinecone-Api-Version` by every doc-index method.
pub const DOC_API_VERSION: &str = "2026-07";

/// The API version required by the backup/backup-schedule control-plane
/// surface — `2026-07`, same generation as [`DOC_API_VERSION`] but an
/// independent constant: the two ship on unrelated schedules (backups vs.
/// document-schema indexes), and pinning them separately is what lets either
/// move again without touching the other. Sent as `X-Pinecone-Api-Version`
/// by every backup/backup-schedule method; every other classic control- and
/// data-plane call (indexes, vectors, inference) stays on [`API_VERSION`]
/// (an explicit per-endpoint version constant, not a global bump).
pub const BACKUP_API_VERSION: &str = "2026-07";

/// This crate's own version — exposed so an embedder (the `pg_pinecone`
/// extension) can fold it into its own `User-Agent` string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// `User-Agent` sent when [`Config::user_agent`] is `None` — identifies this
/// crate directly to Pinecone, per the SDK convention (see
/// docs.pinecone.io/integrations/build-integration/attribute-usage-to-your-integration).
fn default_user_agent() -> String {
    format!("pg_pinecone-client-{VERSION}; lang=rust; source_tag=pg_pinecone")
}

/// Synchronous Pinecone client bound to one `(control_plane_url, api_key)` pair.
///
/// Data-plane hosts are resolved through the control plane and cached per index
/// for the lifetime of the instance, so reuse a `Client` only for the same
/// control plane URL and API key — create a new one for a different pair.
pub struct Client {
    agent: Agent,
    config: Config,
    host_cache: Mutex<HashMap<String, String>>,
    /// Invoked between retry attempts (upsert/delete backoff,
    /// `wait_index_ready` polling). Defaults to [`std::thread::sleep`]; tests
    /// override it via [`Client::with_sleep`] to avoid real delays.
    sleep: Box<dyn Fn(Duration) + Send + Sync>,
}

#[derive(Deserialize, Default)]
struct IndexStatus {
    #[serde(default)]
    ready: bool,
}

#[derive(Deserialize)]
struct IndexDescription {
    host: String,
    #[serde(default)]
    dimension: u32,
    #[serde(default)]
    metric: String,
    #[serde(default)]
    vector_type: String,
    #[serde(default)]
    status: IndexStatus,
    // API version 2026-07 has shown this as a top-level sibling of `spec`;
    // earlier versions nest it under `spec.serverless` — `index_info` checks
    // both, in that order.
    #[serde(default)]
    read_capacity: Option<ApiReadCapacity>,
    #[serde(default)]
    spec: Option<ApiSpec>,
}

#[derive(Deserialize)]
struct DocIndexDescription {
    host: String,
    #[serde(default)]
    status: IndexStatus,
    #[serde(default)]
    schema: Value,
    #[serde(default)]
    deletion_protection: Option<String>,
    #[serde(default)]
    read_capacity: Option<Value>,
}

impl From<DocIndexDescription> for DocIndexInfo {
    fn from(desc: DocIndexDescription) -> Self {
        DocIndexInfo {
            host: normalize_scheme(&desc.host),
            ready: desc.status.ready,
            schema: desc.schema,
            deletion_protection: desc.deletion_protection,
            read_capacity: desc.read_capacity,
        }
    }
}

#[derive(Deserialize, Default)]
struct ApiSpec {
    #[serde(default)]
    serverless: Option<ApiServerlessSpec>,
}

#[derive(Deserialize, Default)]
struct ApiServerlessSpec {
    #[serde(default)]
    read_capacity: Option<ApiReadCapacity>,
}

#[derive(Deserialize, Default)]
struct ApiReadCapacity {
    #[serde(default)]
    mode: String,
    #[serde(default)]
    dedicated: Option<ApiDedicated>,
    #[serde(default)]
    status: Option<ApiReadCapacityStatus>,
}

#[derive(Deserialize, Default)]
struct ApiDedicated {
    #[serde(default)]
    node_type: Option<String>,
    #[serde(default)]
    manual: Option<ApiManual>,
}

#[derive(Deserialize, Default)]
struct ApiManual {
    #[serde(default)]
    shards: Option<u32>,
    #[serde(default)]
    replicas: Option<u32>,
}

#[derive(Deserialize, Default)]
struct ApiReadCapacityStatus {
    #[serde(default)]
    state: Option<String>,
}

#[derive(Deserialize)]
struct QueryResponse {
    #[serde(default)]
    matches: Vec<ApiMatch>,
}

#[derive(Deserialize)]
struct ApiMatch {
    id: String,
    score: f32,
    metadata: Option<Value>,
}

#[derive(Deserialize, Default)]
struct ListVectorsResponse {
    #[serde(default)]
    vectors: Vec<ListedVectorId>,
    #[serde(default)]
    pagination: Option<VectorsListPagination>,
}

#[derive(Deserialize)]
struct ListedVectorId {
    id: String,
}

#[derive(Deserialize)]
struct VectorsListPagination {
    next: Option<String>,
}

#[derive(Deserialize, Default)]
struct UpsertCountResponse {
    #[serde(rename = "upsertedCount", default)]
    upserted_count: u64,
}

#[derive(Deserialize, Default)]
struct EmbedResponse {
    #[serde(default)]
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    /// Absent for sparse models, which this client does not support.
    values: Option<Vec<f32>>,
}

#[derive(Deserialize, Default)]
struct RerankResponse {
    #[serde(default)]
    data: Vec<ApiRerankResult>,
}

#[derive(Deserialize)]
struct ApiRerankResult {
    index: usize,
    score: f32,
}

#[derive(Deserialize, Default)]
struct DocUpsertCountResponse {
    #[serde(default)]
    upserted_count: u64,
}

#[derive(Deserialize, Default)]
struct FetchVectorsResponse {
    #[serde(default)]
    vectors: HashMap<String, ApiFetchedVector>,
}

#[derive(Deserialize, Default)]
struct ApiFetchedVector {
    #[serde(default)]
    values: Vec<f32>,
    #[serde(default)]
    metadata: Option<Value>,
}

/// `POST /vectors/fetch_by_metadata` response — same `vectors` object shape
/// as [`FetchVectorsResponse`] (reuses [`ApiFetchedVector`]) plus a
/// [`ListVectorsResponse`]-style pagination envelope (reuses
/// [`VectorsListPagination`]). Live-verified 2026-09-12 — see
/// [`crate::types::FetchByMetadataRequest`]'s rustdoc.
#[derive(Deserialize, Default)]
struct FetchByMetadataResponse {
    #[serde(default)]
    vectors: HashMap<String, ApiFetchedVector>,
    #[serde(default)]
    pagination: Option<VectorsListPagination>,
}

#[derive(Deserialize, Default)]
struct DescribeIndexStatsResponse {
    #[serde(rename = "totalVectorCount", default)]
    total_vector_count: u64,
    #[serde(default)]
    namespaces: HashMap<String, ApiNamespaceStats>,
}

#[derive(Deserialize, Default)]
struct ApiNamespaceStats {
    #[serde(rename = "vectorCount", default)]
    vector_count: u64,
}

/// A `uint64` Pinecone serializes as a decimal string (OAS `format: uint64`)
/// that may also arrive as a plain number: either parses, anything else
/// (absent, null, garbage, negative) is `None` instead of a failed response.
fn lenient_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(match Option::<Value>::deserialize(deserializer)? {
        Some(Value::Number(n)) => n.as_u64(),
        Some(Value::String(s)) => s.parse().ok(),
        _ => None,
    })
}

/// `NamespaceDescription` (`GET /namespaces`, `GET /namespaces/{namespace}`).
#[derive(Deserialize, Default)]
struct ApiNamespace {
    #[serde(default)]
    name: String,
    #[serde(default, deserialize_with = "lenient_u64")]
    record_count: Option<u64>,
    #[serde(default, deserialize_with = "lenient_u64")]
    size_bytes: Option<u64>,
    #[serde(default)]
    schema: Option<Value>,
    /// `{"fields": ["genre", ...]}`.
    #[serde(default)]
    indexed_fields: Option<Value>,
}

impl From<ApiNamespace> for NamespaceInfo {
    fn from(api: ApiNamespace) -> Self {
        let indexed_fields = api
            .indexed_fields
            .as_ref()
            .and_then(|v| v.get("fields"))
            .and_then(Value::as_array)
            .map(|fields| {
                fields
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        NamespaceInfo {
            name: api.name,
            record_count: api.record_count,
            size_bytes: api.size_bytes,
            indexed_fields,
            schema: api.schema,
        }
    }
}

#[derive(Deserialize, Default)]
struct ListNamespacesResponse {
    #[serde(default)]
    namespaces: Vec<ApiNamespace>,
    #[serde(default)]
    pagination: Option<VectorsListPagination>,
}

/// Backup control-plane response shape — snake_case, like the rest of the
/// classic (`2026-04`-generation) control plane (`/indexes`'s own
/// `vector_type`/`deletion_protection`); unlike the data-plane vector API's
/// camelCase. Assumed from Pinecone's docs, which show no full example
/// payload — validate against a live `create_backup`/`describe_backup` call
/// before relying on any field this doesn't parse ending up silently
/// defaulted away.
#[derive(Deserialize, Default)]
struct ApiBackupInfo {
    #[serde(default)]
    backup_id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    status: String,
    #[serde(default)]
    source_index_name: Option<String>,
    #[serde(default)]
    dimension: Option<u32>,
    #[serde(default)]
    record_count: Option<u64>,
    #[serde(default)]
    namespace_count: Option<u64>,
    #[serde(default)]
    size_bytes: Option<u64>,
    #[serde(default)]
    created_at: Option<String>,
}

impl From<ApiBackupInfo> for BackupInfo {
    fn from(api: ApiBackupInfo) -> Self {
        BackupInfo {
            backup_id: api.backup_id,
            name: api.name,
            status: api.status,
            source_index_name: api.source_index_name,
            dimension: api.dimension,
            record_count: api.record_count,
            namespace_count: api.namespace_count,
            size_bytes: api.size_bytes,
            created_at: api.created_at,
        }
    }
}

/// A page of `GET /indexes/{name}/backups` or `GET /backups` — same
/// `{"data": [...], "pagination": {"next": ...}}` envelope
/// [`ListVectorsResponse`]'s data-plane cousin uses, assumed by analogy
/// (unconfirmed against a live payload — see [`ApiBackupInfo`]'s rustdoc).
#[derive(Deserialize, Default)]
struct BackupListResponse {
    #[serde(default)]
    data: Vec<ApiBackupInfo>,
    #[serde(default)]
    pagination: Option<VectorsListPagination>,
}

#[derive(Deserialize, Default)]
struct DocSearchResponse {
    #[serde(default)]
    matches: Vec<ApiDocMatch>,
}

#[derive(Deserialize)]
struct ApiDocMatch {
    #[serde(rename = "_id")]
    id: String,
    #[serde(rename = "_score")]
    score: f32,
    /// Everything else in the match object, i.e. the document's fields with
    /// `_id`/`_score` stripped out.
    #[serde(flatten)]
    doc: Value,
}

/// Maximum inputs per `/embed` call (Pinecone's documented batch limit).
const EMBED_CHUNK_SIZE: usize = 96;

/// Maximum documents per `documents/upsert` call (Pinecone's documented limit).
const DOC_UPSERT_CHUNK_SIZE: usize = 1000;

/// Approximate maximum serialized bytes (documents) per `documents/upsert`
/// call — same rationale as `pinecone.promote`'s/the sync-trigger's
/// vector-upsert batching budget (`pg_pinecone/src/promote.rs`,
/// `pg_pinecone/src/sync_trigger.rs`).
const DOC_UPSERT_MAX_BYTES: usize = 1_500_000;

/// Maximum ids per `documents/delete` call (Pinecone's documented limit).
const DOC_DELETE_CHUNK_SIZE: usize = 1000;

/// Maximum attempts for the vector/document upserts and `delete_docs` when
/// the server returns 429 or 5xx.
const UPSERT_MAX_ATTEMPTS: u32 = 3;

/// Backoff before attempt 2 and attempt 3 respectively, used when the
/// response carries no `Retry-After` header.
const UPSERT_BACKOFF: [Duration; 2] = [Duration::from_millis(500), Duration::from_secs(2)];

/// Upper bound applied to a server-supplied `Retry-After` (seconds).
const RETRY_AFTER_CAP: Duration = Duration::from_secs(10);

/// `limit` sent on every `/vectors/list` page — Pinecone's documented maximum
/// per call.
const LIST_VECTORS_PAGE_LIMIT: u32 = 100;

/// Safety bound on how many `/vectors/list` pages [`Client::list_vector_ids`]
/// will fetch in one call, so a client bug (or a server that never stops
/// returning `pagination.next`) can't loop forever inside a caller like
/// `ambulkdelete`, which runs synchronously inside a live `VACUUM`. Generous
/// (10M ids at 100/page) — this is a defensive cap, not a normal limit.
const LIST_VECTORS_MAX_PAGES: usize = 100_000;

/// Maximum ids `Client::fetch_vectors` sends per `/vectors/fetch` call —
/// well under Pinecone's documented 1000-id/call hard cap, as a response-size
/// safety margin (768-1536-dimension fp32 vectors serialize to several KB
/// each; Pinecone documents no response-size cap, so this stays
/// conservative until a [live] probe pins one down).
const FETCH_VECTORS_CHUNK_SIZE: usize = 200;

/// `limit` sent on a `/vectors/fetch_by_metadata` call when
/// [`FetchByMetadataRequest::limit`] is `None` — Pinecone's documented
/// per-page maximum.
const FETCH_BY_METADATA_PAGE_LIMIT: u32 = 10_000;

/// `limit` sent on every backup-list page (`Client::list_backups`/
/// `list_project_backups`) — Pinecone's documented per-page maximum.
const BACKUPS_PAGE_LIMIT: u32 = 100;

/// `limit` sent on every [`Client::list_all_namespaces`] page — Pinecone's
/// documented per-page maximum.
const LIST_NAMESPACES_PAGE_LIMIT: u32 = 100;

/// Safety bound on pages [`Client::list_all_namespaces`] fetches in one call
/// (10M namespaces at 100 per page, far past the 1M plan cap) — a defensive
/// stop for a server that never ends its pagination, since the walk runs
/// synchronously inside a Postgres backend.
const LIST_NAMESPACES_MAX_PAGES: usize = 100_000;

impl Client {
    pub fn new(cfg: Config) -> Self {
        Self::with_sleep(cfg, std::thread::sleep)
    }

    /// Like [`Client::new`], but overrides the sleep invoked between retries
    /// (upsert/delete backoff, `wait_index_ready` polling) — tests use this
    /// to avoid real delays.
    pub fn with_sleep(cfg: Config, sleep: impl Fn(Duration) + Send + Sync + 'static) -> Self {
        let user_agent = cfg.user_agent.clone().unwrap_or_else(default_user_agent);
        let agent_config = Agent::config_builder()
            .timeout_global(Some(cfg.timeout))
            .http_status_as_error(false)
            .user_agent(user_agent)
            .build();
        let connector = DefaultConnector::default().chain(ResumeReads);
        Self {
            agent: Agent::with_parts(agent_config, connector, DefaultResolver::default()),
            config: cfg,
            host_cache: Mutex::new(HashMap::new()),
            sleep: Box::new(sleep),
        }
    }

    /// Runs a vector query against `req.index`.
    ///
    /// On a data-plane transport failure the cached host is evicted and
    /// re-resolved exactly once before the error is returned; HTTP error
    /// responses are never retried. A 404 (a namespace nothing was ever
    /// written to) is an empty result, not an error.
    pub fn query(&self, req: &QueryRequest) -> Result<Vec<Match>, Error> {
        self.with_host_retry(&req.index, API_VERSION, |host| self.query_host(host, req))
    }

    /// Upserts ready-made vectors into standard index `req.index`. Returns the
    /// number of vectors Pinecone reports as upserted. An empty batch is a pure
    /// no-op (`Ok(0)`, no HTTP call). Otherwise, same retry and missing-key
    /// policy as [`Client::query`].
    pub fn upsert_vectors(&self, req: &UpsertVectorsRequest) -> Result<u64, Error> {
        if req.vectors.is_empty() {
            return Ok(0);
        }
        self.with_host_retry(&req.index, API_VERSION, |host| {
            self.upsert_vectors_host(host, req)
        })
    }

    /// Embeds `req.inputs` via the Pinecone Inference API, posting straight to
    /// the control plane's `/embed` (no index-host resolution). Returns one
    /// vector per input, in input order; inputs are chunked at
    /// [`EMBED_CHUNK_SIZE`] per call and the results concatenated. An empty
    /// batch is a pure no-op (`Ok(vec![])`, no HTTP call). Otherwise, same
    /// missing-key policy as [`Client::query`]. Errors if the model is sparse
    /// (only dense embedding models are supported).
    pub fn embed(&self, req: &EmbedRequest) -> Result<Vec<Vec<f32>>, Error> {
        if req.inputs.is_empty() {
            return Ok(vec![]);
        }
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let mut embeddings = Vec::with_capacity(req.inputs.len());
        for chunk in req.inputs.chunks(EMBED_CHUNK_SIZE) {
            embeddings.extend(self.embed_chunk(&req.model, chunk, &req.input_type)?);
        }
        Ok(embeddings)
    }

    fn embed_chunk(
        &self,
        model: &str,
        inputs: &[String],
        input_type: &str,
    ) -> Result<Vec<Vec<f32>>, Error> {
        let url = format!(
            "{}/embed",
            self.config.control_plane_url.trim_end_matches('/')
        );
        let body = json!({
            "model": model,
            "parameters": {"input_type": input_type},
            "inputs": inputs.iter().map(|text| json!({"text": text})).collect::<Vec<_>>(),
        });
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .send_json(&body)
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => {
                let parsed: EmbedResponse = read_json(&mut resp)?;
                if parsed.data.len() != inputs.len() {
                    return Err(Error::Api {
                        status: 200,
                        message: format!(
                            "expected {} embeddings, got {}",
                            inputs.len(),
                            parsed.data.len()
                        ),
                    });
                }
                parsed
                    .data
                    .into_iter()
                    .map(|d| {
                        d.values.ok_or_else(|| Error::BadRequest {
                            message: "only dense embedding models are supported (response had no `values`, likely a sparse model)".to_string(),
                        })
                    })
                    .collect()
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Reranks `req.documents` against `req.query` via the Pinecone Inference
    /// API, posting straight to the control plane's `/rerank` (no index-host
    /// resolution). Results are returned in response order. An empty batch is
    /// a pure no-op (`Ok(vec![])`, no HTTP call). Otherwise, same missing-key
    /// policy as [`Client::query`].
    pub fn rerank(&self, req: &RerankRequest) -> Result<Vec<RerankResult>, Error> {
        if req.documents.is_empty() {
            return Ok(vec![]);
        }
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let url = format!(
            "{}/rerank",
            self.config.control_plane_url.trim_end_matches('/')
        );
        let mut body = json!({
            "model": req.model,
            "query": req.query,
            "documents": req.documents.iter().map(|text| json!({"text": text})).collect::<Vec<_>>(),
            "return_documents": false,
        });
        if let Some(top_n) = req.top_n {
            body["top_n"] = json!(top_n);
        }
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .send_json(&body)
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => {
                let parsed: RerankResponse = read_json(&mut resp)?;
                Ok(parsed
                    .data
                    .into_iter()
                    .map(|d| RerankResult {
                        index: d.index,
                        score: d.score,
                    })
                    .collect())
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Creates a standard (non-integrated) index and returns its metadata.
    /// Posts straight to the control plane's `/indexes` (no index-host
    /// resolution — there is no host yet). `req.spec` is sent verbatim as the
    /// `spec` object (e.g. `{"serverless": {...}}` or `{"byoc": {...}}`).
    /// Returns [`Error::Conflict`] if an index with that name already exists.
    pub fn create_index(&self, req: &CreateIndexRequest) -> Result<IndexInfo, Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let url = format!(
            "{}/indexes",
            self.config.control_plane_url.trim_end_matches('/')
        );
        let body = json!({
            "name": req.name,
            "vector_type": "dense",
            "dimension": req.dimension,
            "metric": req.metric,
            "spec": req.spec,
            "deletion_protection": "disabled",
        });
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .send_json(&body)
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            201 => {
                let desc: IndexDescription = read_json(&mut resp)?;
                Ok(index_info(desc))
            }
            400 | 422 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            409 => Err(Error::Conflict {
                name: req.name.clone(),
            }),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Fetches current metadata for `name` from the control plane, sent with
    /// the `2026-04` header. Reused by [`Client::wait_index_ready`] and by
    /// callers validating an existing index's dimension/metric/vector type
    /// before use.
    pub fn describe_index(&self, name: &str) -> Result<IndexInfo, Error> {
        self.describe_index_version(name, API_VERSION)
    }

    /// Like [`Client::describe_index`], but sends `version` as
    /// `X-Pinecone-Api-Version` — the doc-index readiness poll
    /// ([`Client::wait_doc_index_ready`]) needs the `2026-07` header instead
    /// of the classic-index `2026-04` default.
    fn describe_index_version(
        &self,
        name: &str,
        version: &'static str,
    ) -> Result<IndexInfo, Error> {
        self.describe_index_as::<IndexDescription>(name, version)
            .map(index_info)
    }

    /// Describes document-schema index `name` (`2026-07` header), including
    /// its schema. Returns [`Error::IndexNotFound`] if it doesn't exist.
    pub fn describe_doc_index(&self, name: &str) -> Result<DocIndexInfo, Error> {
        self.describe_index_as::<DocIndexDescription>(name, DOC_API_VERSION)
            .map(Into::into)
    }

    fn describe_index_as<T: DeserializeOwned>(
        &self,
        name: &str,
        version: &'static str,
    ) -> Result<T, Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let url = format!(
            "{}/indexes/{}",
            self.config.control_plane_url.trim_end_matches('/'),
            index_segment(name)
        );
        let mut resp = self
            .agent
            .get(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", version)
            .call()
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => read_json(&mut resp),
            404 => Err(Error::IndexNotFound {
                index: name.to_string(),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Polls [`Client::describe_index`] every `poll_every` until `status.ready`
    /// or `deadline` elapses, whichever comes first. Returns [`Error::Timeout`]
    /// (carrying `deadline`, in ms) if the index never becomes ready in time.
    /// Sent with the `2026-04` header; see [`Client::wait_doc_index_ready`]
    /// for the `2026-07` equivalent.
    pub fn wait_index_ready(
        &self,
        name: &str,
        deadline: Duration,
        poll_every: Duration,
    ) -> Result<IndexInfo, Error> {
        self.wait_ready_version(name, deadline, poll_every, API_VERSION)
    }

    /// Like [`Client::wait_index_ready`], but polls with the `2026-07` header
    /// — used by `pinecone.create_doc_index` to wait for a newly created
    /// document-schema index to become ready.
    pub fn wait_doc_index_ready(
        &self,
        name: &str,
        deadline: Duration,
        poll_every: Duration,
    ) -> Result<IndexInfo, Error> {
        self.wait_ready_version(name, deadline, poll_every, DOC_API_VERSION)
    }

    fn wait_ready_version(
        &self,
        name: &str,
        deadline: Duration,
        poll_every: Duration,
        version: &'static str,
    ) -> Result<IndexInfo, Error> {
        let start = Instant::now();
        loop {
            let info = self.describe_index_version(name, version)?;
            if info.ready {
                return Ok(info);
            }
            if start.elapsed() >= deadline {
                return Err(Error::Timeout {
                    ms: timeout_ms(deadline),
                });
            }
            (self.sleep)(poll_every);
        }
    }

    /// Deletes index `name`. No-op on the data plane — this is a control-plane
    /// call only. Returns [`Error::IndexNotFound`] if it doesn't exist. A 403
    /// (as opposed to 401) means the request authenticated fine but was
    /// refused — in practice this is Pinecone's `deletion_protection` guard,
    /// so it maps to [`Error::BadRequest`] naming that as the likely cause
    /// rather than [`Error::Auth`].
    pub fn delete_index(&self, name: &str) -> Result<(), Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let url = format!(
            "{}/indexes/{}",
            self.config.control_plane_url.trim_end_matches('/'),
            index_segment(name)
        );
        let mut resp = self
            .agent
            .delete(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .call()
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 | 202 => {
                drain(&mut resp);
                Ok(())
            }
            404 => Err(Error::IndexNotFound {
                index: name.to_string(),
            }),
            401 => Err(Error::Auth),
            403 => Err(Error::BadRequest {
                message: deletion_protection_forbidden(name, &mut resp),
            }),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Deletes vectors from standard index `req.index`. Exactly one of
    /// `req.ids`, `req.delete_all` or `req.filter` must be set, and a filter
    /// must be a non-empty object; anything else is rejected client-side with
    /// [`Error::BadRequest`] before any network call. A 404 (a namespace
    /// nothing was ever written to) is success. Same retry and missing-key
    /// policy as [`Client::query`] otherwise.
    pub fn delete_vectors(&self, req: &DeleteVectorsRequest) -> Result<(), Error> {
        let selectors = usize::from(req.ids.is_some())
            + usize::from(req.delete_all)
            + usize::from(req.filter.is_some());
        if selectors != 1 {
            return Err(Error::BadRequest {
                message: "delete_vectors requires exactly one of `ids`, `delete_all` or `filter`"
                    .to_string(),
            });
        }
        if let Some(filter) = &req.filter
            && !matches!(filter, Value::Object(m) if !m.is_empty())
        {
            return Err(Error::BadRequest {
                message: "delete_vectors requires a non-empty object as `filter`".to_string(),
            });
        }
        self.with_host_retry(&req.index, API_VERSION, |host| {
            self.delete_vectors_host(host, req)
        })
    }

    /// Lists every vector id currently in standard index `req.namespace`,
    /// walking `/vectors/list`'s pagination internally
    /// ([`LIST_VECTORS_PAGE_LIMIT`] ids per page, up to [`LIST_VECTORS_MAX_PAGES`]
    /// pages) and returning the full, concatenated id list. A namespace that
    /// doesn't exist (or has never been written to) returns an empty list,
    /// not an error — same idempotent-on-absence convention as
    /// [`Client::delete_vectors`]'s 404 handling. Same retry and missing-key
    /// policy as [`Client::query`] otherwise (the whole paginated walk is
    /// retried once, from page 1, on a data-plane transport failure).
    pub fn list_vector_ids(&self, req: &ListVectorsRequest) -> Result<Vec<String>, Error> {
        self.with_host_retry(&req.index, API_VERSION, |host| {
            self.list_vector_ids_host(host, req)
        })
    }

    /// Creates a document-schema (full-text-search) index. Posts straight to
    /// the control plane's `/indexes` (no index-host resolution) with the
    /// `2026-07` header. `req.schema` is sent verbatim as the `schema` object
    /// (the `{"fields": {...}}` shape). Returns [`Error::Conflict`] if an
    /// index with that name already exists.
    pub fn create_doc_index(&self, req: &CreateDocIndexRequest) -> Result<(), Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let url = format!(
            "{}/indexes",
            self.config.control_plane_url.trim_end_matches('/')
        );
        let body = json!({
            "name": req.name,
            "deployment": {
                "deployment_type": "managed",
                "cloud": req.cloud,
                "region": req.region,
            },
            "schema": req.schema,
            "deletion_protection": "disabled",
        });
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", DOC_API_VERSION)
            .send_json(&body)
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            201 => {
                drain(&mut resp);
                Ok(())
            }
            400 | 422 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            409 => Err(Error::Conflict {
                name: req.name.clone(),
            }),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Deletes document-schema index `name`. Separate from [`Client::delete_index`]
    /// because it sends the `2026-07` header — same path (and same 403-vs-401
    /// deletion_protection handling) otherwise.
    pub fn delete_doc_index(&self, name: &str) -> Result<(), Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let url = format!(
            "{}/indexes/{}",
            self.config.control_plane_url.trim_end_matches('/'),
            index_segment(name)
        );
        let mut resp = self
            .agent
            .delete(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", DOC_API_VERSION)
            .call()
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 | 202 => {
                drain(&mut resp);
                Ok(())
            }
            404 => Err(Error::IndexNotFound {
                index: name.to_string(),
            }),
            401 => Err(Error::Auth),
            403 => Err(Error::BadRequest {
                message: deletion_protection_forbidden(name, &mut resp),
            }),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Upserts documents into document-schema index `req.index`, chunking at
    /// [`DOC_UPSERT_CHUNK_SIZE`] documents or ~[`DOC_UPSERT_MAX_BYTES`]
    /// serialized bytes per call — whichever comes first — and summing each
    /// response's `upserted_count`. An empty batch is a pure no-op (`Ok(0)`,
    /// no HTTP call). Otherwise, same retry and missing-key policy as
    /// [`Client::query`], resolved with the `2026-07` header; each chunk also
    /// retries 429/5xx up to 3 attempts, honoring `Retry-After`.
    ///
    /// Chunks are sent sequentially, not atomically: if a later chunk fails,
    /// earlier chunks have already been applied to the index.
    pub fn upsert_docs(&self, req: &UpsertDocsRequest) -> Result<u64, Error> {
        let documents: Vec<&Value> = req.documents.iter().collect();
        self.upsert_documents(&req.index, req.namespace.as_deref(), &documents)
    }

    /// [`Client::upsert_docs`] over documents the caller keeps: nothing is cloned into a request
    /// (a batch of big documents would otherwise exist two more times while it is sent).
    pub fn upsert_documents(
        &self,
        index: &str,
        namespace: Option<&str>,
        documents: &[&Value],
    ) -> Result<u64, Error> {
        if documents.is_empty() {
            return Ok(0);
        }
        let mut total = 0u64;
        let mut start = 0usize;
        let mut chunk_bytes = 0usize;
        for (i, doc) in documents.iter().enumerate() {
            let size = doc.to_string().len();
            if i > start
                && ((i - start) + 1 > DOC_UPSERT_CHUNK_SIZE
                    || chunk_bytes + size > DOC_UPSERT_MAX_BYTES)
            {
                total += self.upsert_docs_chunk(index, namespace, &documents[start..i])?;
                start = i;
                chunk_bytes = 0;
            }
            chunk_bytes += size;
        }
        total += self.upsert_docs_chunk(index, namespace, &documents[start..])?;
        Ok(total)
    }

    fn upsert_docs_chunk(
        &self,
        index: &str,
        namespace: Option<&str>,
        documents: &[&Value],
    ) -> Result<u64, Error> {
        self.with_host_retry(index, DOC_API_VERSION, |host| {
            self.upsert_docs_host(host, namespace, documents)
        })
    }

    fn upsert_docs_host(
        &self,
        host: &str,
        namespace: Option<&str>,
        documents: &[&Value],
    ) -> Result<u64, Error> {
        #[derive(serde::Serialize)]
        struct Body<'a> {
            documents: &'a [&'a Value],
        }
        let namespace = namespace_segment(namespace);
        let url = format!(
            "{}/namespaces/{namespace}/documents/upsert",
            host.trim_end_matches('/')
        );
        let mut resp = self.post_json_retrying(&url, DOC_API_VERSION, &Body { documents })?;
        match resp.status().as_u16() {
            202 => {
                let parsed: DocUpsertCountResponse = read_json(&mut resp)?;
                Ok(parsed.upserted_count)
            }
            400 | 422 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 => Err(Error::Auth),
            403 => Err(forbidden(&mut resp)),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Deletes the selected documents from document-schema index `req.index`.
    /// Ids go out in sequential, non-atomic chunks of at most 1000 (an empty
    /// list makes no HTTP call); a failing chunk leaves later ones unsent. A
    /// data-plane 404 (namespace never written to) is `Ok`. Same retry,
    /// missing-key and header policy as [`Client::upsert_docs`].
    pub fn delete_docs(&self, req: &DeleteDocsRequest) -> Result<(), Error> {
        match &req.selector {
            DocDeleteSelector::Ids(ids) => ids
                .chunks(DOC_DELETE_CHUNK_SIZE)
                .try_for_each(|chunk| self.delete_docs_call(req, json!({ "ids": chunk }))),
            DocDeleteSelector::All => self.delete_docs_call(req, json!({ "delete_all": true })),
        }
    }

    fn delete_docs_call(&self, req: &DeleteDocsRequest, body: Value) -> Result<(), Error> {
        self.with_host_retry(&req.index, DOC_API_VERSION, |host| {
            self.delete_docs_host(host, req.namespace.as_deref(), &body)
        })
    }

    fn delete_docs_host(
        &self,
        host: &str,
        namespace: Option<&str>,
        body: &Value,
    ) -> Result<(), Error> {
        let namespace = namespace_segment(namespace);
        let url = format!(
            "{}/namespaces/{namespace}/documents/delete",
            host.trim_end_matches('/')
        );
        let mut resp = self.post_json_retrying(&url, DOC_API_VERSION, body)?;
        match resp.status().as_u16() {
            // Assumed to mirror `delete_vectors` (404 for a namespace never
            // written to); not yet observed live on document indexes.
            200 | 202 | 404 => {
                drain(&mut resp);
                Ok(())
            }
            400 | 422 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Runs a full-text/hybrid search against document-schema index
    /// `req.index`. Same retry, missing-key and 404 (empty result) policy as
    /// [`Client::query`], resolved with the `2026-07` header.
    pub fn search_docs(&self, req: &SearchDocsRequest) -> Result<Vec<DocMatch>, Error> {
        self.with_host_retry(&req.index, DOC_API_VERSION, |host| {
            self.search_docs_host(host, req)
        })
    }

    fn search_docs_host(
        &self,
        host: &str,
        req: &SearchDocsRequest,
    ) -> Result<Vec<DocMatch>, Error> {
        let namespace = namespace_segment(req.namespace.as_deref());
        let url = format!(
            "{}/namespaces/{namespace}/documents/search",
            host.trim_end_matches('/')
        );
        let mut body = json!({
            "include_fields": req.include_fields,
            "score_by": req.score_by,
            "top_k": req.top_k,
        });
        if let Some(filter) = &req.filter {
            body["filter"] = filter.clone();
        }
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", DOC_API_VERSION)
            .send_json(&body)
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => {
                let parsed: DocSearchResponse = read_json(&mut resp)?;
                Ok(parsed
                    .matches
                    .into_iter()
                    .map(|m| DocMatch {
                        id: m.id,
                        score: m.score,
                        doc: m.doc,
                    })
                    .collect())
            }
            // Same never-written-namespace rule as `query_host`.
            404 => {
                drain(&mut resp);
                Ok(Vec::new())
            }
            400 | 422 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Resolves the data-plane host for `index` and runs `call` against it. On a
    /// transport failure the cached host is evicted and re-resolved exactly once
    /// before retrying; HTTP error responses are never retried. Returns
    /// [`Error::MissingApiKey`] before any network call when no key is set.
    /// `version` is the `X-Pinecone-Api-Version` sent on the host-resolution
    /// (`GET /indexes/{index}`) call only — it has no bearing on `call`, which
    /// sets its own header.
    fn with_host_retry<T>(
        &self,
        index: &str,
        version: &'static str,
        call: impl Fn(&str) -> Result<T, Error>,
    ) -> Result<T, Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let host = self.resolve_host(index, version)?;
        match call(&host) {
            Err(Error::Transport { .. }) => {
                self.evict(index);
                let host = self.resolve_host(index, version)?;
                call(&host)
            }
            result => result,
        }
    }

    fn resolve_host(&self, index: &str, version: &'static str) -> Result<String, Error> {
        if let Some(host) = self.cache().get(index).cloned() {
            return Ok(host);
        }
        let url = format!(
            "{}/indexes/{}",
            self.config.control_plane_url.trim_end_matches('/'),
            index_segment(index)
        );
        let mut resp = self
            .agent
            .get(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", version)
            .call()
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => {
                let desc: IndexDescription = read_json(&mut resp)?;
                let host = normalize_scheme(&desc.host);
                self.cache().insert(index.to_string(), host.clone());
                Ok(host)
            }
            404 => Err(Error::IndexNotFound {
                index: index.to_string(),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    fn query_host(&self, host: &str, req: &QueryRequest) -> Result<Vec<Match>, Error> {
        let url = format!("{}/query", host.trim_end_matches('/'));
        let mut body = json!({
            "vector": req.vector,
            "topK": req.top_k,
            "includeMetadata": req.include_metadata,
        });
        if let Some(namespace) = &req.namespace {
            body["namespace"] = json!(namespace);
        }
        if let Some(filter) = &req.filter {
            body["filter"] = filter.clone();
        }
        if let Some(scan_factor) = req.scan_factor {
            body["scan_factor"] = json!(scan_factor);
        }
        if let Some(max_candidates) = req.max_candidates {
            body["max_candidates"] = json!(max_candidates);
        }
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .send_json(&body)
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => {
                let parsed: QueryResponse = read_json(&mut resp)?;
                Ok(parsed
                    .matches
                    .into_iter()
                    .map(|m| Match {
                        id: m.id,
                        score: m.score,
                        metadata: m.metadata,
                    })
                    .collect())
            }
            // A namespace nothing was ever written to answers 404 "Namespace not
            // found": no matches, not an error (a partitioned table's tenant
            // with no rows yet queries exactly this).
            404 => {
                drain(&mut resp);
                Ok(Vec::new())
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    fn upsert_vectors_host(&self, host: &str, req: &UpsertVectorsRequest) -> Result<u64, Error> {
        // Serialized from the f32s directly: through `json!` each value widens to an f64 and
        // prints with twice the digits.
        #[derive(serde::Serialize)]
        struct VectorBody<'a> {
            id: &'a str,
            values: &'a [f32],
            #[serde(skip_serializing_if = "Option::is_none")]
            metadata: Option<&'a Value>,
        }
        #[derive(serde::Serialize)]
        struct Body<'a> {
            vectors: Vec<VectorBody<'a>>,
            #[serde(skip_serializing_if = "Option::is_none")]
            namespace: Option<&'a str>,
        }
        let url = format!("{}/vectors/upsert", host.trim_end_matches('/'));
        let body = Body {
            vectors: req
                .vectors
                .iter()
                .map(|v| VectorBody {
                    id: &v.id,
                    values: &v.values,
                    metadata: v.metadata.as_ref(),
                })
                .collect(),
            namespace: req.namespace.as_deref(),
        };
        let mut resp = self.post_json_retrying(&url, API_VERSION, &body)?;
        match resp.status().as_u16() {
            200 => {
                let parsed: UpsertCountResponse = read_json(&mut resp)?;
                Ok(parsed.upserted_count)
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 => Err(Error::Auth),
            403 => Err(forbidden(&mut resp)),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// POSTs `body` as JSON, retrying on 429/5xx up to [`UPSERT_MAX_ATTEMPTS`]
    /// total attempts. Backoff between attempts uses the response's
    /// `Retry-After` (seconds, capped at [`RETRY_AFTER_CAP`]) when present,
    /// else [`UPSERT_BACKOFF`]. Returns the last response for the caller to
    /// map. HTTP transport failures (not HTTP-status errors) are not retried
    /// here — that's `with_host_retry`'s job.
    fn post_json_retrying(
        &self,
        url: &str,
        version: &'static str,
        body: &impl serde::Serialize,
    ) -> Result<Response<Body>, Error> {
        // `send_json` pretty-prints, which puts every float of a vector on a line of its own.
        let bytes = serde_json::to_vec(body).map_err(|e| Error::BadRequest {
            message: format!("could not serialize the request: {e}"),
        })?;
        let mut attempt: u32 = 1;
        loop {
            let mut resp = self
                .agent
                .post(url)
                .header("Api-Key", self.config.api_key.as_str())
                .header("X-Pinecone-Api-Version", version)
                .content_type("application/json; charset=utf-8")
                .send(&bytes)
                .map_err(|e| self.map_transport(e, url))?;
            let status = resp.status().as_u16();
            let retryable = status == 429 || (500..600).contains(&status);
            if retryable && attempt < UPSERT_MAX_ATTEMPTS {
                let backoff = retry_after(&resp).unwrap_or(UPSERT_BACKOFF[(attempt - 1) as usize]);
                drain(&mut resp);
                (self.sleep)(backoff);
                attempt += 1;
                continue;
            }
            return Ok(resp);
        }
    }

    fn delete_vectors_host(&self, host: &str, req: &DeleteVectorsRequest) -> Result<(), Error> {
        let url = format!("{}/vectors/delete", host.trim_end_matches('/'));
        let mut body = json!({});
        if let Some(ids) = &req.ids {
            body["ids"] = json!(ids);
        }
        if req.delete_all {
            body["deleteAll"] = json!(true);
        }
        if let Some(filter) = &req.filter {
            body["filter"] = filter.clone();
        }
        if let Some(namespace) = &req.namespace {
            body["namespace"] = json!(namespace);
        }
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .send_json(&body)
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => {
                drain(&mut resp);
                Ok(())
            }
            // Pinecone returns 404 "Namespace not found" for a namespace no
            // vector was ever written to. Deletes are idempotent — nothing to
            // delete is success — and promote()'s rebuild path always hits
            // this on a brand-new namespace.
            404 => {
                drain(&mut resp);
                Ok(())
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    fn list_vector_ids_host(
        &self,
        host: &str,
        req: &ListVectorsRequest,
    ) -> Result<Vec<String>, Error> {
        let mut ids = Vec::new();
        let mut token: Option<String> = None;
        for _ in 0..LIST_VECTORS_MAX_PAGES {
            let page =
                self.list_vector_ids_page_host(host, req.namespace.as_deref(), token.as_deref())?;
            ids.extend(page.ids);
            match page.next_token {
                Some(next) => token = Some(next),
                None => return Ok(ids),
            }
        }
        Err(Error::BadRequest {
            message: format!(
                "list_vector_ids: exceeded {LIST_VECTORS_MAX_PAGES} pages of pagination without \
                 exhausting the namespace — aborting"
            ),
        })
    }

    /// Fetches exactly one `/vectors/list` page — the resumable-cursor
    /// primitive `pinecone.recover_from_remote`'s outbox job needs (one page per
    /// worker drain round, cursor persisted in the job's payload between
    /// rounds); [`Client::list_vector_ids`] instead walks every page
    /// internally via [`Client::list_vector_ids_host`], which now calls this
    /// once per page. Same missing-key/host-retry policy as
    /// [`Client::query`]; a namespace with no page at `req.pagination_token`
    /// (including one that's never been written to at all) returns an empty
    /// `ids` with `next_token: None`, not an error — same idempotent
    /// "nothing there" convention [`Client::delete_vectors`]'s 404 handling
    /// uses.
    pub fn list_vector_ids_page(
        &self,
        req: &ListVectorIdsPageRequest,
    ) -> Result<ListVectorIdsPage, Error> {
        self.with_host_retry(&req.index, API_VERSION, |host| {
            self.list_vector_ids_page_host(
                host,
                req.namespace.as_deref(),
                req.pagination_token.as_deref(),
            )
        })
    }

    fn list_vector_ids_page_host(
        &self,
        host: &str,
        namespace: Option<&str>,
        pagination_token: Option<&str>,
    ) -> Result<ListVectorIdsPage, Error> {
        let mut url = format!(
            "{}/vectors/list?limit={LIST_VECTORS_PAGE_LIMIT}",
            host.trim_end_matches('/')
        );
        if let Some(namespace) = namespace {
            url.push_str(&format!(
                "&namespace={}",
                utf8_percent_encode(namespace, QUERY_COMPONENT)
            ));
        }
        if let Some(t) = pagination_token {
            url.push_str(&format!(
                "&paginationToken={}",
                utf8_percent_encode(t, QUERY_COMPONENT)
            ));
        }
        let mut resp = self
            .agent
            .get(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .call()
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => {
                let parsed: ListVectorsResponse = read_json(&mut resp)?;
                let ids = parsed.vectors.into_iter().map(|v| v.id).collect();
                let next_token = parsed
                    .pagination
                    .and_then(|p| p.next)
                    .filter(|next| !next.is_empty());
                Ok(ListVectorIdsPage { ids, next_token })
            }
            // No namespace at all (never written to) — same idempotent
            // "nothing there" convention as delete_vectors_host's own 404.
            404 => {
                drain(&mut resp);
                Ok(ListVectorIdsPage::default())
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Fetches full vectors (values + metadata) for `req.ids` from standard
    /// index `req.index`, chunking internally at
    /// [`FETCH_VECTORS_CHUNK_SIZE`] ids/call (well under Pinecone's
    /// documented 1000-id/call hard cap — a response-size safety margin).
    /// An id Pinecone has no vector for is simply absent from the returned
    /// map, not an error — callers (e.g.
    /// `pinecone.recover_from_remote`) diff the response's keys against the
    /// requested ids to detect misses. An empty `req.ids` is a pure no-op
    /// (`Ok(HashMap::new())`, no HTTP call). Otherwise, same retry and
    /// missing-key policy as [`Client::query`].
    pub fn fetch_vectors(
        &self,
        req: &FetchVectorsRequest,
    ) -> Result<HashMap<String, Vector>, Error> {
        if req.ids.is_empty() {
            return Ok(HashMap::new());
        }
        self.with_host_retry(&req.index, API_VERSION, |host| {
            self.fetch_vectors_host(host, req.namespace.as_deref(), &req.ids)
        })
    }

    fn fetch_vectors_host(
        &self,
        host: &str,
        namespace: Option<&str>,
        ids: &[String],
    ) -> Result<HashMap<String, Vector>, Error> {
        let mut out = HashMap::with_capacity(ids.len());
        for chunk in ids.chunks(FETCH_VECTORS_CHUNK_SIZE) {
            let chunk_result = self.fetch_vectors_chunk(host, namespace, chunk)?;
            out.extend(chunk_result);
        }
        Ok(out)
    }

    fn fetch_vectors_chunk(
        &self,
        host: &str,
        namespace: Option<&str>,
        ids: &[String],
    ) -> Result<HashMap<String, Vector>, Error> {
        let mut parts: Vec<String> = ids
            .iter()
            .map(|id| format!("ids={}", utf8_percent_encode(id, QUERY_COMPONENT)))
            .collect();
        if let Some(namespace) = namespace {
            parts.push(format!(
                "namespace={}",
                utf8_percent_encode(namespace, QUERY_COMPONENT)
            ));
        }
        let url = format!(
            "{}/vectors/fetch?{}",
            host.trim_end_matches('/'),
            parts.join("&")
        );
        let mut resp = self
            .agent
            .get(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .call()
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => {
                let parsed: FetchVectorsResponse = read_json(&mut resp)?;
                Ok(parsed
                    .vectors
                    .into_iter()
                    .map(|(id, v)| {
                        (
                            id.clone(),
                            Vector {
                                id,
                                values: v.values,
                                metadata: v.metadata,
                            },
                        )
                    })
                    .collect())
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            // No namespace at all — same idempotent "nothing there"
            // convention as `list_vector_ids_page_host`'s own 404 handling.
            404 => {
                drain(&mut resp);
                Ok(HashMap::new())
            }
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Returns per-namespace and total vector counts for `index`
    /// (`POST /describe_index_stats`) — the backup gate's SYNC check and
    /// `pinecone.recover_from_remote`'s job accounting both need this. Same
    /// retry and missing-key policy as [`Client::query`].
    pub fn describe_index_stats(&self, index: &str) -> Result<IndexStats, Error> {
        self.with_host_retry(index, API_VERSION, |host| {
            self.describe_index_stats_host(host)
        })
    }

    fn describe_index_stats_host(&self, host: &str) -> Result<IndexStats, Error> {
        let url = format!("{}/describe_index_stats", host.trim_end_matches('/'));
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .send_json(json!({}))
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => {
                let parsed: DescribeIndexStatsResponse = read_json(&mut resp)?;
                Ok(IndexStats {
                    total_vector_count: parsed.total_vector_count,
                    namespaces: parsed
                        .namespaces
                        .into_iter()
                        .map(|(name, ns)| {
                            (
                                name,
                                NamespaceStats {
                                    vector_count: ns.vector_count,
                                },
                            )
                        })
                        .collect(),
                })
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Fetches one page of vectors (values + metadata) matching `req.filter`
    /// from standard index `req.index` (`POST /vectors/fetch_by_metadata`) —
    /// unlike [`Client::fetch_vectors`] (id-based, ≤1000 ids/call, no pagination),
    /// this is filter-based and paginated, with a documented 10,000-record
    /// page cap. `req.filter` must be non-empty — Pinecone documents no
    /// fetch-all via this endpoint — checked client-side, before any HTTP
    /// call. One page per call, like [`Client::list_vector_ids_page`] (not
    /// auto-walking): callers drive pagination via
    /// [`FetchByMetadataPage::next_token`]. Same retry and missing-key
    /// policy as [`Client::query`] otherwise.
    pub fn fetch_by_metadata(
        &self,
        req: &FetchByMetadataRequest,
    ) -> Result<FetchByMetadataPage, Error> {
        if matches!(&req.filter, Value::Null)
            || matches!(&req.filter, Value::Object(m) if m.is_empty())
        {
            return Err(Error::BadRequest {
                message: "fetch_by_metadata requires a non-empty filter — Pinecone has no \
                          fetch-all via this endpoint"
                    .to_string(),
            });
        }
        self.with_host_retry(&req.index, API_VERSION, |host| {
            self.fetch_by_metadata_host(host, req)
        })
    }

    fn fetch_by_metadata_host(
        &self,
        host: &str,
        req: &FetchByMetadataRequest,
    ) -> Result<FetchByMetadataPage, Error> {
        let url = format!("{}/vectors/fetch_by_metadata", host.trim_end_matches('/'));
        let mut body = json!({
            "filter": req.filter,
            "limit": req.limit.unwrap_or(FETCH_BY_METADATA_PAGE_LIMIT),
        });
        if let Some(namespace) = &req.namespace {
            body["namespace"] = json!(namespace);
        }
        if let Some(token) = &req.pagination_token {
            body["paginationToken"] = json!(token);
        }
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .send_json(&body)
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => {
                let parsed: FetchByMetadataResponse = read_json(&mut resp)?;
                let vectors = parsed
                    .vectors
                    .into_iter()
                    .map(|(id, v)| {
                        (
                            id.clone(),
                            Vector {
                                id,
                                values: v.values,
                                metadata: v.metadata,
                            },
                        )
                    })
                    .collect();
                let next_token = parsed
                    .pagination
                    .and_then(|p| p.next)
                    .filter(|next| !next.is_empty());
                Ok(FetchByMetadataPage {
                    vectors,
                    next_token,
                })
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            // No namespace at all — same idempotent "nothing there"
            // convention as `fetch_vectors_chunk`'s own 404 handling.
            404 => Ok(FetchByMetadataPage::default()),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Merges `req.metadata` into one existing vector's stored metadata
    /// (`POST /vectors/update`) without touching its `values`. See
    /// [`UpdateVectorMetadataRequest`]'s rustdoc for the merge-semantics
    /// caveat (documented, not live-verified by this client). Same retry and
    /// missing-key policy as [`Client::query`].
    pub fn update_vector_metadata(&self, req: &UpdateVectorMetadataRequest) -> Result<(), Error> {
        self.with_host_retry(&req.index, API_VERSION, |host| {
            self.update_vector_metadata_host(host, req)
        })
    }

    fn update_vector_metadata_host(
        &self,
        host: &str,
        req: &UpdateVectorMetadataRequest,
    ) -> Result<(), Error> {
        let url = format!("{}/vectors/update", host.trim_end_matches('/'));
        let mut body = json!({
            "id": req.id,
            "setMetadata": req.metadata,
        });
        if let Some(namespace) = &req.namespace {
            body["namespace"] = json!(namespace);
        }
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .send_json(&body)
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => {
                drain(&mut resp);
                Ok(())
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    // ---- Namespaces (data plane, serverless vector indexes, `API_VERSION`) ----
    // Not verified against document-schema indexes, which speak `DOC_API_VERSION`.

    /// Fetches one page of `index`'s namespaces (`GET /namespaces`), sorted by
    /// name; `req.prefix` narrows them to names starting with it. One page per
    /// call, like [`Client::list_vector_ids_page`]: callers drive pagination
    /// via [`ListNamespacesPage::next`], or use [`Client::list_all_namespaces`].
    /// Same retry and missing-key policy as [`Client::query`]; no status is
    /// special-cased beyond 400 and 401/403, so a 404 is [`Error::Api`].
    pub fn list_namespaces(
        &self,
        req: &ListNamespacesRequest,
    ) -> Result<ListNamespacesPage, Error> {
        self.with_host_retry(&req.index, API_VERSION, |host| {
            self.list_namespaces_host(host, req)
        })
    }

    /// Lists every namespace of `index` whose name starts with `prefix` (all of
    /// them for `None`), walking `GET /namespaces`'s pagination internally
    /// (100 per page, the API maximum, up to 100,000 pages). A token that does
    /// not advance, or a walk past the page cap, is an [`Error::Api`]. The
    /// whole walk is retried once, from page 1, on a data-plane transport
    /// failure.
    pub fn list_all_namespaces(
        &self,
        index: &str,
        prefix: Option<&str>,
    ) -> Result<Vec<NamespaceInfo>, Error> {
        self.with_host_retry(index, API_VERSION, |host| {
            let mut req = ListNamespacesRequest {
                index: index.to_string(),
                prefix: prefix.map(str::to_string),
                limit: Some(LIST_NAMESPACES_PAGE_LIMIT),
                pagination_token: None,
            };
            let mut all = Vec::new();
            for _ in 0..LIST_NAMESPACES_MAX_PAGES {
                let page = self.list_namespaces_host(host, &req)?;
                all.extend(page.namespaces);
                match page.next {
                    None => return Ok(all),
                    Some(next) if req.pagination_token.as_deref() == Some(next.as_str()) => {
                        return Err(Error::Api {
                            status: 200,
                            message: "list_all_namespaces: pagination token did not advance — \
                                      aborting"
                                .to_string(),
                        });
                    }
                    Some(next) => req.pagination_token = Some(next),
                }
            }
            Err(Error::Api {
                status: 200,
                message: format!(
                    "list_all_namespaces: exceeded {LIST_NAMESPACES_MAX_PAGES} pages of \
                     pagination without exhausting the index — aborting"
                ),
            })
        })
    }

    fn list_namespaces_host(
        &self,
        host: &str,
        req: &ListNamespacesRequest,
    ) -> Result<ListNamespacesPage, Error> {
        let mut params = Vec::new();
        if let Some(prefix) = &req.prefix {
            params.push(format!(
                "prefix={}",
                utf8_percent_encode(prefix, QUERY_COMPONENT)
            ));
        }
        if let Some(limit) = req.limit {
            params.push(format!("limit={limit}"));
        }
        if let Some(token) = &req.pagination_token {
            params.push(format!(
                "paginationToken={}",
                utf8_percent_encode(token, QUERY_COMPONENT)
            ));
        }
        let mut url = format!("{}/namespaces", host.trim_end_matches('/'));
        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }
        let mut resp = self
            .agent
            .get(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .call()
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => {
                let parsed: ListNamespacesResponse = read_json(&mut resp)?;
                Ok(ListNamespacesPage {
                    namespaces: parsed.namespaces.into_iter().map(Into::into).collect(),
                    next: parsed
                        .pagination
                        .and_then(|p| p.next)
                        .filter(|next| !next.is_empty()),
                })
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Describes one namespace of `index` (`GET /namespaces/{namespace}`).
    /// [`Error::NamespaceNotFound`] if it does not exist; an empty `namespace`
    /// is rejected client-side ([`Error::BadRequest`], no network call) —
    /// Pinecone's default namespace is `__default__` in paths. Same retry and
    /// missing-key policy as [`Client::query`].
    pub fn describe_namespace(&self, index: &str, namespace: &str) -> Result<NamespaceInfo, Error> {
        require_namespace(namespace, "describe_namespace")?;
        self.with_host_retry(index, API_VERSION, |host| {
            self.describe_namespace_host(host, namespace)
        })
    }

    fn describe_namespace_host(&self, host: &str, namespace: &str) -> Result<NamespaceInfo, Error> {
        let url = format!(
            "{}/namespaces/{}",
            host.trim_end_matches('/'),
            namespace_segment(Some(namespace))
        );
        let mut resp = self
            .agent
            .get(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .call()
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => Ok(read_json::<ApiNamespace>(&mut resp)?.into()),
            404 => {
                drain(&mut resp);
                Err(Error::NamespaceNotFound {
                    namespace: namespace.to_string(),
                })
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Deletes one namespace of `index` and everything in it
    /// (`DELETE /namespaces/{namespace}`, irreversible). Idempotent: a 404
    /// (already gone, or never written to) is success, like
    /// [`Client::delete_vectors`]. A namespace an alias points at is refused
    /// ([`Error::BadRequest`]). Empty `namespace` and key policy as in
    /// [`Client::describe_namespace`].
    pub fn delete_namespace(&self, index: &str, namespace: &str) -> Result<(), Error> {
        require_namespace(namespace, "delete_namespace")?;
        self.with_host_retry(index, API_VERSION, |host| {
            self.delete_namespace_host(host, namespace)
        })
    }

    fn delete_namespace_host(&self, host: &str, namespace: &str) -> Result<(), Error> {
        let url = format!(
            "{}/namespaces/{}",
            host.trim_end_matches('/'),
            namespace_segment(Some(namespace))
        );
        let mut resp = self
            .agent
            .delete(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", API_VERSION)
            .call()
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 | 404 => {
                drain(&mut resp);
                Ok(())
            }
            400 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 | 403 => Err(Error::Auth),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    // ---- Backups & backup schedules (control plane, `BACKUP_API_VERSION`) ----

    /// Creates a whole-index backup of `req.index` (`POST
    /// /indexes/{index}/backups`) — always index-scoped, never per-namespace.
    /// Sent with [`BACKUP_API_VERSION`], not [`API_VERSION`].
    pub fn create_backup(&self, req: &CreateBackupRequest) -> Result<BackupInfo, Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let url = format!(
            "{}/indexes/{}/backups",
            self.config.control_plane_url.trim_end_matches('/'),
            index_segment(&req.index)
        );
        let mut body = json!({});
        if let Some(name) = &req.name {
            body["name"] = json!(name);
        }
        if let Some(description) = &req.description {
            body["description"] = json!(description);
        }
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", BACKUP_API_VERSION)
            .send_json(&body)
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 | 201 => Ok(read_json::<ApiBackupInfo>(&mut resp)?.into()),
            400 | 422 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 => Err(Error::Auth),
            403 => Err(backup_forbidden(&mut resp)),
            404 => Err(Error::IndexNotFound {
                index: req.index.clone(),
            }),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Lists every backup of `index`, walking pagination internally (`GET
    /// /indexes/{index}/backups`, ≤100/page). Sent with [`BACKUP_API_VERSION`].
    pub fn list_backups(&self, index: &str) -> Result<Vec<BackupInfo>, Error> {
        let url_base = format!(
            "{}/indexes/{}/backups",
            self.config.control_plane_url.trim_end_matches('/'),
            index_segment(index)
        );
        self.list_backups_paginated(&url_base)
    }

    /// Lists every backup in the project, regardless of source index (`GET
    /// /backups`), walking pagination internally. Sent with
    /// [`BACKUP_API_VERSION`].
    pub fn list_project_backups(&self) -> Result<Vec<BackupInfo>, Error> {
        let url_base = format!(
            "{}/backups",
            self.config.control_plane_url.trim_end_matches('/')
        );
        self.list_backups_paginated(&url_base)
    }

    fn list_backups_paginated(&self, url_base: &str) -> Result<Vec<BackupInfo>, Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let mut backups = Vec::new();
        let mut token: Option<String> = None;
        for _ in 0..LIST_VECTORS_MAX_PAGES {
            let mut url = format!("{url_base}?limit={BACKUPS_PAGE_LIMIT}");
            if let Some(t) = &token {
                url.push_str(&format!(
                    "&paginationToken={}",
                    utf8_percent_encode(t, QUERY_COMPONENT)
                ));
            }
            let mut resp = self
                .agent
                .get(&url)
                .header("Api-Key", self.config.api_key.as_str())
                .header("X-Pinecone-Api-Version", BACKUP_API_VERSION)
                .call()
                .map_err(|e| self.map_transport(e, &url))?;
            match resp.status().as_u16() {
                200 => {
                    let parsed: BackupListResponse = read_json(&mut resp)?;
                    backups.extend(parsed.data.into_iter().map(BackupInfo::from));
                    match parsed
                        .pagination
                        .and_then(|p| p.next)
                        .filter(|n| !n.is_empty())
                    {
                        Some(next) => token = Some(next),
                        None => return Ok(backups),
                    }
                }
                404 => return Ok(backups),
                401 => return Err(Error::Auth),
                403 => return Err(backup_forbidden(&mut resp)),
                status => {
                    return Err(Error::Api {
                        status,
                        message: read_text(&mut resp),
                    });
                }
            }
        }
        Err(Error::BadRequest {
            message: format!(
                "list_backups: exceeded {LIST_VECTORS_MAX_PAGES} pages of pagination without \
                 exhausting the list — aborting"
            ),
        })
    }

    /// Fetches metadata for one backup by id (`GET /backups/{id}`). Sent with
    /// [`BACKUP_API_VERSION`].
    pub fn describe_backup(&self, backup_id: &str) -> Result<BackupInfo, Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let url = format!(
            "{}/backups/{}",
            self.config.control_plane_url.trim_end_matches('/'),
            index_segment(backup_id)
        );
        let mut resp = self
            .agent
            .get(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", BACKUP_API_VERSION)
            .call()
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => Ok(read_json::<ApiBackupInfo>(&mut resp)?.into()),
            404 => Err(Error::BackupNotFound {
                backup_id: backup_id.to_string(),
            }),
            401 => Err(Error::Auth),
            403 => Err(backup_forbidden(&mut resp)),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Deletes backup `backup_id` (`DELETE /backups/{id}`). Sent with
    /// [`BACKUP_API_VERSION`].
    pub fn delete_backup(&self, backup_id: &str) -> Result<(), Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let url = format!(
            "{}/backups/{}",
            self.config.control_plane_url.trim_end_matches('/'),
            index_segment(backup_id)
        );
        let mut resp = self
            .agent
            .delete(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", BACKUP_API_VERSION)
            .call()
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 | 202 => {
                drain(&mut resp);
                Ok(())
            }
            404 => Err(Error::BackupNotFound {
                backup_id: backup_id.to_string(),
            }),
            401 => Err(Error::Auth),
            403 => Err(backup_forbidden(&mut resp)),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Restores `backup_id` into a brand-new index `new_index_name`
    /// (`POST /backups/{id}/create-index`) — restore is ALWAYS to a new
    /// index, never in place; `pinecone.repoint` is what re-points a tracked
    /// table's registration at it afterward. Sent with [`BACKUP_API_VERSION`].
    pub fn create_index_from_backup(
        &self,
        backup_id: &str,
        new_index_name: &str,
    ) -> Result<IndexInfo, Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let url = format!(
            "{}/backups/{}/create-index",
            self.config.control_plane_url.trim_end_matches('/'),
            index_segment(backup_id)
        );
        let body = json!({ "name": new_index_name });
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", BACKUP_API_VERSION)
            .send_json(&body)
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            201 => {
                let desc: IndexDescription = read_json(&mut resp)?;
                Ok(index_info(desc))
            }
            400 | 422 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 => Err(Error::Auth),
            403 => Err(backup_forbidden(&mut resp)),
            404 => Err(Error::BackupNotFound {
                backup_id: backup_id.to_string(),
            }),
            409 => Err(Error::Conflict {
                name: new_index_name.to_string(),
            }),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Creates (or replaces) `index`'s backup schedule
    /// (`POST /indexes/{index}/backups/schedule`). `schedule` is sent
    /// verbatim — Pinecone's docs don't pin down the exact request shape for
    /// schedules yet; callers should treat the response `Value` as
    /// provisional until a [live] probe confirms the shape. Sent with
    /// [`BACKUP_API_VERSION`].
    pub fn create_backup_schedule(&self, index: &str, schedule: Value) -> Result<Value, Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let url = format!(
            "{}/indexes/{}/backups/schedule",
            self.config.control_plane_url.trim_end_matches('/'),
            index_segment(index)
        );
        let mut resp = self
            .agent
            .post(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", BACKUP_API_VERSION)
            .send_json(&schedule)
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 | 201 => read_json(&mut resp),
            400 | 422 => Err(Error::BadRequest {
                message: extract_message(&mut resp),
            }),
            401 => Err(Error::Auth),
            403 => Err(backup_forbidden(&mut resp)),
            404 => Err(Error::IndexNotFound {
                index: index.to_string(),
            }),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    /// Fetches `index`'s current backup schedule, if any
    /// (`GET /indexes/{index}/backups/schedule`) — `None` if none is
    /// configured. Same provisional-shape caveat as
    /// [`Client::create_backup_schedule`]. Sent with [`BACKUP_API_VERSION`].
    pub fn get_backup_schedule(&self, index: &str) -> Result<Option<Value>, Error> {
        if self.config.api_key.is_empty() {
            return Err(Error::MissingApiKey);
        }
        let url = format!(
            "{}/indexes/{}/backups/schedule",
            self.config.control_plane_url.trim_end_matches('/'),
            index_segment(index)
        );
        let mut resp = self
            .agent
            .get(&url)
            .header("Api-Key", self.config.api_key.as_str())
            .header("X-Pinecone-Api-Version", BACKUP_API_VERSION)
            .call()
            .map_err(|e| self.map_transport(e, &url))?;
        match resp.status().as_u16() {
            200 => Ok(Some(read_json(&mut resp)?)),
            404 => Ok(None),
            401 => Err(Error::Auth),
            403 => Err(backup_forbidden(&mut resp)),
            status => Err(Error::Api {
                status,
                message: read_text(&mut resp),
            }),
        }
    }

    fn cache(&self) -> MutexGuard<'_, HashMap<String, String>> {
        self.host_cache.lock().expect("host cache lock poisoned")
    }

    fn evict(&self, index: &str) {
        self.cache().remove(index);
    }

    fn map_transport(&self, err: ureq::Error, url: &str) -> Error {
        match err {
            ureq::Error::Timeout(_) => Error::Timeout {
                ms: timeout_ms(self.config.timeout),
            },
            other => Error::Transport {
                url: url.to_string(),
                reason: other.to_string(),
            },
        }
    }
}

/// Rejects an empty namespace for the `/namespaces/{namespace}` calls: the
/// empty path segment would address the list endpoint instead.
fn require_namespace(namespace: &str, call: &str) -> Result<(), Error> {
    if namespace.is_empty() {
        return Err(Error::BadRequest {
            message: format!(
                "{call} requires a namespace name — use `{DEFAULT_NAMESPACE}` for the default namespace"
            ),
        });
    }
    Ok(())
}

fn timeout_ms(timeout: Duration) -> u64 {
    u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX)
}

/// Builds an [`IndexInfo`] from a control-plane index description, normalizing
/// the host's scheme.
fn index_info(desc: IndexDescription) -> IndexInfo {
    let read_capacity = desc
        .read_capacity
        .or_else(|| {
            desc.spec
                .and_then(|s| s.serverless)
                .and_then(|s| s.read_capacity)
        })
        .map(read_capacity_info);
    IndexInfo {
        host: normalize_scheme(&desc.host),
        dimension: desc.dimension,
        metric: desc.metric,
        vector_type: desc.vector_type,
        ready: desc.status.ready,
        read_capacity,
    }
}

fn read_capacity_info(rc: ApiReadCapacity) -> ReadCapacity {
    let (node_type, shards, replicas) = match rc.dedicated {
        Some(d) => (
            d.node_type,
            d.manual.as_ref().and_then(|m| m.shards),
            d.manual.and_then(|m| m.replicas),
        ),
        None => (None, None, None),
    };
    ReadCapacity {
        mode: rc.mode,
        node_type,
        shards,
        replicas,
        state: rc.status.and_then(|s| s.state),
    }
}

/// Reads a numeric `Retry-After` (seconds) off a response, capped at
/// [`RETRY_AFTER_CAP`]. `None` if absent or not a plain integer.
fn retry_after(resp: &Response<Body>) -> Option<Duration> {
    let raw = resp.headers().get("Retry-After")?.to_str().ok()?;
    let secs: u64 = raw.trim().parse().ok()?;
    Some(Duration::from_secs(secs).min(RETRY_AFTER_CAP))
}

/// Reads a JSON body, mapping malformed bodies to `Error::Api` (not `Transport`,
/// which would trigger the evict-and-re-resolve retry).
fn read_json<T: DeserializeOwned>(resp: &mut Response<Body>) -> Result<T, Error> {
    let status = resp.status().as_u16();
    resp.body_mut().read_json().map_err(|e| Error::Api {
        status,
        message: format!("could not parse response body as JSON: {e}"),
    })
}

fn read_text(resp: &mut Response<Body>) -> String {
    resp.body_mut().read_to_string().unwrap_or_default()
}

/// Largest body [`drain`] reads; a longer one costs the connection, as an unread one does.
const DRAIN_LIMIT: u64 = 1 << 20;

/// Reads a response the caller has no use for to its end: only a body read to the end lets ureq
/// return the connection to its pool, and a dropped one costs a handshake per call.
fn drain(resp: &mut Response<Body>) {
    let mut reader = resp.body_mut().with_config().limit(DRAIN_LIMIT).reader();
    let _ = std::io::copy(&mut reader, &mut std::io::sink());
}

/// Whether Pinecone's message says an index has reached its plan's limit of namespaces.
pub fn names_namespace_limit(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("namespace")
        && ["limit", "maximum", "quota", "exceed", "reached", "too many"]
            .iter()
            .any(|word| message.contains(word))
}

/// A 403 refuses the credentials, unless its body says the plan's namespace limit was reached:
/// that is the plan refusing a new namespace, which the caller can act on.
fn forbidden(resp: &mut Response<Body>) -> Error {
    let message = extract_message(resp);
    if names_namespace_limit(&message) {
        Error::Api {
            status: 403,
            message,
        }
    } else {
        Error::Auth
    }
}

/// Pulls the server's message out of a JSON error body — the data plane's
/// top-level `{"message": ...}` shape or the control plane's nested
/// `{"error": {"message": ...}}` one — falling back to the raw body text.
fn extract_message(resp: &mut Response<Body>) -> String {
    let text = read_text(resp);
    serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|v| {
            v.get("message")
                .or_else(|| v.get("error").and_then(|e| e.get("message")))
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .unwrap_or(text)
}

/// Builds the message for a `delete_index`/`delete_doc_index` 403. Unlike 401
/// (bad or missing credentials), Pinecone returns 403 for a request that
/// authenticated fine but was refused — in practice this is the index's
/// `deletion_protection` setting, so the message names that as the likely
/// cause and folds in the server's own message when it sent one.
fn deletion_protection_forbidden(name: &str, resp: &mut Response<Body>) -> String {
    let server_message = extract_message(resp);
    let suffix = if server_message.is_empty() {
        String::new()
    } else {
        format!(": {server_message}")
    };
    format!(
        "index '{name}' could not be deleted (403) — likely caused by deletion_protection being \
         enabled on the index; disable it (or delete via the Pinecone console) and retry{suffix}"
    )
}

/// Builds the error for a backup-surface 403. Unlike 401
/// (bad or missing credentials), Pinecone returns 403 for a request that
/// authenticated fine but was refused — on the backup endpoints that is most
/// often the PLAN gate (backups are Standard/Enterprise only; the server's
/// body says "Your organization's plan does not include backups..."). The
/// old `403 => Error::Auth` mapping masked exactly that as "check your API
/// key" in the live DR drill; surfacing the server's own message as
/// [`Error::BadRequest`] keeps plan-ineligibility programmatically
/// distinguishable from bad credentials — the 0.6 backup gate's error
/// taxonomy depends on it.
fn backup_forbidden(resp: &mut Response<Body>) -> Error {
    let server_message = extract_message(resp);
    let detail = if server_message.is_empty() {
        "most likely your Pinecone plan does not include backups (Standard/Enterprise only)"
            .to_string()
    } else {
        server_message
    };
    Error::BadRequest {
        message: format!("backup request refused (403): {detail}"),
    }
}

/// Prepends `https://` when `host` has no scheme; returns it unchanged otherwise.
pub fn normalize_scheme(host: &str) -> String {
    if host.contains("://") {
        host.to_string()
    } else {
        format!("https://{host}")
    }
}

#[cfg(test)]
mod tests {
    use super::{index_segment, names_namespace_limit, normalize_scheme};

    #[test]
    fn the_namespace_limit_is_told_from_other_refusals() {
        for limit in [
            "Request failed. You've reached the maximum number of namespaces for this index (100).",
            "namespace quota exceeded",
            "Too many namespaces",
        ] {
            assert!(names_namespace_limit(limit), "{limit}");
        }
        for other in [
            "",
            "Forbidden",
            "Invalid API Key",
            "namespace is not yours",
            "rate limit exceeded",
        ] {
            assert!(!names_namespace_limit(other), "{other}");
        }
    }

    #[test]
    fn adds_https_when_scheme_missing() {
        assert_eq!(
            normalize_scheme("my-index-123.svc.pinecone.io"),
            "https://my-index-123.svc.pinecone.io"
        );
    }

    #[test]
    fn keeps_explicit_http_scheme() {
        assert_eq!(
            normalize_scheme("http://127.0.0.1:5555"),
            "http://127.0.0.1:5555"
        );
    }

    #[test]
    fn keeps_explicit_https_scheme() {
        assert_eq!(
            normalize_scheme("https://already-scoped.example.com"),
            "https://already-scoped.example.com"
        );
    }

    #[test]
    fn index_segment_leaves_ordinary_names_unchanged() {
        assert_eq!(index_segment("my-index-123").to_string(), "my-index-123");
    }

    #[test]
    fn index_segment_percent_encodes_slash_and_question_mark() {
        assert_eq!(
            index_segment("weird/name?x").to_string(),
            "weird%2Fname%3Fx"
        );
    }
}
