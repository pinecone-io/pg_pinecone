use std::collections::HashMap;

use serde_json::Value;

/// A vector similarity query against a single index.
#[derive(Debug, Clone)]
pub struct QueryRequest {
    pub index: String,
    pub vector: Vec<f32>,
    pub top_k: u32,
    pub namespace: Option<String>,
    pub filter: Option<Value>,
    /// Whether Pinecone should return each match's stored metadata.
    /// `pinecone.topk` (remote) sets this `false` — it only needs id/score.
    pub include_metadata: bool,
    /// Dedicated Read Node query-time tuning (ignored by an on-demand
    /// index); `None` omits the field from the request body entirely rather
    /// than sending a value Pinecone would just ignore anyway.
    pub scan_factor: Option<f64>,
    pub max_candidates: Option<u32>,
}

/// A single scored result from a query.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub id: String,
    pub score: f32,
    pub metadata: Option<Value>,
}

/// A single vector to upsert: an id, its values, and optional metadata.
#[derive(Debug, Clone)]
pub struct Vector {
    pub id: String,
    pub values: Vec<f32>,
    pub metadata: Option<Value>,
}

/// Records per upsert request (`UpsertRequest.vectors.maxItems` in the OpenAPI spec).
pub const UPSERT_MAX_RECORDS: usize = 1_000;

/// Total [`Vector::record_bytes`] per upsert request: 5% under Pinecone's 2 MB, which counts record
/// contents rather than JSON bytes (docs.pinecone.io, "Upsert in batches").
pub const UPSERT_MAX_BYTES: usize = 1_900_000;

const RECORD_FRAMING_BYTES: usize = 32;

impl Vector {
    /// The size Pinecone counts for this record against a request's 2 MB: 4 bytes per value, the id,
    /// the metadata as JSON, and some framing.
    pub fn record_bytes(&self) -> usize {
        let metadata = self.metadata.as_ref().map_or(0, json_len);
        4 * self.values.len() + self.id.len() + metadata + RECORD_FRAMING_BYTES
    }
}

fn json_len(value: &Value) -> usize {
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 += bytes.len();
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter(0);
    let _ = serde_json::to_writer(&mut counter, value);
    counter.0
}

/// A batch of ready-made vectors to upsert into a standard index. Unlike the
/// integrated `records` path, the caller supplies the vectors directly.
#[derive(Debug, Clone)]
pub struct UpsertVectorsRequest {
    pub index: String,
    pub namespace: Option<String>,
    pub vectors: Vec<Vector>,
}

/// A request to embed text into vectors via the Pinecone Inference API. Only
/// dense embedding models are supported; sparse models (e.g.
/// `pinecone-sparse-english-v0`) produce an error.
#[derive(Debug, Clone)]
pub struct EmbedRequest {
    pub model: String,
    pub inputs: Vec<String>,
    /// `"query"` or `"passage"`; matters for asymmetric models like `multilingual-e5-large`.
    pub input_type: String,
}

/// A request to rerank documents against a query via the Pinecone Inference API.
#[derive(Debug, Clone)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    /// Keep only the top `n` results; `None` returns all of them.
    pub top_n: Option<u32>,
}

/// A single reranked result. `index` is 0-based into the request's
/// `documents`, exactly as returned by the API (order is response order, not
/// input order).
#[derive(Debug, Clone, PartialEq)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
}

/// A request to create a new standard index.
#[derive(Debug, Clone)]
pub struct CreateIndexRequest {
    pub name: String,
    pub dimension: u32,
    /// `"cosine"`, `"euclidean"`, or `"dotproduct"`.
    pub metric: String,
    /// The full `spec` object verbatim, e.g. `{"serverless": {...}}` or `{"byoc": {...}}`.
    pub spec: Value,
}

/// Index metadata returned by [`Client::create_index`](crate::Client::create_index),
/// [`Client::describe_index`](crate::Client::describe_index), and
/// [`Client::wait_index_ready`](crate::Client::wait_index_ready).
#[derive(Debug, Clone, PartialEq)]
pub struct IndexInfo {
    /// Data-plane host, normalized to include a scheme.
    pub host: String,
    pub dimension: u32,
    pub metric: String,
    pub vector_type: String,
    pub ready: bool,
    /// `None` for a pod-based index, or a response shape this client doesn't recognize.
    pub read_capacity: Option<ReadCapacity>,
}

/// A serverless index's Dedicated Read Node config. Pinecone has put this
/// object in two different response locations across API versions, so
/// parsing tolerates both (see `client.rs`'s `IndexDescription`).
#[derive(Debug, Clone, PartialEq)]
pub struct ReadCapacity {
    /// `"OnDemand"` or `"Dedicated"`.
    pub mode: String,
    /// `"b1"` or `"t1"` — `None` for `mode = "OnDemand"`.
    pub node_type: Option<String>,
    pub shards: Option<u32>,
    pub replicas: Option<u32>,
    /// `"Ready"`, `"Scaling"`, `"Migrating"`, or `"Error"`, if reported.
    pub state: Option<String>,
}

/// A request to delete vectors from a standard index. Exactly one of `ids`,
/// `delete_all` or `filter` must be set; any other combination is a
/// client-side error.
#[derive(Debug, Clone)]
pub struct DeleteVectorsRequest {
    pub index: String,
    pub namespace: Option<String>,
    pub ids: Option<Vec<String>>,
    pub delete_all: bool,
    /// Delete every vector whose metadata matches (`"filter"` in the request
    /// body). Must be a non-empty JSON object: an empty or null filter is
    /// rejected client-side rather than sent.
    pub filter: Option<Value>,
}

/// A request to list every vector id in a standard index's namespace
/// (`GET /vectors/list`, paginated). [`Client::list_vector_ids`](crate::Client::list_vector_ids)
/// walks every page internally and returns the concatenated id list.
#[derive(Debug, Clone)]
pub struct ListVectorsRequest {
    pub index: String,
    pub namespace: Option<String>,
}

/// A request to create a document-schema (full-text-search) index.
#[derive(Debug, Clone)]
pub struct CreateDocIndexRequest {
    pub name: String,
    /// The `{"fields": {...}}` schema object, sent verbatim.
    pub schema: Value,
    pub cloud: String,
    pub region: String,
}

/// A document-schema index as described by the control plane
/// ([`Client::describe_doc_index`](crate::Client::describe_doc_index)).
#[derive(Debug, Clone, PartialEq)]
pub struct DocIndexInfo {
    /// Data-plane host, normalized to include a scheme.
    pub host: String,
    pub ready: bool,
    /// The `schema` object (`{"fields": {...}}`) verbatim; `Null` if absent.
    pub schema: Value,
    pub deletion_protection: Option<String>,
    /// The raw `read_capacity` object, if reported.
    pub read_capacity: Option<Value>,
}

/// A batch of documents to upsert into a document-schema index. Each document
/// is a JSON object with an `_id` field plus whatever schema/filterable fields
/// it carries.
#[derive(Debug, Clone)]
pub struct UpsertDocsRequest {
    pub index: String,
    pub namespace: Option<String>,
    pub documents: Vec<Value>,
}

/// Which documents a [`DeleteDocsRequest`] removes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocDeleteSelector {
    /// These ids, sent in sequential requests of at most 1000.
    Ids(Vec<String>),
    /// Every document in the namespace.
    All,
}

/// A request to delete documents from a document-schema index.
#[derive(Debug, Clone)]
pub struct DeleteDocsRequest {
    pub index: String,
    pub namespace: Option<String>,
    pub selector: DocDeleteSelector,
}

/// A full-text/hybrid search against a document-schema index.
#[derive(Debug, Clone)]
pub struct SearchDocsRequest {
    pub index: String,
    pub namespace: Option<String>,
    /// Fields to return per match; caller guarantees this is non-empty (e.g.
    /// `["*"]` for all fields).
    pub include_fields: Vec<String>,
    pub filter: Option<Value>,
    /// The `score_by` clause array, sent verbatim; caller normalizes a single
    /// clause into a one-element array.
    pub score_by: Value,
    pub top_k: u32,
}

/// A single scored result from [`Client::search_docs`](crate::Client::search_docs).
/// `doc` carries the match's fields (everything except `_id`/`_score`).
#[derive(Debug, Clone, PartialEq)]
pub struct DocMatch {
    pub id: String,
    pub score: f32,
    pub doc: Value,
}

/// A request to fetch full vectors (values + metadata) by id from a standard
/// index (`GET /vectors/fetch`). [`Client::fetch_vectors`](crate::Client::fetch_vectors)
/// chunks `ids` internally — Pinecone's hard cap is 1000 ids/call, but this
/// client chunks well under that for response-size safety (see the constant
/// in `client.rs`). An id Pinecone doesn't have is simply absent from the
/// returned map — not an error; compare the request's `ids` against the
/// response keys to detect misses/ghosts.
#[derive(Debug, Clone)]
pub struct FetchVectorsRequest {
    pub index: String,
    pub namespace: Option<String>,
    pub ids: Vec<String>,
}

/// A single namespace's vector count, from [`IndexStats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NamespaceStats {
    pub vector_count: u64,
}

/// `POST /describe_index_stats` response — per-namespace vector counts plus
/// the index total. Used both by the (future) backup-gate SYNC check and by
/// `pinecone.recover_from_remote`'s job accounting.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IndexStats {
    pub total_vector_count: u64,
    pub namespaces: HashMap<String, NamespaceStats>,
}

/// A request to set (merge) metadata on one existing vector, without
/// touching its `values` (`POST /vectors/update`). **Merge semantics,
/// live-verified 2026-09-13** (`pg_pinecone-client/tests/update_missing_id_probe.rs`):
/// `metadata` is merged into the vector's existing metadata object —
/// an untouched key survives, a changed key takes the new value, a
/// brand-new key is added; confirmed exactly as documented. It cannot
/// *unset* a key (a `null` value is not documented to delete one, and this
/// client never relies on it doing so).
///
/// **Missing-id behavior, also live-verified**: calling this on an id
/// Pinecone has no vector for returns a plain `200` (`Ok(())`) — a true
/// no-op. It does **not** create a phantom vector (fetching the id
/// afterward finds nothing, and the index's total vector count is
/// unchanged), and it is **never** a `404`. Since that `200` carries no
/// signal that the id didn't exist, a caller that needs to detect this
/// "ghost" case (e.g. `sync_trigger::try_update_metadata`) must check
/// existence itself first (a `fetch_vectors` call) rather than reading
/// anything off this call's own `Result`.
#[derive(Debug, Clone)]
pub struct UpdateVectorMetadataRequest {
    pub index: String,
    pub namespace: Option<String>,
    pub id: String,
    pub metadata: Value,
}

/// A request to create a backup of a whole standard index
/// (`POST /indexes/{name}/backups`). Backups are index-scoped and
/// whole-index only — no per-namespace granularity.
#[derive(Debug, Clone)]
pub struct CreateBackupRequest {
    pub index: String,
    pub name: Option<String>,
    pub description: Option<String>,
}

/// Backup metadata, as returned by [`Client::create_backup`](crate::Client::create_backup),
/// [`Client::list_backups`](crate::Client::list_backups),
/// [`Client::list_project_backups`](crate::Client::list_project_backups), and
/// [`Client::describe_backup`](crate::Client::describe_backup). Fields Pinecone
/// hasn't populated yet (e.g. `record_count` while `status` is still
/// `"Pending"`) are `None` rather than a sentinel like `0`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BackupInfo {
    pub backup_id: String,
    pub name: Option<String>,
    /// `"Pending"`, `"Initializing"`, or `"Ready"`.
    pub status: String,
    pub source_index_name: Option<String>,
    pub dimension: Option<u32>,
    pub record_count: Option<u64>,
    pub namespace_count: Option<u64>,
    pub size_bytes: Option<u64>,
    /// RFC 3339 timestamp, passed through verbatim (not parsed) — every
    /// caller either displays it or compares it as an opaque, lexically
    /// sortable string.
    pub created_at: Option<String>,
}

/// A single page of `GET /vectors/list` — the resumable-cursor primitive
/// [`Client::recover`](crate::Client)-style DR jobs need:
/// [`Client::list_vector_ids`](crate::Client::list_vector_ids) walks every
/// page internally for callers that want the whole namespace at once, but
/// `pinecone.recover_from_remote`'s outbox job persists its cursor between
/// worker drain rounds, so it needs one page at a time instead.
#[derive(Debug, Clone)]
pub struct ListVectorIdsPageRequest {
    pub index: String,
    pub namespace: Option<String>,
    /// `None` requests the first page.
    pub pagination_token: Option<String>,
}

/// One page's worth of ids plus the token for the next page, if any.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ListVectorIdsPage {
    pub ids: Vec<String>,
    pub next_token: Option<String>,
}

/// A filter-based, paginated fetch WITH values (`POST
/// /vectors/fetch_by_metadata`). Unlike [`FetchVectorsRequest`]
/// (id-based, non-paginated, max 1000 ids/call), this is filter-based and
/// paginated (Pinecone's documented max is 10,000 records/page) —
/// [`Client::fetch_by_metadata`](crate::Client::fetch_by_metadata) rejects an
/// empty/null `filter` client-side, before any HTTP call: Pinecone documents
/// no fetch-all via this endpoint, only ever a non-empty filter.
///
/// **Response shape live-verified 2026-09-12**: it matches this client's sibling
/// data-plane endpoints (`/vectors/fetch`'s `vectors` object,
/// `/vectors/list`'s `pagination.next` token); a full 20k-record pagination
/// walk against a real serverless index returned every record exactly once,
/// values bit-exact fp32.
#[derive(Debug, Clone)]
pub struct FetchByMetadataRequest {
    pub index: String,
    pub namespace: Option<String>,
    /// Must be non-empty — enforced client-side before any HTTP call.
    pub filter: Value,
    /// `None` sends Pinecone's documented per-page maximum (10,000).
    pub limit: Option<u32>,
    /// `None` requests the first page; `Some` resumes from a previous
    /// [`FetchByMetadataPage::next_token`].
    pub pagination_token: Option<String>,
}

/// One page of [`Client::fetch_by_metadata`](crate::Client::fetch_by_metadata)
/// — full vectors (values + metadata) plus the token for the next page, if
/// any. Deliberately one-page-per-call (mirrors
/// [`ListVectorIdsPageRequest`]/[`ListVectorIdsPage`], not
/// [`Client::list_vector_ids`](crate::Client::list_vector_ids)'s
/// auto-walking convenience): per-page timing matters for throughput
/// measurement, and a caller such as `pinecone.recover_from_remote` would
/// want the same persisted-cursor primitive `ListVectorIdsPage` gives.
#[derive(Debug, Clone, Default)]
pub struct FetchByMetadataPage {
    pub vectors: HashMap<String, Vector>,
    pub next_token: Option<String>,
}

/// One page of namespaces (`GET /namespaces`, paginated). Only
/// [`Client::list_namespaces`](crate::Client::list_namespaces) takes this;
/// [`Client::list_all_namespaces`](crate::Client::list_all_namespaces) walks
/// every page for callers that want the whole list.
#[derive(Debug, Clone)]
pub struct ListNamespacesRequest {
    pub index: String,
    /// Only namespaces whose name starts with this; `None` lists them all.
    pub prefix: Option<String>,
    /// Page size, 1 to 100; `None` leaves it to Pinecone's default (100).
    pub limit: Option<u32>,
    /// `None` requests the first page; `Some` resumes from a previous
    /// [`ListNamespacesPage::next`].
    pub pagination_token: Option<String>,
}

/// A namespace as `GET /namespaces` and `GET /namespaces/{namespace}`
/// describe it. Parsed leniently: a missing, oddly typed or unparsable field
/// is `None`/empty rather than a failed call (Pinecone sends the counts as
/// decimal strings; a plain number parses too).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NamespaceInfo {
    pub name: String,
    /// Counts may lag behind recent writes and deletes.
    pub record_count: Option<u64>,
    pub size_bytes: Option<u64>,
    /// Names of the indexed metadata fields.
    pub indexed_fields: Vec<String>,
    /// The metadata `schema` object verbatim, if the namespace has one.
    pub schema: Option<Value>,
}

/// One page of [`Client::list_namespaces`](crate::Client::list_namespaces),
/// in Pinecone's order (sorted by name, bitwise), plus the token for the next
/// page (`None` on the last).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ListNamespacesPage {
    pub namespaces: Vec<NamespaceInfo>,
    pub next: Option<String>,
}
