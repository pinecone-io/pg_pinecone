//! Stateful, namespace-aware stand-in for Pinecone's vector (dense) API, for the partition tests.
//!
//! One [`VectorsHandle`] holds the records of every namespace, the recorded calls and the fault
//! rules, shared between the Postgres backend thread and the mock servers' thread. The data
//! plane knows no index name, so it is one logical store. Every `X-Pinecone-Api-Version` is
//! answered (the header is recorded, never checked); a request without `namespace` addresses
//! `__default__`. A namespace exists while it holds a record (or was preloaded); emptying it
//! makes it vanish unless [`VectorsHandle::keep_empty_namespaces`] says otherwise. `/query` on a
//! namespace that holds nothing answers 200 with no matches.

#![allow(dead_code)] // the surface is shared by the test groups; a build uses a part of it

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use serde_json::{Map, Value, json};
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{header, path_regex},
};

use super::MockBackend;

pub const DEFAULT_NAMESPACE: &str = "__default__";

const MAX_ID_BYTES: usize = 512;
const MAX_DELETE_IDS: usize = 1000;
/// Pinecone's limits of one upsert request: 1,000 records and 2 MB, where the 2 MB counts what
/// the records hold (4 bytes a value, the id and the metadata as JSON), not the bytes of the
/// JSON text (docs.pinecone.io, "Upsert in batches": 559 records of 768 values and 500 bytes of
/// metadata). Read as 2,000,000, the stricter of the two meanings of "MB".
const MAX_UPSERT_RECORDS: usize = 1000;
const MAX_UPSERT_BYTES: usize = 2_000_000;
const LIST_PAGE_DEFAULT: usize = 100;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    IndexCreate,
    IndexDescribe,
    IndexDelete,
    Upsert,
    Delete,
    List,
    Fetch,
    FetchByMetadata,
    Update,
    Query,
    DescribeStats,
    NamespaceList,
    NamespaceDescribe,
    NamespaceDelete,
    Unknown,
}

/// One request the mock received.
#[derive(Clone, Debug)]
pub struct VectorCall {
    pub endpoint: Endpoint,
    pub method: String,
    pub path: String,
    /// The namespace as sent (body, query string or path); `None` when the request named none.
    pub namespace: Option<String>,
    pub index: Option<String>,
    pub api_version: Option<String>,
    /// Decoded query string pairs, in order (a repeated `ids` stays repeated).
    pub query: Vec<(String, String)>,
    pub body: Option<Value>,
    pub status: u16,
    /// Bytes of the request body as it came over the wire.
    pub wire_bytes: usize,
}

/// What one upsert request weighed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UpsertSize {
    pub records: usize,
    /// What Pinecone counts against its 2 MB: 4 bytes a value, the id, the metadata as JSON.
    pub counted_bytes: usize,
    /// The body as it came over the wire.
    pub wire_bytes: usize,
    pub status: u16,
}

fn counted_bytes(items: &[Value]) -> usize {
    let one = |item: &Value| {
        let values = item["values"].as_array().map_or(0, Vec::len);
        let id = item["id"].as_str().map_or(0, str::len);
        let metadata = item.get("metadata").map_or(0, |m| m.to_string().len());
        4 * values + id + metadata
    };
    items.iter().map(one).sum()
}

impl VectorCall {
    /// The size of an upsert request; `None` for any other call.
    pub fn upsert_size(&self) -> Option<UpsertSize> {
        if self.endpoint != Endpoint::Upsert {
            return None;
        }
        let items = self.body.as_ref()?["vectors"].as_array()?;
        Some(UpsertSize {
            records: items.len(),
            counted_bytes: counted_bytes(items),
            wire_bytes: self.wire_bytes,
            status: self.status,
        })
    }

    pub fn param(&self, key: &str) -> Option<&str> {
        let pair = self.query.iter().find(|(k, _)| k == key);
        pair.map(|(_, v)| v.as_str())
    }

    pub fn params(&self, key: &str) -> Vec<&str> {
        let pairs = self.query.iter().filter(|(k, _)| k == key);
        pairs.map(|(_, v)| v.as_str()).collect()
    }
}

struct Record {
    values: Vec<f32>,
    metadata: Option<Value>,
}

struct Rule {
    endpoint: Endpoint,
    skip: usize,
    remaining: Option<usize>,
    namespace: Option<String>,
    status: u16,
    message: String,
    /// Answer `message` as a plain-text body instead of the JSON error shape.
    plain: bool,
}

#[derive(Clone)]
struct IndexInfo {
    dimension: u64,
    metric: String,
}

#[derive(Default)]
struct State {
    calls: Vec<VectorCall>,
    indexes: BTreeMap<String, IndexInfo>,
    active: Option<IndexInfo>,
    records: BTreeMap<String, BTreeMap<String, Record>>,
    keep_empty: bool,
    unknown_query_status: Option<u16>,
    namespace_cap: Option<(usize, u16, String)>,
    rules: Vec<Rule>,
    latency: Vec<(Endpoint, Duration)>,
    data_plane_uri: String,
}

impl State {
    fn exists(&self, namespace: &str) -> bool {
        self.records.contains_key(namespace)
    }

    fn settle(&mut self) {
        if !self.keep_empty {
            self.records.retain(|_, store| !store.is_empty());
        }
    }
}

/// Shared handle to the mock's state; clone it freely.
#[derive(Clone, Default)]
pub struct VectorsHandle {
    state: Arc<Mutex<State>>,
}

impl VectorsHandle {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("vectors mock state poisoned")
    }

    pub fn calls(&self) -> Vec<VectorCall> {
        self.lock().calls.clone()
    }

    pub fn calls_to(&self, endpoint: Endpoint) -> Vec<VectorCall> {
        let state = self.lock();
        let calls = state.calls.iter();
        calls.filter(|c| c.endpoint == endpoint).cloned().collect()
    }

    pub fn count(&self, endpoint: Endpoint) -> usize {
        self.calls_to(endpoint).len()
    }

    /// The size of every upsert request received, in arrival order, the refused ones included.
    pub fn upsert_sizes(&self) -> Vec<UpsertSize> {
        let calls = self.calls_to(Endpoint::Upsert);
        calls.iter().filter_map(VectorCall::upsert_size).collect()
    }

    /// Forgets the recorded calls (not the stored records or indexes).
    pub fn clear_calls(&self) {
        self.lock().calls.clear();
    }

    /// Names of the namespaces that exist, sorted.
    pub fn namespaces(&self) -> Vec<String> {
        self.lock().records.keys().cloned().collect()
    }

    pub fn ids(&self, namespace: &str) -> BTreeSet<String> {
        let state = self.lock();
        let store = state.records.get(namespace);
        store
            .map(|s| s.keys().cloned().collect())
            .unwrap_or_default()
    }

    pub fn record_count(&self, namespace: &str) -> usize {
        self.lock().records.get(namespace).map_or(0, BTreeMap::len)
    }

    pub fn total_records(&self) -> usize {
        self.lock().records.values().map(BTreeMap::len).sum()
    }

    /// `{"id", "values", "metadata"?}` of one stored record.
    pub fn record(&self, namespace: &str, id: &str) -> Option<Value> {
        let state = self.lock();
        let record = state.records.get(namespace)?.get(id)?;
        Some(record_json(id, record, true))
    }
}

impl VectorsHandle {
    /// Stores one record as if an upsert had delivered it.
    pub fn preload(&self, namespace: &str, id: &str, values: &[f32], metadata: Option<Value>) {
        let record = Record {
            values: values.to_vec(),
            metadata,
        };
        let mut state = self.lock();
        let store = state.records.entry(namespace.to_string()).or_default();
        store.insert(id.to_string(), record);
    }

    /// Makes an empty namespace exist (`POST /namespaces` creates one ahead of an upsert).
    pub fn preload_namespace(&self, namespace: &str) {
        let mut state = self.lock();
        state.keep_empty = true;
        state.records.entry(namespace.to_string()).or_default();
    }

    /// Makes index `name` exist with `dimension` and `metric` (`cosine`, `dotproduct`, `euclidean`).
    pub fn preload_index(&self, name: &str, dimension: u32, metric: &str) {
        let info = IndexInfo {
            dimension: dimension.into(),
            metric: metric.to_string(),
        };
        let mut state = self.lock();
        state.active = Some(info.clone());
        state.indexes.insert(name.to_string(), info);
    }

    /// Removes a namespace and its records, as a restore or an out-of-band delete would.
    pub fn drop_namespace(&self, namespace: &str) {
        self.lock().records.remove(namespace);
    }

    /// Whether a namespace whose last record went away keeps existing (default: it vanishes).
    pub fn keep_empty_namespaces(&self, keep: bool) {
        let mut state = self.lock();
        state.keep_empty = keep;
        state.settle();
    }

    /// What `/query` answers for a namespace that holds nothing: `None` is 200 with no matches
    /// (the default), `Some(404)` the not-found shape.
    pub fn set_query_unknown_namespace(&self, status: Option<u16>) {
        self.lock().unknown_query_status = status;
    }

    /// An upsert that would create namespace number `cap + 1` answers `status` with `message`.
    pub fn set_namespace_cap(&self, cap: usize, status: u16, message: &str) {
        self.lock().namespace_cap = Some((cap, status, message.to_string()));
    }

    pub fn fail_always(&self, endpoint: Endpoint, status: u16, message: &str) {
        self.add_rule(endpoint, 0, None, None, status, message);
    }

    pub fn fail_next(&self, endpoint: Endpoint, n: usize, status: u16, message: &str) {
        self.add_rule(endpoint, 0, Some(n), None, status, message);
    }

    /// Only the `nth` call to `endpoint` from now on (1-based) answers `status`.
    pub fn fail_nth(&self, endpoint: Endpoint, nth: usize, status: u16, message: &str) {
        let skip = nth.saturating_sub(1);
        self.add_rule(endpoint, skip, Some(1), None, status, message);
    }

    /// `count` calls to `endpoint`, starting with the `from`th from now on (1-based), answer `status`.
    pub fn fail_range(
        &self,
        endpoint: Endpoint,
        from: usize,
        count: usize,
        status: u16,
        message: &str,
    ) {
        let skip = from.saturating_sub(1);
        self.add_rule(endpoint, skip, Some(count), None, status, message);
    }

    /// Every call to `endpoint` addressed to `namespace` answers `status` until healed.
    pub fn fail_namespace(&self, endpoint: Endpoint, namespace: &str, status: u16, message: &str) {
        let namespace = Some(namespace.to_string());
        self.add_rule(endpoint, 0, None, namespace, status, message);
    }

    /// Every call to `endpoint` addressed to `namespace` answers 200 with a body that is no JSON.
    pub fn garble_namespace(&self, endpoint: Endpoint, namespace: &str) {
        let rule = Rule {
            endpoint,
            skip: 0,
            remaining: None,
            namespace: Some(namespace.to_string()),
            status: 200,
            message: "<html>ok</html>".to_string(),
            plain: true,
        };
        self.lock().rules.push(rule);
    }

    pub fn heal(&self, endpoint: Endpoint) {
        self.lock().rules.retain(|r| r.endpoint != endpoint);
    }

    pub fn heal_all(&self) {
        self.lock().rules.clear();
    }

    /// Delays every answer of `endpoint` by `delay` (asynchronously).
    pub fn set_latency(&self, endpoint: Endpoint, delay: Duration) {
        let mut state = self.lock();
        state.latency.retain(|(e, _)| *e != endpoint);
        state.latency.push((endpoint, delay));
    }

    fn add_rule(
        &self,
        endpoint: Endpoint,
        skip: usize,
        remaining: Option<usize>,
        namespace: Option<String>,
        status: u16,
        message: &str,
    ) {
        let message = message.to_string();
        let rule = Rule {
            endpoint,
            skip,
            remaining,
            namespace,
            status,
            message,
            plain: false,
        };
        self.lock().rules.push(rule);
    }
}

/// A [`MockBackend`] whose control plane and data plane are both served by `vectors`.
pub fn start_with_vectors(vectors: &VectorsHandle) -> MockBackend {
    let vectors = vectors.clone();
    MockBackend::start(move |control_plane, data_plane| async move {
        vectors.mount(&control_plane, &data_plane).await;
        (control_plane, data_plane)
    })
}

const CONTROL_ROUTES: &str = r"^/indexes(/[^/]*)?$";
const DATA_ROUTES: &str = r"^/(vectors/[a-z_]+|query|describe_index_stats|namespaces(/.*)?)$";

impl VectorsHandle {
    /// Installs the handlers; call it inside [`MockBackend::start`]'s setup.
    pub async fn mount(&self, control_plane: &MockServer, data_plane: &MockServer) {
        self.mount_filtered(control_plane, data_plane, None).await;
    }

    /// Like [`Self::mount`], but only answers requests that carry `version` as
    /// `X-Pinecone-Api-Version`.
    pub async fn mount_for_version(&self, cp: &MockServer, dp: &MockServer, version: &str) {
        self.mount_filtered(cp, dp, Some(version)).await;
    }

    async fn mount_filtered(&self, cp: &MockServer, dp: &MockServer, version: Option<&str>) {
        self.lock().data_plane_uri = dp.uri();
        for (server, pattern) in [(cp, CONTROL_ROUTES), (dp, DATA_ROUTES)] {
            let handle = self.clone();
            let respond = move |req: &Request| handle.handle(req);
            let mock = Mock::given(path_regex(pattern));
            match version {
                Some(v) => mock
                    .and(header("X-Pinecone-Api-Version", v))
                    .respond_with(respond),
                None => mock.respond_with(respond),
            }
            .mount(server)
            .await;
        }
    }
}

struct Reply {
    status: u16,
    body: Value,
    /// A body that replaces `body` and is no JSON.
    text: Option<String>,
}

fn ok(status: u16, body: Value) -> Reply {
    Reply {
        status,
        body,
        text: None,
    }
}

fn error_reply(status: u16, message: &str) -> Reply {
    let code = match status {
        400 => "INVALID_ARGUMENT",
        401 => "UNAUTHENTICATED",
        403 => "PERMISSION_DENIED",
        404 => "NOT_FOUND",
        409 => "ALREADY_EXISTS",
        429 => "RESOURCE_EXHAUSTED",
        503 => "UNAVAILABLE",
        _ => "UNKNOWN",
    };
    let error = json!({ "code": code, "message": message });
    ok(
        status,
        json!({ "error": error, "status": status, "message": message }),
    )
}

struct Route {
    endpoint: Endpoint,
    index: Option<String>,
    namespace: Option<String>,
}

fn parse_route(method: &str, path: &str) -> Route {
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    let (mut index, mut namespace) = (None, None);
    let endpoint = match (method, segments.as_slice()) {
        ("POST", ["indexes"]) => Endpoint::IndexCreate,
        ("GET", ["indexes", name]) => {
            index = Some(percent_decode(name));
            Endpoint::IndexDescribe
        }
        ("DELETE", ["indexes", name]) => {
            index = Some(percent_decode(name));
            Endpoint::IndexDelete
        }
        ("POST", ["vectors", "upsert"]) => Endpoint::Upsert,
        ("POST", ["vectors", "delete"]) => Endpoint::Delete,
        ("GET", ["vectors", "list"]) => Endpoint::List,
        ("GET", ["vectors", "fetch"]) => Endpoint::Fetch,
        ("POST", ["vectors", "fetch_by_metadata"]) => Endpoint::FetchByMetadata,
        ("POST", ["vectors", "update"]) => Endpoint::Update,
        ("POST", ["query"]) => Endpoint::Query,
        ("POST", ["describe_index_stats"]) => Endpoint::DescribeStats,
        ("GET", ["namespaces"]) => Endpoint::NamespaceList,
        ("GET", ["namespaces", ns]) => {
            namespace = Some(percent_decode(ns));
            Endpoint::NamespaceDescribe
        }
        ("DELETE", ["namespaces", ns]) => {
            namespace = Some(percent_decode(ns));
            Endpoint::NamespaceDelete
        }
        _ => Endpoint::Unknown,
    };
    Route {
        endpoint,
        index,
        namespace,
    }
}

fn percent_decode(segment: &str) -> String {
    let bytes = segment.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| (b as char).to_digit(16);
        let escaped = (bytes[i] == b'%' && i + 2 < bytes.len())
            .then(|| Some(hex(bytes[i + 1])? * 16 + hex(bytes[i + 2])?))
            .flatten();
        match escaped {
            Some(byte) => {
                out.push(byte as u8);
                i += 3;
            }
            None => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn sent_namespace(
    endpoint: Endpoint,
    query: &[(String, String)],
    body: Option<&Value>,
) -> Option<String> {
    match endpoint {
        Endpoint::List | Endpoint::Fetch => {
            let pair = query.iter().find(|(k, _)| k == "namespace");
            pair.map(|(_, v)| v.clone())
        }
        Endpoint::Upsert
        | Endpoint::Delete
        | Endpoint::Query
        | Endpoint::Update
        | Endpoint::FetchByMetadata => {
            let sent = body?.get("namespace")?.as_str();
            sent.map(str::to_string)
        }
        _ => None,
    }
}

fn take_fault(
    state: &mut State,
    endpoint: Endpoint,
    namespace: Option<&str>,
) -> Option<(u16, String, bool)> {
    let effective = namespace.unwrap_or(DEFAULT_NAMESPACE);
    let mut fired = None;
    for rule in state.rules.iter_mut().filter(|r| r.endpoint == endpoint) {
        if rule.namespace.as_deref().is_some_and(|ns| ns != effective) {
            continue;
        }
        if rule.skip > 0 {
            rule.skip -= 1;
            continue;
        }
        match &mut rule.remaining {
            Some(0) => continue,
            Some(n) => *n -= 1,
            None => {}
        }
        fired = Some((rule.status, rule.message.clone(), rule.plain));
        break;
    }
    state.rules.retain(|r| r.remaining != Some(0));
    fired
}

impl VectorsHandle {
    fn handle(&self, req: &Request) -> ResponseTemplate {
        let route = parse_route(req.method.as_str(), req.url.path());
        let body = serde_json::from_slice::<Value>(&req.body).ok();
        let pairs = req.url.query_pairs();
        let query: Vec<(String, String)> = pairs
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        let version = req.headers.get("x-pinecone-api-version");
        let api_version = version.and_then(|v| v.to_str().ok()).map(str::to_string);
        let sent = sent_namespace(route.endpoint, &query, body.as_ref());
        let namespace = route.namespace.clone().or(sent);

        let mut state = self.lock();
        let reply = match take_fault(&mut state, route.endpoint, namespace.as_deref()) {
            Some((status, message, false)) => error_reply(status, &message),
            Some((status, message, true)) => Reply {
                text: Some(message),
                ..ok(status, Value::Null)
            },
            None => {
                let call = Call {
                    route: &route,
                    namespace: namespace.as_deref().unwrap_or(DEFAULT_NAMESPACE),
                    query: &query,
                    body: body.as_ref(),
                };
                dispatch(&mut state, &call)
            }
        };
        state.settle();
        state.calls.push(VectorCall {
            endpoint: route.endpoint,
            method: req.method.to_string(),
            path: req.url.path().to_string(),
            namespace,
            index: route.index.clone().or_else(|| created_name(body.as_ref())),
            api_version,
            query,
            body,
            status: reply.status,
            wire_bytes: req.body.len(),
        });
        let delay = state.latency.iter().find(|(e, _)| *e == route.endpoint);
        let delay = delay.map(|(_, d)| *d);
        drop(state);

        let template = ResponseTemplate::new(reply.status);
        let template = match reply.text {
            Some(text) => template.set_body_string(text),
            None => template.set_body_json(reply.body),
        };
        match delay {
            Some(d) => template.set_delay(d),
            None => template,
        }
    }
}

fn created_name(body: Option<&Value>) -> Option<String> {
    let name = body?.get("name")?.as_str()?;
    Some(name.to_string())
}

struct Call<'a> {
    route: &'a Route,
    namespace: &'a str,
    query: &'a [(String, String)],
    body: Option<&'a Value>,
}

impl Call<'_> {
    fn param(&self, key: &str) -> Option<&str> {
        let pair = self.query.iter().find(|(k, _)| k == key);
        pair.map(|(_, v)| v.as_str())
    }
}

fn dispatch(state: &mut State, call: &Call) -> Reply {
    let (ns, body) = (call.namespace, call.body);
    let index = call.route.index.as_deref().unwrap_or_default();
    match call.route.endpoint {
        Endpoint::IndexCreate => create_index(state, body),
        Endpoint::IndexDescribe => describe_index(state, index),
        Endpoint::IndexDelete => delete_index(state, index),
        Endpoint::Upsert => upsert(state, ns, body),
        Endpoint::Delete => delete(state, ns, body),
        Endpoint::List => list(state, call),
        Endpoint::Fetch => fetch(state, call),
        Endpoint::FetchByMetadata => fetch_by_metadata(state, ns, body),
        Endpoint::Update => update(state, ns, body),
        Endpoint::Query => query(state, ns, body),
        Endpoint::DescribeStats => describe_stats(state),
        Endpoint::NamespaceList => namespace_list(state, call),
        Endpoint::NamespaceDescribe => namespace_describe(state, ns),
        Endpoint::NamespaceDelete => namespace_delete(state, ns),
        Endpoint::Unknown => error_reply(404, "mock: no such vector-API route"),
    }
}

fn index_model(state: &State, name: &str, info: &IndexInfo) -> Value {
    json!({
        "name": name,
        "host": state.data_plane_uri,
        "dimension": info.dimension,
        "metric": info.metric,
        "vector_type": "dense",
        "status": { "ready": true, "state": "Ready" },
        "deletion_protection": "disabled",
    })
}

fn create_index(state: &mut State, body: Option<&Value>) -> Reply {
    let Some(body) = body else {
        return error_reply(400, "mock: request body must be JSON");
    };
    let name = body["name"].as_str().unwrap_or_default();
    if state.indexes.contains_key(name) {
        return error_reply(409, &format!("Resource {name} already exists"));
    }
    let info = IndexInfo {
        dimension: body["dimension"].as_u64().unwrap_or_default(),
        metric: body["metric"].as_str().unwrap_or("cosine").to_string(),
    };
    state.indexes.insert(name.to_string(), info.clone());
    state.active = Some(info.clone());
    ok(201, index_model(state, name, &info))
}

fn describe_index(state: &State, name: &str) -> Reply {
    match state.indexes.get(name) {
        Some(info) => ok(200, index_model(state, name, info)),
        None => error_reply(404, &format!("Resource {name} not found")),
    }
}

fn delete_index(state: &mut State, name: &str) -> Reply {
    match state.indexes.remove(name) {
        Some(_) => ok(202, json!({})),
        None => error_reply(404, &format!("Resource {name} not found")),
    }
}

fn record_json(id: &str, record: &Record, include_values: bool) -> Value {
    let mut out = json!({ "id": id });
    if include_values {
        out["values"] = json!(record.values);
    }
    if let Some(metadata) = &record.metadata {
        out["metadata"] = metadata.clone();
    }
    out
}

fn number_array(value: &Value) -> Option<Vec<f32>> {
    let items = value.as_array()?;
    items.iter().map(|n| n.as_f64().map(|f| f as f32)).collect()
}

fn metadata_value_ok(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::Number(_) | Value::Bool(_) => true,
        Value::Array(items) => items.iter().all(Value::is_string),
        Value::Null | Value::Object(_) => false,
    }
}

fn parse_vector(item: &Value, dimension: Option<usize>) -> Result<(String, Record), String> {
    let id = item["id"].as_str().unwrap_or_default();
    if id.is_empty() || id.len() > MAX_ID_BYTES {
        return Err(format!(
            "vector id must be 1-{MAX_ID_BYTES} bytes, got {id:?}"
        ));
    }
    let values = number_array(&item["values"]).filter(|v| !v.is_empty());
    let values =
        values.ok_or_else(|| format!("vector {id}: values must be a non-empty number array"))?;
    if let Some(d) = dimension
        && values.len() != d
    {
        let got = values.len();
        return Err(format!(
            "vector {id} has dimension {got}, the index has {d}"
        ));
    }
    let metadata = match item.get("metadata") {
        None | Some(Value::Null) => None,
        Some(Value::Object(map)) if map.values().all(metadata_value_ok) => {
            Some(Value::Object(map.clone()))
        }
        Some(_) => return Err(format!("vector {id}: invalid metadata")),
    };
    Ok((id.to_string(), Record { values, metadata }))
}

fn upsert(state: &mut State, ns: &str, body: Option<&Value>) -> Reply {
    let Some(items) = body.and_then(|b| b["vectors"].as_array()) else {
        return error_reply(400, "vectors must be an array");
    };
    if items.is_empty() || items.len() > MAX_UPSERT_RECORDS {
        let got = items.len();
        return error_reply(
            400,
            &format!("a request holds 1-{MAX_UPSERT_RECORDS} vectors (got {got})"),
        );
    }
    let counted = counted_bytes(items);
    if counted > MAX_UPSERT_BYTES {
        return error_reply(
            400,
            &format!(
                "Request size {counted} bytes exceeds the maximum supported size of \
                 {MAX_UPSERT_BYTES} bytes ({} vectors)",
                items.len()
            ),
        );
    }
    let dimension = state.active.as_ref().map(|i| i.dimension as usize);
    let mut parsed = Vec::with_capacity(items.len());
    for item in items {
        match parse_vector(item, dimension) {
            Ok(vector) => parsed.push(vector),
            Err(message) => return error_reply(400, &message),
        }
    }
    if !state.exists(ns)
        && let Some((cap, status, message)) = &state.namespace_cap
        && state.records.len() >= *cap
    {
        return error_reply(*status, message);
    }
    let count = parsed.len();
    let store = state.records.entry(ns.to_string()).or_default();
    store.extend(parsed);
    ok(200, json!({ "upsertedCount": count }))
}

fn delete(state: &mut State, ns: &str, body: Option<&Value>) -> Reply {
    let Some(body) = body else {
        return error_reply(400, "mock: request body must be JSON");
    };
    let ids = body.get("ids");
    let wipe = body["deleteAll"] == json!(true);
    let filter = body.get("filter");
    let forms = usize::from(ids.is_some()) + usize::from(wipe) + usize::from(filter.is_some());
    if forms != 1 {
        return error_reply(400, "exactly one of ids, deleteAll or filter is required");
    }
    if let Some(ids) = ids {
        let ids = ids.as_array().map(Vec::as_slice).unwrap_or_default();
        if !(1..=MAX_DELETE_IDS).contains(&ids.len()) {
            let got = ids.len();
            return error_reply(
                400,
                &format!("ids must hold 1-{MAX_DELETE_IDS} ids (got {got})"),
            );
        }
    }
    let Some(store) = state.records.get_mut(ns) else {
        return error_reply(404, &format!("Namespace {ns} not found"));
    };
    if let Some(ids) = ids {
        let ids = ids
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str);
        ids.for_each(|id| drop(store.remove(id)));
    } else if let Some(filter) = filter {
        store.retain(|_, record| !matches_filter(record.metadata.as_ref(), filter));
    } else {
        store.clear();
    }
    ok(200, json!({}))
}

fn list(state: &State, call: &Call) -> Reply {
    let limit = call.param("limit").and_then(|l| l.parse::<usize>().ok());
    let limit = limit.unwrap_or(LIST_PAGE_DEFAULT).clamp(1, 100);
    let prefix = call.param("prefix").unwrap_or_default();
    let after = call
        .param("paginationToken")
        .and_then(|t| t.strip_prefix("after:"));
    let store = state.records.get(call.namespace);
    let ids = store.into_iter().flat_map(BTreeMap::keys);
    let mut page: Vec<&String> = ids
        .filter(|id| id.starts_with(prefix) && after.is_none_or(|a| id.as_str() > a))
        .take(limit + 1)
        .collect();
    let more = page.len() > limit;
    page.truncate(limit);
    let vectors: Vec<Value> = page.iter().map(|id| json!({ "id": id })).collect();
    let mut body = json!({
        "vectors": vectors,
        "namespace": call.namespace,
        "usage": { "readUnits": 1 },
    });
    if let (true, Some(last)) = (more, page.last()) {
        body["pagination"] = json!({ "next": format!("after:{last}") });
    }
    ok(200, body)
}

fn fetch(state: &State, call: &Call) -> Reply {
    let store = state.records.get(call.namespace);
    let wanted = call.query.iter().filter(|(k, _)| k == "ids");
    let mut vectors = Map::new();
    for (_, id) in wanted {
        if let Some(record) = store.and_then(|s| s.get(id)) {
            vectors.insert(id.clone(), record_json(id, record, true));
        }
    }
    let body = json!({
        "vectors": vectors,
        "namespace": call.namespace,
        "usage": { "readUnits": 1 },
    });
    ok(200, body)
}

fn fetch_by_metadata(state: &State, ns: &str, body: Option<&Value>) -> Reply {
    let filter = body.map(|b| &b["filter"]).filter(|f| f.is_object());
    let Some(filter) = filter else {
        return error_reply(400, "filter must be a non-empty object");
    };
    let store = state.records.get(ns).into_iter().flatten();
    let mut vectors = Map::new();
    for (id, record) in store {
        if matches_filter(record.metadata.as_ref(), filter) {
            vectors.insert(id.clone(), record_json(id, record, true));
        }
    }
    ok(
        200,
        json!({ "vectors": vectors, "namespace": ns, "usage": { "readUnits": 1 } }),
    )
}

fn update(state: &mut State, ns: &str, body: Option<&Value>) -> Reply {
    let id = body.and_then(|b| b["id"].as_str()).unwrap_or_default();
    let Some(store) = state.records.get_mut(ns) else {
        return error_reply(404, &format!("Namespace {ns} not found"));
    };
    let patch = body.and_then(|b| b["setMetadata"].as_object());
    if let (Some(record), Some(patch)) = (store.get_mut(id), patch) {
        let mut merged = match record.metadata.take() {
            Some(Value::Object(map)) => map,
            _ => Map::new(),
        };
        merged.extend(patch.clone());
        record.metadata = Some(Value::Object(merged));
    }
    ok(200, json!({}))
}

fn score(metric: &str, query: &[f32], values: &[f32]) -> f32 {
    let dot: f32 = query.iter().zip(values).map(|(a, b)| a * b).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    match metric {
        "dotproduct" => dot,
        "euclidean" => query
            .iter()
            .zip(values)
            .map(|(a, b)| (a - b) * (a - b))
            .sum(),
        _ => match (norm(query), norm(values)) {
            (a, b) if a == 0.0 || b == 0.0 => 0.0,
            (a, b) => dot / (a * b),
        },
    }
}

fn query(state: &State, ns: &str, body: Option<&Value>) -> Reply {
    let Some(body) = body else {
        return error_reply(400, "mock: request body must be JSON");
    };
    let Some(vector) = number_array(&body["vector"]) else {
        return error_reply(400, "vector must be an array of numbers");
    };
    let top_k = body["topK"].as_u64().filter(|k| (1..=10_000).contains(k));
    let Some(top_k) = top_k else {
        return error_reply(400, "topK must be between 1 and 10000");
    };
    let dimension = state.active.as_ref().map(|i| i.dimension as usize);
    if dimension.is_some_and(|d| d != vector.len()) {
        return error_reply(400, "query vector dimension does not match the index");
    }
    let Some(store) = state.records.get(ns) else {
        return match state.unknown_query_status {
            Some(status) => error_reply(status, &format!("Namespace {ns} not found")),
            None => ok(
                200,
                json!({ "matches": [], "namespace": ns, "usage": { "readUnits": 1 } }),
            ),
        };
    };
    let metric = state
        .active
        .as_ref()
        .map_or("cosine", |i| i.metric.as_str());
    let filter = body.get("filter").filter(|f| f.is_object());
    let include_metadata = body["includeMetadata"].as_bool().unwrap_or(false);
    let mut scored: Vec<(f32, &String, &Record)> = store
        .iter()
        .filter(|(_, r)| filter.is_none_or(|f| matches_filter(r.metadata.as_ref(), f)))
        .map(|(id, r)| (score(metric, &vector, &r.values), id, r))
        .collect();
    scored.sort_by(|a, b| {
        let order = a.0.total_cmp(&b.0);
        let order = if metric == "euclidean" {
            order
        } else {
            order.reverse()
        };
        order.then_with(|| a.1.cmp(b.1))
    });
    let matches: Vec<Value> = scored
        .into_iter()
        .take(top_k as usize)
        .map(|(score, id, record)| {
            let mut m = json!({ "id": id, "score": score });
            if let (true, Some(metadata)) = (include_metadata, &record.metadata) {
                m["metadata"] = metadata.clone();
            }
            m
        })
        .collect();
    ok(
        200,
        json!({ "matches": matches, "namespace": ns, "usage": { "readUnits": 1 } }),
    )
}

fn describe_stats(state: &State) -> Reply {
    let mut namespaces = Map::new();
    for (name, store) in &state.records {
        let key = if name == DEFAULT_NAMESPACE {
            ""
        } else {
            name.as_str()
        };
        namespaces.insert(key.to_string(), json!({ "vectorCount": store.len() }));
    }
    let total: usize = state.records.values().map(BTreeMap::len).sum();
    let dimension = state.active.as_ref().map_or(0, |i| i.dimension);
    ok(
        200,
        json!({
            "namespaces": namespaces,
            "dimension": dimension,
            "indexFullness": 0.0,
            "totalVectorCount": total,
        }),
    )
}

fn namespace_list(state: &State, call: &Call) -> Reply {
    let limit = match call.param("limit").map(str::parse::<usize>) {
        None => 100,
        Some(Ok(n)) if (1..=100).contains(&n) => n,
        Some(_) => return error_reply(400, "limit must be between 1 and 100"),
    };
    let prefix = call.param("prefix").unwrap_or_default();
    let after = call
        .param("paginationToken")
        .and_then(|t| t.strip_prefix("after:"));
    let matching: Vec<(&String, &BTreeMap<String, Record>)> = state
        .records
        .iter()
        .filter(|(name, _)| name.starts_with(prefix))
        .collect();
    let total = matching.len();
    let mut page: Vec<_> = matching
        .into_iter()
        .filter(|(name, _)| after.is_none_or(|a| name.as_str() > a))
        .take(limit + 1)
        .collect();
    let more = page.len() > limit;
    page.truncate(limit);
    let listed: Vec<Value> = page
        .iter()
        .map(|(name, store)| json!({ "name": name, "record_count": store.len().to_string() }))
        .collect();
    let mut body = json!({ "namespaces": listed, "total_count": total });
    if let (true, Some((last, _))) = (more, page.last()) {
        body["pagination"] = json!({ "next": format!("after:{last}") });
    }
    ok(200, body)
}

fn namespace_describe(state: &State, ns: &str) -> Reply {
    match state.records.get(ns) {
        Some(store) => ok(
            200,
            json!({ "name": ns, "record_count": store.len().to_string() }),
        ),
        None if ns == DEFAULT_NAMESPACE => ok(200, json!({ "name": ns, "record_count": "0" })),
        None => error_reply(404, &format!("Namespace {ns} not found")),
    }
}

fn namespace_delete(state: &mut State, ns: &str) -> Reply {
    match state.records.remove(ns) {
        Some(_) => ok(200, json!({})),
        None => error_reply(404, &format!("Namespace {ns} not found")),
    }
}

fn value_eq(actual: &Value, wanted: &Value) -> bool {
    match (actual, wanted) {
        (Value::Array(items), wanted) if !wanted.is_array() => items.iter().any(|i| i == wanted),
        (Value::Number(a), Value::Number(b)) => a.as_f64() == b.as_f64(),
        (a, b) => a == b,
    }
}

fn apply_op(actual: Option<&Value>, op: &str, arg: &Value) -> bool {
    let list = arg.as_array().map(Vec::as_slice).unwrap_or_default();
    let cmp = |f: fn(f64, f64) -> bool| {
        let pair = actual.and_then(Value::as_f64).zip(arg.as_f64());
        pair.is_some_and(|(a, b)| f(a, b))
    };
    match op {
        "$eq" => actual.is_some_and(|a| value_eq(a, arg)),
        "$ne" => actual.is_none_or(|a| !value_eq(a, arg)),
        "$in" => actual.is_some_and(|a| list.iter().any(|w| value_eq(a, w))),
        "$nin" => actual.is_none_or(|a| !list.iter().any(|w| value_eq(a, w))),
        "$exists" => actual.is_some() == arg.as_bool().unwrap_or(true),
        "$gt" => cmp(|a, b| a > b),
        "$gte" => cmp(|a, b| a >= b),
        "$lt" => cmp(|a, b| a < b),
        "$lte" => cmp(|a, b| a <= b),
        _ => false,
    }
}

fn field_matches(actual: Option<&Value>, condition: &Value) -> bool {
    match condition {
        Value::Object(ops) if ops.keys().all(|k| k.starts_with('$')) => {
            ops.iter().all(|(op, arg)| apply_op(actual, op, arg))
        }
        literal => apply_op(actual, "$eq", literal),
    }
}

/// Evaluates a Pinecone metadata filter (shorthand equality, `$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/
/// $exists`, `$and`, `$or`) against one record's metadata.
fn matches_filter(metadata: Option<&Value>, filter: &Value) -> bool {
    let Some(conditions) = filter.as_object() else {
        return false;
    };
    conditions
        .iter()
        .all(|(key, condition)| match key.as_str() {
            "$and" => condition
                .as_array()
                .is_some_and(|c| c.iter().all(|f| matches_filter(metadata, f))),
            "$or" => condition
                .as_array()
                .is_some_and(|c| c.iter().any(|f| matches_filter(metadata, f))),
            field => field_matches(metadata.and_then(|m| m.get(field)), condition),
        })
}

/// Self-tests of the mock: plain HTTP against the handlers, no Postgres. Green by design.
#[cfg(test)]
mod tests {
    use super::*;

    const VERSION: &str = "2026-04";

    fn call(method: &str, url: &str, body: Option<&Value>) -> (u16, Value) {
        let config = ureq::Agent::config_builder().http_status_as_error(false);
        let agent: ureq::Agent = config.build().into();
        let result = match method {
            "GET" => agent
                .get(url)
                .header("X-Pinecone-Api-Version", VERSION)
                .call(),
            "DELETE" => agent
                .delete(url)
                .header("X-Pinecone-Api-Version", VERSION)
                .call(),
            _ => agent
                .post(url)
                .header("X-Pinecone-Api-Version", VERSION)
                .send_json(body.unwrap_or(&Value::Null)),
        };
        let mut response = result.expect("mock reachable");
        let status = response.status().as_u16();
        let body = response
            .body_mut()
            .read_json::<Value>()
            .unwrap_or(Value::Null);
        (status, body)
    }

    /// Starts the mock, creates a 2-d cosine index, returns `(backend, control_url, data_url)`.
    fn started(vectors: &VectorsHandle) -> (MockBackend, String, String) {
        let backend = start_with_vectors(vectors);
        let control = backend.control_plane_url.clone();
        let create = json!({"name": "idx", "dimension": 2, "metric": "cosine"});
        let (status, created) = call("POST", &format!("{control}/indexes"), Some(&create));
        assert_eq!(status, 201, "create: {created}");
        let data = created["host"].as_str().expect("host").to_string();
        (backend, control, data)
    }

    fn upsert(data: &str, ns: Option<&str>, vectors: &[(&str, [f32; 2], Value)]) -> (u16, Value) {
        let items: Vec<Value> = vectors
            .iter()
            .map(|(id, v, m)| json!({"id": id, "values": v, "metadata": m}))
            .collect();
        let mut body = json!({ "vectors": items });
        if let Some(ns) = ns {
            body["namespace"] = json!(ns);
        }
        call("POST", &format!("{data}/vectors/upsert"), Some(&body))
    }

    fn strings(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn index_lifecycle() {
        let vectors = VectorsHandle::new();
        let (_backend, control, _data) = started(&vectors);
        let idx = format!("{control}/indexes/idx");
        let (status, body) = call("GET", &idx, None);
        assert_eq!((status, body["dimension"].clone()), (200, json!(2)));
        let create = json!({"name": "idx", "dimension": 2, "metric": "cosine"});
        assert_eq!(
            call("POST", &format!("{control}/indexes"), Some(&create)).0,
            409
        );
        assert_eq!(call("DELETE", &idx, None).0, 202);
        assert_eq!(call("GET", &idx, None).0, 404);
    }

    #[test]
    fn upsert_query_and_metadata_filters_are_namespace_scoped() {
        let vectors = VectorsHandle::new();
        let (_backend, _control, data) = started(&vectors);
        let a = [
            ("1", [1.0, 0.0], json!({"tenant": 1, "tag": "x"})),
            ("2", [0.0, 1.0], json!({"tenant": 1, "tag": "y"})),
            ("3", [1.0, 1.0], json!({"tenant": 1})),
        ];
        assert_eq!(upsert(&data, Some("t:1"), &a).0, 200);
        let nine = [("9", [1.0, 0.0], json!({"tenant": 2}))];
        assert_eq!(upsert(&data, Some("t:2"), &nine).0, 200);
        assert_eq!(vectors.namespaces(), vec!["t:1", "t:2"]);

        let ask = |ns: &str, filter: Option<Value>, k: u64| -> Vec<String> {
            let mut body = json!({"vector": [1.0, 0.0], "topK": k, "namespace": ns});
            if let Some(f) = filter {
                body["filter"] = f;
            }
            let (status, answer) = call("POST", &format!("{data}/query"), Some(&body));
            assert_eq!(status, 200, "{answer}");
            let matches = answer["matches"].as_array().expect("matches");
            let ids = matches.iter().map(|m| m["id"].as_str().expect("id"));
            ids.map(str::to_string).collect()
        };
        assert_eq!(ask("t:1", None, 10), ["1", "3", "2"]);
        assert_eq!(ask("t:1", None, 2), ["1", "3"]);
        assert_eq!(ask("t:2", None, 10), ["9"]);
        assert_eq!(ask("t:1", Some(json!({"tag": {"$eq": "y"}})), 10), ["2"]);
        assert_eq!(
            ask("t:1", Some(json!({"tag": {"$exists": true}})), 10),
            ["1", "2"]
        );
        let either = json!({"$or": [{"tag": "x"}, {"tenant": {"$gt": 5}}]});
        assert_eq!(ask("t:1", Some(either), 10), ["1"]);
        assert!(ask("t:1", Some(json!({"tenant": {"$in": [2, 3]}})), 10).is_empty());
        assert!(ask("never-written", None, 5).is_empty());

        vectors.set_query_unknown_namespace(Some(404));
        let body = json!({"vector": [1.0, 0.0], "topK": 1, "namespace": "never-written"});
        assert_eq!(call("POST", &format!("{data}/query"), Some(&body)).0, 404);
        let calls = vectors.calls_to(Endpoint::Query);
        assert_eq!(
            calls.last().and_then(|c| c.namespace.as_deref()),
            Some("never-written")
        );
    }

    #[test]
    fn delete_forms_missing_namespaces_and_vanishing() {
        let vectors = VectorsHandle::new();
        let (_backend, _control, data) = started(&vectors);
        let rows = [
            ("1", [1.0, 0.0], json!({"tenant": 1})),
            ("2", [0.0, 1.0], json!({"tenant": 2})),
            ("3", [1.0, 1.0], json!({"tenant": 1})),
        ];
        upsert(&data, Some("n"), &rows);
        let delete = |body: Value| call("POST", &format!("{data}/vectors/delete"), Some(&body));
        assert_eq!(delete(json!({"ids": ["1"], "namespace": "n"})).0, 200);
        assert_eq!(vectors.ids("n"), strings(&["2", "3"]));
        assert_eq!(
            delete(json!({"filter": {"tenant": {"$eq": 1}}, "namespace": "n"})).0,
            200
        );
        assert_eq!(vectors.ids("n"), strings(&["2"]));
        assert_eq!(delete(json!({"namespace": "n"})).0, 400);
        assert_eq!(
            delete(json!({"ids": ["x"], "deleteAll": true, "namespace": "n"})).0,
            400
        );
        assert_eq!(delete(json!({"deleteAll": true, "namespace": "n"})).0, 200);
        assert!(
            vectors.namespaces().is_empty(),
            "an emptied namespace vanishes"
        );
        assert_eq!(delete(json!({"deleteAll": true, "namespace": "n"})).0, 404);
        assert_eq!(
            delete(json!({"ids": ["1"]})).0,
            404,
            "no namespace = the default one"
        );

        vectors.keep_empty_namespaces(true);
        upsert(&data, Some("n"), &rows);
        delete(json!({"deleteAll": true, "namespace": "n"}));
        assert_eq!(vectors.namespaces(), vec!["n"], "kept while empty");
        assert_eq!(delete(json!({"deleteAll": true, "namespace": "n"})).0, 200);
    }

    #[test]
    fn listing_fetching_and_stats_per_namespace() {
        let vectors = VectorsHandle::new();
        let (_backend, _control, data) = started(&vectors);
        let rows: Vec<(String, [f32; 2], Value)> = (1..=5)
            .map(|i| {
                (
                    format!("{}{i}", if i < 4 { "a" } else { "b" }),
                    [1.0, 0.0],
                    json!({"n": i}),
                )
            })
            .collect();
        let rows: Vec<(&str, [f32; 2], Value)> = rows
            .iter()
            .map(|(id, v, m)| (id.as_str(), *v, m.clone()))
            .collect();
        upsert(&data, Some("ns"), &rows);
        upsert(&data, None, &rows[..1]);
        assert_eq!(vectors.namespaces(), vec!["__default__", "ns"]);

        let (status, page) = call(
            "GET",
            &format!("{data}/vectors/list?namespace=ns&limit=2"),
            None,
        );
        assert_eq!(status, 200);
        let ids = |page: &Value| -> Vec<String> {
            let vectors = page["vectors"].as_array().expect("vectors");
            let ids = vectors.iter().map(|v| v["id"].as_str().expect("id"));
            ids.map(str::to_string).collect()
        };
        assert_eq!(ids(&page), ["a1", "a2"]);
        let next = page["pagination"]["next"]
            .as_str()
            .expect("next page")
            .to_string();
        let url = format!("{data}/vectors/list?namespace=ns&limit=2&paginationToken={next}");
        let (_, page2) = call("GET", &url, None);
        assert_eq!(ids(&page2), ["a3", "b4"]);
        let url = format!(
            "{data}/vectors/list?namespace=ns&limit=2&paginationToken={}",
            page2["pagination"]["next"].as_str().expect("next")
        );
        let (_, page3) = call("GET", &url, None);
        assert_eq!(ids(&page3), ["b5"]);
        assert!(
            page3.get("pagination").is_none(),
            "the last page has no token"
        );
        let (_, only_b) = call(
            "GET",
            &format!("{data}/vectors/list?namespace=ns&prefix=b"),
            None,
        );
        assert_eq!(ids(&only_b), ["b4", "b5"]);
        let (status, empty) = call("GET", &format!("{data}/vectors/list?namespace=nope"), None);
        assert_eq!((status, ids(&empty)), (200, Vec::<String>::new()));

        let (_, fetched) = call(
            "GET",
            &format!("{data}/vectors/fetch?ids=a1&ids=zz&namespace=ns"),
            None,
        );
        let found: Vec<&String> = fetched["vectors"]
            .as_object()
            .expect("vectors")
            .keys()
            .collect();
        assert_eq!(found, ["a1"]);
        let (_, stats) = call(
            "POST",
            &format!("{data}/describe_index_stats"),
            Some(&json!({})),
        );
        assert_eq!(stats["namespaces"]["ns"]["vectorCount"], json!(5));
        assert_eq!(
            stats["namespaces"][""]["vectorCount"],
            json!(1),
            "the default namespace is \"\""
        );
        assert_eq!(stats["totalVectorCount"], json!(6));
    }

    #[test]
    fn namespace_operations_answer_the_documented_shapes() {
        let vectors = VectorsHandle::new();
        let (_backend, _control, data) = started(&vectors);
        for i in 0..5 {
            let ns = format!("public.docs:{i}");
            upsert(&data, Some(&ns), &[("1", [1.0, 0.0], json!({}))]);
        }
        upsert(
            &data,
            Some("public.docs2:1"),
            &[("1", [1.0, 0.0], json!({}))],
        );
        upsert(&data, Some("other"), &[("1", [1.0, 0.0], json!({}))]);

        let names = |body: &Value| -> Vec<String> {
            let listed = body["namespaces"].as_array().expect("namespaces");
            let names = listed.iter().map(|n| n["name"].as_str().expect("name"));
            names.map(str::to_string).collect()
        };
        let (status, all) = call("GET", &format!("{data}/namespaces"), None);
        assert_eq!((status, all["total_count"].clone()), (200, json!(7)));
        assert_eq!(all["namespaces"][0]["record_count"], json!("1"));
        let url = format!("{data}/namespaces?prefix=public.docs:&limit=2");
        let (_, first) = call("GET", &url, None);
        assert_eq!(names(&first), ["public.docs:0", "public.docs:1"]);
        assert_eq!(
            first["total_count"],
            json!(5),
            "the total counts the whole prefix"
        );
        let token = first["pagination"]["next"]
            .as_str()
            .expect("next")
            .to_string();
        let (_, second) = call("GET", &format!("{url}&paginationToken={token}"), None);
        assert_eq!(names(&second), ["public.docs:2", "public.docs:3"]);
        let token = second["pagination"]["next"]
            .as_str()
            .expect("next")
            .to_string();
        let (_, last) = call("GET", &format!("{url}&paginationToken={token}"), None);
        assert_eq!(names(&last), ["public.docs:4"]);
        assert!(last.get("pagination").is_none());
        assert_eq!(
            call("GET", &format!("{data}/namespaces?limit=0"), None).0,
            400
        );

        let one = format!("{data}/namespaces/public.docs%3A3");
        assert_eq!(call("GET", &one, None).1["record_count"], json!("1"));
        assert_eq!(call("DELETE", &one, None).0, 200);
        assert_eq!(call("GET", &one, None).0, 404);
        assert_eq!(call("DELETE", &one, None).0, 404);
        assert_eq!(vectors.namespaces().len(), 6);
        assert_eq!(
            call("GET", &format!("{data}/namespaces/__default__"), None).0,
            200
        );
        let deleted = vectors.calls_to(Endpoint::NamespaceDelete);
        assert_eq!(deleted[0].namespace.as_deref(), Some("public.docs:3"));
    }

    #[test]
    fn faults_by_endpoint_and_namespace_and_the_namespace_cap() {
        let vectors = VectorsHandle::new();
        let (_backend, _control, data) = started(&vectors);
        let one = [("1", [1.0, 0.0], json!({}))];
        vectors.fail_namespace(Endpoint::Upsert, "bad", 400, "namespace is poisoned");
        assert_eq!(upsert(&data, Some("good"), &one).0, 200);
        let (status, body) = upsert(&data, Some("bad"), &one);
        assert_eq!(
            (status, body["message"].clone()),
            (400, json!("namespace is poisoned"))
        );
        vectors.heal(Endpoint::Upsert);
        assert_eq!(upsert(&data, Some("bad"), &one).0, 200);

        vectors.fail_nth(Endpoint::Upsert, 2, 503, "later");
        assert_eq!(upsert(&data, Some("c1"), &one).0, 200);
        assert_eq!(upsert(&data, Some("c2"), &one).0, 503);
        assert_eq!(upsert(&data, Some("c2"), &one).0, 200);
        vectors.fail_range(Endpoint::Upsert, 2, 2, 400, "range");
        assert_eq!(upsert(&data, Some("c2"), &one).0, 200);
        assert_eq!(upsert(&data, Some("c2"), &one).0, 400);
        assert_eq!(upsert(&data, Some("c2"), &one).0, 400);
        assert_eq!(upsert(&data, Some("c2"), &one).0, 200);

        vectors.set_namespace_cap(4, 400, "max namespaces reached");
        assert_eq!(
            upsert(&data, Some("c1"), &one).0,
            200,
            "an existing namespace is unaffected"
        );
        assert_eq!(
            upsert(&data, Some("c3"), &one).1["message"],
            json!("max namespaces reached")
        );
        let namespaces = vectors.namespaces();
        assert_eq!(namespaces, ["bad", "c1", "c2", "good"]);
        let sent: Vec<Option<String>> = vectors
            .calls_to(Endpoint::Upsert)
            .into_iter()
            .map(|c| c.namespace)
            .collect();
        assert_eq!(sent.first(), Some(&Some("good".to_string())));
        assert_eq!(sent.len(), 12);
    }

    #[test]
    fn a_garbled_namespace_answers_200_with_a_body_that_is_no_json() {
        let vectors = VectorsHandle::new();
        let (_backend, _control, data) = started(&vectors);
        let one = [("1", [1.0, 0.0], json!({}))];
        vectors.garble_namespace(Endpoint::Upsert, "bad");

        let (status, body) = upsert(&data, Some("good"), &one);
        assert_eq!((status, body["upsertedCount"].clone()), (200, json!(1)));
        assert_eq!(upsert(&data, Some("bad"), &one), (200, Value::Null));
        assert!(vectors.ids("bad").is_empty());
    }

    #[test]
    fn the_real_client_works_against_the_mock() {
        use pg_pinecone_client::{
            Client, Config, CreateIndexRequest, DeleteVectorsRequest, Error, FetchVectorsRequest,
            ListNamespacesRequest, ListVectorIdsPageRequest, QueryRequest, UpsertVectorsRequest,
            Vector,
        };
        let vectors = VectorsHandle::new();
        let backend = start_with_vectors(&vectors);
        let config = Config {
            control_plane_url: backend.control_plane_url.clone(),
            api_key: "test-key".to_string(),
            timeout: Duration::from_secs(5),
            user_agent: None,
        };
        let client = Client::with_sleep(config, |_| {});
        let create = CreateIndexRequest {
            name: "idx".to_string(),
            dimension: 2,
            metric: "cosine".to_string(),
            spec: json!({"serverless": {"cloud": "aws", "region": "us-east-1"}}),
        };
        client.create_index(&create).expect("create");
        let upsert = |ns: &str, ids: &[&str]| UpsertVectorsRequest {
            index: "idx".to_string(),
            namespace: Some(ns.to_string()),
            vectors: ids
                .iter()
                .map(|id| Vector {
                    id: id.to_string(),
                    values: vec![1.0, 0.0],
                    metadata: Some(json!({"tenant": ns})),
                })
                .collect(),
        };
        assert_eq!(
            client
                .upsert_vectors(&upsert("public.t:1", &["a", "b"]))
                .expect("upsert"),
            2
        );
        client
            .upsert_vectors(&upsert("public.t:2", &["c"]))
            .expect("upsert");

        let query = |ns: &str| QueryRequest {
            index: "idx".to_string(),
            vector: vec![1.0, 0.0],
            top_k: 10,
            namespace: Some(ns.to_string()),
            filter: Some(json!({"tenant": {"$eq": ns}})),
            include_metadata: false,
            scan_factor: None,
            max_candidates: None,
        };
        let found = client.query(&query("public.t:1")).expect("query");
        assert_eq!(
            found.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        assert!(
            client
                .query(&query("public.t:9"))
                .expect("query")
                .is_empty()
        );

        let page = client
            .list_vector_ids_page(&ListVectorIdsPageRequest {
                index: "idx".to_string(),
                namespace: Some("public.t:1".to_string()),
                pagination_token: None,
            })
            .expect("list");
        assert_eq!(
            (page.ids, page.next_token),
            (vec!["a".to_string(), "b".to_string()], None)
        );
        let fetch = FetchVectorsRequest {
            index: "idx".to_string(),
            namespace: Some("public.t:2".to_string()),
            ids: vec!["c".to_string(), "zz".to_string()],
        };
        assert_eq!(client.fetch_vectors(&fetch).expect("fetch").len(), 1);
        let stats = client.describe_index_stats("idx").expect("stats");
        assert_eq!(stats.namespaces["public.t:1"].vector_count, 2);

        let wipe = |ns: &str| DeleteVectorsRequest {
            index: "idx".to_string(),
            namespace: Some(ns.to_string()),
            ids: None,
            delete_all: true,
            filter: None,
        };
        client.delete_vectors(&wipe("public.t:1")).expect("wipe");
        client
            .delete_vectors(&wipe("public.t:1"))
            .expect("a missing namespace is a 404 the client accepts");
        assert_eq!(vectors.namespaces(), vec!["public.t:2"]);
        client
            .upsert_vectors(&upsert("public.t:1", &["a", "b"]))
            .expect("upsert again");

        let listed = client
            .list_all_namespaces("idx", Some("public.t:"))
            .expect("list namespaces");
        let names: Vec<&str> = listed.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(names, ["public.t:1", "public.t:2"]);
        assert_eq!(listed[0].record_count, Some(2));
        let first_page = ListNamespacesRequest {
            index: "idx".to_string(),
            prefix: Some("public.t:".to_string()),
            limit: Some(1),
            pagination_token: None,
        };
        let page = client.list_namespaces(&first_page).expect("one page");
        assert_eq!((page.namespaces.len(), page.next.is_some()), (1, true));
        let described = client
            .describe_namespace("idx", "public.t:2")
            .expect("describe");
        assert_eq!(described.record_count, Some(1));
        let missing = client.describe_namespace("idx", "public.t:9");
        assert!(
            matches!(missing, Err(Error::NamespaceNotFound { .. })),
            "{missing:?}"
        );

        let by_filter = DeleteVectorsRequest {
            index: "idx".to_string(),
            namespace: Some("public.t:2".to_string()),
            ids: None,
            delete_all: false,
            filter: Some(json!({"tenant": {"$eq": "public.t:2"}})),
        };
        client.delete_vectors(&by_filter).expect("delete by filter");
        assert_eq!(vectors.namespaces(), vec!["public.t:1"]);
        client
            .delete_namespace("idx", "public.t:1")
            .expect("delete namespace");
        client
            .delete_namespace("idx", "public.t:1")
            .expect("a namespace that is gone is a 404 the client accepts");
        assert!(vectors.namespaces().is_empty());

        vectors.set_query_unknown_namespace(Some(404));
        let answer = client
            .query(&query("public.t:1"))
            .expect("a 404 is an empty result");
        assert!(answer.is_empty());
        let calls = vectors.calls();
        assert!(
            calls
                .iter()
                .all(|c| c.api_version.as_deref() == Some(pg_pinecone_client::API_VERSION))
        );
    }
}
