//! Programmable stand-in for Pinecone's document-schema (full-text-search) API.
//!
//! Test infrastructure for the table-bound FTS tests: one [`DocsHandle`] holds the
//! state (indexes, documents per namespace, recorded calls, fault rules) and is
//! shared between the Postgres backend thread, which drives the extension and
//! inspects the state, and the mock servers' thread, which answers the HTTP calls.
//!
//! What it answers (control plane / data plane of one [`MockBackend`]):
//!
//! - `POST /indexes`, `GET|DELETE /indexes/{name}`: create (201, 409 if it exists),
//!   describe and delete a document index; the schema is validated like the real
//!   server does and described back with the server-applied analyzer keys.
//! - `GET /namespaces/{ns}`: record count (404 for a namespace that holds nothing).
//! - `GET /namespaces` (`prefix`, `limit`, `paginationToken`) and `DELETE /namespaces/{ns}`:
//!   the namespace operations (`{namespaces: [{name, record_count}], pagination, total_count}`,
//!   `record_count` a decimal string as Pinecone sends it; 404 on a delete of a namespace that
//!   holds nothing).
//! - `POST /namespaces/{ns}/documents/{upsert,delete,search}`: an in-memory
//!   document store with Pinecone's documented limits enforced, and canned search
//!   matches. Search clauses must have the 2026-07 shape (`fields` arrays; the
//!   deprecated singular `field` is refused on purpose, so a regression shows up in
//!   every search test). `update` and `fetch` are recorded and answered minimally so
//!   tests can assert that the sync path never calls them.
//!
//! Faults: any endpoint can be told to answer 400/429/5xx for every call, for the
//! next N calls, for the Nth call, or only for requests that touch given document
//! ids (an upsert request fails atomically, as the real API does). Every call is
//! recorded with the `X-Pinecone-Api-Version` header and the parsed JSON body.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{header, path_regex},
};

use super::MockBackend;

/// The `X-Pinecone-Api-Version` every document call must carry. Deliberately a
/// literal and not `pg_pinecone_client::DOC_API_VERSION`: the tests exist to catch a
/// call that sends anything else, including a regression of that constant.
pub const EXPECTED_API_VERSION: &str = "2026-07";

/// Documented per-request and per-document limits (Pinecone docs, 2026-09-29).
const MAX_DOCS_PER_REQUEST: usize = 1000;
const MAX_REQUEST_BYTES: usize = 2 * 1024 * 1024;
const MAX_ID_BYTES: usize = 512;
const MAX_FIELD_NAME_BYTES: usize = 64;
const MAX_FTS_FIELD_BYTES: usize = 100 * 1024;
const MAX_FTS_FIELD_TOKENS: usize = 10_000;
const MAX_METADATA_BYTES: usize = 40 * 1024;
const MAX_TOP_K: u64 = 10_000;
const MAX_DELETE_IDS: usize = 1000;

/// The endpoints the mock distinguishes, for fault rules and call assertions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    /// `POST /indexes`
    IndexCreate,
    /// `GET /indexes/{name}`
    IndexDescribe,
    /// `DELETE /indexes/{name}`
    IndexDelete,
    /// `GET /namespaces/{ns}`
    NamespaceDescribe,
    /// `GET /namespaces`
    NamespaceList,
    /// `DELETE /namespaces/{ns}`
    NamespaceDelete,
    /// `POST /namespaces/{ns}/documents/upsert`
    Upsert,
    /// `POST /namespaces/{ns}/documents/delete`
    Delete,
    /// `POST /namespaces/{ns}/documents/search`
    Search,
    /// `POST /namespaces/{ns}/documents/update` (not used by sync)
    Update,
    /// `POST /namespaces/{ns}/documents/fetch` (not used by sync)
    Fetch,
    /// Anything else; answered 404.
    Unknown,
}

/// One request the mock received.
#[derive(Clone, Debug)]
pub struct DocsCall {
    pub endpoint: Endpoint,
    pub method: String,
    pub path: String,
    /// Percent-decoded namespace of a data-plane call.
    pub namespace: Option<String>,
    /// Index name of a control-plane call (`DELETE`/`GET /indexes/{name}`, or the body's `name` on create).
    pub index: Option<String>,
    /// The `X-Pinecone-Api-Version` header as sent; `None` if absent.
    pub api_version: Option<String>,
    /// Decoded query string pairs, in order.
    #[allow(dead_code)]
    pub query: Vec<(String, String)>,
    pub body: Option<Value>,
    /// The status the mock answered with.
    pub status: u16,
}

impl DocsCall {
    #[allow(dead_code)]
    pub fn param(&self, key: &str) -> Option<&str> {
        let pair = self.query.iter().find(|(k, _)| k == key);
        pair.map(|(_, v)| v.as_str())
    }
}

/// A fault rule: which calls to fail and how.
struct Rule {
    endpoint: Endpoint,
    /// Matching calls to let through before the rule starts to fire.
    skip: usize,
    /// Firings left; `None` fires until the rule is healed.
    remaining: Option<usize>,
    /// If set, only calls that touch at least one of these document ids match.
    ids: Option<BTreeSet<String>>,
    status: u16,
    message: String,
}

#[derive(Default)]
struct State {
    calls: Vec<DocsCall>,
    /// Index name -> schema as the server describes it (`{"fields": {..}}`).
    indexes: BTreeMap<String, Value>,
    /// The schema document validation runs against: the last index created or preloaded.
    active_schema: Option<Value>,
    /// Namespace -> document id -> document.
    docs: BTreeMap<String, BTreeMap<String, Value>>,
    /// Namespace -> record count to report instead of the stored count.
    record_count_override: BTreeMap<String, u64>,
    search_matches: Vec<(String, f32)>,
    /// Namespace -> matches its searches answer with instead of `search_matches`.
    namespace_matches: BTreeMap<String, Vec<(String, f32)>>,
    /// Status a search answers for a namespace that holds nothing and has no canned matches.
    search_missing_status: Option<u16>,
    rules: Vec<Rule>,
    latency: Vec<(Endpoint, Duration)>,
    lenient: bool,
    data_plane_uri: String,
}

/// Shared handle to the mock's state; clone it freely.
#[derive(Clone, Default)]
pub struct DocsHandle {
    state: Arc<Mutex<State>>,
}

impl DocsHandle {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("docs mock state poisoned")
    }

    // ---- recorded calls ----

    /// Every call received so far, in arrival order.
    pub fn calls(&self) -> Vec<DocsCall> {
        self.lock().calls.clone()
    }

    pub fn calls_to(&self, endpoint: Endpoint) -> Vec<DocsCall> {
        let state = self.lock();
        let calls = state.calls.iter();
        calls.filter(|c| c.endpoint == endpoint).cloned().collect()
    }

    pub fn count(&self, endpoint: Endpoint) -> usize {
        self.calls_to(endpoint).len()
    }

    /// Forgets the recorded calls (not the stored documents or indexes).
    pub fn clear_calls(&self) {
        self.lock().calls.clear();
    }

    /// Request bodies of the upsert calls, in arrival order, failed ones included.
    #[allow(dead_code)] // used by the backfill and search tests (later changes)
    pub fn upsert_bodies(&self) -> Vec<Value> {
        let calls = self.calls_to(Endpoint::Upsert);
        calls.into_iter().filter_map(|c| c.body).collect()
    }

    // ---- remote state ----

    /// Ids of the documents the mock currently holds in `namespace`.
    pub fn remote_ids(&self, namespace: &str) -> BTreeSet<String> {
        let state = self.lock();
        let docs = state.docs.get(namespace);
        docs.map(|d| d.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Names of the namespaces that hold at least one document, sorted.
    pub fn namespaces(&self) -> Vec<String> {
        let state = self.lock();
        let held = state.docs.iter().filter(|(_, docs)| !docs.is_empty());
        held.map(|(ns, _)| ns.clone()).collect()
    }

    pub fn remote_doc(&self, namespace: &str, id: &str) -> Option<Value> {
        let state = self.lock();
        state.docs.get(namespace)?.get(id).cloned()
    }

    /// Simulates a lost or restored-empty index: drops every document of `namespace`.
    #[allow(dead_code)] // used by the backfill and search tests (later changes)
    pub fn wipe_namespace(&self, namespace: &str) {
        self.lock().docs.remove(namespace);
    }

    /// Makes `name` exist with `fields` (the `schema.fields` object) as a prior
    /// `POST /indexes` would have stored it, server-applied keys included.
    pub fn preload_index(&self, name: &str, fields: Value) {
        let schema = describe_schema(&json!({ "fields": fields }));
        let mut state = self.lock();
        state.active_schema = Some(schema.clone());
        state.indexes.insert(name.to_string(), schema);
    }

    /// Forgets index `name`, as a deletion in the console would; its documents stay in the store.
    pub fn drop_index(&self, name: &str) {
        self.lock().indexes.remove(name);
    }

    /// Reports `count` for `GET /namespaces/{namespace}` regardless of what is stored.
    #[allow(dead_code)] // used by the backfill and search tests (later changes)
    pub fn set_namespace_record_count(&self, namespace: &str, count: u64) {
        let mut state = self.lock();
        state
            .record_count_override
            .insert(namespace.to_string(), count);
    }

    /// The matches every search answers with, best first.
    pub fn set_search_matches(&self, matches: &[(&str, f32)]) {
        let owned = matches.iter().map(|(id, s)| (id.to_string(), *s));
        self.lock().search_matches = owned.collect();
    }

    /// The matches searches of `namespace` answer with (an empty list: no matches), best first;
    /// other namespaces keep the matches of [`Self::set_search_matches`].
    pub fn set_search_matches_in(&self, namespace: &str, matches: &[(&str, f32)]) {
        let owned = matches.iter().map(|(id, s)| (id.to_string(), *s));
        let owned = owned.collect();
        self.lock()
            .namespace_matches
            .insert(namespace.to_string(), owned);
    }

    /// A search of a namespace that holds no document and has no canned matches answers `status`
    /// (`Some(404)`: the not-found shape) instead of the global matches.
    pub fn set_search_missing_namespace(&self, status: Option<u16>) {
        self.lock().search_missing_status = status;
    }

    /// Stores a minimal document per id in `namespace`, as an earlier writer would have left it.
    pub fn preload_documents(&self, namespace: &str, ids: &[&str]) {
        let mut state = self.lock();
        let store = state.docs.entry(namespace.to_string()).or_default();
        for id in ids {
            store.insert(id.to_string(), json!({ "_id": id }));
        }
    }

    /// Namespaces of the delete calls that asked for `delete_all`, in arrival order.
    pub fn delete_all_namespaces(&self) -> Vec<String> {
        let calls = self.calls_to(Endpoint::Delete);
        let wipes = calls.into_iter().filter(|c| {
            c.body
                .as_ref()
                .is_some_and(|b| b["delete_all"] == json!(true))
        });
        wipes.filter_map(|c| c.namespace).collect()
    }

    /// Turns off limit/schema validation (everything valid is stored as sent). Unused by the
    /// feature tests so far; for tests that need to observe what an invalid request would do.
    #[allow(dead_code)]
    pub fn set_lenient(&self, lenient: bool) {
        self.lock().lenient = lenient;
    }

    // ---- faults and latency ----

    /// Every call to `endpoint` answers `status` until [`Self::heal`].
    pub fn fail_always(&self, endpoint: Endpoint, status: u16, message: &str) {
        self.add_rule(endpoint, 0, None, None, status, message);
    }

    /// The next `n` calls to `endpoint` answer `status`.
    pub fn fail_next(&self, endpoint: Endpoint, n: usize, status: u16, message: &str) {
        self.add_rule(endpoint, 0, Some(n), None, status, message);
    }

    /// Only the `nth` call to `endpoint` from now on (1-based) answers `status`.
    #[allow(dead_code)] // used by the backfill and search tests (later changes)
    pub fn fail_nth(&self, endpoint: Endpoint, nth: usize, status: u16, message: &str) {
        let skip = nth.saturating_sub(1);
        self.add_rule(endpoint, skip, Some(1), None, status, message);
    }

    /// Every call to `endpoint` that touches one of `ids` (documents of an upsert,
    /// `ids` of a delete) answers `status`; the whole request fails, nothing is applied.
    pub fn fail_ids(&self, endpoint: Endpoint, ids: &[&str], status: u16, message: &str) {
        let ids = ids.iter().map(|id| id.to_string()).collect();
        self.add_rule(endpoint, 0, None, Some(ids), status, message);
    }

    /// Removes every fault rule of `endpoint`.
    pub fn heal(&self, endpoint: Endpoint) {
        self.lock().rules.retain(|r| r.endpoint != endpoint);
    }

    pub fn heal_all(&self) {
        self.lock().rules.clear();
    }

    /// Delays every answer of `endpoint` by `delay` (asynchronously, so other calls are not blocked).
    #[allow(dead_code)] // used by the backfill and search tests (later changes)
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
        ids: Option<BTreeSet<String>>,
        status: u16,
        message: &str,
    ) {
        let message = message.to_string();
        let rule = Rule {
            endpoint,
            skip,
            remaining,
            ids,
            status,
            message,
        };
        self.lock().rules.push(rule);
    }
}

/// A [`MockBackend`] whose control plane and data plane are both served by `docs`.
pub fn start_with_docs(docs: &DocsHandle) -> MockBackend {
    let docs = docs.clone();
    MockBackend::start(move |control_plane, data_plane| async move {
        docs.mount(&control_plane, &data_plane).await;
        (control_plane, data_plane)
    })
}

impl DocsHandle {
    /// Installs the handlers; call it inside [`MockBackend::start`]'s setup.
    pub async fn mount(&self, control_plane: &MockServer, data_plane: &MockServer) {
        self.mount_filtered(control_plane, data_plane, None).await;
    }

    /// Like [`Self::mount`], but only answers requests that carry `version` as
    /// `X-Pinecone-Api-Version`, so vector-API mocks (2026-04) mounted after it
    /// can share the same servers.
    pub async fn mount_for_version(&self, cp: &MockServer, dp: &MockServer, version: &str) {
        self.mount_filtered(cp, dp, Some(version)).await;
    }

    async fn mount_filtered(&self, cp: &MockServer, dp: &MockServer, version: Option<&str>) {
        self.lock().data_plane_uri = dp.uri();
        for (server, pattern) in [(cp, r"^/indexes(/[^/]*)?$"), (dp, r"^/namespaces(/.*)?$")] {
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

    fn handle(&self, req: &Request) -> ResponseTemplate {
        let mut route = parse_route(req.method.as_str(), req.url.path());
        route.query = req
            .url
            .query_pairs()
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();
        let body = serde_json::from_slice::<Value>(&req.body).ok();
        let version = req.headers.get("x-pinecone-api-version");
        let api_version = version.and_then(|v| v.to_str().ok()).map(str::to_string);

        let mut state = self.lock();
        let ids = touched_ids(route.endpoint, body.as_ref());
        let reply = match take_fault(&mut state, route.endpoint, &ids) {
            Some((status, message)) => error_reply(status, &message),
            None => dispatch(&mut state, &route, body.as_ref(), req.body.len()),
        };
        state.calls.push(DocsCall {
            endpoint: route.endpoint,
            method: req.method.to_string(),
            path: req.url.path().to_string(),
            namespace: route.namespace.clone(),
            index: route.index.clone().or_else(|| created_name(&body)),
            api_version,
            query: route.query.clone(),
            body,
            status: reply.status,
        });
        let delay = state.latency.iter().find(|(e, _)| *e == route.endpoint);
        let delay = delay.map(|(_, d)| *d);
        drop(state);

        let template = ResponseTemplate::new(reply.status).set_body_json(reply.body);
        match delay {
            Some(d) => template.set_delay(d),
            None => template,
        }
    }
}

struct Reply {
    status: u16,
    body: Value,
}

fn ok(status: u16, body: Value) -> Reply {
    Reply { status, body }
}

/// Pinecone's error envelope: `{"error": {"code", "message"}, "status"}`.
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
    ok(status, json!({ "error": error, "status": status }))
}

struct Route {
    endpoint: Endpoint,
    index: Option<String>,
    namespace: Option<String>,
    query: Vec<(String, String)>,
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
        ("GET", ["namespaces"]) => Endpoint::NamespaceList,
        ("GET", ["namespaces", ns]) => {
            namespace = Some(percent_decode(ns));
            Endpoint::NamespaceDescribe
        }
        ("DELETE", ["namespaces", ns]) => {
            namespace = Some(percent_decode(ns));
            Endpoint::NamespaceDelete
        }
        ("POST", ["namespaces", ns, "documents", op]) => {
            namespace = Some(percent_decode(ns));
            match *op {
                "upsert" => Endpoint::Upsert,
                "delete" => Endpoint::Delete,
                "search" => Endpoint::Search,
                "update" => Endpoint::Update,
                "fetch" => Endpoint::Fetch,
                _ => Endpoint::Unknown,
            }
        }
        _ => Endpoint::Unknown,
    };
    Route {
        endpoint,
        index,
        namespace,
        query: Vec::new(),
    }
}

fn created_name(body: &Option<Value>) -> Option<String> {
    let body = body.as_ref()?;
    Some(body.get("name")?.as_str()?.to_string())
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

/// The document ids a request touches: upsert documents, delete ids.
fn touched_ids(endpoint: Endpoint, body: Option<&Value>) -> BTreeSet<String> {
    let strings = |v: &Value| -> BTreeSet<String> {
        let items = v.as_array().into_iter().flatten();
        items
            .filter_map(|i| i.as_str().map(str::to_string))
            .collect()
    };
    match (endpoint, body) {
        (Endpoint::Upsert, Some(body)) => {
            let docs = body["documents"].as_array().into_iter().flatten();
            docs.filter_map(|d| d["_id"].as_str().map(str::to_string))
                .collect()
        }
        (Endpoint::Delete, Some(body)) => strings(&body["ids"]),
        _ => BTreeSet::new(),
    }
}

/// The fault to apply to this call, if a rule fires.
fn take_fault(
    state: &mut State,
    endpoint: Endpoint,
    ids: &BTreeSet<String>,
) -> Option<(u16, String)> {
    let mut fired = None;
    for rule in state.rules.iter_mut().filter(|r| r.endpoint == endpoint) {
        if let Some(rule_ids) = &rule.ids
            && rule_ids.is_disjoint(ids)
        {
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
        fired = Some((rule.status, rule.message.clone()));
        break;
    }
    state.rules.retain(|r| r.remaining != Some(0));
    fired
}

fn dispatch(state: &mut State, route: &Route, body: Option<&Value>, size: usize) -> Reply {
    let (index, ns) = (route.index.as_deref(), route.namespace.as_deref());
    match (route.endpoint, index, ns) {
        (Endpoint::IndexCreate, _, _) => create_index(state, body),
        (Endpoint::IndexDescribe, Some(name), _) => describe_index(state, name),
        (Endpoint::IndexDelete, Some(name), _) => delete_index(state, name),
        (Endpoint::NamespaceDescribe, _, Some(ns)) => describe_namespace(state, ns),
        (Endpoint::NamespaceList, _, _) => list_namespaces(state, &route.query),
        (Endpoint::NamespaceDelete, _, Some(ns)) => delete_namespace(state, ns),
        (Endpoint::Upsert, _, Some(ns)) => upsert(state, ns, body, size),
        (Endpoint::Delete, _, Some(ns)) => delete(state, ns, body),
        (Endpoint::Search, _, ns) => search(state, ns.unwrap_or_default(), body),
        (Endpoint::Update, _, _) => ok(202, json!({})),
        (Endpoint::Fetch, _, Some(ns)) => ok(200, json!({ "documents": {}, "namespace": ns })),
        _ => error_reply(404, "mock: no such document-API route"),
    }
}

// ---- control plane ----

const LANGUAGES: &[&str] = &[
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "hu", "it", "nl", "no", "pt", "ro", "ru", "sv",
    "ta", "tr",
];

fn index_model(state: &State, name: &str, schema: &Value) -> Value {
    json!({
        "name": name,
        "host": state.data_plane_uri,
        "status": { "ready": true, "state": "Ready" },
        "schema": schema,
        "deletion_protection": "disabled",
        "read_capacity": { "mode": "OnDemand" },
        "deployment": { "deployment_type": "managed", "cloud": "aws", "region": "us-east-1" },
    })
}

fn create_index(state: &mut State, body: Option<&Value>) -> Reply {
    let Some(body) = body else {
        return error_reply(400, "mock: request body must be JSON");
    };
    let name = body["name"].as_str().unwrap_or_default();
    if !state.lenient {
        let valid = validate_index_name(name).and_then(|()| validate_schema(&body["schema"]));
        if let Err(message) = valid {
            return error_reply(400, &message);
        }
    }
    if state.indexes.contains_key(name) {
        return error_reply(409, &format!("Resource {name} already exists"));
    }
    let schema = describe_schema(&body["schema"]);
    state.indexes.insert(name.to_string(), schema.clone());
    state.active_schema = Some(schema.clone());
    ok(201, index_model(state, name, &schema))
}

fn describe_index(state: &State, name: &str) -> Reply {
    match state.indexes.get(name) {
        Some(schema) => ok(200, index_model(state, name, schema)),
        None => error_reply(404, &format!("Index {name} not found")),
    }
}

fn delete_index(state: &mut State, name: &str) -> Reply {
    if state.indexes.remove(name).is_none() {
        return error_reply(404, &format!("Index {name} not found"));
    }
    if state.indexes.is_empty() {
        state.docs.clear();
        state.active_schema = None;
    }
    ok(202, json!({}))
}

fn validate_index_name(name: &str) -> Result<(), String> {
    let valid_chars = name
        .bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
    if name.is_empty() || name.len() > 45 || !valid_chars {
        return Err(format!(
            "Index name '{name}' is invalid: use 1-45 lowercase letters, digits or hyphens"
        ));
    }
    Ok(())
}

/// The schema as the server describes it: analyzer defaults filled in plus the
/// server-applied keys (`lowercase`, `max_term_len`) a naive equality check trips over.
fn describe_schema(schema: &Value) -> Value {
    let mut described = schema.clone();
    let fields = described["fields"].as_object_mut().into_iter().flatten();
    for (_, field) in fields {
        let Some(fts) = field.get_mut("full_text_search") else {
            continue;
        };
        if !fts.is_object() {
            *fts = json!({});
        }
        let options = fts.as_object_mut().expect("just made an object");
        options.entry("language").or_insert(json!("en"));
        options.entry("stemming").or_insert(json!(false));
        options.entry("stop_words").or_insert(json!(false));
        options.entry("lowercase").or_insert(json!(true));
        options.entry("max_term_len").or_insert(json!(40));
    }
    described
}

fn check_field_name(name: &str) -> Result<(), String> {
    if name.is_empty() || name.starts_with('_') || name.starts_with('$') {
        return Err(format!(
            "Field name '{name}' is invalid: it must not start with '_' or '$'"
        ));
    }
    if name.len() > MAX_FIELD_NAME_BYTES {
        return Err(format!(
            "Field name '{name}' exceeds {MAX_FIELD_NAME_BYTES} bytes"
        ));
    }
    Ok(())
}

/// The server-side schema rules a derived schema has to satisfy.
fn validate_schema(schema: &Value) -> Result<(), String> {
    let Some(fields) = schema["fields"].as_object() else {
        return Err("schema.fields must be an object".to_string());
    };
    let (mut text, mut dense, mut sparse) = (0, 0, 0);
    for (name, field) in fields {
        check_field_name(name)?;
        match field["type"].as_str() {
            Some("string") if field.get("full_text_search").is_some() => {
                text += 1;
                validate_fts_options(name, &field["full_text_search"])?;
            }
            Some("dense_vector") => {
                dense += 1;
                let metric = field["metric"].as_str().unwrap_or_default();
                if !field["dimension"].as_u64().is_some_and(|d| d > 0)
                    || !["cosine", "dotproduct", "euclidean"].contains(&metric)
                {
                    return Err(format!(
                        "dense_vector '{name}' needs a dimension and a metric"
                    ));
                }
            }
            Some("sparse_vector") => sparse += 1,
            _ => {
                return Err(format!(
                    "The schema only accepts fields used for search (field types `dense_vector`, \
                     `sparse_vector`, and `string` with `full_text_search` configuration). To use \
                     field '{name}' for filtering (field types `boolean`, `float`, `string`, or \
                     `string_list`), omit it from the schema and include it in documents."
                ));
            }
        }
    }
    if text > 100 || dense > 1 || sparse > 1 {
        return Err(
            "a schema holds at most 100 full-text fields, one dense and one sparse vector".into(),
        );
    }
    Ok(())
}

fn validate_fts_options(field: &str, fts: &Value) -> Result<(), String> {
    let empty = serde_json::Map::new();
    let options = match fts {
        Value::Bool(true) => &empty,
        Value::Object(options) => options,
        _ => {
            return Err(format!(
                "field '{field}': full_text_search must be true or an object"
            ));
        }
    };
    for key in options.keys() {
        if !["language", "stemming", "stop_words", "ngram"].contains(&key.as_str()) {
            return Err(format!(
                "field '{field}': unknown full_text_search option '{key}'"
            ));
        }
    }
    let flag = |key: &str| options.get(key).and_then(Value::as_bool).unwrap_or(false);
    if let Some(language) = options.get("language").and_then(Value::as_str)
        && !LANGUAGES.contains(&language)
    {
        return Err(format!(
            "field '{field}': unsupported language '{language}'"
        ));
    }
    if options.contains_key("ngram") && (flag("stemming") || flag("stop_words")) {
        return Err(format!(
            "field '{field}': ngram cannot be combined with stemming or stop_words"
        ));
    }
    if flag("stop_words") && !flag("stemming") {
        return Err(format!("field '{field}': stop_words requires stemming"));
    }
    Ok(())
}

// ---- data plane ----

fn describe_namespace(state: &State, ns: &str) -> Reply {
    if let Some(count) = state.record_count_override.get(ns) {
        return ok(
            200,
            json!({ "name": ns, "record_count": count.to_string() }),
        );
    }
    match state.docs.get(ns) {
        Some(docs) => ok(
            200,
            json!({ "name": ns, "record_count": docs.len().to_string() }),
        ),
        None => error_reply(404, &format!("Namespace {ns} not found")),
    }
}

fn list_namespaces(state: &State, query: &[(String, String)]) -> Reply {
    let param = |key: &str| {
        query
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    };
    let limit = match param("limit").map(str::parse::<usize>) {
        None => 100,
        Some(Ok(n)) if (1..=100).contains(&n) => n,
        Some(_) => return error_reply(400, "limit must be between 1 and 100"),
    };
    let prefix = param("prefix").unwrap_or_default();
    let after = param("paginationToken").and_then(|t| t.strip_prefix("after:"));
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (ns, docs) in &state.docs {
        counts.insert(ns, docs.len());
    }
    for (ns, count) in &state.record_count_override {
        counts.insert(ns, *count as usize);
    }
    counts.retain(|ns, _| ns.starts_with(prefix));
    let total = counts.len();
    let mut page: Vec<_> = counts
        .into_iter()
        .filter(|(ns, _)| after.is_none_or(|a| *ns > a))
        .take(limit + 1)
        .collect();
    let more = page.len() > limit;
    page.truncate(limit);
    let listed: Vec<Value> = page
        .iter()
        .map(|(ns, count)| json!({ "name": ns, "record_count": count.to_string() }))
        .collect();
    let mut body = json!({ "namespaces": listed, "total_count": total });
    if let (true, Some((last, _))) = (more, page.last()) {
        body["pagination"] = json!({ "next": format!("after:{last}") });
    }
    ok(200, body)
}

fn delete_namespace(state: &mut State, ns: &str) -> Reply {
    let stored = state.docs.remove(ns).is_some();
    let counted = state.record_count_override.remove(ns).is_some();
    if stored || counted {
        ok(200, json!({}))
    } else {
        error_reply(404, &format!("Namespace {ns} not found"))
    }
}

fn upsert(state: &mut State, ns: &str, body: Option<&Value>, size: usize) -> Reply {
    let Some(docs) = body.and_then(|b| b["documents"].as_array()) else {
        return error_reply(400, "documents must be an array");
    };
    if !state.lenient {
        let count_ok = (1..=MAX_DOCS_PER_REQUEST).contains(&docs.len());
        if !count_ok || size > MAX_REQUEST_BYTES {
            return error_reply(
                400,
                &format!(
                    "a request holds 1-{MAX_DOCS_PER_REQUEST} documents and at most 2 MB (got {} documents, {size} bytes)",
                    docs.len()
                ),
            );
        }
        for doc in docs {
            if let Err(message) = validate_document(doc, state.active_schema.as_ref()) {
                return error_reply(400, &message);
            }
        }
    }
    let store = state.docs.entry(ns.to_string()).or_default();
    for doc in docs {
        if let Some(id) = doc["_id"].as_str() {
            store.insert(id.to_string(), doc.clone());
        }
    }
    ok(202, json!({ "upserted_count": docs.len() }))
}

fn metadata_value_ok(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::Number(_) | Value::Bool(_) => true,
        Value::Array(items) => items.iter().all(Value::is_string),
        Value::Null | Value::Object(_) => false,
    }
}

fn validate_document(doc: &Value, schema: Option<&Value>) -> Result<(), String> {
    let Some(fields) = doc.as_object() else {
        return Err("each document must be an object".to_string());
    };
    let id = fields
        .get("_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if id.is_empty() || id.len() > MAX_ID_BYTES {
        return Err(format!(
            "Document _id must be 1-{MAX_ID_BYTES} bytes (got {})",
            id.len()
        ));
    }
    let declared_fields = schema.and_then(|s| s["fields"].as_object());
    let mut metadata_bytes = 0;
    for (name, value) in fields.iter().filter(|(name, _)| *name != "_id") {
        check_field_name(name).map_err(|m| format!("Document '{id}': {m}"))?;
        let declared = declared_fields.and_then(|f| f.get(name));
        match declared.and_then(|f| f["type"].as_str()) {
            Some("string") => {
                let Some(text) = value.as_str() else {
                    return Err(format!("Document '{id}': field '{name}' must be a string"));
                };
                if text.len() > MAX_FTS_FIELD_BYTES {
                    return Err(format!(
                        "Document '{id}': field '{name}' is {} bytes, over the 100 KB limit",
                        text.len()
                    ));
                }
                let tokens = text.split_whitespace().count();
                if tokens > MAX_FTS_FIELD_TOKENS {
                    return Err(format!(
                        "Document '{id}': field '{name}' has {tokens} tokens, over the 10,000 token limit"
                    ));
                }
            }
            Some("dense_vector") => {
                let dimension = declared.map_or(0, |f| f["dimension"].as_u64().unwrap_or(0));
                let numbers = value.as_array().filter(|a| a.iter().all(Value::is_number));
                if numbers.map(|a| a.len() as u64) != Some(dimension) {
                    return Err(format!(
                        "Document '{id}': field '{name}' must be an array of {dimension} numbers"
                    ));
                }
            }
            Some(_) => {}
            None if metadata_value_ok(value) => {
                metadata_bytes += name.len() + value.to_string().len()
            }
            None => {
                return Err(format!(
                    "Document '{id}': field '{name}' has an unsupported value (use a string, number, boolean or list of strings; omit nulls)"
                ));
            }
        }
    }
    if metadata_bytes > MAX_METADATA_BYTES {
        return Err(format!(
            "Document '{id}': filterable metadata is {metadata_bytes} bytes, over the 40 KB limit"
        ));
    }
    Ok(())
}

fn delete(state: &mut State, ns: &str, body: Option<&Value>) -> Reply {
    let Some(body) = body else {
        return error_reply(400, "mock: request body must be JSON");
    };
    let forms = ["ids", "filter", "delete_all"];
    if forms.iter().filter(|k| body.get(**k).is_some()).count() != 1 {
        return error_reply(400, "exactly one of ids, filter or delete_all is required");
    }
    if let Some(ids) = body["ids"].as_array() {
        if !state.lenient && !(1..=MAX_DELETE_IDS).contains(&ids.len()) {
            return error_reply(
                400,
                &format!("ids must hold 1-{MAX_DELETE_IDS} ids (got {})", ids.len()),
            );
        }
        if let Some(store) = state.docs.get_mut(ns) {
            ids.iter()
                .filter_map(Value::as_str)
                .for_each(|id| drop(store.remove(id)));
        }
        return ok(202, json!({}));
    }
    if body["delete_all"] == json!(true) {
        if let Some(store) = state.docs.get_mut(ns) {
            store.clear();
        }
        return ok(202, json!({}));
    }
    // A filter delete: nothing is evaluated, the call is only recorded.
    ok(202, json!({ "matched_records": 0 }))
}

fn search(state: &State, ns: &str, body: Option<&Value>) -> Reply {
    let Some(body) = body else {
        return error_reply(400, "mock: request body must be JSON");
    };
    if !state.lenient
        && let Err(message) = validate_search(body)
    {
        return error_reply(400, &message);
    }
    let holds_documents = state.docs.get(ns).is_some_and(|d| !d.is_empty());
    if let Some(status) = state.search_missing_status
        && !holds_documents
        && !state.namespace_matches.contains_key(ns)
    {
        return error_reply(status, &format!("Namespace {ns} not found"));
    }
    let top_k = body["top_k"].as_u64().unwrap_or(10) as usize;
    let canned = state.namespace_matches.get(ns);
    let matches = canned.unwrap_or(&state.search_matches).iter().take(top_k);
    let matches: Vec<Value> = matches
        .map(|(id, score)| json!({ "_id": id, "_score": score }))
        .collect();
    ok(
        200,
        json!({ "matches": matches, "usage": { "read_units": 1 } }),
    )
}

fn validate_search(body: &Value) -> Result<(), String> {
    if !body["top_k"]
        .as_u64()
        .is_some_and(|k| (1..=MAX_TOP_K).contains(&k))
    {
        return Err(format!("top_k must be between 1 and {MAX_TOP_K}"));
    }
    if !body["include_fields"].is_array() {
        return Err("include_fields is required (use [] for ids and scores only)".to_string());
    }
    let clauses = body["score_by"].as_array().filter(|c| !c.is_empty());
    let clauses = clauses.ok_or("score_by must be a non-empty array")?;
    let types: BTreeSet<&str> = clauses.iter().filter_map(|c| c["type"].as_str()).collect();
    let known = ["text", "query_string", "dense_vector", "sparse_vector"];
    if types.len() != 1 || !types.iter().all(|t| known.contains(t)) {
        return Err(format!(
            "a request uses exactly one scoring type, got {types:?}"
        ));
    }
    clauses.iter().try_for_each(validate_clause)?;
    if body.get("filter").is_some_and(|f| !f.is_object()) {
        return Err("filter must be an object".to_string());
    }
    Ok(())
}

/// One `score_by` clause as the 2026-07 API (`DocumentScoringMethod`) takes it: `text` names one
/// or more fields and a query, `dense_vector` exactly one field and its values, `query_string` a
/// query and no field at all (qualifiers live in the query).
fn validate_clause(clause: &Value) -> Result<(), String> {
    let kind = clause["type"].as_str().unwrap_or_default();
    if clause.get("field").is_some() {
        return Err(format!(
            "{kind}: the singular `field` is deprecated, send a `fields` array"
        ));
    }
    let fields = clause.get("fields").map(|f| {
        let names = f.as_array().map(Vec::as_slice).unwrap_or_default();
        let strings = names.iter().all(Value::is_string);
        (names.len(), strings)
    });
    let query = clause["query"].as_str().map(str::trim);
    match kind {
        "text" | "query_string" if query.is_none_or(str::is_empty) => {
            Err(format!("{kind}: query must be a non-empty string"))
        }
        "text" if !fields.is_some_and(|(n, strings)| n >= 1 && strings) => {
            Err("text: fields must be an array of one or more field names".to_string())
        }
        "query_string" if fields.is_some() => Err(
            "query_string: fields is not allowed, qualify the field in the query (field:(...))"
                .to_string(),
        ),
        "dense_vector" if !fields.is_some_and(|(n, strings)| n == 1 && strings) => {
            Err("dense_vector: fields must name exactly one field".to_string())
        }
        "dense_vector" if !clause["values"].as_array().is_some_and(|v| !v.is_empty()) => {
            Err("dense_vector: values must be a non-empty array".to_string())
        }
        _ => Ok(()),
    }
}

/// Self-tests of the mock: plain HTTP against the handlers, no Postgres. They
/// are green by design (the feature tests that use this mock are the red ones).
#[cfg(test)]
mod tests {
    use super::*;

    fn agent() -> ureq::Agent {
        let config = ureq::Agent::config_builder().http_status_as_error(false);
        config.build().into()
    }

    /// Sends one request and returns `(status, json body)`.
    fn call(method: &str, url: &str, version: Option<&str>, body: Option<&Value>) -> (u16, Value) {
        let agent = agent();
        let header = |name: &str| version.map(|v| (name.to_string(), v.to_string()));
        let version_header = header("X-Pinecone-Api-Version");
        let result = match method {
            "GET" | "DELETE" => {
                let mut request = if method == "GET" {
                    agent.get(url)
                } else {
                    agent.delete(url)
                };
                if let Some((name, value)) = &version_header {
                    request = request.header(name, value);
                }
                request.call()
            }
            _ => {
                let mut request = agent.post(url);
                if let Some((name, value)) = &version_header {
                    request = request.header(name, value);
                }
                request.send_json(body.unwrap_or(&Value::Null))
            }
        };
        let mut response = result.expect("mock reachable");
        let status = response.status().as_u16();
        let body = response
            .body_mut()
            .read_json::<Value>()
            .unwrap_or(Value::Null);
        (status, body)
    }

    fn v(method: &str, url: &str, body: Option<&Value>) -> (u16, Value) {
        call(method, url, Some(EXPECTED_API_VERSION), body)
    }

    fn schema() -> Value {
        json!({ "fields": {
            "body": {"type": "string", "full_text_search": {"language": "en", "stemming": true, "stop_words": true}},
            "embedding": {"type": "dense_vector", "dimension": 2, "metric": "cosine"},
        }})
    }

    /// Starts the mock, creates index `idx` with [`schema`], returns `(control_url, data_url)`.
    fn started(docs: &DocsHandle) -> (MockBackend, String, String) {
        let backend = start_with_docs(docs);
        let control = backend.control_plane_url.clone();
        let create = json!({"name": "idx", "schema": schema()});
        let (status, created) = v("POST", &format!("{control}/indexes"), Some(&create));
        assert_eq!(status, 201, "create: {created}");
        let data = created["host"].as_str().expect("host").to_string();
        (backend, control, data)
    }

    #[test]
    fn index_lifecycle_and_schema_rules() {
        let docs = DocsHandle::new();
        let (_backend, control, _data) = started(&docs);
        let idx = format!("{control}/indexes/idx");

        let (status, body) = v("GET", &idx, None);
        assert_eq!(status, 200);
        let body_fts = &body["schema"]["fields"]["body"]["full_text_search"];
        assert_eq!(
            body_fts["lowercase"],
            json!(true),
            "server-applied keys are described back"
        );
        assert_eq!(body_fts["stemming"], json!(true));

        let (status, _) = v(
            "POST",
            &format!("{control}/indexes"),
            Some(&json!({"name": "idx", "schema": schema()})),
        );
        assert_eq!(status, 409);
        assert_eq!(v("GET", &format!("{control}/indexes/nope"), None).0, 404);

        let bad = |fields: Value| {
            let create = json!({"name": "other", "schema": {"fields": fields}});
            v("POST", &format!("{control}/indexes"), Some(&create))
        };
        let (status, body) = bad(json!({"author": {"type": "string"}}));
        assert_eq!(status, 400);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("omit it from the schema"),
            "{body}"
        );
        let ngram = json!({"type": "string", "full_text_search": {"ngram": {"min_gram": 2, "max_gram": 4}, "stemming": true}});
        assert_eq!(bad(json!({"sku": ngram})).0, 400, "ngram excludes stemming");
        let stop = json!({"type": "string", "full_text_search": {"stop_words": true}});
        assert_eq!(
            bad(json!({"sku": stop})).0,
            400,
            "stop_words needs stemming"
        );
        let named = json!({"type": "string", "full_text_search": {}});
        assert_eq!(bad(json!({"_sku": named})).0, 400, "leading underscore");

        assert_eq!(v("DELETE", &idx, None).0, 202);
        assert_eq!(v("GET", &idx, None).0, 404);

        docs.preload_index(
            "pre",
            json!({"title": {"type": "string", "full_text_search": true}}),
        );
        let (status, body) = v("GET", &format!("{control}/indexes/pre"), None);
        assert_eq!(status, 200);
        let fts = &body["schema"]["fields"]["title"]["full_text_search"];
        assert_eq!(
            fts["language"],
            json!("en"),
            "defaults are filled in: {body}"
        );
    }

    fn upsert_url(data: &str, ns: &str) -> String {
        format!("{data}/namespaces/{ns}/documents/upsert")
    }

    fn doc(id: &str, body: &str) -> Value {
        json!({"_id": id, "body": body, "embedding": [0.5, 0.25], "author": "ann"})
    }

    #[test]
    fn documents_store_delete_and_enforce_limits() {
        let docs = DocsHandle::new();
        let (_backend, _control, data) = started(&docs);
        let url = upsert_url(&data, "public.articles");
        let send =
            |documents: Vec<Value>| v("POST", &url, Some(&json!({ "documents": documents })));

        let (status, body) = send(vec![doc("1", "hello"), doc("2", "world")]);
        assert_eq!(
            (status, &body["upserted_count"]),
            (202, &json!(2)),
            "{body}"
        );
        assert_eq!(docs.remote_ids("public.articles").len(), 2);
        assert_eq!(
            docs.remote_doc("public.articles", "1"),
            Some(doc("1", "hello"))
        );
        assert_eq!(docs.remote_doc("public.articles", "9"), None);
        let ns = format!("{data}/namespaces/public.articles");
        assert_eq!(v("GET", &ns, None).1["record_count"], json!("2"));
        assert_eq!(v("GET", &format!("{data}/namespaces/other"), None).0, 404);
        docs.set_namespace_record_count("other", 12345);
        assert_eq!(
            v("GET", &format!("{data}/namespaces/other"), None).1["record_count"],
            json!("12345")
        );

        let delete = |body: Value| v("POST", &format!("{ns}/documents/delete"), Some(&body));
        assert_eq!(delete(json!({"ids": ["1"]})).0, 202);
        assert_eq!(
            docs.remote_ids("public.articles"),
            BTreeSet::from(["2".to_string()])
        );
        assert_eq!(delete(json!({})).0, 400, "one of ids/filter/delete_all");
        assert_eq!(delete(json!({"delete_all": true})).0, 202);
        assert!(docs.remote_ids("public.articles").is_empty());
        assert_eq!(
            v("GET", &ns, None).1["record_count"],
            json!("0"),
            "delete_all leaves an empty namespace"
        );
        send(vec![doc("1", "again")]);
        docs.wipe_namespace("public.articles");
        assert_eq!(
            v("GET", &ns, None).0,
            404,
            "a wiped namespace is gone, like a lost index"
        );

        let long_id = "x".repeat(513);
        let big = "x ".repeat(10_001);
        let mut null_meta = doc("5", "x");
        null_meta["note"] = Value::Null;
        let mut wrong_dim = doc("6", "x");
        wrong_dim["embedding"] = json!([1.0]);
        let mut reserved = doc("7", "x");
        reserved["_tag"] = json!("t");
        let mut fat = doc("8", "x");
        fat["author"] = json!("a".repeat(41_000));
        let cases = [
            ("long id", send(vec![doc(&long_id, "x")]).0),
            ("100 KB field", send(vec![doc("3", &"x".repeat(102_401))]).0),
            ("10,000 tokens", send(vec![doc("4", &big)]).0),
            ("null value", send(vec![null_meta]).0),
            ("dimension", send(vec![wrong_dim]).0),
            ("reserved name", send(vec![reserved]).0),
            ("40 KB metadata", send(vec![fat]).0),
            ("empty request", send(vec![]).0),
            (
                "1001 documents",
                send((0..1001).map(|i| doc(&i.to_string(), "x")).collect()).0,
            ),
        ];
        for (name, status) in cases {
            assert_eq!(status, 400, "{name} must be rejected");
        }
        assert!(
            docs.remote_ids("public.articles").is_empty(),
            "rejected requests store nothing"
        );
        docs.set_lenient(true);
        assert_eq!(
            send(vec![doc(&long_id, "x")]).0,
            202,
            "lenient mode stores what it is given"
        );
    }

    #[test]
    fn namespaces_are_listed_filtered_paginated_and_deleted() {
        let docs = DocsHandle::new();
        let (_backend, _control, data) = started(&docs);
        for tenant in 1..=5 {
            let url = upsert_url(&data, &format!("public.articles:{tenant}"));
            let documents = json!({"documents": [doc("1", "x"), doc("2", "y")]});
            assert_eq!(v("POST", &url, Some(&documents)).0, 202);
        }
        docs.preload_documents("public.articles2:1", &["a"]);
        docs.set_namespace_record_count("lagging", 7);

        let names = |body: &Value| -> Vec<String> {
            let listed = body["namespaces"].as_array().expect("namespaces");
            let names = listed.iter().map(|n| n["name"].as_str().expect("name"));
            names.map(str::to_string).collect()
        };
        let (status, all) = v("GET", &format!("{data}/namespaces"), None);
        assert_eq!((status, all["total_count"].clone()), (200, json!(7)));
        let url = format!("{data}/namespaces?prefix=public.articles:&limit=2");
        let (_, first) = v("GET", &url, None);
        assert_eq!(names(&first), ["public.articles:1", "public.articles:2"]);
        assert_eq!(first["namespaces"][0]["record_count"], json!("2"));
        assert_eq!(first["total_count"], json!(5));
        let token = first["pagination"]["next"]
            .as_str()
            .expect("next")
            .to_string();
        let (_, second) = v("GET", &format!("{url}&paginationToken={token}"), None);
        assert_eq!(names(&second), ["public.articles:3", "public.articles:4"]);
        let token = second["pagination"]["next"]
            .as_str()
            .expect("next")
            .to_string();
        let (_, last) = v("GET", &format!("{url}&paginationToken={token}"), None);
        assert_eq!(names(&last), ["public.articles:5"]);
        assert!(
            last.get("pagination").is_none(),
            "no token on the last page"
        );
        let (_, lagging) = v("GET", &format!("{data}/namespaces?prefix=lag"), None);
        assert_eq!(lagging["namespaces"][0]["record_count"], json!("7"));
        assert_eq!(
            v("GET", &format!("{data}/namespaces?limit=101"), None).0,
            400
        );

        let one = format!("{data}/namespaces/public.articles%3A3");
        assert_eq!(v("DELETE", &one, None).0, 200);
        assert_eq!(v("GET", &one, None).0, 404);
        assert_eq!(v("DELETE", &one, None).0, 404, "nothing left to delete");
        assert!(docs.remote_ids("public.articles:3").is_empty());
        assert_eq!(docs.count(Endpoint::NamespaceDelete), 2);
        let listed = docs.calls_to(Endpoint::NamespaceList);
        assert_eq!(listed[1].param("prefix"), Some("public.articles:"));
    }

    #[test]
    fn searches_use_per_namespace_matches_and_can_answer_not_found() {
        let docs = DocsHandle::new();
        let (_backend, _control, data) = started(&docs);
        docs.set_search_matches(&[("g", 1.0)]);
        docs.set_search_matches_in("ns:a", &[("a1", 2.0), ("a2", 1.0)]);
        docs.set_search_matches_in("ns:empty", &[]);
        let text = json!({"type": "text", "fields": ["body"], "query": "q"});
        let request = json!({"top_k": 5, "include_fields": [], "score_by": [text]});
        let search = |ns: &str| {
            v(
                "POST",
                &format!("{data}/namespaces/{ns}/documents/search"),
                Some(&request),
            )
        };
        let ids = |body: &Value| -> Vec<String> {
            let matches = body["matches"].as_array().expect("matches");
            let ids = matches.iter().map(|m| m["_id"].as_str().expect("id"));
            ids.map(str::to_string).collect()
        };
        assert_eq!(ids(&search("ns:a").1), ["a1", "a2"]);
        assert!(ids(&search("ns:empty").1).is_empty());
        assert_eq!(
            ids(&search("ns:other").1),
            ["g"],
            "the global matches are the default"
        );
        docs.set_search_missing_namespace(Some(404));
        assert_eq!(
            search("ns:other").0,
            404,
            "never written, no canned matches"
        );
        assert_eq!(search("ns:a").0, 200, "canned matches count as existing");
        docs.preload_documents("ns:written", &["1"]);
        assert_eq!(ids(&search("ns:written").1), ["g"]);
    }

    #[test]
    fn faults_latency_and_recording() {
        let docs = DocsHandle::new();
        let (_backend, _control, data) = started(&docs);
        let url = upsert_url(&data, "ns");
        let send = |id: &str| v("POST", &url, Some(&json!({ "documents": [doc(id, "x")] }))).0;

        docs.fail_next(Endpoint::Upsert, 2, 429, "Pace your requests per second");
        assert_eq!([send("a"), send("a"), send("a")], [429, 429, 202]);
        docs.fail_nth(Endpoint::Upsert, 2, 503, "unavailable");
        assert_eq!([send("b"), send("b"), send("b")], [202, 503, 202]);

        docs.fail_ids(
            Endpoint::Upsert,
            &["poison"],
            400,
            "document poison is invalid",
        );
        let mixed = json!({ "documents": [doc("fine", "x"), doc("poison", "x")] });
        assert_eq!(v("POST", &url, Some(&mixed)).0, 400);
        assert!(
            !docs.remote_ids("ns").contains("fine"),
            "a failed request applies nothing"
        );
        assert_eq!(send("fine"), 202);

        docs.fail_always(Endpoint::Upsert, 500, "boom");
        assert_eq!([send("c"), send("c")], [500, 500]);
        docs.heal(Endpoint::Upsert);
        docs.fail_ids(Endpoint::Upsert, &["poison"], 400, "still");
        assert_eq!(send("c"), 202);
        docs.heal_all();

        docs.set_latency(Endpoint::Upsert, Duration::from_millis(150));
        let started = std::time::Instant::now();
        assert_eq!(send("d"), 202);
        assert!(started.elapsed() >= Duration::from_millis(140));

        let calls = docs.calls_to(Endpoint::Upsert);
        assert_eq!(
            calls.len(),
            12,
            "every call is recorded, faulted ones included"
        );
        assert!(
            calls
                .iter()
                .all(|c| c.api_version.as_deref() == Some(EXPECTED_API_VERSION))
        );
        assert_eq!(calls[0].namespace.as_deref(), Some("ns"));
        assert_eq!(calls[0].status, 429);
        assert_eq!(docs.upsert_bodies().len(), 12);
        docs.clear_calls();
        assert!(docs.calls().is_empty());
        assert_eq!(
            call("GET", &format!("{data}/namespaces/ns"), None, None).0,
            200
        );
        assert_eq!(
            docs.calls()[0].api_version,
            None,
            "a missing header is recorded as missing"
        );
    }

    #[test]
    fn search_returns_canned_matches_and_enforces_one_scoring_type() {
        let docs = DocsHandle::new();
        let (_backend, _control, data) = started(&docs);
        let url = format!("{data}/namespaces/ns/documents/search");
        docs.set_search_matches(&[("7", 3.5), ("3", 1.25), ("9", 0.5)]);
        let text = json!({"type": "text", "fields": ["body"], "query": "q"});
        let dense = json!({"type": "dense_vector", "fields": ["embedding"], "values": [0.5, 0.25]});
        let search = |body: Value| v("POST", &url, Some(&body));

        let request = json!({"top_k": 2, "include_fields": [], "score_by": [text]});
        let (status, body) = search(request);
        assert_eq!(status, 200, "{body}");
        let ids: Vec<&str> = body["matches"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["_id"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            ["7", "3"],
            "truncated to top_k, in the configured order"
        );
        assert_eq!(body["matches"][0]["_score"], json!(3.5));

        let mixed = json!({"top_k": 2, "include_fields": [], "score_by": [text, dense]});
        assert_eq!(search(mixed).0, 400, "two scoring types in one request");
        let no_fields = json!({"top_k": 2, "score_by": [text]});
        assert_eq!(search(no_fields).0, 400, "include_fields is required");
        let too_many = json!({"top_k": 10_001, "include_fields": [], "score_by": [text]});
        assert_eq!(search(too_many).0, 400, "top_k above 10,000");
        let two_text = json!({"top_k": 1, "include_fields": [], "score_by": [text, text]});
        assert_eq!(
            search(two_text).0,
            200,
            "several text clauses are one scoring type"
        );
    }

    #[test]
    fn search_clauses_must_have_the_2026_07_shape() {
        let docs = DocsHandle::new();
        let (_backend, _control, data) = started(&docs);
        let url = format!("{data}/namespaces/ns/documents/search");
        let status_of = |clause: Value| {
            let body = json!({"top_k": 1, "include_fields": [], "score_by": [clause]});
            v("POST", &url, Some(&body)).0
        };
        let ok = [
            json!({"type": "text", "fields": ["a", "b"], "query": "q"}),
            json!({"type": "query_string", "query": "a:(x AND y)"}),
            json!({"type": "dense_vector", "fields": ["e"], "values": [1.0]}),
        ];
        for clause in ok {
            assert_eq!(status_of(clause.clone()), 200, "{clause}");
        }
        let refused = [
            json!({"type": "text", "field": "a", "query": "q"}),
            json!({"type": "text", "query": "q"}),
            json!({"type": "text", "fields": [], "query": "q"}),
            json!({"type": "text", "fields": ["a"], "query": "  "}),
            json!({"type": "query_string", "fields": ["a"], "query": "q"}),
            json!({"type": "query_string", "field": "a", "query": "q"}),
            json!({"type": "dense_vector", "fields": ["a", "b"], "values": [1.0]}),
            json!({"type": "dense_vector", "field": "e", "values": [1.0]}),
            json!({"type": "dense_vector", "fields": ["e"], "values": []}),
        ];
        for clause in refused {
            assert_eq!(status_of(clause.clone()), 400, "{clause}");
        }
    }

    #[test]
    fn version_filtered_mount_ignores_other_api_versions() {
        let docs = DocsHandle::new();
        let handle = docs.clone();
        let backend = MockBackend::start(move |control_plane, data_plane| async move {
            handle
                .mount_for_version(&control_plane, &data_plane, EXPECTED_API_VERSION)
                .await;
            (control_plane, data_plane)
        });
        let describe = format!("{}/indexes/idx", backend.control_plane_url);
        assert_eq!(call("GET", &describe, Some("2026-04"), None).0, 404);
        assert!(
            docs.calls().is_empty(),
            "other versions never reach the docs mock"
        );
        assert_eq!(
            call("GET", &describe, Some(EXPECTED_API_VERSION), None).0,
            404
        );
        assert_eq!(docs.count(Endpoint::IndexDescribe), 1);
    }

    /// The real client (what the extension calls) against the mock: success shapes, the error
    /// mapping the worker classifies on, and the version header.
    #[test]
    fn the_real_client_works_against_the_mock() {
        use pg_pinecone_client::{
            Client, Config, CreateDocIndexRequest, Error, SearchDocsRequest, UpsertDocsRequest,
        };
        let docs = DocsHandle::new();
        let backend = start_with_docs(&docs);
        let config = Config {
            control_plane_url: backend.control_plane_url.clone(),
            api_key: "test-key".to_string(),
            timeout: Duration::from_secs(5),
            user_agent: None,
        };
        let client = Client::with_sleep(config, |_| {});
        let create = CreateDocIndexRequest {
            name: "idx".to_string(),
            schema: schema(),
            cloud: "aws".to_string(),
            region: "us-east-1".to_string(),
        };
        client.create_doc_index(&create).expect("create");
        let (deadline, poll) = (Duration::from_secs(5), Duration::from_millis(50));
        client
            .wait_doc_index_ready("idx", deadline, poll)
            .expect("ready");

        let upsert = |ids: &[&str]| UpsertDocsRequest {
            index: "idx".to_string(),
            namespace: Some("public.articles".to_string()),
            documents: ids.iter().map(|id| doc(id, "text")).collect(),
        };
        assert_eq!(client.upsert_docs(&upsert(&["1", "2"])).expect("upsert"), 2);
        assert_eq!(docs.remote_ids("public.articles").len(), 2);

        docs.set_search_matches(&[("2", 1.5), ("1", 0.5)]);
        let search = SearchDocsRequest {
            index: "idx".to_string(),
            namespace: Some("public.articles".to_string()),
            include_fields: vec![],
            filter: None,
            score_by: json!([{"type": "text", "fields": ["body"], "query": "q"}]),
            top_k: 5,
        };
        let found = client.search_docs(&search).expect("search");
        let ids: Vec<&str> = found.iter().map(|m| m.id.as_str()).collect();
        assert_eq!((ids, found[0].score), (vec!["2", "1"], 1.5));

        // The client retries a 429/5xx up to three times: one failure is absorbed, three surface.
        docs.fail_next(Endpoint::Upsert, 1, 503, "unavailable");
        assert_eq!(client.upsert_docs(&upsert(&["5"])).expect("retried"), 1);
        docs.fail_next(Endpoint::Upsert, 3, 429, "Pace your requests per second");
        let error = client.upsert_docs(&upsert(&["3"])).expect_err("429");
        assert!(matches!(error, Error::Api { status: 429, .. }), "{error:?}");
        docs.fail_ids(Endpoint::Upsert, &["4"], 400, "document 4 is invalid");
        let error = client.upsert_docs(&upsert(&["3", "4"])).expect_err("400");
        assert!(
            matches!(&error, Error::BadRequest { message } if message == "document 4 is invalid"),
            "{error:?}"
        );
        docs.fail_always(Endpoint::Search, 401, "invalid api key");
        assert!(matches!(client.search_docs(&search), Err(Error::Auth)));

        let calls = docs.calls();
        assert!(calls.len() >= 8, "calls = {calls:?}");
        assert!(
            calls
                .iter()
                .all(|c| c.api_version.as_deref() == Some(EXPECTED_API_VERSION))
        );
    }

    /// The namespace calls of the real client (vector API version header) and a 404 search as empty.
    #[test]
    fn the_real_client_lists_describes_and_deletes_namespaces_of_a_document_index() {
        use pg_pinecone_client::{
            Client, Config, CreateDocIndexRequest, Error, SearchDocsRequest, UpsertDocsRequest,
        };
        let docs = DocsHandle::new();
        let backend = start_with_docs(&docs);
        let config = Config {
            control_plane_url: backend.control_plane_url.clone(),
            api_key: "test-key".to_string(),
            timeout: Duration::from_secs(5),
            user_agent: None,
        };
        let client = Client::with_sleep(config, |_| {});
        let create = CreateDocIndexRequest {
            name: "idx".to_string(),
            schema: schema(),
            cloud: "aws".to_string(),
            region: "us-east-1".to_string(),
        };
        client.create_doc_index(&create).expect("create");
        for tenant in 1..=3 {
            let upsert = UpsertDocsRequest {
                index: "idx".to_string(),
                namespace: Some(format!("public.articles:{tenant}")),
                documents: vec![doc("1", "a"), doc("2", "b")],
            };
            client.upsert_docs(&upsert).expect("upsert");
        }

        let listed = client
            .list_all_namespaces("idx", Some("public.articles:"))
            .expect("list");
        let names: Vec<&str> = listed.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "public.articles:1",
                "public.articles:2",
                "public.articles:3"
            ]
        );
        assert_eq!(listed[0].record_count, Some(2));
        let described = client.describe_namespace("idx", "public.articles:2");
        assert_eq!(described.expect("describe").record_count, Some(2));
        let missing = client.describe_namespace("idx", "public.articles:9");
        assert!(matches!(missing, Err(Error::NamespaceNotFound { .. })));
        client
            .delete_namespace("idx", "public.articles:2")
            .expect("delete");
        client
            .delete_namespace("idx", "public.articles:2")
            .expect("gone is success");
        assert!(docs.remote_ids("public.articles:2").is_empty());
        assert_eq!(docs.count(Endpoint::NamespaceDelete), 2);

        docs.set_search_missing_namespace(Some(404));
        let search = SearchDocsRequest {
            index: "idx".to_string(),
            namespace: Some("public.articles:9".to_string()),
            include_fields: vec![],
            filter: None,
            score_by: json!([{"type": "text", "fields": ["body"], "query": "q"}]),
            top_k: 5,
        };
        assert!(
            client
                .search_docs(&search)
                .expect("a 404 is empty")
                .is_empty()
        );
        let listing = docs.calls_to(Endpoint::NamespaceList);
        let sent = listing[0].api_version.as_deref();
        assert_eq!(sent, Some(pg_pinecone_client::API_VERSION));
        let searched = docs.calls_to(Endpoint::Search);
        assert_eq!(
            searched[0].api_version.as_deref(),
            Some(EXPECTED_API_VERSION)
        );
    }

    #[test]
    fn percent_decoding_handles_escapes_and_stray_percent_signs() {
        assert_eq!(percent_decode("public.articles"), "public.articles");
        assert_eq!(percent_decode("a%2Fb%20c"), "a/b c");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("%zz"), "%zz");
    }
}
