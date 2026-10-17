//! Implementation of `pinecone.create_doc_index` / `pinecone.drop_doc_index` /
//! `pinecone.upsert_docs` / `pinecone.search_doc_index`. `#[pg_schema]` only attaches
//! to an inline `mod {}` (see the comment on `pg_pinecone` in `lib.rs`), so its
//! `#[pg_extern]` wrappers are kept thin and delegate here — mirroring
//! `inference.rs`/`promote.rs`.

use std::time::Duration;

use pg_pinecone_client::{
    CreateDocIndexRequest, Error as ClientError, SearchDocsRequest, UpsertDocsRequest,
};
use pgrx::{JsonB, prelude::*};
use serde_json::Value;

use crate::backend::{self, raise, require_non_negative_u32, require_text};

/// How long `pinecone.create_doc_index` waits for a newly created index to
/// report `ready` before giving up — same budget as `pinecone.promote`'s
/// `WAIT_READY_DEADLINE`.
pub(crate) const WAIT_READY_DEADLINE: Duration = Duration::from_secs(120);
/// How often `pinecone.create_doc_index` polls readiness while waiting — same
/// cadence as `pinecone.promote`'s `WAIT_READY_POLL`.
pub(crate) const WAIT_READY_POLL: Duration = Duration::from_secs(2);

/// Implements `pinecone.create_doc_index`: creates a document-schema
/// (full-text search) index and waits for it to report `ready`. `schema`
/// passes through to Pinecone verbatim once its minimum shape (a `"fields"`
/// object) is validated — Pinecone itself validates the field definitions.
/// Same credential resolution as `pinecone.promote` (see
/// `backend::client_for_effective_key`'s rustdoc): a user-invoked remote op,
/// not a sync-trigger internal call.
pub(crate) fn create_doc_index(
    name: Option<&str>,
    schema: Option<JsonB>,
    cloud: Option<&str>,
    region: Option<&str>,
) {
    let request = CreateDocIndexRequest {
        name: require_text("pinecone.create_doc_index", "name", name),
        schema: require_doc_schema(schema),
        cloud: require_text("pinecone.create_doc_index", "cloud", cloud),
        region: require_text("pinecone.create_doc_index", "region", region),
    };

    let client = backend::client_for_effective_key();
    client
        .create_doc_index(&request)
        .unwrap_or_else(|e| raise(e));

    client
        .wait_doc_index_ready(&request.name, WAIT_READY_DEADLINE, WAIT_READY_POLL)
        .unwrap_or_else(|e| match e {
            // A dedicated message: `backend::raise`'s generic `Timeout` text
            // points at `pinecone.timeout_ms`, which is unrelated here — this
            // wait is bounded by `WAIT_READY_DEADLINE`, not the per-request
            // HTTP timeout (mirrors `pinecone.promote`'s own readiness wait).
            ClientError::Timeout { .. } => {
                error!("{}", wait_ready_timeout_message(&request.name))
            }
            other => backend::raise(other),
        });
}

/// The message for a `wait_doc_index_ready` timeout: names the deadline and
/// the index, and deliberately never mentions `pinecone.timeout_ms` (that
/// GUC bounds a single HTTP request, not this multi-minute readiness wait).
/// Mirrors `promote::wait_ready_timeout_message`.
fn wait_ready_timeout_message(index_name: &str) -> String {
    format!(
        "pinecone.create_doc_index: index '{index_name}' did not become ready within the {}s \
         readiness deadline — check the index status in the Pinecone console and retry",
        WAIT_READY_DEADLINE.as_secs()
    )
}

/// Validates that `schema` is a JSON object with a `"fields"` object — the
/// minimum shape `pinecone.create_doc_index` requires before sending it on.
fn require_doc_schema(schema: Option<JsonB>) -> Value {
    let Some(schema) = schema else {
        error!(
            "pinecone.create_doc_index: schema must not be NULL — pass a jsonb object with a \
             \"fields\" object, e.g. '{{\"fields\": {{\"title\": {{\"type\": \"string\"}}}}}}'"
        )
    };
    let Value::Object(map) = &schema.0 else {
        error!(
            "pinecone.create_doc_index: schema must be a JSON object with a \"fields\" object, \
             e.g. '{{\"fields\": {{\"title\": {{\"type\": \"string\"}}}}}}'"
        )
    };
    if !matches!(map.get("fields"), Some(Value::Object(_))) {
        error!(
            "pinecone.create_doc_index: schema must have a \"fields\" object, e.g. \
             '{{\"fields\": {{\"title\": {{\"type\": \"string\"}}}}}}'"
        )
    }
    schema.0
}

/// Implements `pinecone.drop_doc_index`. Same credential resolution as
/// [`create_doc_index`].
pub(crate) fn drop_doc_index(name: Option<&str>) {
    let name = require_text("pinecone.drop_doc_index", "name", name);
    backend::client_for_effective_key()
        .delete_doc_index(&name)
        .unwrap_or_else(|e| raise(e));
}

/// Implements `pinecone.upsert_docs`: upserts `docs` into document-schema
/// index `index`, returning the total number of documents upserted. Same
/// credential resolution as [`create_doc_index`]. `docs` is chunked and sent
/// to Pinecone sequentially (see `Client::upsert_docs`'s rustdoc) — the
/// chunks are not applied atomically, so if a later chunk fails, earlier
/// chunks have already landed in the index.
pub(crate) fn upsert_docs(
    index: Option<&str>,
    docs: Option<JsonB>,
    namespace: Option<String>,
) -> i64 {
    let request = UpsertDocsRequest {
        index: require_text("pinecone.upsert_docs", "index", index),
        namespace,
        documents: require_doc_objects(docs),
    };

    // Blocking call; cancel takes effect within a few multiples of pinecone.timeout_ms
    // (host resolve + upsert, plus one evict-and-retry of both on transport failure,
    // times the number of chunks — at least ceil(docs / 1000), more if the byte
    // budget forces a smaller chunk).
    let count = backend::client_for_effective_key()
        .upsert_docs(&request)
        .unwrap_or_else(|e| raise(e));

    // Counts are bounded by what a single statement can submit; the cast is safe.
    count as i64
}

/// Validates that `docs` is a non-NULL JSON array of objects, each carrying a
/// string `"_id"`, producing a clear error (naming the offending 1-based
/// position) rather than a malformed batch forwarded to Pinecone.
fn require_doc_objects(docs: Option<JsonB>) -> Vec<Value> {
    let Some(docs) = docs else {
        error!(
            "pinecone.upsert_docs: docs must not be NULL — pass a jsonb array of document \
             objects, e.g. '[{{\"_id\": \"a\", \"title\": \"…\"}}]'"
        )
    };
    let Value::Array(elements) = docs.0 else {
        error!(
            "pinecone.upsert_docs: docs must be a JSON array of document objects, e.g. \
             '[{{\"_id\": \"a\", \"title\": \"…\"}}]'"
        )
    };
    for (i, element) in elements.iter().enumerate() {
        let Value::Object(fields) = element else {
            error!(
                "pinecone.upsert_docs: docs[{}] must be a JSON object",
                i + 1
            )
        };
        if !matches!(fields.get("_id"), Some(Value::String(_))) {
            error!(
                "pinecone.upsert_docs: docs[{}] is missing a string \"_id\" field",
                i + 1
            )
        }
    }
    elements
}

/// Implements `pinecone.search_doc_index`: runs a full-text/hybrid search against
/// document-schema index `index`, returning `(id, score, doc)` rows. `k = 0`
/// short-circuits to an empty result with no HTTP call. Same credential
/// resolution as [`create_doc_index`].
pub(crate) fn search_doc_index(
    index: Option<&str>,
    query: Option<JsonB>,
    k: i32,
    filter: Option<JsonB>,
    fields: Option<Vec<Option<String>>>,
    namespace: Option<String>,
) -> Vec<(String, f32, JsonB)> {
    let index = require_text("pinecone.search_doc_index", "index", index);
    let score_by = require_score_by(query);
    let include_fields = require_fields_or_wildcard(fields);
    let k = require_non_negative_u32("pinecone.search_doc_index", "k", k);
    if k == 0 {
        return vec![];
    }

    let request = SearchDocsRequest {
        index,
        namespace,
        include_fields,
        filter: filter.map(|j| j.0),
        score_by,
        top_k: k,
    };

    // Blocking call; same cancel behavior as pinecone.upsert_docs.
    let matches = backend::client_for_effective_key()
        .search_docs(&request)
        .unwrap_or_else(|e| raise(e));

    matches
        .into_iter()
        .map(|m| (m.id, m.score, JsonB(m.doc)))
        .collect()
}

/// Normalizes `query` into a `score_by` clause array: a single clause object
/// becomes a one-element array; a non-empty array passes through unchanged.
/// An empty array (`'[]'::jsonb`) is rejected — Pinecone requires at least
/// one clause. Every clause must be a JSON object with a `"type"` string
/// field.
fn require_score_by(query: Option<JsonB>) -> Value {
    let Some(query) = query else {
        error!(
            "pinecone.search_doc_index: query must not be NULL — pass a score_by clause object or \
             an array of them, e.g. '{{\"type\": \"text\", \"fields\": [\"body\"], \"query\": \
             \"...\"}}'"
        )
    };
    let clauses: Vec<Value> = match query.0 {
        Value::Array(elements) => elements,
        object @ Value::Object(_) => vec![object],
        _ => error!(
            "pinecone.search_doc_index: query must be a score_by clause object or an array of \
             clause objects, e.g. '{{\"type\": \"text\", \"fields\": [\"body\"], \"query\": \
             \"...\"}}'"
        ),
    };
    if clauses.is_empty() {
        error!(
            "pinecone.search_doc_index: query must not be an empty array — pass a score_by clause \
             object or a non-empty array of them, e.g. '{{\"type\": \"text\", \"fields\": \
             [\"body\"], \"query\": \"...\"}}'"
        )
    }
    for (i, clause) in clauses.iter().enumerate() {
        let has_type_string = matches!(clause, Value::Object(map) if matches!(map.get("type"), Some(Value::String(_))));
        if !has_type_string {
            error!(
                "pinecone.search_doc_index: query[{}] must be a JSON object with a \"type\" string \
                 field, e.g. '{{\"type\": \"text\", \"fields\": [\"body\"], \"query\": \"...\"}}'",
                i + 1
            )
        }
    }
    Value::Array(clauses)
}

/// `fields` NULL means "all fields" (`["*"]`); a NULL element within a
/// non-NULL array is an error (mirrors `search::require_field_names`).
fn require_fields_or_wildcard(fields: Option<Vec<Option<String>>>) -> Vec<String> {
    match fields {
        None => vec!["*".to_string()],
        Some(elements) => elements
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                v.unwrap_or_else(|| {
                    error!(
                        "pinecone.search_doc_index: fields[{}] is NULL — all field names must be \
                         non-NULL",
                        i + 1
                    )
                })
            })
            .collect(),
    }
}

// Pure-Rust (no pgrx/pg dependency), unlike this crate's `#[pg_test]`s — runs
// under plain `cargo test`/`cargo pgrx test` alike. Mirrors `promote`'s own
// test of its analogous timeout-message helper.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wait_ready_timeout_message_names_deadline_and_index_not_timeout_ms_guc() {
        let message = wait_ready_timeout_message("pv-public-articles");
        assert!(
            message.contains("pv-public-articles"),
            "message = {message}"
        );
        assert!(message.contains("120s"), "message = {message}");
        assert!(
            !message.contains("pinecone.timeout_ms"),
            "message must not point at the per-request HTTP timeout GUC: {message}"
        );
    }
}
