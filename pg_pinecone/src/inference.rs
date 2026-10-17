//! Implementation of the `pinecone.embed` / `pinecone.embed_batch` / `pinecone.rerank`
//! SQL functions. `#[pg_schema]` only attaches to an inline `mod {}` (see the
//! comment on `pg_pinecone` in `lib.rs`), so its `#[pg_extern]` wrappers are kept
//! thin and delegate here — mirroring how the `tests` schema delegates to
//! `mock_backend`/`smoke`.

use pg_pinecone_client::{EmbedRequest, RerankRequest};
use pgrx::prelude::*;

use crate::backend::{client_for_effective_key, raise, require_text, require_text_array};

/// Implements `pinecone.embed`: embeds a single input, returning its vector.
pub(crate) fn embed(
    input: Option<&str>,
    model: Option<&str>,
    input_type: Option<&str>,
) -> Vec<f32> {
    let request = EmbedRequest {
        model: require_text("pinecone.embed", "model", model),
        inputs: vec![require_text("pinecone.embed", "input", input)],
        input_type: require_text("pinecone.embed", "input_type", input_type),
    };

    // Blocking call; cancel takes effect within a few multiples of pinecone.timeout_ms.
    let mut embeddings = client_for_effective_key()
        .embed(&request)
        .unwrap_or_else(|e| raise(e));

    embeddings
        .pop()
        .unwrap_or_else(|| error!("pinecone.embed: Pinecone returned no embedding for the input"))
}

/// Implements `pinecone.embed_batch`: embeds each of `inputs`, returning
/// `(ordinal, embedding)` pairs with a 1-based `ordinal` matching input position.
pub(crate) fn embed_batch(
    inputs: Option<Vec<Option<String>>>,
    model: Option<&str>,
    input_type: Option<&str>,
) -> Vec<(i32, Vec<f32>)> {
    let request = EmbedRequest {
        model: require_text("pinecone.embed_batch", "model", model),
        inputs: require_text_array("pinecone.embed_batch", "inputs", inputs),
        input_type: require_text("pinecone.embed_batch", "input_type", input_type),
    };

    // Blocking call; the client embeds in chunks of 96 with no interrupt check
    // between them, so a cancel only takes effect once the whole batch finishes —
    // latency scales with ceil(inputs / 96) sequential requests.
    let embeddings = client_for_effective_key()
        .embed(&request)
        .unwrap_or_else(|e| raise(e));

    embeddings
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| (i as i32 + 1, embedding))
        .collect()
}

/// Implements `pinecone.rerank`: reranks `documents` against `query`, returning
/// `(position, document, score)` rows. `position` is 1-based into `documents`
/// (the client's result index is 0-based, in response order).
pub(crate) fn rerank(
    query: Option<&str>,
    documents: Option<Vec<Option<String>>>,
    model: Option<&str>,
    top_n: Option<i32>,
) -> Vec<(i32, String, f32)> {
    let documents = require_text_array("pinecone.rerank", "documents", documents);
    let request = RerankRequest {
        model: require_text("pinecone.rerank", "model", model),
        query: require_text("pinecone.rerank", "query", query),
        documents: documents.clone(),
        top_n: require_non_negative_top_n(top_n),
    };

    // Blocking call; same cancel behavior as pinecone.embed.
    let results = client_for_effective_key()
        .rerank(&request)
        .unwrap_or_else(|e| raise(e));

    results
        .into_iter()
        .map(|r| {
            let document = documents.get(r.index).cloned().unwrap_or_else(|| {
                error!(
                    "pinecone.rerank: Pinecone returned out-of-range document index {}",
                    r.index
                )
            });
            (r.index as i32 + 1, document, r.score)
        })
        .collect()
}

/// Converts the optional `top_n` argument to the client's `Option<u32>`,
/// raising a clear error for a negative value; `None` (no limit) passes through.
fn require_non_negative_top_n(top_n: Option<i32>) -> Option<u32> {
    top_n.map(|n| {
        u32::try_from(n)
            .unwrap_or_else(|_| error!("pinecone.rerank: top_n must be non-negative, got {n}"))
    })
}
