//! `pinecone.secret`: single-row API key store backing `pinecone.set_api_key`/
//! `pinecone.clear_api_key`, and read by `backend::effective_credentials` for
//! user-invoked remote operations (`promote`/`demote`/`sync`, remote-mode
//! `topk`, `embed`/`embed_batch`/`rerank`, and the doc-index functions) when
//! no session GUC key is set.
//!
//! The stored `control_plane_url` is bound to the stored `api_key` in the same
//! row — never the session's own `pinecone.control_plane_url` GUC — so a
//! writer who controls their session GUCs (but not `pinecone.secret` itself)
//! can never redirect the stored key to a server they control. See
//! `backend::effective_credentials`'s rustdoc for the full rationale.
//!
//! Declared here (rather than inside the `pg_pinecone` `#[pg_schema]` module in
//! `lib.rs`) for the same reason as `pinecone.tables` in `registry.rs`: an
//! `extension_sql!` with an explicit `requires` can't live inside an inline
//! `#[pg_schema] mod {}`.
//!
//! Deliberately **not** passed to `pg_extension_config_dump`: a `pg_dump` must
//! never carry the API key. `REVOKE ALL ... FROM PUBLIC` is the whole access
//! model — see `backend::effective_credentials`'s rustdoc for the read side,
//! and [`set_api_key`]'s for the write side. `pinecone._has_stored_api_key`
//! is the one deliberate exception: it's `SECURITY DEFINER` and left
//! executable by anyone, since it only ever answers a boolean ("is a key
//! currently stored?"), never the key itself — `pinecone.promote` uses it to
//! enforce its stored-key requirement even for a caller authenticated purely
//! via the session GUC (see `promote::promote`'s rustdoc).

use pgrx::{prelude::*, spi::Spi};

extension_sql!(
    r#"
    CREATE TABLE pinecone.secret (
        id bool PRIMARY KEY DEFAULT true CHECK (id),
        api_key text NOT NULL,
        control_plane_url text NOT NULL DEFAULT 'https://api.pinecone.io'
    );
    REVOKE ALL ON pinecone.secret FROM PUBLIC;

    CREATE FUNCTION pinecone._has_stored_api_key() RETURNS boolean
    LANGUAGE sql
    SECURITY DEFINER
    STABLE
    SET search_path = pg_catalog, pg_temp
    AS $$ SELECT EXISTS (SELECT 1 FROM pinecone.secret WHERE id AND api_key <> '') $$;
    "#,
    name = "pg_pinecone_secret",
    requires = [pinecone],
);

/// Upserts the stored API key/control-plane URL together. `#[pg_extern]`'s
/// default `SECURITY INVOKER` means this runs with the caller's own
/// privileges, so a role with no grant on `pinecone.secret` gets Postgres's
/// own "permission denied for table secret" — the table ACL *is* the access
/// model, no catch/rewrap needed.
pub(crate) fn set_api_key(key: &str, control_plane_url: &str) {
    Spi::run_with_args(
        "INSERT INTO pinecone.secret (id, api_key, control_plane_url) VALUES (true, $1, $2) \
         ON CONFLICT (id) DO UPDATE SET api_key = EXCLUDED.api_key, \
                                         control_plane_url = EXCLUDED.control_plane_url",
        &[key.into(), control_plane_url.into()],
    )
    .expect("pinecone.set_api_key: insert failed");
}

/// Deletes the stored API key, if any. A no-op (not an error) if none was set.
pub(crate) fn clear_api_key() {
    Spi::run("DELETE FROM pinecone.secret").expect("pinecone.clear_api_key: delete failed");
}

/// Whether `pinecone.secret` currently holds a non-empty API key, checked via
/// the `SECURITY DEFINER` `pinecone._has_stored_api_key()` so it answers
/// truthfully even for a caller with no grant on `pinecone.secret` itself
/// (see the module rustdoc).
pub(crate) fn has_stored_api_key() -> bool {
    Spi::get_one::<bool>("SELECT pinecone._has_stored_api_key()")
        .expect("pg_pinecone: _has_stored_api_key query failed")
        .expect("pg_pinecone: _has_stored_api_key unexpectedly NULL")
}
