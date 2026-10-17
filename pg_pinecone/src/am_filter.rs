//! The filter column of a remote-mode pg_pinecone index (`USING pinecone (vec, col)`): its value is
//! synced to Pinecone as metadata `{"col": v}`, and a scan's equality key on it is pushed down as
//! `{"col": {"$eq": v}}`. The decoded key is also what a partition column turns into a namespace.
use pgrx::{pg_sys, prelude::*};
use serde_json::{Value, json};

use crate::{
    local::{metapage::FilterKind, scan::kind_for_subtype},
    partition::{PartitionValue, canon},
};

/// Integers below this magnitude are exact as f64, which is how Pinecone stores numbers.
const F64_EXACT_LIMIT: u64 = 1 << 53;

/// One equality key as the filter column holds it: an integer (any width, sign-extended) or text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EqKey {
    Int(i64),
    Text(String),
}

impl EqKey {
    /// The value Pinecone filters on: a JSON number or string, the same one `aminsert` and the
    /// backfill write into the metadata.
    pub(crate) fn to_json(&self) -> Value {
        match self {
            EqKey::Int(value) => Value::from(*value),
            EqKey::Text(text) => Value::String(text.clone()),
        }
    }

    /// The key a datum of the filter column itself (its own type, not NULL) holds: what a writer
    /// syncs, and routes on when the column partitions.
    ///
    /// # Safety
    /// `d` must be a valid, non-NULL datum of the Postgres type `kind` was derived from.
    pub(crate) unsafe fn of_datum(kind: FilterKind, d: pg_sys::Datum) -> EqKey {
        unsafe {
            match kind {
                FilterKind::Text => EqKey::Text(FilterKind::datum_to_text(d)),
                _ => EqKey::Int(kind.datum_to_value(d)),
            }
        }
    }

    /// The canonical text of the value, equal to `col::text` (`partition::canon`): what a
    /// namespace is derived from.
    pub(crate) fn canon(&self) -> String {
        match self {
            EqKey::Int(value) => canon(PartitionValue::Int(*value)),
            EqKey::Text(text) => canon(PartitionValue::Text(text.clone())),
        }
    }

    /// Pinecone stores numbers as f64, so it cannot tell integers of 2^53 and beyond from their
    /// neighbours: a match on such a key has to be rechecked against the heap.
    pub(crate) fn inexact_in_pinecone(&self) -> bool {
        matches!(self, EqKey::Int(value) if value.unsigned_abs() >= F64_EXACT_LIMIT)
    }
}

/// What a scan's keys say.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum KeyDecode {
    /// No key: an unfiltered scan.
    Unkeyed,
    /// `col = v` (every key equal).
    Eq(EqKey),
    /// `col = NULL`, or two different equalities: no row can match.
    Nothing,
}

/// `{column: value}`: the metadata one vector carries for its filter key.
pub(crate) fn metadata(column: &str, value: Value) -> Value {
    json!({ column: value })
}

/// `{column: {"$eq": value}}`: the filter a scan pushes down.
pub(crate) fn eq_filter(column: &str, key: &EqKey) -> Value {
    json!({ column: { "$eq": key.to_json() } })
}

/// Decodes the equality scankeys of `scan` (attno 2, strategy 1) against the index's filter
/// `kind`: a cross-type literal sign-extends (`int8col = 5::int4`), equal keys fold into one, a
/// NULL or two different keys can match nothing.
///
/// # Safety
/// `scan` must be a live scan descriptor whose `keyData` holds `numberOfKeys` keys.
pub(crate) unsafe fn decode_keys(scan: pg_sys::IndexScanDesc, kind: FilterKind) -> KeyDecode {
    unsafe {
        let nkeys = (*scan).numberOfKeys as usize;
        if nkeys == 0 {
            return KeyDecode::Unkeyed;
        }
        if kind == FilterKind::None {
            error!(
                "pg_pinecone: scan keys supplied for a pg_pinecone index with no filter column — should \
                 be unreachable"
            );
        }
        let keys = std::slice::from_raw_parts((*scan).keyData, nkeys);
        let mut decoded: Option<EqKey> = None;
        for key in keys {
            if key.sk_attno != 2 || key.sk_strategy != 1 {
                error!(
                    "pg_pinecone: unexpected scan key (attno {}, strategy {}) — the pg_pinecone access \
                     method only supports equality on its filter column",
                    key.sk_attno, key.sk_strategy
                );
            }
            if key.sk_flags & pg_sys::SK_ISNULL as i32 != 0 {
                // `col = NULL` is never true: normally folded away by the planner, but a
                // prepared statement's parameter reaches the AM like this.
                return KeyDecode::Nothing;
            }
            let key_kind = kind_for_subtype(key.sk_subtype, kind);
            if (key_kind == FilterKind::Text) != (kind == FilterKind::Text) {
                error!(
                    "pg_pinecone: scan key of another type than the filter column — should be \
                     unreachable"
                );
            }
            let value = match key_kind {
                FilterKind::Text => EqKey::Text(FilterKind::datum_to_text(key.sk_argument)),
                _ => EqKey::Int(key_kind.datum_to_value(key.sk_argument)),
            };
            decoded = match decoded {
                None => Some(value),
                Some(previous) if previous == value => Some(previous),
                Some(_) => return KeyDecode::Nothing,
            };
        }
        KeyDecode::Eq(decoded.expect("nkeys > 0 always sets a key"))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn keys_are_json_numbers_and_strings() {
        assert_eq!(EqKey::Int(42).to_json(), json!(42));
        assert_eq!(EqKey::Int(-5).to_json(), json!(-5));
        assert_eq!(EqKey::Int(i64::MIN).to_json(), json!(i64::MIN));
        assert_eq!(EqKey::Int(i64::MAX).to_json(), json!(i64::MAX));
        assert_eq!(EqKey::Text("é\"日".into()).to_json(), json!("é\"日"));
        assert_eq!(EqKey::Text(String::new()).to_json(), json!(""));
    }

    #[test]
    fn canon_is_the_text_postgres_gives() {
        assert_eq!(EqKey::Int(-7).canon(), "-7");
        assert_eq!(EqKey::Int(i64::MIN).canon(), i64::MIN.to_string());
        assert_eq!(EqKey::Text("Acme Corp ".into()).canon(), "Acme Corp ");
    }

    #[test]
    fn only_integers_from_two_to_the_53_are_inexact_in_pinecone() {
        let limit = 1_i64 << 53;
        for exact in [0, 1, -1, limit - 1, -(limit - 1)] {
            assert!(!EqKey::Int(exact).inexact_in_pinecone(), "{exact}");
        }
        for inexact in [limit, -limit, limit + 1, i64::MAX, i64::MIN] {
            assert!(EqKey::Int(inexact).inexact_in_pinecone(), "{inexact}");
        }
        assert!(!EqKey::Text("9007199254740993".into()).inexact_in_pinecone());
    }

    #[test]
    fn metadata_and_filter_shapes() {
        assert_eq!(metadata("category", json!("a")), json!({"category": "a"}));
        assert_eq!(metadata("tenant_id", json!(7)), json!({"tenant_id": 7}));
        assert_eq!(
            eq_filter("tenant_id", &EqKey::Int(42)),
            json!({"tenant_id": {"$eq": 42}})
        );
        assert_eq!(
            eq_filter("category", &EqKey::Text("a".into())),
            json!({"category": {"$eq": "a"}})
        );
    }
}
