//! Pinecone-style filter `jsonb` → SQL `WHERE`-clause translator.
//!
//! Pure Rust, no pgrx/pg dependency, so it is unit-tested with plain `#[test]`
//! (see the bottom of this file) rather than `#[pg_test]`. Callers (`pinecone.topk`
//! locally, catalog validation for both modes) add their own error prefix —
//! every [`Err`] returned here is prefix-free.
//!
//! ## SQL-injection safety
//! Every identifier goes through [`quote_ident`] (`"` doubled) and every
//! literal through [`quote_string`] (`'` doubled, NUL rejected) followed by an
//! explicit `::cast` matching the column's [`ColKind`] — there is no code path
//! that copies filter-supplied text into the output without one of these two
//! functions. [`quote_string`] mirrors Postgres's own `quote_literal()`: a
//! value containing a backslash is rendered `E'...'` with both `\` and `'`
//! doubled, everything else as plain `'...'` with `'` doubled. The `E'...'`
//! form is required precisely because it's the one literal syntax whose
//! backslash-escaping is *not* gated by `standard_conforming_strings` — a
//! plain `'...'` literal containing a raw backslash would be misparsed (and a
//! trailing backslash could even swallow the closing quote) if a session ever
//! runs with `standard_conforming_strings = off`.
//!
//! ## Pinned semantics (local is canonical)
//! `$ne`/`$nin` exclude rows where the column is `NULL` (SQL's `<>`/`NOT IN`
//! naturally evaluate to `NULL`, not `TRUE`, against a `NULL` operand) — read
//! as "the field exists and differs". The one special case is `$nin: []`
//! (nothing to exclude): it renders as `<col> IS NOT NULL` rather than a bare
//! `TRUE`, so a `NULL` column still doesn't match "exists, and isn't in the
//! empty set". Range operators (`$gt`/`$gte`/`$lt`/`$lte`) are rejected on
//! `Text`/`TextArray` columns, mirroring Pinecone's own restriction of range
//! filters to numeric fields. These local semantics are canonical; the M9
//! remote-parity suite validates the client's filter translation against
//! them, not the other way around.
//!
//! ## Boolean-grouping strategy
//! `AND` is associative, so sibling object keys, `$and` branches, and
//! multi-operator column groups are all flattened with a bare `" AND "` join —
//! no parentheses needed between them. `$or` is the one place grouping matters
//! (SQL's `AND` binds tighter than `OR`), so an `$or`'s join is always wrapped
//! in its own parens before being handed back to a parent that might `AND` it
//! with siblings. [`translate`] adds exactly one more outer paren pair, so its
//! result is always "a parenthesized SQL boolean expression" regardless of
//! what it contains.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

/// Filter nesting cap (`$and`/`$or` recursion depth): a runaway/adversarial
/// filter can't force unbounded recursion.
const MAX_DEPTH: u32 = 10;

/// Column type used both to validate a filter value against the column it
/// targets and to pick the explicit `::cast` applied to every literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ColKind {
    Text,
    Int,
    Float,
    Numeric,
    Bool,
    TextArray,
    /// A live column whose Postgres type pg_pinecone's filter translator doesn't
    /// recognize (carries `pg_type.typname` for the error message). Never
    /// itself renders a comparison — [`render_entry`] rejects any filter that
    /// references one before it reaches an operator.
    Unsupported(String),
}

/// Message for `$gt`/`$gte`/`$lt`/`$lte` on a `Text`/`TextArray` column:
/// Pinecone itself restricts range filters to numeric fields, and pg_pinecone
/// pins the same restriction locally (see the module doc).
const RANGE_NOT_SUPPORTED_ON_TEXT: &str = "range operators are not supported on text columns — \
     Pinecone restricts range filters to numbers";

/// Collects every key in `filter` that could possibly be a column reference —
/// i.e. every object key that isn't `$and`/`$or` (recursed into) — into `out`.
/// Best-effort and never fails: a malformed filter just yields an incomplete
/// (or empty) set here, which [`translate`] will separately reject with its
/// own clear error. Used by remote-mode column validation
/// (`pinecone.topk`'s `topk_remote`) to resolve only the metadata columns a
/// filter actually touches, rather than every column the table happens to
/// have registered — so a *different*, unreferenced, metadata column that
/// was dropped from the table doesn't break a query that never mentions it.
pub(crate) fn referenced_columns(filter: &Value, out: &mut BTreeSet<String>) {
    let Value::Object(map) = filter else {
        return;
    };
    for (key, val) in map {
        match key.as_str() {
            "$and" | "$or" => {
                if let Value::Array(items) = val {
                    for item in items {
                        referenced_columns(item, out);
                    }
                }
            }
            _ if key.starts_with('$') => {} // unknown top-level operator; translate() rejects it
            _ => {
                out.insert(key.clone());
            }
        }
    }
}

/// The value `filter` pins `column` to with an equality, `{"col": v}` or `{"col": {"$eq": v}}`,
/// at the top level or inside `$and`: what binds a query to one partition. `$or`, `$ne`, `$in`
/// and the ranges bind nothing.
pub(crate) fn equality_on<'a>(filter: &'a Value, column: &str) -> Option<&'a Value> {
    equality_at(filter, column, 1)
}

fn equality_at<'a>(filter: &'a Value, column: &str, depth: u32) -> Option<&'a Value> {
    if depth > MAX_DEPTH {
        return None;
    }
    for (key, value) in filter.as_object()? {
        if key == "$and" {
            let branches = value.as_array().map(Vec::as_slice).unwrap_or_default();
            if let Some(found) = branches
                .iter()
                .find_map(|branch| equality_at(branch, column, depth + 1))
            {
                return Some(found);
            }
        } else if key == column {
            match value {
                Value::Object(operators) => {
                    if let Some(operand) = operators.get("$eq") {
                        return Some(operand);
                    }
                }
                Value::String(_) | Value::Number(_) => return Some(value),
                _ => {}
            }
        }
    }
    None
}

/// Translates a Pinecone-style filter `jsonb` value into a parenthesized SQL
/// boolean expression, e.g. `("genre" = 'drama'::text)`. `columns` maps every
/// column the filter is allowed to reference to its [`ColKind`]; the caller is
/// responsible for catalog-validating that map (this function trusts it).
pub(crate) fn translate(
    filter: &Value,
    columns: &BTreeMap<String, ColKind>,
) -> Result<String, String> {
    Ok(format!("({})", render_object(filter, columns, 1)?))
}

/// Renders a JSON object as an `AND`-joined (bare, unwrapped — see the module
/// doc) list of its entries. Used both for the top-level filter and for each
/// element of a `$and`/`$or` array.
fn render_object(
    value: &Value,
    columns: &BTreeMap<String, ColKind>,
    depth: u32,
) -> Result<String, String> {
    if depth > MAX_DEPTH {
        return Err(format!(
            "filter nesting exceeds the maximum depth of {MAX_DEPTH}"
        ));
    }
    let Value::Object(map) = value else {
        return Err(format!("filter must be a JSON object, got {value}"));
    };
    if map.is_empty() {
        return Err("filter object must have at least one key".to_string());
    }

    let mut parts = Vec::with_capacity(map.len());
    for (key, val) in map {
        parts.push(render_entry(key, val, columns, depth)?);
    }
    Ok(parts.join(" AND "))
}

/// Renders one object entry: a registered column (implicit `$eq` or an
/// operator object) or a `$and`/`$or` logical connective. Column lookup runs
/// first, so a table column literally named e.g. `$eq` still resolves as a
/// column rather than being mistaken for the operator.
fn render_entry(
    key: &str,
    val: &Value,
    columns: &BTreeMap<String, ColKind>,
    depth: u32,
) -> Result<String, String> {
    if let Some(kind) = columns.get(key) {
        if let ColKind::Unsupported(type_name) = kind {
            return Err(format!(
                "column '{key}' has type {type_name}, which pg_pinecone filters do not support"
            ));
        }
        return render_column(&quote_ident(key), kind, val);
    }
    match key {
        "$and" => render_logical("$and", val, columns, depth, "AND", false),
        "$or" => render_logical("$or", val, columns, depth, "OR", true),
        _ if key.starts_with('$') => Err(format!("unknown operator '{key}'")),
        _ => Err(format!("unknown column '{key}'")),
    }
}

/// Renders `$and`/`$or`: `val` must be a non-empty array of filter objects,
/// each recursively rendered at `depth + 1`. `$or`'s join is wrapped in its
/// own parens (`wrap`); `$and`'s is left bare (see the module doc).
fn render_logical(
    name: &str,
    val: &Value,
    columns: &BTreeMap<String, ColKind>,
    depth: u32,
    joiner: &str,
    wrap: bool,
) -> Result<String, String> {
    let Value::Array(items) = val else {
        return Err(format!(
            "{name} requires an array of filter objects, got {val}"
        ));
    };
    if items.is_empty() {
        return Err(format!("{name} requires a non-empty array"));
    }

    let mut parts = Vec::with_capacity(items.len());
    for item in items {
        parts.push(render_object(item, columns, depth + 1)?);
    }
    let joined = parts.join(&format!(" {joiner} "));
    Ok(if wrap { format!("({joined})") } else { joined })
}

/// Renders a column's value: a scalar is implicit `$eq`; an object is one or
/// more `$op: value` pairs, `AND`-joined bare (same associativity argument as
/// `render_object`). `ident` is already `"`-quoted. `kind` is never
/// `Unsupported` here — [`render_entry`] rejects those before this is called.
fn render_column(ident: &str, kind: &ColKind, val: &Value) -> Result<String, String> {
    match val {
        Value::Object(ops) => {
            if ops.is_empty() {
                return Err(format!(
                    "operator object for column {ident} must not be empty"
                ));
            }
            let mut parts = Vec::with_capacity(ops.len());
            for (op, opval) in ops {
                parts.push(render_operator(ident, kind, op, opval)?);
            }
            Ok(parts.join(" AND "))
        }
        _ => render_operator(ident, kind, "$eq", val),
    }
}

/// Renders a single `$op: value` pair against `ident`. `$exists` applies to
/// every [`ColKind`]; `TextArray` columns support only `$eq` (rendered as
/// `ANY(ident)` membership) besides `$exists`; `Text`/`TextArray` reject the
/// range operators (see [`RANGE_NOT_SUPPORTED_ON_TEXT`]); every other kind
/// supports the full comparison/`$in`/`$nin` set.
fn render_operator(ident: &str, kind: &ColKind, op: &str, opval: &Value) -> Result<String, String> {
    if op == "$exists" {
        let Value::Bool(exists) = opval else {
            return Err(format!("$exists requires a boolean value, got {opval}"));
        };
        return Ok(format!(
            "{ident} IS {}NULL",
            if *exists { "NOT " } else { "" }
        ));
    }

    if matches!(kind, ColKind::TextArray) {
        return match op {
            "$eq" => {
                let Value::String(s) = opval else {
                    return Err(format!(
                        "$eq on a text[] column requires a string value, got {opval}"
                    ));
                };
                Ok(format!("{}::text = ANY({ident})", quote_string(s)?))
            }
            "$gt" | "$gte" | "$lt" | "$lte" => Err(RANGE_NOT_SUPPORTED_ON_TEXT.to_string()),
            "$ne" | "$in" | "$nin" => Err(format!(
                "text[] columns only support $eq (membership) and $exists, got '{op}'"
            )),
            _ => Err(format!("unknown operator '{op}'")),
        };
    }

    match op {
        "$eq" => Ok(format!("{ident} = {}", render_literal(opval, kind)?)),
        "$ne" => Ok(format!("{ident} <> {}", render_literal(opval, kind)?)),
        "$gt" => {
            if matches!(kind, ColKind::Text) {
                return Err(RANGE_NOT_SUPPORTED_ON_TEXT.to_string());
            }
            Ok(format!("{ident} > {}", render_literal(opval, kind)?))
        }
        "$gte" => {
            if matches!(kind, ColKind::Text) {
                return Err(RANGE_NOT_SUPPORTED_ON_TEXT.to_string());
            }
            Ok(format!("{ident} >= {}", render_literal(opval, kind)?))
        }
        "$lt" => {
            if matches!(kind, ColKind::Text) {
                return Err(RANGE_NOT_SUPPORTED_ON_TEXT.to_string());
            }
            Ok(format!("{ident} < {}", render_literal(opval, kind)?))
        }
        "$lte" => {
            if matches!(kind, ColKind::Text) {
                return Err(RANGE_NOT_SUPPORTED_ON_TEXT.to_string());
            }
            Ok(format!("{ident} <= {}", render_literal(opval, kind)?))
        }
        "$in" => render_in_list(ident, kind, opval, "$in", "IN", "FALSE".to_string()),
        "$nin" => render_in_list(
            ident,
            kind,
            opval,
            "$nin",
            "NOT IN",
            format!("{ident} IS NOT NULL"),
        ),
        _ => Err(format!("unknown operator '{op}'")),
    }
}

/// Renders `$in`/`$nin`: `val` must be an array; an empty one short-circuits
/// to `empty_result` without ever touching `ident` otherwise (`$in: []` →
/// `FALSE`; `$nin: []` → `<ident> IS NOT NULL`, i.e. "exists, and isn't in the
/// empty set" — see the module doc's pinned-semantics section).
fn render_in_list(
    ident: &str,
    kind: &ColKind,
    val: &Value,
    op_name: &str,
    sql_op: &str,
    empty_result: String,
) -> Result<String, String> {
    let Value::Array(items) = val else {
        return Err(format!("{op_name} requires an array value, got {val}"));
    };
    if items.is_empty() {
        return Ok(empty_result);
    }

    let literals = items
        .iter()
        .map(|item| render_literal(item, kind))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(format!("{ident} {sql_op} ({})", literals.join(", ")))
}

/// Renders a scalar JSON value as a `'literal'::cast` (or bare `TRUE`/`FALSE`
/// cast, or number) matching `kind`, erroring if the JSON type doesn't fit.
/// Never called with `kind == TextArray` — its one supported comparison
/// (`$eq`'s `ANY(...)` membership) casts the scalar as `Text` directly. Also
/// never called with `kind == Unsupported` — [`render_entry`] rejects those
/// before rendering starts; the arm below only exists for exhaustiveness.
fn render_literal(value: &Value, kind: &ColKind) -> Result<String, String> {
    match (kind, value) {
        (ColKind::Text, Value::String(s)) => Ok(format!("{}::text", quote_string(s)?)),
        (ColKind::Bool, Value::Bool(b)) => {
            Ok(format!("{}::bool", if *b { "TRUE" } else { "FALSE" }))
        }
        (ColKind::Int, Value::Number(n)) => {
            if n.is_i64() || n.is_u64() {
                Ok(format!("{value}::int8"))
            } else {
                Err(format!(
                    "value {value} is not valid for an Int column (fractional numbers need Float or Numeric)"
                ))
            }
        }
        (ColKind::Float, Value::Number(_)) => Ok(format!("{value}::float8")),
        (ColKind::Numeric, Value::Number(_)) => Ok(format!("{value}::numeric")),
        (ColKind::Unsupported(type_name), _) => Err(format!(
            "internal error: column of unsupported type {type_name} reached literal rendering — \
             this should have been rejected before rendering started"
        )),
        _ => Err(format!("value {value} does not match column type {kind:?}")),
    }
}

/// Quotes a SQL identifier, doubling any embedded `"`.
fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Quotes a SQL string literal, mirroring Postgres's own `quote_literal()`
/// (rejecting NUL first — Postgres `text` can't store it, so better a clear
/// translator error than a runtime one). A value containing a backslash is
/// rendered `E'...'` with both `\` and `'` doubled: `E'...'` escaping is
/// unconditional (unlike a plain `'...'`, whose backslash handling flips with
/// the `standard_conforming_strings` GUC), so this is the only form that's
/// safe regardless of that session setting. A value with no backslash is
/// rendered as plain `'...'` with `'` doubled, matching `quote_literal()`.
fn quote_string(s: &str) -> Result<String, String> {
    if s.contains('\0') {
        return Err("string values must not contain a NUL byte".to_string());
    }
    if s.contains('\\') {
        let escaped = s.replace('\\', "\\\\").replace('\'', "''");
        Ok(format!("E'{escaped}'"))
    } else {
        Ok(format!("'{}'", s.replace('\'', "''")))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn cols(pairs: &[(&str, ColKind)]) -> BTreeMap<String, ColKind> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    // ---- happy paths per operator/type ----

    #[test]
    fn eq_text_explicit() {
        let filter = json!({"genre": {"$eq": "drama"}});
        let got = translate(&filter, &cols(&[("genre", ColKind::Text)])).unwrap();
        assert_eq!(got, r#"("genre" = 'drama'::text)"#);
    }

    #[test]
    fn implicit_eq_matches_explicit_dollar_eq() {
        let columns = cols(&[("genre", ColKind::Text)]);
        let implicit = translate(&json!({"genre": "drama"}), &columns).unwrap();
        let explicit = translate(&json!({"genre": {"$eq": "drama"}}), &columns).unwrap();
        assert_eq!(implicit, explicit);
        assert_eq!(implicit, r#"("genre" = 'drama'::text)"#);
    }

    #[test]
    fn ne_text() {
        let filter = json!({"genre": {"$ne": "drama"}});
        let got = translate(&filter, &cols(&[("genre", ColKind::Text)])).unwrap();
        assert_eq!(got, r#"("genre" <> 'drama'::text)"#);
    }

    #[test]
    fn gt_gte_lt_lte_int() {
        let columns = cols(&[("score", ColKind::Int)]);
        assert_eq!(
            translate(&json!({"score": {"$gt": 5}}), &columns).unwrap(),
            r#"("score" > 5::int8)"#
        );
        assert_eq!(
            translate(&json!({"score": {"$gte": 5}}), &columns).unwrap(),
            r#"("score" >= 5::int8)"#
        );
        assert_eq!(
            translate(&json!({"score": {"$lt": 5}}), &columns).unwrap(),
            r#"("score" < 5::int8)"#
        );
        assert_eq!(
            translate(&json!({"score": {"$lte": 5}}), &columns).unwrap(),
            r#"("score" <= 5::int8)"#
        );
    }

    #[test]
    fn gt_float() {
        let filter = json!({"weight": {"$gt": 1.5}});
        let got = translate(&filter, &cols(&[("weight", ColKind::Float)])).unwrap();
        assert_eq!(got, r#"("weight" > 1.5::float8)"#);
    }

    #[test]
    fn gte_numeric() {
        let filter = json!({"price": {"$gte": 9.99}});
        let got = translate(&filter, &cols(&[("price", ColKind::Numeric)])).unwrap();
        assert_eq!(got, r#"("price" >= 9.99::numeric)"#);
    }

    #[test]
    fn eq_bool_true_and_false() {
        let columns = cols(&[("active", ColKind::Bool)]);
        assert_eq!(
            translate(&json!({"active": {"$eq": true}}), &columns).unwrap(),
            r#"("active" = TRUE::bool)"#
        );
        assert_eq!(
            translate(&json!({"active": false}), &columns).unwrap(),
            r#"("active" = FALSE::bool)"#
        );
    }

    #[test]
    fn in_int() {
        let filter = json!({"score": {"$in": [1, 2, 3]}});
        let got = translate(&filter, &cols(&[("score", ColKind::Int)])).unwrap();
        assert_eq!(got, r#"("score" IN (1::int8, 2::int8, 3::int8))"#);
    }

    #[test]
    fn nin_text() {
        let filter = json!({"genre": {"$nin": ["a", "b"]}});
        let got = translate(&filter, &cols(&[("genre", ColKind::Text)])).unwrap();
        assert_eq!(got, r#"("genre" NOT IN ('a'::text, 'b'::text))"#);
    }

    #[test]
    fn empty_in_is_false() {
        let filter = json!({"score": {"$in": []}});
        let got = translate(&filter, &cols(&[("score", ColKind::Int)])).unwrap();
        assert_eq!(got, "(FALSE)");
    }

    #[test]
    fn empty_nin_is_exists() {
        // "nothing to exclude" is "exists, and isn't in the empty set", not a
        // vacuous TRUE — a NULL column still shouldn't match (see the module
        // doc's pinned-semantics section).
        let filter = json!({"score": {"$nin": []}});
        let got = translate(&filter, &cols(&[("score", ColKind::Int)])).unwrap();
        assert_eq!(got, r#"("score" IS NOT NULL)"#);
    }

    #[test]
    fn exists_true_and_false() {
        let columns = cols(&[("genre", ColKind::Text)]);
        assert_eq!(
            translate(&json!({"genre": {"$exists": true}}), &columns).unwrap(),
            r#"("genre" IS NOT NULL)"#
        );
        assert_eq!(
            translate(&json!({"genre": {"$exists": false}}), &columns).unwrap(),
            r#"("genre" IS NULL)"#
        );
    }

    #[test]
    fn exists_non_bool_errors() {
        let filter = json!({"genre": {"$exists": "yes"}});
        let err = translate(&filter, &cols(&[("genre", ColKind::Text)])).unwrap_err();
        assert!(err.contains("$exists"), "err = {err}");
    }

    #[test]
    fn text_array_eq_renders_any_membership() {
        let columns = cols(&[("tags", ColKind::TextArray)]);
        let explicit = translate(&json!({"tags": {"$eq": "x"}}), &columns).unwrap();
        assert_eq!(explicit, r#"('x'::text = ANY("tags"))"#);
        // Implicit form is the same shorthand.
        let implicit = translate(&json!({"tags": "x"}), &columns).unwrap();
        assert_eq!(implicit, explicit);
    }

    #[test]
    fn text_array_exists_still_works() {
        let filter = json!({"tags": {"$exists": true}});
        let got = translate(&filter, &cols(&[("tags", ColKind::TextArray)])).unwrap();
        assert_eq!(got, r#"("tags" IS NOT NULL)"#);
    }

    #[test]
    fn text_array_non_range_operators_error() {
        let columns = cols(&[("tags", ColKind::TextArray)]);
        for op in ["$ne", "$in", "$nin"] {
            let filter = json!({"tags": {op: "x"}});
            let err = translate(&filter, &columns).unwrap_err();
            assert!(err.contains("text[]"), "op = {op}, err = {err}");
            assert!(err.contains(op), "op = {op}, err = {err}");
        }
    }

    #[test]
    fn text_array_range_operators_error() {
        let columns = cols(&[("tags", ColKind::TextArray)]);
        for op in ["$gt", "$gte", "$lt", "$lte"] {
            let filter = json!({"tags": {op: "x"}});
            let err = translate(&filter, &columns).unwrap_err();
            assert_eq!(err, RANGE_NOT_SUPPORTED_ON_TEXT, "op = {op}");
        }
    }

    #[test]
    fn text_range_operators_error() {
        let columns = cols(&[("genre", ColKind::Text)]);
        for op in ["$gt", "$gte", "$lt", "$lte"] {
            let filter = json!({"genre": {op: "x"}});
            let err = translate(&filter, &columns).unwrap_err();
            assert_eq!(err, RANGE_NOT_SUPPORTED_ON_TEXT, "op = {op}");
        }
    }

    #[test]
    fn text_array_eq_requires_string_value() {
        let filter = json!({"tags": {"$eq": 1}});
        let err = translate(&filter, &cols(&[("tags", ColKind::TextArray)])).unwrap_err();
        assert!(err.contains("string"), "err = {err}");
    }

    // ---- nesting: $and / $or ----

    #[test]
    fn multi_key_object_is_and() {
        let filter = json!({"a": 1, "b": 2});
        let got = translate(&filter, &cols(&[("a", ColKind::Int), ("b", ColKind::Int)])).unwrap();
        assert_eq!(got, r#"("a" = 1::int8 AND "b" = 2::int8)"#);
    }

    #[test]
    fn multi_operator_same_column_is_and() {
        let filter = json!({"price": {"$gte": 5, "$lte": 10}});
        let got = translate(&filter, &cols(&[("price", ColKind::Numeric)])).unwrap();
        assert_eq!(got, r#"("price" >= 5::numeric AND "price" <= 10::numeric)"#);
    }

    #[test]
    fn dollar_and_matches_equivalent_multi_key_object() {
        let columns = cols(&[("a", ColKind::Int), ("b", ColKind::Int)]);
        let and_form = translate(&json!({"$and": [{"a": 1}, {"b": 2}]}), &columns).unwrap();
        let flat_form = translate(&json!({"a": 1, "b": 2}), &columns).unwrap();
        assert_eq!(and_form, flat_form);
    }

    #[test]
    fn dollar_or_is_self_parenthesized() {
        let columns = cols(&[("a", ColKind::Int), ("b", ColKind::Int)]);
        let got = translate(&json!({"$or": [{"a": 1}, {"b": 2}]}), &columns).unwrap();
        assert_eq!(got, r#"(("a" = 1::int8 OR "b" = 2::int8))"#);
    }

    #[test]
    fn dollar_or_mixed_with_sibling_column() {
        let columns = cols(&[
            ("a", ColKind::Int),
            ("b", ColKind::Int),
            ("c", ColKind::Int),
        ]);
        let filter = json!({"$or": [{"a": 1}, {"b": 2}], "c": 3});
        let got = translate(&filter, &columns).unwrap();
        assert_eq!(
            got,
            r#"(("a" = 1::int8 OR "b" = 2::int8) AND "c" = 3::int8)"#
        );
    }

    #[test]
    fn empty_and_array_errors() {
        let filter = json!({"$and": []});
        let err = translate(&filter, &cols(&[("a", ColKind::Int)])).unwrap_err();
        assert!(err.contains("$and"), "err = {err}");
    }

    #[test]
    fn empty_or_array_errors() {
        let filter = json!({"$or": []});
        let err = translate(&filter, &cols(&[("a", ColKind::Int)])).unwrap_err();
        assert!(err.contains("$or"), "err = {err}");
    }

    #[test]
    fn and_value_must_be_array() {
        let filter = json!({"$and": {"a": 1}});
        let err = translate(&filter, &cols(&[("a", ColKind::Int)])).unwrap_err();
        assert!(err.contains("array"), "err = {err}");
    }

    // ---- error paths ----

    #[test]
    fn unknown_column_errors() {
        let filter = json!({"ghost": 1});
        let err = translate(&filter, &cols(&[("a", ColKind::Int)])).unwrap_err();
        assert!(err.contains("unknown column"), "err = {err}");
        assert!(err.contains("ghost"), "err = {err}");
    }

    #[test]
    fn unsupported_column_type_names_the_type() {
        let filter = json!({"created_at": "2024-01-01"});
        let err = translate(
            &filter,
            &cols(&[(
                "created_at",
                ColKind::Unsupported("timestamptz".to_string()),
            )]),
        )
        .unwrap_err();
        assert_eq!(
            err,
            "column 'created_at' has type timestamptz, which pg_pinecone filters do not support"
        );
    }

    #[test]
    fn unknown_operator_errors() {
        let filter = json!({"a": {"$bogus": 1}});
        let err = translate(&filter, &cols(&[("a", ColKind::Int)])).unwrap_err();
        assert!(err.contains("unknown operator"), "err = {err}");
        assert!(err.contains("$bogus"), "err = {err}");
    }

    #[test]
    fn wrong_value_type_for_column_errors() {
        let filter = json!({"a": "not-a-number"});
        let err = translate(&filter, &cols(&[("a", ColKind::Int)])).unwrap_err();
        assert!(err.contains("column type Int"), "err = {err}");
    }

    #[test]
    fn fractional_number_on_int_column_errors() {
        let filter = json!({"score": {"$eq": 3.5}});
        let err = translate(&filter, &cols(&[("score", ColKind::Int)])).unwrap_err();
        assert!(err.contains("Int column"), "err = {err}");
    }

    #[test]
    fn in_requires_array() {
        let filter = json!({"a": {"$in": 5}});
        let err = translate(&filter, &cols(&[("a", ColKind::Int)])).unwrap_err();
        assert!(err.contains("$in"), "err = {err}");
    }

    #[test]
    fn empty_operator_object_errors() {
        let filter = json!({"a": {}});
        let err = translate(&filter, &cols(&[("a", ColKind::Int)])).unwrap_err();
        assert!(err.contains("empty"), "err = {err}");
    }

    #[test]
    fn top_level_must_be_object() {
        let err = translate(
            &json!(["not", "an", "object"]),
            &cols(&[("a", ColKind::Int)]),
        )
        .unwrap_err();
        assert!(err.contains("JSON object"), "err = {err}");
    }

    #[test]
    fn empty_top_level_object_errors() {
        let err = translate(&json!({}), &cols(&[("a", ColKind::Int)])).unwrap_err();
        assert!(err.contains("at least one key"), "err = {err}");
    }

    // ---- depth cap ----

    fn nest(n: usize, leaf: Value) -> Value {
        let mut v = leaf;
        for _ in 0..n {
            v = json!({"$and": [v]});
        }
        v
    }

    #[test]
    fn depth_at_cap_succeeds() {
        let columns = cols(&[("a", ColKind::Int)]);
        // 9 wrapping `$and` layers around a depth-1 leaf = depth 10 (== MAX_DEPTH).
        let filter = nest(9, json!({"a": 1}));
        assert!(translate(&filter, &columns).is_ok());
    }

    #[test]
    fn depth_over_cap_errors() {
        let columns = cols(&[("a", ColKind::Int)]);
        // 10 wrapping layers = depth 11 (> MAX_DEPTH).
        let filter = nest(10, json!({"a": 1}));
        let err = translate(&filter, &columns).unwrap_err();
        assert!(err.contains("depth"), "err = {err}");
    }

    // ---- adversarial quoting ----

    #[test]
    fn single_quote_in_value_is_escaped() {
        let filter = json!({"name": {"$eq": "O'Brien"}});
        let got = translate(&filter, &cols(&[("name", ColKind::Text)])).unwrap();
        assert_eq!(got, r#"("name" = 'O''Brien'::text)"#);
    }

    #[test]
    fn double_quote_in_column_name_is_escaped() {
        let columns = cols(&[("a\"b", ColKind::Text)]);
        let filter = json!({"a\"b": "x"});
        let got = translate(&filter, &columns).unwrap();
        assert_eq!(got, r#"("a""b" = 'x'::text)"#);
    }

    #[test]
    fn sql_injection_attempt_is_rendered_as_inert_data() {
        let filter = json!({"name": {"$eq": "'; DROP TABLE users; --"}});
        let got = translate(&filter, &cols(&[("name", ColKind::Text)])).unwrap();
        assert_eq!(got, r#"("name" = '''; DROP TABLE users; --'::text)"#);
    }

    #[test]
    fn unicode_value_round_trips() {
        let filter = json!({"name": {"$eq": "héllo wörld 😀"}});
        let got = translate(&filter, &cols(&[("name", ColKind::Text)])).unwrap();
        assert_eq!(got, "(\"name\" = 'héllo wörld 😀'::text)");
    }

    #[test]
    fn backslash_in_value_renders_as_escape_string() {
        // A value containing a backslash renders `E'...'` with both `\` and
        // `'` doubled — see quote_string's doc comment for why this (rather
        // than a plain `'...'`) is required for GUC-independent safety.
        let filter = json!({"path": {"$eq": "C:\\temp\\x"}});
        let got = translate(&filter, &cols(&[("path", ColKind::Text)])).unwrap();
        assert_eq!(got, r#"("path" = E'C:\\temp\\x'::text)"#);
    }

    #[test]
    fn trailing_backslash_in_value_is_doubled() {
        // A trailing backslash is the classic injection vector under
        // standard_conforming_strings=off: left single (in a plain '...'
        // literal) it would escape the closing quote. Doubling it inside an
        // E'...' literal is safe regardless of that GUC.
        let filter = json!({"path": {"$eq": "abc\\"}});
        let got = translate(&filter, &cols(&[("path", ColKind::Text)])).unwrap();
        assert_eq!(got, r#"("path" = E'abc\\'::text)"#);
    }

    #[test]
    fn backslash_and_quote_combo_are_both_doubled() {
        let filter = json!({"path": {"$eq": "a\\'b"}});
        let got = translate(&filter, &cols(&[("path", ColKind::Text)])).unwrap();
        assert_eq!(got, r#"("path" = E'a\\''b'::text)"#);
    }

    #[test]
    fn nul_byte_in_value_errors() {
        let filter = json!({"name": {"$eq": "bad\0value"}});
        let err = translate(&filter, &cols(&[("name", ColKind::Text)])).unwrap_err();
        assert!(err.contains("NUL"), "err = {err}");
    }

    // ---- referenced_columns ----

    #[test]
    fn referenced_columns_flat_object() {
        let mut out = BTreeSet::new();
        referenced_columns(&json!({"category": "x", "score": {"$gte": 1}}), &mut out);
        assert_eq!(
            out,
            BTreeSet::from(["category".to_string(), "score".to_string()])
        );
    }

    #[test]
    fn referenced_columns_recurses_into_and_or() {
        let mut out = BTreeSet::new();
        referenced_columns(
            &json!({"$or": [{"a": 1}, {"$and": [{"b": 2}, {"c": 3}]}]}),
            &mut out,
        );
        assert_eq!(
            out,
            BTreeSet::from(["a".to_string(), "b".to_string(), "c".to_string()])
        );
    }

    #[test]
    fn referenced_columns_ignores_dropped_unreferenced_column() {
        // A column that isn't mentioned anywhere in the filter must never
        // show up here, even if it's otherwise a registered metadata column —
        // that's what lets remote validation skip it.
        let mut out = BTreeSet::new();
        referenced_columns(&json!({"category": "x"}), &mut out);
        assert!(!out.contains("views"));
    }

    #[test]
    fn column_named_dollar_eq_is_treated_as_a_column() {
        let columns = cols(&[("$eq", ColKind::Text)]);
        let filter = json!({"$eq": "v"});
        let got = translate(&filter, &columns).unwrap();
        assert_eq!(got, r#"("$eq" = 'v'::text)"#);
    }

    #[test]
    fn equality_on_finds_the_shorthand_and_the_eq_operator() {
        assert_eq!(equality_on(&json!({"t": 7}), "t"), Some(&json!(7)));
        assert_eq!(equality_on(&json!({"t": "a"}), "t"), Some(&json!("a")));
        assert_eq!(equality_on(&json!({"t": {"$eq": 7}}), "t"), Some(&json!(7)));
        assert_eq!(
            equality_on(&json!({"t": {"$gt": 1, "$eq": 7}}), "t"),
            Some(&json!(7))
        );
    }

    #[test]
    fn equality_on_looks_into_and_and_nothing_else() {
        let nested = json!({"$and": [{"c": "x"}, {"$and": [{"t": {"$eq": 7}}]}]});
        assert_eq!(equality_on(&nested, "t"), Some(&json!(7)));

        for filter in [
            json!({"$or": [{"t": {"$eq": 7}}, {"t": {"$eq": 8}}]}),
            json!({"t": {"$ne": 7}}),
            json!({"t": {"$in": [7]}}),
            json!({"t": {"$gte": 7}}),
            json!({"t": [7]}),
            json!({"t": null}),
            json!({"other": 7}),
            json!({"$and": "t"}),
            json!(7),
        ] {
            assert_eq!(equality_on(&filter, "t"), None, "{filter}");
        }
    }

    #[test]
    fn equality_on_stops_at_the_nesting_cap_translate_enforces() {
        let mut filter = json!({"t": 7});
        for _ in 0..MAX_DEPTH {
            filter = json!({"$and": [filter]});
        }
        assert_eq!(equality_on(&filter, "t"), None);
    }
}
