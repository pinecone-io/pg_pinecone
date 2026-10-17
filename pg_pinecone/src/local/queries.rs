//! The queries a build calibrates on, when the user names them: `calibration_queries =
//! 'schema.table'` or `'schema.table(column)'`.
//!
//! A build draws up to 200 distinct queries of the table, seeded by `seed`, in one statement run as the
//! owner of the indexed table: `index_build()` switches to that owner and restricts the search path to
//! `pg_catalog` (hence the schema in the name). The owner's privileges are checked first, explicitly, so
//! a rebuild a superuser drives (REINDEX, `reindex_native_indexes()`, the maintenance worker) never
//! reads what the owner cannot. Everything runs in a subtransaction that is rolled back: its locks end
//! with the read and any ERROR comes out as an answer.
//!
//! A problem is an ERROR at CREATE INDEX. At a rebuild it is a WARNING and the build calibrates on table
//! rows instead: a REINDEX must not fail because a table the owner dropped is gone.

use std::{ffi::CString, panic::AssertUnwindSafe};

use pgrx::{
    PgSqlErrorCode,
    prelude::*,
    spi::{Spi, quote_identifier, quote_qualified_identifier},
};

use super::calibrate::MIN_SAMPLES;
use crate::docs_guard::{self, Caught, sqlstate};

/// Fewer queries than this cannot calibrate (the floor of the row sample).
pub(crate) const MIN_QUERIES: usize = MIN_SAMPLES;

/// A table of more rows than this is refused: the draw sorts all of them (about 2 s for 100,000 768-d
/// rows), and a few hundred queries are all it uses. The planner's estimate refuses early; a count up to
/// the cap decides, as the estimate can be stale (a load after the last ANALYZE) or absent (a view).
const MAX_ROWS: f32 = 100_000.0;

/// How long the read waits for a lock the table's indexes or TOAST need, at most: a busy table is a
/// problem to report, not a reason to stall a rebuild that holds the indexed table.
const LOCK_WAIT_MS: i32 = 2_000;

/// Why a table of queries cannot be used.
#[derive(Debug)]
pub(crate) struct Problem {
    pub(crate) code: PgSqlErrorCode,
    pub(crate) message: String,
    pub(crate) hint: Option<String>,
}

impl Problem {
    fn new(code: PgSqlErrorCode, message: String, hint: &str) -> Problem {
        Problem {
            code,
            message,
            hint: Some(hint.to_string()),
        }
    }

    /// An ERROR at CREATE INDEX, a WARNING at a rebuild (the caller then calibrates on rows).
    pub(crate) fn raise(self, rebuild: bool) {
        let (text, hint, level) = if rebuild {
            (
                format!(
                    "pg_pinecone: calibration_queries: {}; calibrating on table rows instead",
                    self.message
                ),
                Some(
                    "ALTER INDEX ... RESET (calibration_queries) stops this; fix or restore the \
                     table and REINDEX to calibrate on it again."
                        .to_string(),
                ),
                PgLogLevel::WARNING,
            )
        } else {
            (
                format!("pg_pinecone: calibration_queries: {}", self.message),
                self.hint,
                PgLogLevel::ERROR,
            )
        };
        let code = if rebuild {
            PgSqlErrorCode::ERRCODE_WARNING
        } else {
            self.code
        };
        let mut report = pgrx::pg_sys::panic::ErrorReport::new(code, text, pgrx::function_name!());
        if let Some(hint) = hint {
            report = report.set_hint(hint);
        }
        report.report(level);
    }
}

/// `schema.table` or `schema.table(column)`, as parsed from the reloption.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Spec {
    pub(crate) schema: String,
    pub(crate) table: String,
    pub(crate) column: Option<String>,
}

struct Cursor<'a> {
    chars: &'a [char],
    at: usize,
}

impl Cursor<'_> {
    fn skip_blanks(&mut self) {
        while self.chars.get(self.at).is_some_and(|c| c.is_whitespace()) {
            self.at += 1;
        }
    }

    fn eat(&mut self, want: char) -> bool {
        self.skip_blanks();
        let found = self.chars.get(self.at) == Some(&want);
        self.at += usize::from(found);
        found
    }

    /// A SQL identifier: `"quoted"` as written (`""` is one quote), or bare, folded to lower case.
    fn identifier(&mut self) -> Option<String> {
        self.skip_blanks();
        if self.chars.get(self.at) == Some(&'"') {
            self.at += 1;
            let mut name = String::new();
            loop {
                match self.chars.get(self.at)? {
                    '"' if self.chars.get(self.at + 1) == Some(&'"') => {
                        name.push('"');
                        self.at += 2;
                    }
                    '"' => {
                        self.at += 1;
                        break;
                    }
                    c => {
                        name.push(*c);
                        self.at += 1;
                    }
                }
            }
            return (!name.is_empty()).then_some(name);
        }
        let first = |c: char| c.is_ascii_alphabetic() || c == '_' || !c.is_ascii();
        let rest = |c: char| first(c) || c.is_ascii_digit() || c == '$';
        let start = self.at;
        if !self.chars.get(self.at).copied().is_some_and(first) {
            return None;
        }
        while self.chars.get(self.at).copied().is_some_and(rest) {
            self.at += 1;
        }
        Some(
            self.chars[start..self.at]
                .iter()
                .map(char::to_ascii_lowercase)
                .collect(),
        )
    }

    fn at_end(&mut self) -> bool {
        self.skip_blanks();
        self.at == self.chars.len()
    }
}

/// Parses the reloption: the schema is required because a build runs with a restricted search path.
pub(crate) fn parse_spec(raw: &str) -> Result<Spec, String> {
    let chars: Vec<char> = raw.chars().collect();
    let mut cursor = Cursor {
        chars: &chars,
        at: 0,
    };
    let parsed = (|| {
        let schema = cursor.identifier()?;
        cursor.eat('.').then_some(())?;
        let table = cursor.identifier()?;
        let column = if cursor.eat('(') {
            let column = cursor.identifier()?;
            cursor.eat(')').then_some(())?;
            Some(column)
        } else {
            None
        };
        cursor.at_end().then_some(Spec {
            schema,
            table,
            column,
        })
    })();
    parsed.ok_or_else(|| format!("\"{raw}\" must be schema.table or schema.table(column)"))
}

/// [`parse_spec`] with the error a user sees.
pub(crate) fn parse(raw: &str) -> Result<Spec, Problem> {
    parse_spec(raw).map_err(|message| {
        Problem::new(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            message,
            "Name the schema: 'public.queries' or 'public.queries(embedding)'.",
        )
    })
}

/// A live column of the queries table.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Column {
    pub(crate) name: String,
    /// For messages: `format_type` of the column's type.
    pub(crate) type_name: String,
    /// `atttypmod`: the dimension of a `vector(n)`, -1 for a bare `vector`.
    pub(crate) typmod: i32,
    /// Whether the column has the indexed column's type (its oid, not just a type called `vector`: a
    /// domain or composite of that name would hand the read something else).
    pub(crate) is_vector: bool,
}

fn describe(column: &Column) -> String {
    if column.typmod >= 0 {
        format!("{} vector({})", column.name, column.typmod)
    } else {
        format!("{} vector", column.name)
    }
}

/// The column the queries are read from: the one named, or the one vector column of the indexed
/// dimension (a bare `vector` column has no dimension to match, so it is only ever named).
pub(crate) fn choose_column<'a>(
    table: &str,
    columns: &'a [Column],
    wanted: Option<&str>,
    dim: u32,
) -> Result<&'a Column, Problem> {
    if let Some(wanted) = wanted {
        let column = columns.iter().find(|c| c.name == wanted).ok_or_else(|| {
            Problem::new(
                PgSqlErrorCode::ERRCODE_UNDEFINED_COLUMN,
                format!("table \"{table}\" has no column \"{wanted}\""),
                "Name a vector column: 'schema.table(column)'.",
            )
        })?;
        if !column.is_vector {
            return Err(Problem::new(
                PgSqlErrorCode::ERRCODE_DATATYPE_MISMATCH,
                format!(
                    "column \"{wanted}\" of \"{table}\" has type {}, not vector({dim})",
                    column.type_name
                ),
                "The queries are embedded with the model of the indexed column.",
            ));
        }
        if column.typmod >= 0 && column.typmod != dim as i32 {
            return Err(Problem::new(
                PgSqlErrorCode::ERRCODE_DATATYPE_MISMATCH,
                format!(
                    "column \"{wanted}\" of \"{table}\" is vector({}), the indexed column is \
                     vector({dim})",
                    column.typmod
                ),
                "The queries are embedded with the model of the indexed column.",
            ));
        }
        return Ok(column);
    }
    let fits: Vec<&Column> = columns
        .iter()
        .filter(|c| c.is_vector && c.typmod == dim as i32)
        .collect();
    match fits.as_slice() {
        [one] => Ok(one),
        [] => {
            let vectors: Vec<String> = columns
                .iter()
                .filter(|c| c.is_vector)
                .map(describe)
                .collect();
            if vectors.is_empty() {
                Err(Problem::new(
                    PgSqlErrorCode::ERRCODE_UNDEFINED_COLUMN,
                    format!("table \"{table}\" has no vector column"),
                    "Name a vector column: 'schema.table(column)'.",
                ))
            } else {
                Err(Problem::new(
                    PgSqlErrorCode::ERRCODE_DATATYPE_MISMATCH,
                    format!(
                        "table \"{table}\" has no vector({dim}) column (its vector columns: {})",
                        vectors.join(", ")
                    ),
                    "Name a column: 'schema.table(column)'; a bare vector column can be named.",
                ))
            }
        }
        many => {
            let names: Vec<String> = many.iter().map(|c| format!("\"{}\"", c.name)).collect();
            Err(Problem::new(
                PgSqlErrorCode::ERRCODE_AMBIGUOUS_COLUMN,
                format!(
                    "table \"{table}\" has {} vector({dim}) columns ({})",
                    many.len(),
                    names.join(", ")
                ),
                &format!("Name one of them: '{table}(column)'."),
            ))
        }
    }
}

/// A query must have the index's dimension and finite components.
pub(crate) fn check_vector(table: &str, vector: &[f32], dim: usize) -> Result<(), Problem> {
    if vector.len() != dim {
        return Err(Problem::new(
            PgSqlErrorCode::ERRCODE_DATATYPE_MISMATCH,
            format!(
                "a query in \"{table}\" has {} dimensions, the index has {dim}",
                vector.len()
            ),
            "Embed the queries with the model of the indexed column.",
        ));
    }
    if let Some(at) = super::metric::first_non_finite(vector) {
        let what = if vector[at].is_nan() {
            "NaN"
        } else {
            "Infinity"
        };
        return Err(Problem::new(
            PgSqlErrorCode::ERRCODE_DATA_EXCEPTION,
            format!(
                "a query in \"{table}\" has a non-finite value ({what}) in component {}",
                at + 1
            ),
            "Remove the row, or re-embed it.",
        ));
    }
    Ok(())
}

/// The queries a build calibrates on, and the table they came from.
pub(crate) struct Drawn {
    /// `schema.table`, quoted as needed.
    pub(crate) table: String,
    pub(crate) vectors: Vec<Vec<f32>>,
}

/// What `calibration_queries` gives a build.
pub(crate) enum Draw {
    /// Not set: calibrate on table rows.
    Unset,
    /// Set but unusable at a rebuild (a WARNING said why): calibrate on table rows.
    Fallback,
    Drawn(Drawn),
}

/// Errors that say nothing about the table: a cancel, a shutdown, a deadlock, exhausted resources. A
/// lock timeout is not one: `fetch` sets it, and it means the table is busy.
fn is_interruption(caught: &Caught) -> bool {
    caught.code != PgSqlErrorCode::ERRCODE_LOCK_NOT_AVAILABLE
        && (caught.transient
            || matches!(
                caught.code,
                PgSqlErrorCode::ERRCODE_ADMIN_SHUTDOWN | PgSqlErrorCode::ERRCODE_CRASH_SHUTDOWN
            ))
}

/// Reads up to `limit` distinct queries of the table `raw` names, as the owner of `heap` would. A
/// problem is an ERROR at CREATE INDEX and, at a rebuild, a WARNING and [`Draw::Fallback`]; an
/// interruption is raised either way.
pub(crate) fn draw(
    raw: Option<&str>,
    heap: pg_sys::Relation,
    heap_attnum: i16,
    dim: u32,
    seed: u64,
    limit: usize,
    rebuild: bool,
) -> Draw {
    let Some(raw) = raw.map(str::trim).filter(|raw| !raw.is_empty()) else {
        return Draw::Unset;
    };
    let spec = match parse(raw) {
        Ok(spec) => spec,
        Err(problem) => {
            problem.raise(rebuild);
            return Draw::Fallback;
        }
    };
    let (heap_id, owner) = unsafe { ((*heap).rd_id, (*(*heap).rd_rel).relowner) };
    let indexed_type = unsafe { pg_sys::get_atttype(heap_id, heap_attnum) };
    let attempt = docs_guard::rolled_back_caught(AssertUnwindSafe(|| {
        fetch(&spec, owner, heap_id, indexed_type, dim, seed, limit)
    }));
    match attempt {
        Ok(Ok(drawn)) => Draw::Drawn(drawn),
        Ok(Err(problem)) => {
            problem.raise(rebuild);
            Draw::Fallback
        }
        Err(caught) => {
            let rebuild = rebuild && !is_interruption(&caught);
            // A rebuild's WARNING reaches whoever ran it, not the owner: the text of an error a view
            // raised stays in the server log.
            let message = if rebuild {
                log!(
                    "pg_pinecone: calibration_queries: reading the table failed: {}",
                    caught.message
                );
                format!(
                    "reading the table failed with SQLSTATE {} (the server log has the message)",
                    sqlstate(caught.code)
                )
            } else {
                caught.message
            };
            Problem {
                code: caught.code,
                message,
                hint: None,
            }
            .raise(rebuild);
            Draw::Fallback
        }
    }
}

fn role_name(role: pg_sys::Oid) -> Option<String> {
    unsafe {
        let name = pg_sys::GetUserNameFromId(role, true);
        (!name.is_null()).then(|| {
            std::ffi::CStr::from_ptr(name)
                .to_string_lossy()
                .into_owned()
        })
    }
}

fn grant_hint(grant: &str, role: &Option<String>) -> Option<String> {
    role.as_ref()
        .map(|role| format!("GRANT {grant} TO {};", quote_identifier(role)))
}

/// The catalog and privilege checks, the lock and the read, in that order: nothing is locked or read
/// that the owner of the indexed table could not read.
fn fetch(
    spec: &Spec,
    owner: pg_sys::Oid,
    heap_id: pg_sys::Oid,
    indexed_type: pg_sys::Oid,
    dim: u32,
    seed: u64,
    limit: usize,
) -> Result<Drawn, Problem> {
    let heap_name = crate::catalog::display_name_of(heap_id);
    let name = quote_qualified_identifier(&spec.schema, &spec.table);
    let role = role_name(owner);
    let who = role.as_deref().unwrap_or("an unknown role");
    let missing = || {
        Problem::new(
            PgSqlErrorCode::ERRCODE_UNDEFINED_TABLE,
            format!("table \"{name}\" does not exist"),
            "Create the table, or leave calibration_queries out.",
        )
    };
    let schema = CString::new(spec.schema.as_str()).map_err(|_| missing())?;
    let table = CString::new(spec.table.as_str()).map_err(|_| missing())?;
    let namespace = unsafe { pg_sys::get_namespace_oid(schema.as_ptr(), true) };
    if namespace == pg_sys::InvalidOid {
        return Err(missing());
    }
    let may_use_schema = Spi::get_one_with_args::<bool>(
        "SELECT pg_catalog.has_schema_privilege($1::oid, $2::oid, 'USAGE')",
        &[owner.into(), namespace.into()],
    )
    .expect("pg_pinecone: has_schema_privilege failed")
    .unwrap_or(false);
    if !may_use_schema {
        return Err(Problem {
            code: PgSqlErrorCode::ERRCODE_INSUFFICIENT_PRIVILEGE,
            message: format!(
                "permission denied for schema \"{}\": a build reads the table as the owner of \
                 \"{heap_name}\" ({who})",
                spec.schema
            ),
            hint: grant_hint(
                &format!("USAGE ON SCHEMA {}", quote_identifier(&spec.schema)),
                &role,
            ),
        });
    }
    let relid = unsafe { pg_sys::get_relname_relid(table.as_ptr(), namespace) };
    if relid == pg_sys::InvalidOid {
        return Err(missing());
    }
    if relid == heap_id {
        return Err(Problem::new(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "\"{name}\" is the indexed table itself: its rows are the default sample, and as \
                 queries each would find itself"
            ),
            "Name a table of queries that are not rows of the indexed table.",
        ));
    }
    let kind = unsafe { pg_sys::get_rel_relkind(relid) } as u8 as char;
    if kind == 'f' {
        return Err(Problem::new(
            PgSqlErrorCode::ERRCODE_WRONG_OBJECT_TYPE,
            format!(
                "\"{name}\" is a foreign table: a remote that does not answer would stall the build"
            ),
            "Copy the queries into a local table or a materialized view.",
        ));
    }
    if !"rpvm".contains(kind) {
        return Err(Problem::new(
            PgSqlErrorCode::ERRCODE_WRONG_OBJECT_TYPE,
            format!("\"{name}\" is not a table or view"),
            "Name a table or view that holds the queries.",
        ));
    }
    let rows = Spi::get_one_with_args::<f32>(
        "SELECT c.reltuples FROM pg_catalog.pg_class c WHERE c.oid = $1::oid",
        &[relid.into()],
    )
    .expect("pg_pinecone: reltuples query failed")
    .unwrap_or(-1.0);
    let too_big = |rows: &str| {
        Problem::new(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "table \"{name}\" has {rows} rows: drawing the queries sorts all of them, and a few \
                 hundred are enough"
            ),
            "Name a table or a view over a sample of at most 100,000 queries.",
        )
    };
    if rows > MAX_ROWS {
        return Err(too_big(&format!("about {rows:.0}")));
    }
    let columns = columns_of(relid, indexed_type);
    let column = choose_column(&name, &columns, spec.column.as_deref(), dim)?;
    let may_select = Spi::get_one_with_args::<bool>(
        "SELECT pg_catalog.has_column_privilege($1::oid, $2::oid, $3, 'SELECT')",
        &[owner.into(), relid.into(), column.name.as_str().into()],
    )
    .expect("pg_pinecone: has_column_privilege failed")
    .unwrap_or(false);
    if !may_select {
        return Err(Problem {
            code: PgSqlErrorCode::ERRCODE_INSUFFICIENT_PRIVILEGE,
            message: format!(
                "permission denied for table \"{name}\": a build reads it as the owner of \
                 \"{heap_name}\" ({who})"
            ),
            hint: grant_hint(&format!("SELECT ON {name}"), &role),
        });
    }
    // No wait: a refresh that holds the table must not stall a rebuild, which holds the indexed
    // table's lock. The subtransaction's rollback gives the lock back, and the timeout with it.
    Spi::run(&format!(
        "SELECT pg_catalog.set_config('lock_timeout', \
                (CASE WHEN s.setting::int = 0 OR s.setting::int > {LOCK_WAIT_MS} \
                      THEN {LOCK_WAIT_MS} ELSE s.setting::int END)::text, true) \
         FROM pg_catalog.pg_settings s WHERE s.name = 'lock_timeout'"
    ))
    .expect("pg_pinecone: setting lock_timeout failed");
    if !unsafe { pg_sys::ConditionalLockRelationOid(relid, pg_sys::AccessShareLock as i32) } {
        return Err(Problem::new(
            PgSqlErrorCode::ERRCODE_LOCK_NOT_AVAILABLE,
            format!("table \"{name}\" is locked by another transaction"),
            "Retry once the other transaction has finished.",
        ));
    }
    // A table dropped or renamed before the lock was granted is not the one that was checked.
    if unsafe { pg_sys::get_relname_relid(table.as_ptr(), namespace) } != relid {
        return Err(missing());
    }
    // The estimate above is a fast refusal, not a pass: it is stale after a load or absent for a view.
    if count_past_cap(&name, &column.name) {
        return Err(too_big(&format!("more than {MAX_ROWS:.0}")));
    }
    read(&name, &column.name, indexed_type, dim, seed, limit)
}

/// Whether `table` holds more than [`MAX_ROWS`] queries in `column`, reading no more rows than that.
fn count_past_cap(table: &str, column: &str) -> bool {
    let counted = Spi::get_one::<i64>(&format!(
        "SELECT pg_catalog.count(*) FROM (SELECT 1 FROM {table} WHERE {col} IS NOT NULL LIMIT {cap}) s",
        col = quote_identifier(column),
        cap = MAX_ROWS as i64 + 1,
    ))
    .expect("pg_pinecone: counting calibration_queries failed")
    .unwrap_or(0);
    counted as f32 > MAX_ROWS
}

fn columns_of(relid: pg_sys::Oid, indexed_type: pg_sys::Oid) -> Vec<Column> {
    Spi::connect(|client| {
        client
            .select(
                "SELECT a.attname::text AS name, pg_catalog.format_type(a.atttypid, NULL) AS type_name, \
                        a.atttypmod AS typmod, a.atttypid = $2::oid AS is_vector \
                 FROM pg_catalog.pg_attribute a \
                 WHERE a.attrelid = $1::oid AND a.attnum > 0 AND NOT a.attisdropped \
                 ORDER BY a.attnum",
                None,
                &[relid.into(), indexed_type.into()],
            )
            .expect("pg_pinecone: column catalog query failed")
            .map(|row| Column {
                name: row
                    .get_by_name::<String, _>("name")
                    .expect("pg_pinecone: column name read failed")
                    .expect("pg_pinecone: attname is never NULL"),
                type_name: row
                    .get_by_name::<String, _>("type_name")
                    .expect("pg_pinecone: type name read failed")
                    .expect("pg_pinecone: format_type is never NULL"),
                typmod: row
                    .get_by_name::<i32, _>("typmod")
                    .expect("pg_pinecone: typmod read failed")
                    .expect("pg_pinecone: atttypmod is never NULL"),
                is_vector: row
                    .get_by_name::<bool, _>("is_vector")
                    .expect("pg_pinecone: type match read failed")
                    .expect("pg_pinecone: the type comparison is never NULL"),
            })
            .collect()
    })
}

/// Up to `limit` distinct queries, drawn by a seeded hash of their text (not by place in the table, so
/// any scan order and table size draws the same ones; equal vectors hash equal and count once), each
/// checked.
fn read(
    table: &str,
    column: &str,
    indexed_type: pg_sys::Oid,
    dim: u32,
    seed: u64,
    limit: usize,
) -> Result<Drawn, Problem> {
    let col = quote_identifier(column);
    let sql = format!(
        "SELECT d.q FROM (\
             SELECT DISTINCT ON (s.h) s.q, s.h FROM (\
                 SELECT {col} AS q, pg_catalog.hashtextextended({col}::pg_catalog.text, $1) AS h \
                 FROM {table} WHERE {col} IS NOT NULL) s \
             ORDER BY s.h) d \
         ORDER BY d.h LIMIT {limit}"
    );
    let mut vectors = Vec::new();
    let mut rejected = None;
    Spi::connect(|client| {
        let mut rows = client
            .select(&sql, None, &[(seed as i64).into()])
            .expect("pg_pinecone: reading calibration_queries failed");
        // The column was checked before the lock; what the statement returns is what is parsed.
        let returned = rows
            .column_type_oid(1)
            .expect("pg_pinecone: calibration_queries result type unreadable")
            .value();
        if returned != indexed_type {
            rejected = Some(Problem::new(
                PgSqlErrorCode::ERRCODE_DATATYPE_MISMATCH,
                format!("column \"{column}\" of \"{table}\" changed type while it was being read"),
                "Retry.",
            ));
            return;
        }
        while rows.next().is_some() {
            let Some(datum) = rows
                .get_datum_by_ordinal(1)
                .expect("pg_pinecone: reading a calibration query failed")
            else {
                continue;
            };
            let vector = unsafe { crate::am::parse_vector(datum) };
            if let Err(problem) = check_vector(table, &vector, dim as usize) {
                rejected = Some(problem);
                return;
            }
            vectors.push(vector);
        }
    });
    if let Some(problem) = rejected {
        return Err(problem);
    }
    if vectors.len() < MIN_QUERIES {
        return Err(Problem::new(
            PgSqlErrorCode::ERRCODE_INVALID_PARAMETER_VALUE,
            format!(
                "table \"{table}\" has {} distinct queries in column \"{column}\": calibration \
                 needs at least {MIN_QUERIES}",
                vectors.len()
            ),
            "Add a few hundred real queries, embedded with the model of the indexed column.",
        ));
    }
    Ok(Drawn {
        table: table.to_string(),
        vectors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(schema: &str, table: &str, column: Option<&str>) -> Spec {
        Spec {
            schema: schema.into(),
            table: table.into(),
            column: column.map(Into::into),
        }
    }

    #[test]
    fn a_name_is_schema_dot_table_with_an_optional_column() {
        for (raw, want) in [
            ("public.queries", spec("public", "queries", None)),
            (
                "public.queries(embedding)",
                spec("public", "queries", Some("embedding")),
            ),
            (
                "  public . queries ( embedding )  ",
                spec("public", "queries", Some("embedding")),
            ),
            (
                "Public.Real_Queries(Emb1)",
                spec("public", "real_queries", Some("emb1")),
            ),
            ("app.q_1$x", spec("app", "q_1$x", None)),
            (
                "\"My Schema\".\"Q \"\"x\"\"\"(\"Emb\")",
                spec("My Schema", "Q \"x\"", Some("Emb")),
            ),
            ("\"a.b\".\"c(d)\"", spec("a.b", "c(d)", None)),
            ("s.t(\"A B\")", spec("s", "t", Some("A B"))),
        ] {
            assert_eq!(parse_spec(raw), Ok(want), "{raw:?}");
        }
    }

    #[test]
    fn anything_else_is_refused_with_the_shape_it_wants() {
        for raw in [
            "",
            "   ",
            "queries",
            "queries(embedding)",
            "a.b.c",
            "public.",
            ".queries",
            "public.queries(",
            "public.queries()",
            "public.queries(a, b)",
            "public.queries(a)(b)",
            "public.queries) ",
            "public.queries embedding",
            "public.\"queries",
            "public.\"\"",
            "public.q-1",
            "public.1q",
            "public.q;drop table t",
        ] {
            let error = parse_spec(raw).expect_err(raw);
            assert!(error.contains("schema.table"), "{raw:?}: {error}");
        }
    }

    fn vector_column(name: &str, dim: i32) -> Column {
        Column {
            name: name.into(),
            type_name: "vector".into(),
            typmod: dim,
            is_vector: true,
        }
    }

    fn other_column(name: &str, type_name: &str) -> Column {
        Column {
            name: name.into(),
            type_name: type_name.into(),
            typmod: -1,
            is_vector: false,
        }
    }

    #[test]
    fn the_default_column_is_the_one_vector_column_of_the_indexed_dimension() {
        let columns = [
            other_column("id", "int8"),
            vector_column("raw", 1536),
            vector_column("embedding", 768),
            vector_column("loose", -1),
        ];
        assert_eq!(
            choose_column("public.q", &columns, None, 768).unwrap().name,
            "embedding"
        );
    }

    #[test]
    fn two_columns_of_the_dimension_are_ambiguous_and_the_error_names_both() {
        let columns = [vector_column("a", 8), vector_column("b", 8)];
        let problem = choose_column("public.q", &columns, None, 8).unwrap_err();
        assert_eq!(problem.code, PgSqlErrorCode::ERRCODE_AMBIGUOUS_COLUMN);
        assert!(
            problem.message.contains("\"a\"") && problem.message.contains("\"b\""),
            "{problem:?}"
        );
        assert!(problem.hint.unwrap().contains("public.q(column)"));
        assert_eq!(
            choose_column("public.q", &columns, Some("b"), 8)
                .unwrap()
                .name,
            "b"
        );
    }

    #[test]
    fn no_default_column_says_which_vector_columns_there_are() {
        let none = [other_column("id", "int8"), other_column("q", "text")];
        let problem = choose_column("public.q", &none, None, 8).unwrap_err();
        assert_eq!(problem.code, PgSqlErrorCode::ERRCODE_UNDEFINED_COLUMN);
        assert!(problem.message.contains("no vector column"), "{problem:?}");

        let wrong = [vector_column("embedding", 1536), vector_column("loose", -1)];
        let problem = choose_column("public.q", &wrong, None, 8).unwrap_err();
        assert_eq!(problem.code, PgSqlErrorCode::ERRCODE_DATATYPE_MISMATCH);
        assert!(
            problem.message.contains("vector(8)")
                && problem.message.contains("embedding vector(1536)")
                && problem.message.contains("loose vector"),
            "{problem:?}"
        );
    }

    #[test]
    fn a_type_that_is_only_called_vector_is_not_one() {
        let lookalike = [Column {
            name: "q".into(),
            type_name: "vector".into(),
            typmod: -1,
            is_vector: false,
        }];
        let named = choose_column("public.q", &lookalike, Some("q"), 8).unwrap_err();
        assert_eq!(named.code, PgSqlErrorCode::ERRCODE_DATATYPE_MISMATCH);
        let default = choose_column("public.q", &lookalike, None, 8).unwrap_err();
        assert_eq!(default.code, PgSqlErrorCode::ERRCODE_UNDEFINED_COLUMN);
    }

    #[test]
    fn a_named_column_must_exist_be_a_vector_and_fit_the_dimension() {
        let columns = [
            other_column("id", "int8"),
            other_column("arr", "_float4"),
            vector_column("embedding", 8),
            vector_column("big", 1536),
            vector_column("loose", -1),
        ];
        let pick = |name: &str| choose_column("public.q", &columns, Some(name), 8);
        assert_eq!(pick("embedding").unwrap().name, "embedding");
        assert_eq!(pick("loose").unwrap().name, "loose");

        let missing = pick("nope").unwrap_err();
        assert_eq!(missing.code, PgSqlErrorCode::ERRCODE_UNDEFINED_COLUMN);
        assert!(missing.message.contains("\"nope\""), "{missing:?}");

        let wrong_type = pick("arr").unwrap_err();
        assert_eq!(wrong_type.code, PgSqlErrorCode::ERRCODE_DATATYPE_MISMATCH);
        assert!(wrong_type.message.contains("_float4"), "{wrong_type:?}");

        let wrong_dim = pick("big").unwrap_err();
        assert_eq!(wrong_dim.code, PgSqlErrorCode::ERRCODE_DATATYPE_MISMATCH);
        assert!(
            wrong_dim.message.contains("vector(1536)") && wrong_dim.message.contains("vector(8)"),
            "{wrong_dim:?}"
        );
    }

    #[test]
    fn a_query_must_have_the_dimension_and_finite_components() {
        assert!(check_vector("public.q", &[0.0, -1.5, 3.0e30], 3).is_ok());

        let short = check_vector("public.q", &[1.0, 2.0], 3).unwrap_err();
        assert_eq!(short.code, PgSqlErrorCode::ERRCODE_DATATYPE_MISMATCH);
        assert!(
            short.message.contains("2 dimensions") && short.message.contains("3"),
            "{short:?}"
        );

        for (value, name) in [
            (f32::NAN, "NaN"),
            (f32::INFINITY, "Infinity"),
            (f32::NEG_INFINITY, "Infinity"),
        ] {
            let bad = check_vector("public.q", &[1.0, value, 2.0], 3).unwrap_err();
            assert_eq!(bad.code, PgSqlErrorCode::ERRCODE_DATA_EXCEPTION);
            assert!(
                bad.message.contains(name) && bad.message.contains("component 2"),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn the_floor_is_the_row_samples() {
        assert_eq!(MIN_QUERIES, 50);
    }
}
