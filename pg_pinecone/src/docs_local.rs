//! `engine => 'local'` of `pinecone.search_docs` and `pinecone.headline`: Postgres full-text
//! search through SPI as the calling role (privileges and RLS apply); creates no index.

use pgrx::{
    PgRelation, PgSqlErrorCode as Code,
    datum::DatumWithOid,
    spi::{Spi, quote_identifier, quote_qualified_identifier},
};

use crate::{
    backend,
    catalog::{self, ColumnCatalog},
    docs_outbox,
    docs_registry::{self, DocRegistryRow},
    docs_schema::{self, TextColumn},
    docs_search::{SearchError, invalid, undeclared_field},
    partition::PartitionSpec,
};

const HEADLINE: &str = "pinecone.headline";
const REGCONFIGS: [(&str, &str); 14] = [
    ("da", "danish"),
    ("de", "german"),
    ("en", "english"),
    ("es", "spanish"),
    ("fi", "finnish"),
    ("fr", "french"),
    ("hu", "hungarian"),
    ("it", "italian"),
    ("nl", "dutch"),
    ("no", "norwegian"),
    ("pt", "portuguese"),
    ("ru", "russian"),
    ("sv", "swedish"),
    ("tr", "turkish"),
];
const SIMPLE: &str = "simple";
const UNDECLARED: &str = "english";

fn regconfig(language: Option<&str>, stemming: bool) -> &'static str {
    let found = language.filter(|_| stemming).and_then(|language| {
        REGCONFIGS
            .iter()
            .find(|(code, _)| code.eq_ignore_ascii_case(language))
    });
    found.map_or(SIMPLE, |(_, config)| config)
}

fn config_literal(config: &str) -> String {
    format!("'pg_catalog.{config}'::pg_catalog.regconfig")
}

struct Group {
    config: &'static str,
    quoted_columns: Vec<String>,
}

pub(crate) struct Target {
    quoted_table: String,
    quoted_id: String,
    id_expr: String,
    groups: Vec<Group>,
    scope: Option<Scope>,
}

/// The partition a search is limited to: `AND <column> = <value>`, the equality the vector
/// lane's local `topk` adds. The value is the canonical text of the column's type, passed as `$3`.
pub(crate) struct Scope {
    quoted_column: String,
    type_name: &'static str,
    value: String,
}

impl Scope {
    pub(crate) fn new(spec: &PartitionSpec, value: String) -> Self {
        Self {
            quoted_column: quote_identifier(&spec.column),
            type_name: spec.kind.as_str(),
            value,
        }
    }
}

impl Target {
    pub(crate) fn scoped(mut self, scope: Option<Scope>) -> Self {
        self.scope = scope;
        self
    }

    /// One match per configuration, OR-ed, their ranks added. With one it is `to_tsvector(cfg, c1)
    /// || ... @@ websearch_to_tsquery(cfg, $1)`, which a GIN index on that expression serves.
    fn sql(&self) -> String {
        let vector = |group: &Group| {
            let config = config_literal(group.config);
            let parts = group.quoted_columns.iter().map(|column| {
                format!("pg_catalog.to_tsvector({config}, coalesce(src.{column}, ''))")
            });
            parts.collect::<Vec<_>>().join(" OPERATOR(pg_catalog.||) ")
        };
        let queries = self.groups.iter().enumerate().map(|(at, group)| {
            let config = config_literal(group.config);
            format!("pg_catalog.websearch_to_tsquery({config}, $1) AS q{at}(q)")
        });
        let ranks = (self.groups.iter().enumerate())
            .map(|(at, group)| format!("pg_catalog.ts_rank_cd({}, q{at}.q)", vector(group)));
        let matches = (self.groups.iter().enumerate())
            .map(|(at, group)| format!("{} OPERATOR(pg_catalog.@@) q{at}.q", vector(group)));
        let matches = matches.collect::<Vec<_>>().join(" OR ");
        let condition = match &self.scope {
            Some(scope) => format!(
                "({matches}) AND src.{} = $3::pg_catalog.{}",
                scope.quoted_column, scope.type_name
            ),
            None => matches,
        };
        format!(
            "SELECT {} AS id, {} AS score FROM {} AS src, {} WHERE {condition} \
             ORDER BY score DESC, src.{} LIMIT $2",
            self.id_expr,
            ranks.collect::<Vec<_>>().join(" + "),
            self.quoted_table,
            queries.collect::<Vec<_>>().join(", "),
            self.quoted_id,
        )
    }
}

/// The rendering of the documents' `_id` (`docs_drain`), so ids join across engines.
fn id_expr(quoted_id: &str, id_type: &str) -> String {
    match id_type {
        "bpchar" => format!("pg_catalog.rtrim(src.{quoted_id}::text)"),
        _ => format!("src.{quoted_id}::text"),
    }
}

/// Name order keeps the searched expression predictable, so an index can be written for it.
fn declared(row: &DocRegistryRow) -> Result<Vec<TextColumn>, String> {
    let mut columns = docs_schema::parse_text_columns(&row.text_columns).map_err(|error| {
        format!(
            "the registered declaration is unreadable: {}",
            error.message
        )
    })?;
    columns.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(columns)
}

fn select(
    display: &str,
    mut columns: Vec<TextColumn>,
    field: Option<&str>,
) -> Result<Vec<TextColumn>, SearchError> {
    let Some(field) = field else {
        return Ok(columns);
    };
    match columns.iter().position(|column| column.name == field) {
        Some(at) => Ok(vec![columns.swap_remove(at)]),
        None => {
            let names: Vec<String> = columns.into_iter().map(|column| column.name).collect();
            Err(undeclared_field(display, field, &names))
        }
    }
}

fn grouped(columns: &[TextColumn]) -> Vec<(&'static str, Vec<&str>)> {
    let mut groups: Vec<(&'static str, Vec<&str>)> = Vec::new();
    for column in columns {
        let config = regconfig(column.language.as_deref(), column.stemming);
        match groups.iter_mut().find(|(known, _)| *known == config) {
            Some((_, names)) => names.push(&column.name),
            None => groups.push((config, vec![&column.name])),
        }
    }
    groups
}

fn tracked(
    display: &str,
    quoted_table: String,
    row: &DocRegistryRow,
    field: Option<&str>,
) -> Result<Target, SearchError> {
    let columns =
        declared(row).map_err(|message| SearchError::new(Code::ERRCODE_DATA_CORRUPTED, message))?;
    let columns = select(display, columns, field)?;
    let groups = grouped(&columns).into_iter().map(|(config, names)| Group {
        config,
        quoted_columns: names.into_iter().map(quote_identifier).collect(),
    });
    let quoted_id = quote_identifier(&row.id_column);
    Ok(Target {
        quoted_table,
        id_expr: id_expr(&quoted_id, &row.id_type),
        quoted_id,
        groups: groups.collect(),
        scope: None,
    })
}

fn field_required(display: &str) -> SearchError {
    invalid(format!(
        "field => is required for the local engine on {display}, which is not tracked — name \
         the text column to search"
    ))
    .hint(
        "pass field => '<column>' (searched as english), or track_docs the table to declare its \
         columns and languages",
    )
}

fn check_untracked_field(
    display: &str,
    field: &str,
    column_type: Option<&str>,
) -> Result<(), SearchError> {
    match column_type {
        None => Err(SearchError::new(
            Code::ERRCODE_UNDEFINED_COLUMN,
            format!("column \"{field}\" does not exist in {display}"),
        )),
        Some("text" | "varchar" | "bpchar") => Ok(()),
        Some(other) => Err(SearchError::new(
            Code::ERRCODE_DATATYPE_MISMATCH,
            format!(
                "column \"{field}\" of {display} has type {other} — the local engine searches \
                 text, varchar or char columns"
            ),
        )),
    }
}

fn untracked_id(display: &str, keys: &[ColumnCatalog]) -> Result<(String, String), SearchError> {
    match keys {
        [] => Err(SearchError::new(
            Code::ERRCODE_INVALID_TABLE_DEFINITION,
            format!("{display} has no primary key — its values are the ids of the result"),
        )
        .hint("add a single-column primary key")),
        [key] if catalog::DOC_ID_COLUMN_TYPES.contains(&key.type_name.as_str()) => {
            Ok((key.name.clone(), key.type_name.clone()))
        }
        [key] => Err(SearchError::new(
            Code::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "id column \"{}\" of {display} has type {} — supported: smallint, integer, \
                 bigint, uuid, text, varchar, char",
                key.name, key.type_name
            ),
        )),
        keys => Err(SearchError::new(
            Code::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "the primary key of {display} has {} columns — this release supports \
                 single-column keys",
                keys.len()
            ),
        )),
    }
}

fn untracked(
    tbl: &PgRelation,
    display: &str,
    quoted_table: String,
    field: Option<&str>,
) -> Result<Target, SearchError> {
    let field = field.ok_or_else(|| field_required(display))?;
    let relid = tbl.oid();
    check_untracked_field(
        display,
        field,
        catalog::column_type(relid, field).as_deref(),
    )?;
    let (id, id_type) = untracked_id(display, &catalog::primary_key_columns(relid))?;
    let quoted_id = quote_identifier(id);
    Ok(Target {
        quoted_table,
        id_expr: id_expr(&quoted_id, &id_type),
        quoted_id,
        groups: vec![Group {
            config: UNDECLARED,
            quoted_columns: vec![quote_identifier(field)],
        }],
        scope: None,
    })
}

pub(crate) fn resolve(
    tbl: &PgRelation,
    row: Option<&DocRegistryRow>,
    field: Option<&str>,
) -> Result<Target, SearchError> {
    let display = catalog::display_name(tbl);
    let quoted_table = quote_qualified_identifier(tbl.namespace(), tbl.name());
    match row {
        Some(row) => tracked(&display, quoted_table, row, field),
        None => untracked(tbl, &display, quoted_table, field),
    }
}

pub(crate) fn search(target: &Target, query: &str, k: u32) -> Vec<(String, f32, i32)> {
    let mut args: Vec<DatumWithOid> = vec![query.into(), i64::from(k).into()];
    if let Some(scope) = &target.scope {
        args.push(scope.value.as_str().into());
    }
    let found: Vec<(String, f32)> = Spi::connect(|client| {
        client
            .select(&target.sql(), None, &args)
            .expect("pg_pinecone: local search failed")
            .map(|row| {
                let id = row
                    .get_by_name::<String, _>("id")
                    .expect("pg_pinecone: local id read failed")
                    .expect("pg_pinecone: local id unexpectedly NULL");
                let score = row
                    .get_by_name::<f32, _>("score")
                    .expect("pg_pinecone: local score read failed")
                    .expect("pg_pinecone: local score unexpectedly NULL");
                (id, score)
            })
            .collect()
    });
    found
        .into_iter()
        .zip(1..)
        .map(|((id, score), rank)| (id, score, rank))
        .collect()
}

fn headline_error(code: Code, message: String) -> ! {
    backend::error_with_code(code, format!("{HEADLINE}: {message}"), None, None)
}

fn headline_config(tbl: &PgRelation, column: &str) -> &'static str {
    let relid = tbl.oid();
    // Reading the declaration is a reader's right: SELECT on the table, nothing on the registry.
    docs_outbox::require_select(relid, tbl.name());
    if let Some(row) = docs_registry::as_owner(|| docs_registry::lookup(relid)) {
        let columns = declared(&row)
            .unwrap_or_else(|message| headline_error(Code::ERRCODE_DATA_CORRUPTED, message));
        if let Some(found) = columns.iter().find(|candidate| candidate.name == column) {
            return regconfig(found.language.as_deref(), found.stemming);
        }
    }
    if catalog::column_type(relid, column).is_none() {
        let message = format!(
            "column \"{column}\" does not exist in {}",
            catalog::display_name(tbl)
        );
        headline_error(Code::ERRCODE_UNDEFINED_COLUMN, message);
    }
    UNDECLARED
}

pub(crate) fn headline(
    tbl: &PgRelation,
    column: &str,
    document: &str,
    query: &str,
    options: &str,
) -> String {
    let config = config_literal(headline_config(tbl, column));
    let sql = format!(
        "SELECT pg_catalog.ts_headline({config}, $1, \
         pg_catalog.websearch_to_tsquery({config}, $2), $3)"
    );
    let args = [document.into(), query.into(), options.into()];
    Spi::get_one_with_args::<String>(&sql, &args)
        .expect("pg_pinecone: headline query failed")
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn columns(declared: &str) -> Vec<TextColumn> {
        let json: serde_json::Value = serde_json::from_str(declared).expect("json");
        docs_schema::parse_text_columns(&json).expect("a valid declaration")
    }

    fn example() -> Vec<TextColumn> {
        columns(
            r#"{"title": {"language": "en"}, "body": {},
                "sku": {"stemming": false, "stop_words": false}}"#,
        )
    }

    fn named(columns: &[TextColumn]) -> Vec<String> {
        columns.iter().map(|column| column.name.clone()).collect()
    }

    #[test]
    fn every_language_postgres_stems_maps_to_its_configuration() {
        let table = [
            ("en", "english"),
            ("de", "german"),
            ("fr", "french"),
            ("es", "spanish"),
            ("it", "italian"),
            ("pt", "portuguese"),
            ("nl", "dutch"),
            ("ru", "russian"),
            ("sv", "swedish"),
            ("no", "norwegian"),
            ("da", "danish"),
            ("fi", "finnish"),
            ("hu", "hungarian"),
            ("tr", "turkish"),
        ];
        assert_eq!(table.len(), REGCONFIGS.len());
        for (language, config) in table {
            assert_eq!(regconfig(Some(language), true), config, "{language}");
        }
    }

    #[test]
    fn a_column_without_stemming_or_a_known_language_is_matched_as_written() {
        for language in ["en", "de", "xx", "ar"] {
            assert_eq!(regconfig(Some(language), false), "simple", "{language}");
        }
        assert_eq!(regconfig(Some("xx"), true), "simple");
        assert_eq!(regconfig(Some("ar"), true), "simple");
        assert_eq!(regconfig(None, true), "simple");
        assert_eq!(regconfig(None, false), "simple");
        assert_eq!(regconfig(Some("DE"), true), "german");
    }

    #[test]
    fn declared_columns_map_through_their_resolved_options() {
        let columns = columns(
            r#"{"b": {"language": "de"}, "c": {"language": "German", "stemming": false,
                "stop_words": false}, "n": {"ngram": {"min_gram": 2, "max_gram": 3}}}"#,
        );
        let config = |name: &str| {
            let column = columns.iter().find(|c| c.name == name).expect("column");
            regconfig(column.language.as_deref(), column.stemming)
        };
        assert_eq!(config("b"), "german");
        assert_eq!(config("c"), "simple", "stemming off");
        assert_eq!(config("n"), "simple", "ngram has no stemming");
    }

    #[test]
    fn columns_are_grouped_by_configuration() {
        let mut columns = example();
        columns.sort_by(|a, b| a.name.cmp(&b.name));
        assert_eq!(
            grouped(&columns),
            [("english", vec!["body", "title"]), ("simple", vec!["sku"])]
        );
        let german = self::columns(r#"{"a": {"language": "de"}, "b": {"language": "en"}}"#);
        assert_eq!(
            grouped(&german),
            [("german", vec!["a"]), ("english", vec!["b"])]
        );
    }

    #[test]
    fn field_narrows_to_one_declared_column_or_names_the_declared_ones() {
        assert_eq!(named(&select("t", example(), None).expect("all")).len(), 3);
        let one = select("public.articles", example(), Some("sku")).expect("one");
        assert_eq!(named(&one), ["sku"]);
        let error = select("public.articles", example(), Some("note")).expect_err("undeclared");
        assert_eq!(error.code, Code::ERRCODE_UNDEFINED_COLUMN);
        assert!(error.message.contains("note"), "{error:?}");
        let hint = error.hint.expect("a hint");
        for column in ["body", "sku", "title"] {
            assert!(hint.contains(column), "{hint}");
        }
    }

    fn target(groups: Vec<Group>) -> Target {
        Target {
            quoted_table: "\"public\".\"articles\"".to_string(),
            quoted_id: "\"id\"".to_string(),
            id_expr: id_expr("\"id\"", "int8"),
            groups,
            scope: None,
        }
    }

    fn group(config: &'static str, columns: &[&str]) -> Group {
        Group {
            config,
            quoted_columns: columns.iter().map(|c| format!("\"{c}\"")).collect(),
        }
    }

    #[test]
    fn one_configuration_is_the_plain_websearch_ts_rank_cd_statement() {
        let sql = target(vec![group("english", &["body", "title"])]).sql();
        let english = "'pg_catalog.english'::pg_catalog.regconfig";
        let vector = format!(
            "pg_catalog.to_tsvector({english}, coalesce(src.\"body\", '')) \
             OPERATOR(pg_catalog.||) \
             pg_catalog.to_tsvector({english}, coalesce(src.\"title\", ''))"
        );
        assert_eq!(
            sql,
            format!(
                "SELECT src.\"id\"::text AS id, pg_catalog.ts_rank_cd({vector}, q0.q) AS score \
                 FROM \"public\".\"articles\" AS src, \
                 pg_catalog.websearch_to_tsquery({english}, $1) AS q0(q) \
                 WHERE {vector} OPERATOR(pg_catalog.@@) q0.q \
                 ORDER BY score DESC, src.\"id\" LIMIT $2"
            )
        );
    }

    #[test]
    fn a_partition_ands_a_typed_equality_after_the_whole_match() {
        let groups = vec![group("english", &["body"]), group("simple", &["sku"])];
        let plain = target(groups).sql();
        let groups = vec![group("english", &["body"]), group("simple", &["sku"])];
        let mut scoped = target(groups);
        scoped.scope = Some(Scope {
            quoted_column: "\"org\"".to_string(),
            type_name: "int8",
            value: "7".to_string(),
        });
        let (matched, rest) = plain.split_once(" WHERE ").expect("a WHERE clause");
        let (condition, order) = rest.split_once(" ORDER BY ").expect("an ORDER BY");
        assert_eq!(
            scoped.sql(),
            format!(
                "{matched} WHERE ({condition}) AND src.\"org\" = $3::pg_catalog.int8 \
                 ORDER BY {order}"
            ),
            "the alternatives are one condition, the partition another"
        );
    }

    #[test]
    fn several_configurations_match_alternatively_and_their_ranks_add_up() {
        let groups = vec![group("english", &["body"]), group("simple", &["sku"])];
        let sql = target(groups).sql();
        let simple = "'pg_catalog.simple'::pg_catalog.regconfig";
        assert!(sql.contains(&format!("({simple}, $1) AS q1(q)")), "{sql}");
        assert!(sql.contains("q0.q) + pg_catalog.ts_rank_cd("), "{sql}");
        assert!(sql.contains("OPERATOR(pg_catalog.@@) q0.q OR "), "{sql}");
        assert!(
            sql.ends_with("q1.q ORDER BY score DESC, src.\"id\" LIMIT $2"),
            "{sql}"
        );
    }

    #[test]
    fn a_padded_char_id_renders_without_its_padding() {
        let quoted = "\"id\"";
        assert_eq!(
            id_expr(quoted, "bpchar"),
            "pg_catalog.rtrim(src.\"id\"::text)"
        );
        for id_type in ["int2", "int4", "int8", "uuid", "text", "varchar"] {
            assert_eq!(id_expr(quoted, id_type), "src.\"id\"::text", "{id_type}");
        }
    }

    fn key(name: &str, type_name: &str) -> ColumnCatalog {
        ColumnCatalog {
            name: name.to_string(),
            attnum: 1,
            type_name: type_name.to_string(),
        }
    }

    #[test]
    fn an_untracked_table_takes_its_single_supported_primary_key() {
        for type_name in catalog::DOC_ID_COLUMN_TYPES {
            let found = untracked_id("t", &[key("pk", type_name)]).expect("a usable key");
            assert_eq!(found, ("pk".to_string(), (*type_name).to_string()));
        }
    }

    #[test]
    fn an_untracked_table_without_a_usable_key_is_refused() {
        let none = untracked_id("public.t", &[]).expect_err("no key");
        assert_eq!(none.code, Code::ERRCODE_INVALID_TABLE_DEFINITION);
        assert!(
            none.message.contains("public.t has no primary key"),
            "{none:?}"
        );
        assert!(none.hint.is_some());

        let two = [key("a", "int4"), key("b", "int4")];
        let composite = untracked_id("public.t", &two).expect_err("composite");
        assert_eq!(composite.code, Code::ERRCODE_FEATURE_NOT_SUPPORTED);
        assert!(composite.message.contains("2 columns"), "{composite:?}");

        let typed = untracked_id("public.t", &[key("ts", "timestamptz")]).expect_err("type");
        assert_eq!(typed.code, Code::ERRCODE_FEATURE_NOT_SUPPORTED);
        assert!(typed.message.contains("timestamptz"), "{typed:?}");
        assert!(typed.message.contains("public.t"), "{typed:?}");
    }

    #[test]
    fn an_untracked_field_must_be_a_text_column_of_the_table() {
        for type_name in ["text", "varchar", "bpchar"] {
            assert!(check_untracked_field("t", "body", Some(type_name)).is_ok());
        }
        let missing = check_untracked_field("public.t", "bod", None).expect_err("missing");
        assert_eq!(missing.code, Code::ERRCODE_UNDEFINED_COLUMN);
        assert!(
            missing
                .message
                .contains("\"bod\" does not exist in public.t")
        );
        let json = check_untracked_field("public.t", "meta", Some("jsonb")).expect_err("jsonb");
        assert_eq!(json.code, Code::ERRCODE_DATATYPE_MISMATCH);
        assert!(json.message.contains("jsonb"), "{json:?}");
    }

    #[test]
    fn field_is_required_on_an_untracked_table() {
        let error = field_required("public.t");
        assert_eq!(error.code, Code::ERRCODE_INVALID_PARAMETER_VALUE);
        assert!(error.message.contains("not tracked"), "{error:?}");
        assert!(error.hint.is_some_and(|hint| hint.contains("track_docs")));
    }
}
