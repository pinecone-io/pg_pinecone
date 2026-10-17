//! Pure document-schema logic behind `track_docs`: declaration parsing, schema
//! derivation, attach comparison, size policy and metadata typing. No SPI, so
//! it is unit-tested without Postgres.

use pgrx::PgSqlErrorCode as Code;
use serde_json::{Map, Value, json};

use crate::backend;

pub(crate) const MAX_FIELD_NAME_BYTES: usize = 64;
pub(crate) const MAX_TEXT_FIELDS: usize = 100;
pub(crate) const MAX_DIMENSION: u32 = 20_000;
pub(crate) const DEFAULT_LANGUAGE: &str = "en";
/// Prefix of the fields pg_pinecone adds to a document; columns may not use it.
pub(crate) const RESERVED_PREFIX: &str = "pg_pinecone_";
/// Set to `true` on a document whose text [`truncate_text`] cut.
pub(crate) const TRUNCATED_FIELD: &str = "pg_pinecone_truncated";
/// Pinecone allows 100 KB and 10,000 tokens (256 bytes each) per full-text field. Tokens are
/// estimated by whitespace, hence 10% headroom.
pub(crate) const MAX_FIELD_TOKENS: usize = 9_000;
pub(crate) const MAX_FIELD_BYTES: usize = 100_000;
pub(crate) const MAX_TOKEN_BYTES: usize = 256;

const PREFIX: &str = "pinecone.track_docs: ";
const OPTION_KEYS: [&str; 5] = ["language", "stemming", "stop_words", "ngram", "on_oversize"];
const SUPPORTED_METADATA_TYPES: &str = "smallint, integer, bigint, real, double precision, \
     numeric, boolean, text, varchar, char, uuid, date, timestamp, timestamptz, text[]";
const LANGUAGES: [(&str, &str); 18] = [
    ("ar", "arabic"),
    ("da", "danish"),
    ("de", "german"),
    ("el", "greek"),
    ("en", "english"),
    ("es", "spanish"),
    ("fi", "finnish"),
    ("fr", "french"),
    ("hu", "hungarian"),
    ("it", "italian"),
    ("nl", "dutch"),
    ("no", "norwegian"),
    ("pt", "portuguese"),
    ("ro", "romanian"),
    ("ru", "russian"),
    ("sv", "swedish"),
    ("ta", "tamil"),
    ("tr", "turkish"),
];

/// A declaration problem: SQLSTATE, message and HINT, ready for [`backend::error_with_code`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DocsError {
    pub(crate) code: Code,
    pub(crate) message: String,
    pub(crate) hint: Option<String>,
}

impl DocsError {
    pub(crate) fn new(code: Code, message: impl Into<String>) -> Self {
        Self {
            code,
            message: format!("{PREFIX}{}", message.into()),
            hint: None,
        }
    }

    pub(crate) fn hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    pub(crate) fn raise(self) -> ! {
        backend::error_with_code(self.code, self.message, None, self.hint.as_deref())
    }
}

fn column_error(name: &str, message: impl std::fmt::Display) -> DocsError {
    DocsError::new(
        Code::ERRCODE_INVALID_PARAMETER_VALUE,
        format!("column \"{name}\": {message}"),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OnOversize {
    Truncate,
    Error,
}

impl OnOversize {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            OnOversize::Truncate => "truncate",
            OnOversize::Error => "error",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ngram {
    pub(crate) min_gram: u32,
    pub(crate) max_gram: u32,
    pub(crate) prefix_only: bool,
}

/// One declared full-text column with every default resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TextColumn {
    pub(crate) name: String,
    /// Always set for word-tokenised columns; an ngram column has one only if it declared it.
    pub(crate) language: Option<String>,
    pub(crate) stemming: bool,
    pub(crate) stop_words: bool,
    pub(crate) ngram: Option<Ngram>,
    pub(crate) on_oversize: OnOversize,
}

/// The one dense field; `metric` is pg_pinecone' name (`cosine`, `l2`, `ip`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VectorField {
    pub(crate) column: String,
    pub(crate) dimension: u32,
    pub(crate) metric: String,
}

/// Pinecone's field-name rules plus the reserved prefix; `role` words the error message.
pub(crate) fn validate_field_name(role: &str, name: &str) -> Result<(), DocsError> {
    if name.starts_with(RESERVED_PREFIX) {
        return Err(DocsError::new(
            Code::ERRCODE_INVALID_NAME,
            format!("{role} \"{name}\" collides with a reserved field name"),
        )
        .hint(format!(
            "field names starting with \"{RESERVED_PREFIX}\" are reserved for pg_pinecone"
        )));
    }
    if name.is_empty() || name.starts_with(['_', '$']) || name.len() > MAX_FIELD_NAME_BYTES {
        return Err(DocsError::new(
            Code::ERRCODE_INVALID_NAME,
            format!(
                "{role} \"{name}\" is not a valid Pinecone field name — names may not start \
                 with \"_\" or \"$\" and are at most {MAX_FIELD_NAME_BYTES} bytes"
            ),
        )
        .hint("expose it under a valid name, e.g. through a generated column"));
    }
    Ok(())
}

/// `["title", "body"]` is shorthand for `{"title": {}, "body": {}}`: defaults for every column.
fn columns_from_names(names: &[Value]) -> Result<Value, DocsError> {
    let mut columns = Map::new();
    for name in names {
        let Some(name) = name.as_str() else {
            return Err(DocsError::new(
                Code::ERRCODE_INVALID_PARAMETER_VALUE,
                "text_columns as a list must hold column names (strings) only",
            )
            .hint(r#"use the object form for options: text_columns => '{"body": {"language": "de"}}'"#));
        };
        if columns.insert(name.to_string(), json!({})).is_some() {
            return Err(DocsError::new(
                Code::ERRCODE_INVALID_PARAMETER_VALUE,
                format!("column \"{name}\" is listed twice in text_columns"),
            ));
        }
    }
    Ok(Value::Object(columns))
}

/// Parses `{"<column>": {<options>}, ...}` (or a list of column names, meaning defaults) with
/// every default resolved, sorted by column name.
pub(crate) fn parse_text_columns(declared: &Value) -> Result<Vec<TextColumn>, DocsError> {
    let listed;
    let declared = match declared {
        Value::Array(names) => {
            listed = columns_from_names(names)?;
            &listed
        }
        other => other,
    };
    let Some(map) = declared.as_object().filter(|map| !map.is_empty()) else {
        return Err(DocsError::new(
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            "text_columns must be a non-empty JSON object mapping column names to options, or a list of column names",
        )
        .hint(r#"e.g. text_columns => '{"title": {"language": "en"}, "body": {}}'"#));
    };
    if map.len() > MAX_TEXT_FIELDS {
        return Err(DocsError::new(
            Code::ERRCODE_PROGRAM_LIMIT_EXCEEDED,
            format!(
                "text_columns declares {} columns — Pinecone allows at most {MAX_TEXT_FIELDS} \
                 full-text fields per index",
                map.len()
            ),
        ));
    }
    let mut columns = map
        .iter()
        .map(|(name, spec)| parse_column(name, spec))
        .collect::<Result<Vec<_>, _>>()?;
    columns.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(columns)
}

fn present<'a>(options: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    options.get(key).filter(|value| !value.is_null())
}

fn flag(name: &str, options: &Map<String, Value>, key: &str) -> Result<Option<bool>, DocsError> {
    match present(options, key) {
        None => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(column_error(name, format!("{key} must be true or false"))),
    }
}

fn parse_column(name: &str, spec: &Value) -> Result<TextColumn, DocsError> {
    validate_field_name("column", name)?;
    let no_options = Map::new();
    let options = match spec {
        Value::Null => &no_options,
        Value::Object(map) => map,
        _ => {
            return Err(column_error(
                name,
                r#"options must be a JSON object, e.g. {"language": "en"}"#,
            ));
        }
    };
    if let Some(key) = options.keys().find(|k| !OPTION_KEYS.contains(&k.as_str())) {
        return Err(column_error(name, format!("unknown option \"{key}\""))
            .hint(format!("allowed: {}", OPTION_KEYS.join(", "))));
    }
    let language = match present(options, "language") {
        None => None,
        Some(Value::String(raw)) => Some(parse_language(name, raw)?),
        Some(_) => return Err(column_error(name, "language must be a string, e.g. \"en\"")),
    };
    let on_oversize = match present(options, "on_oversize").map(Value::as_str) {
        None => OnOversize::Truncate,
        Some(Some("truncate")) => OnOversize::Truncate,
        Some(Some("error")) => OnOversize::Error,
        Some(_) => {
            return Err(column_error(
                name,
                "on_oversize must be 'truncate' or 'error'",
            ));
        }
    };
    let stemming = flag(name, options, "stemming")?;
    let stop_words = flag(name, options, "stop_words")?;
    let ngram = present(options, "ngram")
        .map(|value| parse_ngram(name, value))
        .transpose()?;
    resolve_analyzer(name, language, stemming, stop_words, ngram, on_oversize)
}

/// Maps a language name to Pinecone's short code; unknown ones pass through for Pinecone to
/// judge, so a newly supported language needs no release here.
fn parse_language(name: &str, raw: &str) -> Result<String, DocsError> {
    let plausible = raw.len() <= 32
        && raw
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if raw.is_empty() {
        return Err(column_error(name, "language must not be empty"));
    }
    if !plausible {
        return Err(column_error(
            name,
            format!("language \"{raw}\" is not a language code or name"),
        ));
    }
    Ok(normalise_language(raw))
}

fn parse_ngram(name: &str, value: &Value) -> Result<Ngram, DocsError> {
    let invalid = |why: &str| {
        column_error(name, format!("ngram {why}"))
            .hint(r#"e.g. "ngram": {"min_gram": 2, "max_gram": 3}"#)
    };
    let Value::Object(map) = value else {
        return Err(invalid("must be an object with min_gram and max_gram"));
    };
    if let Some(key) = map
        .keys()
        .find(|k| !["min_gram", "max_gram", "prefix_only"].contains(&k.as_str()))
    {
        return Err(invalid(&format!("has unknown option \"{key}\"")));
    }
    let gram = |key: &str| {
        map.get(key)
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| *n >= 1)
            .ok_or_else(|| invalid(&format!("{key} must be an integer of at least 1")))
    };
    let (min_gram, max_gram) = (gram("min_gram")?, gram("max_gram")?);
    if min_gram > max_gram {
        return Err(invalid("min_gram must not exceed max_gram"));
    }
    let prefix_only = match map.get("prefix_only") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(invalid("prefix_only must be true or false")),
    };
    Ok(Ngram {
        min_gram,
        max_gram,
        prefix_only,
    })
}

/// Stemming and stop words default on (stop words follow stemming and need it); ngram takes neither.
fn resolve_analyzer(
    name: &str,
    language: Option<String>,
    stemming: Option<bool>,
    stop_words: Option<bool>,
    ngram: Option<Ngram>,
    on_oversize: OnOversize,
) -> Result<TextColumn, DocsError> {
    let (language, stemming, stop_words) = if ngram.is_some() {
        if stemming == Some(true) || stop_words == Some(true) {
            return Err(
                column_error(name, "ngram cannot be combined with stemming or stop_words")
                    .hint("identifier columns: stemming => false, stop_words => false"),
            );
        }
        (language, false, false)
    } else {
        let stemming = stemming.unwrap_or(true);
        let stop_words = stop_words.unwrap_or(stemming);
        if stop_words && !stemming {
            return Err(column_error(name, "stop_words requires stemming")
                .hint("set stop_words => false, or enable stemming"));
        }
        let language = language.unwrap_or_else(|| DEFAULT_LANGUAGE.to_string());
        (Some(language), stemming, stop_words)
    };
    Ok(TextColumn {
        name: name.to_string(),
        language,
        stemming,
        stop_words,
        ngram,
        on_oversize,
    })
}

fn ngram_json(ngram: Ngram) -> Value {
    let mut out = Map::new();
    out.insert("min_gram".to_string(), json!(ngram.min_gram));
    out.insert("max_gram".to_string(), json!(ngram.max_gram));
    if ngram.prefix_only {
        out.insert("prefix_only".to_string(), json!(true));
    }
    Value::Object(out)
}

/// The resolved declaration as stored in `doc_tables.text_columns`; [`parse_text_columns`] reads it back.
pub(crate) fn text_columns_json(columns: &[TextColumn]) -> Value {
    let mut out = Map::new();
    for column in columns {
        let mut options = Map::new();
        if let Some(language) = &column.language {
            options.insert("language".to_string(), json!(language));
        }
        options.insert("stemming".to_string(), json!(column.stemming));
        options.insert("stop_words".to_string(), json!(column.stop_words));
        if let Some(ngram) = column.ngram {
            options.insert("ngram".to_string(), ngram_json(ngram));
        }
        options.insert(
            "on_oversize".to_string(),
            json!(column.on_oversize.as_str()),
        );
        out.insert(column.name.clone(), Value::Object(options));
    }
    Value::Object(out)
}

fn pinecone_metric(metric: &str) -> Result<&'static str, DocsError> {
    match metric {
        "cosine" => Ok("cosine"),
        "l2" => Ok("euclidean"),
        "ip" => Ok("dotproduct"),
        other => Err(DocsError::new(
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            format!("vector metric \"{other}\" is not supported"),
        )
        .hint("use cosine, l2 or ip")),
    }
}

fn text_field(column: &TextColumn) -> Value {
    let mut config = Map::new();
    if let Some(language) = &column.language {
        config.insert("language".to_string(), json!(language));
    }
    match column.ngram {
        Some(ngram) => {
            config.insert("ngram".to_string(), ngram_json(ngram));
        }
        None => {
            config.insert("stemming".to_string(), json!(column.stemming));
            config.insert("stop_words".to_string(), json!(column.stop_words));
        }
    }
    json!({"type": "string", "full_text_search": Value::Object(config)})
}

/// The Pinecone schema: one full-text field per text column plus the optional dense field;
/// local options such as `on_oversize` stay out.
pub(crate) fn derive_schema(
    text: &[TextColumn],
    vector: Option<&VectorField>,
) -> Result<Value, DocsError> {
    if text.is_empty() || text.len() > MAX_TEXT_FIELDS {
        return Err(DocsError::new(
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            format!("text_columns must declare 1 to {MAX_TEXT_FIELDS} columns"),
        ));
    }
    let mut fields = Map::new();
    for column in text {
        validate_field_name("column", &column.name)?;
        fields.insert(column.name.clone(), text_field(column));
    }
    if let Some(vector) = vector {
        validate_field_name("vector column", &vector.column)?;
        if fields.contains_key(&vector.column) {
            return Err(DocsError::new(
                Code::ERRCODE_INVALID_PARAMETER_VALUE,
                format!(
                    "column \"{}\" is declared as both a text column and the vector column",
                    vector.column
                ),
            ));
        }
        if !(1..=MAX_DIMENSION).contains(&vector.dimension) {
            return Err(DocsError::new(
                Code::ERRCODE_INVALID_PARAMETER_VALUE,
                format!(
                    "vector column \"{}\" has dimension {} — Pinecone supports 1 to {MAX_DIMENSION}",
                    vector.column, vector.dimension
                ),
            ));
        }
        let metric = pinecone_metric(&vector.metric)?;
        fields.insert(
            vector.column.clone(),
            json!({"type": "dense_vector", "dimension": vector.dimension, "metric": metric}),
        );
    }
    Ok(json!({"fields": Value::Object(fields)}))
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SchemaMismatch {
    pub(crate) field: String,
    /// The remote spec, `None` if the field is missing there.
    pub(crate) remote: Option<Value>,
    /// The declared spec, `None` if the remote has a field that was not declared.
    pub(crate) declared: Option<Value>,
}

impl SchemaMismatch {
    /// `field "body" is {...} remotely, {...} declared` (or the missing/undeclared variants).
    pub(crate) fn describe(&self) -> String {
        let field = &self.field;
        match (&self.remote, &self.declared) {
            (Some(remote), Some(declared)) => {
                format!("field \"{field}\" is {remote} remotely, {declared} declared")
            }
            (None, Some(declared)) => {
                format!("field \"{field}\" is missing remotely, {declared} declared")
            }
            (Some(remote), None) => {
                format!("field \"{field}\" exists remotely as {remote} but is not declared")
            }
            (None, None) => format!("field \"{field}\" differs"),
        }
    }
}

/// A full-text analyzer with the server's defaults resolved (language `en`, stemming and stop
/// words off) and everything else dropped: Pinecone describes a field back with defaults filled
/// in and server-applied keys (`lowercase`, `max_term_len`, ...) added.
fn analyzer(config: &Value) -> Value {
    let language = config
        .get("language")
        .and_then(Value::as_str)
        .map_or_else(|| DEFAULT_LANGUAGE.to_string(), normalise_language);
    let flag = |key: &str| config.get(key).and_then(Value::as_bool).unwrap_or(false);
    let mut out = Map::new();
    out.insert("language".to_string(), json!(language));
    out.insert("stemming".to_string(), json!(flag("stemming")));
    out.insert("stop_words".to_string(), json!(flag("stop_words")));
    if let Some(ngram) = config.get("ngram").and_then(Value::as_object) {
        let prefix_only = ngram.get("prefix_only").and_then(Value::as_bool);
        out.insert(
            "ngram".to_string(),
            json!({"min_gram": ngram.get("min_gram"), "max_gram": ngram.get("max_gram"),
                   "prefix_only": prefix_only.unwrap_or(false)}),
        );
    }
    Value::Object(out)
}

fn normalise_language(raw: &str) -> String {
    let lowered = raw.to_ascii_lowercase();
    LANGUAGES
        .iter()
        .find(|(code, long)| lowered == *code || lowered == *long)
        .map_or_else(|| raw.to_string(), |(code, _)| (*code).to_string())
}

/// What decides whether two field specs are the same field.
fn comparable(spec: &Value) -> Value {
    let kind = spec.get("type").and_then(Value::as_str);
    match (kind, spec.get("full_text_search")) {
        (Some("string"), Some(config)) => analyzer(config),
        (Some("dense_vector"), _) => json!({"type": "dense_vector",
            "dimension": spec.get("dimension"), "metric": spec.get("metric")}),
        (Some("sparse_vector"), _) => json!({"type": "sparse_vector"}),
        _ => {
            let mut other = spec.clone();
            if let Some(map) = other.as_object_mut() {
                map.remove("description");
            }
            other
        }
    }
}

/// Fields a schema can declare; the rest of a described schema is response-only metadata
/// that Pinecone indexed from upserted documents.
fn is_searchable(spec: &Value) -> bool {
    match spec.get("type").and_then(Value::as_str) {
        Some("dense_vector" | "sparse_vector") => true,
        Some("string") => spec.get("full_text_search").is_some(),
        _ => false,
    }
}

/// The first difference between a declared schema and an index's described one (declared fields
/// in name order, then undeclared searchable ones), or `None` if it can be attached. Key order,
/// server-resolved defaults, server-applied keys and response-only metadata fields do not count.
pub(crate) fn schema_mismatch(declared: &Value, remote: &Value) -> Option<SchemaMismatch> {
    let none = Map::new();
    let declared_fields = declared
        .get("fields")
        .and_then(Value::as_object)
        .unwrap_or(&none);
    let remote_fields = remote
        .get("fields")
        .and_then(Value::as_object)
        .unwrap_or(&none);
    for (name, spec) in declared_fields {
        let want = comparable(spec);
        let have = remote_fields.get(name).map(comparable);
        if have.as_ref() != Some(&want) {
            return Some(SchemaMismatch {
                field: name.clone(),
                remote: have,
                declared: Some(want),
            });
        }
    }
    remote_fields
        .iter()
        .find(|(name, spec)| !declared_fields.contains_key(*name) && is_searchable(spec))
        .map(|(name, spec)| SchemaMismatch {
            field: name.clone(),
            remote: Some(comparable(spec)),
            declared: None,
        })
}

/// What Pinecone's per-field limits are measured on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TextSize {
    pub(crate) bytes: usize,
    /// The estimated token count, see [`estimate_tokens`].
    pub(crate) tokens: usize,
    /// The longest whitespace-separated run in bytes, an upper bound for the longest token.
    pub(crate) longest_token: usize,
}

/// The runs of alphanumeric characters of `s` with their byte offsets: what a standard analyzer
/// makes tokens of, however many of them a whitespace-separated word holds ("a,b,c" is three).
fn alphanumeric_runs(s: &str) -> impl Iterator<Item = (usize, &str)> {
    let base = s.as_ptr() as usize;
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|run| !run.is_empty())
        .map(move |run| (run.as_ptr() as usize - base, run))
}

pub(crate) fn text_size(s: &str) -> TextSize {
    let (mut tokens, mut longest_token) = (0, 0);
    for word in s.split_whitespace() {
        tokens += alphanumeric_runs(word).count();
        longest_token = longest_token.max(word.len());
    }
    TextSize {
        bytes: s.len(),
        tokens,
        longest_token,
    }
}

/// Whether [`truncate_text`] at the field limits would cut: the `on_oversize => 'error'`
/// policy refuses exactly these values.
pub(crate) fn exceeds_field_limits(size: &TextSize) -> bool {
    size.bytes > MAX_FIELD_BYTES
        || size.tokens > MAX_FIELD_TOKENS
        || size.longest_token > MAX_TOKEN_BYTES
}

/// What the `error` policy says about a value over the limits, naming the limit it breaks: a
/// value that is only one long word must not be blamed on its size.
pub(crate) fn oversize_message(column: &str, size: &TextSize) -> String {
    if size.bytes > MAX_FIELD_BYTES || size.tokens > MAX_FIELD_TOKENS {
        format!(
            "column \"{column}\" is {} bytes (~{} tokens) — Pinecone allows 100 KB and 10000 \
             tokens per field",
            size.bytes, size.tokens
        )
    } else {
        format!(
            "column \"{column}\" contains a {}-byte token — Pinecone allows {MAX_TOKEN_BYTES} \
             bytes per token",
            size.longest_token
        )
    }
}

/// Estimated token count: runs of alphanumeric characters, which tracks Pinecone's analyzer on
/// punctuation-dense text where counting words would not; [`MAX_FIELD_TOKENS`] leaves headroom.
pub(crate) fn estimate_tokens(s: &str) -> usize {
    alphanumeric_runs(s).count()
}

fn clip_token(token: &str) -> &str {
    if token.len() <= MAX_TOKEN_BYTES {
        return token;
    }
    let mut end = MAX_TOKEN_BYTES;
    while !token.is_char_boundary(end) {
        end -= 1;
    }
    &token[..end]
}

/// The longest prefix of `word` that holds at most `max_runs` alphanumeric runs and `max_bytes`
/// bytes and ends where a run ends; all of `word` if it fits.
fn prefix_within(word: &str, max_runs: usize, max_bytes: usize) -> &str {
    if word.len() <= max_bytes && alphanumeric_runs(word).count() <= max_runs {
        return word;
    }
    let mut end = 0;
    for (kept, (start, run)) in alphanumeric_runs(word).enumerate() {
        if kept == max_runs || start + run.len() > max_bytes {
            break;
        }
        end = start + run.len();
    }
    &word[..end]
}

/// Cuts `s` to at most `max_tokens` alphanumeric runs and `max_bytes` bytes, at the end of a
/// run, clipping any word over [`MAX_TOKEN_BYTES`] on a character boundary; reports whether
/// anything was cut. Apart from clipped words the result is a prefix of `s`.
pub(crate) fn truncate_text(s: &str, max_tokens: usize, max_bytes: usize) -> (String, bool) {
    let size = text_size(s);
    if size.bytes <= max_bytes && size.tokens <= max_tokens && size.longest_token <= MAX_TOKEN_BYTES
    {
        return (s.to_string(), false);
    }
    let base = s.as_ptr() as usize;
    let (mut out, mut cursor, mut tokens) = (String::new(), 0, 0);
    for word in s.split_whitespace() {
        if tokens >= max_tokens {
            break;
        }
        let start = word.as_ptr() as usize - base;
        let separator = &s[cursor..start];
        let room = max_bytes.saturating_sub(out.len() + separator.len());
        let piece = prefix_within(clip_token(word), max_tokens - tokens, room);
        if piece.is_empty() {
            break;
        }
        out.push_str(separator);
        out.push_str(piece);
        tokens += alphanumeric_runs(piece).count();
        cursor = start + word.len();
    }
    (out, true)
}

/// How a metadata column's value reaches Pinecone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetaKind {
    Int,
    Float,
    Numeric,
    Bool,
    Text,
    Uuid,
    TextArray,
    Date,
    Timestamp,
    Timestamptz,
}

/// The kind for a `pg_type.typname`, or `None` if the type has no Pinecone counterpart
/// (`jsonb`, composites, arrays other than `text[]`, ...).
pub(crate) fn metadata_kind(type_name: &str) -> Option<MetaKind> {
    Some(match type_name {
        "int2" | "int4" | "int8" => MetaKind::Int,
        "float4" | "float8" => MetaKind::Float,
        "numeric" => MetaKind::Numeric,
        "bool" => MetaKind::Bool,
        "text" | "varchar" | "bpchar" => MetaKind::Text,
        "uuid" => MetaKind::Uuid,
        "_text" => MetaKind::TextArray,
        "date" => MetaKind::Date,
        "timestamp" => MetaKind::Timestamp,
        "timestamptz" => MetaKind::Timestamptz,
        _ => return None,
    })
}

impl MetaKind {
    /// The JSON type the document carries; dates and timestamps are epoch seconds.
    #[allow(dead_code)] // the search filters (a later change) type their values with it
    pub(crate) fn json_type(self) -> &'static str {
        match self {
            MetaKind::Bool => "boolean",
            MetaKind::Text | MetaKind::Uuid => "string",
            MetaKind::TextArray => "string[]",
            _ => "number",
        }
    }

    /// SQL turning `column` (already quoted) into the document value. Epoch seconds ignore the
    /// session time zone (`timestamp` and `date` count as UTC), infinities become NULL, and
    /// `::text` strips `char(n)` padding.
    pub(crate) fn value_expr(self, column: &str) -> String {
        match self {
            MetaKind::Int | MetaKind::Float | MetaKind::Bool | MetaKind::TextArray => {
                column.to_string()
            }
            MetaKind::Text | MetaKind::Uuid => format!("{column}::text"),
            MetaKind::Numeric => format!("{column}::float8"),
            MetaKind::Date => epoch(column, &format!("{column}::timestamp")),
            MetaKind::Timestamp | MetaKind::Timestamptz => epoch(column, column),
        }
    }
}

fn epoch(column: &str, instant: &str) -> String {
    format!(
        "CASE WHEN pg_catalog.isfinite({column}) THEN pg_catalog.date_part('epoch', {instant}) END"
    )
}

/// Validates a metadata column's name and type; the error names the column.
pub(crate) fn check_metadata_column(name: &str, type_name: &str) -> Result<MetaKind, DocsError> {
    validate_field_name("metadata column", name)?;
    metadata_kind(type_name).ok_or_else(|| {
        let shown = type_name
            .strip_prefix('_')
            .map_or_else(|| type_name.to_string(), |inner| format!("{inner}[]"));
        DocsError::new(
            Code::ERRCODE_FEATURE_NOT_SUPPORTED,
            format!(
                "metadata column \"{name}\" has type {shown} — supported: {SUPPORTED_METADATA_TYPES}"
            ),
        )
        .hint("list a generated column of a supported type instead")
    })
}

#[cfg(test)]
mod tests {
    use pgrx::PgSqlErrorCode as Code;
    use serde_json::{Value, json};

    use super::*;

    const EXAMPLE: &str = r#"{"title": {"language": "en"},
        "body": {"language": "en", "stemming": true, "on_oversize": "truncate"},
        "sku": {"stemming": false, "stop_words": false}}"#;

    fn parse(declared: &str) -> Result<Vec<TextColumn>, DocsError> {
        parse_text_columns(&serde_json::from_str(declared).expect("test JSON"))
    }

    fn parse_ok(declared: &str) -> Vec<TextColumn> {
        parse(declared).unwrap_or_else(|e| panic!("{declared}: {e:?}"))
    }

    fn parse_err(declared: &str) -> DocsError {
        parse(declared).expect_err(declared)
    }

    fn assert_error(error: &DocsError, code: Code, message: &[&str], hint: Option<&str>) {
        assert_eq!(error.code, code, "{error:?}");
        for part in message {
            assert!(error.message.contains(part), "{part:?} not in {error:?}");
        }
        assert!(
            error.message.starts_with("pinecone.track_docs: "),
            "{error:?}"
        );
        if let Some(part) = hint {
            let text = error.hint.as_deref().unwrap_or_default();
            assert!(text.contains(part), "{part:?} not in hint of {error:?}");
        }
    }

    fn fts(language: &str, stemming: bool, stop_words: bool) -> Value {
        json!({"type": "string", "full_text_search":
            {"language": language, "stemming": stemming, "stop_words": stop_words}})
    }

    fn vector(dimension: u32, metric: &str) -> VectorField {
        VectorField {
            column: "embedding".to_string(),
            dimension,
            metric: metric.to_string(),
        }
    }

    // ---- declaration parsing and schema derivation ----

    #[test]
    fn example_declaration_derives_the_documented_schema() {
        let columns = parse_ok(EXAMPLE);
        let schema = derive_schema(&columns, Some(&vector(3, "cosine"))).expect("schema");
        assert_eq!(
            schema,
            json!({"fields": {
                "title": fts("en", true, true),
                "body": fts("en", true, true),
                "sku": fts("en", false, false),
                "embedding": {"type": "dense_vector", "dimension": 3, "metric": "cosine"},
            }})
        );
    }

    #[test]
    fn defaults_are_english_stemming_stop_words_and_truncate() {
        for declared in [r#"{"body": {}}"#, r#"{"body": null}"#] {
            let columns = parse_ok(declared);
            assert_eq!(columns.len(), 1, "{declared}");
            let body = &columns[0];
            assert_eq!(body.name, "body");
            assert_eq!(body.language.as_deref(), Some("en"));
            assert!(body.stemming && body.stop_words, "{body:?}");
            assert_eq!(body.ngram, None);
            assert_eq!(body.on_oversize, OnOversize::Truncate);
        }
    }

    #[test]
    fn derived_schema_has_no_vector_field_without_a_vector_column_and_no_local_options() {
        let columns = parse_ok(r#"{"body": {"on_oversize": "error"}}"#);
        let schema = derive_schema(&columns, None).expect("schema");
        assert_eq!(schema, json!({"fields": {"body": fts("en", true, true)}}));
    }

    #[test]
    fn stop_words_default_follows_stemming_but_explicit_true_needs_stemming() {
        let off = parse_ok(r#"{"sku": {"stemming": false}}"#);
        assert!(!off[0].stemming && !off[0].stop_words, "{:?}", off[0]);
        let error = parse_err(r#"{"sku": {"stemming": false, "stop_words": true}}"#);
        assert_error(
            &error,
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            &["column \"sku\": stop_words requires stemming"],
            Some("stop_words => false"),
        );
    }

    #[test]
    fn ngram_column_does_not_inherit_the_stemming_defaults() {
        let columns = parse_ok(r#"{"sku": {"ngram": {"min_gram": 2, "max_gram": 4}}}"#);
        assert!(
            !columns[0].stemming && !columns[0].stop_words,
            "{:?}",
            columns[0]
        );
        assert_eq!(columns[0].language, None, "language only when declared");
        let schema = derive_schema(&columns, None).expect("schema");
        assert_eq!(
            schema,
            json!({"fields": {"sku": {"type": "string", "full_text_search":
                {"ngram": {"min_gram": 2, "max_gram": 4}}}}})
        );
    }

    #[test]
    fn ngram_keeps_an_explicit_language_and_prefix_only_and_accepts_explicit_false() {
        let declared = r#"{"name": {"language": "German", "stemming": false, "stop_words": false,
            "ngram": {"min_gram": 1, "max_gram": 3, "prefix_only": true}}}"#;
        let schema = derive_schema(&parse_ok(declared), None).expect("schema");
        assert_eq!(
            schema["fields"]["name"]["full_text_search"],
            json!({"language": "de",
                   "ngram": {"min_gram": 1, "max_gram": 3, "prefix_only": true}})
        );
    }

    #[test]
    fn ngram_combined_with_stemming_or_stop_words_is_refused_with_a_hint() {
        for option in [r#""stemming": true"#, r#""stop_words": true"#] {
            let declared =
                format!(r#"{{"sku": {{"ngram": {{"min_gram": 2, "max_gram": 4}}, {option}}}}}"#);
            assert_error(
                &parse_err(&declared),
                Code::ERRCODE_INVALID_PARAMETER_VALUE,
                &["column \"sku\": ngram cannot be combined with stemming or stop_words"],
                Some("identifier columns: stemming => false, stop_words => false"),
            );
        }
    }

    #[test]
    fn ngram_options_are_validated() {
        for ngram in [
            r#""x""#,
            r#"{}"#,
            r#"{"min_gram": 2}"#,
            r#"{"max_gram": 2}"#,
            r#"{"min_gram": 0, "max_gram": 2}"#,
            r#"{"min_gram": 3, "max_gram": 2}"#,
            r#"{"min_gram": 1.5, "max_gram": 2}"#,
            r#"{"min_gram": "2", "max_gram": 3}"#,
            r#"{"min_gram": 2, "max_gram": 3, "prefix_only": "yes"}"#,
            r#"{"min_gram": 2, "max_gram": 3, "edge": true}"#,
        ] {
            let error = parse_err(&format!(r#"{{"sku": {{"ngram": {ngram}}}}}"#));
            assert_error(
                &error,
                Code::ERRCODE_INVALID_PARAMETER_VALUE,
                &["column \"sku\": ngram"],
                Some("min_gram"),
            );
        }
    }

    #[test]
    fn unknown_option_is_named_with_the_allowed_list() {
        assert_error(
            &parse_err(r#"{"body": {"stemmer": true}}"#),
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            &["column \"body\": unknown option \"stemmer\""],
            Some("allowed: language, stemming, stop_words, ngram, on_oversize"),
        );
    }

    #[test]
    fn option_values_are_type_checked() {
        for (declared, part) in [
            (r#"{"b": {"language": 5}}"#, "language must be a string"),
            (r#"{"b": {"language": ""}}"#, "language must not be empty"),
            (r#"{"b": {"language": "en us"}}"#, "language"),
            (
                r#"{"b": {"stemming": "yes"}}"#,
                "stemming must be true or false",
            ),
            (
                r#"{"b": {"stop_words": 1}}"#,
                "stop_words must be true or false",
            ),
            (
                r#"{"b": {"on_oversize": "cut"}}"#,
                "on_oversize must be 'truncate' or 'error'",
            ),
            (
                r#"{"b": {"on_oversize": true}}"#,
                "on_oversize must be 'truncate' or 'error'",
            ),
            (r#"{"b": "en"}"#, "options must be a JSON object"),
            (r#"{"b": [1]}"#, "options must be a JSON object"),
        ] {
            let error = parse_err(declared);
            assert_error(&error, Code::ERRCODE_INVALID_PARAMETER_VALUE, &[part], None);
        }
    }

    #[test]
    fn language_names_normalise_to_short_codes_and_unknown_ones_pass_through() {
        for (given, expected) in [
            ("en", "en"),
            ("english", "en"),
            ("English", "en"),
            ("EN", "en"),
            ("german", "de"),
            ("de", "de"),
            ("Norwegian", "no"),
            ("tr", "tr"),
            ("zh-cn", "zh-cn"),
            ("klingon", "klingon"),
        ] {
            let columns = parse_ok(&format!(r#"{{"b": {{"language": "{given}"}}}}"#));
            assert_eq!(columns[0].language.as_deref(), Some(expected), "{given}");
        }
    }

    #[test]
    fn on_oversize_accepts_truncate_and_error() {
        let columns =
            parse_ok(r#"{"a": {"on_oversize": "error"}, "b": {"on_oversize": "truncate"}}"#);
        assert_eq!(columns[0].on_oversize, OnOversize::Error);
        assert_eq!(columns[1].on_oversize, OnOversize::Truncate);
    }

    #[test]
    fn field_names_follow_pinecone_rules() {
        let long = "a".repeat(65);
        let wide_ok = "é".repeat(32);
        let wide_bad = "é".repeat(33);
        for bad in ["_body", "$price", "", long.as_str(), wide_bad.as_str()] {
            let declared = json!({ bad: {} });
            let error = parse_text_columns(&declared).expect_err(bad);
            assert_error(
                &error,
                Code::ERRCODE_INVALID_NAME,
                &[&format!("\"{bad}\" is not a valid Pinecone field name")],
                None,
            );
        }
        for good in [
            "body",
            "a",
            "a_b$",
            "x".repeat(64).as_str(),
            wide_ok.as_str(),
            "PgPinecone_x",
        ] {
            parse_text_columns(&json!({ good: {} })).unwrap_or_else(|e| panic!("{good}: {e:?}"));
        }
    }

    #[test]
    fn reserved_prefix_is_refused() {
        for reserved in ["pg_pinecone_truncated", "pg_pinecone_extra", "pg_pinecone_"] {
            let error = parse_text_columns(&json!({ reserved: {} })).expect_err(reserved);
            assert_error(
                &error,
                Code::ERRCODE_INVALID_NAME,
                &[&format!(
                    "\"{reserved}\" collides with a reserved field name"
                )],
                Some("pg_pinecone_"),
            );
        }
        assert!(parse_text_columns(&json!({"pg_pinecone": {}})).is_ok());
    }

    #[test]
    fn text_columns_also_accepts_a_list_of_column_names_meaning_defaults() {
        assert_eq!(
            parse_ok(r#"["title", "body"]"#),
            parse_ok(r#"{"title": {}, "body": {}}"#)
        );
        assert_eq!(parse_ok(r#"["body"]"#), parse_ok(r#"{"body": null}"#));
    }

    #[test]
    fn a_list_of_text_columns_refuses_mixed_shapes_non_strings_and_repeats() {
        for bad in [
            r#"["title", {"body": {"language": "de"}}]"#,
            r#"["title", 5]"#,
            r#"["title", null]"#,
            r#"[["title"]]"#,
        ] {
            let error = parse_err(bad);
            assert_error(
                &error,
                Code::ERRCODE_INVALID_PARAMETER_VALUE,
                &["text_columns", "column names"],
                Some("object form"),
            );
        }
        let error = parse_err(r#"["body", "body"]"#);
        assert_error(
            &error,
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            &["\"body\" is listed twice"],
            None,
        );
        let error = parse_err(r#"["_body"]"#);
        assert_error(
            &error,
            Code::ERRCODE_INVALID_NAME,
            &["\"_body\" is not a valid Pinecone field name"],
            None,
        );
    }

    #[test]
    fn text_columns_must_be_a_non_empty_object_within_pinecones_limit() {
        for bad in ["[]", r#""body""#, "5", "null", "{}"] {
            let declared: Value = serde_json::from_str(bad).expect("JSON");
            let error = parse_text_columns(&declared).expect_err(bad);
            assert_error(
                &error,
                Code::ERRCODE_INVALID_PARAMETER_VALUE,
                &["text_columns"],
                None,
            );
        }
        let many = |n: usize| -> Value {
            Value::Object((0..n).map(|i| (format!("c{i:03}"), json!({}))).collect())
        };
        assert_eq!(
            parse_text_columns(&many(100)).expect("100 allowed").len(),
            100
        );
        let error = parse_text_columns(&many(101)).expect_err("101 refused");
        assert_error(
            &error,
            Code::ERRCODE_PROGRAM_LIMIT_EXCEEDED,
            &["101", "100"],
            None,
        );
    }

    #[test]
    fn parsed_columns_are_sorted_by_name() {
        let names: Vec<_> = parse_ok(r#"{"z": {}, "a": {}, "m": {}}"#)
            .into_iter()
            .map(|c| c.name)
            .collect();
        assert_eq!(names, ["a", "m", "z"]);
    }

    #[test]
    fn vector_field_uses_pinecone_metric_names() {
        let columns = parse_ok(r#"{"body": {}}"#);
        for (metric, pinecone) in [
            ("cosine", "cosine"),
            ("l2", "euclidean"),
            ("ip", "dotproduct"),
        ] {
            let schema = derive_schema(&columns, Some(&vector(768, metric))).expect(metric);
            assert_eq!(
                schema["fields"]["embedding"],
                json!({"type": "dense_vector", "dimension": 768, "metric": pinecone})
            );
        }
    }

    #[test]
    fn vector_field_is_validated() {
        let columns = parse_ok(r#"{"body": {}}"#);
        for dimension in [0, 20_001] {
            let error =
                derive_schema(&columns, Some(&vector(dimension, "cosine"))).expect_err("dim");
            assert_error(
                &error,
                Code::ERRCODE_INVALID_PARAMETER_VALUE,
                &["dimension"],
                None,
            );
        }
        assert!(derive_schema(&columns, Some(&vector(20_000, "cosine"))).is_ok());
        let error = derive_schema(&columns, Some(&vector(3, "hamming"))).expect_err("metric");
        assert_error(
            &error,
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            &["hamming"],
            Some("cosine"),
        );

        let clash = VectorField {
            column: "body".to_string(),
            ..vector(3, "cosine")
        };
        let error = derive_schema(&columns, Some(&clash)).expect_err("clash");
        assert_error(
            &error,
            Code::ERRCODE_INVALID_PARAMETER_VALUE,
            &["\"body\"", "both"],
            None,
        );
        let reserved = VectorField {
            column: "pg_pinecone_vec".to_string(),
            ..vector(3, "cosine")
        };
        let error = derive_schema(&columns, Some(&reserved)).expect_err("reserved");
        assert_error(
            &error,
            Code::ERRCODE_INVALID_NAME,
            &["\"pg_pinecone_vec\""],
            None,
        );
    }

    #[test]
    fn stored_text_columns_round_trip_with_resolved_defaults() {
        for declared in [
            EXAMPLE,
            r#"{"sku": {"ngram": {"min_gram": 2, "max_gram": 4}, "on_oversize": "error"}}"#,
            r#"{"n": {"language": "german", "ngram": {"min_gram": 1, "max_gram": 2, "prefix_only": true}}}"#,
            r#"{"b": {}}"#,
        ] {
            let columns = parse_ok(declared);
            let stored = text_columns_json(&columns);
            let again = parse_text_columns(&stored).expect("stored json parses");
            assert_eq!(again, columns, "{declared} -> {stored}");
        }
        let stored = text_columns_json(&parse_ok(r#"{"b": {}}"#));
        assert_eq!(
            stored,
            json!({"b": {"language": "en", "stemming": true, "stop_words": true,
                         "on_oversize": "truncate"}})
        );
    }

    // ---- attach comparison ----

    fn example_schema() -> Value {
        derive_schema(&parse_ok(EXAMPLE), Some(&vector(3, "cosine"))).expect("schema")
    }

    /// The schema as the control plane describes it back: analyzer defaults filled in and
    /// server-applied keys added.
    fn described(schema: &Value) -> Value {
        let mut out = schema.clone();
        for (_, field) in out["fields"].as_object_mut().expect("fields").iter_mut() {
            if let Some(cfg) = field
                .get_mut("full_text_search")
                .and_then(Value::as_object_mut)
            {
                cfg.entry("language").or_insert(json!("en"));
                cfg.entry("stemming").or_insert(json!(false));
                cfg.entry("stop_words").or_insert(json!(false));
                cfg.insert("lowercase".to_string(), json!(true));
                cfg.insert("max_term_len".to_string(), json!(40));
            }
        }
        out
    }

    #[test]
    fn identical_schemas_match_whatever_the_key_order() {
        let declared = example_schema();
        assert_eq!(schema_mismatch(&declared, &declared), None);
        let reversed: Value = serde_json::from_str(
            r#"{"fields": {"embedding": {"metric": "cosine", "dimension": 3, "type": "dense_vector"},
                "sku": {"full_text_search": {"stop_words": false, "stemming": false, "language": "en"},
                        "type": "string"},
                "body": {"full_text_search": {"stop_words": true, "stemming": true, "language": "en"},
                         "type": "string"},
                "title": {"type": "string", "full_text_search":
                          {"stemming": true, "language": "en", "stop_words": true}}}}"#,
        )
        .expect("JSON");
        assert_eq!(schema_mismatch(&declared, &reversed), None);
    }

    #[test]
    fn server_applied_keys_and_filled_in_defaults_are_not_a_difference() {
        let declared = example_schema();
        assert_eq!(schema_mismatch(&declared, &described(&declared)), None);

        let ngram = derive_schema(
            &parse_ok(r#"{"sku": {"ngram": {"min_gram": 2, "max_gram": 4}}}"#),
            None,
        )
        .expect("schema");
        let remote = json!({"fields": {"sku": {"type": "string", "full_text_search": {
            "language": "en", "stemming": false, "stop_words": false, "lowercase": true,
            "ngram": {"min_gram": 2, "max_gram": 4, "prefix_only": false}}}}});
        assert_eq!(schema_mismatch(&ngram, &remote), None);

        let defaults = json!({"fields": {"b": {"type": "string", "full_text_search": {}}}});
        let explicit = json!({"fields": {"b": fts("en", false, false)}});
        assert_eq!(schema_mismatch(&explicit, &defaults), None);
    }

    #[test]
    fn a_different_analyzer_names_the_field_and_both_specs() {
        let declared = example_schema();
        let mut remote = described(&declared);
        remote["fields"]["body"]["full_text_search"]["language"] = json!("de");
        let mismatch = schema_mismatch(&declared, &remote).expect("language differs");
        assert_eq!(mismatch.field, "body");
        assert_eq!(
            mismatch.describe(),
            r#"field "body" is {"language":"de","stemming":true,"stop_words":true} remotely, {"language":"en","stemming":true,"stop_words":true} declared"#
        );

        for (key, value) in [("stemming", json!(false)), ("stop_words", json!(false))] {
            let mut remote = described(&declared);
            remote["fields"]["title"]["full_text_search"][key] = value;
            let mismatch = schema_mismatch(&declared, &remote).expect(key);
            assert_eq!(mismatch.field, "title", "{key}");
        }
        let mut remote = described(&declared);
        remote["fields"]["sku"]["full_text_search"]["ngram"] =
            json!({"min_gram": 2, "max_gram": 3});
        assert_eq!(
            schema_mismatch(&declared, &remote).expect("ngram").field,
            "sku"
        );
    }

    #[test]
    fn dense_field_dimension_and_metric_must_match() {
        let declared = example_schema();
        for (key, value) in [("dimension", json!(4)), ("metric", json!("euclidean"))] {
            let mut remote = described(&declared);
            remote["fields"]["embedding"][key] = value;
            let mismatch = schema_mismatch(&declared, &remote).expect(key);
            assert_eq!(mismatch.field, "embedding", "{key}");
            assert!(
                mismatch.describe().contains("remotely"),
                "{}",
                mismatch.describe()
            );
        }
    }

    #[test]
    fn missing_retyped_and_extra_searchable_fields_are_differences() {
        let declared = example_schema();

        let mut remote = described(&declared);
        remote["fields"]
            .as_object_mut()
            .expect("fields")
            .remove("sku");
        let mismatch = schema_mismatch(&declared, &remote).expect("missing");
        assert_eq!(
            (mismatch.field.as_str(), mismatch.remote.is_none()),
            ("sku", true)
        );
        assert_eq!(
            mismatch.describe(),
            r#"field "sku" is missing remotely, {"language":"en","stemming":false,"stop_words":false} declared"#
        );

        let mut remote = described(&declared);
        remote["fields"]["title"] =
            json!({"type": "dense_vector", "dimension": 3, "metric": "cosine"});
        assert_eq!(
            schema_mismatch(&declared, &remote).expect("retyped").field,
            "title"
        );

        for extra in [
            fts("en", true, true),
            json!({"type": "dense_vector", "dimension": 3, "metric": "cosine"}),
            json!({"type": "sparse_vector"}),
        ] {
            let mut remote = described(&declared);
            remote["fields"]["extra"] = extra;
            let mismatch = schema_mismatch(&declared, &remote).expect("extra searchable field");
            assert_eq!(mismatch.field, "extra");
            assert!(mismatch.declared.is_none());
            assert!(
                mismatch.describe().contains("not declared"),
                "{}",
                mismatch.describe()
            );
        }
    }

    #[test]
    fn response_only_metadata_fields_of_the_remote_are_ignored() {
        let declared = example_schema();
        let mut remote = described(&declared);
        for (name, spec) in [
            ("tags", json!({"type": "string_list", "filterable": true})),
            ("featured", json!({"type": "boolean", "filterable": true})),
            ("views", json!({"type": "integer", "filterable": true})),
            ("rating", json!({"type": "float", "filterable": true})),
            ("author", json!({"type": "string", "filterable": true})),
        ] {
            remote["fields"][name] = spec;
        }
        assert_eq!(schema_mismatch(&declared, &remote), None);
    }

    #[test]
    fn a_remote_without_fields_differs_in_the_first_declared_field() {
        let declared = example_schema();
        for remote in [Value::Null, json!({}), json!({"fields": []})] {
            let mismatch = schema_mismatch(&declared, &remote).expect("nothing remotely");
            assert_eq!(mismatch.field, "body", "fields are compared in name order");
        }
    }

    // ---- size policy ----

    #[test]
    fn text_within_the_limits_is_returned_unchanged() {
        for text in ["", "a", "  a  b\n", "ünï cödé", "   "] {
            assert_eq!(
                truncate_text(text, MAX_FIELD_TOKENS, MAX_FIELD_BYTES),
                (text.to_string(), false)
            );
        }
        let exact_tokens = "ab ".repeat(MAX_FIELD_TOKENS);
        assert_eq!(
            truncate_text(&exact_tokens, MAX_FIELD_TOKENS, usize::MAX),
            (exact_tokens.clone(), false)
        );
        let exact_bytes = "a".repeat(100);
        assert_eq!(
            truncate_text(&exact_bytes, 10, 100),
            (exact_bytes.clone(), false)
        );
        let exact_token = "x".repeat(MAX_TOKEN_BYTES);
        assert_eq!(
            truncate_text(&exact_token, 10, 1000),
            (exact_token.clone(), false)
        );
    }

    #[test]
    fn token_cap_keeps_the_first_tokens_and_their_separators() {
        assert_eq!(
            truncate_text("a  b\tc d", 2, 100),
            ("a  b".to_string(), true)
        );
        assert_eq!(truncate_text("  a b c", 1, 100), ("  a".to_string(), true));
        assert_eq!(truncate_text("a b c", 0, 100), (String::new(), true));
        let long = "ab ".repeat(22_000);
        let (cut, truncated) = truncate_text(&long, MAX_FIELD_TOKENS, MAX_FIELD_BYTES);
        assert!(truncated);
        assert_eq!(estimate_tokens(&cut), MAX_FIELD_TOKENS);
        assert!(long.starts_with(&cut));
    }

    #[test]
    fn byte_cap_cuts_at_a_token_boundary() {
        assert_eq!(
            truncate_text("abcde fghij klmno", 100, 11),
            ("abcde fghij".to_string(), true)
        );
        assert_eq!(
            truncate_text("abcde fghij klmno", 100, 10),
            ("abcde".to_string(), true)
        );
        assert_eq!(truncate_text("abcdef", 100, 3), (String::new(), true));
        let long = "abcdefghijklmnopqrst ".repeat(6_000);
        let (cut, truncated) = truncate_text(&long, MAX_FIELD_TOKENS, MAX_FIELD_BYTES);
        assert!(truncated);
        assert!(
            cut.len() <= MAX_FIELD_BYTES && cut.len() > MAX_FIELD_BYTES - 21,
            "{}",
            cut.len()
        );
        assert!(long.starts_with(&cut));
        assert!(estimate_tokens(&cut) < MAX_FIELD_TOKENS);
    }

    #[test]
    fn whichever_cap_binds_first_wins() {
        let long = "abcdefghi ".repeat(15_000);
        let (cut, truncated) = truncate_text(&long, MAX_FIELD_TOKENS, MAX_FIELD_BYTES);
        assert!(truncated);
        assert_eq!(
            estimate_tokens(&cut),
            MAX_FIELD_TOKENS,
            "9,000 tokens bind before 100,000 bytes"
        );
        assert!(cut.len() < MAX_FIELD_BYTES);
    }

    #[test]
    fn a_token_over_256_bytes_is_clipped_on_a_character_boundary() {
        let (cut, truncated) = truncate_text(&format!("a {} b", "x".repeat(300)), 100, 10_000);
        assert!(truncated);
        assert_eq!(cut, format!("a {} b", "x".repeat(MAX_TOKEN_BYTES)));

        let euros = "€".repeat(100);
        let (cut, truncated) = truncate_text(&format!("{euros} end"), 100, 10_000);
        assert!(truncated);
        let clipped = cut.split_whitespace().next().expect("first token");
        assert_eq!(clipped.len(), 255, "256 falls inside a 3-byte character");
        assert!(euros.starts_with(clipped));
        assert!(cut.ends_with(" end"));

        let wide = "é".repeat(200);
        let (cut, _) = truncate_text(&wide, 100, 10_000);
        assert_eq!(cut.len(), MAX_TOKEN_BYTES);
    }

    #[test]
    fn clipping_counts_toward_the_byte_cap() {
        let text = format!("{} {}", "x".repeat(300), "y".repeat(300));
        let (cut, truncated) = truncate_text(&text, 100, 260);
        assert!(truncated);
        assert_eq!(cut, "x".repeat(MAX_TOKEN_BYTES));
    }

    #[test]
    fn whitespace_only_text_over_the_byte_cap_is_emptied() {
        assert_eq!(
            truncate_text(&" ".repeat(50), 10, 20),
            (String::new(), true)
        );
    }

    #[test]
    fn token_estimate_counts_runs_of_alphanumeric_characters() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("   "), 0);
        assert_eq!(estimate_tokens(" a\tb\nc  d "), 4);
        assert_eq!(estimate_tokens("a\u{00a0}b\u{3000}c"), 3);
        assert_eq!(estimate_tokens("one-token, still_one"), 4);
        assert_eq!(estimate_tokens("https://example.com/a/b?x=1"), 7);
        assert_eq!(
            estimate_tokens("--- ,,, €€"),
            0,
            "no letters or digits, no tokens"
        );
        assert_eq!(
            estimate_tokens("ünï 1é2"),
            2,
            "letters and digits of any script"
        );
    }

    #[test]
    fn punctuation_dense_text_counts_by_its_runs_not_its_words() {
        let csv = "a,b,c ".repeat(4_000);
        assert_eq!(csv.split_whitespace().count(), 4_000, "under 9,000 words");
        assert_eq!(text_size(&csv).tokens, 12_000);
        assert!(
            exceeds_field_limits(&text_size(&csv)),
            "but over 9,000 tokens"
        );
        let (cut, truncated) = truncate_text(&csv, MAX_FIELD_TOKENS, MAX_FIELD_BYTES);
        assert!(truncated);
        assert_eq!(estimate_tokens(&cut), MAX_FIELD_TOKENS);
        assert!(csv.starts_with(&cut));
        let words = "a b c ".repeat(2_999);
        assert!(
            !exceeds_field_limits(&text_size(&words)),
            "8,997 tokens fit"
        );
    }

    #[test]
    fn a_word_of_several_runs_is_cut_at_the_end_of_a_run() {
        assert_eq!(
            truncate_text("a,b,c,d xyz", 2, 100),
            ("a,b".to_string(), true)
        );
        assert_eq!(
            truncate_text("a,b,c,d xyz", 100, 4),
            ("a,b".to_string(), true),
            "bytes: \"a,b,\" would end between runs"
        );
        assert_eq!(
            truncate_text("a,b xyz", 3, 100),
            ("a,b xyz".to_string(), false)
        );
        let (cut, _) = truncate_text("w1 a,b,c w2", 3, 100);
        assert_eq!(cut, "w1 a,b", "tokens: w1, a, b");
    }

    #[test]
    fn the_message_names_the_limit_a_value_breaks() {
        let long_word = text_size(&format!("see {} here", "u".repeat(300)));
        let message = oversize_message("body", &long_word);
        assert!(
            message.contains("column \"body\" contains a 300-byte token"),
            "{message}"
        );
        assert!(message.contains("256 bytes per token"), "{message}");
        assert!(!message.contains("100 KB"), "{message}");

        let big = text_size(&"ab ".repeat(22_000));
        let message = oversize_message("body", &big);
        assert!(message.contains("66000 bytes (~22000 tokens)"), "{message}");
        assert!(message.contains("100 KB and 10000 tokens"), "{message}");

        let both = text_size(&format!("{} {}", "x".repeat(300), "ab ".repeat(40_000)));
        assert!(
            oversize_message("body", &both).contains("100 KB"),
            "size first"
        );
    }

    #[test]
    fn text_size_reports_bytes_tokens_and_the_longest_token() {
        let size = text_size("ab cd  é\tlonger");
        assert_eq!(
            (size.bytes, size.tokens, size.longest_token),
            (16, 4, 6),
            "{size:?}"
        );
        let empty = text_size("");
        assert_eq!((empty.bytes, empty.tokens, empty.longest_token), (0, 0, 0));
    }

    #[test]
    fn field_limits_flag_bytes_tokens_and_long_tokens_exactly_as_truncation_cuts() {
        let samples = [
            String::new(),
            "short".to_string(),
            "ab ".repeat(MAX_FIELD_TOKENS),
            "ab ".repeat(MAX_FIELD_TOKENS + 1),
            "x".repeat(MAX_FIELD_BYTES),
            "x ".repeat(MAX_FIELD_BYTES / 2 + 1),
            "y".repeat(MAX_TOKEN_BYTES),
            "y".repeat(MAX_TOKEN_BYTES + 1),
            format!("a {} b", "z".repeat(300)),
        ];
        for text in &samples {
            let cut = truncate_text(text, MAX_FIELD_TOKENS, MAX_FIELD_BYTES).1;
            let over = exceeds_field_limits(&text_size(text));
            assert_eq!(over, cut, "{} bytes", text.len());
        }
        assert!(!exceeds_field_limits(&text_size(&samples[2])));
        assert!(exceeds_field_limits(&text_size(&samples[3])));
    }

    // ---- metadata typing ----

    #[test]
    fn metadata_types_pass_through_or_convert() {
        for (type_name, kind, json_type) in [
            ("int2", MetaKind::Int, "number"),
            ("int4", MetaKind::Int, "number"),
            ("int8", MetaKind::Int, "number"),
            ("float4", MetaKind::Float, "number"),
            ("float8", MetaKind::Float, "number"),
            ("numeric", MetaKind::Numeric, "number"),
            ("bool", MetaKind::Bool, "boolean"),
            ("text", MetaKind::Text, "string"),
            ("varchar", MetaKind::Text, "string"),
            ("bpchar", MetaKind::Text, "string"),
            ("uuid", MetaKind::Uuid, "string"),
            ("_text", MetaKind::TextArray, "string[]"),
            ("date", MetaKind::Date, "number"),
            ("timestamp", MetaKind::Timestamp, "number"),
            ("timestamptz", MetaKind::Timestamptz, "number"),
        ] {
            assert_eq!(metadata_kind(type_name), Some(kind), "{type_name}");
            assert_eq!(kind.json_type(), json_type, "{type_name}");
            let checked = check_metadata_column("col", type_name);
            assert_eq!(checked.expect(type_name), kind);
        }
    }

    #[test]
    fn metadata_types_without_a_pinecone_type_are_refused_naming_the_column() {
        for type_name in [
            "jsonb",
            "json",
            "_int4",
            "_varchar",
            "_float8",
            "my_composite",
            "bytea",
            "interval",
            "time",
            "timetz",
            "money",
            "inet",
            "point",
            "tsvector",
            "oid",
            "vector",
        ] {
            let error = check_metadata_column("payload", type_name).expect_err(type_name);
            assert_error(
                &error,
                Code::ERRCODE_FEATURE_NOT_SUPPORTED,
                &["metadata column \"payload\" has type"],
                Some("generated column"),
            );
        }
        let error = check_metadata_column("payload", "jsonb").expect_err("jsonb");
        assert!(error.message.contains("has type jsonb"), "{error:?}");
        let error = check_metadata_column("scores", "_int4").expect_err("int4[]");
        assert!(error.message.contains("has type int4[]"), "{error:?}");
    }

    #[test]
    fn metadata_column_names_follow_the_field_rules() {
        for (name, code) in [
            ("_tag", Code::ERRCODE_INVALID_NAME),
            ("$price", Code::ERRCODE_INVALID_NAME),
            ("pg_pinecone_truncated", Code::ERRCODE_INVALID_NAME),
        ] {
            let error = check_metadata_column(name, "text").expect_err(name);
            assert_error(&error, code, &[&format!("\"{name}\""), "field name"], None);
        }
    }

    #[test]
    fn metadata_value_expressions_convert_to_pinecone_types() {
        let col = r#""t"."c""#;
        for (kind, expected) in [
            (MetaKind::Int, r#""t"."c""#.to_string()),
            (MetaKind::Float, r#""t"."c""#.to_string()),
            (MetaKind::Bool, r#""t"."c""#.to_string()),
            (MetaKind::Text, r#""t"."c"::text"#.to_string()),
            (MetaKind::TextArray, r#""t"."c""#.to_string()),
            (MetaKind::Uuid, r#""t"."c"::text"#.to_string()),
            (MetaKind::Numeric, r#""t"."c"::float8"#.to_string()),
            (
                MetaKind::Timestamptz,
                r#"CASE WHEN pg_catalog.isfinite("t"."c") THEN pg_catalog.date_part('epoch', "t"."c") END"#.to_string(),
            ),
            (
                MetaKind::Timestamp,
                r#"CASE WHEN pg_catalog.isfinite("t"."c") THEN pg_catalog.date_part('epoch', "t"."c") END"#.to_string(),
            ),
            (
                MetaKind::Date,
                r#"CASE WHEN pg_catalog.isfinite("t"."c") THEN pg_catalog.date_part('epoch', "t"."c"::timestamp) END"#.to_string(),
            ),
        ] {
            assert_eq!(kind.value_expr(col), expected, "{kind:?}");
        }
    }

    // ---- ids ----

    #[test]
    fn document_ids_add_bpchar_without_touching_the_vector_list() {
        use crate::catalog::{DOC_ID_COLUMN_TYPES, ID_COLUMN_TYPES};
        assert!(DOC_ID_COLUMN_TYPES.contains(&"bpchar"));
        assert!(
            !ID_COLUMN_TYPES.contains(&"bpchar"),
            "the vector path is unchanged"
        );
        for kept in ID_COLUMN_TYPES {
            assert!(DOC_ID_COLUMN_TYPES.contains(kept), "{kept}");
        }
        assert_eq!(DOC_ID_COLUMN_TYPES.len(), ID_COLUMN_TYPES.len() + 1);
    }
}
