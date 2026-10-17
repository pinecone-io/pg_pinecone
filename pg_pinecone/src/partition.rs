//! Partition column naming: one Pinecone namespace per value, and the ownership claims that keep
//! tables from sharing one. Pure Rust, no Postgres.

use std::fmt;

use crate::docs_track::valid_namespace;

pub(crate) const MAX_NAMESPACE_BYTES: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartitionKind {
    Int2,
    Int4,
    Int8,
    Text,
    Uuid,
}

impl PartitionKind {
    pub(crate) const ALL: [PartitionKind; 5] = [
        PartitionKind::Int2,
        PartitionKind::Int4,
        PartitionKind::Int8,
        PartitionKind::Text,
        PartitionKind::Uuid,
    ];

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            PartitionKind::Int2 => "int2",
            PartitionKind::Int4 => "int4",
            PartitionKind::Int8 => "int8",
            PartitionKind::Text => "text",
            PartitionKind::Uuid => "uuid",
        }
    }

    pub(crate) fn parse(kind: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.as_str() == kind)
    }

    /// `pg_type.typname` of a partition column; `varchar` partitions as `text`.
    pub(crate) fn from_typname(typname: &str) -> Option<Self> {
        match typname {
            "varchar" => Some(PartitionKind::Text),
            other => Self::parse(other),
        }
    }

    /// Whether `text` is what [`canon`] makes of a value of this kind: the shape a namespace of a
    /// column of this kind can decode to. No write of the table produces any other.
    pub(crate) fn is_canonical(self, text: &str) -> bool {
        match self {
            PartitionKind::Int2 => text.parse::<i16>().is_ok_and(|n| n.to_string() == text),
            PartitionKind::Int4 => text.parse::<i32>().is_ok_and(|n| n.to_string() == text),
            PartitionKind::Int8 => text.parse::<i64>().is_ok_and(|n| n.to_string() == text),
            PartitionKind::Uuid => is_canonical_uuid(text),
            PartitionKind::Text => true,
        }
    }
}

/// Lowercase hex in the 8-4-4-4-12 grouping.
fn is_canonical_uuid(text: &str) -> bool {
    text.len() == 36
        && text.bytes().enumerate().all(|(i, byte)| match i {
            8 | 13 | 18 | 23 => byte == b'-',
            _ => matches!(byte, b'0'..=b'9' | b'a'..=b'f'),
        })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PartitionSpec {
    pub(crate) column: String,
    pub(crate) kind: PartitionKind,
}

impl PartitionSpec {
    /// The registry's two nullable columns, which must be both set or both NULL.
    pub(crate) fn from_columns(
        column: Option<String>,
        kind: Option<String>,
    ) -> Result<Option<Self>, String> {
        match (column, kind) {
            (None, None) => Ok(None),
            (Some(column), Some(kind)) => match PartitionKind::parse(&kind) {
                Some(kind) => Ok(Some(Self { column, kind })),
                None => Err(format!("unknown partition_kind '{kind}'")),
            },
            (Some(_), None) => Err("partition_column is set without partition_kind".to_string()),
            (None, Some(_)) => Err("partition_kind is set without partition_column".to_string()),
        }
    }
}

/// Which registry (and sync lane) a partition belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lane {
    Vector,
    Docs,
}

impl Lane {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Lane::Vector => "vector",
            Lane::Docs => "docs",
        }
    }

    #[allow(dead_code, reason = "the tests read the lane column back through it")]
    pub(crate) fn parse(lane: &str) -> Option<Self> {
        [Lane::Vector, Lane::Docs]
            .into_iter()
            .find(|l| l.as_str() == lane)
    }
}

/// A partition value as the writers see it, before it becomes text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PartitionValue {
    Int(i64),
    #[allow(
        dead_code,
        reason = "no writer builds one: Postgres renders a uuid as text for us"
    )]
    Uuid([u8; 16]),
    Text(String),
}

/// The canonical text of a value: what `col::text` gives in plpgsql for the supported types.
pub(crate) fn canon(value: PartitionValue) -> String {
    match value {
        PartitionValue::Int(int) => int.to_string(),
        PartitionValue::Uuid(bytes) => uuid_text(&bytes),
        PartitionValue::Text(text) => text,
    }
}

fn uuid_text(bytes: &[u8; 16]) -> String {
    let mut out = String::with_capacity(36);
    for (i, byte) in bytes.iter().enumerate() {
        if matches!(i, 4 | 6 | 8 | 10) {
            out.push('-');
        }
        out.push(hex_digit(byte >> 4, HEX_LOWER));
        out.push(hex_digit(byte & 15, HEX_LOWER));
    }
    out
}

const HEX_LOWER: &[u8; 16] = b"0123456789abcdef";
const HEX_UPPER: &[u8; 16] = b"0123456789ABCDEF";

fn hex_digit(nibble: u8, table: &[u8; 16]) -> char {
    char::from(table[usize::from(nibble)])
}

fn unreserved(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
}

/// Percent-encodes every byte outside `[A-Za-z0-9._-]` as uppercase `%XX`, so the result never
/// holds `:` and distinct values stay distinct.
pub(crate) fn enc(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for &byte in value.as_bytes() {
        if unreserved(byte) {
            out.push(char::from(byte));
        } else {
            out.push('%');
            out.push(hex_digit(byte >> 4, HEX_UPPER));
            out.push(hex_digit(byte & 15, HEX_UPPER));
        }
    }
    out
}

pub(crate) fn family_prefix(prefix: &str) -> String {
    format!("{prefix}:")
}

pub(crate) fn namespace_for(prefix: &str, canon: &str) -> String {
    format!("{prefix}:{}", enc(canon))
}

fn dec(encoded: &str) -> Option<String> {
    let bytes = encoded.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let byte = bytes[i];
        if byte == b'%' {
            let high = hex_upper(*bytes.get(i + 1)?)?;
            let low = hex_upper(*bytes.get(i + 2)?)?;
            let escaped = (high << 4) | low;
            if unreserved(escaped) {
                return None;
            }
            out.push(escaped);
            i += 3;
        } else if unreserved(byte) {
            out.push(byte);
            i += 1;
        } else {
            return None;
        }
    }
    String::from_utf8(out).ok()
}

fn hex_upper(digit: u8) -> Option<u8> {
    match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'A'..=b'F' => Some(digit - b'A' + 10),
        _ => None,
    }
}

/// The inverse of [`namespace_for`]; `None` for a namespace outside the family, and for text
/// `enc` would never produce (so two namespaces never decode to one value).
pub(crate) fn parse_value(prefix: &str, namespace: &str) -> Option<String> {
    dec(namespace.strip_prefix(prefix)?.strip_prefix(':')?)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NamespaceError {
    Empty,
    TooLong { bytes: usize },
    NotPrintableAscii,
}

impl fmt::Display for NamespaceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NamespaceError::Empty => write!(f, "namespace is empty"),
            NamespaceError::TooLong { bytes } => write!(
                f,
                "namespace is {bytes} bytes, Pinecone allows at most {MAX_NAMESPACE_BYTES}"
            ),
            NamespaceError::NotPrintableAscii => write!(f, "namespace must be printable ASCII"),
        }
    }
}

/// Whether Pinecone and this repo accept `namespace`: at most 512 bytes of printable ASCII.
pub(crate) fn fits_pinecone(namespace: &str) -> Result<(), NamespaceError> {
    if namespace.is_empty() {
        Err(NamespaceError::Empty)
    } else if namespace.len() > MAX_NAMESPACE_BYTES {
        Err(NamespaceError::TooLong {
            bytes: namespace.len(),
        })
    } else if !valid_namespace(namespace) {
        Err(NamespaceError::NotPrintableAscii)
    } else {
        Ok(())
    }
}

/// What a registration holds in one Pinecone index: a single namespace, or every namespace
/// starting with `prefix:` (a table with a partition column).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Claim<'a> {
    Exact(&'a str),
    Family(&'a str),
}

impl<'a> Claim<'a> {
    pub(crate) fn of(namespace: &'a str, partitioned: bool) -> Self {
        if partitioned {
            Claim::Family(namespace)
        } else {
            Claim::Exact(namespace)
        }
    }

    pub(crate) fn namespace(self) -> &'a str {
        match self {
            Claim::Exact(namespace) | Claim::Family(namespace) => namespace,
        }
    }

    /// Whether two registrations in one index would share a namespace. A family also occupies
    /// its prefix as a registry key, so an exact namespace equal to the prefix clashes too.
    #[allow(dead_code, reason = "the SQL overlap test holds the registries to it")]
    pub(crate) fn conflicts_with(self, other: Claim<'_>) -> bool {
        match (self, other) {
            (Claim::Exact(a), Claim::Exact(b)) => a == b,
            (Claim::Exact(namespace), Claim::Family(prefix))
            | (Claim::Family(prefix), Claim::Exact(namespace)) => {
                namespace == prefix || namespace.starts_with(&family_prefix(prefix))
            }
            (Claim::Family(a), Claim::Family(b)) => {
                let (a, b) = (family_prefix(a), family_prefix(b));
                a.starts_with(&b) || b.starts_with(&a)
            }
        }
    }
}

/// The message of a collision involving a partition family; `owner_table` is the display name of
/// the table that already holds `owner`.
pub(crate) fn clash_message(
    func: &str,
    index: &str,
    claim: Claim<'_>,
    owner_table: &str,
    owner: Claim<'_>,
) -> String {
    let subject = match claim {
        Claim::Exact(namespace) => format!("namespace '{namespace}'"),
        Claim::Family(prefix) => format!("partition family '{prefix}:*'"),
    };
    let relation = match (claim, owner) {
        (Claim::Exact(_), Claim::Family(prefix)) => {
            format!("falls in the partition family '{prefix}:*' of")
        }
        (Claim::Family(_), Claim::Exact(namespace)) => {
            format!("collides with namespace '{namespace}' of")
        }
        (Claim::Family(_), Claim::Family(prefix)) => {
            format!("overlaps the partition family '{prefix}:*' of")
        }
        (Claim::Exact(_), Claim::Exact(_)) => "is already owned by".to_string(),
    };
    format!("{func}: index '{index}' {subject} {relation} {owner_table}")
}

pub(crate) const CLASH_HINT: &str = "a partitioned table reserves every namespace that starts \
     with its prefix and ':'; pass a different index or namespace";

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    const PREFIX: &str = "public.docs";

    #[test]
    fn kinds_round_trip_through_their_registry_text() {
        let texts: Vec<_> = PartitionKind::ALL.iter().map(|k| k.as_str()).collect();
        assert_eq!(texts, ["int2", "int4", "int8", "text", "uuid"]);
        for kind in PartitionKind::ALL {
            assert_eq!(PartitionKind::parse(kind.as_str()), Some(kind));
        }
        for bad in ["", "INT4", "varchar", "bigint", "uuid ", "float8"] {
            assert_eq!(PartitionKind::parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn typnames_map_to_kinds_and_everything_else_is_unsupported() {
        let cases = [
            ("int2", Some(PartitionKind::Int2)),
            ("int4", Some(PartitionKind::Int4)),
            ("int8", Some(PartitionKind::Int8)),
            ("text", Some(PartitionKind::Text)),
            ("varchar", Some(PartitionKind::Text)),
            ("uuid", Some(PartitionKind::Uuid)),
            ("bpchar", None),
            ("numeric", None),
            ("float8", None),
            ("bool", None),
            ("timestamptz", None),
            ("citext", None),
            ("_int4", None),
        ];
        for (typname, kind) in cases {
            assert_eq!(PartitionKind::from_typname(typname), kind, "{typname}");
        }
    }

    #[test]
    fn only_what_canon_writes_is_canonical_for_a_kind() {
        let accepts = [
            (PartitionKind::Int2, "-32768"),
            (PartitionKind::Int2, "0"),
            (PartitionKind::Int4, "2147483647"),
            (PartitionKind::Int8, "-9223372036854775808"),
            (PartitionKind::Uuid, "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"),
            (PartitionKind::Text, ""),
            (PartitionKind::Text, "Acme Corp"),
        ];
        for (kind, text) in accepts {
            assert!(kind.is_canonical(text), "{kind:?} {text:?}");
        }
        let refuses = [
            (PartitionKind::Int2, "32768"),
            (PartitionKind::Int4, "007"),
            (PartitionKind::Int4, "+7"),
            (PartitionKind::Int8, "abc"),
            (PartitionKind::Int8, ""),
            (PartitionKind::Int8, "7 "),
            (PartitionKind::Int8, "-0"),
            (PartitionKind::Uuid, "A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11"),
            (PartitionKind::Uuid, "a0eebc999c0b4ef8bb6d6bb9bd380a11"),
            (PartitionKind::Uuid, "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a1"),
            (PartitionKind::Uuid, "g0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"),
        ];
        for (kind, text) in refuses {
            assert!(!kind.is_canonical(text), "{kind:?} {text:?}");
        }
    }

    #[test]
    fn what_canon_writes_is_canonical_for_its_kind() {
        let ints = [0, 1, -1, 42, i64::from(i16::MAX), i64::MIN, i64::MAX];
        for n in ints {
            assert!(PartitionKind::Int8.is_canonical(&canon(PartitionValue::Int(n))));
        }
        let uuid = canon(PartitionValue::Uuid([0xAB; 16]));
        assert!(PartitionKind::Uuid.is_canonical(&uuid), "{uuid}");
    }

    #[test]
    fn spec_columns_are_both_set_or_both_null() {
        let text = |s: &str| Some(s.to_string());
        assert_eq!(
            PartitionSpec::from_columns(text("tenant_id"), text("int8")),
            Ok(Some(PartitionSpec {
                column: "tenant_id".to_string(),
                kind: PartitionKind::Int8
            }))
        );
        assert_eq!(PartitionSpec::from_columns(None, None), Ok(None));
        for (column, kind) in [
            (text("tenant_id"), None),
            (None, text("int8")),
            (text("tenant_id"), text("bogus")),
        ] {
            assert!(PartitionSpec::from_columns(column, kind).is_err());
        }
    }

    #[test]
    fn lanes_round_trip_through_their_registry_text() {
        assert_eq!(Lane::Vector.as_str(), "vector");
        assert_eq!(Lane::Docs.as_str(), "docs");
        for lane in [Lane::Vector, Lane::Docs] {
            assert_eq!(Lane::parse(lane.as_str()), Some(lane));
        }
        assert_eq!(Lane::parse("fts"), None);
    }

    #[test]
    fn canon_writes_ints_in_decimal() {
        for (value, text) in [
            (0, "0"),
            (42, "42"),
            (-1, "-1"),
            (i64::from(i16::MIN), "-32768"),
            (i64::from(i32::MAX), "2147483647"),
            (i64::MIN, "-9223372036854775808"),
            (i64::MAX, "9223372036854775807"),
        ] {
            assert_eq!(canon(PartitionValue::Int(value)), text);
        }
    }

    #[test]
    fn canon_writes_uuids_lowercase_and_hyphenated() {
        let bytes = [
            0xA0, 0xEE, 0xBC, 0x99, 0x9C, 0x0B, 0x4E, 0xF8, 0xBB, 0x6D, 0x6B, 0xB9, 0xBD, 0x38,
            0x0A, 0x11,
        ];
        assert_eq!(
            canon(PartitionValue::Uuid(bytes)),
            "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11"
        );
        assert_eq!(
            canon(PartitionValue::Uuid([0; 16])),
            "00000000-0000-0000-0000-000000000000"
        );
    }

    #[test]
    fn canon_keeps_text_raw() {
        for text in ["", "acme", "Acme Corp", "a:b%c", "é日😀", "__default__"] {
            assert_eq!(canon(PartitionValue::Text(text.to_string())), text);
        }
    }

    #[test]
    fn enc_keeps_unreserved_bytes_and_escapes_the_rest_in_uppercase_hex() {
        for (value, encoded) in [
            ("", ""),
            ("acme", "acme"),
            ("Az09._-", "Az09._-"),
            ("__default__", "__default__"),
            (":", "%3A"),
            ("%", "%25"),
            ("%41", "%2541"),
            (" ", "%20"),
            ("a/b", "a%2Fb"),
            ("é", "%C3%A9"),
            ("日", "%E6%97%A5"),
            ("😀", "%F0%9F%98%80"),
            ("\u{7f}", "%7F"),
            ("\n", "%0A"),
            ("\0", "%00"),
        ] {
            assert_eq!(enc(value), encoded, "{value:?}");
        }
    }

    #[test]
    fn namespaces_are_prefix_colon_encoded_value() {
        assert_eq!(family_prefix(PREFIX), "public.docs:");
        assert_eq!(namespace_for(PREFIX, "42"), "public.docs:42");
        assert_eq!(namespace_for(PREFIX, ""), "public.docs:");
        assert_eq!(namespace_for(PREFIX, "a:b"), "public.docs:a%3Ab");
        assert_eq!(
            namespace_for(PREFIX, "__default__"),
            "public.docs:__default__"
        );
        assert_eq!(namespace_for("a:b", "1"), "a:b:1");
    }

    #[test]
    fn parse_value_inverts_namespace_for() {
        for value in [
            "",
            "42",
            "acme",
            "a:b",
            "100%",
            "é日😀",
            "%41",
            "__default__",
            " ",
        ] {
            let namespace = namespace_for(PREFIX, value);
            assert_eq!(parse_value(PREFIX, &namespace), Some(value.to_string()));
        }
    }

    #[test]
    fn parse_value_is_none_outside_the_family() {
        for namespace in [
            "public.docs",
            "public.docsx:1",
            "public.doc:1",
            "other:1",
            "public.docs;1",
            "Public.docs:1",
            "x:public.docs:1",
            "",
            "__default__",
        ] {
            assert_eq!(parse_value(PREFIX, namespace), None, "{namespace:?}");
        }
        assert_eq!(parse_value(PREFIX, "public.docs:"), Some(String::new()));
    }

    #[test]
    fn parse_value_rejects_text_that_enc_never_produces() {
        for tail in [
            "a b", "a:b", "%", "%4", "%G0", "%zz", "%41", "%2d", "%5f", "%7e", "%c3%a9", "%FF",
            "%C3", "é", "a+b", "a/b", "%0", "%%",
        ] {
            let namespace = format!("{PREFIX}:{tail}");
            assert_eq!(parse_value(PREFIX, &namespace), None, "{namespace:?}");
        }
    }

    #[test]
    fn parse_value_handles_prefixes_with_colons() {
        assert_eq!(parse_value("a:b", "a:b:1"), Some("1".to_string()));
        assert_eq!(parse_value("a", "a:b:1"), None);
        assert_eq!(parse_value("a:b", "a:1"), None);
    }

    #[test]
    fn fits_pinecone_allows_up_to_512_printable_ascii_bytes() {
        assert_eq!(fits_pinecone("public.docs:42"), Ok(()));
        assert_eq!(fits_pinecone(&"x".repeat(512)), Ok(()));
        assert_eq!(
            fits_pinecone(&"x".repeat(513)),
            Err(NamespaceError::TooLong { bytes: 513 })
        );
        assert_eq!(fits_pinecone(""), Err(NamespaceError::Empty));
        for namespace in ["a\u{7f}b", "a\nb", "é", "a\0b", "日本"] {
            assert_eq!(
                fits_pinecone(namespace),
                Err(NamespaceError::NotPrintableAscii),
                "{namespace:?}"
            );
        }
        assert_eq!(fits_pinecone("a b~"), Ok(()));
    }

    #[test]
    fn fits_pinecone_counts_bytes_not_characters() {
        let long = "é".repeat(257);
        assert_eq!(
            fits_pinecone(&long),
            Err(NamespaceError::TooLong { bytes: 514 })
        );
    }

    #[test]
    fn namespace_errors_say_what_is_wrong() {
        assert!(NamespaceError::Empty.to_string().contains("empty"));
        let long = NamespaceError::TooLong { bytes: 600 }.to_string();
        assert!(long.contains("600") && long.contains("512"), "{long}");
        assert!(
            NamespaceError::NotPrintableAscii
                .to_string()
                .contains("ASCII")
        );
    }

    fn corpus() -> Vec<String> {
        let mut values: Vec<String> = [
            "",
            ":",
            "::",
            ":a",
            "a:",
            "%",
            "%%",
            "%41",
            "%2541",
            "%3A",
            "A",
            "a",
            "a b",
            " ",
            "/",
            "\\",
            "'",
            "\"",
            "é",
            "É",
            "e\u{301}",
            "日本語",
            "😀",
            "__default__",
            "default",
            "docs:42",
            "42",
            "042",
            "-1",
            ".",
            "..",
            "_",
            "-",
            "a:b",
            "a%3Ab",
            "\u{7f}",
            "\n",
            "\0",
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        values.push("x".repeat(500));
        values.push(format!("{}b", "a".repeat(499)));
        values.push(format!("{}c", "a".repeat(499)));
        values.push("é".repeat(250));
        values.push(":".repeat(500));
        values.push("%".repeat(170));
        values.sort();
        values.dedup();
        values
    }

    fn strings_up_to(alphabet: &[&str], max_len: usize) -> Vec<String> {
        let mut all = vec![String::new()];
        let mut layer = vec![String::new()];
        for _ in 0..max_len {
            layer = layer
                .iter()
                .flat_map(|s| alphabet.iter().map(move |c| format!("{s}{c}")))
                .collect();
            all.extend(layer.iter().cloned());
        }
        all
    }

    fn assert_injective(prefix: &str, values: &[String]) {
        let mut seen: HashMap<String, &String> = HashMap::new();
        for value in values {
            let namespace = namespace_for(prefix, value);
            if let Some(other) = seen.insert(namespace.clone(), value) {
                panic!("{other:?} and {value:?} both map to {namespace:?}");
            }
            assert_eq!(parse_value(prefix, &namespace).as_deref(), Some(&value[..]));
        }
    }

    #[test]
    fn namespaces_are_injective_over_a_hostile_corpus() {
        let values = corpus();
        assert!(values.len() > 40);
        assert_injective(PREFIX, &values);
        assert_injective("a:b", &values);
        assert_injective("", &values);
    }

    #[test]
    fn namespaces_are_injective_over_every_short_string_of_a_tricky_alphabet() {
        let alphabet = ["a", "A", "4", "1", "%", ":", "_", ".", "-", " ", "é", "日"];
        let values = strings_up_to(&alphabet, 4);
        assert!(values.len() > 20_000);
        assert_injective(PREFIX, &values);
    }

    #[test]
    fn only_canonical_text_decodes_so_two_namespaces_never_share_a_value() {
        let alphabet = ["%", "4", "1", "A", "a", "f", "F", "g", ":", " ", "-"];
        let mut decoded = 0;
        for tail in strings_up_to(&alphabet, 5) {
            let namespace = format!("{PREFIX}:{tail}");
            if let Some(value) = parse_value(PREFIX, &namespace) {
                assert_eq!(namespace_for(PREFIX, &value), namespace);
                decoded += 1;
            }
        }
        assert!(decoded > 1_000, "decoded {decoded}");
    }

    #[test]
    fn a_500_byte_value_fits_only_with_a_short_prefix_and_plain_bytes() {
        let plain = "x".repeat(500);
        assert_eq!(fits_pinecone(&namespace_for("public.docs", &plain)), Ok(()));
        let colons = ":".repeat(500);
        assert!(matches!(
            fits_pinecone(&namespace_for("public.docs", &colons)),
            Err(NamespaceError::TooLong { bytes: 1512 })
        ));
        let edge = "x".repeat(512 - "public.docs:".len());
        assert_eq!(fits_pinecone(&namespace_for("public.docs", &edge)), Ok(()));
        assert!(fits_pinecone(&namespace_for("public.docs", &format!("{edge}x"))).is_err());
    }

    #[test]
    fn every_encoded_namespace_is_printable_ascii_when_the_prefix_is() {
        for value in corpus().iter().filter(|v| v.len() <= 100) {
            assert_eq!(
                fits_pinecone(&namespace_for(PREFIX, value)),
                Ok(()),
                "{value:?}"
            );
        }
    }

    #[test]
    fn a_namespace_always_has_a_colon_so_it_is_never_the_default_namespace() {
        for value in corpus() {
            let namespace = namespace_for(PREFIX, &value);
            assert!(namespace.contains(':'), "{namespace:?}");
            assert_ne!(namespace, "__default__");
        }
    }

    fn claims() -> Vec<Claim<'static>> {
        let names = ["a", "a:1", "a:1:2", "a:", "ab", "b", "b:a", "a:b", "", ":"];
        names
            .into_iter()
            .flat_map(|n| [Claim::Exact(n), Claim::Family(n)])
            .collect()
    }

    #[test]
    fn exact_claims_clash_only_when_equal() {
        assert!(Claim::Exact("a").conflicts_with(Claim::Exact("a")));
        for other in ["b", "a:1", "ab", "A", ""] {
            assert!(
                !Claim::Exact("a").conflicts_with(Claim::Exact(other)),
                "{other}"
            );
        }
    }

    #[test]
    fn an_exact_claim_clashes_with_the_family_it_falls_into() {
        let family = Claim::Family("public.docs");
        for inside in [
            "public.docs:42",
            "public.docs:",
            "public.docs:a%3Ab",
            "public.docs:a:b",
            "public.docs",
        ] {
            assert!(Claim::Exact(inside).conflicts_with(family), "{inside}");
            assert!(family.conflicts_with(Claim::Exact(inside)), "{inside}");
        }
        for outside in [
            "public.docs2",
            "public.docs2:1",
            "public.doc:1",
            "other:public.docs:1",
            "Public.docs:1",
            "public.do",
        ] {
            assert!(!Claim::Exact(outside).conflicts_with(family), "{outside}");
        }
    }

    #[test]
    fn families_clash_when_their_reservations_overlap() {
        let family = Claim::Family("a");
        for other in ["a", "a:b", "a:b:c", "a:"] {
            assert!(family.conflicts_with(Claim::Family(other)), "{other}");
            assert!(Claim::Family(other).conflicts_with(family), "{other}");
        }
        for other in ["ab", "b", "", "b:a", "A"] {
            assert!(!family.conflicts_with(Claim::Family(other)), "{other}");
        }
    }

    #[test]
    fn clashes_are_symmetric_and_reflexive() {
        for left in claims() {
            assert!(left.conflicts_with(left), "{left:?}");
            for right in claims() {
                assert_eq!(
                    left.conflicts_with(right),
                    right.conflicts_with(left),
                    "{left:?} {right:?}"
                );
            }
        }
    }

    #[test]
    fn every_namespace_a_family_can_write_falls_in_its_reservation() {
        for value in corpus() {
            let namespace = namespace_for(PREFIX, &value);
            assert!(Claim::Exact(&namespace).conflicts_with(Claim::Family(PREFIX)));
            assert!(!Claim::Exact(&namespace).conflicts_with(Claim::Family("public.doc")));
        }
    }

    #[test]
    fn claims_know_their_namespace_and_how_they_were_registered() {
        assert_eq!(Claim::of("a", false), Claim::Exact("a"));
        assert_eq!(Claim::of("a", true), Claim::Family("a"));
        assert_eq!(Claim::Exact("x").namespace(), "x");
        assert_eq!(Claim::Family("y").namespace(), "y");
    }

    #[test]
    fn clash_messages_name_both_sides_and_the_family_wildcard() {
        let exact_in_family = clash_message(
            "pinecone.promote",
            "pv-public-t",
            Claim::Exact("public.t:7"),
            "public.t",
            Claim::Family("public.t"),
        );
        assert_eq!(
            exact_in_family,
            "pinecone.promote: index 'pv-public-t' namespace 'public.t:7' falls in the \
             partition family 'public.t:*' of public.t"
        );
        let family_over_exact = clash_message(
            "pinecone.create_index",
            "idx",
            Claim::Family("public.t"),
            "public.\"t:7\"",
            Claim::Exact("public.t:7"),
        );
        assert_eq!(
            family_over_exact,
            "pinecone.create_index: index 'idx' partition family 'public.t:*' collides with \
             namespace 'public.t:7' of public.\"t:7\""
        );
        let family_over_family = clash_message(
            "pinecone.track_docs",
            "idx",
            Claim::Family("a"),
            "public.b",
            Claim::Family("a:b"),
        );
        assert_eq!(
            family_over_family,
            "pinecone.track_docs: index 'idx' partition family 'a:*' overlaps the partition \
             family 'a:b:*' of public.b"
        );
        let exact_exact = clash_message("f", "i", Claim::Exact("n"), "public.t", Claim::Exact("n"));
        assert_eq!(
            exact_exact,
            "f: index 'i' namespace 'n' is already owned by public.t"
        );
        assert!(CLASH_HINT.contains("different index or namespace"));
    }
}
