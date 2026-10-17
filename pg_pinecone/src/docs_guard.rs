//! Subtransactions around SQL whose ERROR must not take the surrounding transaction down: the
//! document lane's (the worker runs a round in one transaction, so an ERROR from one table's rows
//! would abort it for every table) and `pinecone.reindex_native_indexes`'s (one index per
//! subtransaction). [`isolated`] confines an ERROR to its step; [`rolled_back_caught`] runs one read
//! of a table's rows, gives its locks back before the HTTP calls that follow and hands an ERROR
//! back as an answer.

use std::panic::UnwindSafe;

use pgrx::{PgSqlErrorCode, PgTryBuilder, pg_sys, pg_sys::submodules::panic::CaughtError};

/// A Postgres ERROR (or panic) caught at a lane boundary.
#[derive(Debug, Clone)]
pub(crate) struct Caught {
    pub(crate) message: String,
    /// Whether the same step can succeed later without anyone changing anything.
    pub(crate) transient: bool,
    pub(crate) code: PgSqlErrorCode,
}

/// The five-character SQLSTATE of `code`.
pub(crate) fn sqlstate(code: PgSqlErrorCode) -> String {
    let packed = code as i32;
    let digit = |at: u32| (((packed >> (6 * at)) & 0x3F) as u8 + b'0') as char;
    (0..5).map(digit).collect()
}

/// Deadlocks and serialization failures (class 40), exhausted resources (class 53), I/O and
/// system errors (class 58), connection errors (class 08), a lock that was not available (55P03),
/// a cancelled statement (57014) and a server that cannot accept work now (57P03) say nothing
/// about the data.
fn is_transient(sqlstate: &str) -> bool {
    ["08", "40", "53", "58"]
        .iter()
        .any(|class| sqlstate.starts_with(class))
        || matches!(sqlstate, "55P03" | "57014" | "57P03")
}

impl Caught {
    fn new(error: &CaughtError) -> Self {
        let report = match error {
            CaughtError::PostgresError(e)
            | CaughtError::ErrorReport(e)
            | CaughtError::RustPanic { ereport: e, .. } => e,
        };
        Self {
            message: report.message().to_string(),
            transient: is_transient(&sqlstate(report.sql_error_code())),
            code: report.sql_error_code(),
        }
    }

    /// Whether the error says a relation, or the schema it was named in, does not exist.
    pub(crate) fn names_a_missing_relation(&self) -> bool {
        matches!(
            self.code,
            PgSqlErrorCode::ERRCODE_UNDEFINED_TABLE | PgSqlErrorCode::ERRCODE_INVALID_SCHEMA_NAME
        )
    }
}

struct Frame {
    context: pg_sys::MemoryContext,
    owner: pg_sys::ResourceOwner,
}

/// Starts a subtransaction like plpgsql does for a `BEGIN ... EXCEPTION` block.
fn begin() -> Frame {
    unsafe {
        let frame = Frame {
            context: pg_sys::CurrentMemoryContext,
            owner: pg_sys::CurrentResourceOwner,
        };
        pg_sys::BeginInternalSubTransaction(std::ptr::null());
        // The body's allocations belong in the caller's context, not the subtransaction's.
        pg_sys::MemoryContextSwitchTo(frame.context);
        frame
    }
}

fn end(frame: Frame, commit: bool) {
    unsafe {
        if commit {
            pg_sys::ReleaseCurrentSubTransaction();
        } else {
            pg_sys::RollbackAndReleaseCurrentSubTransaction();
        }
        pg_sys::MemoryContextSwitchTo(frame.context);
        pg_sys::CurrentResourceOwner = frame.owner;
    }
}

/// Runs `body` in a subtransaction. An ERROR inside rolls everything `body` wrote back and comes
/// out as `Err`; otherwise the subtransaction commits into the surrounding transaction.
pub(crate) fn isolated<T>(body: impl FnOnce() -> T + UnwindSafe) -> Result<T, Caught> {
    let frame = begin();
    let result = PgTryBuilder::new(|| Ok(body()))
        .catch_others(|caught| Err(Caught::new(&caught)))
        .execute();
    end(frame, result.is_ok());
    result
}

/// Runs `body`, which must only read, in a subtransaction that is always rolled back, which
/// releases the locks it took (a statement's locks otherwise live to the end of the
/// transaction). An ERROR comes out as `Err` after the rollback; nothing is raised.
pub(crate) fn rolled_back_caught<T>(body: impl FnOnce() -> T + UnwindSafe) -> Result<T, Caught> {
    let frame = begin();
    let result = PgTryBuilder::new(|| Ok(body()))
        .catch_others(|caught| Err(Caught::new(&caught)))
        .execute();
    end(frame, false);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_classes_that_say_nothing_about_the_data_are_transient() {
        for transient in [
            "40001", "40P01", "53200", "53300", "55P03", "57014", "57P03", "58030", "58P01",
            "08006",
        ] {
            assert!(is_transient(transient), "{transient}");
        }
        for permanent in [
            "22003", "22P02", "42703", "42P01", "XX000", "23505", "57P01",
        ] {
            assert!(!is_transient(permanent), "{permanent}");
        }
    }

    #[test]
    fn only_a_missing_relation_or_schema_names_a_missing_relation() {
        let caught = |code| Caught {
            message: String::new(),
            transient: false,
            code,
        };
        for missing in [
            PgSqlErrorCode::ERRCODE_UNDEFINED_TABLE,
            PgSqlErrorCode::ERRCODE_INVALID_SCHEMA_NAME,
        ] {
            assert!(caught(missing).names_a_missing_relation(), "{missing:?}");
        }
        for other in [
            PgSqlErrorCode::ERRCODE_LOCK_NOT_AVAILABLE,
            PgSqlErrorCode::ERRCODE_QUERY_CANCELED,
            PgSqlErrorCode::ERRCODE_INSUFFICIENT_PRIVILEGE,
            PgSqlErrorCode::ERRCODE_UNDEFINED_COLUMN,
        ] {
            assert!(!caught(other).names_a_missing_relation(), "{other:?}");
        }
    }

    #[test]
    fn a_sqlstate_unpacks_to_its_five_characters() {
        assert_eq!(
            sqlstate(PgSqlErrorCode::ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
            "22003"
        );
        assert_eq!(
            sqlstate(PgSqlErrorCode::ERRCODE_T_R_DEADLOCK_DETECTED),
            "40P01"
        );
        assert_eq!(sqlstate(PgSqlErrorCode::ERRCODE_INTERNAL_ERROR), "XX000");
    }
}
