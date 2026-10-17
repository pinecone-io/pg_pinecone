//! The native local vector engine: local ambuild /
//! amgettuple / small-N exact bootstrap. Everything
//! under this module runs entirely inside the index relation's own pages —
//! no Pinecone client, no `pinecone.tables` registry row, no outbox — the
//! opposite of `am.rs`'s pre-existing remote-mode path, which this module
//! sits alongside (never replaces): `am.rs` reads a build's metapage
//! ([`metapage::read`]) to decide, per index, which of the two this scan/
//! build/insert/vacuum call should run.
//!
//! Module map:
//! - [`metapage`]: block 0's on-disk format + version-gated read/write.
//! - [`options`]: `WITH (mode=..., ...)` reloptions.
//! - [`metric`]: exact distance math.
//! - [`distance`]: the SIMD f32 kernels (L2, dot, cosine) `metric` calls.
//! - [`rng`]: the deterministic generator behind the rotation, the query rounding and k-means.
//! - [`kmeans`]: sample-based IVF training.
//! - [`pages`]: shared on-disk layout math for the centroid-directory and
//!   list pages (used by `build`, `scan`, and `vacuum`).
//! - [`stats`]: the per-index filter-value sketch feeding the two-mode
//!   filtered scan's cardinality estimate.
//! - [`build`]: `ambuild`'s local-mode path.
//! - [`calibrate`]: the build-time sample that picks the default probe count and rescore pool for a
//!   recall target.
//! - [`queries`]: the table of real queries a build can calibrate on (`calibration_queries`).
//! - [`blockscan`]: the block scan of a list page and the error-bound early stop of the rescore (pure
//!   functions: no Postgres).
//! - [`scan`]: `amgettuple`'s local-mode path (mode-H IVF probe scored by RaBitQ codes, mode-L
//!   filter-first exact scan, and the small-N exact bootstrap).
//! - [`insert`]: `aminsert`'s local-mode path (nearest-centroid append,
//!   flat-mode counter + transition warning, stats-staleness counter).
//! - [`cost`]: mode-aware `amcostestimate` math + the insert-path
//!   pricing notes.
//! - [`rotation`] / [`rabitq`] / [`kernel`]: the RaBitQ quantiser — random orthonormal rotation,
//!   `D`-bit codes with an unbiased distance estimator and error bound, and the block-table
//!   (FastScan-style) and popcount kernels that evaluate it.

pub(crate) mod blockscan;
pub(crate) mod build;
pub(crate) mod calibrate;
pub(crate) mod cost;
pub(crate) mod distance;
pub(crate) mod insert;
pub(crate) mod kernel;
pub(crate) mod kmeans;
pub(crate) mod metapage;
pub(crate) mod metric;
pub(crate) mod options;
pub(crate) mod pages;
pub(crate) mod queries;
pub(crate) mod rabitq;
pub(crate) mod rng;
pub(crate) mod rotation;
pub(crate) mod scan;
pub(crate) mod stats;
#[cfg(any(test, feature = "pg_test"))]
pub(crate) mod testkit;
pub(crate) mod vacuum;
