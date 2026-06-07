//! # iqdb-hnsw
//!
//! Hierarchical Navigable Small World approximate nearest-neighbour
//! index for the iQDB vector-database spine, implementing the
//! Malkov–Yashunin (2016) algorithm. [`HnswIndex`] builds a multi-layer
//! proximity graph at insert time and answers top-`k` queries by
//! beam-search descent through the layers.
//!
//! At the default `ef_search = 64`, recall@10 against the exact
//! `iqdb_flat::FlatIndex` oracle clears `0.95` on real data — SIFT-1M
//! (dim=128) measures recall@10 = 0.9644 at this default.
//!
//! ## Design
//!
//! - Storage is `Vec<Arc<[f32]>>` per row plus parallel `Vec`s for
//!   `VectorId`, `Option<Metadata>`, insertion-sequence number,
//!   tombstone flag, and top-layer index. The row payload is wrapped
//!   in `Arc` so the engine shares one allocation between this index
//!   and its record store.
//! - Distance math is delegated to
//!   [`iqdb_distance::compute_batch`] — HNSW never reimplements a
//!   metric. `DotProduct` is negated at the boundary so
//!   [`Hit::distance`] is *smaller-is-nearer* across all five metrics.
//! - The layer-assignment PRNG is a hand-rolled SplitMix64 seeded
//!   from [`HnswConfig::seed`]; no external `rand` dependency.
//!   Identical insert order + identical seed produces a byte-
//!   identical graph and identical query results.
//! - Per-layer adjacency is bounded by [`HnswConfig::m`] (`2 * m` at
//!   layer 0). Insert (Alg 1) selects diverse neighbours via the
//!   Alg 4 heuristic, links bidirectionally, and re-runs the
//!   heuristic on any neighbour that overflows its cap.
//! - Search (Alg 5) greedy-descends from the global entry through
//!   the upper layers with `ef = 1`, then runs SEARCH-LAYER (Alg 2)
//!   at layer 0 with `ef = max(ef_search, k)`.
//! - Delete is tombstone-only. Tombstoned nodes stay in graph
//!   traversal for connectivity but are never returned as `Hit`s.
//! - Filter handling is post-filter via
//!   [`iqdb_filter::FilterEvaluator`]. The beam widens by
//!   [`HnswConfig::filter_widen`] when a filter is present;
//!   selective filters can still under-return.
//!
//! ## Example
//!
//! ```
//! use std::sync::Arc;
//!
//! use iqdb_hnsw::{HnswConfig, HnswIndex};
//! use iqdb_index::{Index, IndexCore};
//! use iqdb_types::{DistanceMetric, SearchParams, VectorId};
//!
//! # fn main() -> iqdb_types::Result<()> {
//! let mut idx = HnswIndex::new(2, DistanceMetric::Euclidean, HnswConfig::default())?;
//! idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0][..]), None)?;
//! idx.insert(VectorId::from(2u64), Arc::<[f32]>::from(&[3.0, 4.0][..]), None)?;
//! idx.insert(VectorId::from(3u64), Arc::<[f32]>::from(&[1.0, 0.0][..]), None)?;
//!
//! let hits = idx.search(&[0.0, 0.0], &SearchParams::new(2, DistanceMetric::Euclidean))?;
//! assert_eq!(hits.len(), 2);
//! assert_eq!(hits[0].id, VectorId::U64(1));
//! assert_eq!(hits[1].id, VectorId::U64(3));
//! # Ok(())
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(warnings)]
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unused_must_use)]
#![deny(unused_results)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::print_stdout)]
#![deny(clippy::print_stderr)]
#![deny(clippy::dbg_macro)]
#![deny(clippy::unreachable)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![forbid(unsafe_code)]

mod config;
mod graph;
mod index;
mod insert;
mod rng;
mod search;
mod topk;

pub use crate::config::HnswConfig;
pub use crate::index::HnswIndex;

// Re-export the `Hit` type that searches return, so callers can drive
// `HnswIndex` without a second `use` line for the result type.
pub use iqdb_types::Hit;

/// The version of this crate, taken from `Cargo.toml` at compile time.
///
/// # Examples
///
/// ```
/// let version = iqdb_hnsw::VERSION;
/// assert_eq!(version.split('.').count(), 3);
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
