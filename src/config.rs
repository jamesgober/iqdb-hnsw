//! [`HnswConfig`] — the typed configuration consumed by
//! [`iqdb_index::Index::new`] for [`crate::HnswIndex`].
//!
//! Defaults are the recall/latency operating point documented in the
//! crate `README.md`: `m = 16`, `ef_construction = 200`,
//! `ef_search = 64`, `filter_widen = 4`, `seed = 0xDEADBEEFCAFEF00D`.
//! Use [`HnswConfig::default`] for the operating point, or the
//! builder-style `with_*` methods to override a single field.

/// Default seed for the layer-assignment PRNG.
const DEFAULT_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

/// Default max neighbours per node above layer 0; layer 0 uses `2 * M`.
const DEFAULT_M: usize = 16;

/// Default beam width during insert (Alg 1 `ef_construction`).
const DEFAULT_EF_CONSTRUCTION: usize = 200;

/// Default beam width during search (Alg 5 `ef_search`).
///
/// The production default is calibrated to real-data recall, not the
/// synthetic worst case: on SIFT-1M (dim=128) `iqdb-hnsw` measures
/// recall@10 = 0.9644 at `ef_search = 64`, clearing the 0.95 floor at
/// roughly half the search cost of `ef_search = 128`. The headline
/// uniform-random recall gate (`tests/recall.rs`) is HNSW's worst case
/// and is exercised at an explicit `ef_search = 128`, decoupled from
/// this default.
const DEFAULT_EF_SEARCH: usize = 64;

/// Default multiplier applied to `ef_search` when a metadata filter
/// is supplied at query time.
const DEFAULT_FILTER_WIDEN: usize = 4;

/// Configuration for [`crate::HnswIndex`] construction (see
/// [`iqdb_index::Index::new`]).
///
/// All fields have documented defaults; see the field-level docs and
/// the crate `README.md` for the tradeoffs each one controls.
///
/// # Examples
///
/// ```
/// use iqdb_hnsw::HnswConfig;
///
/// let cfg = HnswConfig::default();
/// assert_eq!(cfg.m, 16);
/// assert_eq!(cfg.ef_construction, 200);
/// assert_eq!(cfg.ef_search, 64);
/// assert_eq!(cfg.filter_widen, 4);
///
/// let tuned = HnswConfig::default()
///     .with_m(32)
///     .with_ef_search(128);
/// assert_eq!(tuned.m, 32);
/// assert_eq!(tuned.ef_search, 128);
/// assert_eq!(tuned.ef_construction, 200);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HnswConfig {
    /// Max neighbours per node above layer 0; layer 0 uses `2 * m`.
    ///
    /// Larger `m` raises recall, build cost, and memory; smaller `m`
    /// is faster to build but recall degrades. Default `16`.
    pub m: usize,
    /// Beam width during insert (Alg 1 `ef_construction`).
    ///
    /// Larger values raise recall at higher build cost; query cost
    /// is unchanged. Must be `>= m`. Default `200`.
    pub ef_construction: usize,
    /// Beam width during search (Alg 5 `ef_search`).
    ///
    /// Larger values raise recall at higher per-query cost. At
    /// query time the effective beam width is `max(ef_search, k)`.
    /// Default `64` — calibrated to real-data recall: on SIFT-1M
    /// (dim=128) `iqdb-hnsw` measures recall@10 = 0.9644 at
    /// `ef_search = 64`, clearing the 0.95 floor at roughly half the
    /// search cost of `ef_search = 128`. The headline uniform-random
    /// recall gate (`tests/recall.rs`) is HNSW's worst case and is
    /// exercised at an explicit `ef_search = 128`, decoupled from
    /// this default.
    pub ef_search: usize,
    /// Multiplier applied to the effective beam width when a filter
    /// is supplied at query time.
    ///
    /// Mitigates HNSW post-filter under-return: a selective filter
    /// can leave fewer than `k` survivors in a narrow beam. Default
    /// `4`.
    pub filter_widen: usize,
    /// Seed for the internal SplitMix64 PRNG that assigns each new
    /// node's top layer.
    ///
    /// Identical insert order + identical `seed` → byte-identical
    /// graph and identical search results.
    pub seed: u64,
}

impl HnswConfig {
    /// Override `m`.
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Override `ef_construction`.
    #[must_use]
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Override `ef_search`.
    #[must_use]
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    /// Override `filter_widen`.
    #[must_use]
    pub fn with_filter_widen(mut self, filter_widen: usize) -> Self {
        self.filter_widen = filter_widen;
        self
    }

    /// Override the PRNG seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: DEFAULT_M,
            ef_construction: DEFAULT_EF_CONSTRUCTION,
            ef_search: DEFAULT_EF_SEARCH,
            filter_widen: DEFAULT_FILTER_WIDEN,
            seed: DEFAULT_SEED,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_values_are_the_documented_operating_point() {
        let cfg = HnswConfig::default();
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.ef_construction, 200);
        assert_eq!(cfg.ef_search, 64);
        assert_eq!(cfg.filter_widen, 4);
        assert_eq!(cfg.seed, 0xDEAD_BEEF_CAFE_F00D);
    }

    #[test]
    fn with_helpers_compose() {
        let cfg = HnswConfig::default()
            .with_m(8)
            .with_ef_construction(100)
            .with_ef_search(32)
            .with_filter_widen(2)
            .with_seed(42);
        assert_eq!(cfg.m, 8);
        assert_eq!(cfg.ef_construction, 100);
        assert_eq!(cfg.ef_search, 32);
        assert_eq!(cfg.filter_widen, 2);
        assert_eq!(cfg.seed, 42);
    }
}
