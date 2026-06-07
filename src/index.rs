//! [`HnswIndex`] — the columnar storage and graph state that
//! implements [`iqdb_index::IndexCore`] and [`iqdb_index::Index`].
//!
//! The algorithm logic (SEARCH-LAYER, k-NN, INSERT-NODE, the
//! heuristic) lives in [`crate::search`] and [`crate::insert`]; this
//! module owns the data those algorithms read and mutate, plus the
//! trait-impl shims that route the public `insert`/`delete`/`search`
//! calls into them.
//!
//! ## Storage shape
//!
//! Five parallel "columnar" `Vec`s indexed by [`NodeIdx`] (a `u32`):
//! `vectors`, `ids`, `metadata`, `seqs`, `tombstoned`, `node_layer`.
//! Plus `layers: Vec<Vec<Vec<NodeIdx>>>` indexed by `[node][layer]`
//! holding the adjacency list at each layer this node sits on, and
//! `id_to_node: HashMap<VectorId, NodeIdx>` for `O(1)` duplicate
//! detection and delete lookup.
//!
//! ## Audit M1 — shared `Arc<[f32]>`
//!
//! `vectors: Vec<Arc<[f32]>>` stores the caller's `Arc` verbatim;
//! `insert` never allocates a fresh `[f32]`. The engine's record
//! store and this index share the same allocation through that
//! `Arc`, same contract `FlatIndex` honours.
//!
//! ## Tombstone semantics
//!
//! `delete(id)` removes `id` from `id_to_node` and sets
//! `tombstoned[idx] = true`. `live_count` decrements; `len()`
//! reports `live_count`. The columnar row and the graph edges are
//! NOT freed; search treats tombstoned nodes as traversal-only
//! (their out-edges remain valid neighbours) but never returns
//! them as Hits. Memory is reclaimed only when the index is
//! dropped.

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;

use iqdb_index::{Index, IndexCore, IndexStats};
use iqdb_types::{DistanceMetric, Hit, IqdbError, Metadata, Result, SearchParams, VectorId};

use crate::config::HnswConfig;
use crate::graph::NodeIdx;
use crate::rng::SplitMix64;
use crate::{insert as insert_algo, search as search_algo};

/// Hierarchical Navigable Small World approximate nearest-neighbour
/// index. Implements both [`iqdb_index::IndexCore`] (object-safe) and
/// [`iqdb_index::Index`] (typed construction).
///
/// See the crate-level docs for the recall/latency operating point
/// and the [`HnswConfig`] field-level docs for tuning.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use iqdb_hnsw::{HnswConfig, HnswIndex};
/// use iqdb_index::{Index, IndexCore};
/// use iqdb_types::{DistanceMetric, SearchParams, VectorId};
///
/// # fn main() -> iqdb_types::Result<()> {
/// let mut idx = HnswIndex::new(2, DistanceMetric::Euclidean, HnswConfig::default())?;
/// idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0][..]), None)?;
/// idx.insert(VectorId::from(2u64), Arc::<[f32]>::from(&[3.0, 4.0][..]), None)?;
///
/// let hits = idx.search(&[0.0, 0.0], &SearchParams::new(1, DistanceMetric::Euclidean))?;
/// assert_eq!(hits.len(), 1);
/// assert_eq!(hits[0].id, VectorId::U64(1));
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct HnswIndex {
    pub(crate) dim: usize,
    pub(crate) metric: DistanceMetric,
    pub(crate) cfg: HnswConfig,
    /// Precomputed `1 / ln(m)` for the layer-assignment draw.
    pub(crate) m_l_inv: f64,

    /// Row payload, indexed by [`NodeIdx`]. Stores the caller's `Arc`
    /// verbatim (audit M1).
    pub(crate) vectors: Vec<Arc<[f32]>>,
    /// Parallel to `vectors`: the id the caller gave each row.
    pub(crate) ids: Vec<VectorId>,
    /// Parallel to `vectors`: the optional metadata each row.
    pub(crate) metadata: Vec<Option<Metadata>>,
    /// Monotonic insertion-sequence number per row, parallel to
    /// `vectors`. Top-`k` selection tie-breaks on this — *not* on the
    /// row's position — so reordering inside the graph's adjacency
    /// updates does not change query results.
    pub(crate) seqs: Vec<u64>,
    /// `true` when the corresponding row has been deleted. Tombstoned
    /// rows stay in graph traversal but are never returned as Hits.
    pub(crate) tombstoned: Vec<bool>,
    /// Highest layer this node sits on; the node's adjacency lists
    /// have length `node_layer[i] as usize + 1`.
    pub(crate) node_layer: Vec<u8>,

    /// Per-node adjacency: `layers[node][layer]` is the list of
    /// neighbour [`NodeIdx`]s at that layer. Capped by
    /// [`crate::graph::cap_at_layer`].
    pub(crate) layers: Vec<Vec<Vec<NodeIdx>>>,

    /// Live `id → NodeIdx`. Maintained on insert (push) and delete
    /// (remove). Iteration is forbidden on any result-affecting path.
    pub(crate) id_to_node: HashMap<VectorId, NodeIdx>,

    /// Global graph entry point. `None` until the first insert.
    pub(crate) entry: Option<NodeIdx>,
    /// Highest layer any node currently sits on.
    pub(crate) top_layer: u8,

    /// Next monotonic sequence number to assign on insert.
    pub(crate) next_seq: u64,
    /// Layer-assignment PRNG. Single-writer-internal, so a plain
    /// owned generator is enough — no `Mutex`.
    pub(crate) rng: SplitMix64,
    /// Count of non-tombstoned rows. Equals `ids.len() - tombstoned_count`.
    pub(crate) live_count: usize,
}

impl HnswIndex {
    /// Build an empty index for `dim`-component vectors compared
    /// under `metric`, using the supplied `cfg`.
    ///
    /// Returns [`IqdbError::InvalidConfig`] on `dim == 0`, `m == 0`,
    /// `ef_construction < m`, `ef_search == 0`, or
    /// `filter_widen == 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use iqdb_hnsw::{HnswConfig, HnswIndex};
    /// use iqdb_types::DistanceMetric;
    ///
    /// let idx = HnswIndex::new_unconfigured(3, DistanceMetric::Cosine, HnswConfig::default()).unwrap();
    /// assert_eq!(idx.dim(), 3);
    /// assert!(idx.is_empty());
    /// ```
    pub fn new_unconfigured(dim: usize, metric: DistanceMetric, cfg: HnswConfig) -> Result<Self> {
        if dim == 0 {
            return Err(IqdbError::InvalidConfig {
                reason: "HnswIndex dim must be greater than zero",
            });
        }
        if cfg.m == 0 {
            return Err(IqdbError::InvalidConfig {
                reason: "HnswConfig.m must be greater than zero",
            });
        }
        if cfg.ef_construction < cfg.m {
            return Err(IqdbError::InvalidConfig {
                reason: "HnswConfig.ef_construction must be >= m",
            });
        }
        if cfg.ef_search == 0 {
            return Err(IqdbError::InvalidConfig {
                reason: "HnswConfig.ef_search must be greater than zero",
            });
        }
        if cfg.filter_widen == 0 {
            return Err(IqdbError::InvalidConfig {
                reason: "HnswConfig.filter_widen must be greater than zero",
            });
        }
        let m_l_inv = 1.0_f64 / (cfg.m as f64).ln();
        Ok(Self {
            dim,
            metric,
            cfg,
            m_l_inv,
            vectors: Vec::new(),
            ids: Vec::new(),
            metadata: Vec::new(),
            seqs: Vec::new(),
            tombstoned: Vec::new(),
            node_layer: Vec::new(),
            layers: Vec::new(),
            id_to_node: HashMap::new(),
            entry: None,
            top_layer: 0,
            next_seq: 0,
            rng: SplitMix64::new(cfg.seed),
            live_count: 0,
        })
    }

    /// The dimensionality the index was built for.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The distance metric the index was built for.
    #[must_use]
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// The number of searchable (non-tombstoned) vectors in the
    /// index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live_count
    }

    /// Returns `true` when the index holds no searchable vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// Returns the current [`HnswConfig`] (a `Copy` snapshot).
    #[must_use]
    pub fn config(&self) -> HnswConfig {
        self.cfg
    }

    /// Histogram of assigned top layers across every node ever
    /// inserted into this index, including tombstoned ones.
    ///
    /// The returned `Vec` is indexed by layer: `out[L]` is the count
    /// of nodes whose top layer is exactly `L`. Length equals
    /// `top_layer + 1` if the index has any nodes, or `0` if empty.
    ///
    /// At default `HnswConfig` and a non-trivial corpus the histogram
    /// should decay geometrically with rate `1 / m`. This is exposed
    /// for diagnostics and integration testing of the determinism
    /// contract; the data it reflects is internal and stable only
    /// within a given seed.
    #[must_use]
    pub fn node_layer_histogram(&self) -> Vec<usize> {
        if self.node_layer.is_empty() {
            return Vec::new();
        }
        let max_layer = self.node_layer.iter().copied().max().unwrap_or(0);
        let mut out = vec![0_usize; (max_layer as usize) + 1];
        for &l in &self.node_layer {
            out[l as usize] = out[l as usize].saturating_add(1);
        }
        out
    }

    /// Query the index with an explicit beam width, bypassing the stored
    /// [`HnswConfig::ef_search`].
    ///
    /// This is `#[doc(hidden)]` and not part of the stable public API.
    /// Intended for recall-curve diagnostics and benchmarks only.
    /// Does not apply the `filter_widen` multiplier beyond what [`search`]
    /// would for the same `ef` override.
    ///
    /// [`search`]: iqdb_index::IndexCore::search
    #[doc(hidden)]
    pub fn search_with_ef(
        &self,
        query: &[f32],
        params: &SearchParams,
        ef: usize,
    ) -> Result<Vec<Hit>> {
        search_algo::search_with_ef(self, query, params, ef)
    }

    pub(crate) fn check_dim(&self, vector_len: usize) -> Result<()> {
        if vector_len != self.dim {
            return Err(IqdbError::DimensionMismatch {
                expected: self.dim,
                found: vector_len,
            });
        }
        Ok(())
    }

    /// Approximate resident footprint of the index, in bytes.
    fn approximate_memory_bytes(&self) -> usize {
        let arc_header_bytes = 2 * size_of::<usize>();
        let vectors_bytes = self
            .vectors
            .iter()
            .map(|arc| arc.len() * size_of::<f32>() + arc_header_bytes)
            .sum::<usize>()
            + self.vectors.capacity() * size_of::<Arc<[f32]>>();
        let ids_bytes = self.ids.capacity() * size_of::<VectorId>();
        let metadata_bytes = self.metadata.capacity() * size_of::<Option<Metadata>>();
        let seqs_bytes = self.seqs.capacity() * size_of::<u64>();
        let tombstoned_bytes = self.tombstoned.capacity() * size_of::<bool>();
        let node_layer_bytes = self.node_layer.capacity() * size_of::<u8>();
        let layers_bytes: usize = self
            .layers
            .iter()
            .map(|per_node| {
                size_of::<Vec<Vec<NodeIdx>>>() * per_node.capacity()
                    + per_node
                        .iter()
                        .map(|adj| adj.capacity() * size_of::<NodeIdx>())
                        .sum::<usize>()
            })
            .sum();
        let id_to_node_bytes =
            self.id_to_node.capacity() * (size_of::<VectorId>() + size_of::<NodeIdx>());
        vectors_bytes
            + ids_bytes
            + metadata_bytes
            + seqs_bytes
            + tombstoned_bytes
            + node_layer_bytes
            + layers_bytes
            + id_to_node_bytes
    }
}

impl IndexCore for HnswIndex {
    fn insert(
        &mut self,
        id: VectorId,
        vector: Arc<[f32]>,
        metadata: Option<Metadata>,
    ) -> Result<()> {
        insert_algo::insert_node(self, id, vector, metadata)
    }

    fn delete(&mut self, id: &VectorId) -> Result<()> {
        let node = self.id_to_node.remove(id).ok_or(IqdbError::NotFound)?;
        self.tombstoned[node as usize] = true;
        self.live_count = self.live_count.saturating_sub(1);
        Ok(())
    }

    /// Top-`k` nearest-neighbour search via beam-search descent
    /// through the layered graph.
    ///
    /// Returns [`IqdbError::DimensionMismatch`] if `query.len() != self.dim`,
    /// [`IqdbError::InvalidMetric`] if `params.metric` does not match
    /// the index's, and [`IqdbError::InvalidFilter`] if a supplied
    /// `params.filter` is malformed or exceeds the
    /// [`iqdb_filter::MAX_FILTER_DEPTH`] / [`iqdb_filter::MAX_IN_VALUES`]
    /// caps.
    ///
    /// **Limitation:** when `params.filter.is_some()`, post-filter
    /// may return fewer than `params.k` hits if the filter is highly
    /// selective. The internal beam is widened by
    /// `HnswConfig::filter_widen` to mitigate. See the crate
    /// `README.md` "Filter handling" section.
    fn search(&self, query: &[f32], params: &SearchParams) -> Result<Vec<Hit>> {
        search_algo::search(self, query, params)
    }

    fn len(&self) -> usize {
        HnswIndex::len(self)
    }

    fn is_empty(&self) -> bool {
        HnswIndex::is_empty(self)
    }

    fn dim(&self) -> usize {
        HnswIndex::dim(self)
    }

    fn metric(&self) -> DistanceMetric {
        HnswIndex::metric(self)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn stats(&self) -> IndexStats {
        IndexStats {
            n_vectors: self.live_count,
            memory_bytes: self.approximate_memory_bytes(),
            disk_bytes: None,
            index_type: "hnsw",
            extra: None,
        }
    }
}

impl Index for HnswIndex {
    type Config = HnswConfig;

    fn new(dim: usize, metric: DistanceMetric, config: Self::Config) -> Result<Self> {
        Self::new_unconfigured(dim, metric, config)
    }
}
