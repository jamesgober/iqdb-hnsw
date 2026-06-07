# iqdb-hnsw &mdash; API Reference

> Complete reference for **every** public item in `iqdb-hnsw` as of **v1.0.0**:
> what it is, its parameters and return shape, the contract it carries, and
> worked examples for each use case.
>
> **Status: stable (1.0).** The surface below is implemented, tested, and
> recall-validated against the exact `iqdb-flat` ground truth, and is frozen for
> the 1.x series (recorded in [`dev/ROADMAP.md`](../dev/ROADMAP.md)); only
> additive, non-breaking changes are made until 2.0.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Crate constants](#crate-constants)
  - [`VERSION`](#version)
- [`HnswConfig`](#hnswconfig)
  - [Fields & defaults](#fields--defaults)
  - [Builder overrides](#builder-overrides)
- [`HnswIndex`](#hnswindex)
  - [Construction](#construction)
  - [Accessors](#accessors)
  - [The `IndexCore` operations](#the-indexcore-operations)
  - [Diagnostics](#diagnostics)
- [Result ordering & determinism](#result-ordering--determinism)
- [Filter handling](#filter-handling)
- [Errors](#errors)
- [Feature flags](#feature-flags)
- [Trait implementation matrix](#trait-implementation-matrix)

---

## Overview

`iqdb-hnsw` is the production approximate-nearest-neighbour index of the iQDB
spine: a Hierarchical Navigable Small World graph (Malkov–Yashunin, 2016) that
answers top-`k` queries by beam-search descent through a multi-layer proximity
graph. It implements the [`iqdb_index::Index`](https://docs.rs/iqdb-index) trait,
so it is interchangeable with `iqdb-flat` (exact) and `iqdb-ivf` (clustered)
behind one interface.

```rust
use std::sync::Arc;
use iqdb_hnsw::{HnswConfig, HnswIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

# fn main() -> iqdb_types::Result<()> {
let mut idx = HnswIndex::new(2, DistanceMetric::Euclidean, HnswConfig::default())?;
idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0][..]), None)?;
idx.insert(VectorId::from(2u64), Arc::<[f32]>::from(&[3.0, 4.0][..]), None)?;
idx.insert(VectorId::from(3u64), Arc::<[f32]>::from(&[1.0, 0.0][..]), None)?;

let hits = idx.search(&[0.0, 0.0], &SearchParams::new(2, DistanceMetric::Euclidean))?;
assert_eq!(hits.len(), 2);
assert_eq!(hits[0].id, VectorId::U64(1));
assert_eq!(hits[1].id, VectorId::U64(3));
# Ok(()) }
```

At the default `ef_search = 64`, recall@10 against the exact
[`iqdb_flat::FlatIndex`](https://docs.rs/iqdb-flat) ground truth clears `0.95` on
real data (SIFT-1M, dim=128: recall@10 = 0.9644).

---

## Installation

```toml
[dependencies]
iqdb-hnsw = "1.0"
```

There are no feature flags. HNSW construction is single-writer-internal (the
engine guards each index with an external lock), so there is no parallel toggle,
and persistence is the separate `iqdb-persist` crate's job.

---

## Crate constants

### `VERSION`

```rust
pub const VERSION: &str;
```

The crate's compile-time version (`CARGO_PKG_VERSION`), a `major.minor.patch`
SemVer core.

```rust
let v = iqdb_hnsw::VERSION;
assert_eq!(v.split('.').count(), 3);
```

---

## `HnswConfig`

```rust
pub struct HnswConfig {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub filter_widen: usize,
    pub seed: u64,
}
```

The typed configuration consumed by [`Index::new`](#construction). **Derives:**
`Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Default`.

### Fields & defaults

| Field | Default | Controls |
|---|---|---|
| `m` | `16` | Max neighbours per node above layer 0 (`2 * m` at layer 0). Larger ⇒ higher recall, build cost, and memory. |
| `ef_construction` | `200` | Beam width during insert (Alg 1). Larger ⇒ higher recall at higher build cost; query cost unchanged. Must be `>= m`. |
| `ef_search` | `64` | Beam width during search (Alg 5); effective width is `max(ef_search, k)`. Larger ⇒ higher recall at higher per-query cost. |
| `filter_widen` | `4` | Multiplier on the effective beam width when a query carries a filter, to mitigate post-filter under-return. |
| `seed` | `0xDEADBEEFCAFEF00D` | Seed for the layer-assignment PRNG. Same insert order + seed ⇒ byte-identical graph. |

```rust
use iqdb_hnsw::HnswConfig;

let cfg = HnswConfig::default();
assert_eq!(cfg.m, 16);
assert_eq!(cfg.ef_construction, 200);
assert_eq!(cfg.ef_search, 64);
assert_eq!(cfg.filter_widen, 4);
```

### Builder overrides

```rust
pub fn with_m(self, m: usize) -> Self;
pub fn with_ef_construction(self, ef_construction: usize) -> Self;
pub fn with_ef_search(self, ef_search: usize) -> Self;
pub fn with_filter_widen(self, filter_widen: usize) -> Self;
pub fn with_seed(self, seed: u64) -> Self;
```

Consuming, `#[must_use]` setters that override one field and return the config,
so they compose:

```rust
use iqdb_hnsw::HnswConfig;

let tuned = HnswConfig::default().with_m(32).with_ef_search(128);
assert_eq!(tuned.m, 32);
assert_eq!(tuned.ef_search, 128);
assert_eq!(tuned.ef_construction, 200); // untouched
```

---

## `HnswIndex`

```rust
pub struct HnswIndex { /* private */ }
```

The HNSW index. Implements both [`iqdb_index::IndexCore`] (object-safe) and
[`iqdb_index::Index`] (typed construction). **Derives:** `Debug`. It is
`Send + Sync` (via the `IndexCore` supertrait bound). Storage is columnar:
`Vec<Arc<[f32]>>` row payloads plus parallel `Vec`s for ids, metadata,
insertion-sequence, tombstone flags, and per-node layer, with per-node, per-layer
adjacency lists and an `id → node` map for `O(1)` duplicate/delete lookup.

[`iqdb_index::Index`]: https://docs.rs/iqdb-index/latest/iqdb_index/trait.Index.html
[`iqdb_index::IndexCore`]: https://docs.rs/iqdb-index/latest/iqdb_index/trait.IndexCore.html

### Construction

```rust
// via the trait:
impl Index for HnswIndex { type Config = HnswConfig; fn new(dim, metric, config) -> Result<Self>; }
// or the inherent constructor (same behaviour):
pub fn new_unconfigured(dim: usize, metric: DistanceMetric, cfg: HnswConfig) -> Result<Self>;
```

Build an empty index for `dim`-component vectors compared under `metric`.

**Errors:** [`InvalidConfig`](https://docs.rs/iqdb-types/latest/iqdb_types/enum.IqdbError.html#variant.InvalidConfig)
on `dim == 0`, `m == 0`, `ef_construction < m`, `ef_search == 0`, or
`filter_widen == 0`.

```rust
use iqdb_hnsw::{HnswConfig, HnswIndex};
use iqdb_index::Index;
use iqdb_types::DistanceMetric;

let idx = HnswIndex::new(3, DistanceMetric::Cosine, HnswConfig::default()).unwrap();
assert_eq!(idx.dim(), 3);
assert!(idx.is_empty());
```

### Accessors

```rust
pub fn dim(&self) -> usize;            // configured dimensionality
pub fn metric(&self) -> DistanceMetric; // configured metric
pub fn len(&self) -> usize;            // searchable (non-tombstoned) vectors
pub fn is_empty(&self) -> bool;        // len() == 0
pub fn config(&self) -> HnswConfig;    // a Copy snapshot of the config
```

### The `IndexCore` operations

`HnswIndex` implements the full [`IndexCore`] surface: `insert`, `delete`,
`search`, `insert_batch` / `search_batch` (trait defaults), `len`, `is_empty`,
`dim`, `metric`, `flush` (a no-op — purely in-memory), and `stats`.

- **`insert(id, vector, metadata)`** — adds a node via INSERT-NODE (Alg 1). The
  `Arc<[f32]>` is stored verbatim (zero-copy). Returns
  [`DimensionMismatch`](https://docs.rs/iqdb-types/latest/iqdb_types/enum.IqdbError.html#variant.DimensionMismatch)
  on a length mismatch and
  [`Duplicate`](https://docs.rs/iqdb-types/latest/iqdb_types/enum.IqdbError.html#variant.Duplicate)
  on a repeated id.
- **`delete(id)`** — tombstones the node: removed from results, retained in
  traversal for connectivity. Returns
  [`NotFound`](https://docs.rs/iqdb-types/latest/iqdb_types/enum.IqdbError.html#variant.NotFound)
  if the id is not live. `len()` decrements.
- **`search(query, params)`** — top-`k` beam search; see
  [Result ordering](#result-ordering--determinism) and [Filter handling](#filter-handling).

```rust
# use std::sync::Arc;
# use iqdb_hnsw::{HnswConfig, HnswIndex};
# use iqdb_index::{Index, IndexCore};
# use iqdb_types::{DistanceMetric, SearchParams, VectorId};
# fn main() -> iqdb_types::Result<()> {
let mut idx = HnswIndex::new(2, DistanceMetric::Euclidean, HnswConfig::default())?;
idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0][..]), None)?;
idx.insert(VectorId::from(2u64), Arc::<[f32]>::from(&[9.0, 0.0][..]), None)?;
assert_eq!(idx.len(), 2);

idx.delete(&VectorId::from(2u64))?;
assert_eq!(idx.len(), 1);
let hits = idx.search(&[0.0, 0.0], &SearchParams::new(5, DistanceMetric::Euclidean))?;
assert!(hits.iter().all(|h| h.id != VectorId::U64(2))); // tombstoned, never returned
# Ok(()) }
```

### Diagnostics

```rust
pub fn node_layer_histogram(&self) -> Vec<usize>;
```

A histogram of assigned top layers across every node ever inserted (including
tombstoned). `out[L]` is the count of nodes whose top layer is exactly `L`. At a
non-trivial corpus the histogram decays geometrically with rate `1 / m`. Exposed
for diagnostics and determinism testing; the data it reflects is internal and
stable only within a given seed.

---

## Result ordering & determinism

[`Hit::distance`](https://docs.rs/iqdb-types/latest/iqdb_types/struct.Hit.html) is
**smaller-is-nearer** for all five metrics, and `search` returns hits best-first.
Distance math is delegated to
[`iqdb_distance::compute_batch`](https://docs.rs/iqdb-distance); HNSW never
reimplements a metric. For `DotProduct` (a similarity — larger is more similar)
the raw inner product is **negated at the boundary**, so one ordering invariant
holds across the whole family.

Construction is **deterministic**: the layer-assignment PRNG is a hand-rolled
SplitMix64 seeded from [`HnswConfig::seed`] (no external `rand`). Identical insert
order + identical seed produces a byte-identical graph and identical query
results, on every platform.

---

## Filter handling

When `SearchParams::filter` is set, it is evaluated through
[`iqdb_filter`](https://docs.rs/iqdb-filter) during result collection. HNSW
filtering is **post-filter**: the beam is widened by
[`HnswConfig::filter_widen`] (default `4×`) to keep enough survivors, but a highly
selective filter can still return fewer than `params.k` hits. A malformed filter,
or one exceeding the `iqdb-filter` depth / `IN`-value caps, returns
[`InvalidFilter`](https://docs.rs/iqdb-types/latest/iqdb_types/enum.IqdbError.html#variant.InvalidFilter).

Set the public `SearchParams::filter` field to a [`Filter`](https://docs.rs/iqdb-types)
predicate. Only hits whose stored [`Metadata`](https://docs.rs/iqdb-types) satisfies
it are returned:

```rust
use std::sync::Arc;
use iqdb_hnsw::{HnswConfig, HnswIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_types::{DistanceMetric, Filter, Metadata, SearchParams, Value, VectorId};

# fn main() -> iqdb_types::Result<()> {
let mut idx = HnswIndex::new(2, DistanceMetric::Euclidean, HnswConfig::default())?;

let red: Metadata = [("color".to_string(), Value::String("red".to_string()))]
    .into_iter()
    .collect();
let blue: Metadata = [("color".to_string(), Value::String("blue".to_string()))]
    .into_iter()
    .collect();

idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0][..]), Some(red))?;
idx.insert(VectorId::from(2u64), Arc::<[f32]>::from(&[0.1, 0.0][..]), Some(blue))?;

// Nearest "red" vector, even though id 2 is closer overall.
let params = SearchParams {
    filter: Some(Filter::eq("color", Value::String("red".to_string()))),
    ..SearchParams::new(5, DistanceMetric::Euclidean)
};
let hits = idx.search(&[0.0, 0.0], &params)?;
assert!(hits.iter().all(|h| h.id == VectorId::U64(1)));
# Ok(()) }
```

For a tighter filter, raise [`HnswConfig::filter_widen`] (or `ef_search`) so the
widened beam keeps enough survivors to fill `params.k`.

---

## Errors

Every fallible method returns
[`iqdb_types::Result<T>`](https://docs.rs/iqdb-types/latest/iqdb_types/type.Result.html),
whose error is the shared
[`IqdbError`](https://docs.rs/iqdb-types/latest/iqdb_types/enum.IqdbError.html).

| Variant | Raised when |
|---|---|
| `InvalidConfig { reason }` | `new` got `dim == 0` or an out-of-range `HnswConfig` field. |
| `DimensionMismatch { expected, found }` | A vector or query length ≠ `dim()`. |
| `Duplicate` | `insert` collided with a live id. |
| `NotFound` | `delete` named an id that is not live. |
| `InvalidMetric` | `params.metric` did not match the index's `metric()`. |
| `InvalidFilter` | A query filter was malformed or exceeded the caps. |

`IqdbError` is `#[non_exhaustive]`; a `match` on it must carry a wildcard arm.

---

## Feature flags

`iqdb-hnsw` has **no** feature flags. Construction is single-writer-internal, so
there is no parallel toggle; persistence is the separate `iqdb-persist` crate's
job. The default build is the whole surface.

---

## Trait implementation matrix

| Item | Kind | Object-safe | Key bound |
|---|---|:---:|---|
| `HnswIndex` as [`IndexCore`] | trait impl | ✅ (`Box<dyn IndexCore>`) | `Send + Sync` |
| `HnswIndex` as [`Index`] | trait impl | — (by design) | `Config = HnswConfig` |
| `HnswConfig` | struct | n/a | `Debug + Clone + Copy + PartialEq + Eq + Default` |

---

<sub>Copyright &copy; 2026 <strong>James Gober</strong>.</sub>
