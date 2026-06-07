# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

### Changed

### Fixed

### Security

---

## [0.6.0] - 2026-06-06

The HNSW index is **feature-complete, recall-validated, and API-frozen**. This
release lands the implementation on top of the v0.1.0 scaffold — graph storage,
insert, beam search, tombstone delete, and filtered traversal — wired to the
stable (`1.0`) iQDB spine, and commits the public surface for the 1.x series. The
roadmap's v0.2 (graph + insert + search), v0.3 (neighbour heuristic + tombstone),
v0.4 (filtered traversal + feature freeze), and v0.5 (recall + API freeze)
milestones land together here; the release is numbered `0.6.0` to align with the
iQDB family's version line (siblings `iqdb-flat` / `iqdb-build` are already at
`0.5.0`+). See `dev/ROADMAP.md`.

### Added

- **`HnswIndex`** — a Malkov–Yashunin (2016) Hierarchical Navigable Small World
  index implementing `iqdb_index::IndexCore` + `Index` (`type Config = HnswConfig`).
  Insert, delete (tombstone), batch insert, single and batch search, `flush`, and
  `stats`. Inherent `new_unconfigured`, `dim`, `metric`, `len`, `is_empty`,
  `config`, and a `node_layer_histogram` diagnostic.
- **`HnswConfig`** — `m`, `ef_construction`, `ef_search`, `filter_widen`, and
  `seed`, with builder-style `with_*` overrides and documented defaults
  (`m = 16`, `ef_construction = 200`, `ef_search = 64`, `filter_widen = 4`).
- **Graph construction** — columnar storage (`Vec<Arc<[f32]>>` rows + parallel
  `Vec`s + per-node, per-layer adjacency), INSERT-NODE (Alg 1) with the Alg 4
  diverse-neighbour heuristic, bidirectional linking, and overflow re-pruning.
- **Beam search** — greedy descent through the upper layers (`ef = 1`) then
  SEARCH-LAYER (Alg 2) at layer 0 with `ef = max(ef_search, k)`.
- **One ordering invariant** — distance math delegated to
  `iqdb_distance::compute_batch`; `DotProduct` negated at the boundary so
  `Hit.distance` is *smaller-is-nearer* across all five metrics.
- **Determinism** — a hand-rolled SplitMix64 (no `rand` dependency) seeded from
  `HnswConfig::seed`; identical insert order + seed ⇒ byte-identical graph and
  identical results.
- **Tombstone delete** — deleted nodes stay in traversal for connectivity but are
  never returned as hits; `len()` reports the live count.
- **Filtered traversal** — `SearchParams::filter` evaluated via `iqdb-filter`,
  with the beam widened by `filter_widen` to mitigate post-filter under-return.
- **Zero-copy insert** — the caller's `Arc<[f32]>` payload is stored verbatim.
- **`VERSION`** — the crate's compile-time SemVer string.
- **Tests** — unit, contract, determinism, edge, layer-distribution, tombstone,
  and a headline recall@10 ≥ 0.95 gate across all five metrics measured against
  an inline exact full-scan oracle; an `#[ignore]`'d SIFT-1M recall diagnostic.
- **Benchmark** — `criterion` search latency at two corpus scales and three
  `ef_search` widths, with recall@10 reported in each bench name.

### Changed

- **Public API frozen.** The committed surface is recorded in `dev/ROADMAP.md`
  (§ v0.5.0 / v0.6.0); only additive, non-breaking changes are made through 1.x.
  `iqdb_types::IqdbError` is `#[non_exhaustive]`, so new error variants remain
  non-breaking.
- Wired dependencies to the stable iQDB spine: `iqdb-types`, `iqdb-distance`,
  `iqdb-index`, and `iqdb-filter` (all `1.0`).
- Added Matt Callahan to the crate authors.
- Removed the scaffold's `std` / `serde` feature split; the crate has no feature
  flags (single-writer-internal construction; persistence is `iqdb-persist`).

---

## [0.1.0] - 2026-05-30

Initial scaffold and repository bootstrap. No domain logic yet &mdash; this release establishes the structure, tooling, and quality gates the implementation will be built on.

### Added

- `Cargo.toml` with crate metadata, Rust 2024 edition, MSRV 1.87.
- Dual `Apache-2.0 OR MIT` license files.
- `README.md`, `CHANGELOG.md`, and a documentation skeleton.
- `REPS.md` compliance baseline.
- `.github/workflows/ci.yml` CI matrix; `deny.toml`, `clippy.toml`, `rustfmt.toml`.
- `dev/DIRECTIVES.md` and `dev/ROADMAP.md` (committed engineering standards + plan).
[Unreleased]: https://github.com/jamesgober/iqdb-hnsw/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/jamesgober/iqdb-hnsw/compare/v0.1.0...v0.6.0
[0.1.0]: https://github.com/jamesgober/iqdb-hnsw/releases/tag/v0.1.0
