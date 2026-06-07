# iqdb-hnsw -- Roadmap

> Path from scaffold to a stable 1.0. Hard parts are front-loaded; each phase has hard exit criteria.
>
> **Anti-deferral rule:** no listed hard task moves to a later phase unless this file records the move and the reason.

---

## v0.1.0 -- Scaffold (DONE)

Compiles, CI green, structure correct, no domain logic.

- [x] Manifest, README, CHANGELOG, REPS, license, CI, lints in place.
- [x] API surface sketched in `docs/API.md`.

---

## v0.2.0 -- graph storage + insert + search (single-threaded) (THE HARD PART, NOT DEFERRED) (DONE)

Columnar graph storage (`Vec<Arc<[f32]>>` rows + parallel `Vec`s + per-node
adjacency), INSERT-NODE (Alg 1) with the layer-assignment SplitMix64 draw, and
SEARCH (Alg 5 greedy descent + Alg 2 SEARCH-LAYER) all land here.

Exit criteria:
- [x] Every public item has rustdoc + a runnable example.
- [x] Core invariants property-tested.

Landed together with 0.3/0.4/0.5 in the consolidated v0.6.0 release (see CHANGELOG).

---

## v0.3.0 -- neighbor-selection heuristics + tombstone delete (+ compaction DEFERRED) (DONE)

Exit criteria:
- [x] New surface tested and benchmarked where it is a hot path.

The Alg 4 diverse-neighbour heuristic (with bidirectional linking + overflow
re-pruning) and tombstone delete shipped in v0.6.0.

**Deferral (recorded per the anti-deferral rule).** *Slot compaction* (reclaiming
tombstoned rows + graph-repair delete) is deferred to a post-freeze internal
optimisation. Rationale: delete is tombstone-only — a tombstoned node stays in
graph traversal for connectivity but is never returned as a `Hit`, so the
*observable* deletion contract holds without compaction. Reclaiming the slot is a
memory optimisation, internal and non-breaking, that can land in any later
0.x/1.x. It does not block the 0.5 API freeze.

---

## v0.4.0 -- filtered traversal via iqdb-filter + feature freeze (DONE)

Exit criteria:
- [x] No `todo!`/`unimplemented!`. Feature freeze declared.

Metadata filtering via `iqdb-filter` (post-filter, with the beam widened by
`HnswConfig::filter_widen` to mitigate under-return) shipped in v0.6.0.
**Feature freeze is declared:** the feature set — graph build, insert, beam
search, tombstone delete, and filtered traversal — is complete. Only the
compaction optimisation above remains, and it is not a feature-surface change.

---

## v0.5.0 -- recall validation + API freeze (DONE)

Exit criteria:
- [x] Public API frozen (recorded below). `cargo audit` + `cargo deny` clean.

A headline recall@10 ≥ 0.95 gate across all five metrics (`tests/recall.rs`),
measured against an inline exact full-scan oracle, validates the graph at scale;
an `#[ignore]`'d SIFT-1M diagnostic (`tests/sift_recall.rs`) is available for
real-data validation when the dataset is present. An hnswlib comparison is
deferred to the alpha phase as an external benchmark (not an API gate).

### Frozen public API (1.x compatibility surface)

Recorded here at the API freeze. Everything below is committed; only **additive,
non-breaking** changes are made through 1.x. `iqdb_types::IqdbError` is
`#[non_exhaustive]`, so new variants are not breaking.

- **`iqdb_hnsw::VERSION: &str`** — compile-time SemVer string.
- **`iqdb_hnsw::Hit`** — re-export of `iqdb_types::Hit` (the search result type).
- **`HnswConfig`** — fields `m`, `ef_construction`, `ef_search`, `filter_widen`,
  `seed`; builder `with_*` overrides; `derive(Debug, Clone, Copy, PartialEq, Eq,
  Default)`.
- **`HnswIndex`** — `derive(Debug)`; `Send + Sync`. Inherent methods:
  `new_unconfigured`, `dim`, `metric`, `len`, `is_empty`, `config`,
  `node_layer_histogram`.
- **`impl iqdb_index::Index for HnswIndex`** — `type Config = HnswConfig`; `new`.
- **`impl iqdb_index::IndexCore for HnswIndex`** — `insert`, `insert_batch`*,
  `delete`, `search`, `search_batch`*, `len`, `is_empty`, `dim`, `metric`,
  `flush`, `stats` (* = trait default, not overridden).
- **No feature flags.**

Behavioural contracts frozen with the surface: `Hit.distance` is
smaller-is-nearer for all five metrics (`DotProduct` negated); deterministic graph
under a fixed seed + insert order; tombstone delete (a deleted id never reappears
in `search` until re-inserted); no method panics on any input; zero `unsafe`.

---

## v0.6.0 -- consolidated implementation release (alpha entry) (DONE)

The full implementation (roadmap v0.2–v0.5: graph, insert, beam search, neighbour
heuristic, tombstone delete, filtered traversal, recall validation, API freeze)
landed in a single consolidated release tagged **v0.6.0** — the crate goes from
the v0.1.0 scaffold straight to 0.6.0 to align with the iQDB family's version line
(`iqdb-flat` / `iqdb-build` are already at `0.5.0`+). This is the alpha entry point.

---

## v0.7.x -> v0.9.x -- Alpha / Beta -> RC (CONSOLIDATED INTO v1.0.0)

The alpha/beta/RC band collapsed into the `1.0.0` release. With the whole
dependency family published and stable at `1.0` — including the recall-oracle
sibling `iqdb-flat` — the remaining alpha task that actually gated the surface
(recall validation against `iqdb-flat`, not a hand-rolled oracle) was completed
directly, and the feature set has been frozen and unchanged since `0.6.0`. The
hnswlib external comparison stays an out-of-tree diagnostic (`tests/sift_recall.rs`
with the published SIFT ground truth and an hnswlib reference line), never an API
gate, so it does not block the stable tag.

---

## v1.0.0 -- Stable (DONE)

- [x] Definition of Done (DIRECTIVES section 7) satisfied.
- [x] Public API frozen until 2.0 (unchanged from the `0.6.0` freeze).
- [x] Recall validated against the published `iqdb-flat` ground truth
      (DIRECTIVES §8); filtered-traversal path covered end to end.
- [x] Release note written (`docs/release/v1.0.0.md`). Publish + tag are the
      maintainer's step.

---

## Out of scope for 1.0

- Persistence -- `iqdb-persist` wraps it.
- Distributed graph -- reserved distributed phase.
