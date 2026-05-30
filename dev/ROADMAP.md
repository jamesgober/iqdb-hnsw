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

## v0.2.0 -- graph storage + insert + search (single-threaded) (THE HARD PART, NOT DEFERRED)

Exit criteria:
- [ ] Every public item has rustdoc + a runnable example.
- [ ] Core invariants property-tested.

---

## v0.3.0 -- neighbor-selection heuristics + tombstone delete + compaction

Exit criteria:
- [ ] New surface tested and benchmarked where it is a hot path.

---

## v0.4.0 -- filtered traversal via iqdb-filter + feature freeze

Exit criteria:
- [ ] No `todo!`/`unimplemented!`. Feature freeze declared.

---

## v0.5.0 -- recall vs flat on SIFT1M/GIST1M + hnswlib comparison + API freeze

Exit criteria:
- [ ] Public API frozen (recorded here). `cargo audit` + `cargo deny` clean.

---

## v0.6.0 -> v0.9.x -- Alpha / Beta -> RC

- 0.6.x-0.7.x: integrate against real consumers; MINOR-compatible additions only.
- 0.8.x (beta): bug fixes; broader testing; final benchmarks.
- 0.9.x (rc): critical fixes + doc polish.

---

## v1.0.0 -- Stable

- [ ] Definition of Done (DIRECTIVES section 7) satisfied.
- [ ] Public API frozen until 2.0.
- [ ] Release note written; published to crates.io; tag pushed.

---

## Out of scope for 1.0

- Persistence -- `iqdb-persist` wraps it.
- Distributed graph -- reserved distributed phase.
