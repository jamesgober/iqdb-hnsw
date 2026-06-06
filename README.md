<h1 align="center">
    <img width="99" alt="Rust logo" src="https://raw.githubusercontent.com/jamesgober/rust-collection/72baabd71f00e14aa9184efcb16fa3deddda3a0a/assets/rust-logo.svg">
    <br>
    <b>iqdb-hnsw</b>
    <br>
    <sub><sup>iQDB HNSW INDEX</sup></sub>
</h1>

<div align="center">
    <a href="https://crates.io/crates/iqdb-hnsw"><img alt="Crates.io" src="https://img.shields.io/crates/v/iqdb-hnsw"></a>
    <a href="https://crates.io/crates/iqdb-hnsw"><img alt="Downloads" src="https://img.shields.io/crates/d/iqdb-hnsw?color=%230099ff"></a>
    <a href="https://docs.rs/iqdb-hnsw"><img alt="docs.rs" src="https://img.shields.io/docsrs/iqdb-hnsw"></a>
    <a href="https://github.com/jamesgober/iqdb-hnsw/actions"><img alt="CI" src="https://github.com/jamesgober/iqdb-hnsw/actions/workflows/ci.yml/badge.svg"></a>
    <a href="https://github.com/rust-lang/rfcs/blob/master/text/2495-min-rust-version.md"><img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.87%2B-blue"></a>
</div>

<br>

<div align="left">
    <p>
        <strong>iqdb-hnsw</strong> is the primary approximate nearest-neighbor algorithm for production vector search: sub-millisecond search over millions of vectors. After the types crate, it is the most important crate in the family.
    </p>
    <p>
        Correctness is proven by recall benchmarks against `iqdb-flat`; the goal is to be fast, tunable, and readable enough to follow the algorithm.
    </p>
    <br>
    <hr>
    <p>
        <strong>MSRV is 1.87+</strong> (Rust 2024 edition). Production ANN. Recall-validated against flat. Filtered traversal.
    </p>
    <blockquote>
        <strong>Status: pre-1.0, in active development.</strong> The public API is being designed across the 0.x series and frozen at <code>1.0.0</code>. See <a href="./CHANGELOG.md"><code>CHANGELOG.md</code></a>.
    </blockquote>
</div>

<hr>
<br>

<h2>What it does</h2>

- **HNSW graph** &mdash; hierarchical navigable small-world graph per Malkov & Yashunin
- **Standard knobs** &mdash; M, efConstruction, efSearch, optional fixed seed for reproducibility
- **Insert / search / delete** &mdash; deletion via tombstones with compaction
- **Filtered traversal** &mdash; filter during traversal via iqdb-filter, not just after retrieval
- **Competitive** &mdash; targets within 2x of hnswlib on standard benchmarks


<br>

## Installation

```toml
[dependencies]
iqdb-hnsw = "0.1"
```

<br>

## Status

This is the <code>v0.1.0</code> scaffold: structure, tooling, and quality gates are in place; the implementation lands across the 0.x series per the <a href="./dev/ROADMAP.md"><code>ROADMAP</code></a> and <a href="./docs/API.md"><code>docs/API.md</code></a>.

<hr>
<br>

## Where It Fits

`iqdb-hnsw` is the flagship Phase-3 index. It builds on:

- `iqdb-types` &mdash; core types
- `iqdb-distance` &mdash; distance kernels
- `iqdb-index` &mdash; implements the trait
- `iqdb-filter` &mdash; filter during graph traversal

It is unblocked once index/distance/filter exist; no external dependency.

<br>

## Contributing

See <a href="./dev/DIRECTIVES.md"><code>dev/DIRECTIVES.md</code></a> for engineering standards and the definition of done. Before a PR: `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --all-features` must be clean.

<br>

<div id="license">
    <h2>License</h2>
    <p>Licensed under either of</p>
    <ul>
        <li><b>Apache License, Version 2.0</b> &mdash; <a href="./LICENSE-APACHE">LICENSE-APACHE</a></li>
        <li><b>MIT License</b> &mdash; <a href="./LICENSE-MIT">LICENSE-MIT</a></li>
    </ul>
    <p>at your option.</p>
</div>

<div align="center">
  <h2></h2>
  <sup>COPYRIGHT <small>&copy;</small> 2026 <strong>JAMES GOBER.</strong></sup>
</div>
