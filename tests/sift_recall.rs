//! Real-data recall validation: `iqdb-hnsw` recall@10 on the SIFT
//! (TEXMEX) corpus, the canonical published ANN benchmark for HNSW.
//!
//! This is a `#[ignore]`'d diagnostic — it is opt-in, prints a table,
//! and never asserts a pass/fail threshold. The default `cargo test`
//! suite is unaffected.
//!
//! ## Why this exists
//!
//! Synthetic recall tests are inconclusive about whether
//! `iqdb-hnsw` carries an implementation inefficiency vs reference
//! HNSW. Uniform-random rows trigger distance concentration (HNSW's
//! worst case); tight Gaussian blobs produce near-ties that make
//! recall@10 brittle for a different reason. SIFT (128-dim,
//! Euclidean, ~10k–1M vectors, with published per-query 100-NN
//! ground truth) is what every reference HNSW implementation is
//! quoted against.
//!
//! ## How to run
//!
//! 1. Fetch the dataset(s) into a local gitignored directory:
//!
//!    ```sh
//!    mkdir -p iqdb-hnsw/.bench-data
//!    cd iqdb-hnsw/.bench-data
//!    curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
//!    curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
//!    tar -xzf siftsmall.tar.gz
//!    tar -xzf sift.tar.gz
//!    ```
//!
//! 2. Run from the workspace root:
//!
//!    ```sh
//!    cargo test -p iqdb-hnsw --test sift_recall \
//!        sift_recall_siftsmall -- --include-ignored --nocapture
//!    cargo test --release -p iqdb-hnsw --test sift_recall \
//!        sift_recall_sift1m -- --include-ignored --nocapture
//!    ```
//!
//! If the dataset directory is missing, the test prints a `SKIP`
//! line and returns; it does not fail. The `iqdb-hnsw/.bench-data/`
//! path is gitignored at the workspace root.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use iqdb_hnsw::{HnswConfig, HnswIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

const K: usize = 10;
const EF_VALUES: [usize; 3] = [64, 128, 256];

/// Published `hnswlib` recall@10 on SIFT1M at `(M=16, ef_construction=200,
/// ef_search=64)`. Sourced from the ann-benchmarks "sift-128-euclidean"
/// dashboard. Used only for an informational verdict line — never
/// asserted as a threshold.
const HNSWLIB_SIFT1M_REF_RECALL_AT_EF64: f64 = 0.97;

struct DatasetSpec {
    name: &'static str,
    dir: &'static str,
    base: &'static str,
    query: &'static str,
    groundtruth: &'static str,
}

/// Read a `.fvecs` file. Each record on disk is a little-endian
/// `i32 dim`, followed by `dim` × `f32`. Returns one `Vec<f32>` per
/// record.
fn read_fvecs(path: &Path) -> std::io::Result<Vec<Vec<f32>>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut out = Vec::new();
    let mut dim_buf = [0u8; 4];
    loop {
        match r.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;
        let mut payload = vec![0u8; dim * 4];
        r.read_exact(&mut payload)?;
        let row: Vec<f32> = payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.push(row);
    }
    Ok(out)
}

/// Read an `.ivecs` file. Identical layout to `.fvecs`, but the
/// payload is `dim` × `i32` (the per-query ground-truth neighbour
/// IDs; SIFT IDs are always non-negative so `u32` is the natural fit).
fn read_ivecs(path: &Path) -> std::io::Result<Vec<Vec<u32>>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut out = Vec::new();
    let mut dim_buf = [0u8; 4];
    loop {
        match r.read_exact(&mut dim_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;
        let mut payload = vec![0u8; dim * 4];
        r.read_exact(&mut payload)?;
        let row: Vec<u32> = payload
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        out.push(row);
    }
    Ok(out)
}

fn run(spec: &DatasetSpec) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(spec.dir);
    let base_path = root.join(spec.base);
    let query_path = root.join(spec.query);
    let gt_path = root.join(spec.groundtruth);

    if !base_path.exists() || !query_path.exists() || !gt_path.exists() {
        println!(
            "SKIP {}: missing dataset under {}.\n  \
             Fetch with:\n    \
             mkdir -p iqdb-hnsw/.bench-data && cd iqdb-hnsw/.bench-data\n    \
             curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz\n    \
             curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz\n    \
             tar -xzf siftsmall.tar.gz && tar -xzf sift.tar.gz",
            spec.name,
            root.display(),
        );
        return;
    }

    println!("=== {} ===", spec.name);
    let t_load = Instant::now();
    let base = read_fvecs(&base_path).unwrap();
    let query = read_fvecs(&query_path).unwrap();
    let gt = read_ivecs(&gt_path).unwrap();
    println!(
        "loaded base={} query={} gt={} in {:.2}s",
        base.len(),
        query.len(),
        gt.len(),
        t_load.elapsed().as_secs_f64(),
    );
    assert!(!base.is_empty(), "empty base set");
    assert_eq!(query.len(), gt.len(), "query/gt length mismatch");
    let dim = base[0].len();
    assert!(
        base.iter().all(|r| r.len() == dim),
        "ragged dim in base set",
    );
    assert!(
        query.iter().all(|r| r.len() == dim),
        "ragged dim in query set",
    );
    assert!(
        gt.iter().all(|r| r.len() >= K),
        "ground-truth row shorter than K={K}",
    );

    let cfg = HnswConfig::default();
    println!(
        "config: m={} ef_construction={} ef_search_default={} seed={:#x}",
        cfg.m, cfg.ef_construction, cfg.ef_search, cfg.seed,
    );

    let t_build = Instant::now();
    let mut idx = HnswIndex::new(dim, DistanceMetric::Euclidean, cfg).unwrap();
    for (i, row) in base.iter().enumerate() {
        idx.insert(VectorId::from(i as u64), Arc::<[f32]>::from(&row[..]), None)
            .unwrap();
    }
    let build_secs = t_build.elapsed().as_secs_f64();
    println!(
        "built N={} dim={} in {:.2}s ({:.0} ins/s)",
        base.len(),
        dim,
        build_secs,
        base.len() as f64 / build_secs.max(1e-9),
    );

    let params = SearchParams::new(K, DistanceMetric::Euclidean);
    println!();
    println!(
        "{:<10}  {:>3}   {:>10}   {:>14}",
        "dataset", "ef", "recall@10", "mean_us/query",
    );
    println!("{}", "-".repeat(48));

    let mut recall_at_ef64: Option<f64> = None;
    for &ef in &EF_VALUES {
        let t0 = Instant::now();
        let mut overlap_total = 0usize;
        let mut count_total = 0usize;
        for (qi, q) in query.iter().enumerate() {
            let hits = idx.search_with_ef(q, &params, ef).unwrap();
            assert_eq!(
                hits.len(),
                K,
                "expected {K} hits at ef={ef}, got {} for query {qi}",
                hits.len(),
            );
            let truth: HashSet<u64> = gt[qi].iter().take(K).map(|&id| id as u64).collect();
            let overlap = hits
                .iter()
                .filter(|h| matches!(&h.id, VectorId::U64(u) if truth.contains(u)))
                .count();
            overlap_total += overlap;
            count_total += K;
        }
        let mean_us = t0.elapsed().as_micros() as f64 / query.len() as f64;
        let recall = overlap_total as f64 / count_total as f64;
        println!(
            "{:<10}  {:>3}   {:>10.4}   {:>14.1}",
            spec.name, ef, recall, mean_us,
        );
        if ef == 64 {
            recall_at_ef64 = Some(recall);
        }
    }

    if let Some(r) = recall_at_ef64 {
        let delta = r - HNSWLIB_SIFT1M_REF_RECALL_AT_EF64;
        let verdict = if r >= 0.95 {
            "competitive (clears 0.95 floor)"
        } else if delta.abs() <= 0.01 {
            "matches hnswlib reference"
        } else {
            "below hnswlib reference — implementation gap likely"
        };
        println!();
        println!(
            "verdict ({}): iqdb-hnsw recall@10 @ ef=64 = {:.4}",
            spec.name, r,
        );
        println!(
            "  hnswlib SIFT1M reference @ ef=64 (M=16, efC=200) ≈ {:.4}  (ann-benchmarks)",
            HNSWLIB_SIFT1M_REF_RECALL_AT_EF64,
        );
        println!("  delta = {delta:+.4}  → {verdict}");
    }
    println!();
}

/// Fast first pass on siftsmall (10k base / 100 query). Catches harness
/// bugs in seconds before we commit a 1M-vector build.
#[test]
#[ignore]
fn sift_recall_siftsmall() {
    run(&DatasetSpec {
        name: "siftsmall",
        dir: ".bench-data/siftsmall",
        base: "siftsmall_base.fvecs",
        query: "siftsmall_query.fvecs",
        groundtruth: "siftsmall_groundtruth.ivecs",
    });
}

/// Rigorous run on SIFT1M (1M base / 10k query). Run in release mode —
/// debug builds are too slow for the 1M-vector insert phase and skew
/// the per-query latency numbers.
#[test]
#[ignore]
fn sift_recall_sift1m() {
    run(&DatasetSpec {
        name: "sift1m",
        dir: ".bench-data/sift",
        base: "sift_base.fvecs",
        query: "sift_query.fvecs",
        groundtruth: "sift_groundtruth.ivecs",
    });
}
