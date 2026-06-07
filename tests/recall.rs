//! Headline recall test: `iqdb-hnsw` recall@10 vs the exact `iqdb-flat`
//! oracle across every distance metric.
//!
//! The ground truth is [`iqdb_flat::FlatIndex`] — the family's brute-force
//! exact index — built over the identical corpus. Recall therefore measures
//! the *graph*: of the genuine nearest neighbours `FlatIndex` returns by full
//! scan, what fraction does HNSW's beam search recover. Both indexes share the
//! one ordering contract (`Hit.distance` smaller-is-nearer, `DotProduct`
//! negated at the boundary) and the same insertion-order tie-break, so the
//! comparison is apples-to-apples. Validating against the published sibling —
//! not a hand-rolled scan — is the DIRECTIVES §8 mandate.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

use std::collections::HashSet;
use std::sync::Arc;

use iqdb_flat::{FlatConfig, FlatIndex};
use iqdb_hnsw::{HnswConfig, HnswIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

const N: usize = 5_000;
const DIM: usize = 128;
const K: usize = 10;
const QUERIES: usize = 200;
const RECALL_FLOOR: f64 = 0.95;

/// Explicit beam width used by the headline recall gate, decoupled from
/// `HnswConfig::default().ef_search`. The gate's corpus is uniform-random
/// — HNSW's worst case — and recall@10 on uniform-random only clears 0.95
/// at ef >= 128. The production default is calibrated to real-data
/// (SIFT-1M, recall@10 = 0.9644 at ef=64); the gate keeps a real 0.95
/// floor by pinning the ef rather than inheriting it from the default.
const GATE_EF_SEARCH: usize = 128;

/// Local SplitMix64 — we don't reach into the crate's private `rng`
/// module from integration tests, so this is a 12-line copy with
/// the same constants. Test-only.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f32_unit(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1_u32 << 24) as f32)
    }

    /// Approximate N(0,1) sample via sum of 12 uniforms (CLT).
    fn next_f32_gaussian(&mut self) -> f32 {
        let mut s = 0.0_f64;
        for _ in 0..12 {
            s += self.next_f32_unit() as f64;
        }
        (s - 6.0) as f32
    }

    fn next_usize_below(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

fn arc(v: Vec<f32>) -> Arc<[f32]> {
    Arc::from(v)
}

fn make_row(rng: &mut Rng, dim: usize, metric: DistanceMetric) -> Vec<f32> {
    match metric {
        // Hamming on f32 is bit-equality; binary 0/1 inputs give a
        // meaningful distribution.
        DistanceMetric::Hamming => (0..dim)
            .map(|_| {
                if rng.next_u64() & 1 == 0 {
                    0.0_f32
                } else {
                    1.0
                }
            })
            .collect(),
        _ => (0..dim).map(|_| rng.next_f32_unit() * 2.0 - 1.0).collect(),
    }
}

fn build_flat(metric: DistanceMetric, rows: &[Vec<f32>]) -> FlatIndex {
    let mut flat = FlatIndex::new(DIM, metric, FlatConfig).unwrap();
    for (i, row) in rows.iter().enumerate() {
        flat.insert(VectorId::from(i as u64), arc(row.clone()), None)
            .unwrap();
    }
    flat
}

fn build_hnsw(metric: DistanceMetric, rows: &[Vec<f32>]) -> HnswIndex {
    // Uses the production default (`ef_search = 64`) at build time; the
    // headline gate queries at an explicit `GATE_EF_SEARCH = 128` via
    // `search_with_ef`, since uniform-random is HNSW's worst case and
    // does not clear 0.95 recall at ef=64.
    let cfg = HnswConfig::default();
    let mut idx = HnswIndex::new(DIM, metric, cfg).unwrap();
    for (i, row) in rows.iter().enumerate() {
        idx.insert(VectorId::from(i as u64), arc(row.clone()), None)
            .unwrap();
    }
    idx
}

/// Number of Gaussian / binary blob centers for clustered data.
const N_CLUSTERS: usize = 50;
/// Gaussian noise std added to each dimension around a center.
const CLUSTER_STD: f32 = 0.1;
/// Per-bit flip probability for the Hamming clustered case.
const CLUSTER_FLIP_P: f32 = 0.05;

/// Generate `n` clustered vectors: pick a random center from `N_CLUSTERS`
/// pre-generated centroids, then add Gaussian noise (or flip bits for
/// Hamming). Deterministic via `seed`.
fn make_clustered_rows(metric: DistanceMetric, n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    let centers: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|_| make_row(&mut rng, dim, metric))
        .collect();

    (0..n)
        .map(|_| {
            let c = &centers[rng.next_usize_below(N_CLUSTERS)];
            match metric {
                DistanceMetric::Hamming => c
                    .iter()
                    .map(|&b| {
                        if rng.next_f32_unit() < CLUSTER_FLIP_P {
                            1.0 - b
                        } else {
                            b
                        }
                    })
                    .collect(),
                _ => c
                    .iter()
                    .map(|&v| v + rng.next_f32_gaussian() * CLUSTER_STD)
                    .collect(),
            }
        })
        .collect()
}

/// Recall@K against the flat oracle using `search_with_ef` (one graph build,
/// multiple ef values).
fn compute_recall_at_ef(
    flat: &FlatIndex,
    hnsw: &HnswIndex,
    queries: &[Vec<f32>],
    metric: DistanceMetric,
    ef: usize,
) -> f64 {
    let params = SearchParams::new(K, metric);
    let mut total = 0.0_f64;
    for query in queries {
        let flat_hits = flat.search(query, &params).unwrap();
        let hnsw_hits = hnsw.search_with_ef(query, &params, ef).unwrap();
        let flat_set: HashSet<_> = flat_hits.iter().map(|h| h.id.clone()).collect();
        let overlap = hnsw_hits
            .iter()
            .filter(|h| flat_set.contains(&h.id))
            .count();
        total += overlap as f64 / K as f64;
    }
    total / queries.len() as f64
}

fn metric_name(m: DistanceMetric) -> &'static str {
    match m {
        DistanceMetric::Cosine => "Cosine",
        DistanceMetric::DotProduct => "DotProduct",
        DistanceMetric::Euclidean => "Euclidean",
        DistanceMetric::Manhattan => "Manhattan",
        DistanceMetric::Hamming => "Hamming",
        _ => "Unknown",
    }
}

fn metric_seed_offset(metric: DistanceMetric) -> u64 {
    match metric {
        DistanceMetric::Cosine => 1,
        DistanceMetric::DotProduct => 2,
        DistanceMetric::Euclidean => 3,
        DistanceMetric::Manhattan => 4,
        DistanceMetric::Hamming => 5,
        _ => 6,
    }
}

fn recall_for(metric: DistanceMetric) -> f64 {
    let mut data_rng = Rng::new(0x1234_5678_9ABC_DEF0_u64.wrapping_add(metric_seed_offset(metric)));
    let rows: Vec<Vec<f32>> = (0..N)
        .map(|_| make_row(&mut data_rng, DIM, metric))
        .collect();

    let flat = build_flat(metric, &rows);
    let hnsw = build_hnsw(metric, &rows);

    let mut query_rng = Rng::new(0x000C_AFEB_ABE0_u64.wrapping_add(metric_seed_offset(metric)));
    let params = SearchParams::new(K, metric);

    let mut total: f64 = 0.0;
    for _ in 0..QUERIES {
        let query = make_row(&mut query_rng, DIM, metric);
        let flat_hits = flat.search(&query, &params).unwrap();
        let hnsw_hits = hnsw
            .search_with_ef(&query, &params, GATE_EF_SEARCH)
            .unwrap();
        let flat_set: HashSet<_> = flat_hits.iter().map(|h| h.id.clone()).collect();
        let overlap = hnsw_hits
            .iter()
            .filter(|h| flat_set.contains(&h.id))
            .count();
        total += overlap as f64 / K as f64;
    }
    total / QUERIES as f64
}

#[test]
fn recall_at_10_meets_floor_cosine() {
    let recall = recall_for(DistanceMetric::Cosine);
    assert!(
        recall >= RECALL_FLOOR,
        "Cosine recall@10 = {recall:.4} < {RECALL_FLOOR}",
    );
}

#[test]
fn recall_at_10_meets_floor_dotproduct() {
    let recall = recall_for(DistanceMetric::DotProduct);
    assert!(
        recall >= RECALL_FLOOR,
        "DotProduct recall@10 = {recall:.4} < {RECALL_FLOOR}",
    );
}

#[test]
fn recall_at_10_meets_floor_euclidean() {
    let recall = recall_for(DistanceMetric::Euclidean);
    assert!(
        recall >= RECALL_FLOOR,
        "Euclidean recall@10 = {recall:.4} < {RECALL_FLOOR}",
    );
}

#[test]
fn recall_at_10_meets_floor_manhattan() {
    let recall = recall_for(DistanceMetric::Manhattan);
    assert!(
        recall >= RECALL_FLOOR,
        "Manhattan recall@10 = {recall:.4} < {RECALL_FLOOR}",
    );
}

#[test]
fn recall_at_10_meets_floor_hamming() {
    let recall = recall_for(DistanceMetric::Hamming);
    assert!(
        recall >= RECALL_FLOOR,
        "Hamming recall@10 = {recall:.4} < {RECALL_FLOOR}",
    );
}

/// Recall@10 vs ef_search for uniform-random vs Gaussian-blob clustered data.
///
/// Run with:
/// ```text
/// cargo test --test recall recall_curve_uniform_vs_clustered -- --include-ignored --nocapture
/// ```
///
/// Diagnostic, not a gate. The production default (`ef_search = 64`) is
/// calibrated to real-data recall — SIFT-1M recall@10 = 0.9644 at ef=64
/// (see `tests/sift_recall.rs`). Uniform-random is HNSW's worst case and
/// is exercised by the headline gate at explicit `ef_search = 128`; this
/// curve is kept around to make the uniform-vs-clustered gap visible.
#[test]
#[ignore]
fn recall_curve_uniform_vs_clustered() {
    const EF_VALUES: [usize; 3] = [32, 64, 128];
    const METRICS: [DistanceMetric; 5] = [
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::Hamming,
    ];

    println!(
        "\n{:<12}  {:>6}   {:>8}   {:>9}",
        "metric", "ef", "uniform", "clustered"
    );
    println!("{}", "-".repeat(46));

    for metric in METRICS {
        let seed_base = 0xABCD_1234_5678_u64.wrapping_add(metric_seed_offset(metric));

        // Uniform-random corpus — same distribution as the headline tests.
        let mut u_data_rng = Rng::new(seed_base);
        let u_rows: Vec<Vec<f32>> = (0..N)
            .map(|_| make_row(&mut u_data_rng, DIM, metric))
            .collect();
        let mut u_q_rng = Rng::new(seed_base.wrapping_add(0x0101));
        let u_queries: Vec<Vec<f32>> = (0..QUERIES)
            .map(|_| make_row(&mut u_q_rng, DIM, metric))
            .collect();

        // Clustered corpus: 50 Gaussian (or binary) blobs.
        let c_rows = make_clustered_rows(metric, N, DIM, seed_base.wrapping_add(0xBEEF));
        let c_queries = make_clustered_rows(metric, QUERIES, DIM, seed_base.wrapping_add(0xCAFE));

        // Build once per data distribution; query at each ef.
        let flat_u = build_flat(metric, &u_rows);
        let hnsw_u = build_hnsw(metric, &u_rows);
        let flat_c = build_flat(metric, &c_rows);
        let hnsw_c = build_hnsw(metric, &c_rows);

        for ef in EF_VALUES {
            let r_u = compute_recall_at_ef(&flat_u, &hnsw_u, &u_queries, metric, ef);
            let r_c = compute_recall_at_ef(&flat_c, &hnsw_c, &c_queries, metric, ef);
            println!(
                "{:<12}  ef={:>3}   {:.4}     {:.4}",
                metric_name(metric),
                ef,
                r_u,
                r_c
            );
        }
        println!();
    }
}
