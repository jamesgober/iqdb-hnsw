//! Criterion benches for `iqdb-hnsw` search latency at two corpus scales and
//! three `ef_search` widths, with recall@10 reported in each bench name so the
//! latency / recall curve is documented on every run.
//!
//! Recall is measured against the exact `iqdb_flat::FlatIndex` oracle — the
//! same brute-force ground truth the headline recall gate uses — so the bench
//! and the gate share one notion of "true top-k". This is a documentation
//! bench, not a regression gate.
//!
//! Data is synthetic and seeded (sin/cos columns from the row index plus a
//! dimension offset) so the recall numbers in the bench names are reproducible.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use iqdb_flat::{FlatConfig, FlatIndex};
use iqdb_hnsw::{HnswConfig, HnswIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

fn synthetic_row(i: usize, dim: usize) -> Vec<f32> {
    (0..dim).map(|j| ((i + j * 7) as f32).sin()).collect()
}

fn query_vector(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim).map(|j| ((seed + j * 11) as f32).cos()).collect()
}

fn build_hnsw(n: usize, dim: usize, metric: DistanceMetric, ef_search: usize) -> HnswIndex {
    let cfg = HnswConfig::default().with_ef_search(ef_search);
    let mut idx = HnswIndex::new(dim, metric, cfg).expect("valid dim");
    for i in 0..n {
        let row = synthetic_row(i, dim);
        idx.insert(VectorId::from(i as u64), arc(&row), None)
            .expect("fresh id");
    }
    idx
}

/// Build the exact `FlatIndex` oracle over the same corpus as the HNSW index.
fn build_flat(n: usize, dim: usize, metric: DistanceMetric) -> FlatIndex {
    let mut flat = FlatIndex::new(dim, metric, FlatConfig).expect("valid dim");
    for i in 0..n {
        let row = synthetic_row(i, dim);
        flat.insert(VectorId::from(i as u64), arc(&row), None)
            .expect("fresh id");
    }
    flat
}

fn recall_at_10(
    flat: &FlatIndex,
    hnsw: &HnswIndex,
    dim: usize,
    metric: DistanceMetric,
    n_queries: usize,
) -> f64 {
    let params = SearchParams::new(10, metric);
    let mut total = 0.0_f64;
    for q in 0..n_queries {
        let query = query_vector(q.wrapping_mul(37), dim);
        let exact: std::collections::HashSet<u64> = flat
            .search(&query, &params)
            .expect("flat search")
            .iter()
            .filter_map(|h| match h.id {
                VectorId::U64(v) => Some(v),
                _ => None,
            })
            .collect();
        let hnsw_hits = hnsw.search(&query, &params).expect("hnsw search");
        let overlap = hnsw_hits
            .iter()
            .filter(|h| matches!(&h.id, VectorId::U64(v) if exact.contains(v)))
            .count();
        total += overlap as f64 / 10.0_f64;
    }
    total / n_queries as f64
}

fn bench_hnsw(c: &mut Criterion) {
    let dim = 128_usize;
    let metric = DistanceMetric::Euclidean;
    let scales = [10_000_usize, 100_000_usize];
    let ef_search_values = [32_usize, 64, 128];

    for n in scales {
        let flat = build_flat(n, dim, metric);
        for ef in ef_search_values {
            let hnsw = build_hnsw(n, dim, metric, ef);
            let recall = recall_at_10(&flat, &hnsw, dim, metric, 64);
            let recall_pct = (recall * 1000.0).round() as i64;
            let query = query_vector(0, dim);
            let params = SearchParams::new(10, metric);
            let name =
                format!("hnsw_bench/hnsw/n{n}/d{dim}/ef{ef}/recall_at_10_x1000_{recall_pct}");
            let _ = c.bench_function(&name, |bencher| {
                bencher.iter(|| hnsw.search(black_box(&query), black_box(&params)));
            });
        }
    }
}

criterion_group!(benches, bench_hnsw);
criterion_main!(benches);
