//! Criterion benches for `iqdb-hnsw` search latency at two corpus scales and
//! three `ef_search` widths, with recall@10 reported in each bench name so the
//! latency / recall curve is documented on every run.
//!
//! Recall is measured against an inline exact full-scan oracle (the same
//! `iqdb_distance` kernel HNSW uses, `DotProduct` negated), so the bench needs
//! no sibling index crate. This is a documentation bench, not a regression gate.
//!
//! Data is synthetic and seeded (sin/cos columns from the row index plus a
//! dimension offset) so the recall numbers in the bench names are reproducible.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use iqdb_distance::compute_batch;
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

/// Exact top-`k` ids by full scan — the recall oracle.
fn exact_top_ids(rows: &[Vec<f32>], query: &[f32], metric: DistanceMetric, k: usize) -> Vec<u64> {
    let slices: Vec<&[f32]> = rows.iter().map(Vec::as_slice).collect();
    let mut dists = vec![0.0_f32; slices.len()];
    compute_batch(metric, query, &slices, &mut dists).expect("distance");
    let negate = matches!(metric, DistanceMetric::DotProduct);
    let mut scored: Vec<(u64, f32)> = dists
        .iter()
        .enumerate()
        .map(|(i, &d)| (i as u64, if negate { -d } else { d }))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}

fn recall_at_10(
    rows: &[Vec<f32>],
    hnsw: &HnswIndex,
    dim: usize,
    metric: DistanceMetric,
    n_queries: usize,
) -> f64 {
    let params = SearchParams::new(10, metric);
    let mut total = 0.0_f64;
    for q in 0..n_queries {
        let query = query_vector(q.wrapping_mul(37), dim);
        let exact: std::collections::HashSet<u64> = exact_top_ids(rows, &query, metric, 10)
            .into_iter()
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
        let rows: Vec<Vec<f32>> = (0..n).map(|i| synthetic_row(i, dim)).collect();
        for ef in ef_search_values {
            let hnsw = build_hnsw(n, dim, metric, ef);
            let recall = recall_at_10(&rows, &hnsw, dim, metric, 64);
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
