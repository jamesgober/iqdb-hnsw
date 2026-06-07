//! Determinism: identical insert order + identical seed → identical
//! search results across two independent builds.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use iqdb_hnsw::{HnswConfig, HnswIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_types::{DistanceMetric, Hit, SearchParams, VectorId};

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

fn synthetic_row(i: usize, dim: usize) -> Vec<f32> {
    (0..dim).map(|j| ((i + j * 7) as f32).sin()).collect()
}

fn query_vector(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim).map(|j| ((seed + j * 11) as f32).cos()).collect()
}

fn build(seed: u64, n: usize, dim: usize, metric: DistanceMetric) -> HnswIndex {
    let cfg = HnswConfig::default().with_seed(seed);
    let mut idx = HnswIndex::new(dim, metric, cfg).unwrap();
    for i in 0..n {
        idx.insert(VectorId::from(i as u64), arc(&synthetic_row(i, dim)), None)
            .unwrap();
    }
    idx
}

fn hits_bit_equal(a: &[Hit], b: &[Hit]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        x.id == y.id && x.distance.to_bits() == y.distance.to_bits() && x.metadata == y.metadata
    })
}

#[test]
fn identical_seed_and_order_yields_byte_identical_hits_euclidean() {
    let dim = 32_usize;
    let n = 500_usize;
    let seed = 0xABCD_1234_5678_9ABC;
    let metric = DistanceMetric::Euclidean;

    let a = build(seed, n, dim, metric);
    let b = build(seed, n, dim, metric);

    let params = SearchParams::new(10, metric);
    for q in 0..50 {
        let query = query_vector(q, dim);
        let ha = a.search(&query, &params).unwrap();
        let hb = b.search(&query, &params).unwrap();
        assert!(
            hits_bit_equal(&ha, &hb),
            "divergent hits at query {q}: a={ha:?}, b={hb:?}",
        );
    }
}

#[test]
fn identical_seed_and_order_yields_byte_identical_hits_cosine() {
    let dim = 16_usize;
    let n = 400_usize;
    let seed = 7;
    let metric = DistanceMetric::Cosine;

    let a = build(seed, n, dim, metric);
    let b = build(seed, n, dim, metric);

    let params = SearchParams::new(5, metric);
    for q in 0..40 {
        let query = query_vector(q, dim);
        let ha = a.search(&query, &params).unwrap();
        let hb = b.search(&query, &params).unwrap();
        assert!(hits_bit_equal(&ha, &hb));
    }
}

// (Removed) negative-control test "different seeds produce different
// results" — at the default wide beam, both seeds converge to the
// same hits regardless of layer assignment, which doesn't actually
// test the seed. The `layer_dist.rs` integration test, which inspects
// the assigned-layer histogram directly, is the right place to
// confirm the seed drives layer assignment.
