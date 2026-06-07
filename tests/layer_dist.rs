//! Sanity test: the assigned-layer histogram matches the expected
//! geometric decay for the seed.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use iqdb_hnsw::{HnswConfig, HnswIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_types::{DistanceMetric, VectorId};

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

fn synthetic_row(i: usize, dim: usize) -> Vec<f32> {
    (0..dim).map(|j| ((i + j * 7) as f32).sin()).collect()
}

#[test]
fn layer_zero_holds_about_fifteen_sixteenths_at_m16() {
    let dim = 16_usize;
    let n = 5_000_usize;
    let cfg = HnswConfig::default(); // m = 16
    let mut idx = HnswIndex::new(dim, DistanceMetric::Euclidean, cfg).unwrap();
    for i in 0..n {
        idx.insert(VectorId::from(i as u64), arc(&synthetic_row(i, dim)), None)
            .unwrap();
    }

    let hist = idx.node_layer_histogram();
    assert!(!hist.is_empty());
    let total: usize = hist.iter().sum();
    assert_eq!(total, n);

    let l0 = hist[0] as f64 / total as f64;
    // Expected ~ 15/16 = 0.9375.
    assert!(
        (l0 - 0.9375).abs() < 0.02,
        "layer-0 fraction {l0:.4} not near 15/16",
    );
}

#[test]
fn histogram_decays_monotonically_in_the_head() {
    let dim = 16_usize;
    let n = 5_000_usize;
    let cfg = HnswConfig::default();
    let mut idx = HnswIndex::new(dim, DistanceMetric::Euclidean, cfg).unwrap();
    for i in 0..n {
        idx.insert(VectorId::from(i as u64), arc(&synthetic_row(i, dim)), None)
            .unwrap();
    }
    let hist = idx.node_layer_histogram();
    assert!(
        hist.len() >= 3,
        "histogram has fewer than 3 layers: {hist:?}",
    );
    assert!(
        hist[0] > hist[1],
        "hist[0]={} not > hist[1]={}",
        hist[0],
        hist[1],
    );
    assert!(
        hist[1] >= hist[2],
        "hist[1]={} unexpectedly < hist[2]={}",
        hist[1],
        hist[2],
    );
}

#[test]
fn empty_index_has_empty_histogram() {
    let cfg = HnswConfig::default();
    let idx = HnswIndex::new(4, DistanceMetric::Euclidean, cfg).unwrap();
    assert!(idx.node_layer_histogram().is_empty());
}
