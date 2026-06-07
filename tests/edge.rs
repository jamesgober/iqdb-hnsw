//! Edge-case coverage for `iqdb-hnsw`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use iqdb_hnsw::{HnswConfig, HnswIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_types::{DistanceMetric, IqdbError, SearchParams, VectorId};

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

fn new_empty(dim: usize, metric: DistanceMetric) -> HnswIndex {
    HnswIndex::new(dim, metric, HnswConfig::default()).unwrap()
}

#[test]
fn search_on_empty_index_returns_empty() {
    let idx = new_empty(3, DistanceMetric::Euclidean);
    let hits = idx
        .search(
            &[0.0, 0.0, 0.0],
            &SearchParams::new(5, DistanceMetric::Euclidean),
        )
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn search_with_k_zero_returns_empty() {
    let mut idx = new_empty(2, DistanceMetric::Euclidean);
    idx.insert(VectorId::from(1u64), arc(&[0.0, 0.0]), None)
        .unwrap();
    let hits = idx
        .search(
            &[0.0, 0.0],
            &SearchParams::new(0, DistanceMetric::Euclidean),
        )
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn single_vector_search_returns_that_vector() {
    let mut idx = new_empty(2, DistanceMetric::Euclidean);
    idx.insert(VectorId::from(42u64), arc(&[1.5, 2.5]), None)
        .unwrap();
    let hits = idx
        .search(
            &[0.0, 0.0],
            &SearchParams::new(1, DistanceMetric::Euclidean),
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, VectorId::U64(42));
}

#[test]
fn all_identical_vectors_search_does_not_panic_and_orders_deterministically() {
    let mut idx = new_empty(3, DistanceMetric::Euclidean);
    for i in 0..16_u64 {
        idx.insert(VectorId::from(i), arc(&[1.0, 1.0, 1.0]), None)
            .unwrap();
    }
    let hits = idx
        .search(
            &[1.0, 1.0, 1.0],
            &SearchParams::new(5, DistanceMetric::Euclidean),
        )
        .unwrap();
    assert_eq!(hits.len(), 5);
    for hit in &hits {
        assert!(
            hit.distance.abs() < 1e-6,
            "expected zero distance, got {}",
            hit.distance,
        );
    }
    let again = idx
        .search(
            &[1.0, 1.0, 1.0],
            &SearchParams::new(5, DistanceMetric::Euclidean),
        )
        .unwrap();
    let ids_first: Vec<_> = hits.iter().map(|h| h.id.clone()).collect();
    let ids_again: Vec<_> = again.iter().map(|h| h.id.clone()).collect();
    assert_eq!(ids_first, ids_again);
}

#[test]
fn equal_distance_ties_break_deterministically_by_insertion_order() {
    let mut idx = new_empty(1, DistanceMetric::Euclidean);
    idx.insert(VectorId::from(10u64), arc(&[1.0]), None)
        .unwrap();
    idx.insert(VectorId::from(20u64), arc(&[1.0]), None)
        .unwrap();
    idx.insert(VectorId::from(30u64), arc(&[1.0]), None)
        .unwrap();

    let hits = idx
        .search(&[0.0], &SearchParams::new(3, DistanceMetric::Euclidean))
        .unwrap();
    let ids: Vec<VectorId> = hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(
        ids,
        vec![VectorId::U64(10), VectorId::U64(20), VectorId::U64(30)],
        "tiebreaker should be insertion order",
    );
}

#[test]
fn query_dim_mismatch_returns_typed_error_not_panic() {
    let mut idx = new_empty(4, DistanceMetric::Euclidean);
    idx.insert(VectorId::from(1u64), arc(&[1.0, 0.0, 0.0, 0.0]), None)
        .unwrap();
    let err = idx
        .search(
            &[1.0, 0.0],
            &SearchParams::new(1, DistanceMetric::Euclidean),
        )
        .unwrap_err();
    assert_eq!(
        err,
        IqdbError::DimensionMismatch {
            expected: 4,
            found: 2,
        }
    );
}

#[test]
fn large_n_inserts_without_panic_or_overflow() {
    let mut idx = new_empty(8, DistanceMetric::Euclidean);
    for i in 0..1_000_u64 {
        let v: Vec<f32> = (0..8).map(|j| ((i + j as u64) as f32).sin()).collect();
        idx.insert(VectorId::from(i), arc(&v), None).unwrap();
    }
    assert_eq!(idx.len(), 1_000);
    let hits = idx
        .search(
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &SearchParams::new(10, DistanceMetric::Euclidean),
        )
        .unwrap();
    assert_eq!(hits.len(), 10);
}
