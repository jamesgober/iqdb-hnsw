//! Tombstone-delete semantics for `iqdb-hnsw`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use iqdb_hnsw::{HnswConfig, HnswIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

fn new_loaded(metric: DistanceMetric, n: u64) -> HnswIndex {
    let mut idx = HnswIndex::new(1, metric, HnswConfig::default()).unwrap();
    for i in 0..n {
        idx.insert(VectorId::from(i), arc(&[i as f32]), None)
            .unwrap();
    }
    idx
}

#[test]
fn deleted_id_never_appears_in_search() {
    let mut idx = new_loaded(DistanceMetric::Euclidean, 50);
    idx.delete(&VectorId::from(7u64)).unwrap();

    let hits = idx
        .search(&[7.0], &SearchParams::new(10, DistanceMetric::Euclidean))
        .unwrap();
    for hit in &hits {
        assert_ne!(hit.id, VectorId::U64(7), "deleted id surfaced in results");
    }
    assert_eq!(idx.len(), 49);
}

#[test]
fn delete_then_reinsert_returns_id_again() {
    let mut idx = new_loaded(DistanceMetric::Euclidean, 30);
    idx.delete(&VectorId::from(5u64)).unwrap();
    idx.insert(VectorId::from(5u64), arc(&[99.0]), None)
        .unwrap();
    assert_eq!(idx.len(), 30);

    let hits = idx
        .search(&[99.0], &SearchParams::new(5, DistanceMetric::Euclidean))
        .unwrap();
    let ids: Vec<VectorId> = hits.iter().map(|h| h.id.clone()).collect();
    assert!(
        ids.contains(&VectorId::U64(5)),
        "re-inserted id missing: {ids:?}",
    );
}

#[test]
fn delete_all_leaves_search_returning_empty() {
    let mut idx = new_loaded(DistanceMetric::Euclidean, 8);
    for i in 0..8_u64 {
        idx.delete(&VectorId::from(i)).unwrap();
    }
    assert!(idx.is_empty());

    let hits = idx
        .search(&[0.0], &SearchParams::new(10, DistanceMetric::Euclidean))
        .unwrap();
    assert!(hits.is_empty());
}

#[test]
fn delete_then_search_returns_at_most_remaining() {
    let mut idx = new_loaded(DistanceMetric::Euclidean, 20);
    for i in 0..10_u64 {
        idx.delete(&VectorId::from(i)).unwrap();
    }
    let hits = idx
        .search(&[0.0], &SearchParams::new(50, DistanceMetric::Euclidean))
        .unwrap();
    assert!(hits.len() <= 10);
    for hit in &hits {
        if let VectorId::U64(n) = hit.id {
            assert!(n >= 10, "deleted id {n} surfaced");
        }
    }
}

#[test]
fn stats_n_vectors_excludes_tombstoned() {
    let mut idx = new_loaded(DistanceMetric::Euclidean, 10);
    idx.delete(&VectorId::from(0u64)).unwrap();
    idx.delete(&VectorId::from(1u64)).unwrap();
    assert_eq!(idx.stats().n_vectors, 8);
}
