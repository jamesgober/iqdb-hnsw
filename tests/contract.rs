//! IndexCore contract coverage for `iqdb-hnsw`.

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
fn search_results_are_ascending_distance() {
    let mut idx = new_empty(1, DistanceMetric::Euclidean);
    for i in 0..50_u64 {
        idx.insert(VectorId::from(i), arc(&[i as f32]), None)
            .unwrap();
    }
    let hits = idx
        .search(&[0.0], &SearchParams::new(10, DistanceMetric::Euclidean))
        .unwrap();
    assert_eq!(hits.len(), 10);
    for window in hits.windows(2) {
        assert!(
            window[0].distance <= window[1].distance,
            "results not ascending: {window:?}",
        );
    }
}

#[test]
fn search_k_greater_than_n_returns_all_live() {
    let mut idx = new_empty(1, DistanceMetric::Euclidean);
    idx.insert(VectorId::from(1u64), arc(&[1.0]), None).unwrap();
    idx.insert(VectorId::from(2u64), arc(&[2.0]), None).unwrap();
    let hits = idx
        .search(&[0.0], &SearchParams::new(100, DistanceMetric::Euclidean))
        .unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn search_k_equal_n_returns_all() {
    let mut idx = new_empty(1, DistanceMetric::Euclidean);
    idx.insert(VectorId::from(1u64), arc(&[1.0]), None).unwrap();
    idx.insert(VectorId::from(2u64), arc(&[2.0]), None).unwrap();
    let hits = idx
        .search(&[0.0], &SearchParams::new(2, DistanceMetric::Euclidean))
        .unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn insert_dimension_mismatch_returns_typed_error() {
    let mut idx = new_empty(3, DistanceMetric::Euclidean);
    let err = idx
        .insert(VectorId::from(1u64), arc(&[0.0, 0.0]), None)
        .unwrap_err();
    assert_eq!(
        err,
        IqdbError::DimensionMismatch {
            expected: 3,
            found: 2,
        }
    );
}

#[test]
fn search_dimension_mismatch_returns_typed_error() {
    let idx = new_empty(3, DistanceMetric::Euclidean);
    let err = idx
        .search(
            &[0.0, 0.0],
            &SearchParams::new(1, DistanceMetric::Euclidean),
        )
        .unwrap_err();
    assert_eq!(
        err,
        IqdbError::DimensionMismatch {
            expected: 3,
            found: 2,
        }
    );
}

#[test]
fn search_metric_mismatch_returns_invalid_metric() {
    let mut idx = new_empty(2, DistanceMetric::Euclidean);
    idx.insert(VectorId::from(1u64), arc(&[1.0, 0.0]), None)
        .unwrap();
    let err = idx
        .search(&[0.0, 0.0], &SearchParams::new(1, DistanceMetric::Cosine))
        .unwrap_err();
    assert_eq!(err, IqdbError::InvalidMetric);
}

#[test]
fn insert_duplicate_id_returns_duplicate() {
    let mut idx = new_empty(1, DistanceMetric::Euclidean);
    idx.insert(VectorId::from(1u64), arc(&[0.0]), None).unwrap();
    let err = idx
        .insert(VectorId::from(1u64), arc(&[1.0]), None)
        .unwrap_err();
    assert_eq!(err, IqdbError::Duplicate);
}

#[test]
fn delete_missing_returns_not_found() {
    let mut idx = new_empty(1, DistanceMetric::Euclidean);
    let err = idx.delete(&VectorId::from(99u64)).unwrap_err();
    assert_eq!(err, IqdbError::NotFound);
}

#[test]
fn dim_and_metric_reflect_construction() {
    let idx = new_empty(7, DistanceMetric::Cosine);
    assert_eq!(idx.dim(), 7);
    assert_eq!(idx.metric(), DistanceMetric::Cosine);
}

#[test]
fn len_and_is_empty_track_inserts_and_deletes() {
    let mut idx = new_empty(1, DistanceMetric::Euclidean);
    assert!(idx.is_empty());
    assert_eq!(idx.len(), 0);

    idx.insert(VectorId::from(1u64), arc(&[1.0]), None).unwrap();
    assert!(!idx.is_empty());
    assert_eq!(idx.len(), 1);

    idx.insert(VectorId::from(2u64), arc(&[2.0]), None).unwrap();
    assert_eq!(idx.len(), 2);

    idx.delete(&VectorId::from(1u64)).unwrap();
    assert_eq!(idx.len(), 1);
}

#[test]
fn flush_is_ok_for_hnsw() {
    let mut idx = new_empty(1, DistanceMetric::Euclidean);
    idx.flush().unwrap();
}

#[test]
fn stats_reports_hnsw_index_type_and_counts() {
    let mut idx = new_empty(3, DistanceMetric::Euclidean);
    idx.insert(VectorId::from(1u64), arc(&[0.0, 0.0, 0.0]), None)
        .unwrap();
    let stats = idx.stats();
    assert_eq!(stats.n_vectors, 1);
    assert_eq!(stats.index_type, "hnsw");
    assert_eq!(stats.disk_bytes, None);
    assert!(stats.memory_bytes > 0);
    assert!(stats.extra.is_none());
}

#[test]
fn hnsw_index_is_object_safe_through_dyn_index_core() {
    let mut idx: Box<dyn IndexCore> =
        Box::new(HnswIndex::new(2, DistanceMetric::Cosine, HnswConfig::default()).unwrap());
    assert_eq!(idx.dim(), 2);
    assert_eq!(idx.metric(), DistanceMetric::Cosine);
    assert!(idx.is_empty());
    idx.insert(VectorId::from(1u64), arc(&[1.0, 0.0]), None)
        .unwrap();
    assert_eq!(idx.len(), 1);
    idx.flush().unwrap();
}

#[test]
fn invalid_configs_are_rejected_at_new() {
    let err = HnswIndex::new(0, DistanceMetric::Euclidean, HnswConfig::default()).unwrap_err();
    assert!(matches!(err, IqdbError::InvalidConfig { .. }));

    let cfg = HnswConfig::default().with_m(0);
    let err = HnswIndex::new(2, DistanceMetric::Euclidean, cfg).unwrap_err();
    assert!(matches!(err, IqdbError::InvalidConfig { .. }));

    let cfg = HnswConfig::default().with_m(64).with_ef_construction(10);
    let err = HnswIndex::new(2, DistanceMetric::Euclidean, cfg).unwrap_err();
    assert!(matches!(err, IqdbError::InvalidConfig { .. }));

    let cfg = HnswConfig::default().with_ef_search(0);
    let err = HnswIndex::new(2, DistanceMetric::Euclidean, cfg).unwrap_err();
    assert!(matches!(err, IqdbError::InvalidConfig { .. }));

    let cfg = HnswConfig::default().with_filter_widen(0);
    let err = HnswIndex::new(2, DistanceMetric::Euclidean, cfg).unwrap_err();
    assert!(matches!(err, IqdbError::InvalidConfig { .. }));
}
