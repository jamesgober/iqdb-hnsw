//! Public-surface smoke test for `iqdb-hnsw`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use iqdb_hnsw::{HnswConfig, HnswIndex, VERSION};
use iqdb_index::{Index, IndexCore};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

#[test]
fn version_is_semver_triplet() {
    assert_eq!(VERSION.split('.').count(), 3);
    assert!(VERSION.split('.').all(|part| !part.is_empty()));
}

#[test]
fn end_to_end_insert_then_search() {
    let mut idx = HnswIndex::new(2, DistanceMetric::Euclidean, HnswConfig::default()).unwrap();
    idx.insert(VectorId::from(1u64), arc(&[0.0, 0.0]), None)
        .unwrap();
    idx.insert(VectorId::from(2u64), arc(&[3.0, 4.0]), None)
        .unwrap();
    idx.insert(VectorId::from(3u64), arc(&[1.0, 0.0]), None)
        .unwrap();

    let hits = idx
        .search(
            &[0.0, 0.0],
            &SearchParams::new(2, DistanceMetric::Euclidean),
        )
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, VectorId::U64(1));
    assert_eq!(hits[1].id, VectorId::U64(3));
}

#[test]
fn hnsw_config_satisfies_index_config_trait_bounds() {
    fn ensure_default_clone<T: Default + Clone>() -> T {
        let value = T::default();
        value.clone()
    }
    let _config: HnswConfig = ensure_default_clone();
}
