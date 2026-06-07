//! Filtered-traversal coverage for `iqdb-hnsw`.
//!
//! Exercises the `SearchParams::filter` path end to end: a metadata predicate
//! supplied at query time is evaluated through `iqdb-filter` during result
//! collection, the beam is widened by `HnswConfig::filter_widen`, and only the
//! survivors are returned — still in ascending-distance order, still never
//! returning a tombstoned id. This is the worked example from `docs/API.md`,
//! kept honest as a compiled test.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use iqdb_hnsw::{HnswConfig, HnswIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_types::{DistanceMetric, Filter, Metadata, SearchParams, Value, VectorId};

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

fn color(c: &str) -> Metadata {
    [("color".to_string(), Value::String(c.to_string()))]
        .into_iter()
        .collect()
}

fn eq_color(c: &str) -> Filter {
    Filter::eq("color", Value::String(c.to_string()))
}

/// The headline filtered-search contract: only records whose metadata
/// satisfies the predicate come back, even when a non-matching record is
/// strictly nearer.
#[test]
fn filter_returns_only_matching_metadata() {
    let mut idx = HnswIndex::new(2, DistanceMetric::Euclidean, HnswConfig::default()).unwrap();

    // id 2 ("blue") is nearest to the origin, but the filter wants "red".
    idx.insert(VectorId::from(1u64), arc(&[0.0, 0.0]), Some(color("red")))
        .unwrap();
    idx.insert(VectorId::from(2u64), arc(&[0.1, 0.0]), Some(color("blue")))
        .unwrap();
    idx.insert(VectorId::from(3u64), arc(&[2.0, 0.0]), Some(color("red")))
        .unwrap();

    let params = SearchParams {
        filter: Some(eq_color("red")),
        ..SearchParams::new(5, DistanceMetric::Euclidean)
    };
    let hits = idx.search(&[0.0, 0.0], &params).unwrap();

    assert_eq!(hits.len(), 2, "exactly the two red records survive");
    assert!(
        hits.iter().all(|h| h.id != VectorId::U64(2)),
        "the blue record must never appear under an eq(\"red\") filter",
    );
    assert_eq!(hits[0].id, VectorId::U64(1));
    assert_eq!(hits[1].id, VectorId::U64(3));
    for window in hits.windows(2) {
        assert!(
            window[0].distance <= window[1].distance,
            "filtered results must stay ascending: {window:?}",
        );
    }
}

/// A predicate that matches nothing yields an empty result set, not an error.
#[test]
fn filter_with_no_matches_returns_empty() {
    let mut idx = HnswIndex::new(2, DistanceMetric::Euclidean, HnswConfig::default()).unwrap();
    idx.insert(VectorId::from(1u64), arc(&[0.0, 0.0]), Some(color("red")))
        .unwrap();
    idx.insert(VectorId::from(2u64), arc(&[1.0, 1.0]), Some(color("red")))
        .unwrap();

    let params = SearchParams {
        filter: Some(eq_color("green")),
        ..SearchParams::new(5, DistanceMetric::Euclidean)
    };
    let hits = idx.search(&[0.0, 0.0], &params).unwrap();
    assert!(hits.is_empty(), "no record is green");
}

/// A record with no metadata at all is rejected by an equality predicate —
/// the filter sees `None` and cannot match a concrete value.
#[test]
fn filter_excludes_records_without_metadata() {
    let mut idx = HnswIndex::new(2, DistanceMetric::Euclidean, HnswConfig::default()).unwrap();
    idx.insert(VectorId::from(1u64), arc(&[0.0, 0.0]), None)
        .unwrap();
    idx.insert(VectorId::from(2u64), arc(&[0.5, 0.0]), Some(color("red")))
        .unwrap();

    let params = SearchParams {
        filter: Some(eq_color("red")),
        ..SearchParams::new(5, DistanceMetric::Euclidean)
    };
    let hits = idx.search(&[0.0, 0.0], &params).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, VectorId::U64(2));
}

/// A tombstoned record never returns, even when it satisfies the filter.
#[test]
fn filter_still_skips_tombstoned() {
    let mut idx = HnswIndex::new(2, DistanceMetric::Euclidean, HnswConfig::default()).unwrap();
    idx.insert(VectorId::from(1u64), arc(&[0.0, 0.0]), Some(color("red")))
        .unwrap();
    idx.insert(VectorId::from(2u64), arc(&[1.0, 0.0]), Some(color("red")))
        .unwrap();
    idx.delete(&VectorId::from(1u64)).unwrap();

    let params = SearchParams {
        filter: Some(eq_color("red")),
        ..SearchParams::new(5, DistanceMetric::Euclidean)
    };
    let hits = idx.search(&[0.0, 0.0], &params).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, VectorId::U64(2));
}
