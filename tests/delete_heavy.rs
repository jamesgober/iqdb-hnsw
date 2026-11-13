//! Search quality and memory under heavy deletes, plus the size bounds on
//! caller- and config-supplied parameters.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::sync::Arc;

use iqdb_flat::{FlatConfig, FlatIndex};
use iqdb_hnsw::{HnswConfig, HnswIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_types::{DistanceMetric, Filter, Hit, IqdbError, Metadata, SearchParams, Value, VectorId};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| r.unit()).collect())
        .collect()
}

fn ids(hits: &[Hit]) -> HashSet<VectorId> {
    hits.iter().map(|h| h.id.clone()).collect()
}

fn load_both(data: &[Vec<f32>], metric: DistanceMetric) -> (HnswIndex, FlatIndex) {
    let dim = data[0].len();
    let mut h = HnswIndex::new(dim, metric, HnswConfig::default()).unwrap();
    let mut f = FlatIndex::new(dim, metric, FlatConfig).unwrap();
    for (i, v) in data.iter().enumerate() {
        let a: Arc<[f32]> = Arc::from(v.as_slice());
        h.insert(VectorId::from(i as u64), a.clone(), None).unwrap();
        f.insert(VectorId::from(i as u64), a, None).unwrap();
    }
    (h, f)
}

#[test]
fn deleting_the_whole_neighbourhood_still_returns_k_live_hits() {
    let dim = 16;
    let data = corpus(4000, dim, 7);
    let (mut h, mut f) = load_both(&data, DistanceMetric::Euclidean);
    let q = vec![0.0f32; dim];

    // Delete the 3000 vectors nearest the query.
    let near = f
        .search(&q, &SearchParams::new(3000, DistanceMetric::Euclidean))
        .unwrap();
    for hit in &near {
        h.delete(&hit.id).unwrap();
        f.delete(&hit.id).unwrap();
    }
    assert_eq!(h.len(), 1000);

    let p = SearchParams::new(10, DistanceMetric::Euclidean);
    let got = h.search(&q, &p).unwrap();
    let truth = f.search(&q, &p).unwrap();
    assert_eq!(
        got.len(),
        10,
        "only {} hits with 1000 live vectors",
        got.len()
    );
    let overlap = ids(&got).intersection(&ids(&truth)).count();
    assert!(overlap >= 8, "recall@10 = {overlap}/10 after heavy deletes");
}

#[test]
fn recall_holds_against_flat_after_random_deletes() {
    let dim = 16;
    let data = corpus(3000, dim, 11);
    let queries = corpus(30, dim, 12);
    let (mut h, mut f) = load_both(&data, DistanceMetric::Cosine);
    let mut r = Rng(13);
    for i in 0..3000u64 {
        if r.next() % 10 < 8 {
            h.delete(&VectorId::from(i)).unwrap();
            f.delete(&VectorId::from(i)).unwrap();
        }
    }
    let p = SearchParams::new(10, DistanceMetric::Cosine);
    let mut found = 0usize;
    for q in &queries {
        let got = h.search(q, &p).unwrap();
        assert_eq!(got.len(), 10);
        found += ids(&got)
            .intersection(&ids(&f.search(q, &p).unwrap()))
            .count();
    }
    let recall = found as f32 / (10 * queries.len()) as f32;
    assert!(recall >= 0.9, "recall@10 = {recall} after 80% deletes");
}

#[test]
fn selective_filter_returns_every_match_it_can_reach() {
    let dim = 8;
    let data = corpus(2000, dim, 21);
    let mut h = HnswIndex::new(dim, DistanceMetric::Euclidean, HnswConfig::default()).unwrap();
    for (i, v) in data.iter().enumerate() {
        let tag = if i % 200 == 0 { "rare" } else { "common" };
        let meta: Metadata = [("tag".to_string(), Value::String(tag.to_string()))]
            .into_iter()
            .collect();
        h.insert(
            VectorId::from(i as u64),
            Arc::from(v.as_slice()),
            Some(meta),
        )
        .unwrap();
    }
    let mut p = SearchParams::new(10, DistanceMetric::Euclidean);
    p.filter = Some(Filter::Eq {
        field: "tag".to_string(),
        value: Value::String("rare".to_string()),
    });
    let hits = h.search(&data[1], &p).unwrap();
    // Ten ids match, and the graph reaches all of them.
    assert_eq!(
        hits.len(),
        10,
        "{} of 10 filtered matches returned",
        hits.len()
    );
}

#[test]
fn compact_frees_deleted_nodes_and_keeps_results() {
    let dim = 8;
    let mut h = HnswIndex::new(dim, DistanceMetric::Euclidean, HnswConfig::default()).unwrap();
    // 100 live rows, each replaced 20 times.
    for round in 0..20u64 {
        for i in 0..100u64 {
            let id = VectorId::from(i);
            let _ = h.delete(&id);
            let v: Vec<f32> = (0..dim as u64)
                .map(|j| (i * 8 + j + round) as f32)
                .collect();
            h.insert(id, Arc::from(v.as_slice()), None).unwrap();
        }
    }
    assert_eq!(h.len(), 100);
    assert_eq!(h.tombstones(), 1900);

    let q = vec![40.0f32; dim];
    let p = SearchParams::new(10, DistanceMetric::Euclidean);
    let before = h.search(&q, &p).unwrap();

    h.compact().unwrap();
    assert_eq!(h.tombstones(), 0);
    assert_eq!(h.len(), 100);
    assert_eq!(h.node_layer_histogram().iter().sum::<usize>(), 100);

    let after = h.search(&q, &p).unwrap();
    assert_eq!(ids(&before), ids(&after));

    // Still a working index: deletes, inserts and duplicates behave.
    h.delete(&VectorId::from(0u64)).unwrap();
    assert!(matches!(
        h.insert(VectorId::from(1u64), Arc::from(&[0.0f32; 8][..]), None),
        Err(IqdbError::Duplicate)
    ));
    h.insert(VectorId::from(500u64), Arc::from(&[40.0f32; 8][..]), None)
        .unwrap();
    assert_eq!(h.search(&q, &p).unwrap()[0].id, VectorId::from(500u64));
}

#[test]
fn compact_is_deterministic() {
    let data = corpus(500, 8, 31);
    let build = || {
        let (mut h, _) = load_both(&data, DistanceMetric::Euclidean);
        for i in (0..500u64).step_by(3) {
            h.delete(&VectorId::from(i)).unwrap();
        }
        h.compact().unwrap();
        h
    };
    let (a, b) = (build(), build());
    let p = SearchParams::new(10, DistanceMetric::Euclidean);
    for q in corpus(10, 8, 32) {
        let ha: Vec<VectorId> = a
            .search(&q, &p)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let hb: Vec<VectorId> = b
            .search(&q, &p)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(ha, hb);
    }
}

#[test]
fn huge_k_is_clamped_to_the_live_count() {
    let mut h = HnswIndex::new(2, DistanceMetric::Euclidean, HnswConfig::default()).unwrap();
    for i in 0..5u64 {
        h.insert(VectorId::from(i), Arc::from(&[i as f32, 0.0][..]), None)
            .unwrap();
    }
    let hits = h
        .search(
            &[0.0, 0.0],
            &SearchParams::new(usize::MAX / 2, DistanceMetric::Euclidean),
        )
        .unwrap();
    assert_eq!(hits.len(), 5);
    let mut p = SearchParams::new(usize::MAX, DistanceMetric::Euclidean);
    p.ef = Some(usize::MAX);
    assert_eq!(h.search(&[0.0, 0.0], &p).unwrap().len(), 5);
    assert_eq!(
        h.search_with_ef(&[0.0, 0.0], &p, usize::MAX).unwrap().len(),
        5
    );
}

#[test]
fn oversized_m_is_rejected_at_construction() {
    let cfg = HnswConfig::default()
        .with_m(usize::MAX / 2 + 1)
        .with_ef_construction(usize::MAX);
    assert!(matches!(
        HnswIndex::new(2, DistanceMetric::Euclidean, cfg),
        Err(IqdbError::InvalidConfig { .. })
    ));
    let ok = HnswConfig::default()
        .with_m(1024)
        .with_ef_construction(1024);
    assert!(HnswIndex::new(2, DistanceMetric::Euclidean, ok).is_ok());
    let big = HnswConfig::default()
        .with_m(1025)
        .with_ef_construction(2048);
    assert!(HnswIndex::new(2, DistanceMetric::Euclidean, big).is_err());
}

#[test]
fn huge_ef_construction_is_clamped_on_insert() {
    let cfg = HnswConfig::default().with_ef_construction(usize::MAX);
    let mut h = HnswIndex::new(2, DistanceMetric::Euclidean, cfg).unwrap();
    for i in 0..20u64 {
        h.insert(VectorId::from(i), Arc::from(&[i as f32, 1.0][..]), None)
            .unwrap();
    }
    assert_eq!(h.len(), 20);
}

#[test]
fn params_ef_sets_the_beam_width() {
    let dim = 16;
    let data = corpus(3000, dim, 41);
    let cfg = HnswConfig::default().with_ef_search(10);
    let mut h = HnswIndex::new(dim, DistanceMetric::Euclidean, cfg).unwrap();
    for (i, v) in data.iter().enumerate() {
        h.insert(VectorId::from(i as u64), Arc::from(v.as_slice()), None)
            .unwrap();
    }
    let q = vec![0.5f32; dim];
    let mut p = SearchParams::new(10, DistanceMetric::Euclidean);
    p.ef = Some(2000);
    let via_params: Vec<VectorId> = h
        .search(&q, &p)
        .unwrap()
        .into_iter()
        .map(|x| x.id)
        .collect();
    let via_override: Vec<VectorId> = h
        .search_with_ef(&q, &p, 2000)
        .unwrap()
        .into_iter()
        .map(|x| x.id)
        .collect();
    assert_eq!(via_params, via_override);
}

#[test]
fn overflowing_cosine_candidate_does_not_outrank_an_exact_match() {
    let mut h = HnswIndex::new(2, DistanceMetric::Cosine, HnswConfig::default()).unwrap();
    h.insert(VectorId::from(1u64), Arc::from(&[1.0f32, 1.0][..]), None)
        .unwrap();
    h.insert(
        VectorId::from(2u64),
        Arc::from(
            &[
                std::hint::black_box(1e20f32),
                std::hint::black_box(-1e20f32),
            ][..],
        ),
        None,
    )
    .unwrap();
    let q = [std::hint::black_box(1e20f32), std::hint::black_box(1e20f32)];
    let hits = h
        .search(&q, &SearchParams::new(2, DistanceMetric::Cosine))
        .unwrap();
    assert_eq!(hits[0].id, VectorId::from(1u64), "{hits:?}");
}
