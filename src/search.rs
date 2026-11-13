//! HNSW SEARCH-LAYER (Alg 2) and the k-NN query path (Alg 5).
//!
//! `search_layer` is the beam-search primitive: it walks the graph at
//! one layer from a set of entry points and returns up to `ef`
//! candidates in a max-heap keyed by `(distance, seq)`. `knn_search`
//! is the query-time driver: greedy-descend from the global entry
//! through the upper layers with `ef = 1`, then run SEARCH-LAYER at
//! layer 0 with `ef = max(ef_search, k)` and take the closest `k`.
//!
//! ## Tombstone and filter behaviour
//!
//! Deleted (tombstoned) nodes are **traversed** for graph connectivity,
//! since their out-edges still lead to live nodes, but at layer 0 of a
//! query they never enter the result beam and never count toward it. A
//! metadata filter works the same way: a node that fails it is walked
//! through but not collected. The beam therefore holds up to `ef` nodes
//! that can actually be returned, and the walk only stops early once it
//! has that many, so a neighbourhood full of deleted or filtered-out
//! nodes widens the search instead of starving it. When fewer than `k`
//! such nodes are reachable, the walk covers the whole reachable graph
//! and returns what it found.
//!
//! The insert path keeps tombstoned nodes in its candidate beam (they
//! are still useful routing points) and filters them out before they
//! can become a freshly linked neighbour.
//!
//! ## Size bounds
//!
//! Every buffer is sized from the beam width clamped to the number of
//! stored nodes, never from the caller's `k` or `ef` alone, so a huge
//! `k` costs no more than an exhaustive walk.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use iqdb_distance::compute_batch;
use iqdb_filter::FilterEvaluator;
use iqdb_types::{DistanceMetric, Hit, IqdbError, Result, SearchParams};

use crate::graph::NodeIdx;
use crate::index::HnswIndex;
use crate::topk::{Scored, cmp_distance, take_topk_sorted};

/// Public-facing search entry point. Called from
/// [`crate::HnswIndex::search`] (via the trait-impl shim in
/// `index.rs`). The base beam width is `params.ef` when set, else the
/// configured `ef_search`.
pub(crate) fn search(idx: &HnswIndex, query: &[f32], params: &SearchParams) -> Result<Vec<Hit>> {
    let ef = params.ef.unwrap_or(idx.cfg.ef_search);
    search_impl(idx, query, params, ef)
}

/// Like [`search`] but overrides the stored `ef_search` with a
/// caller-supplied beam width.
///
/// Intended for recall-curve diagnostics and benchmarks. Applies
/// `filter_widen` when a filter is present (same as [`search`]) but
/// uses `ef` instead of `idx.cfg.ef_search` as the base beam width.
pub(crate) fn search_with_ef(
    idx: &HnswIndex,
    query: &[f32],
    params: &SearchParams,
    ef: usize,
) -> Result<Vec<Hit>> {
    search_impl(idx, query, params, ef)
}

fn search_impl(
    idx: &HnswIndex,
    query: &[f32],
    params: &SearchParams,
    ef: usize,
) -> Result<Vec<Hit>> {
    idx.check_dim(query.len())?;
    if params.metric != idx.metric {
        return Err(IqdbError::InvalidMetric);
    }
    if params.k == 0 || idx.live_count == 0 || idx.entry.is_none() {
        return Ok(Vec::new());
    }

    let ef_base = ef.max(params.k);
    let ef_effective = if params.filter.is_some() {
        ef_base.saturating_mul(idx.cfg.filter_widen)
    } else {
        ef_base
    };
    // The result beam only ever holds live nodes, so a width past the
    // live count buys nothing; clamping here keeps a huge `k` from
    // sizing anything.
    let ef_effective = ef_effective.min(idx.live_count);

    let scored = match &params.filter {
        None => {
            let accept = |n: NodeIdx| !idx.tombstoned[n as usize];
            knn_search(idx, query, ef_effective, Some(&accept))?
        }
        Some(filter) => {
            let evaluator = FilterEvaluator::new(filter.clone())?;
            let accept = |n: NodeIdx| {
                !idx.tombstoned[n as usize] && evaluator.evaluate(idx.metadata[n as usize].as_ref())
            };
            knn_search(idx, query, ef_effective, Some(&accept))?
        }
    };

    Ok(scored
        .into_iter()
        .take(params.k)
        .map(|s| Hit {
            id: idx.ids[s.node as usize].clone(),
            distance: s.dist,
            metadata: idx.metadata[s.node as usize].clone(),
        })
        .collect())
}

/// Which nodes may enter a SEARCH-LAYER result beam. `None` admits
/// every node.
pub(crate) type Accept<'a> = Option<&'a dyn Fn(NodeIdx) -> bool>;

/// HNSW k-NN search (Alg 5): greedy-descend through the upper layers
/// at `ef = 1`, then SEARCH-LAYER at layer 0 with the supplied `ef`.
/// Returns up to `ef` Scored entries in ascending-distance order, all
/// admitted by `accept`.
pub(crate) fn knn_search(
    idx: &HnswIndex,
    query: &[f32],
    ef: usize,
    accept: Accept<'_>,
) -> Result<Vec<Scored>> {
    let entry = match idx.entry {
        Some(e) => e,
        None => return Ok(Vec::new()),
    };
    let entry_dist = distance_to(idx, query, entry)?;
    let mut cur = Scored {
        dist: entry_dist,
        seq: idx.seqs[entry as usize],
        node: entry,
    };

    // Greedy descent through layers `top_layer..=1` with ef=1. This is
    // routing only, so every node, deleted or not, is a valid step.
    let mut layer = idx.top_layer;
    while layer >= 1 {
        let result_heap = search_layer(idx, query, &[cur], layer, 1, None)?;
        if let Some(nearest) = best_of(&result_heap) {
            cur = nearest;
        }
        layer -= 1;
    }

    let result_heap = search_layer(idx, query, &[cur], 0, ef, accept)?;
    Ok(take_topk_sorted(result_heap, ef))
}

/// HNSW SEARCH-LAYER (Alg 2): beam search at one layer.
///
/// `entries` seed the candidate min-heap (every entry is "visited"),
/// and those admitted by `accept` also seed the result max-heap. The
/// loop pops the nearest unexpanded candidate, batches the distance
/// computation to its unvisited neighbours through
/// [`iqdb_distance::compute_batch`], and prunes against the result
/// heap's worst-best entry. Nodes `accept` rejects are still expanded
/// (they keep the graph connected) but never enter the result heap,
/// so the stop check only fires once `ef` admitted nodes are held.
pub(crate) fn search_layer(
    idx: &HnswIndex,
    query: &[f32],
    entries: &[Scored],
    layer: u8,
    ef: usize,
    accept: Accept<'_>,
) -> Result<BinaryHeap<Scored>> {
    // A result heap can never hold more than every stored node, so the
    // node count bounds every buffer whatever `ef` the caller asked for.
    let n_nodes = idx.vectors.len().max(1);
    let ef = ef.clamp(1, n_nodes);
    let admits = |n: NodeIdx| accept.is_none_or(|f| f(n));
    let mut visited: HashSet<NodeIdx> = HashSet::with_capacity(ef.saturating_mul(2).min(n_nodes));
    let mut candidates: BinaryHeap<Reverse<Scored>> = BinaryHeap::with_capacity(ef);
    let mut results: BinaryHeap<Scored> = BinaryHeap::with_capacity(ef);

    for e in entries {
        if !visited.insert(e.node) {
            continue;
        }
        candidates.push(Reverse(*e));
        if admits(e.node) {
            push_to_results(&mut results, *e, ef);
        }
    }

    // Peek the nearest unexpanded candidate first. If it is already
    // worse than the worst entry in the (full) result heap, stop
    // before doing the pop+expand work — this is the standard Alg 2
    // stop check. `if let` is nested (rather than `&& let` chained)
    // to stay on the MSRV 1.87 stable feature set.
    while let Some(Reverse(next)) = candidates.peek().copied() {
        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if cmp_distance(next.dist, worst.dist) == Ordering::Greater {
                    break;
                }
            }
        }
        let _ = candidates.pop();
        let c = next;

        let c_node = c.node as usize;
        let c_layers = &idx.layers[c_node];
        if (layer as usize) >= c_layers.len() {
            // c does not sit on this layer — nothing to traverse.
            continue;
        }
        let adj = &c_layers[layer as usize];

        // Batch the unvisited neighbours so the distance kernel sees
        // them as one `compute_batch` call. This amortises the
        // SIMD/scalar dispatch cost per pop.
        let mut new_neighbours: Vec<NodeIdx> = Vec::with_capacity(adj.len());
        for &n in adj {
            if visited.insert(n) {
                new_neighbours.push(n);
            }
        }
        if new_neighbours.is_empty() {
            continue;
        }

        let slices: Vec<&[f32]> = new_neighbours
            .iter()
            .map(|&n| &idx.vectors[n as usize][..])
            .collect();
        let mut dists = vec![0.0_f32; new_neighbours.len()];
        compute_batch(idx.metric, query, &slices, &mut dists)?;

        for (i, &n) in new_neighbours.iter().enumerate() {
            let mut d = dists[i];
            if matches!(idx.metric, DistanceMetric::DotProduct) {
                d = -d;
            }
            let scored = Scored {
                dist: d,
                seq: idx.seqs[n as usize],
                node: n,
            };
            // Standard Alg 2: only push to candidates when this
            // neighbour is competitive with the current beam (better
            // than the worst result, or the result heap isn't full
            // yet). Unconditional push pollutes the min-heap with
            // far candidates that we pop one-by-one, effectively
            // shrinking the beam.
            let competitive = if results.len() < ef {
                true
            } else if let Some(worst) = results.peek() {
                scored < *worst
            } else {
                true
            };
            if competitive {
                candidates.push(Reverse(scored));
                if admits(n) {
                    push_to_results(&mut results, scored, ef);
                }
            }
        }
    }

    Ok(results)
}

/// Distance from `query` to a single graph node, with the
/// "smaller is nearer" sign convention applied (DotProduct is
/// negated at the boundary).
pub(crate) fn distance_to(idx: &HnswIndex, query: &[f32], node: NodeIdx) -> Result<f32> {
    let mut buf = [0.0_f32; 1];
    let slice: [&[f32]; 1] = [&idx.vectors[node as usize][..]];
    compute_batch(idx.metric, query, &slice, &mut buf)?;
    let raw = buf[0];
    Ok(if matches!(idx.metric, DistanceMetric::DotProduct) {
        -raw
    } else {
        raw
    })
}

/// Distance between two graph nodes (used by the heuristic's
/// neighbour-vs-neighbour comparisons).
pub(crate) fn distance_between(idx: &HnswIndex, a: NodeIdx, b: NodeIdx) -> Result<f32> {
    let mut buf = [0.0_f32; 1];
    let slice: [&[f32]; 1] = [&idx.vectors[b as usize][..]];
    compute_batch(idx.metric, &idx.vectors[a as usize], &slice, &mut buf)?;
    let raw = buf[0];
    Ok(if matches!(idx.metric, DistanceMetric::DotProduct) {
        -raw
    } else {
        raw
    })
}

/// Push `scored` to the bounded result max-heap. Maintains
/// `results.len() <= ef`; evicts the worst when a better entry
/// arrives.
fn push_to_results(results: &mut BinaryHeap<Scored>, scored: Scored, ef: usize) {
    if results.len() < ef {
        results.push(scored);
    } else if let Some(worst) = results.peek() {
        if scored < *worst {
            let _evicted = results.pop();
            results.push(scored);
        }
    }
}

/// Best (smallest distance) entry currently in the result heap, or
/// `None` if empty. Used at the boundary of greedy-descent steps to
/// pick the next layer's entry candidate.
fn best_of(results: &BinaryHeap<Scored>) -> Option<Scored> {
    results.iter().min().copied()
}
