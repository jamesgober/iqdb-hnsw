//! [`Scored`] — one scored graph node — and helpers for the
//! bounded-heap result selection inside SEARCH-LAYER and at the
//! query boundary.
//!
//! Ordering is `(dist, seq)`: smaller distance wins, and ties on
//! distance are broken by smaller insertion sequence — the same
//! tiebreaker `iqdb-flat` uses, for the same reason: deterministic
//! results across runs without coupling to storage position.
//!
//! `f32` ordering goes through [`f32::total_cmp`]; `partial_cmp`
//! returns `None` on NaN, which would panic any `BinaryHeap`
//! operation that round-trips through `Ord`. `total_cmp` defines
//! a total order on every f32 payload a distance computation
//! might produce.

use core::cmp::Ordering;
use std::collections::BinaryHeap;

/// One scored graph node: a distance, the node's insertion-sequence
/// number (the stable tiebreaker), and the node's index into the
/// columnar storage.
///
/// Ordering keys off `(dist, seq)`; `node` is carried for output
/// addressing only and does not affect comparison.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Scored {
    pub(crate) dist: f32,
    pub(crate) seq: u64,
    pub(crate) node: u32,
}

impl Scored {
    fn cmp_key(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then(self.seq.cmp(&other.seq))
    }
}

impl PartialEq for Scored {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_key(other) == Ordering::Equal
    }
}
impl Eq for Scored {}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_key(other)
    }
}

/// Drain the SEARCH-LAYER result max-heap into a Vec of the best
/// `k` entries in ascending order (best-first).
///
/// `heap` is a max-heap keyed by `Scored`'s `(dist, seq)` ordering:
/// its root is the "worst-best" entry seen during the search. After
/// pulling out the top `k`, the entries are returned best-first so
/// the caller can map them directly into `Hit`s.
///
/// Returns an empty `Vec` if `k == 0` or `heap` is empty. If `k`
/// exceeds the heap size, returns every entry.
pub(crate) fn take_topk_sorted(heap: BinaryHeap<Scored>, k: usize) -> Vec<Scored> {
    if k == 0 {
        return Vec::new();
    }
    let mut sorted = heap.into_sorted_vec();
    if sorted.len() > k {
        sorted.truncate(k);
    }
    sorted
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn s(dist: f32, seq: u64, node: u32) -> Scored {
        Scored { dist, seq, node }
    }

    #[test]
    fn ordering_is_by_distance_then_seq() {
        assert!(s(1.0, 100, 0) < s(2.0, 0, 1));
        assert!(s(1.0, 0, 0) < s(1.0, 1, 1));
        assert_eq!(s(1.0, 7, 0), s(1.0, 7, 99));
    }

    #[test]
    fn take_topk_sorted_returns_best_first() {
        let mut heap = BinaryHeap::new();
        for (dist, node) in [(5.0, 0), (1.0, 1), (4.0, 2), (2.0, 3), (3.0, 4)] {
            heap.push(s(dist, node as u64, node));
        }
        let top = take_topk_sorted(heap, 3);
        let nodes: Vec<u32> = top.iter().map(|x| x.node).collect();
        assert_eq!(nodes, vec![1, 3, 4]);
    }

    #[test]
    fn take_topk_sorted_k_zero_is_empty() {
        let mut heap = BinaryHeap::new();
        heap.push(s(1.0, 0, 0));
        assert!(take_topk_sorted(heap, 0).is_empty());
    }

    #[test]
    fn take_topk_sorted_k_greater_than_heap_returns_all() {
        let mut heap = BinaryHeap::new();
        heap.push(s(2.0, 1, 1));
        heap.push(s(1.0, 0, 0));
        let top = take_topk_sorted(heap, 10);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].node, 0);
        assert_eq!(top[1].node, 1);
    }

    #[test]
    fn nan_is_handled_via_total_cmp() {
        let mut heap = BinaryHeap::new();
        heap.push(s(f32::NAN, 0, 0));
        heap.push(s(1.0, 1, 1));
        heap.push(s(2.0, 2, 2));
        let top = take_topk_sorted(heap, 3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].node, 1);
        assert_eq!(top[1].node, 2);
        assert!(top[2].dist.is_nan());
    }
}
