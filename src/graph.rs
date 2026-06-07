//! Per-layer graph geometry: node index alias, neighbour-cap rule,
//! and the layer-assignment draw.
//!
//! Storage of the graph itself (adjacency lists, columnar per-node
//! arrays, entry point) lives on [`crate::HnswIndex`] directly.
//! This module holds the small, isolated geometry pieces that insert
//! and search both depend on, so the two algorithm modules don't
//! drift in their cap and layer rules.

use crate::rng::SplitMix64;

/// Index of a node inside the columnar storage.
///
/// `u32` is enough for ~4 billion vectors per shard, well above any
/// in-memory corpus this index targets; the four-byte width keeps
/// adjacency lists half the size of `usize` indices on 64-bit
/// platforms, which matters because the SEARCH-LAYER hot loop walks
/// these.
pub(crate) type NodeIdx = u32;

/// The maximum number of neighbours a node may hold at `layer`.
///
/// Layer 0 uses `2 * m` (Malkov–Yashunin's `M0` rule); every higher
/// layer caps at `m`. The cap is a hard contract on the adjacency
/// list lengths: Alg 1's prune step re-runs the heuristic on any
/// neighbour that grows beyond this after a bidirectional link.
#[inline]
pub(crate) const fn cap_at_layer(m: usize, layer: u8) -> usize {
    if layer == 0 { 2 * m } else { m }
}

/// Pick the top layer for a freshly-inserted node.
///
/// Formula from Malkov–Yashunin: `l = floor(-ln(u) * mL)` with `u`
/// drawn from `(0, 1]` and `mL = 1 / ln(M)`. The result is bounded
/// at [`u8::MAX`] so the per-node `node_layer: Vec<u8>` field can
/// hold it; that ceiling is purely defensive — typical corpora
/// produce layer assignments well below 30.
///
/// `m_l_inv` is the precomputed `1.0 / ln(M)` so callers do not pay
/// a logarithm per insert.
pub(crate) fn pick_layer(rng: &mut SplitMix64, m_l_inv: f64) -> u8 {
    let u = rng.next_open_unit();
    let raw = (-u.ln()) * m_l_inv;
    let layer = raw.floor();
    if !layer.is_finite() || layer < 0.0 {
        return 0;
    }
    if layer >= u8::MAX as f64 {
        return u8::MAX;
    }
    layer as u8
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn cap_at_layer_zero_is_two_m() {
        assert_eq!(cap_at_layer(16, 0), 32);
        assert_eq!(cap_at_layer(8, 0), 16);
    }

    #[test]
    fn cap_at_layer_above_zero_is_m() {
        assert_eq!(cap_at_layer(16, 1), 16);
        assert_eq!(cap_at_layer(16, 5), 16);
    }

    #[test]
    fn pick_layer_distribution_decays_geometrically_at_m16() {
        let m = 16usize;
        let m_l_inv = 1.0 / (m as f64).ln();
        let n = 20_000usize;
        let mut rng = SplitMix64::new(1);
        let mut counts = [0usize; 16];
        for _ in 0..n {
            let l = pick_layer(&mut rng, m_l_inv);
            let bucket = (l as usize).min(15);
            counts[bucket] += 1;
        }
        let layer0_frac = counts[0] as f64 / n as f64;
        assert!(
            (layer0_frac - 0.9375).abs() < 0.03,
            "layer-0 fraction {layer0_frac} not near 15/16",
        );
        assert!(counts[1] < counts[0]);
        assert!(counts[0] > counts[1]);
        assert!(counts[1] >= counts[2]);
    }

    #[test]
    fn pick_layer_does_not_panic_on_extreme_seed() {
        let mut rng = SplitMix64::new(0xFFFF_FFFF_FFFF_FFFF);
        let _layer = pick_layer(&mut rng, 1.0 / (16.0_f64).ln());
    }
}
