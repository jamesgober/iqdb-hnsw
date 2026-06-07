//! HNSW insert (Alg 1) and the heuristic neighbour-selection (Alg 4).
//!
//! `insert_node` is the trait-impl entry point; it owns the borrow on
//! `&mut HnswIndex`, allocates the row, runs greedy descent and
//! `search_layer` to find candidates, picks diverse neighbours via
//! the heuristic, wires up the bidirectional links, prunes any
//! neighbour that overflows its layer cap, and bumps the global
//! entry point if the new node's top layer is the new high.
//!
//! `select_heuristic` is Alg 4 from the paper: walk candidates in
//! ascending distance, admit `c` only if no already-selected
//! neighbour `s` is closer to `c` than the query is. This produces a
//! diverse neighbourhood (better long-range connectivity) instead of
//! the M nearest.

use std::sync::Arc;

use iqdb_types::{IqdbError, Metadata, Result, VectorId};

use crate::graph::{NodeIdx, cap_at_layer, pick_layer};
use crate::index::HnswIndex;
use crate::search::{distance_between, distance_to, search_layer};
use crate::topk::{Scored, take_topk_sorted};

/// Insert one vector into the HNSW graph (Alg 1).
///
/// Returns [`IqdbError::DimensionMismatch`] for a vector that does
/// not match `idx.dim`, [`IqdbError::Duplicate`] for an already-known
/// id, and [`IqdbError::InvalidConfig`] for an internal sequence
/// counter overflow (a contract violation reachable only after
/// `u64::MAX` inserts on a single index).
pub(crate) fn insert_node(
    idx: &mut HnswIndex,
    id: VectorId,
    vector: Arc<[f32]>,
    metadata: Option<Metadata>,
) -> Result<()> {
    idx.check_dim(vector.len())?;
    if idx.id_to_node.contains_key(&id) {
        return Err(IqdbError::Duplicate);
    }

    let seq = idx.next_seq;
    let next_seq = idx
        .next_seq
        .checked_add(1)
        .ok_or(IqdbError::InvalidConfig {
            reason: "HnswIndex insertion sequence counter overflowed u64",
        })?;
    idx.next_seq = next_seq;

    let layer = pick_layer(&mut idx.rng, idx.m_l_inv);

    // Allocate row + empty per-layer adjacency. The new node sits at
    // every layer `0..=layer`.
    let new_node: NodeIdx = idx.vectors.len() as NodeIdx;
    idx.vectors.push(Arc::clone(&vector));
    idx.ids.push(id.clone());
    idx.metadata.push(metadata);
    idx.seqs.push(seq);
    idx.tombstoned.push(false);
    idx.node_layer.push(layer);
    let mut per_layer_adj: Vec<Vec<NodeIdx>> = Vec::with_capacity((layer as usize) + 1);
    for lc in 0..=layer {
        per_layer_adj.push(Vec::with_capacity(cap_at_layer(idx.cfg.m, lc)));
    }
    idx.layers.push(per_layer_adj);
    let _prev = idx.id_to_node.insert(id, new_node);

    // First-insert short-circuit. Set entry, bump live count, done.
    let entry = match idx.entry {
        Some(e) => e,
        None => {
            idx.entry = Some(new_node);
            idx.top_layer = layer;
            idx.live_count = idx.live_count.saturating_add(1);
            return Ok(());
        }
    };

    // Greedy descent from the global entry through layers above the
    // new node's layer (ef = 1).
    let entry_dist = distance_to(idx, &vector, entry)?;
    let mut cur = Scored {
        dist: entry_dist,
        seq: idx.seqs[entry as usize],
        node: entry,
    };
    let top = idx.top_layer;
    if top > layer {
        let mut lc = top;
        while lc > layer {
            let result_heap = search_layer(idx, &vector, &[cur], lc, 1)?;
            if let Some(nearest) = result_heap.iter().min().copied() {
                cur = nearest;
            }
            lc -= 1;
        }
    }

    // Insert at each layer from min(layer, top) down to 0.
    let mut entry_points: Vec<Scored> = vec![cur];
    let start_lc = layer.min(top);
    let mut lc = start_lc;
    loop {
        let result_heap = search_layer(idx, &vector, &entry_points, lc, idx.cfg.ef_construction)?;
        let w_sorted = take_topk_sorted(result_heap, idx.cfg.ef_construction);

        let live_candidates: Vec<Scored> = w_sorted
            .iter()
            .copied()
            .filter(|s| !idx.tombstoned[s.node as usize] && s.node != new_node)
            .collect();
        let m_cap = cap_at_layer(idx.cfg.m, lc);
        let chosen = select_heuristic(idx, &vector, &live_candidates, m_cap)?;

        // Bidirectional links. When `s`'s neighbourhood overflows
        // the layer cap, re-trim it but pin `new_node` so the link
        // we just established stays in place. Letting the heuristic
        // silently drop it would leave the new node unreachable
        // from `s` even though `new_node → s` remains, breaking the
        // observable bidirectional invariant the recall test
        // depends on.
        for s in &chosen {
            idx.layers[new_node as usize][lc as usize].push(s.node);
            idx.layers[s.node as usize][lc as usize].push(new_node);
            if idx.layers[s.node as usize][lc as usize].len() > m_cap {
                trim_neighbourhood(idx, s.node, lc, m_cap, Some(new_node))?;
            }
        }

        // Next iteration's entries: the full W from this layer
        // (per Alg 1's `ep ← W`).
        entry_points = w_sorted;
        if lc == 0 {
            break;
        }
        lc -= 1;
    }

    if layer > idx.top_layer {
        idx.entry = Some(new_node);
        idx.top_layer = layer;
    }

    idx.live_count = idx.live_count.saturating_add(1);
    Ok(())
}

/// Diagnostic-only simple top-M selection (Alg 3). Not currently
/// used; kept available for A/B experiments against the heuristic.
#[allow(dead_code)]
fn select_simple(_idx: &HnswIndex, candidates: &[Scored], m_max: usize) -> Vec<Scored> {
    if m_max == 0 || candidates.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<Scored> = candidates.to_vec();
    sorted.sort();
    sorted.truncate(m_max);
    sorted
}

/// HNSW heuristic neighbour selection (Alg 4) with the standard
/// `keepPrunedConnections = true` top-up.
///
/// Walks `candidates` in ascending `(dist, seq)` order. Admits `c`
/// when no already-selected neighbour `s` is closer to `c` than the
/// query is; otherwise `s` "covers" `c` and `c` is stashed as
/// pruned. After the diverse pass, if fewer than `m_max` were
/// selected, the pruned candidates are added back in their original
/// distance order until `m_max` is reached.
///
/// The top-up matches what reference HNSW implementations
/// (including hnswlib) do for layer 0 and is what carries recall
/// above 0.95 at the default parameters on random corpora — without
/// it, a tight cluster's candidates all "cover" each other and the
/// returned set can be much smaller than `m_max`, leaving the graph
/// sparse.
pub(crate) fn select_heuristic(
    idx: &HnswIndex,
    query: &[f32],
    candidates: &[Scored],
    m_max: usize,
) -> Result<Vec<Scored>> {
    let _ = query;
    if m_max == 0 || candidates.is_empty() {
        return Ok(Vec::new());
    }
    let mut sorted: Vec<Scored> = candidates.to_vec();
    sorted.sort();
    let mut selected: Vec<Scored> = Vec::with_capacity(m_max);
    let mut pruned: Vec<Scored> = Vec::new();

    for c in sorted {
        if selected.len() >= m_max {
            break;
        }
        let mut covered = false;
        for s in &selected {
            let d_cs = distance_between(idx, c.node, s.node)?;
            if d_cs < c.dist {
                covered = true;
                break;
            }
        }
        if covered {
            pruned.push(c);
        } else {
            selected.push(c);
        }
    }
    // keepPrunedConnections top-up.
    if selected.len() < m_max {
        for c in pruned {
            if selected.len() >= m_max {
                break;
            }
            selected.push(c);
        }
    }
    Ok(selected)
}

/// Re-run the heuristic over `node`'s existing neighbourhood at
/// `layer` to fit it back into `cap` after a fresh bidirectional
/// link pushed it over.
///
/// `pinned`, when supplied, MUST end up in the trimmed adjacency —
/// even if the heuristic would otherwise have discarded it. This is
/// used by the insert path to preserve the just-added link from
/// `node` to the new node; without it, search-time traversal from
/// `node` cannot reach the new node and recall sags by several
/// percent on the headline test.
fn trim_neighbourhood(
    idx: &mut HnswIndex,
    node: NodeIdx,
    layer: u8,
    cap: usize,
    pinned: Option<NodeIdx>,
) -> Result<()> {
    let current_adj: Vec<NodeIdx> = idx.layers[node as usize][layer as usize].clone();
    let mut candidates: Vec<Scored> = Vec::with_capacity(current_adj.len());
    for &nb in &current_adj {
        let d = distance_between(idx, node, nb)?;
        candidates.push(Scored {
            dist: d,
            seq: idx.seqs[nb as usize],
            node: nb,
        });
    }
    let node_vec = Arc::clone(&idx.vectors[node as usize]);
    let mut chosen = select_heuristic(idx, &node_vec, &candidates, cap)?;

    if let Some(pin) = pinned {
        let already_in = chosen.iter().any(|s| s.node == pin);
        if !already_in {
            // Find the pinned candidate's Scored entry to swap in.
            if let Some(pin_scored) = candidates.iter().copied().find(|s| s.node == pin) {
                if chosen.len() >= cap {
                    // Drop the worst chosen (largest dist) so the
                    // pin fits without growing past the cap.
                    if let Some((worst_idx, _)) =
                        chosen.iter().enumerate().max_by(|(_, a), (_, b)| a.cmp(b))
                    {
                        let _evicted = chosen.swap_remove(worst_idx);
                    }
                }
                chosen.push(pin_scored);
            }
        }
    }

    idx.layers[node as usize][layer as usize] = chosen.iter().map(|s| s.node).collect();
    Ok(())
}
