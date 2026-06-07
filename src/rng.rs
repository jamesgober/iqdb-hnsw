//! A small seeded deterministic PRNG.
//!
//! [`SplitMix64`] is the standard 64-bit `SplitMix` generator from Vigna,
//! used here as the seed source for HNSW's layer-assignment loop. It is
//! intentionally hand-rolled and dependency-free so the determinism
//! contract — same `seed` + same insert order → same graph — does not
//! depend on an external crate's version drift.
//!
//! The generator is not cryptographically secure and is not intended to
//! be. It is fast, has good enough distribution for the small floating
//! draws layer assignment makes, and produces the same byte sequence on
//! every platform.

/// A deterministic 64-bit PRNG seeded once at construction.
///
/// State is a single `u64` updated by Vigna's SplitMix64 mix function
/// on each [`next_u64`](SplitMix64::next_u64) call. Cloning the
/// generator copies the state — two clones at the same point produce
/// identical subsequent draws.
#[derive(Debug, Clone)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Build a generator seeded with `seed`.
    ///
    /// A `seed` of `0` is fine — the SplitMix mix function does not
    /// degenerate on a zero state.
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the generator and return the next `u64`.
    ///
    /// The mix constants are from Vigna's reference implementation.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Sample an `f64` in `(0, 1]`.
    ///
    /// The HNSW layer-assignment formula `floor(-ln(u) * mL)` requires
    /// `u > 0` to avoid `ln(0) = -inf`; this method re-rolls if the
    /// top-53-bit mantissa draw is zero, and falls back to `2^-53`
    /// after a bounded number of attempts so termination is provable
    /// from the source.
    pub(crate) fn next_open_unit(&mut self) -> f64 {
        for _ in 0..4 {
            let bits = self.next_u64() >> 11;
            if bits != 0 {
                return (bits as f64) * (1.0 / ((1_u64 << 53) as f64));
            }
        }
        1.0 / ((1_u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn same_seed_produces_same_sequence() {
        let mut a = SplitMix64::new(0xCAFE_F00D);
        let mut b = SplitMix64::new(0xCAFE_F00D);
        for _ in 0..1024 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        let mut equal = 0;
        for _ in 0..1024 {
            if a.next_u64() == b.next_u64() {
                equal += 1;
            }
        }
        assert!(equal < 4, "too much agreement: {equal}/1024");
    }

    #[test]
    fn open_unit_is_in_unit_interval_exclusive_zero() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..10_000 {
            let u = rng.next_open_unit();
            assert!(u > 0.0 && u <= 1.0, "u = {u}");
        }
    }

    #[test]
    fn open_unit_never_returns_zero_from_zero_seed() {
        let mut rng = SplitMix64::new(0);
        for _ in 0..10_000 {
            assert!(rng.next_open_unit() > 0.0);
        }
    }
}
