//! A tiny, dependency-free, deterministic PRNG used everywhere the local
//! engine needs reproducible randomness (the RaBitQ rotation's signs and
//! permutations — [`crate::local::rotation`] — the query rounding offsets
//! — [`crate::local::rabitq`] — and k-means sampling/initialization —
//! [`crate::local::kmeans`]): each must regenerate *exactly* the same
//! sequence of "random" values on every call given the same `seed`, since
//! neither the rotation nor the build's sample selection is persisted to
//! disk — a codes page is only meaningful against the rotation its seed
//! regenerates.
//!
//! [SplitMix64](https://prng.di.unimi.it/splitmix64.c) (Steele, Lea, Flood
//! 2014) rather than pulling in the `rand` crate: it's ~15 lines, has no
//! transitive dependencies to keep in step with pgrx's own dependency tree,
//! and its statistical quality is more than sufficient for generating random
//! signs and a Fisher–Yates permutation (neither is cryptographic, and
//! SplitMix64 is a standard, well-studied choice for exactly this kind of
//! seeded-simulation use).

/// Domain-separation constants for [`SplitMix64::for_domain`] — fixed
/// before any real index existed to make it a build-format change: two
/// consumers of the metapage's single `seed` used to call
/// `SplitMix64::new(seed)` directly with the SAME value.
/// `SplitMix64::new` doesn't hash its input (`Self(seed)` — the very first
/// `next_u64` call is what starts mixing), so two generators constructed
/// from the identical `seed` produce the identical output sequence — their
/// draws were silently lock-step correlated with each other for every single
/// index, rather than independent as the design (conceptually unrelated
/// randomized components) intends. XOR-ing in a distinct constant per
/// consumer before seeding fixes this: the guarantees this crate relies on
/// (Lloyd's algorithm, the random rotation) hold for *any* fixed random
/// draw, so this changes nothing about correctness — only decorrelates which
/// draw each consumer uses.
pub(crate) mod domain {
    pub(crate) const KMEANS: u64 = 0x4B4D45414E535F31; // ASCII-ish "KMEANS_1"
    /// The RaBitQ rotation's stream (`rotation.rs`): on-disk format, like `fnv1a64` — an index's codes
    /// are only meaningful against the rotation this constant and the seed regenerate. ASCII "ROTATION".
    pub(crate) const ROTATION: u64 = 0x524F544154494F4E;
    /// The query rounding offsets of `rabitq::prepare_query_into`: not on-disk format (they decide
    /// which of two neighbouring 4-bit levels a query coordinate rounds to), so ranks can differ
    /// between builds that change it. ASCII "ROUNDING".
    pub(crate) const ROUNDING: u64 = 0x524F_554E_4449_4E47;
    /// The sample queries of the build-time calibration (`calibrate.rs`): not on-disk format, they
    /// decide which rows stand in for queries. ASCII "CALIBRAT".
    pub(crate) const CALIBRATION: u64 = 0x4341_4C49_4252_4154;
}

/// A SplitMix64 generator, seeded once and then advanced via [`next_u64`](Self::next_u64).
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// [`new`](Self::new), but XOR-ing in one of [`domain`]'s constants first
    /// — see [`domain`]'s own rustdoc for why every caller seeding from a
    /// shared, index-wide `seed` value (`kmeans`, the rotation, the query
    /// rounding, all keyed off the metapage's single `seed` field) must go
    /// through this instead of [`new`] directly.
    pub(crate) fn for_domain(seed: u64, domain: u64) -> Self {
        Self::new(seed ^ domain)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// A uniform `f32` in `[0, 1)` — a general-purpose primitive exercised by
    /// this module's and `kmeans`'s own test fixtures (synthetic
    /// data generation); no non-test build code needs it yet.
    #[allow(dead_code)]
    pub(crate) fn next_f32(&mut self) -> f32 {
        // 24 bits of mantissa is all an f32 can represent anyway.
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// A pseudo-random `{-1.0, +1.0}` sign, each with probability 1/2 — the
    /// rotation's random-sign diagonals (Ailon–Chazelle 2006).
    pub(crate) fn next_sign(&mut self) -> f32 {
        if self.next_u64() & 1 == 0 { 1.0 } else { -1.0 }
    }

    /// A uniform index in `0..bound` (`bound > 0`), via Lemire's rejection-free
    /// bias reduction — used by [`shuffle`](Self::shuffle) and k-means's
    /// sampling/reinit picks.
    pub(crate) fn next_bounded(&mut self, bound: u32) -> u32 {
        // 64-bit widening keeps the modulo bias low enough for our use (all
        // `bound`s here are at most a few million — sample sizes/dimensions).
        (((self.next_u64() >> 32) * bound as u64) >> 32) as u32
    }

    /// Fisher–Yates shuffle of the *entire* slice.
    pub(crate) fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_bounded(i as u32 + 1) as usize;
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_given_same_seed() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn domain_separation_decorrelates_same_seed() {
        // The exact bug scenario: consumers fed the SAME metapage `seed`
        // value — must no longer draw identical sequences.
        let draws = |domain: u64| -> Vec<u64> {
            let mut rng = SplitMix64::for_domain(42, domain);
            (0..10).map(|_| rng.next_u64()).collect()
        };
        let all = [
            draws(domain::KMEANS),
            draws(domain::ROTATION),
            draws(domain::ROUNDING),
            draws(domain::CALIBRATION),
        ];
        for i in 0..all.len() {
            for j in i + 1..all.len() {
                assert_ne!(all[i], all[j], "domains {i} and {j}");
            }
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        let seq_a: Vec<u64> = (0..10).map(|_| a.next_u64()).collect();
        let seq_b: Vec<u64> = (0..10).map(|_| b.next_u64()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn next_bounded_stays_in_range() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..10_000 {
            let v = rng.next_bounded(17);
            assert!(v < 17);
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = SplitMix64::new(99);
        let mut v: Vec<usize> = (0..256).collect();
        rng.shuffle(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..256).collect::<Vec<_>>());
    }

    #[test]
    fn next_f32_in_unit_range() {
        let mut rng = SplitMix64::new(3);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "v = {v}");
        }
    }
}
