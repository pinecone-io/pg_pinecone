//! Exact distance/similarity math shared by k-means training, exact-rescore,
//! and the small-N bootstrap's brute-force scan. Sign/scale conventions match
//! pgvector's own operators exactly (not just "some sensible distance") since
//! the tests compare the local engine's output order/values against
//! `ORDER BY col <-> $1` etc. run straight through pgvector:
//! - `vector_l2_ops` (`<->`): Euclidean distance, `sqrt(sum((a_i-b_i)^2))`.
//! - `vector_cosine_ops` (`<=>`): *cosine distance*, `1 - cosine_similarity`.
//! - `vector_ip_ops` (`<#>`): *negative* inner product, `-(a . b)` — pgvector's
//!   own choice, so that ascending order (closest first) still means
//!   "smallest value first" like the other two.

use super::distance;

/// The three opclasses `am::metric_for_opclass` maps to — plain `u8` on disk
/// (metapage v2's `metric` field), not a bare `String`, so the metapage has a
/// fixed-width, allocation-free representation like every other field in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Metric {
    L2 = 0,
    Cosine = 1,
    Ip = 2,
}

impl Metric {
    pub(crate) fn from_opclass_metric(metric: &str) -> Self {
        match metric {
            "l2" => Metric::L2,
            "cosine" => Metric::Cosine,
            "ip" => Metric::Ip,
            other => panic!("pg_pinecone: unknown metric '{other}' — should be unreachable"),
        }
    }

    /// Round-trips through the metapage's `u8` encoding — panics on a value
    /// this build never wrote itself (metapage version validation is what
    /// protects this from ever seeing a foreign/corrupt byte in practice).
    pub(crate) fn from_u8(v: u8) -> Self {
        match v {
            0 => Metric::L2,
            1 => Metric::Cosine,
            2 => Metric::Ip,
            other => panic!("pg_pinecone: corrupt metapage — unknown metric code {other}"),
        }
    }

    pub(crate) fn as_u8(self) -> u8 {
        self as u8
    }
}

pub(crate) fn l2_norm(a: &[f32]) -> f32 {
    distance::dot(a, a).sqrt()
}

/// The exact distance `am::metric_for_opclass`'s metric would compute for
/// `a`/`b` — used for k-means assignment (which needs *a* distance, exact or
/// not) and exact-rescore (which needs *the* distance pgvector's own operator
/// would return, since tests compare against it directly).
pub(crate) fn exact_distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    key_to_distance(metric, exact_key(metric, a, b))
}

/// The exact distance in the units of the RaBitQ keys: [`exact_distance`] but squared for L2, so the
/// scan can compare it with a key.
pub(crate) fn exact_key(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::L2 => distance::l2_sq(a, b),
        Metric::Cosine => distance::cosine(a, b),
        Metric::Ip => -distance::dot(a, b),
    }
}

/// [`exact_key`] back to the distance pgvector's operator returns.
pub(crate) fn key_to_distance(metric: Metric, key: f32) -> f32 {
    match metric {
        Metric::L2 => key.sqrt(),
        Metric::Cosine | Metric::Ip => key,
    }
}

/// `v / |v|` into `out` — the form a cosine index encodes and scores (a zero vector, and one whose
/// norm overflows, is copied as it is: the exact rescore decides those rows).
pub(crate) fn normalise_into(v: &[f32], out: &mut Vec<f32>) {
    let norm = l2_norm(v);
    out.clear();
    if norm > 0.0 && norm.is_finite() {
        out.extend(v.iter().map(|x| x / norm));
    } else {
        out.extend_from_slice(v);
    }
}

/// The position of the first NaN or infinite component of `v`: the native engine encodes and
/// scores finite vectors only (pgvector never stores any other), so a caller raises on `Some`.
pub(crate) fn first_non_finite(v: &[f32]) -> Option<usize> {
    v.iter().position(|x| !x.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_matches_hand_computation() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 6.0, 3.0];
        // sqrt(9 + 16 + 0) = 5
        assert!((exact_distance(Metric::L2, &a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_distance_zero_for_identical_direction() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 4.0, 6.0]; // same direction, different magnitude
        assert!(exact_distance(Metric::Cosine, &a, &b).abs() < 1e-5);
    }

    #[test]
    fn cosine_distance_two_for_opposite_direction() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        assert!((exact_distance(Metric::Cosine, &a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ip_is_negative_dot() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        assert!((exact_distance(Metric::Ip, &a, &b) - (-11.0)).abs() < 1e-6);
    }

    #[test]
    fn normalise_gives_unit_vectors_and_leaves_zero_and_overflowing_ones_alone() {
        let mut out = Vec::new();
        normalise_into(&[3.0, 4.0], &mut out);
        assert!((out[0] - 0.6).abs() < 1e-6 && (out[1] - 0.8).abs() < 1e-6);
        assert!((l2_norm(&out) - 1.0).abs() < 1e-6);
        normalise_into(&[0.0, 0.0, 0.0], &mut out);
        assert_eq!(out, [0.0, 0.0, 0.0]);
        normalise_into(&[3e30, 4e30], &mut out);
        assert_eq!(out, [3e30, 4e30], "an overflowing norm is not divided by");
        let v = lcg_vector(8, 768);
        normalise_into(&v, &mut out);
        assert!((l2_norm(&out) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn non_finite_components_are_found_by_position() {
        assert_eq!(first_non_finite(&[1.0, 2.0, 3.0]), None);
        assert_eq!(first_non_finite(&[1.0, f32::NAN, 3.0]), Some(1));
        assert_eq!(first_non_finite(&[f32::INFINITY, f32::NAN]), Some(0));
        assert_eq!(first_non_finite(&[1.0, f32::NEG_INFINITY]), Some(1));
        assert_eq!(first_non_finite(&[]), None);
    }

    fn lcg_vector(seed: u64, n: usize) -> Vec<f32> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 40) as u32) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn oracle(metric: Metric, a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
        match metric {
            Metric::L2 => a
                .iter()
                .zip(b)
                .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
                .sum::<f64>()
                .sqrt(),
            Metric::Cosine => {
                let na: f64 = a.iter().map(|&x| x as f64 * x as f64).sum();
                let nb: f64 = b.iter().map(|&y| y as f64 * y as f64).sum();
                1.0 - dot / (na.sqrt() * nb.sqrt())
            }
            Metric::Ip => -dot,
        }
    }

    #[test]
    fn exact_distance_matches_the_f64_oracle_for_every_metric() {
        for (i, &dim) in [3usize, 64, 128, 768, 1536].iter().enumerate() {
            let a = lcg_vector(11 + i as u64, dim);
            let b = lcg_vector(97 + i as u64, dim);
            for metric in [Metric::L2, Metric::Cosine, Metric::Ip] {
                let got = exact_distance(metric, &a, &b) as f64;
                let want = oracle(metric, &a, &b);
                let scale = want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= 1e-5 * scale,
                    "{metric:?} dim {dim}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn l2_norm_matches_the_oracle() {
        for dim in [1usize, 17, 768, 2000] {
            let a = lcg_vector(5, dim);
            let want = a.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
            assert!(
                (l2_norm(&a) as f64 - want).abs() <= 1e-5 * want,
                "dim {dim}"
            );
        }
    }

    #[test]
    fn cosine_of_a_zero_vector_is_one_not_nan() {
        let v = lcg_vector(3, 40);
        assert_eq!(exact_distance(Metric::Cosine, &[0.0; 40], &v), 1.0);
        assert_eq!(exact_distance(Metric::Cosine, &v, &[0.0; 40]), 1.0);
    }

    #[test]
    fn metric_u8_round_trips() {
        for m in [Metric::L2, Metric::Cosine, Metric::Ip] {
            assert_eq!(Metric::from_u8(m.as_u8()), m);
        }
    }
}
