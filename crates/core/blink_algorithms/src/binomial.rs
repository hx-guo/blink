//! 二项分布单边尾概率。

use statrs::function::beta::beta_reg;

/// `n` 次试验、单次概率 `p` 时成功 **至少** `k` 次的概率 `P(X ≥ k)`。
///
/// 走正则化不完全贝塔 `I_p(k, n − k + 1)`，不做 `1 − cdf`：电子束的判据在
/// 1e-9 量级，相减会把尾巴吃掉。
pub fn sf_at_least(n: u32, k: u32, p: f64) -> f64 {
    match k {
        0 => 1.0,
        k if k > n => 0.0,
        k => beta_reg(k as f64, (n - k + 1) as f64, p),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_direct_sum() {
        let direct = |n: u32, k: u32, p: f64| -> f64 {
            (k..=n)
                .map(|i| {
                    let ln_choose = statrs::function::factorial::ln_binomial(n as u64, i as u64);
                    (ln_choose + i as f64 * p.ln() + (n - i) as f64 * (1.0 - p).ln()).exp()
                })
                .sum()
        };
        for &(n, k, p) in &[(10, 3, 0.5), (40, 35, 0.56), (60, 60, 0.56), (25, 0, 0.3)] {
            let expected = direct(n, k, p);
            let got = sf_at_least(n, k, p);
            assert!(
                (got - expected).abs() <= 1e-12 * expected.max(1e-300) + 1e-15,
                "n={n} k={k}: {got} vs {expected}"
            );
        }
        assert_eq!(sf_at_least(5, 6, 0.5), 0.0);
        // 尾巴深处仍有相对精度：60 次全中，0.56^60 ≈ 7.6e-16
        assert!((sf_at_least(60, 60, 0.56) / 0.56f64.powi(60) - 1.0).abs() < 1e-9);
    }
}
