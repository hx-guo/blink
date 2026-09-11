use crate::constants::DAYS_PER_YEAR;
use statrs::distribution::{DiscreteCDF, Poisson};
use uom::si::f64::*;

/// 观测到 `count` 个计数、本底期望 `mean` 时的单边 p 值，即 **`P(X ≥ count)`**。
///
/// # 这里曾经少算一项，2026-09-11 改
///
/// 原来直接调 statrs 的 `DiscreteCDF::sf(count)`，而它是 **`P(X > count)`**
/// ——**把"恰好观测到这么多"那一项漏掉了**。在我们关心的极端尾巴上那一项就是
/// 尾巴的主体，所以报出来的 `sf` 与 `fa` 系统性偏小，低估的倍数是
/// **`(count+1)/λ`**：三台仪器实测中位 SVOM 6.54、HXMT 19.62、GECAM-A 1.16e4、
/// GECAM-C 36。
///
/// **改动不要求重跑搜索**：`P(X ≥ k) > P(X > k)` 恒成立 ⇒ 每一格的 `fa` 逐点
/// 只增 ⇒ 取最小格之后也只增 ⇒ **旧口径选出的候选池是新口径那个池的超集**，
/// 离线拿 `count` 与 `mean` 重算即可，一个候选都不会漏。
/// **但离线重算的是旧的最佳格**：`sf` 也参与最佳格的选择，
/// `min_b sf_new(b) ≤ sf_new(best_old)`，**所以离线值是新 `fa` 的上界**
/// ——"离线判过阈"是硬的，"离线判掉出"只是掉出数的上界。
pub fn sf(mean: f64, count: u32) -> f64 {
    match (mean, count) {
        // 观测到 ≥ 0 个是必然事件，与本底多大无关
        (_, 0) => 1.0,
        // 本底期望为零时 `Poisson::new` 构造不出分布。数学上 `P(X ≥ k) = 0`
        // （k ≥ 1），但那等于宣称"无穷显著"，而 `mean == 0` 恰恰意味着这一格
        // 没有可用的本底估计。返回 1.0 是保守的那一侧，沿用改动前的取值。
        (0.0, _) => 1.0,
        _ => Poisson::new(mean)
            .inspect_err(|e| {
                eprintln!("Error in Poisson distribution: {}", e);
                eprintln!("Mean: {}", mean);
            })
            .unwrap()
            // `DiscreteCDF::sf(x)` 是 `P(X > x)`，所以 `P(X ≥ count)` 要取
            // `sf(count − 1)`。上面已经挡掉 `count == 0`，这里不会下溢。
            .sf((count - 1) as u64),
    }
}

pub fn false_positive_per_year(sf: f64, duration: Time) -> f64 {
    sf * (Time::new::<uom::si::time::second>(3600.0 * 24.0 * DAYS_PER_YEAR) / duration)
        .get::<uom::si::ratio::ratio>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::{DiscreteCDF, Poisson};
    use statrs::function::gamma::ln_gamma;

    /// 与被测实现无关的参照：直接按定义把 PMF 从 `count` 往上加。
    ///
    /// **单项必须在对数空间算**——极端尾巴上 `P(X = k)` 会小到 `1e−300` 以下，
    /// 中间量先乘再除会整段下溢成 0，看着像"算出来是零"，其实是把最显著那一批
    /// 悄悄丢了。
    fn tail_from_pmf(mean: f64, count: u32) -> f64 {
        let mut total = 0.0_f64;
        // 均值 + 若干倍标准差之外没有可观质量，加到那里就够
        let top = (count as f64 + mean + 50.0 * (mean.sqrt() + 1.0)) as u32;
        for k in count..=top {
            total += (-mean + k as f64 * mean.ln() - ln_gamma(k as f64 + 1.0)).exp();
        }
        total
    }

    #[test]
    fn the_tail_includes_the_observed_count_itself() {
        // 这一条就是 2026-09-11 修掉的那个错：漏掉了 `P(X = count)` 那一项
        for &(mean, count) in &[(0.19_f64, 9_u32), (1.2, 12), (0.1, 12), (10.0, 30)] {
            let ours = sf(mean, count);
            let reference = tail_from_pmf(mean, count);
            assert!(
                (ours - reference).abs() <= reference * 1e-9,
                "sf({mean}, {count}) = {ours}，独立实现给 {reference}"
            );
        }
    }

    #[test]
    fn the_old_semantics_differ_by_the_factor_we_measured() {
        // 低估倍数应当是 (count+1)/λ —— 三台仪器上实测到的就是这个比值
        for &(mean, count) in &[(0.19_f64, 9_u32), (1.2, 12), (0.1, 12), (0.0012, 8)] {
            let strictly_greater = Poisson::new(mean).unwrap().sf(count as u64);
            let ratio = sf(mean, count) / strictly_greater;
            let predicted = (count as f64 + 1.0) / mean;
            assert!(
                (ratio / predicted - 1.0).abs() < 0.05,
                "mean={mean} count={count}：实测倍数 {ratio}，(count+1)/λ 给 {predicted}"
            );
        }
    }

    #[test]
    fn the_correction_only_ever_increases_the_tail() {
        // 候选池严格套嵌就靠这一条：`P(X ≥ k) > P(X > k)` 恒成立 ⇒ 旧口径选出的
        // 池是新口径那个池的超集 ⇒ 改动不要求重跑搜索。
        for count in 1..40_u32 {
            for mean in [0.001_f64, 0.05, 0.5, 2.0, 9.0] {
                let strictly_greater = Poisson::new(mean).unwrap().sf(count as u64);
                assert!(
                    sf(mean, count) >= strictly_greater,
                    "mean={mean} count={count} 时尾概率反而变小了"
                );
            }
        }
    }

    #[test]
    fn observing_at_least_zero_is_certain() {
        // 旧实现在 (0.0, 0) 上返回 0.0，即"无穷显著"，对 count == 0 是荒谬的
        assert_eq!(sf(0.0, 0), 1.0);
        assert_eq!(sf(3.7, 0), 1.0);
    }

    #[test]
    fn a_missing_background_estimate_stays_on_the_conservative_side() {
        // `mean == 0` 不是"本底真的为零"，是这一格没有可用的本底估计
        assert_eq!(sf(0.0, 5), 1.0);
    }

    #[test]
    fn the_pruning_bound_is_below_the_tail_it_bounds() {
        // 剪枝拿 `P(X = count)` 当 `P(X ≥ count)` 的下界。界必须真的是下界，
        // 否则会剪掉本该触发的组 —— 这正是 `sf` 改语义时必须同步挪的那个界
        // （从 `count+1` 处的 PMF 挪到 `count` 处）。
        for count in 1..40_u32 {
            for mean in [0.001_f64, 0.05, 0.5, 2.0, 9.0] {
                let ln_pmf = -mean + count as f64 * mean.ln() - ln_gamma(count as f64 + 1.0);
                assert!(
                    ln_pmf.exp() <= sf(mean, count) * (1.0 + 1e-12),
                    "mean={mean} count={count}：下界 {} 超过了它要界的尾巴 {}",
                    ln_pmf.exp(),
                    sf(mean, count)
                );
            }
        }
    }
}
