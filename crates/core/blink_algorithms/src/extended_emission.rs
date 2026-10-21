//! GRB 判据：候选之外有没有延展辐射。
//!
//! TGF 一毫秒以内就结束，GRB 却持续几百毫秒以上，里面几毫秒的脉冲照样能过
//! 显著性判据。于是看候选**之外**：起点前后 [−0.1, +0.3) s（扣掉候选自身的
//! full interval）的事例数，与 [−2, −0.5) ∪ [+0.5, +2) s 本底率推得的期望比，
//! 超出折成高斯 σ 的 `Z`。GRB 照亮整台仪器，超出由各组分摊；单组时间标签错、
//! 单组毛刺之类的仪器效应只在一组。HXMT 论文（4.6 节）的判据：
//! `Z > 5` 且最大单组占超出的比例 `< 0.6`，再把被判掉者 1 s 内的候选一并算作
//! 同一个暴（这一步要看整个候选表，由目录生成做，这里只给逐候选的量）。
//!
//! 原实现是 `hxmt-catalog` 的 `grb_check/rule.py`，按 10 ms / 100 ms 分箱算；
//! 这里改按事例计，候选扣除区间与本底长度都精确到事例。分组超出的扣除也从
//! "全组平摊候选计数"改为逐组扣候选自己的事例。
//!
//! 时间一律相对候选**起点**（不是峰值），与论文与原脚本一致。

use crate::poisson;
use blink_core::traits::Event;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};

/// 判据的时间窗与阈值，单位秒，相对候选起点。
#[derive(Clone, Debug)]
pub struct ExtendedEmissionConfig {
    /// 检验区间 [lo, hi)
    pub test: (f64, f64),
    /// 本底区间，各自 [lo, hi)
    pub background: [(f64, f64); 2],
    /// 候选 [start, stop] 两侧再多扣的余量。原脚本按 10 ms 箱扣到 ±1 ms。
    pub exclude_pad: f64,
    /// 本底区间（含两端边界）里允许的最长无事例间隔；更长就说明数据有缺口
    /// （缺小时文件、饱和、停机），本底率不可信，`Z` 不给。原脚本的等价条件
    /// 是"本底里有空的 10 ms 箱"。
    pub max_background_gap: f64,
    pub z_threshold: f64,
    pub max_group_share: f64,
}

impl Default for ExtendedEmissionConfig {
    fn default() -> Self {
        Self {
            test: (-0.1, 0.3),
            background: [(-2.0, -0.5), (0.5, 2.0)],
            exclude_pad: 0.001,
            max_background_gap: 0.01,
            z_threshold: 5.0,
            max_group_share: 0.6,
        }
    }
}

impl ExtendedEmissionConfig {
    /// 判据需要的事例范围（相对起点），调用方据此决定要载入哪些数据。
    pub fn span(&self) -> (f64, f64) {
        let lo = self.test.0.min(self.background[0].0);
        let hi = self.test.1.max(self.background[1].1);
        (lo, hi)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExtendedEmission {
    /// 检验区间（扣掉候选）内的事例数
    pub observed: u32,
    /// 本底率推得的期望
    pub expected: f64,
    /// `P(X ≥ observed)` 折成的高斯 σ；p 值下限截在 1e-300。本底有缺口时为 `None`。
    pub z: Option<f64>,
    /// 逐组超出（观测 − 本底期望），下标即组号
    pub group_excess: Vec<f64>,
    /// 最大单组超出 / max(总超出, 1)，与原脚本同口径
    pub max_group_share: f64,
    /// `z > z_threshold && max_group_share < max_group_share`
    pub flagged: bool,
}

/// 对时间有序的事例计算延展辐射判据。`start`/`stop` 为候选窗（MET 秒），
/// `group` 把事例映射到 `0..n_groups`（HXMT 是机箱，GBM 可用 NaI/BGO）。
pub fn extended_emission<E: Event>(
    events: &[E],
    start: f64,
    stop: f64,
    n_groups: usize,
    group: impl Fn(&E) -> usize,
    config: &ExtendedEmissionConfig,
) -> ExtendedEmission {
    let at = |t: f64| events.partition_point(|e| e.time().met() < t);
    let slice = |lo: f64, hi: f64| &events[at(start + lo)..at(start + hi)];

    // 本底：两段的事例与长度，同时量最长间隔（区间端点算作边界，事例缺在
    // 区间开头或结尾同样是缺口）
    let mut n_bg = vec![0u32; n_groups];
    let mut bg_length = 0.0;
    let mut max_gap: f64 = 0.0;
    for &(lo, hi) in &config.background {
        let mut previous = start + lo;
        for event in slice(lo, hi) {
            let t = event.time().met();
            max_gap = max_gap.max(t - previous);
            previous = t;
            if let Some(slot) = n_bg.get_mut(group(event)) {
                *slot += 1;
            }
        }
        max_gap = max_gap.max(start + hi - previous);
        bg_length += hi - lo;
    }

    // 检验区间扣掉候选：[start + test.0, start + test.1) \ [start − pad, stop + pad]
    let core = (start - config.exclude_pad, stop + config.exclude_pad);
    let test = (start + config.test.0, start + config.test.1);
    let overlap = (test.1.min(core.1) - test.0.max(core.0)).max(0.0);
    let test_length = (test.1 - test.0) - overlap;
    let mut n_test = vec![0u32; n_groups];
    for event in slice(config.test.0, config.test.1) {
        let t = event.time().met();
        if t >= core.0 && t <= core.1 {
            continue;
        }
        if let Some(slot) = n_test.get_mut(group(event)) {
            *slot += 1;
        }
    }

    let observed: u32 = n_test.iter().sum();
    let expected = n_bg.iter().sum::<u32>() as f64 / bg_length * test_length;
    let group_excess: Vec<f64> = n_test
        .iter()
        .zip(&n_bg)
        .map(|(&n, &b)| n as f64 - b as f64 / bg_length * test_length)
        .collect();
    let total_excess: f64 = group_excess.iter().sum();
    let max_excess = group_excess.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let max_group_share = max_excess / total_excess.max(1.0);

    let z = (max_gap <= config.max_background_gap).then(|| {
        let p = poisson::sf(expected, observed).max(1e-300);
        -Normal::standard().inverse_cdf(p)
    });
    let flagged = z.is_some_and(|z| z > config.z_threshold)
        && max_group_share < config.max_group_share;

    ExtendedEmission {
        observed,
        expected,
        z,
        group_excess,
        max_group_share,
        flagged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TestEvent;

    /// 均匀本底：`rate` 每秒，在 [lo, hi) 上等间隔，组号轮流
    fn uniform(lo: f64, hi: f64, rate: f64, n_groups: usize) -> Vec<(f64, usize)> {
        let n = ((hi - lo) * rate) as usize;
        (0..n)
            .map(|i| (lo + (i as f64 + 0.5) / rate, i % n_groups))
            .collect()
    }

    fn run(mut raw: Vec<(f64, usize)>, start: f64, stop: f64) -> ExtendedEmission {
        raw.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let events: Vec<TestEvent> = raw
            .into_iter()
            .map(|(seconds, group)| TestEvent {
                seconds,
                group: group as u8,
            })
            .collect();
        extended_emission(&events, start, stop, 3, |e| e.group as usize, &Default::default())
    }

    #[test]
    fn flat_background_is_not_flagged() {
        // 1e4/s 平本底 + 候选本身 30 个事例：候选被扣掉，检验区间无超出
        let mut raw = uniform(-3.0, 3.0, 1e4, 3);
        raw.extend((0..30).map(|i| (1e-5 * i as f64 + 1e-7, i % 3)));
        let result = run(raw, 0.0, 3e-4);
        let z = result.z.expect("background is complete");
        assert!(z.abs() < 1.0, "z = {z}");
        assert!(!result.flagged);
        // 期望 = 率 × (0.4 s − 候选与余量)
        assert!((result.expected - 1e4 * (0.4 - 0.0023)).abs() < 2.0);
    }

    #[test]
    fn extended_emission_shared_by_groups_is_flagged() {
        // 检验区间里多 2000 个事例，三组平分：GRB
        let mut raw = uniform(-3.0, 3.0, 1e4, 3);
        raw.extend(uniform(0.01, 0.21, 1e4, 3));
        let result = run(raw, 0.0, 3e-4);
        assert!(result.z.unwrap() > 5.0);
        assert!(result.max_group_share < 0.4);
        assert!(result.flagged);
    }

    #[test]
    fn excess_in_one_group_is_not_flagged() {
        // 同样的超出全在第 0 组：单组仪器效应，不算 GRB
        let mut raw = uniform(-3.0, 3.0, 1e4, 3);
        raw.extend(uniform(0.01, 0.21, 1e4, 1));
        let result = run(raw, 0.0, 3e-4);
        assert!(result.z.unwrap() > 5.0);
        assert!(result.max_group_share > 0.95);
        assert!(!result.flagged);
    }

    #[test]
    fn background_gap_gives_no_z() {
        // 本底右段 [1.0, 1.2) 缺数据（如邻小时文件缺失）
        let raw = uniform(-3.0, 3.0, 1e4, 3)
            .into_iter()
            .filter(|&(t, _)| !(1.0..1.2).contains(&t))
            .collect();
        let result = run(raw, 0.0, 3e-4);
        assert!(result.z.is_none());
        assert!(!result.flagged);
        // 数据只到 1.5 s：区间末端的缺口同样算
        let raw = uniform(-3.0, 1.5, 1e4, 3);
        assert!(run(raw, 0.0, 3e-4).z.is_none());
    }
}
