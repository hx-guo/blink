//! 候选窗口的 ACD 符合计数。
//!
//! 电子必须穿过反符合塑闪才能到达 NaI/CsI，ACD 几乎必然同时着火；伽马对
//! 薄塑闪只有几个百分点的康普顿效率。候选窗内 ACD 着火事例占比因此是
//! REP（电子沉降）与 TGF（伽马）的直接物理判别量，与候选密度（train）
//! 判据正交。搜索时统计并随候选保存；`blink acd-audit` 用同一实现离线
//! 复算认证样本。

use crate::types::Event;
use blink_core::traits::Event as _;
use blink_core::types::AcdCounts;

/// 基线窗：候选两侧各 1 s，紧贴候选的 10 ms hollow 挖掉，避免瞬变自身拖尾
/// 污染基线。
///
/// **与搜索的本底窗不是同一个宽度，别拿 `n_bg` 直接当搜索的 `mean` 用。**
/// `SearchConfig` 的 `neighbor = 1 s` / `hollow = 10 ms` 是**总宽**，
/// `search_new` 里两侧各取 `neighbor / 2`、`hollow / 2`，即 ±0.5 s 与 ±5 ms；
/// 这里两侧各取 1 s 与 10 ms，正好是搜索的两倍。本底率稳定时
/// `n_bg ≈ 2 × pure_mean_number`，`mean ∝ n_bg × bin_size_best`，但机箱级停机
/// 这类窗内不均匀的情形下两者不成比例。
pub(crate) const NEIGHBOR_S: f64 = 1.0;
pub(crate) const HOLLOW_S: f64 = 0.01;

/// 对时间有序的 kept 事例数组统计候选窗 [start, stop]（闭区间）与两侧
/// 基线窗 [start−1s, start−10ms) ∪ (stop+10ms, stop+1s] 的 ACD 计数。
/// 窗口越过数据边缘时按实际存在的事例截断——搜索按小时分块，基线在小时
/// 边缘天然截断，审计侧同一行为，两边数字可比。
pub fn acd_counts(events: &[Event], start_met: f64, stop_met: f64) -> AcdCounts {
    // (总数, 任意着火数, ≥2 块着火数)；区间端点由调用处的 partition_point 谓词决定
    let count = |i0: usize, i1: usize| -> (u32, u32, u32) {
        let mut n = 0u32;
        let mut n_acd = 0u32;
        let mut n_multi = 0u32;
        for event in &events[i0..i1] {
            n += 1;
            let fired = event.acds.iter().filter(|&&b| b).count();
            if fired >= 1 {
                n_acd += 1;
            }
            if fired >= 2 {
                n_multi += 1;
            }
        }
        (n, n_acd, n_multi)
    };
    let below = |t: f64| events.partition_point(|e| e.time().met() < t);
    let at_or_below = |t: f64| events.partition_point(|e| e.time().met() <= t);

    let (n, n_acd, n_acd_multi) = count(below(start_met), at_or_below(stop_met));
    let (n_left, acd_left, _) = count(below(start_met - NEIGHBOR_S), below(start_met - HOLLOW_S));
    let (n_right, acd_right, _) =
        count(at_or_below(stop_met + HOLLOW_S), at_or_below(stop_met + NEIGHBOR_S));

    AcdCounts {
        n,
        n_acd,
        n_acd_multi,
        n_bg: n_left + n_right,
        n_acd_bg: acd_left + acd_right,
    }
}

/// 顶部 ACD 的标志位下标（其余 12 块在侧面）。来源：hx-guo/astro_sift
/// `plot/marker/detector.py`，HXMT 论文 6.1 节沿用。
pub const TOP_ACDS: [usize; 6] = [4, 5, 10, 11, 16, 17];

/// 电子束检验的参考：闪电关联 TGF 中，"只有顶部 ACD 着火"占"只有顶部或只有
/// 侧面着火"事例的比例（HXMT 论文 6.1 节，v9 目录 1071 个关联 TGF 实测 0.56）。
/// 定成常数而不在线重估：实时管线拿不到闪电数据，参考样本也不该随目录漂移。
pub const TOP_SHARE_REFERENCE: f64 = 0.56;

/// 电子束判据：单边二项 `P < 1e-9`。v9 目录里 7 个电子束都在 1e-9 以下，
/// 其余 TGF 最小 2.5e-6，中间是空的。
pub const BEAM_P_THRESHOLD: f64 = 1e-9;

/// 候选窗内只有顶部 ACD 着火、只有侧面 ACD 着火的事例数。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopSide {
    pub top_only: u32,
    pub side_only: u32,
}

/// 对时间有序的 kept 事例统计候选窗 [start, stop]（闭区间）的顶/侧 ACD 计数。
pub fn acd_top_side(events: &[Event], start_met: f64, stop_met: f64) -> TopSide {
    let lo = events.partition_point(|e| e.time().met() < start_met);
    let hi = events.partition_point(|e| e.time().met() <= stop_met);
    let mut counts = TopSide {
        top_only: 0,
        side_only: 0,
    };
    for event in &events[lo..hi] {
        let top = TOP_ACDS.iter().any(|&i| event.acds[i]);
        let side = event
            .acds
            .iter()
            .enumerate()
            .any(|(i, &fired)| fired && !TOP_ACDS.contains(&i));
        match (top, side) {
            (true, false) => counts.top_only += 1,
            (false, true) => counts.side_only += 1,
            _ => {}
        }
    }
    counts
}

/// 电子从上方进来时"只有顶部着火"的事例异常多：在参考比例下观测到至少这么多
/// 的单边二项概率。没有只顶/只侧的事例时为 1。
pub fn beam_probability(counts: TopSide) -> f64 {
    blink_algorithms::binomial::sf_at_least(
        counts.top_only + counts.side_only,
        counts.top_only,
        TOP_SHARE_REFERENCE,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Detector, Scintillator};
    use blink_core::types::MissionElapsedTime;

    fn event(met: f64, fired_paddles: usize) -> Event {
        let mut acds = [false; 18];
        acds.iter_mut().take(fired_paddles).for_each(|b| *b = true);
        Event::new(
            MissionElapsedTime::new(met),
            100,
            Detector {
                id: 0,
                scintillator: Scintillator::Csi,
            },
            false,
            acds,
        )
    }

    #[test]
    fn window_and_baseline_partition() {
        // 候选窗 [10.0, 10.001]：3 事例，其中 1 个单块着火、1 个三块着火。
        // 左基线 [9.0, 9.99)：2 事例 1 着火；右基线 (10.011, 11.001]：1 事例未着火。
        // hollow 内（9.995、10.005）的事例任何窗都不计。
        let events = vec![
            event(9.5, 0),
            event(9.8, 1),
            event(9.995, 5),  // 左 hollow，丢弃
            event(10.0, 0),   // 窗内（闭区间左端）
            event(10.0005, 1),
            event(10.001, 3), // 窗内（闭区间右端）
            event(10.005, 5), // 右 hollow，丢弃
            event(10.5, 0),
        ];
        let counts = acd_counts(&events, 10.0, 10.001);
        assert_eq!(counts.n, 3);
        assert_eq!(counts.n_acd, 2);
        assert_eq!(counts.n_acd_multi, 1);
        assert_eq!(counts.n_bg, 3);
        assert_eq!(counts.n_acd_bg, 1);
    }

    #[test]
    fn empty_and_edge_truncation() {
        let counts = acd_counts(&[], 10.0, 10.001);
        assert_eq!((counts.n, counts.n_bg), (0, 0));
        // 数据只覆盖候选之后：左基线截断为空，不出错
        let events = vec![event(10.0, 1), event(10.7, 2)];
        let counts = acd_counts(&events, 10.0, 10.001);
        assert_eq!(counts.n, 1);
        assert_eq!(counts.n_acd, 1);
        assert_eq!(counts.n_bg, 1);
        assert_eq!(counts.n_acd_bg, 1);
    }

    #[test]
    fn top_and_side_are_counted_exclusively() {
        let with = |met: f64, paddles: &[usize]| {
            let mut e = event(met, 0);
            paddles.iter().for_each(|&i| e.acds[i] = true);
            e
        };
        let events = vec![
            with(9.0, &[4]),        // 窗外
            with(10.0, &[4]),       // 只顶
            with(10.0002, &[5, 17]), // 只顶（两块）
            with(10.0004, &[0]),    // 只侧
            with(10.0006, &[4, 0]), // 顶+侧：不计
            with(10.0008, &[]),     // 无 ACD
        ];
        let counts = acd_top_side(&events, 10.0, 10.001);
        assert_eq!(counts, TopSide { top_only: 2, side_only: 1 });
    }

    #[test]
    fn a_beam_from_the_top_passes_and_a_tgf_does_not() {
        // 电子束：60 个只顶、1 个只侧（40 顶 2 侧还只有 1.4e-8，不够）
        let beam = beam_probability(TopSide { top_only: 60, side_only: 1 });
        assert!(beam < BEAM_P_THRESHOLD, "{beam}");
        // 典型 TGF：只顶/只侧接近参考比例
        let tgf = beam_probability(TopSide { top_only: 5, side_only: 4 });
        assert!(tgf > 0.1, "{tgf}");
        assert_eq!(beam_probability(TopSide { top_only: 0, side_only: 0 }), 1.0);
    }
}
