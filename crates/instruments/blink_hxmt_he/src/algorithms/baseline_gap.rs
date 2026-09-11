//! 基线窗里最长的一段"没有事例"的时间。
//!
//! 搜索把本底率算成 `n_bg / 窗宽`，窗宽按墙钟取。本底窗要是伸进了数据缺口，
//! 分子少了分母没少，本底被压低、显著性被做高——这是 HXMT 侧已知的一类假信号
//! 来源（`OPEN-QUESTIONS.md` 第 7 条）。
//!
//! 原先的诊断量拿搜索的 ±0.5 s 本底率比审计的 ±1 s 本底率。**那个比值对最要命
//! 的情形不敏感**：缺口比两个窗都大时，两边被削掉的比例一样，比值仍是 1。要直接
//! 量，就得看本底窗里到底有没有一段时间是空的。
//!
//! 这里给的是参数无关的原始量：基线窗内最长的无事例间隔（含窗边到首/末事例的
//! 两段）。判据留到离线——泊松下最长间隔的期望是 `ln(n_bg) / 率`，实测值比它大
//! 一个量级以上就是真缺口，不必先定阈值。窗宽与 `acd_counts` 逐字一致，
//! `n_bg` 与这个量配套使用。

use crate::algorithms::acd::{HOLLOW_S, NEIGHBOR_S};
use crate::types::Event;
use blink_core::traits::Event as _;

/// 一段区间内最长的无事例间隔，窗边也算：`[lo, 首事例]` 与 `[末事例, hi]` 都参与
/// 取大。区间内一个事例都没有时返回整段宽度。
fn longest_gap(events: &[Event], lo: f64, hi: f64) -> f64 {
    if hi <= lo {
        return 0.0;
    }
    let i0 = events.partition_point(|e| e.time().met() < lo);
    let i1 = events.partition_point(|e| e.time().met() <= hi);
    let mut previous = lo;
    let mut longest: f64 = 0.0;
    for event in &events[i0..i1] {
        let t = event.time().met();
        longest = longest.max(t - previous);
        previous = t;
    }
    longest.max(hi - previous)
}

/// 候选两侧基线窗内最长的无事例间隔（秒），两侧取大。
///
/// 窗口与 `acd::acd_counts` 同一定义：`[start−1s, start−10ms)` 与
/// `(stop+10ms, stop+1s]`。事例数组必须按时间有序。
pub fn baseline_max_gap(events: &[Event], start_met: f64, stop_met: f64) -> f64 {
    let left = longest_gap(events, start_met - NEIGHBOR_S, start_met - HOLLOW_S);
    let right = longest_gap(events, stop_met + HOLLOW_S, stop_met + NEIGHBOR_S);
    left.max(right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Detector, Scintillator};
    use blink_core::types::MissionElapsedTime;

    fn event(met: f64) -> Event {
        Event::new(
            MissionElapsedTime::new(met),
            100,
            Detector {
                id: 0,
                scintillator: Scintillator::Csi,
            },
            false,
            [false; 18],
        )
    }

    /// 基线窗铺满事例时，最长间隔就是事例间隔本身，不是窗宽。
    #[test]
    fn dense_baseline_reports_the_event_spacing() {
        let mut events: Vec<Event> = Vec::new();
        let mut t = 9.0;
        while t <= 11.01 {
            events.push(event(t));
            t += 0.001;
        }
        let gap = baseline_max_gap(&events, 10.0, 10.001);
        assert!(gap < 0.0025, "gap = {gap}");
    }

    /// 左半窗整段空着：应当报出左半窗的宽度（1 s − 10 ms），而不是 0。
    #[test]
    fn empty_left_half_reports_its_full_width() {
        let events = vec![event(10.02), event(10.5), event(10.51), event(11.0)];
        let gap = baseline_max_gap(&events, 10.0, 10.001);
        assert!((gap - (NEIGHBOR_S - HOLLOW_S)).abs() < 1e-9, "gap = {gap}");
    }

    /// 窗内部的一段缺口要能量出来，包括窗边到首事例那一段。
    #[test]
    fn interior_hole_is_measured() {
        // 左窗 [9.0, 9.99)：事例只到 9.3，之后空到窗尾 → 0.69 s
        // 右窗 (10.011, 11.001]：事例密集 → 很小
        let mut events = vec![event(9.05), event(9.1), event(9.3)];
        let mut t = 10.02;
        while t <= 11.0 {
            events.push(event(t));
            t += 0.001;
        }
        let gap = baseline_max_gap(&events, 10.0, 10.001);
        assert!((gap - 0.69).abs() < 1e-6, "gap = {gap}");
    }

    /// 完全没有事例：两半窗各报满宽，取大仍是满宽。
    #[test]
    fn no_events_at_all() {
        let gap = baseline_max_gap(&[], 10.0, 10.001);
        assert!((gap - (NEIGHBOR_S - HOLLOW_S)).abs() < 1e-9, "gap = {gap}");
    }
}
