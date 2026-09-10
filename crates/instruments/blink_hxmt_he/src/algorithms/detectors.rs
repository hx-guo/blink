//! 候选窗与基线窗内逐路探头的计数。
//!
//! 18 路相里探测器在搜索时合成一路（`SearchConfig::coincidence = 1`），逐路
//! 的分布因此不进判据，也不出现在候选表里——事后要问就得把 1K 事例流重读一遍。
//! 与 `AcdCounts` 同理，这是只能在候选生成现场取到的量，当场存下来。
//!
//! 三件事要用它：
//!
//! 1. **单路毛刺**。`MAX_DETECTOR_FRACTION = 0.8` 是四仪器共用的定值，接进
//!    HXMT 时只在 v6 的 4270 个显著候选上审计过一遍（实测最高 0.417）。存下
//!    整条向量，阈值日后要复议就不必再重跑全量。
//! 2. **是不是照亮整台仪器**。18 路各自的本底率实测能差一成（一小时 1.86M–
//!    2.06M），只看窗内计数分不清"这一路亮"和"这一路本来就快"，所以基线窗
//!    一并存。
//! 3. **方向**。18 路装在同一块板上，高能光子穿过结构时相互遮挡不同，星体
//!    在下、准直器在上——相对计数携带入射方向的信息，HE 的 CsI 当全天监视器
//!    做 GRB 定位走的就是这条路。TGF 从下方来、GRB 从上方来，这一问正是本
//!    项目未决的那条。
//!
//! 只存计数不存比例：保泊松误差，判据可复议而不必重跑。
//!
//! 窗口定义与 `acd::acd_counts` 逐字一致，于是 `window` 之和恒等于
//! `AcdCounts::n`、`baseline` 之和恒等于 `AcdCounts::n_bg`——两个量互为校验。

use crate::algorithms::acd::{HOLLOW_S, NEIGHBOR_S};
use crate::types::Event;
use blink_core::traits::Event as _;
use blink_core::types::DetectorCounts;

/// HE 的相里探测器路数，1K 的 `Det_ID` 取值 0..=17。
///
/// 机箱划分实测过（2026-09-10，用 1B 检出的 FIFO reset 缺口当探针：某机箱复位
/// 期间只有它自己那 6 路不出事例，另外 12 路照常；每个机箱各取 40 个缺口）：
/// **Box A = 0..=5，Box B = 6..=11，Box C = 12..=17**。Box C 里含那个盲探测器。
pub const N_DETECTORS: usize = 18;

/// 对时间有序的 kept 事例数组统计候选窗 [start, stop]（闭区间）与两侧基线窗
/// [start−1s, start−10ms) ∪ (stop+10ms, stop+1s] 的逐路计数。窗口越过数据边缘
/// 时按实际存在的事例截断，与 `acd_counts` 同一行为。
pub fn detector_counts(events: &[Event], start_met: f64, stop_met: f64) -> DetectorCounts {
    let below = |t: f64| events.partition_point(|e| e.time().met() < t);
    let at_or_below = |t: f64| events.partition_point(|e| e.time().met() <= t);

    let mut window = vec![0u32; N_DETECTORS];
    let mut baseline = vec![0u32; N_DETECTORS];
    let tally = |slice: &[Event], into: &mut Vec<u32>| {
        for event in slice {
            // Det_ID 超出 0..=17 的事例不可能出现在标称的 1K 表里；真出现了
            // 也只能丢，不能让它把别人的格子算脏。
            if let Some(slot) = into.get_mut(event.detector.id as usize) {
                *slot += 1;
            }
        }
    };

    tally(
        &events[below(start_met)..at_or_below(stop_met)],
        &mut window,
    );
    tally(
        &events[below(start_met - NEIGHBOR_S)..below(start_met - HOLLOW_S)],
        &mut baseline,
    );
    tally(
        &events[at_or_below(stop_met + HOLLOW_S)..at_or_below(stop_met + NEIGHBOR_S)],
        &mut baseline,
    );

    DetectorCounts {
        window,
        baseline,
        baseline_seconds: 2.0 * (NEIGHBOR_S - HOLLOW_S),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::acd::acd_counts;
    use crate::types::{Detector, Scintillator};
    use blink_core::types::MissionElapsedTime;

    fn event(met: f64, detector_id: u8) -> Event {
        Event::new(
            MissionElapsedTime::new(met),
            100,
            Detector {
                id: detector_id,
                scintillator: Scintillator::Csi,
            },
            false,
            [false; 18],
        )
    }

    #[test]
    fn window_and_baseline_partition() {
        // 候选窗 [10.0, 10.001]：0 号 2 个、5 号 1 个。
        // 左基线 [9.0, 9.99)：3 号 1 个、0 号 1 个；右基线 (10.011, 11.001]：17 号 1 个。
        // hollow 内（9.995、10.005）的事例任何窗都不计。
        let events = vec![
            event(8.0, 9),     // 左基线之外
            event(9.5, 3),     // 左基线
            event(9.8, 0),     // 左基线
            event(9.995, 1),   // 左 hollow，丢弃
            event(10.0, 0),    // 窗内（闭区间左端）
            event(10.0005, 5), // 窗内
            event(10.001, 0),  // 窗内（闭区间右端）
            event(10.005, 2),  // 右 hollow，丢弃
            event(10.5, 17),   // 右基线
            event(12.0, 4),    // 右基线之外
        ];
        let counts = detector_counts(&events, 10.0, 10.001);
        assert_eq!(counts.window.len(), N_DETECTORS);
        assert_eq!(counts.window[0], 2);
        assert_eq!(counts.window[5], 1);
        assert_eq!(counts.window.iter().sum::<u32>(), 3);
        assert_eq!(counts.baseline[3], 1);
        assert_eq!(counts.baseline[0], 1);
        assert_eq!(counts.baseline[17], 1);
        assert_eq!(counts.baseline.iter().sum::<u32>(), 3);
        assert!((counts.baseline_seconds - 1.98).abs() < 1e-12);

        // 与 ACD 侧同一套窗口：两个和必须逐个对上，这是两边的互校
        let acd = acd_counts(&events, 10.0, 10.001);
        assert_eq!(counts.window.iter().sum::<u32>(), acd.n);
        assert_eq!(counts.baseline.iter().sum::<u32>(), acd.n_bg);
    }

    #[test]
    fn empty_and_edge_truncation() {
        let counts = detector_counts(&[], 10.0, 10.001);
        assert_eq!(counts.window.iter().sum::<u32>(), 0);
        assert_eq!(counts.baseline.iter().sum::<u32>(), 0);
        // 数据只覆盖候选之后：左基线截断为空，不出错
        let events = vec![event(10.0, 7), event(10.7, 11)];
        let counts = detector_counts(&events, 10.0, 10.001);
        assert_eq!(counts.window[7], 1);
        assert_eq!(counts.baseline[11], 1);
        assert_eq!(counts.window.iter().sum::<u32>(), 1);
        assert_eq!(counts.baseline.iter().sum::<u32>(), 1);
    }

    #[test]
    fn out_of_range_detector_id_is_dropped_not_folded() {
        let events = vec![event(10.0, 18), event(10.0005, 3)];
        let counts = detector_counts(&events, 10.0, 10.001);
        assert_eq!(counts.window[3], 1);
        assert_eq!(counts.window.iter().sum::<u32>(), 1);
    }
}
