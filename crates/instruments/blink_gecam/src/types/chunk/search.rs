use blink_algorithms::detector_share::{MAX_DETECTOR_FRACTION, max_detector_fraction};
use blink_algorithms::snapshot_stepping::{SearchConfig, search_new};
use blink_core::traits::Event as _;
use blink_core::types::{
    AcdCounts, Attitude, DetectorCounts, MissionElapsedTime, Position, Signal, Trajectory,
};
use std::sync::atomic::Ordering;
use uom::si::f64::*;

use crate::types::instrument::{Gecam, Satellite};
use crate::types::{Chunk, Event};

/// CPD 统计用的基线窗半宽（秒）。候选窗两侧各取这么长作本底，与搜索的
/// `neighbor` 同一量级，好让 `n_acd / n_acd_bg` 有可比的分母。
const CPD_BASELINE_SECONDS: f64 = 1.0;

pub(super) fn search<S: Satellite>(chunk: &Chunk<S>) -> Vec<Signal<Event<S>>> {
    // 事例准入见 `Event::keep`；再按 GTI 过滤。
    //
    // 分子分母必须是同一份 GTI。SVOM/GRM 上只在分母上用 GTI，事例流照单全收，
    // 结果是一直在搜 L1B 判为坏时段的数据：数据缺口两端各留着半秒事例，它们
    // 的本底窗有一半落在空区，均值被压低，普通计数就显得极显著——那批「1 Hz
    // 假信号」和「高本底假信号」就是这么来的。GECAM 的 GTI 缺口更多更长
    // （一小时常被切成一到四段），同样的坑只会更深。
    let mut n_outside_gti = 0usize;
    let mut events = chunk
        .evt_file
        .into_iter()
        .filter(|event| event.keep())
        .filter(|event| {
            let inside = chunk.evt_file.gti_contains(event.time().met());
            if !inside {
                n_outside_gti += 1;
            }
            inside
        })
        .collect::<Vec<_>>();
    // 各路内部会有几处量化步级别的时间抖动，k 路归并把它原样传了出来，而
    // `search_new` 假定输入有序。幅度大到有害的回跳已在 `exclusion` 挡掉，
    // 剩下的排一下就干净了——数组本来就基本有序，代价很低。
    events.sort();

    let gti: Vec<[MissionElapsedTime<Gecam<S>>; 2]> = chunk
        .evt_file
        .gti_segments()
        .into_iter()
        .map(|(start, stop)| {
            [
                MissionElapsedTime::new(start),
                MissionElapsedTime::new(stop),
            ]
        })
        .collect();

    let results = search_new(
        &events,
        1,
        chunk.span[0],
        chunk.span[1],
        &gti,
        SearchConfig {
            min_duration: Time::new::<uom::si::time::microsecond>(0.0),
            // 窗长上限沿用 SVOM/GRM 的 1 ms。GECAM 自己的 TGF 样本还没攒出来，
            // 攒出来之后要按同样的办法复核：拿证实样本量 T90，看 1 ms 够不够。
            max_duration: Time::new::<uom::si::time::millisecond>(1.0),
            neighbor: Time::new::<uom::si::time::second>(1.0),
            hollow: Time::new::<uom::si::time::millisecond>(10.0),
            false_positive_per_year: 20.0,
            min_number: 8,
            // 单组：各路 GRD 朝向不同，合成一路，符合判据退化
            coincidence: 1,
        },
    );

    // 两条轨迹各建一次，而不是每个候选重建一遍（1 Hz 采样，一小时约 3700 点）
    let positions = Trajectory::<MissionElapsedTime<Gecam<S>>, Position>::from(&chunk.posatt_file);
    let attitudes = Trajectory::<MissionElapsedTime<Gecam<S>>, Attitude>::from(&chunk.posatt_file);

    let mut n_dropped = 0usize;
    let mut n_without_attitude = 0usize;
    let mut n_single_detector = 0usize;
    let signals = results
        .into_iter()
        .filter_map(|candidate| {
            // 单路毛刺否决：候选窗里一路探测器占了绝大多数计数就不是暴发。
            // 四台仪器共用同一条判据、同一个阈值，按整窗算而不是按最显著一格。
            if max_detector_fraction(&events, candidate.start, candidate.stop, |e| e.detector_id)
                > MAX_DETECTOR_FRACTION
            {
                n_single_detector += 1;
                return None;
            }
            let peak = candidate.start + candidate.bin_size_best / 2.0;
            let Some(position) = positions.interpolate(peak) else {
                n_dropped += 1;
                return None;
            };
            // 姿态只是元数据，缺了不丢候选，留空并记账
            let attitude = attitudes.interpolate(peak).map(|a| a.state);
            if attitude.is_none() {
                n_without_attitude += 1;
            }

            Some(Signal {
                start: candidate.start,
                stop: candidate.stop,
                bin_size_min: candidate.bin_size_min,
                bin_size_max: candidate.bin_size_max,
                bin_size_best: candidate.bin_size_best,
                delay: candidate.delay,
                count: candidate.count,
                mean: candidate.mean,
                sf: candidate.sf(),
                false_positive_per_year: candidate.false_positive_per_year(),
                attitude,
                position: position.state,
                acd: Some(cpd_counts(chunk, &events, candidate.start, candidate.stop)),
                detectors: Some(detector_counts(
                    chunk.evt_file.detector_count(),
                    &events,
                    candidate.start,
                    candidate.stop,
                )),
            })
        })
        .collect::<Vec<_>>();

    chunk
        .dropped_no_ephemeris
        .store(n_dropped, Ordering::Relaxed);
    chunk
        .without_attitude
        .store(n_without_attitude, Ordering::Relaxed);
    chunk
        .dropped_single_detector
        .store(n_single_detector, Ordering::Relaxed);
    chunk
        .events_outside_gti
        .store(n_outside_gti, Ordering::Relaxed);

    signals
}

/// 候选窗里的 CPD 符合计数，随候选一起存下来。
///
/// 复用 `AcdCounts` 这个结构，但字段在 GECAM 上的含义要说清楚：HXMT HE 的
/// ACD 是逐事例带的标志位，GECAM 的 CPD 是一路独立的探测器、有自己的事例流，
/// 所以这里是「窗内 GRD 计数」对「窗内 CPD 计数」，不是同一批事例的两种标注。
///
/// * `n` / `n_bg`：候选窗、基线窗内通过准入的 GRD 事例数
/// * `n_acd` / `n_acd_bg`：同两个窗内的 CPD 事例数
/// * `n_acd_multi`：候选窗内、10 µs 内有另一路 CPD 一起响的事例数。一路响
///   可能是本底，多路同时响基本只能是穿过整台仪器的带电粒子。
///
/// **只统计，不否决。** 符合到什么程度才算带电粒子得拿真候选定标；在那之前
/// 存原始计数而不是比例，泊松误差还在，阈值日后可以复议而不必重跑全量。
fn cpd_counts<S: Satellite>(
    chunk: &Chunk<S>,
    events: &[Event<S>],
    start: MissionElapsedTime<Gecam<S>>,
    stop: MissionElapsedTime<Gecam<S>>,
) -> AcdCounts {
    let (start, stop) = (start.met(), stop.met());
    let (before, after) = (start - CPD_BASELINE_SECONDS, stop + CPD_BASELINE_SECONDS);

    // events 已按时间排好，二分即可：候选一多，线性扫 1600 万个事例就不像话了
    let grd_within = |from: f64, to: f64| {
        let lower = events.partition_point(|event| event.time().met() < from);
        let upper = events.partition_point(|event| event.time().met() <= to);
        (upper - lower) as u32
    };

    let n = grd_within(start, stop);
    AcdCounts {
        n,
        n_acd: chunk.cpd_file.count_within(start, stop),
        n_acd_multi: chunk.cpd_file.count_multi_within(start, stop),
        // 基线取候选窗两侧，挖掉候选本身
        n_bg: grd_within(before, after).saturating_sub(n),
        n_acd_bg: chunk
            .cpd_file
            .count_within(before, after)
            .saturating_sub(chunk.cpd_file.count_within(start, stop)),
    }
}

/// 候选窗与基线窗内逐路 GRD 的计数。
///
/// 这是方向分析唯一的原料，而且只能在这里取：候选表落盘后事例流就不在手边，
/// 事后要补就得重跑全量（GECAM-C 5.8 TB、GECAM-B 110 TB）。所以哪怕方向分析
/// 还没开工，这个向量也当场存下来。
///
/// 25 路 GRD 朝向各不相同，逐路计数的相对高低就编码了入射方向：CALDB 的
/// `mc_rsp` 给了 401 个方向 × 25 路的蒙卡响应，拿这个向量去拟合就能定方向。
/// 更省事的一问是"从天上来还是从地球来"——朝地那几路亮不亮，直接分开 TGF 与
/// GRB、TGF 与 TEB，而这正是天格上卡住的那个问题。
///
/// 基线窗一并存，否则分不清"这一路亮"和"这一路本来就快"（各路本底率实测能
/// 差一倍以上）。
fn detector_counts<S: Satellite>(
    detector_count: usize,
    events: &[Event<S>],
    start: MissionElapsedTime<Gecam<S>>,
    stop: MissionElapsedTime<Gecam<S>>,
) -> DetectorCounts {
    let (start, stop) = (start.met(), stop.met());
    let (before, after) = (start - CPD_BASELINE_SECONDS, stop + CPD_BASELINE_SECONDS);

    // 探头号从 1 数起，下标 0 对应 EVENTS01
    let mut window = vec![0u32; detector_count];
    let mut baseline = vec![0u32; detector_count];
    let lower = events.partition_point(|event| event.time().met() < before);
    let upper = events.partition_point(|event| event.time().met() <= after);
    for event in &events[lower..upper] {
        let Some(slot) = (event.detector_id as usize).checked_sub(1) else {
            continue;
        };
        if slot >= detector_count {
            continue;
        }
        let time = event.time().met();
        if time >= start && time <= stop {
            window[slot] += 1;
        } else {
            baseline[slot] += 1;
        }
    }

    DetectorCounts {
        window,
        baseline,
        baseline_seconds: 2.0 * CPD_BASELINE_SECONDS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::instrument::SatC;

    fn event(time: f64, detector_id: u8) -> Event<SatC> {
        Event {
            time: MissionElapsedTime::new(time),
            channel: 100,
            detector_id,
            gain_type: 0,
            dead_time: 4.0,
            evt_type: 1,
            flag: 0,
        }
    }

    #[test]
    fn detector_ids_are_one_based_so_slot_zero_is_events01() {
        let events = vec![event(10.0, 1), event(10.0, 3)];
        let counts = detector_counts(
            3,
            &events,
            MissionElapsedTime::new(9.9),
            MissionElapsedTime::new(10.1),
        );
        assert_eq!(counts.window, vec![1, 0, 1]);
    }

    #[test]
    fn events_outside_the_candidate_window_land_in_the_baseline() {
        let events = vec![
            event(9.0, 2),  // 基线：候选前
            event(10.0, 2), // 候选窗内
            event(11.0, 2), // 基线：候选后
        ];
        let counts = detector_counts(
            2,
            &events,
            MissionElapsedTime::new(9.9),
            MissionElapsedTime::new(10.1),
        );
        assert_eq!(counts.window, vec![0, 1]);
        assert_eq!(counts.baseline, vec![0, 2]);
        assert_eq!(counts.baseline_seconds, 2.0 * CPD_BASELINE_SECONDS);
    }

    #[test]
    fn events_beyond_the_baseline_window_are_not_counted_at_all() {
        // 基线半宽 1 s，±5 s 的事例既不算窗内也不算基线
        let events = vec![event(5.0, 1), event(10.0, 1), event(15.0, 1)];
        let counts = detector_counts(
            1,
            &events,
            MissionElapsedTime::new(9.9),
            MissionElapsedTime::new(10.1),
        );
        assert_eq!(counts.window, vec![1]);
        assert_eq!(counts.baseline, vec![0]);
    }

    #[test]
    fn a_detector_id_out_of_range_is_skipped_rather_than_panicking() {
        // 探头数由文件里的 EVENTS 表数决定；万一对不上也不能越界
        let events = vec![event(10.0, 1), event(10.0, 99)];
        let counts = detector_counts(
            2,
            &events,
            MissionElapsedTime::new(9.9),
            MissionElapsedTime::new(10.1),
        );
        assert_eq!(counts.window, vec![1, 0]);
    }
}
