use blink_algorithms::detector_share::{MAX_DETECTOR_FRACTION, max_detector_fraction};
use blink_algorithms::snapshot_stepping::{SearchConfig, search_new};
use blink_core::traits::Event as _;
use blink_core::types::{AcdCounts, Attitude, MissionElapsedTime, Position, Signal, Trajectory};
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
