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

/// 搜索本底窗的**全宽**（秒）：`search_new` 实际取候选两侧各 `neighbor / 2`。
const SEARCH_NEIGHBOR_SECONDS: f64 = 1.0;
/// 搜索空窗的**全宽**（秒）：候选两侧各 `hollow / 2` 从本底里挖掉。
const SEARCH_HOLLOW_MILLISECONDS: f64 = 10.0;

/// 审计统计（`cpd_counts`、`detector_counts`）用的基线窗**半宽**（秒）。
///
/// **它不等于搜索的本底窗，正好是搜索的两倍宽**，而且还差两件事。数值上
/// `CPD_BASELINE_SECONDS == SEARCH_NEIGHBOR_SECONDS`，但前者是半宽、后者是
/// 全宽，所以审计取的是候选两侧各 1 s、搜索取的是各 0.5 s。除此之外：
///
/// * **审计不挖空窗。** 搜索从本底里挖掉候选两侧各 `hollow / 2` = 5 ms，
///   审计只挖掉候选窗本身。
/// * **审计不夹 GTI。** 搜索把本底窗夹到候选所在的那一段活时间里再算时长，
///   审计的 `baseline_seconds` 是**标称的 2 s，不是活时间**。
///
/// **后果，用这些数之前必须清楚：**
///
/// * `n_acd_bg / baseline_seconds` 只有在本底率**平稳且窗内没有 GTI 缺口**时
///   才是"该候选处的 CPD 率"。GECAM 的候选恰恰常落在率剧变的地方（GTI 边界、
///   高磁纬进出、粒子增强期），而 GECAM 的 GTI 一小时常被切成一到四段。
///   **窗跨了缺口就会低估率**，跟着高估任何"观测/期望"的超出。要准就回
///   CPD 事例流按活时间自己数，别用这里的标称分母。
/// * **不能拿这些计数去反推搜索的 `mean`**，会差两倍。
///
/// **不改窗、只把话说清楚**：两倍窗给的统计量更多、更稳，而改窗会让已经落盘
/// 的候选表前后不可比。
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
    let n_merged = dedupe_gain_pairs(&mut events);

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
            neighbor: Time::new::<uom::si::time::second>(SEARCH_NEIGHBOR_SECONDS),
            hollow: Time::new::<uom::si::time::millisecond>(SEARCH_HOLLOW_MILLISECONDS),
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
                acd: cpd_counts(chunk, &events, candidate.start, candidate.stop),
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
    chunk
        .merged_gain_duplicates
        .store(n_merged, Ordering::Relaxed);

    signals
}

/// 双增益重复记录去重：同一路探头、同一时戳的两条并成一条。返回并掉的条数。
///
/// **同一个物理光子会被 ADC 的两个增益支路各写一行。** 实测 GECAM-C
/// 2023-06-15 12h 全 12 路：同探头同时戳的事例对占原始事例 6.635%，这些对
/// **100%** 是 `GAIN_TYPE` 一高一低、`FLAG` 一个 0 一个 10、`EVT_PAIR`
/// 两端都为真，而 PI 相同的占 0%。所以它们不是两个光子，是一个光子的两条
/// 记录。GECAM-A 2024-01-11 独立复现（123 万对），并排除了「其实是同探头
/// 29.8 ns 内的两个独立事例」：按单路 748 c/s 算偶然同戳对的期望是 11 对／
/// 路／过境，实测 49,214 对／路，**99.98% 是真双增益读出**。
///
/// 多数时候高增益那条已经饱和进溢出道、被 `Event::keep` 的 `PI < 448` 挡掉，
/// 一个物理事例只剩一条（占 98.8%）。但剩下的 1.2% 两条都过了准入，于是
/// **约 3.25% 的准入后计数是重复计数**——而候选窗内这个比例是 6.82%，
/// 富集 2.1 倍，**50.4% 的候选窗里至少含一对重复**。两条记录的时戳完全相同，
/// 泊松独立性直接破掉，跟带电粒子跨探头同时点亮是同一类机制，只是发生在
/// 一路探头内部。粗估反标定表明约 28% 的候选是这么造出来的。
///
/// **保留低增益那条，理由是量程不是饱和。** 「高增益支路会饱和」这个先前的
/// 说法**实测不成立**：GECAM-A 2024-01-11 的 123 万对里，高增益条
/// `PI >= 448` 的占 **0.00%**。真正的理由是两档的量程差一个数量级——低增益
/// 支路的高能端撑到 **ch≈379（约 4.0 MeV）**，高增益支路在 **ch≈157
/// （约 215 keV）** 就到顶了，**留高增益条会把 215 keV 以上的信息整段丢掉**。
///
/// **两档不是同一把尺子，而且偏差随能量发散。** 同一对里 PI 完全相同的只有
/// 3.06%，高−低的差中位 −8 道、5–95% 为 −46..+9；按低增益道号分段，
/// ch54–100 段差中位 +2、ch100–200 段 −13、ch200–300 段 −64。所以留哪条
/// **对计数可忽略、对能量不可忽略**：会因为留哪条而跨过 `MIN_CHANNEL` 的对
/// 占 0.18%（折到全部事例约 0.02%），但道号差中位 −8 道 ≈ 11% 能量。
///
/// 判据用「同探头同时戳」这个实测事实，不用 `EVT_PAIR` 那一列：实测
/// `EVT_PAIR` 为真的事例占 19.9%，而同戳对只占 7.9%，两者对不上（伴侣可能
/// 被上游过滤掉、或差一个时戳格），语义没核实清楚之前不拿它当判据。
///
/// 输入必须已按时间排好。同一时戳的一串通常只有几条，最多不过探头路数，
/// 所以串内 O(k²) 的比对可以忽略。
fn dedupe_gain_pairs<S: Satellite>(events: &mut Vec<Event<S>>) -> usize {
    let mut keep_run: Vec<bool> = Vec::with_capacity(64);
    let mut write = 0usize;
    let mut merged = 0usize;
    let mut start = 0usize;

    while start < events.len() {
        let mut stop = start + 1;
        while stop < events.len() && events[stop].time == events[start].time {
            stop += 1;
        }

        keep_run.clear();
        for a in start..stop {
            // 同一路探头在这一串里另有一条时就并掉：留低增益（`gain_type` 大）
            // 那条，两条增益档相同时留先出现的，规则与归并顺序无关
            let superseded = (start..stop).any(|b| {
                b != a
                    && events[b].detector_id == events[a].detector_id
                    && (events[b].gain_type > events[a].gain_type
                        || (events[b].gain_type == events[a].gain_type && b < a))
            });
            keep_run.push(!superseded);
        }

        // 决定已经算完，往前挪不会影响本串的判断；`write <= start + offset`
        // 恒成立，且 swap 只碰更靠前的位置，后面待挪的元素动不到
        for (offset, keep) in keep_run.iter().enumerate() {
            if *keep {
                events.swap(write, start + offset);
                write += 1;
            } else {
                merged += 1;
            }
        }
        start = stop;
    }

    events.truncate(write);
    merged
}

/// 候选窗里的 CPD 符合计数，随候选一起存下来。
///
/// 复用 `AcdCounts` 这个结构，但字段在 GECAM 上的含义要说清楚：HXMT HE 的
/// ACD 是逐事例带的标志位，GECAM 的 CPD 是一路独立的探测器、有自己的事例流，
/// 所以这里是「窗内 GRD 计数」对「窗内 CPD 计数」，不是同一批事例的两种标注。
///
/// * `n` / `n_bg`：候选窗、基线窗内通过准入的 GRD 事例数
/// * `n_acd` / `n_acd_bg`：同两个窗内的 CPD 事例数
///
/// **基线窗是候选两侧各 `CPD_BASELINE_SECONDS`，正好是搜索本底窗的两倍宽，
/// 而且不挖空窗、不夹 GTI**——口径与后果见 `CPD_BASELINE_SECONDS` 的注释。
/// * `n_acd_multi`：候选窗内、10 µs 内有另一路 CPD 一起响的事例数。一路响
///   可能是本底，多路同时响基本只能是穿过整台仪器的带电粒子。
///
/// **只统计，不否决。** 符合到什么程度才算带电粒子得拿真候选定标；在那之前
/// 存原始计数而不是比例，泊松误差还在，阈值日后可以复议而不必重跑全量。
///
/// 这一小时没有 CPD 文件时返回 `None` 而不是一组零。GECAM-A 在 2025-03-12
/// 之前整段都没有 CPD 产品，若填零，日后定标 CPD 判据时会把「没测」当成
/// 「测了是零」统计进去，直接把判据往松了标。
fn cpd_counts<S: Satellite>(
    chunk: &Chunk<S>,
    events: &[Event<S>],
    start: MissionElapsedTime<Gecam<S>>,
    stop: MissionElapsedTime<Gecam<S>>,
) -> Option<AcdCounts> {
    let cpd = chunk.cpd_file.as_ref()?;
    let (start, stop) = (start.met(), stop.met());
    let (before, after) = (start - CPD_BASELINE_SECONDS, stop + CPD_BASELINE_SECONDS);

    // events 已按时间排好，二分即可：候选一多，线性扫 1600 万个事例就不像话了
    let grd_within = |from: f64, to: f64| {
        let lower = events.partition_point(|event| event.time().met() < from);
        let upper = events.partition_point(|event| event.time().met() <= to);
        (upper - lower) as u32
    };

    let n = grd_within(start, stop);
    Some(AcdCounts {
        n,
        n_acd: cpd.count_within(start, stop),
        n_acd_multi: cpd.count_multi_within(start, stop),
        // 基线取候选窗两侧，挖掉候选本身
        n_bg: grd_within(before, after).saturating_sub(n),
        n_acd_bg: cpd
            .count_within(before, after)
            .saturating_sub(cpd.count_within(start, stop)),
    })
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
/// 差一倍以上）。**`baseline_seconds` 是标称的 2 s，不是活时间**，与搜索的
/// 本底窗也不是同一个窗——见 `CPD_BASELINE_SECONDS`。
///
/// 逐路基线还能看见一件 GTI 根本不表达的事：**探头级的部分停机**。GTI 是全
/// 仪器概念，25 路里掉一路、仪器照样"活着"，只是有效面积在变，而 `mean` 的
/// 分母不会跟着改、`fa` 就偏低。基线向量里的零格是唯一看得见它的地方。
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

    /// 审计窗与搜索窗的关系是个容易写错的地方（原注释就写成了"与搜索一致"，
    /// 实际是两倍），钉在这里，日后谁改了 `neighbor` 就会看见这条。
    #[test]
    fn the_audit_baseline_window_is_twice_the_search_one() {
        // 搜索取候选两侧各 `neighbor / 2`，审计取两侧各 `CPD_BASELINE_SECONDS`
        let search_half_width = SEARCH_NEIGHBOR_SECONDS / 2.0;
        assert_eq!(CPD_BASELINE_SECONDS, 2.0 * search_half_width);
        // `DetectorCounts::baseline_seconds` 存的是两侧合计的标称时长，不是活时间
        let counts = detector_counts(
            1,
            &[event(10.0, 1)],
            MissionElapsedTime::new(9.9),
            MissionElapsedTime::new(10.1),
        );
        assert_eq!(counts.baseline_seconds, 4.0 * search_half_width);
    }

    fn gain_event(time: f64, detector_id: u8, gain_type: u8, channel: i16) -> Event<SatC> {
        Event {
            time: MissionElapsedTime::new(time),
            channel,
            detector_id,
            gain_type,
            dead_time: 4.0,
            evt_type: 1,
            flag: if gain_type == 0 { 0 } else { 10 },
        }
    }

    #[test]
    fn a_dual_gain_pair_counts_once_and_the_low_gain_row_is_the_one_kept() {
        // 同一路探头、同一时戳的一高一低两条 = 同一个物理光子的两条记录
        let mut events = vec![
            gain_event(10.0, 3, 0, 440), // 高增益，接近饱和
            gain_event(10.0, 3, 1, 377), // 低增益，有效测量
        ];
        assert_eq!(dedupe_gain_pairs(&mut events), 1);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].gain_type, 1);
        assert_eq!(events[0].channel, 377);
    }

    #[test]
    fn the_merge_order_does_not_change_which_row_survives() {
        // 高增益那条先到还是后到，留下来的都得是低增益那条
        for order in [[0usize, 1], [1, 0]] {
            let rows = [gain_event(10.0, 3, 0, 440), gain_event(10.0, 3, 1, 377)];
            let mut events = vec![rows[order[0]].clone(), rows[order[1]].clone()];
            assert_eq!(dedupe_gain_pairs(&mut events), 1);
            assert_eq!(events[0].gain_type, 1);
        }
    }

    #[test]
    fn same_timestamp_across_detectors_is_a_particle_not_a_duplicate_and_is_kept() {
        // 带电粒子穿过整台仪器：各路同时响，这是要留下来给判据看的东西，
        // 不能跟一路探头内部的双增益重复混为一谈
        let mut events = vec![
            gain_event(10.0, 1, 0, 300),
            gain_event(10.0, 2, 0, 300),
            gain_event(10.0, 7, 0, 300),
        ];
        assert_eq!(dedupe_gain_pairs(&mut events), 0);
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn duplicates_are_merged_without_disturbing_the_time_order_around_them() {
        let mut events = vec![
            gain_event(9.0, 1, 0, 300),
            gain_event(10.0, 2, 0, 440),
            gain_event(10.0, 2, 1, 377),
            gain_event(10.0, 5, 0, 300),
            gain_event(11.0, 1, 0, 300),
        ];
        assert_eq!(dedupe_gain_pairs(&mut events), 1);
        let times: Vec<f64> = events.iter().map(|e| e.time.met()).collect();
        assert_eq!(times, vec![9.0, 10.0, 10.0, 11.0]);
        assert!(events.windows(2).all(|w| w[0].time <= w[1].time));
        // 10.0 那一串里留下的是探头 2 的低增益条和探头 5
        assert_eq!(events[1].detector_id, 2);
        assert_eq!(events[1].gain_type, 1);
        assert_eq!(events[2].detector_id, 5);
    }

    #[test]
    fn an_event_stream_without_duplicates_comes_back_untouched() {
        let mut events = vec![event(9.0, 1), event(10.0, 2), event(11.0, 3)];
        let before: Vec<f64> = events.iter().map(|e| e.time.met()).collect();
        assert_eq!(dedupe_gain_pairs(&mut events), 0);
        let after: Vec<f64> = events.iter().map(|e| e.time.met()).collect();
        assert_eq!(before, after);
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
