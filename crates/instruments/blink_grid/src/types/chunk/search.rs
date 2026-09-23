use blink_algorithms::detector_share::{MAX_DETECTOR_FRACTION, max_detector_fraction};
use blink_algorithms::snapshot_stepping::{SearchConfig, search_new};
use blink_core::traits::Event as _;
use blink_core::types::{MissionElapsedTime, Signal};
use std::sync::atomic::Ordering;
use uom::si::f64::*;

use super::Chunk;
use crate::io::posatt::{attitude_trajectory, interpolate_sampled, position_trajectory};
use crate::types::Event;
use crate::types::instrument::{Grid, Satellite};

/// 判为"读出空洞"的门槛：本底窗里最长的空段在本地速率下应有的计数 r·L。
///
/// 天格的读出在高计数率下会成帧丢数：辐射带里（|lat| > 40°，5–75 kc/s）事例
/// 以 5–16 ms 的密集帧到达，帧间 3–12 ms 一个事例都没有，四个探测器同步。
/// 搜索窗落在帧内、本底窗横跨帧和空洞，均值被帧间空洞拉低，帧内的普通计数
/// 就成了 fa=1e-199 的"暴发"——GRID-03B 一天 9213 个候选全是这种。实测
/// 2–15 kc/s 的秒空洞占比中位 0.0–0.7%，安静时段一天搜不出任何候选，所以
/// 只在本底窗里出现统计上不可能的空段时否决：r·L > 9.2 即 P(0) < 1e-4。
/// 在 1 kc/s 要 9 ms 的空段才触发（一个本底窗里约百次机会，误否决期望 1%），
/// 在 13 kc/s 只要 0.7 ms。全量首跑用 14（P<1e-6）时，辐射带 10–17 kc/s 的
/// 帧边界（空段接近 1 ms）漏了一批，幸存的显著候选一半仍在 |lat| ≥ 40°。
///
/// 速率 r 取本底窗速率与整次过境速率中的大者：成片丢数时窗内速率本身被压到
/// 几个 c/s（GRID-04 有 count=8、mean=0.00、率 2 c/s 的"候选"），按窗内速率
/// 算 r·L 永远到不了门槛，按过境速率算半秒空段就是几百。
///
/// 更正的做法是把空洞当 GTI 缺口、让本底按真实活时间归一，但 `search_new`
/// 的活时间夹取假设候选所在的一段 GTI 覆盖整个本底窗，毫秒级的洞会把本底窗
/// 夹到一帧里去，统计反而更差。见 `OPEN-QUESTIONS.md`。
const DEAD_GAP_EXPECTED_COUNTS: f64 = 9.2;

/// 读出可信的计数率上限（四路合计，c/s）。
///
/// 成帧丢数从这个量级开始：GRID-02 与 04 在 5 kc/s 附近就成片出现空洞，03B
/// 到 15 kc/s 才成片，但 13 kc/s 时已经有另一种形态——四路同时一个 1 ms 的
/// 尖峰（86 个对本底 4 个/ms）后面跟几毫秒的稀疏，合并四路后最长空段只有
/// 0.6 ms，空洞判据抓不到。全量首跑显著候选的一半落在 |lat| ≥ 40° 的高速率
/// 区，本底率中位 5.4 kc/s；加这道门后 |lat| < 40° 的显著候选一个不少
/// （03B 277/277），失去的只有辐射带。取本底窗速率与过境速率中的大者比较。
const RATE_CEILING: f64 = 5000.0;

/// 带电粒子否决：最显著一格里 ≥3 重同戳簇的个数 k₃ 与它的偶然期望 λ₃ 做泊松检验，
/// `P(K₃ ≥ k₃ | SAFETY·λ₃) < MAX_TRIPLE_P` 即否决。
///
/// 挡的是穿过整星的带电粒子：四块 GAGG 同时响应、各留一个大沉积，时间戳完全相同
/// （同一探头从不同戳，1.11e8 个探头内相邻对里严格为 0，所以同戳必然是跨探头的）。
/// 四路合成一路搜索没有组间符合可用，这道门是天格挡带电粒子的唯一手段。
///
/// **为什么不再用固定的占比阈（v11 的 f₃ > 0.5）。** 同戳簇最多 4 重（4 路、同探头
/// 不重格），`min_number = 8` ⇒ 只含一个簇的候选 f₃ ≤ 4/8 = 0.5，所以 f₃ > 0.5 对
/// 最典型的"一次四路穿越"恒不触发，实际等价于"窗内 ≥2 个三重簇才否决"。亮而短的
/// 注入网格（九格 × 6 万次）里它连一次都没响过。见 `OPEN-QUESTIONS.md` 未决项 19。
///
/// **λ₃ 两项相加。**
/// 1. 量化偶然：窗内各路事例不相干、撞进同一个 2⁻²² s 格。逐探头口径（同探头不重格、
///    三重必须来自 ≥3 路），`a_d = n_d/m`、`m = T/q`，
///    `λ₃ᵠ = m·Σ_{|S|≥3} Π_{d∈S} a_d · Π_{d∉S}(1−a_d)`，四路就 5 项。
/// 2. 本底粒子落进窗：本底三重簇率 r₃ × T。**这一项大 34 倍**（全池 1.82 对 0.054），
///    只算第一项会把 λ₃ 低估一个多数量级、偏向误杀。
///
/// **安全系数 2**：均匀独立假设在真暴发成团的时间结构下低估 λ₃，保结构的两种做法
/// （逐探头随机平移、局部密度解析）在 26 个干净候选上给出的低估上界都 ≤ 2 倍。
///
/// **阈 1e-3**：显著池（fa ≤ 1e-5）误杀期望 0.007 个候选，亮而短注入网格的误杀率
/// ≤ 0.07%；7 个闪电认证的 TGF k₃ = 0、p = 1，任何阈下都不碰。
const MAX_TRIPLE_P: f64 = 1e-3;
const TRIPLE_LAMBDA_SAFETY: f64 = 2.0;

/// 一簇同戳事例要几个才算粒子签名。二重不算：偶然期望 1% 量级，压不住噪声。
const TRIPLE: usize = 3;

/// 事例时戳的量化步：`TIME × 2²²` 的小数部分四颗星都严格为 0。
const TICK_S: f64 = 1.0 / 4_194_304.0;

/// 天格每颗星 4 路探头
const N_DETECTORS: usize = 4;

/// 本底窗 `[from, to]`（已夹到候选所在的 GTI 段内）里是否有读出空洞。
///
/// `events` 已按时间排好。空段包括窗口两端到最近事例的距离——窗口已夹在
/// GTI 内，端点上没有事例不是过境边界造成的。
fn has_dead_gap<S: Satellite>(events: &[Event<S>], from: f64, to: f64, pass_rate: f64) -> bool {
    if to <= from {
        return false;
    }
    let lo = events.partition_point(|e| e.time().met() < from);
    let hi = events.partition_point(|e| e.time().met() <= to);
    let window = &events[lo..hi];
    let longest = if window.is_empty() {
        to - from
    } else {
        let mut longest =
            (window[0].time().met() - from).max(to - window[window.len() - 1].time().met());
        for pair in window.windows(2) {
            longest = longest.max(pair[1].time().met() - pair[0].time().met());
        }
        longest
    };
    let rate = (window.len() as f64 / (to - from)).max(pass_rate);
    rate * longest > DEAD_GAP_EXPECTED_COUNTS
}

/// `events[lo..hi]` 里重数 ≥3 的同戳簇个数。时间戳相同的事例必然相邻，扫一遍分段即可。
fn triple_clusters<S: Satellite>(window: &[Event<S>]) -> usize {
    let (mut clusters, mut run) = (0usize, 1usize);
    for i in 1..=window.len() {
        if i < window.len() && window[i].time() == window[i - 1].time() {
            run += 1;
        } else {
            if run >= TRIPLE {
                clusters += 1;
            }
            run = 1;
        }
    }
    clusters
}

/// 时刻落在闭区间 `[from, to]` 内的那一段事例。
fn slice_between<S: Satellite>(events: &[Event<S>], from: f64, to: f64) -> &[Event<S>] {
    let lo = events.partition_point(|e| e.time().met() < from);
    let hi = events.partition_point(|e| e.time().met() <= to);
    &events[lo..hi.max(lo)]
}

/// 量化偶然的三重簇期望（逐探头口径），见 `MAX_TRIPLE_P`。
fn chance_triples(per_detector: &[usize; N_DETECTORS], duration: f64) -> f64 {
    let m = duration / TICK_S;
    if m <= 0.0 {
        return 0.0;
    }
    let a: Vec<f64> = per_detector
        .iter()
        .map(|&n| (n as f64 / m).min(1.0))
        .collect();
    let mut sum = 0.0;
    for subset in 0u32..(1 << N_DETECTORS) {
        if (subset.count_ones() as usize) < TRIPLE {
            continue;
        }
        sum += (0..N_DETECTORS)
            .map(|d| {
                if subset >> d & 1 == 1 {
                    a[d]
                } else {
                    1.0 - a[d]
                }
            })
            .product::<f64>();
    }
    m * sum
}

/// `P(X ≥ k)`，`X ~ Poisson(lam)`。
///
/// 直接累尾，不用 `1 − 前 k 项`：后者在尾部小时灾难性相消，λ ~ 1e-3、k ≥ 5 直接得 0
/// （真值 8.3e-18），而这里 λ₃ 能小到 1e-6。
fn poisson_tail(k: usize, lam: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if lam <= 0.0 {
        return 0.0;
    }
    // 首项 e^{-λ} λ^k / k! 在对数里算，之后逐项乘 λ/(i+1)
    let ln_first = -lam + k as f64 * lam.ln() - (1..=k).map(|i| (i as f64).ln()).sum::<f64>();
    let mut term = ln_first.exp();
    let mut total = 0.0;
    for i in k..k + 1000 {
        total += term;
        if term < 1e-18 * total.max(1e-300) {
            break;
        }
        term *= lam / (i + 1) as f64;
    }
    total.min(1.0)
}

/// 最显著一格 `[start, stop]` 的粒子检验 p 值：`P(K₃ ≥ k₃ | SAFETY·λ₃)`。
///
/// `background_rate` 是本底三重簇率（个/s）。两端都是事例本身的时刻（`Candidate` 的
/// start、delay、bin_size_best 都来自事例），闭区间比较，边界上那一簇不会漏。
fn triple_p_value<S: Satellite>(
    events: &[Event<S>],
    start: MissionElapsedTime<Grid<S>>,
    stop: MissionElapsedTime<Grid<S>>,
    background_rate: f64,
) -> f64 {
    let window = slice_between(events, start.met(), stop.met());
    let k3 = triple_clusters(window);
    if k3 == 0 {
        return 1.0;
    }
    let mut per_detector = [0usize; N_DETECTORS];
    for e in window {
        per_detector[(e.detector as usize).min(N_DETECTORS - 1)] += 1;
    }
    let duration = stop.met() - start.met();
    let lam = chance_triples(&per_detector, duration) + background_rate * duration;
    poisson_tail(k3, TRIPLE_LAMBDA_SAFETY * lam)
}

pub(super) fn search<S: Satellite>(chunk: &Chunk<S>) -> Vec<Signal<Event<S>>> {
    let gti: Vec<[MissionElapsedTime<Grid<S>>; 2]> = chunk
        .gti
        .iter()
        .map(|g| [MissionElapsedTime::new(g[0]), MissionElapsedTime::new(g[1])])
        .collect();
    let inside = |t: f64| gti.iter().any(|g| t >= g[0].met() && t <= g[1].met());

    // 四路探测器、若干次过境拼在一起再排序。事例准入见 `Event::keep`；实测事例
    // 都落在各自文件的 GTI 内，这里再按 GTI 过滤一次是为了和曝光核算同一口径，
    // 丢掉的数目记进诊断。
    let mut n_outside = 0usize;
    let mut events: Vec<Event<S>> = chunk
        .passes
        .iter()
        .flat_map(|p| p.events::<S>())
        .filter(|e| e.keep())
        .filter(|e| {
            let ok = inside(e.time().met());
            if !ok {
                n_outside += 1;
            }
            ok
        })
        .collect();
    events.sort();

    let results = search_new(
        &events,
        1,
        chunk.span[0],
        chunk.span[1],
        // 活时间就是这几段过境。天格一小时里往往只有十几分钟有数据，两头是
        // 硬边界，本底窗必须按活时间归一，否则边界候选的本底会被压低。
        &gti,
        SearchConfig {
            min_duration: Time::new::<uom::si::time::microsecond>(0.0),
            max_duration: Time::new::<uom::si::time::millisecond>(1.0),
            neighbor: Time::new::<uom::si::time::second>(1.0),
            hollow: Time::new::<uom::si::time::millisecond>(10.0),
            false_positive_per_year: 20.0,
            min_number: 8,
            // 单组：4 个 GAGG 同型，合成一路
            coincidence: 1,
        },
    );

    // 各次过境的本底三重簇率（个/s），粒子否决用，见 `MAX_TRIPLE_P`
    let pass_triple_rates: Vec<f64> = chunk
        .passes
        .iter()
        .map(|p| {
            // 事例只取到本小时内，过境时长也截到本小时
            let (from, to) = (
                p.start.max(chunk.span[0].met()),
                p.stop.min(chunk.span[1].met()),
            );
            if S::SHARED_FRAME || to <= from {
                return 0.0;
            }
            triple_clusters(slice_between(&events, from, to)) as f64 / (to - from)
        })
        .collect();

    let attitudes = attitude_trajectory::<S>(&chunk.posatt);
    let positions = position_trajectory::<S>(&chunk.posatt);
    let fitted = crate::io::orbit_fit::trajectory::<S>(&chunk.orbit_fit);

    let mut n_dropped = 0usize;
    let mut n_fitted = 0usize;
    let mut n_dead_gap = 0usize;
    let mut n_high_rate = 0usize;
    let mut n_simultaneous = 0usize;
    let mut n_no_attitude = 0usize;
    let mut n_single_detector = 0usize;
    let half_neighbor = 0.5_f64;
    let signals = results
        .into_iter()
        .filter_map(|candidate| {
            // 读出空洞否决。本底窗夹到候选所在的那段 GTI 里，过境边界不算空洞。
            let (cs, ce) = (candidate.start.met(), candidate.stop.met());
            let seg = chunk.gti.iter().find(|g| cs >= g[0] && cs <= g[1]);
            let (from, to) = match seg {
                Some(g) => (
                    (cs - half_neighbor).max(g[0]),
                    (ce + half_neighbor).min(g[1]),
                ),
                None => (cs - half_neighbor, ce + half_neighbor),
            };
            // 过境速率：准入后的事例数 / 过境时长
            let pass_rate = chunk
                .passes
                .iter()
                .find(|p| cs >= p.start && cs <= p.stop)
                .map(|p| {
                    let a = events.partition_point(|e| e.time().met() < p.start);
                    let b = events.partition_point(|e| e.time().met() <= p.stop);
                    (b - a) as f64 / (p.stop - p.start).max(1e-9)
                })
                .unwrap_or(0.0);
            if has_dead_gap(&events, from, to, pass_rate) {
                n_dead_gap += 1;
                return None;
            }
            let window_rate = {
                let a = events.partition_point(|e| e.time().met() < from);
                let b = events.partition_point(|e| e.time().met() <= to);
                (b - a) as f64 / (to - from).max(1e-9)
            };
            if window_rate.max(pass_rate) > RATE_CEILING {
                n_high_rate += 1;
                return None;
            }
            // 带电粒子否决看最显著的那一格：由 start 偏移 delay 得到，两者都是
            // 事例的时刻，差与和在 f64 下精确。
            //
            // 共帧读出的星整个跳过这道门：那里任一路击中就开一个 28.6 µs 的帧、
            // 帧内所有事例共用触发那一击的时戳，**同戳是读出结构的必然产物不是
            // 物理**。实测把 03B 的真暴按共帧读出重放，这道门会否掉一半左右的真
            // 暴发（最亮的那个有闪电认证的 TGF 有 76% 的试验被它砍掉），而穿星
            // 粒子在共帧下只留 4 个计数、根本够不着候选门，轮不到这道门。
            //
            // 本底三重簇率取两者中的大者：候选所在 ±0.5 s 本底窗（扣掉候选窗本身，
            // 免得粒子候选用自己的簇抬高本底）与整次过境。本底窗里期望只有一个上下
            // 的簇，单独用它常常是 0，λ₃ 被压低就偏向误杀；取大者只会让门更难触发。
            let best_start = candidate.start + candidate.delay;
            let best_stop = best_start + candidate.bin_size_best;
            if !S::SHARED_FRAME {
                let local = {
                    let all = triple_clusters(slice_between(&events, from, to));
                    let own = triple_clusters(slice_between(&events, cs, ce));
                    let live = (to - from) - (ce - cs);
                    if live > 0.0 {
                        all.saturating_sub(own) as f64 / live
                    } else {
                        0.0
                    }
                };
                let pass = chunk
                    .passes
                    .iter()
                    .position(|p| cs >= p.start && cs <= p.stop)
                    .map(|i| pass_triple_rates[i])
                    .unwrap_or(0.0);
                if triple_p_value(&events, best_start, best_stop, local.max(pass)) < MAX_TRIPLE_P {
                    n_simultaneous += 1;
                    return None;
                }
            }
            // 单路毛刺否决。四块 GAGG 并排同向，真暴发四路均分：v3 全量真候选的单路
            // 最大占比中位 0.36、最高 0.56；超过 0.9 的 3 个全是一路探测器自己在闹
            // （03B 2023-08-08T19:53:50 一路占 100%、PI 6；2023-08-10 三分钟内两个
            // 候选同一路占 90–95%）。单组搜索没有组间符合，这是挡单路毛刺的唯一手段。
            if max_detector_fraction(&events, candidate.start, candidate.stop, |e| e.detector)
                > MAX_DETECTOR_FRACTION
            {
                n_single_detector += 1;
                return None;
            }
            let peak = candidate.start + candidate.bin_size_best / 2.0;
            // 2024-02 之后位姿文件里没有位置解（POS_TYPE=0，全 NaN），这样的
            // 候选定不了位。丢可以，静默丢不行——记账见 `diagnostics`。
            // 位姿有位置解就用它；没有（2024-02-09 起）退回拟合轨道表；都没有才丢
            let position = match interpolate_sampled(&positions, peak) {
                Some(p) => p,
                None => match interpolate_sampled(&fitted, peak) {
                    Some(p) => {
                        n_fitted += 1;
                        p
                    }
                    None => {
                        n_dropped += 1;
                        return None;
                    }
                },
            };
            // 姿态解也会整段缺失（v3 全量 82 个候选、3 个显著落在里面）。候选的
            // 实质是时间加位置，姿态只是元数据：缺了照留，留空并记账。
            let attitude = interpolate_sampled(&attitudes, peak);
            if attitude.is_none() {
                n_no_attitude += 1;
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
                position,
                // 天格没有反符合探测器
                acd: None,
                // 暂未填：天格 4 路，方向分析的收益还没评估，见 blink_core::DetectorCounts
                detectors: None,
            })
        })
        .collect::<Vec<_>>();

    chunk
        .dropped_no_ephemeris
        .store(n_dropped, Ordering::Relaxed);
    chunk
        .positions_from_orbit_fit
        .store(n_fitted, Ordering::Relaxed);
    chunk.events_outside_gti.store(n_outside, Ordering::Relaxed);
    chunk.dropped_dead_gap.store(n_dead_gap, Ordering::Relaxed);
    chunk
        .dropped_high_rate
        .store(n_high_rate, Ordering::Relaxed);
    chunk
        .dropped_simultaneous
        .store(n_simultaneous, Ordering::Relaxed);
    chunk
        .without_attitude
        .store(n_no_attitude, Ordering::Relaxed);
    chunk
        .dropped_single_detector
        .store(n_single_detector, Ordering::Relaxed);
    signals
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Sat03B;

    fn at(times: &[f64]) -> Vec<Event<Sat03B>> {
        times
            .iter()
            .map(|t| Event {
                time: MissionElapsedTime::new(*t),
                channel: 20,
                detector: 0,
                evt_type: 1,
                energy_kev: 100.0,
                overflow: false,
            })
            .collect()
    }

    fn on_detectors(times_and_detectors: &[(f64, u8)]) -> Vec<Event<Sat03B>> {
        times_and_detectors
            .iter()
            .map(|(t, d)| Event {
                time: MissionElapsedTime::new(*t),
                channel: 20,
                detector: *d,
                evt_type: 1,
                energy_kev: 100.0,
                overflow: false,
            })
            .collect()
    }

    fn detector_share(events: &[Event<Sat03B>]) -> f64 {
        max_detector_fraction(
            events,
            events[0].time(),
            events[events.len() - 1].time(),
            |e| e.detector,
        )
    }

    #[test]
    fn a_burst_shared_by_the_four_detectors_is_kept() {
        let events = on_detectors(
            &(0..12)
                .map(|i| (100.0 + i as f64 * 1e-5, (i % 4) as u8))
                .collect::<Vec<_>>(),
        );
        assert!((detector_share(&events) - 0.25).abs() < 1e-12);
        assert!(detector_share(&events) <= MAX_DETECTOR_FRACTION);
    }

    #[test]
    fn a_glitch_in_one_detector_is_vetoed_even_with_a_background_count() {
        // 8 个计数里 7 个来自 2 号探测器：7/8 = 0.875
        let mut v: Vec<(f64, u8)> = (0..7).map(|i| (100.0 + i as f64 * 1e-5, 2)).collect();
        v.push((100.00005, 0));
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert!(detector_share(&on_detectors(&v)) > MAX_DETECTOR_FRACTION);
    }

    fn p_value(times_and_detectors: &[(f64, u8)], background_rate: f64) -> f64 {
        let events = on_detectors(times_and_detectors);
        triple_p_value(
            &events,
            events[0].time(),
            events[events.len() - 1].time(),
            background_rate,
        )
    }

    /// 8 个事例铺在 `span` 秒上、依次落在 4 路，`crossing` 处插一次四路同戳。
    fn with_crossings(span: f64, crossings: &[f64]) -> Vec<(f64, u8)> {
        let mut v: Vec<(f64, u8)> = (0..8)
            .map(|i| (100.0 + span * i as f64 / 7.0, (i % 4) as u8))
            .collect();
        for &c in crossings {
            v.extend((0..4).map(|d| (100.0 + c, d as u8)));
        }
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        v
    }

    #[test]
    fn a_single_four_detector_crossing_is_caught() {
        // v11 的 f₃ > 0.5 对这种形态恒不触发（4/12 = 0.33）；泊松检验看的是
        // "300 µs 里出现一次三重簇"本身有多不可能：量化 7e-5 + 本底 1 个/s × 300 µs
        let p = p_value(&with_crossings(300e-6, &[150e-6]), 1.0);
        assert!(p < MAX_TRIPLE_P, "p = {p}");
    }

    #[test]
    fn a_single_crossing_in_a_dense_window_sits_near_the_threshold() {
        // 判据最薄的一处：12 个计数挤在 100 µs 里，量化偶然本身就有 6e-4，一个簇的
        // p ≈ 1.4e-3，放过。这与 v10 那 4 个单簇候选 p = 4–5e-4（×2 后 8–10e-4）同一档，
        // 单簇的证据强度本来就只有这么多。见 `OPEN-QUESTIONS.md` 未决项 19
        let p = p_value(&with_crossings(100e-6, &[50e-6]), 1.0);
        assert!(p > 1e-4 && p < 1e-2, "p = {p}");
    }

    #[test]
    fn repeated_crossings_are_caught() {
        let p = p_value(&with_crossings(300e-6, &[50e-6, 200e-6]), 1.0);
        assert!(p < 1e-6, "p = {p}");
    }

    #[test]
    fn a_crossing_is_let_through_where_particles_are_common() {
        // 本底三重簇率高到 20 个/s 时，1 ms 窗里套进一个粒子的期望 0.02（×2 = 0.04），
        // 一个簇的 p ≈ 0.04，判据自己让路
        let p = p_value(&with_crossings(1e-3, &[0.5e-3]), 20.0);
        assert!(p > MAX_TRIPLE_P, "p = {p}");
    }

    #[test]
    fn a_burst_without_clusters_is_never_vetoed() {
        let v: Vec<(f64, u8)> = (0..20)
            .map(|i| (100.0 + i as f64 * 5e-6, (i % 4) as u8))
            .collect();
        assert_eq!(p_value(&v, 1.0), 1.0);
    }

    #[test]
    fn pairs_on_one_timestamp_do_not_count() {
        // 二重同戳的偶然期望是 1% 量级，压不住噪声，所以只数三重及以上
        let v: Vec<(f64, u8)> = (0..8)
            .map(|i| (100.0 + (i / 2) as f64 * 2e-4, (i % 2) as u8))
            .collect();
        assert_eq!(triple_clusters(&on_detectors(&v)), 0);
        assert_eq!(p_value(&v, 1.0), 1.0);
    }

    #[test]
    fn the_cluster_on_the_window_edge_is_counted() {
        // 簇正好落在窗口末端：闭区间比较才数得到它
        let v = [
            (100.0, 0),
            (100.0001, 1),
            (100.0002, 2),
            (100.0003, 0),
            (100.0003, 1),
            (100.0003, 2),
            (100.0003, 3),
        ];
        assert_eq!(triple_clusters(&on_detectors(&v)), 1);
        assert!(p_value(&v, 0.0) < 1.0);
    }

    #[test]
    fn chance_triples_match_the_closed_form_for_equal_detectors() {
        // 四路各 n 个、m 格：λ₃ = m·[4a³(1−a) + a⁴]，a = n/m
        let t = 1e-3;
        let m = t / TICK_S;
        let a = 5.0 / m;
        let want = m * (4.0 * a.powi(3) * (1.0 - a) + a.powi(4));
        let got = chance_triples(&[5, 5, 5, 5], t);
        assert!((got / want - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_poisson_tail_does_not_cancel() {
        // 1 − 前 k 项在这里会得 0；直接累尾：P(X≥5 | 1e-3) ≈ λ⁵/5!·e^{-λ}
        let lam: f64 = 1e-3;
        let want = lam.powi(5) / 120.0 * (-lam).exp();
        assert!((poisson_tail(5, lam) / want - 1.0).abs() < 1e-3);
        assert_eq!(poisson_tail(0, lam), 1.0);
        assert!((poisson_tail(1, 2.0) - (1.0 - (-2.0f64).exp())).abs() < 1e-12);
    }

    #[test]
    fn shared_frame_satellites_skip_the_particle_veto() {
        // 共帧星上同戳是读出结构的必然产物，不是物理——这道门对它们整个跳过
        assert!(!Sat03B::SHARED_FRAME);
        assert!(crate::types::Sat02::SHARED_FRAME);
        assert!(crate::types::Sat04::SHARED_FRAME);
        assert!(crate::types::Sat07::SHARED_FRAME);
    }

    #[test]
    fn a_uniform_stream_has_no_dead_gap() {
        // 2000 c/s 均匀铺满 1 s：最长空段 0.5 ms，r·L = 1
        let events = at(&(0..2000)
            .map(|i| 100.0 + i as f64 * 5e-4)
            .collect::<Vec<_>>());
        assert!(!has_dead_gap(&events, 100.0, 101.0, 2000.0));
    }

    #[test]
    fn a_frame_gap_at_high_rate_is_a_dead_gap() {
        // 帧结构：每 10 ms 前 5 ms 有 75 kc/s 的事例，后 5 ms 全空——r·L ≈ 37500×0.005 = 187
        let mut times = Vec::new();
        for frame in 0..100 {
            let t0 = 100.0 + frame as f64 * 0.01;
            times.extend((0..375).map(|i| t0 + i as f64 * 5e-3 / 375.0));
        }
        assert!(has_dead_gap(&at(&times), 100.0, 101.0, 37500.0));
    }

    #[test]
    fn a_long_but_poisson_plausible_gap_at_low_rate_is_not_a_dead_gap() {
        // 300 c/s，最长空段 20 ms：r·L = 6，P(0) = 2.5e-3，不能算读出空洞
        let mut times: Vec<f64> = (0..300).map(|i| 100.0 + i as f64 / 300.0).collect();
        times.retain(|t| !(100.50..100.52).contains(t));
        assert!(!has_dead_gap(&at(&times), 100.0, 101.0, 300.0));
    }

    #[test]
    fn the_window_edges_count_as_gaps() {
        // 窗口前 200 ms 一个事例都没有，之后 5 kc/s：r·L = 4000×0.2 = 800
        let events = at(&(0..4000)
            .map(|i| 100.2 + i as f64 * 2e-4)
            .collect::<Vec<_>>());
        assert!(has_dead_gap(&events, 100.0, 101.0, 2000.0));
    }

    #[test]
    fn a_nearly_empty_window_is_judged_by_the_pass_rate() {
        // 窗里只有 8 个事例挤在 1 ms 内（窗内速率 8 c/s），但整次过境是 1 kc/s：
        // 半秒的空段按过境速率是 500 个期望计数——这是成片丢数里的孤帧，不是暴发
        let events = at(&(0..8).map(|i| 100.5 + i as f64 * 1e-4).collect::<Vec<_>>());
        assert!(has_dead_gap(&events, 100.0, 101.0, 1000.0));
        // 同样的 8 个事例，如果整次过境本来就只有 8 c/s，那半秒空段是正常的
        assert!(!has_dead_gap(&events, 100.0, 101.0, 8.0));
    }

    #[test]
    fn a_millisecond_frame_edge_at_belt_rates_is_caught() {
        // 13 kc/s 的帧边界留 0.9 ms 空段：r·L = 11.7，P(0) = 8e-6——用 1e-6 的门槛会漏
        let mut times: Vec<f64> = (0..13000).map(|i| 100.0 + i as f64 / 13000.0).collect();
        times.retain(|t| !(100.5000..100.5009).contains(t));
        assert!(has_dead_gap(&at(&times), 100.0, 101.0, 13000.0));
    }
}
