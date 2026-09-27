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

/// 读出空洞否决：本底窗里最长的空段，在窗内实际计数率下出现的概率低于这个值就否决。
///
/// **要防的是什么。** 候选窗落在有数据的一段里，本底窗里却夹着数据没记下来的时间，
/// 本底被拉低，普通计数就成了"暴发"。天格有两种丢法，在数据里都表现为泊松涨落解释
/// 不了的空白（2026-09-27 实测，`OPEN-QUESTIONS.md` 未决项 23）：
///
/// 1. **高计数率下缓冲区按个数装满就停**：03B 每 1260 个事例停约 4 ms 的整数倍、每
///    20160 个停约 340 ms，约 100 kc/s 以上出现；02/04/07 每帧 352 个，几千 c/s 起，
///    计数率越高每秒交出的帧越少。全在辐射带里。
/// 2. **03B 四路恰好同时丢包**：03B 每路 20 个事例一包，约 3% 的包没有到手；单路丢包
///    时其余三路照常，四路合计里只是计数率少 1/4，不构成空白，这里不管；四路同时丢
///    才留下几百毫秒的空白，约 1% 的本底窗。
///
/// **计数率怎么估：`r = ln2 / 窗内相邻间隔的中位数`。** 泊松过程的间隔中位数是 ln2/r；
/// 丢数时绝大多数间隔落在有数据的段里，中位数反映的是有数据时的真实计数率，不被空白
/// 拉低，也不被几百秒外的辐射带抬高。候选窗本身的间隔不计入，免得暴发把 r 抬高。
///
/// **阈怎么定：按整个窗里"最长的那个空段"算概率。** 窗里 n 个间隔（含两端到窗边）中
/// 最长的一个 ≥ L 的概率是 `1 − (1 − e^{−rL})^n`，小于 `DEAD_GAP_ALPHA` 即否决——每个
/// 窗的纯统计误砍固定在 α 量级，不随计数率与窗内事例数变化。
///
/// **旧判据（v10–v14）错在哪。** 旧的是 `max(窗内率, 整次过境率) · L > 9.2`：9.2 是单个
/// 间隔的 1e-4，一个窗里几百到上千个间隔合起来误砍 4–10%；更糟的是过境一旦经过辐射带，
/// 过境平均率被抬高几倍，低纬正常本底里的普通涨落也触发——随机取的 6.8 万个安静窗里
/// 误砍 43%（03B），另外三星 24–49%。
///
/// **实测（四星各 18 天，同一批窗）**：以读出缓冲的帧大小作独立标签的丢数窗，检出
/// 99.6–99.9%（旧 99.9–100%）；安静窗触发 0.9–1.6%（旧 24–49%），其中 P < 1e-12 的
/// 是真空白（四路同时丢包、MCU 星上被压低的成帧段），贴着阈值的纯统计误砍约 0.2%。
const DEAD_GAP_ALPHA: f64 = 1e-3;

// 计数率上限（v10–v15 的 RATE_CEILING = 5000 c/s，取本底窗率与整次过境率的大者）已删除
// （2026-09-27）。本底高本身不造假信号——泊松检验已按本底算显著性；这道门是"读出坏了"的
// 替代指标，而读出坏了的直接判据（空白）已由 `has_dead_gap` 按实测验证过。它还和旧空洞门一样
// 用了整次过境率，经过辐射带的过境里低纬安静段也被砍。去掉后高纬、高本底的事件会进候选，
// 由下游按时长、能谱、位置分类。见 `OPEN-QUESTIONS.md` 未决项 24。

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

/// 候选门的最少计数，与 `SearchConfig.min_number` 同值；扣簇后的剩余计数也要够它。
///
/// 泊松检验问的是"窗里出现 k₃ 个簇有多意外"，但一个窗之所以成为候选，可能**正是因为**
/// 那一簇的 3–4 个同戳计数凑满了 8——条件在"已经是候选"上，簇就不再意外，检验看不出来。
/// v12/v13 回归日里有 3 个显著候选是这种形态：窗里恰好一次四路同戳、另有 4–6 个计数，
/// p = 1.2–1.9e-3 刚过阈，扣掉簇只剩 4–6 个。同戳计数是粒子不是光子，扣掉之后不够候选门
/// 就不是候选。真暴发里偶然混进一簇的，扣 3–4 个之后计数通常仍 ≥ 8；7 个闪电认证 k₃ = 0，
/// 这道门碰不到它们。
const MIN_COUNTS: u32 = 8;

/// 事例时戳的量化步：`TIME × 2²²` 的小数部分四颗星都严格为 0。
const TICK_S: f64 = 1.0 / 4_194_304.0;

/// 天格每颗星 4 路探头
const N_DETECTORS: usize = 4;

/// 片上标定脉冲：搜索之前把脉冲事例从流里删掉（v17 起；此前是否决脉冲开着时的候选）。
///
/// 脉冲一发让四路在同一个时戳上各记一个 500–900 keV 的事例，按固定周期重复，每天在固定的
/// UTC 时段开一段（03B 20/23 个脉冲显著候选在 05:01–07:23），占 3–5% 的过境。03B 1 kHz，一个
/// 1 ms 窗装得下两发 = 8 个计数 = `min_number`，脉冲自己就能凑出"暴发"；共帧三星 500 Hz，一窗
/// 最多一发。**删事例而不是否决候选**：脉冲开着时天空照样在被观测，删掉脉冲后这段照常搜，
/// 也就不存在"候选被砍了、曝光却还算着"的账。
///
/// 周期：03B 按脉冲序号线性回归 4194.7672 tick = 1.000110 ms（2022-03-19；2022-05、2023-08、
/// 2023-11 的脉冲窗相邻簇落在梳齿上的占比 0.995–0.998）；共帧三星 8388/8389 tick
/// （2.000011–2.000023 ms，`evidence/grid04` 的 `pulser_period.py`），取 8388.68。
fn pulser_period_ticks<S: Satellite>() -> f64 {
    if S::SHARED_FRAME { 8388.68 } else { 4194.7672 }
}
/// 梳齿容差（tick）：时戳量化把相邻间隔打散成相邻两档；共帧三星的周期各差零点零几 tick，
/// 最多 4 倍谐波下累积不到 0.2 tick。
const PULSER_TOLERANCE_TICKS: f64 = 1.5;
/// 只看最近的 4 次谐波：漏发几发脉冲的间隔落在 2–4 倍齿上。
const PULSER_MAX_HARMONIC: f64 = 4.0;

/// 本底窗 `[from, to]`（已夹到候选所在的 GTI 段内）里是否有读出空洞，见 `DEAD_GAP_ALPHA`。
///
/// `events` 已按时间排好。空段包括窗口两端到最近事例的距离——窗口已夹在 GTI 内，端点上
/// 没有事例不是过境边界造成的。
///
/// **按不同的时戳算，不按事例算。** 空洞问的是"有没有一段时间读出什么都没有"，单位是
/// 读出发生的时刻。共帧星（02/04/07）一帧里几路共用触发那一击的时戳，成帧时一半以上的
/// 相邻间隔是 0，按事例算中位数就是 0、计数率无从估计——第一版在这里直接放行，让 MCU 星
/// 的成帧段整个绕过了这道门（回归日 GRID-07 2024-03-15 多出 10 个显著的孤帧）。
///
/// **候选窗 `[cs, ce]` 的间隔只在它占窗内不到一半时才排除。** 排除是为了不让亮暴发把
/// 计数率抬高；但成帧丢数里的孤帧整个就是候选（本底窗 352 个时戳里 348–350 个在候选窗内），
/// 排除后只剩一两个跨过空白的长间隔，计数率被估成几 c/s，空白反倒显得正常。
fn has_dead_gap<S: Satellite>(events: &[Event<S>], from: f64, to: f64, cs: f64, ce: f64) -> bool {
    if to <= from {
        return false;
    }
    let lo = events.partition_point(|e| e.time().met() < from);
    let hi = events.partition_point(|e| e.time().met() <= to);
    let mut stamps: Vec<f64> = events[lo..hi].iter().map(|e| e.time().met()).collect();
    stamps.dedup();
    if stamps.len() < 3 {
        // 候选自身就有 ≥ 8 个事例落在窗里；到这里是整窗几乎只有一两个读出时刻
        return true;
    }
    let first = stamps[0];
    let last = stamps[stamps.len() - 1];
    let mut longest = (first - from).max(to - last);
    let mut all: Vec<f64> = Vec::with_capacity(stamps.len());
    let mut outside: Vec<f64> = Vec::with_capacity(stamps.len());
    for pair in stamps.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        longest = longest.max(b - a);
        all.push(b - a);
        if !(a >= cs && b <= ce) {
            outside.push(b - a);
        }
    }
    let intervals = if 2 * outside.len() > all.len() {
        &mut outside
    } else {
        &mut all
    };
    let mid = intervals.len() / 2;
    let median = *intervals
        .select_nth_unstable_by(mid, |x, y| x.partial_cmp(y).unwrap())
        .1;
    let rate = std::f64::consts::LN_2 / median;
    // n 个间隔（含两端）里最长的 ≥ L 的概率：1 − (1 − e^{−rL})^n
    let n = (stamps.len() + 1) as f64;
    let p_longest = -(n * (-(-rate * longest).exp()).ln_1p()).exp_m1();
    p_longest < DEAD_GAP_ALPHA
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

/// 从按时间排好的事例流里删掉片上标定脉冲，返回删掉的事例数。见 `pulser_period_ticks`。
///
/// 脉冲簇 = 同一时戳上 ≥ 3 个事例（一路偶尔正处在死时间或丢了包，只剩三路），**且前后两侧
/// 都有**另一个这样的簇与它相隔周期的 1–4 倍 ± 容差。只看一侧时，共帧星高计数率下自然成帧的
/// 三四路同戳簇偶然对上梳齿的机会约 0.7%；两侧同时对上约 5e-5。天然穿星粒子约每秒一次，
/// 落在 ±1.5 tick 梳齿上的机会约 4e-4，而且同样要两侧都有脉冲才会被删。
fn remove_pulser<S: Satellite>(events: &mut Vec<Event<S>>) -> usize {
    let period = pulser_period_ticks::<S>();
    // 同戳簇：(tick, 起, 止)
    let mut clusters: Vec<(f64, usize, usize)> = Vec::new();
    let mut i = 0;
    while i < events.len() {
        let mut j = i + 1;
        while j < events.len() && events[j].time() == events[i].time() {
            j += 1;
        }
        if j - i >= TRIPLE {
            clusters.push(((events[i].time().met() / TICK_S).round(), i, j));
        }
        i = j;
    }
    let on_comb = |d: f64| {
        let k = (d / period).round();
        (1.0..=PULSER_MAX_HARMONIC).contains(&k) && (d - k * period).abs() <= PULSER_TOLERANCE_TICKS
    };
    let reach = PULSER_MAX_HARMONIC * period + PULSER_TOLERANCE_TICKS;
    let mut drop = vec![false; events.len()];
    let mut removed = 0;
    for (c, &(tick, a, b)) in clusters.iter().enumerate() {
        let before = clusters[..c]
            .iter()
            .rev()
            .take_while(|x| tick - x.0 <= reach)
            .any(|x| on_comb(tick - x.0));
        let after = clusters[c + 1..]
            .iter()
            .take_while(|x| x.0 - tick <= reach)
            .any(|x| on_comb(x.0 - tick));
        if before && after {
            drop[a..b].iter_mut().for_each(|d| *d = true);
            removed += b - a;
        }
    }
    if removed > 0 {
        let mut k = 0;
        events.retain(|_| {
            k += 1;
            !drop[k - 1]
        });
    }
    removed
}

/// 最显著一格里不在 ≥3 重同戳簇上的事例数。
fn residual_counts<S: Satellite>(window: &[Event<S>]) -> usize {
    let (mut in_clusters, mut run) = (0usize, 1usize);
    for i in 1..=window.len() {
        if i < window.len() && window[i].time() == window[i - 1].time() {
            run += 1;
        } else {
            if run >= TRIPLE {
                in_clusters += run;
            }
            run = 1;
        }
    }
    window.len() - in_clusters
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
    let n_pulser = remove_pulser(&mut events);

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
            min_number: MIN_COUNTS,
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
    let mut n_simultaneous = 0usize;
    let mut n_no_attitude = 0usize;
    let mut n_single_detector = 0usize;
    let mut n_residual = 0usize;
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
            if has_dead_gap(&events, from, to, cs, ce) {
                n_dead_gap += 1;
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
                if residual_counts(slice_between(&events, best_start.met(), best_stop.met()))
                    < MIN_COUNTS as usize
                {
                    n_residual += 1;
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
        .dropped_simultaneous
        .store(n_simultaneous, Ordering::Relaxed);
    chunk
        .without_attitude
        .store(n_no_attitude, Ordering::Relaxed);
    chunk
        .dropped_single_detector
        .store(n_single_detector, Ordering::Relaxed);
    chunk.dropped_pulser.store(n_pulser, Ordering::Relaxed);
    chunk.dropped_residual.store(n_residual, Ordering::Relaxed);
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

    /// 1 kHz 脉冲串（每发四路同戳）+ 零星本底，时刻落在 2⁻²² s 栅格上
    fn pulser_train(from: f64, n: usize, missing_every: usize) -> Vec<(f64, u8)> {
        let mut v = Vec::new();
        for i in 0..n {
            if missing_every > 0 && i % missing_every == 3 {
                continue;
            }
            let tick = (from / TICK_S).round() + (i as f64 * 4194.7672).round();
            v.extend((0..4).map(|d| (tick * TICK_S, d as u8)));
        }
        for i in 0..200 {
            v.push((from + 0.0013 + i as f64 * 0.004_97, (i % 4) as u8));
        }
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        v
    }

    #[test]
    fn a_pulser_train_is_removed_and_the_background_is_kept() {
        // 1000 发、每 7 发漏一发：两端各一发只有一侧邻居，留下；其余全删，本底 200 个一个不动
        let mut events = on_detectors(&pulser_train(100.0, 1000, 7));
        let n0 = events.len();
        let removed = remove_pulser(&mut events);
        let pulses = (0..1000).filter(|i| i % 7 != 3).count();
        assert_eq!(n0, pulses * 4 + 200);
        assert_eq!(removed, (pulses - 2) * 4);
        assert_eq!(events.len(), 200 + 8);
    }

    #[test]
    fn a_three_detector_pulse_is_removed_too() {
        // 一路正处在死时间：那一发只剩三路同戳，照样按梳齿认出来
        let mut v = pulser_train(100.0, 10, 0);
        let t5 = ((100.0 / TICK_S).round() + (5.0 * 4194.7672_f64).round()) * TICK_S;
        v.retain(|x| !(x.0 == t5 && x.1 == 2));
        let mut events = on_detectors(&v);
        assert_eq!(remove_pulser(&mut events), 8 * 4 - 1);
    }

    #[test]
    fn scattered_particles_are_not_removed() {
        // 30 次四路穿越、间隔不规则：簇有，但不在梳齿上
        let mut v = Vec::new();
        for i in 0..30 {
            let t = 100.0 + 0.0317 * i as f64 + 0.0011 * ((i * i) % 7) as f64;
            let t = (t / TICK_S).round() * TICK_S;
            v.extend((0..4).map(|d| (t, d as u8)));
        }
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut events = on_detectors(&v);
        assert_eq!(remove_pulser(&mut events), 0);
    }

    #[test]
    fn a_particle_one_period_after_another_is_not_removed() {
        // 两次穿越恰好相隔一个周期：只有一侧邻居，不删
        let t0 = (100.0 / TICK_S).round();
        let mut v: Vec<(f64, u8)> = (0..4).map(|d| (t0 * TICK_S, d as u8)).collect();
        v.extend((0..4).map(|d| ((t0 + 4195.0) * TICK_S, d as u8)));
        let mut events = on_detectors(&v);
        assert_eq!(remove_pulser(&mut events), 0);
    }

    #[test]
    fn a_burst_is_never_touched() {
        // 20 个光子铺在 80 µs 里，没有三路同戳
        let v: Vec<(f64, u8)> = (0..20)
            .map(|i| {
                (
                    ((100.0 + i as f64 * 4e-6) / TICK_S).round() * TICK_S,
                    (i % 4) as u8,
                )
            })
            .collect();
        let mut events = on_detectors(&v);
        assert_eq!(remove_pulser(&mut events), 0);
        assert_eq!(events.len(), 20);
    }

    #[test]
    fn one_crossing_that_fills_the_candidate_leaves_too_few_counts() {
        // 窗里一次四路同戳 + 5 个零散计数：9 个够候选门，扣掉簇只剩 5
        let v = with_crossings(300e-6, &[150e-6]);
        let events = on_detectors(&v[..]);
        let mut short: Vec<(f64, u8)> =
            v.iter().copied().filter(|x| x.0 < 100.0 + 150e-6).collect();
        short.extend((0..4).map(|d| (100.0 + 150e-6, d as u8)));
        assert_eq!(residual_counts(&events), 8);
        assert_eq!(residual_counts(&on_detectors(&short)), short.len() - 4);
        assert!(residual_counts(&on_detectors(&short)) < MIN_COUNTS as usize);
    }

    #[test]
    fn a_bright_burst_with_a_stray_cluster_keeps_enough_counts() {
        // 20 个光子铺开 + 偶然混进一个三重：剩 20，远够
        let mut v: Vec<(f64, u8)> = (0..20)
            .map(|i| (100.0 + i as f64 * 5e-6, (i % 4) as u8))
            .collect();
        v.extend((0..3).map(|d| (100.00003, d as u8)));
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        assert!(residual_counts(&on_detectors(&v)) >= MIN_COUNTS as usize);
    }

    #[test]
    fn shared_frame_satellites_skip_the_particle_veto() {
        // 共帧星上同戳是读出结构的必然产物，不是物理——这道门对它们整个跳过
        assert!(!Sat03B::SHARED_FRAME);
        assert!(crate::types::Sat02::SHARED_FRAME);
        assert!(crate::types::Sat04::SHARED_FRAME);
        assert!(crate::types::Sat07::SHARED_FRAME);
    }

    fn gap(times: &[f64]) -> bool {
        has_dead_gap(&at(times), 100.0, 101.0, -1.0, -1.0)
    }

    /// 可复现的泊松本底：splitmix64 生成均匀数，指数间隔。（xorshift 在小种子上头几个
    /// 输出很差，会造出一个特别长的首间隔、被当成窗口开头的空白。）
    fn poisson_stream(rate: f64, from: f64, to: f64, seed: u64) -> Vec<f64> {
        let mut x = seed;
        let mut t = from;
        let mut out = Vec::new();
        loop {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let u = ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            t += -u.ln() / rate;
            if t > to {
                return out;
            }
            out.push(t);
        }
    }

    #[test]
    fn a_uniform_stream_has_no_dead_gap() {
        let times: Vec<f64> = (0..2000).map(|i| 100.0 + i as f64 * 5e-4).collect();
        assert!(!gap(&times));
    }

    #[test]
    fn a_frame_gap_at_high_rate_is_a_dead_gap() {
        // 帧结构：每 10 ms 前 5 ms 有 75 kc/s 的事例，后 5 ms 全空
        let mut times = Vec::new();
        for frame in 0..100 {
            let t0 = 100.0 + frame as f64 * 0.01;
            times.extend((0..375).map(|i| t0 + i as f64 * 5e-3 / 375.0));
        }
        assert!(gap(&times));
    }

    #[test]
    fn a_long_but_poisson_plausible_gap_at_low_rate_is_not_a_dead_gap() {
        // 300 c/s，挖掉 20 ms：一个 1 s 窗里 300 个间隔，最长的到 20 ms 并不稀奇
        let mut times: Vec<f64> = (0..300).map(|i| 100.0 + i as f64 / 300.0).collect();
        times.retain(|t| !(100.50..100.52).contains(t));
        assert!(!gap(&times));
    }

    #[test]
    fn a_lost_packet_on_all_four_detectors_is_a_dead_gap() {
        // 400 c/s 的本底里四路同时丢了 300 ms（03B 四路同时丢包的典型长度）
        let mut times = poisson_stream(400.0, 100.0, 101.0, 7);
        times.retain(|t| !(100.40..100.70).contains(t));
        assert!(gap(&times));
    }

    #[test]
    fn the_window_edges_count_as_gaps() {
        // 窗口前 200 ms 一个事例都没有，之后 5 kc/s
        let times: Vec<f64> = (0..4000).map(|i| 100.2 + i as f64 * 2e-4).collect();
        assert!(gap(&times));
    }

    #[test]
    fn an_isolated_frame_is_caught_without_the_pass_rate() {
        // 成片丢数里的孤帧：窗里只有 8 个事例挤在 1 ms 内。旧判据要靠整次过境率才抓得到；
        // 按间隔中位数估的计数率是帧内的 7 kc/s，半秒空白一看就不是涨落
        let times: Vec<f64> = (0..8).map(|i| 100.5 + i as f64 * 1e-4).collect();
        assert!(gap(&times));
    }

    #[test]
    fn an_isolated_frame_covered_by_the_candidate_is_caught() {
        // GRID-07 2024-03-15T03:54:57.233：窗里一整帧（这里 88 个时刻、每个时刻四路共戳），
        // 候选窗盖住了整帧。排除候选窗的间隔会只剩跨空白的长间隔、把计数率估成几 c/s
        let mut v = Vec::new();
        for i in 0..88 {
            let t = 100.5 + i as f64 * 8e-5;
            v.extend((0..4).map(|d| (t, d as u8)));
        }
        let events = on_detectors(&v);
        assert!(has_dead_gap(&events, 100.0, 101.0, 100.4999, 100.5071));
    }

    #[test]
    fn shared_timestamps_do_not_make_the_rate_zero() {
        // 共帧读出在高计数率下一半以上的相邻事例同戳：按事例算间隔中位数为 0。
        // 按不同时戳算：每 10 ms 前 5 ms 每 30 µs 一帧（四路共戳），后 5 ms 全空
        let mut v = Vec::new();
        for frame in 0..100 {
            let t0 = 100.0 + frame as f64 * 0.01;
            for i in 0..166 {
                let t = t0 + i as f64 * 3e-5;
                v.extend((0..4).map(|d| (t, d as u8)));
            }
        }
        assert!(has_dead_gap(&on_detectors(&v), 100.0, 101.0, -1.0, -1.0));
    }

    #[test]
    fn a_passing_radiation_belt_does_not_raise_the_rate() {
        // 旧判据的毛病：过境平均率被辐射带抬到 1134 c/s 时，387 c/s 的正常本底有四成的窗
        // 被砍。新判据只看窗内，不存在这个输入
        let mut fired = 0;
        for seed in 1..=400u64 {
            let times = poisson_stream(387.0, 100.0, 101.0, seed * 7919);
            if gap(&times) {
                fired += 1;
            }
        }
        // 设计误砍 α = 1e-3；间隔中位数有抖动，400 个窗里允许到 4 个
        assert!(fired <= 4, "fired {fired}/400");
    }

    #[test]
    fn the_burst_itself_does_not_raise_the_rate() {
        // 200 c/s 本底里一个 20 计数 / 80 µs 的暴。暴内间隔计入中位数会把 r 抬高、
        // 让普通涨落更容易被判成空洞；排除候选窗后结果与没有暴时一致
        let mut times = poisson_stream(200.0, 100.0, 101.0, 3);
        let burst: Vec<f64> = (0..20).map(|i| 100.5 + i as f64 * 4e-6).collect();
        times.extend(&burst);
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let events = at(&times);
        let with = has_dead_gap(&events, 100.0, 101.0, 100.5, 100.5 + 80e-6);
        let without = has_dead_gap(
            &at(&poisson_stream(200.0, 100.0, 101.0, 3)),
            100.0,
            101.0,
            -1.0,
            -1.0,
        );
        assert_eq!(with, without);
    }

    #[test]
    fn a_millisecond_gap_at_belt_rates_is_a_normal_fluctuation() {
        // 13 kc/s 均匀流里 0.9 ms 的空段：1 s 里 13000 个间隔，最长的到 0.9 ms 统计上并不稀奇。
        // 旧判据会砍它。0.9 ms 的空段在这里本来就是正常涨落
        let mut times: Vec<f64> = (0..13000).map(|i| 100.0 + i as f64 / 13000.0).collect();
        times.retain(|t| !(100.5000..100.5009).contains(t));
        assert!(!gap(&times));
    }
}
