use blink_core::traits::Instrument;
use blink_core::types::{TemporalState, UnifiedSignal};
use blink_lightning::{algorithms::coincidence_prob, database::coverage, database::get_lightnings};
use blink_load::load_all;
use chrono::TimeDelta;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use uom::si::f64::*;

/// 列车窗口半宽：±600 s 覆盖一次辐射带沉降区穿越（REP 微暴列车持续几分钟
/// 到几十分钟）；单个雷暴单体在星下可见仅 1–2 分钟，又不至于把同一轨道
/// 前后两个独立雷暴缝在一起。
pub const TRAIN_HALF_WINDOW_US: i64 = 600 * 1_000_000;

/// 列车判定的向后兼容地板（已冻结）：WWLLN 关联认证 TGF（2547 个）的
/// neighbors_10min 分布 99 分位。阈值只由 TGF 样本单方面定标，REP 样本不参与；
/// 认证 REP 人群从 ~60 起步（中位 656），阈值坐在两人群之间的空沟里，取 30
/// 还是 50 结果几乎不变。原计划的 ACD 逐事例符合交叉验证已做且判据不成立——
/// HE 里的 REP 信号是沉降电子的轫致辐射光子（NaI 截止层几乎无信号，ACD 符合
/// 超出 <1%），与 TGF 伽马对 ACD 同盲——故本阈值以 TGF 侧定标独立成立。
///
/// 这个 34 现在是**地板**而不是唯一的阈：见 `train_threshold`。地板存在的
/// 理由是历史的——不动已冻结的 HXMT v6 结果与 SVOM v8 结果，不是什么物理
/// 尺度。SVOM 池的 neighbors_10min 中位是 0，`RATIO * 0 = 0`，全靠这个地板兜着。
pub const TRAIN_THRESHOLD: u32 = 34;

/// 阈值相对池尺度的倍率，**由 HXMT 的冻结阈反解得来**：HXMT v6 全池
/// （1,709,490 个候选）的 neighbors_10min 中位是 9，冻结阈 34，34/9 = 3.78。
/// 它的含义只有一句——"保持 HXMT 上那一档有效严格度，按各池自己的尺度输出
/// 到别的仪器"。不是物理常数，别当物理常数用。
const RATIO: f64 = 34.0 / 9.0;

/// 池尺度与理论期望不一致的报警倍数，见 `train_threshold`。
const SCALE_SANITY_FACTOR: f64 = 2.0;

/// 逐池定阈：`max(TRAIN_THRESHOLD, round(RATIO * median(neighbors_10min)))`。
///
/// 为什么不能用绝对阈：`neighbors_10min` 在均匀池下的期望就是 `R · 2W`
/// （池率 × 1200 s），所以它随池率线性变，绝对阈是量纲错误。实测四个池的
/// 中位分别是 HXMT v6 = 9、SVOM v8 = 0、Fermi/GBM 2014 = 5、
/// Fermi/GBM 2019 = 34：GBM 2019 的池率 0.0290 /s 给出期望 34.8，而绝对阈
/// 正好是 34，于是阈值不偏不倚坐在本底中位数上、把整池的 49.97% 标成列车。
/// 换成本函数后 GBM 2019 得到 128、标记 1.32%，另外三个池的阈仍是 34、
/// 标记逐位不变（见 `blink_fermi_gbm/evidence/train_threshold/`）。
///
/// 这个形式只会把阈往上抬、不会往下降，所以已冻结的结果不会动。
///
/// 中位数是稳健估计，但它的前提是**列车在池里是少数**；真遇到一个被列车主导
/// 的池，中位本身就被污染、阈值会跟着抬上去正好放过它们。所以并排算一个与
/// 中位互不相关的理论期望 `R · 2W`（`R` 由相邻候选间隔的中位反解，
/// `R = ln2 / median(Δt)`，列车只压低少数间隔、压不动中位），两者差过
/// `SCALE_SANITY_FACTOR` 倍就报警。这是判据不是数据完整性，所以只打警告、
/// 既不静默降级也不 panic。
/// 开几条工作线程。
///
/// **不能直接用 `available_parallelism()`**：它认 `sched_getaffinity` 与 cgroup
/// 配额，**不认 `OMP_NUM_THREADS`**，农场作业里实测两者差 24 倍——`OMP_NUM_THREADS=1`
/// 让 `nproc` 返回 1，而 affinity 仍是 24。而每条线程要开一条 WWLLN 连接、
/// 各吃 2 GiB 虚拟地址空间（`RLIMIT_AS` 数的是虚拟地址空间，同一个文件映射
/// N 次就占 N 份），于是一个**看起来单核**的作业会起 25 × 2 GiB = 50 GiB
/// 地址空间，然后被无声杀掉——`.err` 是空的、没有任何报错。
///
/// 所以两道闸，**但两道挡的不是同一件事，别指望后一道能替前一道**：
///
/// * `BLINK_THREADS` 显式指定时一律听它的。**控制线程数不能靠 `taskset`
///   也不能靠 `nproc`**，只能靠程序自己的参数，这就是那个参数。**农场上那个
///   无声被杀的场景只能靠这一道**——见下。
/// * 没有指定时，按 `RLIMIT_AS` 还能放下几条连接封顶。不设 `RLIMIT_AS` 的
///   环境（开发机）维持原行为。
///
/// **第二道在农场上基本不触发，别误以为它兜住了。** `hep_sub -mem 20000`
/// **并不压地址空间**，那一档下 `ulimit -v` 仍是 95 GB（实测，见
/// `blink_lightning::database::mmap_budget` 的注释）：95 GB 放得下 32 条连接，
/// 而 affinity 给的是 24，**闸不会合上**。那里杀掉作业的是 RSS 超 `-mem`，
/// 不是地址空间。第二道只在有人真的 `ulimit -v` 设紧时才管用。
///
/// 线程数不影响结果：每条线程带原下标收回、最后按下标排序，顺序与串行一致。
fn worker_threads() -> usize {
    let detected = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    if let Some(requested) = std::env::var("BLINK_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
    {
        return requested;
    }
    match blink_lightning::database::max_connections() {
        Some(cap) if cap < detected => {
            eprintln!(
                "filter: 地址空间只放得下 {cap} 条 WWLLN 连接，线程数从 {detected} 压到 {cap}\
                 （要自己指定用 BLINK_THREADS）"
            );
            cap
        }
        _ => detected,
    }
}

fn train_threshold(times_us: &[i64], neighbors: &[u32]) -> u32 {
    if neighbors.is_empty() {
        return TRAIN_THRESHOLD;
    }
    let median_neighbors = median_u32(neighbors);
    let scaled = (RATIO * median_neighbors as f64).round() as u32;
    let threshold = TRAIN_THRESHOLD.max(scaled);

    // 与中位互不相关的第二把尺子：由间隔中位反解池率，再乘窗长。
    let window_seconds = 2.0 * (TRAIN_HALF_WINDOW_US as f64 / 1e6);
    if let Some(rate) = pool_rate_hz(times_us) {
        let expected = rate * window_seconds;
        let observed = median_neighbors as f64;
        // 两边都可能是 0（低率池），比值只在都非零时才有意义
        if expected > 0.0 && observed > 0.0 {
            let ratio = (observed / expected).max(expected / observed);
            if ratio > SCALE_SANITY_FACTOR {
                eprintln!(
                    "filter: WARNING pool scale disagrees — neighbors_10min median {observed:.0} \
                     vs expected R·2W = {expected:.1} (rate {rate:.5} /s from median gap); \
                     ratio {ratio:.1}× > {SCALE_SANITY_FACTOR}×. \
                     The median may be train-contaminated and the threshold too lax."
                );
            }
        }
        eprintln!(
            "filter: pool scale — neighbors_10min median {median_neighbors}, expected R·2W {expected:.1}, \
             threshold {threshold}"
        );
    } else {
        eprintln!(
            "filter: pool scale — neighbors_10min median {median_neighbors}, expected R·2W unavailable, \
             threshold {threshold}"
        );
    }
    threshold
}

/// 计数的中位数。偶数个时取偏下的那个，跟阈值一样是整数计数，不做插值。
fn median_u32(values: &[u32]) -> u32 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[(sorted.len() - 1) / 2]
}

/// 由相邻候选时间间隔的中位反解池率：均匀泊松下间隔中位是 `ln2 / R`。
/// 用中位而不是 `N / 跨度`，是因为池里有没搜的天、跨度不等于活时间；
/// 也因为列车只压低少数间隔、压不动中位。实测对照已知活时间，
/// GBM 2014 偏 +2.3%、2019 偏 +7.2%。
fn pool_rate_hz(times_us: &[i64]) -> Option<f64> {
    if times_us.len() < 2 {
        return None;
    }
    let mut sorted = times_us.to_vec();
    sorted.sort_unstable();
    let mut gaps: Vec<i64> = sorted
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&g| g > 0)
        .collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_unstable();
    let median_gap_us = gaps[(gaps.len() - 1) / 2] as f64;
    Some(std::f64::consts::LN_2 / (median_gap_us / 1e6))
}

#[derive(Serialize, Deserialize)]
pub struct LightningInfo {
    pub associated: bool,
    /// 偶合概率。覆盖外为 `None`——那里既没查也算不出，不是 0 也不是 1。
    pub coincidence_probability: Option<f64>,
    /// 候选时刻是否落在 WWLLN 库的覆盖区间内。
    ///
    /// 库止于 2024-12-31，而 SVOM 的候选到 2026-08——覆盖外的候选查出来必然
    /// 是空，`associated` 于是恒为 false。那是"查不到"，不是"没有闪电"，
    /// 下游不能把两者当成一回事。旧的 tgfs.json 没有这一列，读回时按 true
    /// 处理：HXMT 的搜索范围整个落在覆盖内。
    #[serde(default = "covered_by_default")]
    pub in_coverage: bool,
}

fn covered_by_default() -> bool {
    true
}

#[derive(Serialize, Deserialize)]
pub struct TrainInfo {
    /// 全初始候选池（fa < 20 /yr，含亚阈）中，start 落在本候选 start 的
    /// 半开窗 [t−600s, t+600s) 内的其他候选数
    pub neighbors_10min: u32,
    /// `neighbors_10min > train_threshold(池)`：列车成员，目录阶段池级摘除。
    ///
    /// 阈不再是常数 34，而是按池尺度缩放，见 `train_threshold`。
    ///
    /// 摘掉的是什么，随仪器而变，不要一律当成 REP：HXMT 上这一刀摘的是
    /// REP 微暴列车（认证 REP 人群 neighbors 中位 656），而 Fermi 轨道倾角
    /// 只有 25.6°、候选纬度最大到 |lat| ≈ 26°，**根本够不到外辐射带足点**，
    /// GBM 上被标出来的那批（标记者 |lat| 中位 18.9°、|lat| > 45° 占 0.0%）
    /// 因此不是 REP。它们是什么**尚未定性**。
    pub is_train: bool,
}

/// tgfs.json 的单条记录。`blink wwlln` 写出（富集不筛选），
/// `blink catalog` 读回做池级清洁与判选。
#[derive(Serialize, Deserialize)]
pub struct Tgf {
    pub signal: UnifiedSignal,
    pub lightning: LightningInfo,
    pub train: TrainInfo,
}

/// 逐候选数出半开窗 [t−600s, t+600s) 内的其他时刻数（排除自身）。
/// 与原型 np.searchsorted 的左侧语义逐点一致。
fn neighbor_counts_us(times_us: &[i64]) -> Vec<u32> {
    let mut sorted = times_us.to_vec();
    sorted.sort_unstable();
    times_us
        .iter()
        .map(|&t| {
            let lo = sorted.partition_point(|&x| x < t - TRAIN_HALF_WINDOW_US);
            let hi = sorted.partition_point(|&x| x < t + TRAIN_HALF_WINDOW_US);
            (hi - lo - 1) as u32
        })
        .collect()
}

/// 列车密度特征。数的是**全量初始候选池**而非仅显著候选：REP 列车过境会
/// 点亮上千个亚阈候选（2025-09-30 单趟 ±10 min 内 4491 个），宽池里列车
/// 藏不住；TGF 一次过境下方只有一个雷暴系统，连亚阈算上也只有几个到
/// 二十几个（中位 8）。
fn train_neighbor_counts(signals: &[UnifiedSignal]) -> (Vec<i64>, Vec<u32>) {
    let times: Vec<i64> = signals.iter().map(|s| s.start.timestamp_micros()).collect();
    let counts = neighbor_counts_us(&times);
    (times, counts)
}

/// 对单个候选做 WWLLN 闪电关联 + 虚警概率。每次调用的两个 `get_lightnings`
/// 查询走线程本地只读连接（见 blink_lightning::database），可安全并行。
fn associate(
    signal: &UnifiedSignal,
    neighbors_10min: u32,
    train_threshold: u32,
    coverage: (chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>),
    window: TimeDelta,
) -> Tgf {
    let peak_time = signal.peak_time();
    let in_coverage = peak_time >= coverage.0 && peak_time <= coverage.1;
    if !in_coverage {
        // 覆盖外就不查了：查也是空，白付两次百万级检索的代价。
        return Tgf {
            signal: signal.clone(),
            lightning: LightningInfo {
                associated: false,
                coincidence_probability: None,
                in_coverage: false,
            },
            train: TrainInfo {
                neighbors_10min,
                is_train: neighbors_10min > train_threshold,
            },
        };
    }
    let position = TemporalState {
        timestamp: peak_time,
        state: signal.position.clone(),
    };
    let lightnings = get_lightnings(
        peak_time - TimeDelta::seconds(1),
        peak_time + TimeDelta::seconds(1),
    )
    .into_iter()
    .filter(|lightning| {
        lightning.is_associated(
            &position,
            window,
            Length::new::<uom::si::length::kilometer>(ASSOCIATION_KILOMETERS),
        )
    })
    .collect::<Vec<_>>();

    Tgf {
        signal: signal.clone(),
        lightning: LightningInfo {
            associated: !lightnings.is_empty(),
            in_coverage: true,
            coincidence_probability: Some(coincidence_prob(
                &position,
                window,
                Length::new::<uom::si::length::kilometer>(ASSOCIATION_KILOMETERS),
                TimeDelta::minutes(2),
            )),
        },
        train: TrainInfo {
            neighbors_10min,
            is_train: neighbors_10min > train_threshold,
        },
    }
}

/// 关联窗口。±5 ms 与 800 km 是 HXMT 上定标的：800 km 约合 550 km 轨道高度
/// 下的可见地平，5 ms 覆盖光行时差与两边的计时不确定度。SVOM 轨道高约
/// 625 km、GBM 约 535 km，几何上同量级，先沿用同一组值；哪颗星要单独定标，
/// 得先有它自己的认证样本。
/// 默认的时间窗半宽；`run` 可按需放宽（如电子束的足点检验）。
pub const ASSOCIATION_MILLISECONDS: i64 = 5;
const ASSOCIATION_KILOMETERS: f64 = 800.0;

pub fn run<I: Instrument>(window_ms: i64) {
    let window = TimeDelta::milliseconds(window_ms);
    if window_ms != ASSOCIATION_MILLISECONDS {
        eprintln!(
            "filter: association window ±{window_ms} ms (default {ASSOCIATION_MILLISECONDS})"
        );
    }
    let signals = load_all::<I>();
    let total = signals.len();
    eprintln!("filter: {} candidates to associate ({})", total, I::name());

    // 列车密度在全量池上一次算完（排序 + 二分，O(n log n)），并行阶段按
    // 下标取用即可。
    let coverage = coverage();
    eprintln!(
        "filter: WWLLN coverage {} .. {}",
        coverage.0.format("%Y-%m-%d"),
        coverage.1.format("%Y-%m-%d")
    );
    let n_outside = signals
        .iter()
        .filter(|s| s.peak_time() < coverage.0 || s.peak_time() > coverage.1)
        .count();
    if n_outside > 0 {
        eprintln!(
            "filter: {n_outside}/{total} candidates fall outside it and are left unqueried \
             (in_coverage = false, not `no lightning`)"
        );
    }

    let (times_us, neighbor_counts) = train_neighbor_counts(&signals);
    let threshold = train_threshold(&times_us, &neighbor_counts);
    let n_train = neighbor_counts.iter().filter(|&&n| n > threshold).count();
    eprintln!("filter: {n_train}/{total} flagged as train members (neighbors > {threshold})");

    // 每候选做 2 次 WWLLN 查询（±1s 关联 + ±62s 虚警概率），成本随该时段闪电
    // 密度差几十倍（活跃季 ±62s 窗返回上万条闪电）。静态分块会严重失衡（空段线程
    // 早退、忙段线程拖尾），故用原子取号做工作窃取：每线程反复领下一个待处理下标，
    // 忙闲自动均衡，56 核吃满到最后。结果带原下标收回后排序，保持原顺序。
    let n_threads = worker_threads();
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let signals_ref = &signals;
    let counts_ref = &neighbor_counts;

    let mut collected: Vec<(usize, Tgf)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n_threads)
            .map(|_| {
                let next = &next;
                let done = &done;
                scope.spawn(move || {
                    let mut local: Vec<(usize, Tgf)> = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= total {
                            break;
                        }
                        local.push((
                            i,
                            associate(&signals_ref[i], counts_ref[i], threshold, coverage, window),
                        ));
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if n % 100_000 == 0 {
                            eprintln!("filter: {n}/{total}");
                        }
                    }
                    local
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });

    collected.sort_by_key(|(i, _)| *i);
    let tgfs: Vec<Tgf> = collected.into_iter().map(|(_, tgf)| tgf).collect();

    eprintln!("filter: {total}/{total} associated, writing tgfs.json");
    let json = serde_json::to_string_pretty(&tgfs).expect("failed to serialize to json");
    // 原子写：先写临时文件再 rename，避免下游（pipeline 的 cp / git）读到半截 json。
    let tmp = format!("tgfs.json.{}.tmp", nanoid::nanoid!(6));
    std::fs::write(&tmp, json).expect("failed to write tgfs.json tmp");
    std::fs::rename(&tmp, "tgfs.json").expect("failed to rename tgfs.json");
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: i64 = 1_000_000; // 1 s in µs

    #[test]
    fn isolated_candidates_count_zero() {
        assert_eq!(neighbor_counts_us(&[]), Vec::<u32>::new());
        assert_eq!(neighbor_counts_us(&[42 * S]), vec![0]);
        // 相距远超 ±600s 的两个候选互不计数
        assert_eq!(neighbor_counts_us(&[0, 2000 * S]), vec![0, 0]);
    }

    #[test]
    fn train_members_count_each_other() {
        // 1 s 内的三个候选各自数到另外两个，自身不计
        let times = [0, S / 2, S];
        assert_eq!(neighbor_counts_us(&times), vec![2, 2, 2]);
        // 输入乱序不影响结果（计数走排序副本）
        let times = [S, 0, S / 2];
        assert_eq!(neighbor_counts_us(&times), vec![2, 2, 2]);
    }

    #[test]
    fn window_is_half_open_like_searchsorted() {
        // [t−600s, t+600s)：右端开、左端闭，与原型 np.searchsorted 逐点一致。
        // 对 t=0，+600s 处的邻居落在窗外；对 t=600s，0 处的邻居落在窗内。
        let times = [0, 600 * S];
        assert_eq!(neighbor_counts_us(&times), vec![0, 1]);
    }

    #[test]
    fn duplicated_timestamps_are_neighbors_not_self() {
        // 同一时刻的多条候选互为邻居，但都不数自己
        let times = [7 * S, 7 * S, 7 * S];
        assert_eq!(neighbor_counts_us(&times), vec![2, 2, 2]);
    }

    /// 池率 `rate` 的均匀候选流，间隔取常数（间隔中位因此精确）。
    fn uniform_pool(n: usize, rate_hz: f64) -> Vec<i64> {
        let step = (1e6 / rate_hz) as i64;
        (0..n as i64).map(|i| i * step).collect()
    }

    #[test]
    fn the_median_is_the_lower_one_for_an_even_count() {
        // 阈值是整数计数，偶数个时取偏下的那个而不是插值
        assert_eq!(median_u32(&[1, 2, 3, 4]), 2);
        assert_eq!(median_u32(&[1, 2, 3]), 2);
        assert_eq!(median_u32(&[7]), 7);
    }

    #[test]
    fn the_pool_rate_comes_back_from_the_median_gap() {
        // 常数间隔 100 ms -> 10 /s；间隔中位反解 ln2/0.1 = 6.93 /s。
        // 常数间隔不是泊松，所以差一个 ln2 因子是意料之中的——这里只钉住
        // 反解本身没算错，实际池上的偏差（GBM 实测 +2.3% / +7.2%）见 evidence/。
        let times = uniform_pool(1000, 10.0);
        let rate = pool_rate_hz(&times).expect("有间隔就有估计");
        assert!(
            (rate - std::f64::consts::LN_2 * 10.0).abs() < 1e-6,
            "{rate}"
        );
    }

    #[test]
    fn a_pool_with_no_gaps_has_no_rate() {
        assert!(pool_rate_hz(&[]).is_none());
        assert!(pool_rate_hz(&[42 * S]).is_none());
        // 全部同一时刻：间隔全是 0，反解不出率，不能除零
        assert!(pool_rate_hz(&[5 * S, 5 * S, 5 * S]).is_none());
    }

    #[test]
    fn a_low_rate_pool_keeps_the_frozen_floor() {
        // SVOM 那种池：候选稀疏，邻居中位 0，RATIO*0 = 0，靠地板兜住。
        // 每 1 小时一个候选，±600 s 内谁也够不着谁。
        let times = uniform_pool(200, 1.0 / 3600.0);
        let counts = neighbor_counts_us(&times);
        assert_eq!(median_u32(&counts), 0);
        assert_eq!(train_threshold(&times, &counts), TRAIN_THRESHOLD);
    }

    #[test]
    fn a_high_rate_pool_scales_the_threshold_up() {
        // GBM 2019 那种池：邻居中位与绝对阈同量级时，绝对阈会砍掉半个池。
        // 缩放后的阈必须显著高于中位，且严格大于地板。
        let times = uniform_pool(20_000, 0.03);
        let counts = neighbor_counts_us(&times);
        let median = median_u32(&counts);
        assert!(
            median > TRAIN_THRESHOLD / 2,
            "本例要的就是中位与地板同量级: {median}"
        );
        let threshold = train_threshold(&times, &counts);
        assert!(threshold > TRAIN_THRESHOLD, "高率池必须抬阈: {threshold}");
        assert_eq!(threshold, (RATIO * median as f64).round() as u32);
        // 均匀池里一个都不该被标成列车
        assert_eq!(counts.iter().filter(|&&n| n > threshold).count(), 0);
    }

    #[test]
    fn the_threshold_never_drops_below_the_frozen_floor() {
        // 「只抬不降」是这个形式的全部安全性来源：已冻结的结果不会被改动。
        for rate in [1.0 / 86400.0, 1.0 / 600.0, 0.004, 0.03, 0.2] {
            let times = uniform_pool(5_000, rate);
            let counts = neighbor_counts_us(&times);
            assert!(
                train_threshold(&times, &counts) >= TRAIN_THRESHOLD,
                "rate {rate}"
            );
        }
    }

    #[test]
    fn an_empty_pool_falls_back_to_the_floor() {
        assert_eq!(train_threshold(&[], &[]), TRAIN_THRESHOLD);
    }

    /// 邻居数中位恰好是 `median` 的一个向量。取值本身与真池不同，
    /// 定阈只看中位，所以够用。
    fn pool_with_median(median: u32) -> Vec<u32> {
        let mut v = vec![0u32; 101];
        for (i, slot) in v.iter_mut().enumerate() {
            *slot = if i <= 50 {
                median.saturating_sub(1)
            } else {
                median + 1
            };
        }
        v[50] = median;
        v
    }

    #[test]
    fn the_four_real_pools_map_to_the_measured_thresholds() {
        // 四个真池实测的 neighbors_10min 中位 -> 阈值。三个冻结池必须原地不动，
        // 只有 GBM 2019 抬起来。数字来源见
        // `blink_fermi_gbm/evidence/train_threshold/`。
        for (pool, median, expected) in [
            ("HXMT v6", 9u32, 34u32),
            ("SVOM v8", 0, 34),
            ("Fermi/GBM 2014", 5, 34),
            ("Fermi/GBM 2019", 34, 128),
            ("Fermi/GBM 2014+2019 合池", 33, 125),
        ] {
            let counts = pool_with_median(median);
            assert_eq!(median_u32(&counts), median, "{pool} 的构造中位不对");
            let got = train_threshold(&[], &counts);
            assert_eq!(
                got, expected,
                "{pool}: 中位 {median} 应给出阈 {expected}，得到 {got}"
            );
        }
    }

    #[test]
    fn the_hxmt_median_lands_exactly_on_the_frozen_threshold() {
        // RATIO 是 34/9 反解出来的，所以 HXMT 的中位 9 必须精确回到 34——
        // 浮点上偏一点点四舍五入成 35，整个 v6 目录就动了。
        assert_eq!((RATIO * 9.0).round() as u32, TRAIN_THRESHOLD);
    }
}
