use blink_core::traits::Instrument;
use blink_core::types::{TemporalState, UnifiedSignal};
use blink_lightning::{algorithms::coincidence_prob, database::coverage, database::get_lightnings};
use blink_load::load_all;
use chrono::TimeDelta;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use uom::si::f64::*;

pub use blink_algorithms::train::{TRAIN_HALF_WINDOW_US, TRAIN_THRESHOLD};
use blink_algorithms::train::{train_neighbor_counts, train_threshold};

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
/// **第二道在农场上基本不触发，别误以为它兜住了。** `hep_sub -mem` **并不压
/// 地址空间**：作业内实测 `ulimit -v` 是 **100 GB 的固定值、不随 `-mem` 变**，
/// 而 cgroup 的 `memory.max` / `memory.high` / `cpu.max` 在作业内**一个都读不到**。
/// 100 GB 放得下 34 条连接，而 affinity 给的是 24 到 64，**24 核那档闸不会
/// 合上**（64 核那档会，但那是碰巧不是设计）。
/// 第二道只在有人真的把 `ulimit -v` 设紧时才管用。
///
/// **农场上作业被杀的机制目前未定，这里只记观测、不记机制**（本仓库的规矩：
/// 唯象的东西不补物理机制 narrative）：
///
/// * 见过**无声被杀**——`.err` 是空的、没有任何报错；
/// * **`-mem` 是不是强制上限，两个方向的证据都有**：GECAM-B 在 `-g hxmt` 上
///   三路只差 `-mem` 的夹逼实验里 16000 与 20000 两路**被 held**
///   （`MemoryUsage` 15965 / 20053）、22000 那路跑完；SVOM 在 `-mem 8000`
///   的边界上**差 0.5% 被 held**；而 GBM 把 `-mem` 从 1 扫到 4000 九个值
///   **全部跑完、一个都没被拦**，`-mem 1`（申请 1 MB）实用 490 MiB 照跑。
///
/// **所以不能靠"没被 hold"推断没超，也不能因为别人没被 hold 就放心超额申请。**
/// 无论机制是什么，**线程数失控都会把内存占用抬上去**，而这里唯一能确定拦住
/// 它的就是 `BLINK_THREADS`。
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
