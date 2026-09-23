use super::rec_sci_data::reconstruct_with_wrap_tracking;
use crate::io::level_1b::SciFile;

/// 饱和类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaturationType {
    /// 整包丢失：FIFOAFullReset 触发，整个 FIFO A 被清空。
    /// 表现为重建时间序列中出现远大于正常包间隔的空洞。
    FifoReset,
}

/// 单个饱和区间
#[derive(Debug, Clone)]
pub struct SaturationInterval {
    /// 空洞起始 MET（前一个包的最晚事件时间）
    pub start_met: f64,
    /// 空洞结束 MET（后一个包的最早事件时间）
    pub stop_met: f64,
    /// 空洞持续时间（秒）
    pub gap_seconds: f64,
    /// 空洞前一个包的索引（排序后）
    pub prev_pkt_idx: usize,
    /// 空洞后一个包的索引（排序后）
    pub next_pkt_idx: usize,
    /// 饱和类型
    pub saturation_type: SaturationType,
}

/// 每个 CCSDS 包的时间摘要
struct PacketTimeSummary {
    /// 原始包索引（在 SciFile.ccsds 中的位置）
    pkt_idx: usize,
    /// 包内最早事件的重建 MET
    min_met: f64,
    /// 包内最晚事件的重建 MET
    max_met: f64,
    /// 包内有效事件数
    n_events: usize,
    /// 包头 `RATE_WINDOW_EVENTS` 个事例的率 (events/s)
    head_rate: Option<f64>,
    /// 包尾 `RATE_WINDOW_EVENTS` 个事例的率 (events/s)
    tail_rate: Option<f64>,
}

/// 稳定率估不出来时的回退倍率：`edge_rate × gap` 要超过它才算。
///
/// 这是 `MIN_POISSON_EXPONENT` 的同一个测试，但除的是 16 事例的边缘率（σ≈26%
/// 而不是 1.4%），所以门槛必须留得高得多。只在 `stable_event_rate` 返回
/// `None`（率窗口里一个包都没有，通常是数据边缘）时走这条路。
///
/// 实测**没有任何检出依赖这条分支**：把它推到 0（永远通过）和 1e9（永远不过），
/// 四个小时（安静 / 亮暴 / 斩波 / 断档）的检出数一个不变。
///
/// # 关于「持续斩波检不出」
///
/// 这里原先写着：深饱和时探测器进入逐机箱连续斩波（约 7 ms 填满 FIFO → 复位
/// → 约 14 ms 死区，周期约 21 ms；2024-05-11 Gannon REP episode 实测零段起点
/// 间隔中位 21.0 ms、范围 16-22 ms，n=97），基线被压短后 14 ms 的空洞够不到
/// 100 倍，「整段斩波一个 reset 都不报」。
///
/// **这句与实测不符，已删。** 2024-05-11T20（该 episode 所在小时）实跑：旧判据
/// 报 26866 条，新判据 26871 条，洞长中位 18.47 ms，正落在那个 16-22 ms 的梳齿
/// 上。斩波是报出来的。
///
/// 下游的后果不受影响，仍然成立：深饱和段的包络是真通量的下界（逐箱占空比
/// 约 1/3），该段候选的显著性也不可靠（本底被死时间稀释）。对 TGF 计数无影响
/// ——风暴日在目录阶段就被 train 判据摘除了；但 REP episode 的定量必须回 1B 做
/// 占空比改正。诊断图见 `scripts/plot_gannon_fifo_chopping.py`。
const GAP_FACTOR: f64 = 100.0;

/// 泊松指数下限：`λ_stable × gap` 要超过它才算 FIFO 复位。
///
/// # 为什么 `GAP_FACTOR` 那个数读不出误报率
///
/// `gap > GAP_FACTOR / rate` 移项就是 `rate × gap > GAP_FACTOR`。如果 `rate`
/// 是真实泊松率 λ，那 `rate × gap` 就是 **λT**——「这段时间一个事例都没来」的
/// 泊松指数，概率 e^−λT，常量本身就有单位，能直接算出全任务冒出几个假空洞。
///
/// 但 `GAP_FACTOR` 除的是 `local_event_rate`，16 个事例估出来的 λ̂，相对标准差
/// 1/√15 = 25.8%。λT 进的是指数，±26% 就是 e^±26 ≈ 10^±11 的误报率摆动。两个
/// 方向都实测到了：向上涨落让噪声进来（名义降到 30 时 12 小时冒出 13 条 <5 ms
/// 的洞，而真 λT=30 的算术预期是全任务 0.32 条，差七个数量级）；向下涨落把真洞
/// 挡在外面（HEB211105190 的 11.87 ms 洞，λ̂T=47.5 不过，稳定 λT 是 61.8）。
///
/// # 30 是怎么定的
///
/// 改用 ±0.5 s 窗口的稳定 λ（σ≈1.4%）后，在 8 个整小时上分两群量：
///
/// | | n | 洞长中位 | 稳定 λT（p1 / 中位 / max） |
/// |---|---|---|---|
/// | 判据已接受的 | 1778 | 16.06 ms | 47.5 / 373.9 / 806.6 |
/// | 只在名义阈降到 10 才冒出的 | 2836 | 1.50 ms | 0 / 6.4 / **16.1** |
///
/// 噪声上界 16.1、真值最低（非零）37.8，**16–38 之间是空的**，阈值落在空谷里。
/// 自洽性：这 8 小时 ×3 箱 ×4000 evt/s ≈ 3.5e8 个间隔，纯泊松下最大 λT 的期望是
/// ln(3.5e8) = 19.7，实测噪声最大 16.1——换成稳定 λ 之后尾巴确实是泊松的，
/// 试验次数那套算术才可以用。
///
/// 全任务约 3.4e12 个间隔：λT=30 期望 0.32 条假空洞，λT=40 期望 1.4e-5。取 30
/// 而不是 40，是因为两头的代价不对称——误报只多 ±1 s 掩模（实测曝光影响 0.3%），
/// 漏检则是真数据缺口没被标出来。30 距噪声上界 1.9 倍、距最低真值 1.26 倍。
const MIN_POISSON_EXPONENT: f64 = 30.0;

/// 估计稳定率用的半窗（秒）。要长到把率估计的涨落压进百分之几（本底率下
/// 1 s 约 5000 个事例，σ≈1.4%），又短到跟得上进出 SAA 那种真实的率变化。
///
/// 实测在当前阈值下不敏感：13 个整小时上扫 0.1 / 0.25 / 0.5 / 1.0 / 2.0（以及
/// 0.001 和 10），**结果逐条相同**——两群的稳定 λT 离 30 太远，几个百分点的率
/// 差挪不动任何一条。把 `MIN_POISSON_EXPONENT` 推到 100（制造大量边缘样本）后
/// 旋钮就转了：安静小时 26 / 19 / 1 / 1 / 1，窗越窄率估得越高、越宽松（窄窗贴着
/// 洞前的高率段）。所以 0.5 不是中性的，只是当前阈值离敏感区够远。
const STABLE_RATE_HALF_WINDOW: f64 = 0.5;

/// FIFO Reset gap 的最大持续时间（秒）。超过此值的 gap 不认为是 FIFO 复位，
/// 而是数据传输中断、SAA 等其他原因。正常 FIFO reset gap 在 8ms~100ms 量级。
///
/// # 实测它现在是空转的
///
/// 13 个整小时上扫 1 / 2 / 5 / 10 / 30 / 100 s：**结果逐条相同**，一条也没多出来
/// ——这些小时里根本没有超过 1 s 还能过另外两关的候选洞。往下扫旋钮是转的
/// （20 ms 砍掉三成，5 ms 几乎砍光），所以不是扫描失灵。
///
/// 原因是 `LOCAL_RATE_FLOOR` 接管了它本来的活：长洞出现在中断段，那里洞旁的
/// 事例本来就稀，边缘率过不了闸门。也就是说这个上界目前是冗余的，**没有在漏检
/// 方向挡住东西**——这是查过的结论，不是假设。
const MAX_FIFO_RESET_GAP: f64 = 1.0;

/// MCU 从 FIFO A 读出的速率 (events/s)：109 events / ~7 ms ≈ 15,600，取 15000
/// 留一点余量。**这是个物理常量**，`reconstruct_gaps` 在 gap 两端都估不出率时
/// 拿它当形状函数的地板填充值。检测判据的闸门是另一回事，见 `LOCAL_RATE_FLOOR`
/// ——两者数值曾经相同，但含义不同，不要再合成一个。
const MCU_READ_RATE_FLOOR: f64 = 15000.0;

/// 本地事例率下限 (events/s)：低于此值的空洞不判为 FIFO 复位。
///
/// **不是物理阈值**。MCU 从 FIFO A 读出的速率是 109 events / ~7 ms ≈ 15,600
/// evt/s，只有入口率超过它 FIFO 才可能溢出，这条常量原先就取在那个数上。但
/// 把它设成物理值是把两件事压在一个阈值上：泊松否决已经由 `GAP_FACTOR` 承担
/// （100 倍平均间隔是 e^-100 的涨落，噪声根本够不到），闸门真正要挡的只是
/// 中断段那类**低速率长洞**——SAA 关机、传输中断，那里的包横跨几秒到上百秒才
/// 凑够 109 个事例，率只有 1–19 evt/s。
///
/// 而且记录下来的率是被审查过的下界：FIFO 一满就堵写，堵掉的事例恰恰不在数据
/// 里。拿物理值去卡一个系统性偏低的观测量，只会漏检。
///
/// # 阈值怎么定的
///
/// 12 个整小时上扫（含 SAA 进出、断档小时、221009A、2024-05-11T00 深饱和小时）：
///
/// | 阈值 evt/s | 检出 | 相对旧判据 |
/// |---|---|---|
/// | 500 / 1000 / 2000 / 3000 / 4000 / 6000 / 8000 | 27526（逐条相同） | +67, −0 |
/// | 10000 | 27523 | 开始丢 |
/// | 12000 | 27503 | −20 |
/// | 15000 | 27109 | −406 |
///
/// 500 到 8000 跨一个多数量级结果一字不差——判据对它不敏感，取值落在空谷里。
/// 15000 会丢 406 条，都在斩波段：洞旁那 16 个事例的率到不了 15 kHz，旧判据是靠
/// ±5 包在几百毫秒外找到的尖峰才放行的。
///
/// 取 2000：比中断段包的 1–19 evt/s 高两个数量级，比低速率箱的约 400 evt/s 高
/// 5 倍，又比平台上沿 8000 低 4 倍，两边都有余量。任何阈值下都没有出现整秒端点
/// 的空洞，也没有 >50 ms 的。
const LOCAL_RATE_FLOOR: f64 = 2000.0;

/// 量本地率用的事例数。
///
/// # 为什么不能用整包平均率
///
/// 一个包裹 109 个事例，本底率（约 4000 evt/s）下横跨约 27 ms。撑满 FIFO 只
/// 需要几毫秒的爆发，用 `n_events / span` 去量它，等于把毫秒级尖峰摊到整包
/// 上，稀释约一个数量级。两道关卡原先量到的都是这个平均：闸门比的是
/// `n_events / span`，`GAP_FACTOR` 的基线是它的倒数。于是一起漏检——洞长约等于
/// 455/尖峰率，门槛约等于 100/包级率，要通过就得「包级率 ≥ 尖峰率的 22%」。
///
/// 实测代价（HEB_list 832 个源窗，共 6.4 小时）：满足 `gap > 100×baseline`
/// 且 `≤ 1 s` 的空洞里，14478 条通过、25 条被整包平均挡掉。被挡掉的 25 条全部
/// 落在 15.6–17.9 ms，与通过的那批同一段；>100 ms 和 <10 ms 都是 0 条——
/// 挡掉的不是**另一类**东西，只是同一人群的低速率尾巴。
///
/// 16 个事例在 15,600 evt/s 下横跨约 1 ms，足以分辨那种爆发；再小则率估计
/// 自身的泊松涨落会明显偏高。
///
/// 实测这个值现在不敏感：13 个整小时上扫 8 / 12 / 16 / 24 / 32 / 48，**结果逐条
/// 相同**，连 2 和 100 都一样。它只喂 `edge_rate`，而 `LOCAL_RATE_FLOOR` 本身
/// 空转（见那里的扫描），所以窗口大小无从体现。把闸门推到 12000（开始咬）之后
/// 旋钮就转了（安静小时 44→38），说明扫描是活的。
const RATE_WINDOW_EVENTS: usize = 16;
/// 一段（包头或包尾）事例的率 (events/s)。`times` 必须升序。
///
/// k 个事例张开 T 秒里有 k−1 个间隔，所以率取 (k−1)/T：泊松过程下
/// E[T] = (k−1)/λ，用 k/T 会系统性高估 k/(k−1)。
fn edge_event_rate(times: &[f64]) -> Option<f64> {
    if times.len() < 2 {
        return None;
    }
    let span = times[times.len() - 1] - times[0];
    (span > 1e-9).then(|| (times.len() - 1) as f64 / span)
}

/// 空洞处的本地事例率 (events/s)：洞前最后 `RATE_WINDOW_EVENTS` 个事例、
/// 洞后最前 `RATE_WINDOW_EVENTS` 个事例，两者取高。
///
/// **只看洞旁边**。判据问的两件事——入口率够不够撑满 FIFO、这个洞比本地事例
/// 间隔大多少倍——都只关于空洞紧邻的那一小段时间。原来的写法一是把率算成整包
/// 109 个事例的平均，二是在 ±5 个包的范围里取最大（本底率下约 ±135 ms；中断段
/// 里一个包能横跨 108 s，就是 ±几百秒）。拿几百毫秒外的尖峰去论证这个洞旁边
/// FIFO 撑满了，站不住；实测也确实会把中断段起止落在整秒的空洞放进来。
///
/// ±5 包那个加宽是旧写法的遗留：一个包只给一个率，只能靠横向多取几个包压涨落。
/// 换成事例级窗口后一个包里就有约 109 个测量，这个理由不成立了。
fn local_event_rate(prev: &PacketTimeSummary, next: &PacketTimeSummary) -> Option<f64> {
    match (prev.tail_rate, next.head_rate) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// 空洞附近 ±`STABLE_RATE_HALF_WINDOW` 秒的稳定事例率 (events/s)。
///
/// 计数按包与窗口的交叠比例分摊，除以窗口内被包覆盖的**并集**时长。
///
/// # 为什么分母是并集而不是逐包求和
///
/// **FIFO 严格保序，所以同一个盒的包在时间上本不该重叠**：包 N 的最后一个事例
/// 必须早于包 N+1 的第一个。实测也确实如此——2023-06-16T20、2019-05-30T10、
/// 2022-10-09T13、2024-05-11T20、2021-02-13T06 五个小时近两百万个包，
/// `max_met[i] > min_met[i+1]` 的违规**一次都没有**。
///
/// 但 2026-01-25T19 有 23 次（34 万包里，0.007%），而且不是物理现象：查到包内，
/// 是**单个事例的重建时间错了**。pkt 112554 的 109 个事例里 108 个挤在 2.8 ms
/// 内，剩一个晚了 591.7 ms；pkt 114342 的主体 3.0 ms，离群那个晚了 1051.7 ms
/// ——约等于一个 ptime 回绕周期（1.048576 s），指向 wrap 误判。这类包在包内按
/// `evt_idx` 排序时会出现时间倒退，正是 FIFO 保序不允许的。
///
/// 这一个杂散事例把包的 `max_met` 拉远，于是该包与后面几百个包"重叠"。逐包
/// 交叠求和会把它的近 1 s 全算进分母却只带 109 个事例，率被稀释约十倍
/// （3,289/s 对真实的约 35,000/s），**是漏检方向**。取并集就免疫。
///
/// 中断段里还有一类**合法**的长包：约 1 evt/s 时一百多秒才凑够 109 个事例，
/// 包内无倒退。并集同样处理得对。
///
/// 同理往前回退找起点不能一见 `max_met < lo` 就停：`reach[i]` 是
/// `max_met[0..=i]` 的前缀最大值，单调不减，据它回退才不漏包。（改之前 Rust
/// 提前停、恰好把被污染的包排除了，另写的 Python 不停，两边算同一个量差出
/// 两三倍，这个缺陷就是这样暴露的。）
///
/// **根因没修**：`min_met`/`max_met` 仍取包内事例时间的极值，对单个杂散事例
/// 不稳健。后果不止稀释率——`gap = 后包.min_met − 前包.max_met` 也被撑大的
/// `max_met` 切碎：那一秒里 1B 报出 1.6–6.9 ms 一串碎片，1K 在同位置看到的是
/// 一个完整的 15.1–15.4 ms 空洞。`blink_old` 报一样的碎片，所以是既有问题。
///
/// 这里用包级平均是对的：反对包级平均是因为要分辨毫秒尖峰（那是 FIFO 闸门的
/// 活，见 `RATE_WINDOW_EVENTS`），而泊松零假设要的恰恰是长窗口的平均率。
///
/// `summaries` 必须已按 `min_met` 升序，`reach` 是它的 `max_met` 前缀最大值。
/// 窗口内没有任何包覆盖时返回 `None`，由调用方回退到边缘率判据。
fn stable_event_rate(
    summaries: &[PacketTimeSummary],
    reach: &[f64],
    center: f64,
) -> Option<f64> {
    let (lo, hi) = (
        center - STABLE_RATE_HALF_WINDOW,
        center + STABLE_RATE_HALF_WINDOW,
    );
    let end = summaries.partition_point(|s| s.min_met <= hi);
    let mut i = end;
    while i > 0 && reach[i - 1] >= lo {
        i -= 1;
    }

    let mut counts = 0.0;
    let mut live = 0.0;
    let mut frontier = lo;
    for s in &summaries[i..end] {
        let a = s.min_met.max(lo);
        let b = s.max_met.min(hi);
        if b <= a {
            continue;
        }
        let span = s.max_met - s.min_met;
        counts += if span > 1e-9 {
            s.n_events as f64 * (b - a) / span
        } else {
            s.n_events as f64
        };
        // 包按 min_met 升序 ⇒ a 单调不减，一次扫描就能并集
        if b > frontier {
            live += b - a.max(frontier);
            frontier = b;
        }
    }
    (live > 1e-6).then_some(counts / live)
}

/// 丢掉时间与打包顺序矛盾的包，返回保留掩码。
///
/// # 为什么可以这么判
///
/// FIFO A 严格保序，MCU 按序读出、按序打包，所以**按 `pkt_idx` 排序后各包的
/// `min_met` 必须不减**。这是仪器的硬约束，不是经验规则。
///
/// 实测它几乎处处成立：2023-06-16T20、2019-05-30T10、2022-10-09T13（221009A）、
/// 2024-05-11T20（斩波）、2021-02-13T06（断档）、2024-05-11T00 六个小时约 260 万
/// 个包，违规**零次**。只有 2026-01-25T19 有 3 处（34 万包里），而且违规处的时间
/// 位移中位 **1.0303 s**——对上 ptime 回绕周期 1.048576 s（2^19 × 2 µs）。
///
/// # 违反它的是什么
///
/// 是**整块包的 wrap 被判错**。深饱和斩波下每约 21 ms 出 2 个包（约 95 包/s），
/// 所以「包号差 100」对应约 1.05 s，正好一个 wrap。那一段实测：
///
/// ```text
/// 114341 [200.994142, 200.996914]     ← 真实梳齿的「开」相位
/// 114342 [200.996920, 202.048572]     ← 末事例晚了一个 wrap
/// 114538 [201.004632, 201.007984]     ← 包号 +200，却排在这里
/// 114440 [201.006846, 201.009948]     ← 包号 +100，早了一个 wrap
/// 114343 [201.015244, 201.018314]     ← 回到真实梳齿
/// ```
///
/// 同段 1K 是干净的 15.2 ms 梳齿（约 6 ms 有数据 → 15.2 ms 死区）。被错配的包落进
/// 死区里，把真实空洞**填上**，`detect` 于是只看到残余的 1.6–6.9 ms 碎片。
///
/// # 为什么是按包号贪心，不是最长不减子序列
///
/// 先试的是最长不减子序列，**挑错了边**。错配块与被它覆盖的正确块长度相当
/// （那一小时各约 100 个包），对最长链是近似平局，LIS 没有信息分辨谁对，实测它
/// 保留了错配的 1144xx、丢掉了真实的 1143xx，报出的洞随之整体偏 13 ms。
///
/// 可分辨的信息在包号里：**错配块的包号更大、时间却更早**（被判早了一个 wrap）。
/// 所以按包号顺序走一遍、时间一旦倒退就剔，先被保住的必然是正确的那支。
///
/// 这条对「被判**晚**一个 wrap」的块不成立——那时贪心会先接受错的、把随后正确的
/// 整片剔掉。实测的三处都是判早，没有遇到判晚的；真出现要靠包的 UTC 尾戳
/// （`rec_sci_data.rs` 已用它给 wrap 候选定上界）另行判别，这里不兜底。
///
/// 根因在 `rec_sci_data.rs` 的 wrap 指派（深饱和时 SEC 锚点稀疏/损坏）。这里只做
/// 剔除，不做修复：被剔的包时间不可信，它们覆盖的区间也就不再产出饱和结论。
/// 包**内**的同一条不变量：事例按 `evt_idx` 读出，时间必须不减。
/// 返回保序的子串（原序，无需排序）。
///
/// 违反它的是**单个事例的时间错了**，离群量实测约等于一个 ptime wrap：
/// pkt 114342 的 109 个事例里，0–103 正常升序、第 104 个跳了 **1048.6 ms**
/// （wrap = 1048.576 ms）、105–108 又回到正常。留着它 `max_met` 被撑到 202.05，
/// `gap = 后包.min_met − 前包.max_met` 算成负数，那个真实空洞**整个被吞掉**
/// ——实测正是这样漏掉了 1K 看得见的一齿。
///
/// # 为什么这里剔尖刺，包级那里用贪心
///
/// 两层的形状不同。包内是「1 个离群对 108 个正常」，一眼就知道谁错，剔掉孤立
/// 尖刺即可。包级是「两支长度相当」（错配块与它覆盖的正确块各约 100 个包），
/// 谁对要靠包号顺序来断。**按包号贪心用在包内会正好做反**：它会接受那个尖刺、
/// 把后面 4 个正常事例全剔掉，`max_met` 反而更糟——实测踩过。
fn fifo_ordered_prefix(times: &[f64]) -> Vec<f64> {
    let v: Vec<f64> = times.iter().copied().filter(|t| !t.is_nan()).collect();
    if v.len() < 3 {
        return v;
    }
    // 一遍剔掉「比两侧都晚」的孤立尖刺。只判向上这一侧：向下的离群交给后面的
    // 单调兜底去掉，若这里也判向下，尖刺**后面**那个正常事例会被连坐误剔。
    let mut out = Vec::with_capacity(v.len());
    for i in 0..v.len() {
        let prev = if i == 0 { f64::NEG_INFINITY } else { v[i - 1] };
        let next = if i + 1 == v.len() { f64::INFINITY } else { v[i + 1] };
        if !(v[i] > next && v[i] > prev) {
            out.push(v[i]);
        }
    }
    // 兜底：`edge_event_rate` 要求升序，剩下的再过一遍单调
    let mut kept = Vec::with_capacity(out.len());
    let mut frontier = f64::NEG_INFINITY;
    for t in out {
        if t >= frontier {
            frontier = t;
            kept.push(t);
        }
    }
    kept
}

fn fifo_consistent_mask(summaries: &[PacketTimeSummary]) -> Vec<bool> {
    let mut keep = vec![true; summaries.len()];
    let mut frontier = f64::NEG_INFINITY;
    for (i, s) in summaries.iter().enumerate() {
        if s.min_met < frontier {
            keep[i] = false;
        } else {
            frontier = s.min_met;
        }
    }
    keep
}

/// 检测整包丢失（FIFO reset）造成的饱和区间。
///
/// 算法：
/// 1. 对每个 CCSDS 包重建所有事件的 MET 时间，提取 (min_met, max_met, n_events)
/// 2. 按 min_met 排序
/// 3. 对每对相邻包：
///    - edge_rate = 空洞两侧各 `RATE_WINDOW_EVENTS` 个事例的率，取高者（瞬时）
///    - 若 edge_rate < LOCAL_RATE_FLOOR → 跳过（中断段那类低速率长洞）
///    - 泊松否决用 ±`STABLE_RATE_HALF_WINDOW` 秒的稳定率：
///      `stable_rate × gap > MIN_POISSON_EXPONENT` → 标记为 FifoReset；
///      稳定率估不出来时回退到 `edge_rate × gap > GAP_FACTOR`
///
/// 两个测试要用两个不同的 λ：FIFO 那一问要毫秒尺度的瞬时率，泊松那一问要
/// 稳定的本底率。共用一个估计量是原判据的病根，见 `MIN_POISSON_EXPONENT`。
pub fn detect_fifo_reset_intervals(sci_data: &SciFile, offset: f64) -> Vec<SaturationInterval> {
    let packet_times = reconstruct_with_wrap_tracking(sci_data, offset);

    let mut summaries: Vec<PacketTimeSummary> = Vec::new();
    let mut n_evt_dropped = 0usize;
    let mut n_evt_total = 0usize;
    for (pkt_idx, times) in packet_times.iter().enumerate() {
        // FIFO 保序：包内按 evt_idx 时间必须不减，违反的是时间错了的事例。
        // 保序子串本身就是升序的，不用再排。
        let n_valid = times.iter().filter(|t| !t.is_nan()).count();
        let valid = fifo_ordered_prefix(times);
        n_evt_total += n_valid;
        n_evt_dropped += n_valid - valid.len();
        if valid.is_empty() {
            continue;
        }
        summaries.push(PacketTimeSummary {
            pkt_idx,
            min_met: valid[0],
            max_met: valid[valid.len() - 1],
            n_events: valid.len(),
            head_rate: edge_event_rate(&valid[..RATE_WINDOW_EVENTS.min(valid.len())]),
            tail_rate: edge_event_rate(&valid[valid.len().saturating_sub(RATE_WINDOW_EVENTS)..]),
        });
    }
    if n_evt_dropped > 0 {
        eprintln!(
            "  warning: {} of {} events have times that contradict the read-out order \
             within their packet (wrap misassignment); dropped from gap detection",
            n_evt_dropped, n_evt_total
        );
    }

    // FIFO 保序：按 pkt_idx 排序后 min_met 必须不减。违反的是 wrap 被判错的包，
    // 留下它们会把真实空洞填上、切成碎片。见 `fifo_consistent_mask`。
    let keep = fifo_consistent_mask(&summaries);
    let n_dropped = keep.iter().filter(|k| !**k).count();
    if n_dropped > 0 {
        eprintln!(
            "  warning: {} of {} packets have times that contradict the packing order \
             (wrap misassignment); dropped from gap detection",
            n_dropped,
            summaries.len()
        );
        let mut it = keep.iter();
        summaries.retain(|_| *it.next().unwrap());
    }

    summaries.sort_by(|a, b| a.min_met.partial_cmp(&b.min_met).unwrap());

    // max_met 的前缀最大值：包可能互相重叠（拥塞宽包），按它回退才不漏包。
    let mut reach: Vec<f64> = Vec::with_capacity(summaries.len());
    let mut run = f64::NEG_INFINITY;
    for s in &summaries {
        run = run.max(s.max_met);
        reach.push(run);
    }

    let mut intervals = Vec::new();
    for window in summaries.windows(2) {
        let gap = window[1].min_met - window[0].max_met;
        if gap <= 0.0 {
            continue;
        }

        let edge_rate = match local_event_rate(&window[0], &window[1]) {
            Some(r) => r,
            None => continue,
        };
        if edge_rate < LOCAL_RATE_FLOOR {
            continue;
        }

        let beats_poisson = match stable_event_rate(&summaries, &reach, window[0].max_met) {
            Some(stable) => gap * stable > MIN_POISSON_EXPONENT,
            None => gap * edge_rate > GAP_FACTOR,
        };
        if beats_poisson && gap <= MAX_FIFO_RESET_GAP {
            intervals.push(SaturationInterval {
                start_met: window[0].max_met,
                stop_met: window[1].min_met,
                gap_seconds: gap,
                prev_pkt_idx: window[0].pkt_idx,
                next_pkt_idx: window[1].pkt_idx,
                saturation_type: SaturationType::FifoReset,
            });
        }
    }

    intervals
}

/// 每个包的时间摘要（公开版本，用于重建）
#[derive(Debug, Clone)]
pub struct PacketInfo {
    pub pkt_idx: usize,
    pub min_met: f64,
    pub max_met: f64,
    pub n_events: usize,
}

impl PacketInfo {
    pub fn span(&self) -> f64 {
        self.max_met - self.min_met
    }
}

/// 从 SciFile 提取包时间摘要列表（按 min_met 排序）
pub fn extract_packet_infos(sci_data: &SciFile, offset: f64) -> Vec<PacketInfo> {
    let packet_times = reconstruct_with_wrap_tracking(sci_data, offset);
    let mut infos: Vec<PacketInfo> = packet_times
        .iter()
        .enumerate()
        .filter_map(|(pkt_idx, times)| {
            let valid: Vec<f64> = times.iter().copied().filter(|t| !t.is_nan()).collect();
            if valid.is_empty() {
                return None;
            }
            let min_met = valid.iter().cloned().reduce(f64::min).unwrap();
            let max_met = valid.iter().cloned().reduce(f64::max).unwrap();
            Some(PacketInfo {
                pkt_idx,
                min_met,
                max_met,
                n_events: valid.len(),
            })
        })
        .collect();
    infos.sort_by(|a, b| a.min_met.partial_cmp(&b.min_met).unwrap());
    infos
}

/// 不可信时间区间（FIFO reset gap 或拥塞宽包的时间覆盖）
#[derive(Debug, Clone)]
pub struct UnreliableInterval {
    pub start: f64,
    pub stop: f64,
}

/// 单个 box 的饱和重建数据
#[derive(Debug)]
pub struct BoxReconstructionData {
    /// 原始事件 MET 时间（已排序）
    pub events: Vec<f64>,
    /// 与 events 一一对应的 wrapped channel（SEC 槽 = CHANNEL_SEC）
    pub channels: Vec<u16>,
    /// 与 events 一一对应的脉宽 pulinfo（NaI/CsI 甄别；SEC 槽 = 0）
    pub pulse_widths: Vec<u8>,
    /// FIFO reset 区间
    pub gaps: Vec<SaturationInterval>,
    /// 包信息
    pub packets: Vec<PacketInfo>,
    /// 每个包内的事件时间（索引 = 原始包号，内部已排序）
    pub packet_events: Vec<Vec<f64>>,
    /// 不可信区间（FIFO reset gap），用于交叉参考时排除
    pub unreliable: Vec<UnreliableInterval>,
}

/// 检测不可信时间区间：仅 FIFO reset gap。
///
/// 拥塞宽包和包内泊松异常检测已移除：
/// - 拥塞宽包：实际触发多为 SAA 开关机导致的包跨时异常，非真正拥塞
/// - 包内异常：与静默丢数检测同一判据（泊松 log₁₀(p) < -10），
///   因 λ 在单包时间跨度内不稳定导致大量误报
pub fn detect_unreliable_intervals(
    gaps: &[SaturationInterval],
    _packets: &[PacketInfo],
    _packet_events: &[Vec<f64>],
) -> Vec<UnreliableInterval> {
    let mut intervals: Vec<UnreliableInterval> = Vec::new();

    for g in gaps {
        intervals.push(UnreliableInterval {
            start: g.start_met,
            stop: g.stop_met,
        });
    }

    // 按 start 排序
    intervals.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    intervals
}

/// `intervals` 由 `detect_unreliable_intervals` 按 `start` 排好，且同一盒内互不
/// 重叠（每段取自相邻两包之间），所以二分与逐个扫等价。
///
/// 原来是线性 `any()`，逐事例、逐形状格各扫一遍全表。复位稀疏时无所谓，但
/// 2024-09-18T20 那种一小时两万条复位的深饱和小时，每盒七八千段 × 上百万次查询
/// 就成了实际意义上的 O(n²)：实测 707 段的小时 8.3 gap/s，7000 段的小时掉到
/// 0.107 gap/s，慢 78 倍，整小时要跑几十小时。
fn is_in_unreliable(t: f64, intervals: &[UnreliableInterval]) -> bool {
    debug_assert!(
        intervals.windows(2).all(|w| w[0].start <= w[1].start && w[0].stop <= w[1].start),
        "unreliable intervals must be sorted and disjoint for the binary search"
    );
    let i = intervals.partition_point(|iv| iv.stop < t);
    i < intervals.len() && intervals[i].start <= t
}

/// 单个参考盒对某 cross-ref gap 的标定描述子(spec §13/§5b)。
/// σ_k 由计数给:(σ_k/k)² = 1/C_a_cal + 1/C_ref_cal;跨-ref 相关用共用分子
/// C_a_cal(评审①):Σ_k[b,b'] = k_b k_b'(δ/C_ref_cal + 1/C_a_cal)。
#[derive(Debug, Clone)]
pub struct GapRefCalib {
    /// 在 `references` 切片里的下标
    pub ref_idx: usize,
    /// 标定系数 k = 目标/参考 率比(标定窗内)
    pub k: f64,
    /// 标定窗内 target 计数(所有 ref 共用分子)
    pub c_a_cal: f64,
    /// 标定窗内该参考盒计数
    pub c_ref_cal: f64,
}

/// gap 的协方差块描述子(spec §13)。cross-ref 用 `refs`(每参考盒的 k 与标定窗计数);
/// degenerate 用端点率 `r_pre/r_post` 与实际事例数 `n_pre/n_post`(自由度)。
/// `sys_bias_scale` 是退化外推偏差的粗代理 |r_post−r_pre|/(r_pre+r_post)(评审④,
/// 下界启发式);cross-ref 段为 0。`maskable`=(None,None) 地板填充,不可信、可屏蔽(评审③)。
#[derive(Debug, Clone, Default)]
pub struct GapCovBlock {
    /// cross-ref:参与形状重建的各参考盒标定(空 = degenerate)
    pub refs: Vec<GapRefCalib>,
    /// degenerate:gap 前端点率(None = 该端无有效率)
    pub r_pre: Option<f64>,
    /// degenerate:gap 后端点率
    pub r_post: Option<f64>,
    /// degenerate:前端点 packet 实际事例数(自由度 = n_pre−1)
    pub n_pre: Option<f64>,
    /// degenerate:后端点 packet 实际事例数
    pub n_post: Option<f64>,
    /// (None,None) 地板填充:率无从估、不入协方差、下游可屏蔽(评审③)
    pub maskable: bool,
    /// 系统偏差量级粗代理(评审④);cross-ref 段为 0
    pub sys_bias_scale: f64,
    /// cross-ref 重整因子 ρ=N_lost/Σshape(spec §7,S 系数 ρk_b/n_m);退化段为 0
    pub rho: f64,
}

/// gap 内单个 1ms 格的 kind(spec ③ gapbins)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapBinKind {
    /// 有参考盒覆盖:S 用恒等,filler↔参考系数 ρk_b/n_m
    Measured,
    /// 无参考盒,靠端点插值:S 借 left/right 端点格的参考事件
    Empty,
}

/// gap 内单个 1ms 格的结构描述(spec ③ gapbins)。measured 格记有效参考盒数 n_m;
/// empty 格记插值端点 (left_bin,right_bin) 与权重 τ,下游据此在 1ms 网格精确拼 S,
/// 不必重猜插值算子(避免与 Rust 漂移)。
#[derive(Debug, Clone)]
pub struct GapBinInfo {
    /// gap 内 1ms 格序号(0..n_sbins)
    pub bin_index: usize,
    /// 该格下沿的 MET 时间
    pub t_lo: f64,
    /// measured:有效参考盒数;empty 不适用(下游用端点格的 n_m),置 0
    pub n_m: usize,
    /// measured / empty
    pub kind: GapBinKind,
    /// empty:插值左端点 bin_index(measured 为 None)
    pub left_bin: Option<usize>,
    /// empty:插值右端点 bin_index(measured 为 None)
    pub right_bin: Option<usize>,
    /// empty:插值权重 τ=(i−l)/(r−l)(measured 为 None)
    pub tau: Option<f64>,
}

/// 重建后的补全事件
#[derive(Debug, Clone)]
pub struct ReconstructedGap {
    /// 对应的 gap 索引
    pub gap_idx: usize,
    /// 补全的事件 MET 时间
    pub filled_events: Vec<f64>,
    /// N_lost
    pub n_lost: usize,
    /// 是否使用了交叉参考
    pub has_cross_ref: bool,
    /// 每个 filler 的“仅方差”权重 v。cross-ref gap = 0（方差由参考事件承载）；
    /// 退化 gap = √(σ²_gap / N_lost)，σ²_gap 由 pre/post 包率涨落传播，使方差
    /// 局域在 gap 内（挂 filler）而非畸变相邻真实事件。
    pub filler_weight: f64,
    /// spec §13 协方差块描述子(下游装配协方差用)
    pub cov: GapCovBlock,
    /// spec ③:gap 内每 1ms 格结构(measured/empty、n_m、插值端点/τ)。cross-ref
    /// gap 才有;退化 gap 为空(方差走 r項/filler_weight,不经 S filler↔参考)。
    pub bins: Vec<GapBinInfo>,
}

const SHAPE_BIN_WIDTH: f64 = 0.001; // 1ms

/// 对单个 box 的 FIFO reset gap 进行光变曲线重建。
///
/// 算法：
/// 1. 用参考 box 的事件分布构建校准后的形状函数（1ms bin）
/// 2. N_lost = shape 总和（校准后的参考计数直接给出丢失事件数）
/// 3. 按形状分配事件到各 bin
///
/// 当参考 box 不可用（所有 box 同时饱和）时，退化为 post-reset 包率估算 + 均匀分配。
pub fn reconstruct_gaps(
    target: &BoxReconstructionData,
    references: &[&BoxReconstructionData],
) -> (Vec<ReconstructedGap>, Vec<Vec<f64>>) {
    let mut results = Vec::new();
    // per-reference-box, per-event particle-weight contribution from THIS
    // target's gaps (aligned with `references`); accumulated below.
    let mut ref_weights: Vec<Vec<f64>> = references
        .iter()
        .map(|r| vec![0.0f64; r.events.len()])
        .collect();

    for (gap_idx, gap) in target.gaps.iter().enumerate() {
        let gap_start = gap.start_met;
        let gap_stop = gap.stop_met;
        let gap_dur = gap_stop - gap_start;
        if gap_dur <= 0.0 {
            continue;
        }

        // 步骤一：构建形状 bin
        let n_sbins = ((gap_dur / SHAPE_BIN_WIDTH).ceil() as usize).max(1);
        let actual_sbin = gap_dur / n_sbins as f64;
        let mut shape = vec![0.0f64; n_sbins];
        // 每个 bin 记录贡献它的 (ref_idx, 事件起, 事件止, k)，供逐事件权重反算
        let mut bin_refs: Vec<Vec<(usize, usize, usize, f64)>> =
            vec![Vec::new(); n_sbins];

        // 汇总所有参考 box 的事件，统一构建形状函数
        // k 只依赖 (gap, 参考盒)，与形状 bin 无关，整段 gap 算一次即可；
        // 原来放在逐 bin 循环里，每个 gap 要重复调用 n_sbins 次（等价改写）。
        let ks: Vec<f64> = references
            .iter()
            .map(|ref_data| {
                calibrate_ratio_sorted(
                    &target.events,
                    &ref_data.events,
                    &target.unreliable,
                    &ref_data.unreliable,
                    gap_start,
                    gap_stop,
                    0.5,
                )
            })
            .collect();
        let mut has_ref = false;
        // M1:每格『可用(可信)参考盒数』——形状/权重/n_m 的正确分母。饱和(is_in_unreliable
        // 被排除)才不算可用;可信但该格 count=0 的盒仍算可用(0 计数入分子、+1 入分母)。
        // 『饱和』与『count=0』是两件事,不能都从分母踢掉。
        let mut avail = vec![0usize; n_sbins];
        for si in 0..n_sbins {
            let bin_lo = gap_start + si as f64 * actual_sbin;
            let bin_hi = bin_lo + actual_sbin;
            let bin_mid = (bin_lo + bin_hi) / 2.0;

            let mut total_ref_count = 0.0;
            let mut n_avail = 0usize;

            for (ref_idx, ref_data) in references.iter().enumerate() {
                if is_in_unreliable(bin_mid, &ref_data.unreliable) {
                    continue;
                }
                n_avail += 1;

                let lo_idx = ref_data.events.partition_point(|&t| t < bin_lo);
                let hi_idx = ref_data.events.partition_point(|&t| t < bin_hi);
                let count = (hi_idx - lo_idx) as f64;

                if count > 0.0 {
                    let k = ks[ref_idx];
                    total_ref_count += count * k;
                    bin_refs[si].push((ref_idx, lo_idx, hi_idx, k));
                }
            }

            avail[si] = n_avail;
            if n_avail > 0 {
                shape[si] = total_ref_count / n_avail as f64;
                if shape[si] > 0.0 {
                    has_ref = true;
                }
            }
        }

        // ── diagnostic: calibration ratio breakdown ──
        for (ri, ref_data) in references.iter().enumerate() {
            let dw = [(gap_start - 0.5, gap_start), (gap_stop, gap_stop + 0.5)];
            let (mut tc, mut rc) = (0usize, 0usize);
            let (mut te, mut re) = (0.0f64, 0.0f64);
            for &(wl, wh) in &dw {
                let t = effective_duration(wl, wh, &target.unreliable);
                if t > 1e-6 {
                    let a = target.events.partition_point(|&x| x < wl);
                    let b = target.events.partition_point(|&x| x < wh);
                    tc += target.events[a..b].iter()
                        .filter(|&&x| !is_in_unreliable(x, &target.unreliable)).count();
                    te += t;
                }
                let r = effective_duration(wl, wh, &ref_data.unreliable);
                if r > 1e-6 {
                    let a = ref_data.events.partition_point(|&x| x < wl);
                    let b = ref_data.events.partition_point(|&x| x < wh);
                    rc += ref_data.events[a..b].iter()
                        .filter(|&&x| !is_in_unreliable(x, &ref_data.unreliable)).count();
                    re += r;
                }
            }
            if re > 1e-6 && rc > 10 && te > 1e-6 {
                let tr = tc as f64 / te;
                let rr = rc as f64 / re;
                eprintln!("  gap[{gap_idx}] ref[{ri}]: k={:.4}  tgt={tc}/{te:.4}s={tr:.0}/s  ref={rc}/{re:.4}s={rr:.0}/s", tr/rr);
            } else {
                eprintln!("  gap[{gap_idx}] ref[{ri}]: k=1.0(default)  tgt={tc}/{te:.4}s  ref={rc}/{re:.4}s");
            }
        }

        // 步骤二：确定 N_lost、退化方差、是否真用了 cross-ref
        let n_lost;
        // None = cross-ref(filler 权重 0，方差在参考事件)；Some(σ²) = 退化
        // (Some(NaN) = (None,None) 地板填充，按 ~100% 不确定处理)
        let filler_sigma2: Option<f64>;
        let used_cross_ref: bool;
        // spec ③/§7:cross-ref 分支填这两项(退化分支保持默认:ρ=0、空格结构)。
        let mut cross_rho = 0.0f64;
        let mut gap_bins: Vec<GapBinInfo> = Vec::new();
        if has_ref {
            let n_filled = shape.iter().filter(|&&v| v > 0.0).count();
            if n_filled * 100 / n_sbins >= 30 {
                // 参考覆盖充分：用 shape 总和作为 N_lost
                let (lambda, interp) = interpolate_empty_bins(&mut shape, &avail);
                let total: f64 = shape.iter().sum();
                n_lost = total.round() as usize;
                // 逐参考事件累加权重贡献 w_contrib = ρ · k/n_m · (1+Λ)。
                // 每个 filler 都完整归回它的源参考事件（守恒：Σ contrib = n_lost）。
                let rho = if total > 0.0 { n_lost as f64 / total } else { 0.0 };
                for (si2, refs_in_bin) in bin_refs.iter().enumerate() {
                    if refs_in_bin.is_empty() {
                        continue;
                    }
                    // 分母用可用盒数(M1),不是非零盒数
                    let amp = rho * (1.0 + lambda[si2]) / avail[si2] as f64;
                    for &(ref_idx, lo, hi, k) in refs_in_bin {
                        let contrib = amp * k;
                        for ev in lo..hi {
                            ref_weights[ref_idx][ev] += contrib;
                        }
                    }
                }
                // spec ③:把逐格结构序列化出来(内部已算好,别让下游重推)。interp[si2]
                // None ⟺ 该格 measured(有参考);Some((l,r,τ)) ⟺ 空格,借端点插值。
                cross_rho = rho;
                gap_bins = (0..n_sbins)
                    .map(|si2| {
                        let t_lo = gap_start + si2 as f64 * actual_sbin;
                        match interp[si2] {
                            None => GapBinInfo {
                                bin_index: si2,
                                t_lo,
                                n_m: avail[si2],
                                kind: GapBinKind::Measured,
                                left_bin: None,
                                right_bin: None,
                                tau: None,
                            },
                            Some((l, r, t)) => GapBinInfo {
                                bin_index: si2,
                                t_lo,
                                n_m: 0,
                                kind: GapBinKind::Empty,
                                left_bin: Some(l),
                                right_bin: Some(r),
                                tau: Some(t),
                            },
                        }
                    })
                    .collect();
                filler_sigma2 = None; // 方差由参考事件承载
                used_cross_ref = true;
                eprintln!("gap[{gap_idx}]: {gap_dur:.4}s  n_lost={n_lost}  cross-ref  cov={n_filled}/{n_sbins}");
            } else {
                // 参考覆盖不足：退化为 pre/post 率线性插值（不是 cross-ref）
                fill_shape_fallback(&mut shape, gap, &target.packets);
                n_lost = (shape.iter().sum::<f64>() * actual_sbin).round() as usize;
                filler_sigma2 = Some(degenerate_gap_variance(&target.packets, gap, gap_dur));
                used_cross_ref = false;
                eprintln!("gap[{gap_idx}]: {gap_dur:.4}s  n_lost={n_lost}  FALLBACK  cov={n_filled}/{n_sbins}");
            }
        } else {
            // 无参考：pre/post 率线性插值
            fill_shape_fallback(&mut shape, gap, &target.packets);
            n_lost = (shape.iter().sum::<f64>() * actual_sbin).round() as usize;
            filler_sigma2 = Some(degenerate_gap_variance(&target.packets, gap, gap_dur));
            used_cross_ref = false;
            eprintln!("gap[{gap_idx}]: {gap_dur:.4}s  n_lost={n_lost}  NO-REF");
        }

        if n_lost == 0 {
            continue;
        }

        // 步骤三：按形状分配事件
        let total: f64 = shape.iter().sum();
        if total <= 0.0 {
            continue;
        }

        let mut filled_events = Vec::with_capacity(n_lost);
        for (si, &s) in shape.iter().enumerate() {
            let n_in_bin = (s / total * n_lost as f64).round() as usize;
            if n_in_bin > 0 {
                let bin_lo = gap_start + si as f64 * actual_sbin;
                let bin_hi = bin_lo + actual_sbin;
                let step = (bin_hi - bin_lo) / n_in_bin as f64;
                for j in 0..n_in_bin {
                    filled_events.push(bin_lo + (j as f64 + 0.5) * step);
                }
            }
        }

        // 退化 filler 的“仅方差”权重 v=√(σ²_gap/实际filler数),使 Σv²=σ²_gap。
        // 用 filled_events.len()(逐 bin round 后可 ≠ n_lost),否则下游按实际
        // filler 数求和时不守恒。cross-ref=0；(None,None) 地板填充按 σ²=filler数²
        // (~100% 不确定)→ v=√(filler数),绝不给 0。
        let n_filled_actual = filled_events.len();
        let filler_weight = match filler_sigma2 {
            None => 0.0,
            Some(_) if n_filled_actual == 0 => 0.0,
            Some(s2) if s2.is_nan() => (n_filled_actual as f64).sqrt(),
            Some(s2) => (s2 / n_filled_actual as f64).sqrt(),
        };

        // spec §13 协方差块描述子
        let cov = if used_cross_ref {
            // 收集参与形状重建的参考盒(bin_refs 里出现过),各算 k 与标定窗计数
            let mut seen = vec![false; references.len()];
            let mut refs_calib: Vec<GapRefCalib> = Vec::new();
            for refs_in_bin in &bin_refs {
                for &(ref_idx, _, _, _) in refs_in_bin {
                    if !seen[ref_idx] {
                        seen[ref_idx] = true;
                        let (k, c_a, c_ref) = calibrate_counts(
                            &target.events, &references[ref_idx].events,
                            &target.unreliable, &references[ref_idx].unreliable,
                            gap_start, gap_stop, 0.5,
                        );
                        refs_calib.push(GapRefCalib {
                            ref_idx, k, c_a_cal: c_a as f64, c_ref_cal: c_ref as f64,
                        });
                    }
                }
            }
            refs_calib.sort_by_key(|r| r.ref_idx);
            GapCovBlock { refs: refs_calib, rho: cross_rho, ..Default::default() }
        } else {
            // degenerate:端点率、实际事例数(自由度)、可屏蔽、系统偏代理
            let pre = packet_rate_and_n(&target.packets, gap.prev_pkt_idx);
            let post = packet_rate_and_n(&target.packets, gap.next_pkt_idx);
            let (r_pre, n_pre) = pre.map_or((None, None), |(r, n)| (Some(r), Some(n)));
            let (r_post, n_post) = post.map_or((None, None), |(r, n)| (Some(r), Some(n)));
            let maskable = pre.is_none() && post.is_none();
            // 评审④:退化外推偏差粗代理 = 端点率相对变化;地板段 ~100%;单侧变化未知→0
            let sys_bias_scale = match (r_pre, r_post) {
                (Some(a), Some(b)) if a + b > 0.0 => (b - a).abs() / (a + b),
                _ if maskable => 1.0,
                _ => 0.0,
            };
            GapCovBlock {
                r_pre, r_post, n_pre, n_post, maskable, sys_bias_scale,
                ..Default::default()
            }
        };

        results.push(ReconstructedGap {
            gap_idx,
            filled_events,
            n_lost,
            has_cross_ref: used_cross_ref,
            filler_weight,
            cov,
            bins: gap_bins,
        });
    }

    (results, ref_weights)
}

/// 退化(fallback)gap 的 gap 级方差 σ²_gap,由 pre/post 包率涨落传播:
/// Var(r)=r²/(n_events−1)。**评审②:率的分子与自由度都用各端点 packet 的实际
/// n_events**(reset 前后常是残包,≠名义 109),两端各用各的。两端有率 →
/// (T/2)²(r_pre²/(n_pre−1)+r_post²/(n_post−1)),单侧外推 → T²r²/(n−1)。方差最终挂在
/// gap 内的 filler 上(局域、不畸变相邻真实观测事件),每 filler v=√(σ²_gap/实际filler
/// 数),在分配后算(见调用处)。
/// (None,None):无相邻包率、只能用 MCU 地板率硬填,方差无从估 → 返回 NaN,
/// 由调用方按 ~100% 不确定(σ²=filler数²)处理,绝不给 0。
fn degenerate_gap_variance(
    packets: &[PacketInfo],
    gap: &SaturationInterval,
    gap_dur: f64,
) -> f64 {
    let dof = |n: f64| (n - 1.0).max(1.0);
    let r_pre = packet_rate_and_n(packets, gap.prev_pkt_idx);
    let r_post = packet_rate_and_n(packets, gap.next_pkt_idx);
    match (r_pre, r_post) {
        (Some((rp, np)), Some((rn, nn))) => {
            (gap_dur / 2.0).powi(2) * (rp * rp / dof(np) + rn * rn / dof(nn))
        }
        (Some((r, n)), None) | (None, Some((r, n))) => gap_dur.powi(2) * r * r / dof(n),
        (None, None) => f64::NAN,
    }
}

/// 从包的 span 估算事件率与实际事例数(自由度用)。评审②:分子用该包实际 n_events。
fn packet_rate_and_n(packets: &[PacketInfo], pkt_idx: usize) -> Option<(f64, f64)> {
    packets.iter().find(|p| p.pkt_idx == pkt_idx).and_then(|info| {
        let span = info.span();
        if span > 1e-9 {
            Some((info.n_events as f64 / span, info.n_events as f64))
        } else {
            None
        }
    })
}

/// 从包的 span 估算事件率。
fn packet_rate(packets: &[PacketInfo], pkt_idx: usize) -> Option<f64> {
    packet_rate_and_n(packets, pkt_idx).map(|(r, _)| r)
}

/// 无参考时的 fallback：用 pre/post-reset 包的率线性插值构建 shape。
fn fill_shape_fallback(
    shape: &mut [f64],
    gap: &SaturationInterval,
    packets: &[PacketInfo],
) {
    let r_pre = packet_rate(packets, gap.prev_pkt_idx);
    let r_post = packet_rate(packets, gap.next_pkt_idx);
    let n = shape.len();

    match (r_pre, r_post) {
        (Some(rp), Some(rn)) => {
            for (i, s) in shape.iter_mut().enumerate() {
                let t = (i as f64 + 0.5) / n as f64;
                *s = rp * (1.0 - t) + rn * t;
            }
        }
        (Some(r), None) | (None, Some(r)) => {
            shape.iter_mut().for_each(|v| *v = r);
        }
        (None, None) => {
            shape.iter_mut().for_each(|v| *v = MCU_READ_RATE_FLOOR);
        }
    }
}

/// 计算 target/ref 的事件率比值，用于将参考 box 的计数换算为 target box 的计数。
///
/// 在 gap 前后各 margin 秒的窗口内统计双方的事件率（events/有效秒），
/// 排除各自的 unreliable 区间。
/// 标定窗(gap 前后 ±margin)内 target/ref 的率比 k,连同两侧**事件计数**
/// (spec §5b 算 σ_k 用:(σ_k/k)²=1/C_a_cal+1/C_ref_cal)。返回 (k, C_a_cal, C_ref_cal);
/// 参考计数不足(≤10)或时长为 0 时 k=1.0(默认)。
fn calibrate_counts(
    target_events: &[f64],
    ref_events: &[f64],
    target_unreliable: &[UnreliableInterval],
    ref_unreliable: &[UnreliableInterval],
    gap_start: f64,
    gap_stop: f64,
    margin: f64,
) -> (f64, usize, usize) {
    let windows = [
        (gap_start - margin, gap_start),
        (gap_stop, gap_stop + margin),
    ];

    let mut target_count = 0usize;
    let mut ref_count = 0usize;
    let mut target_effective = 0.0f64;
    let mut ref_effective = 0.0f64;

    for &(win_lo, win_hi) in &windows {
        // target 侧：排除 target 的 unreliable 区间
        let t_eff = effective_duration(win_lo, win_hi, target_unreliable);
        if t_eff > 1e-6 {
            let a = target_events.partition_point(|&t| t < win_lo);
            let b = target_events.partition_point(|&t| t < win_hi);
            // 只计落在可信时段内的事件
            let cnt = target_events[a..b]
                .iter()
                .filter(|&&t| !is_in_unreliable(t, target_unreliable))
                .count();
            target_count += cnt;
            target_effective += t_eff;
        }

        // ref 侧：排除 ref 的 unreliable 区间
        let r_eff = effective_duration(win_lo, win_hi, ref_unreliable);
        if r_eff > 1e-6 {
            let a = ref_events.partition_point(|&t| t < win_lo);
            let b = ref_events.partition_point(|&t| t < win_hi);
            let cnt = ref_events[a..b]
                .iter()
                .filter(|&&t| !is_in_unreliable(t, ref_unreliable))
                .count();
            ref_count += cnt;
            ref_effective += r_eff;
        }
    }

    // 用事件率比值（而非事件数比值），补偿双方有效时长不同
    let k = if ref_effective > 1e-6 && ref_count > 10 && target_effective > 1e-6 {
        let target_rate = target_count as f64 / target_effective;
        let ref_rate = ref_count as f64 / ref_effective;
        target_rate / ref_rate
    } else {
        1.0
    };
    (k, target_count, ref_count)
}

/// 只要 k 的薄封装(bin 级 shape 构建用,不需要计数)。
fn calibrate_ratio_sorted(
    target_events: &[f64],
    ref_events: &[f64],
    target_unreliable: &[UnreliableInterval],
    ref_unreliable: &[UnreliableInterval],
    gap_start: f64,
    gap_stop: f64,
    margin: f64,
) -> f64 {
    calibrate_counts(
        target_events, ref_events, target_unreliable, ref_unreliable, gap_start,
        gap_stop, margin,
    )
    .0
}

/// 计算窗口 [lo, hi] 内排除 unreliable 区间后的有效时长。
fn effective_duration(lo: f64, hi: f64, unreliable: &[UnreliableInterval]) -> f64 {
    let mut excluded = 0.0;
    for iv in unreliable {
        let overlap_lo = iv.start.max(lo);
        let overlap_hi = iv.stop.min(hi);
        if overlap_hi > overlap_lo {
            excluded += overlap_hi - overlap_lo;
        }
    }
    (hi - lo - excluded).max(0.0)
}




/// 空 bin 插值：从最近的有值 bin 做线性插值，边缘用最近有值 bin 常数外推。
#[cfg(test)]
mod rate_window_tests {
    use super::*;

    fn even_stream(rate: f64, n: usize, t0: f64) -> Vec<f64> {
        (0..n).map(|i| t0 + i as f64 / rate).collect()
    }

    fn summary(times: &[f64]) -> PacketTimeSummary {
        PacketTimeSummary {
            pkt_idx: 0,
            min_met: times[0],
            max_met: times[times.len() - 1],
            n_events: times.len(),
            head_rate: edge_event_rate(&times[..RATE_WINDOW_EVENTS.min(times.len())]),
            tail_rate: edge_event_rate(&times[times.len().saturating_sub(RATE_WINDOW_EVENTS)..]),
        }
    }

    /// 均匀流上，边缘窗率就是那个率——(k−1)/T 的取法不能带偏置。
    #[test]
    fn uniform_stream_gives_its_own_rate() {
        let t = even_stream(4_000.0, 109, 0.0);
        let r = edge_event_rate(&t[..RATE_WINDOW_EVENTS]).unwrap();
        assert!((r - 4_000.0).abs() < 1.0, "edge rate {r}, expected 4000");
    }

    /// 判据缺陷的回归测试。一个包裹 109 个事例：64 个是 4 kHz 本底，45 个是
    /// 1.5 ms 内的 30 kHz 爆发，爆发压在包尾——照着 HEB 源窗里实测的洞前形态造的
    /// （1 ms 分辨下峰值 23–35 kHz）。整包平均率约 6 kHz，两道关卡按包级平均都
    /// 挡得住它；按洞旁边的事例算就挡不住。
    #[test]
    fn a_spike_at_the_edge_survives_packet_averaging() {
        let mut t = even_stream(4_000.0, 64, 0.0);
        let t0 = t[t.len() - 1];
        for i in 0..45 {
            t.push(t0 + (i as f64 + 1.0) / 30_000.0);
        }
        let s = summary(&t);
        let hole = 0.015;

        let packet_average = (s.max_met - s.min_met) / s.n_events as f64;
        assert!(
            hole < packet_average * GAP_FACTOR,
            "the packet average puts the bar at {} s, above the hole — that is the defect",
            packet_average * GAP_FACTOR
        );
        let rate = local_event_rate(&s, &s).unwrap();
        assert!(rate > 25_000.0, "the edge window has to see the spike, got {rate}");
        assert!(hole > GAP_FACTOR / rate, "and the hole has to clear the bar it sets");
    }

    /// 只看洞旁边：同样的爆发挪到包中间，边缘窗就不该看见它。
    /// 这是整包峰值写法放进中断段整秒空洞的那个毛病的单元版。
    #[test]
    fn a_spike_far_from_the_gap_is_not_local_rate() {
        let mut t = even_stream(4_000.0, 64, 0.0);
        let mid = t[32];
        for i in 0..45 {
            t.push(mid + (i as f64 + 1.0) / 30_000.0);
        }
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rate = local_event_rate(&summary(&t), &summary(&t)).unwrap();
        assert!(
            rate < 10_000.0,
            "a spike in the middle of the packet must not count as the rate at the gap, got {rate}"
        );
    }

    /// 事例数不足一个窗时退化为整段，不能直接返回 None——
    /// 包尾/包首的短包也要参与本地率估计。
    #[test]
    fn short_packet_degrades_to_whole_span() {
        let t = even_stream(20_000.0, 5, 0.0);
        let r = edge_event_rate(&t).unwrap();
        assert!((r - 20_000.0).abs() < 1.0, "edge rate {r}, expected 20000");
        assert!(edge_event_rate(&t[..1]).is_none(), "one event has no rate");
        assert!(edge_event_rate(&[]).is_none(), "no events, no rate");
    }

    /// 所有事例同一时刻（重建退化）不能产生无穷大率。
    #[test]
    fn zero_span_is_not_a_rate() {
        assert!(edge_event_rate(&[1.0; 20]).is_none());
    }

    /// 造一串包：每包 109 个事例，率 `rate`，从 `t0` 起连续铺，
    /// `skip` 给出要挖掉的包序号（模拟空洞）。
    /// `max_met` 的前缀最大值，和 `detect_fifo_reset_intervals` 里建的一样。
    fn reach_of(s: &[PacketTimeSummary]) -> Vec<f64> {
        let mut run = f64::NEG_INFINITY;
        s.iter().map(|p| { run = run.max(p.max_met); run }).collect()
    }

    fn packet_train(rate: f64, n_pkt: usize, t0: f64, skip: &[usize]) -> Vec<PacketTimeSummary> {
        let span = 108.0 / rate;
        (0..n_pkt)
            .filter(|i| !skip.contains(i))
            .map(|i| {
                let a = t0 + i as f64 * 109.0 / rate;
                summary(&even_stream(rate, 109, a))
            })
            .collect()
    }

    /// 均匀包流上，稳定率就是那个率。
    #[test]
    fn stable_rate_recovers_a_uniform_stream() {
        let s = packet_train(4_000.0, 60, 0.0, &[]);
        let center = s[30].max_met;
        let r = stable_event_rate(&s, &reach_of(&s), center).unwrap();
        assert!((r - 4_000.0).abs() < 60.0, "stable rate {r}, expected 4000");
    }

    /// 窗口里有空洞时，率要按**活时间**算。除以窗长会把率压低、λT 跟着压低，
    /// 那是漏检方向——这条守住那个除数。
    #[test]
    fn stable_rate_divides_by_live_time_not_wall_clock() {
        // 抠掉窗口中间 6 个包，约占窗口的四分之一
        let s = packet_train(4_000.0, 60, 0.0, &[31, 32, 33, 34, 35, 36]);
        let center = s[30].max_met;
        let r = stable_event_rate(&s, &reach_of(&s), center).unwrap();
        assert!(
            (r - 4_000.0).abs() < 200.0,
            "a hole inside the window must not drag the rate down, got {r}"
        );
    }

    /// 拥塞时会出现「宽包」：109 个事例横跨几百毫秒，压在几十个正常包之上。
    /// 分母若按逐包交叠求和，宽包每个白送近一整个窗宽却只带 109 个事例，率被
    /// 稀释；必须用被覆盖时间的**并集**。2026-01-25T19 实测稀释了约十倍。
    #[test]
    fn a_wide_packet_lying_across_the_window_must_not_dilute_the_rate() {
        let mut v = packet_train(36_000.0, 400, 0.0, &[]);
        let (lo, hi) = (v[0].min_met, v[v.len() - 1].max_met);
        // 一个横跨全窗的宽包，事例数和正常包一样
        v.push(PacketTimeSummary {
            pkt_idx: 9999,
            min_met: lo,
            max_met: hi,
            n_events: 109,
            head_rate: Some(109.0 / (hi - lo)),
            tail_rate: Some(109.0 / (hi - lo)),
        });
        v.sort_by(|a, b| a.min_met.partial_cmp(&b.min_met).unwrap());
        let center = (lo + hi) / 2.0;
        let r = stable_event_rate(&v, &reach_of(&v), center).unwrap();
        assert!(
            r > 30_000.0,
            "a single wide packet must not drag 36 kHz down to {r}"
        );
    }

    /// 宽包的 `max_met` 远在后面，按 `min_met` 排序后往前回退不能一见
    /// `max_met < lo` 就停——那样会把它漏掉，两个实现就会算出不同的率。
    #[test]
    fn the_walk_back_uses_the_prefix_reach_not_the_neighbour() {
        let mut v = vec![PacketTimeSummary {
            pkt_idx: 0, min_met: 0.0, max_met: 10.0, n_events: 109,
            head_rate: Some(10.9), tail_rate: Some(10.9),
        }];
        // 中间塞一批 max_met 都小于窗口下界的窄包
        v.extend(packet_train(4_000.0, 20, 0.1, &[]));
        v.sort_by(|a, b| a.min_met.partial_cmp(&b.min_met).unwrap());
        let reach = reach_of(&v);
        // 窗口落在宽包尾部、窄包之后
        let r = stable_event_rate(&v, &reach, 9.5);
        assert!(r.is_some(), "the wide packet still covers this window and must be found");
    }

    /// 包内单个事例的时间跳了一个 wrap：剔掉它，前后正常的一个不少。
    /// 按包号贪心用在这里会做反——它会接受尖刺、把后面正常的全剔掉。
    #[test]
    fn a_single_event_spiking_by_one_wrap_is_dropped() {
        const WRAP: f64 = 524_288.0 * 2e-6;
        let mut t = even_stream(36_000.0, 109, 100.0);
        let good = t.clone();
        t[104] += WRAP;
        let kept = fifo_ordered_prefix(&t);
        assert_eq!(kept.len(), 108, "exactly the spike goes, got {}", kept.len());
        assert!(
            (kept[kept.len() - 1] - good[108]).abs() < 1e-9,
            "the packet must end where it really ends, not a wrap later"
        );
        assert!(kept.windows(2).all(|w| w[0] <= w[1]));
    }

    /// 干净的包一个事例都不该剔。
    #[test]
    fn a_clean_packet_keeps_every_event() {
        let t = even_stream(36_000.0, 109, 0.0);
        assert_eq!(fifo_ordered_prefix(&t).len(), 109);
    }

    /// 时间与打包顺序一致时一个都不该剔。
    #[test]
    fn a_well_ordered_stream_loses_nothing() {
        let s = packet_train(4_000.0, 50, 0.0, &[]);
        assert!(fifo_consistent_mask(&s).iter().all(|k| *k));
    }

    /// 一小块包被判早了一个 wrap，落进前面的死区里：必须整块剔掉，
    /// 正确的那一支一个不少。照 2026-01-25T19 的形态造的——两支长度相当，
    /// 最长不减子序列在这种近似平局下会挑错边，按包号贪心不会。
    #[test]
    fn a_block_displaced_by_one_wrap_is_dropped() {
        const WRAP: f64 = 524_288.0 * 2e-6; // 1.048576 s
        let mut v = packet_train(36_000.0, 200, 0.0, &[]);
        let good = v.len();
        // 取靠后的一小块，整体前移一个 wrap —— 包号仍在后面，时间却跑到前面去了
        let mut bad = packet_train(36_000.0, 20, WRAP + 0.05, &[]);
        for b in &mut bad {
            b.min_met -= WRAP;
            b.max_met -= WRAP;
        }
        v.extend(bad);
        let keep = fifo_consistent_mask(&v);
        assert_eq!(keep.iter().filter(|k| **k).count(), good, "the good run must survive whole");
        assert!(keep[..good].iter().all(|k| *k), "and it must be the good run that survives");
        assert!(keep[good..].iter().all(|k| !*k), "the displaced block must go");
    }

    /// 窗口里一个包都没有时返回 None，由调用方回退到边缘率判据。
    #[test]
    fn stable_rate_is_none_when_the_window_is_empty() {
        let s = packet_train(4_000.0, 4, 0.0, &[]);
        assert!(stable_event_rate(&s, &reach_of(&s), 1_000.0).is_none());
    }

    /// 判据的分工：泊松那一关吃稳定率，FIFO 闸门吃边缘率。
    /// 数字取自 HEB211105190——洞 11.87 ms、稳定 λ 5203/s：λ̂T 不到 100 所以旧
    /// 判据拒了它，稳定 λT 是 61.8，远在 30 之上。
    #[test]
    fn the_poisson_test_reads_the_stable_rate() {
        let gap = 0.011_872;
        let stable = 5_203.0;
        let edge = 4_000.0; // 洞旁 16 个事例恰好偏稀，向下涨落
        assert!(
            gap * edge < GAP_FACTOR,
            "the old test on the edge rate rejects this hole — that is the miss"
        );
        assert!(
            gap * stable > MIN_POISSON_EXPONENT,
            "the stable rate has to clear the Poisson bar: {} vs {MIN_POISSON_EXPONENT}",
            gap * stable
        );
        // 噪声那一群的稳定 λT 实测最大 16.1，必须仍然拒
        assert!(16.1 < MIN_POISSON_EXPONENT, "the junk population must stay out");
    }
}

#[cfg(test)]
mod weight_tests {
    use super::*;

    fn si(lo: f64, hi: f64) -> SaturationInterval {
        SaturationInterval {
            start_met: lo,
            stop_met: hi,
            gap_seconds: hi - lo,
            prev_pkt_idx: 0,
            next_pkt_idx: 0,
            saturation_type: SaturationType::FifoReset,
        }
    }

    fn make_box(events: Vec<f64>, gaps: Vec<SaturationInterval>) -> BoxReconstructionData {
        let n = events.len();
        let unreliable = gaps
            .iter()
            .map(|g| UnreliableInterval { start: g.start_met, stop: g.stop_met })
            .collect();
        BoxReconstructionData {
            events,
            channels: vec![100u16; n],
            pulse_widths: vec![60u8; n],
            gaps,
            packets: Vec::new(),
            packet_events: Vec::new(),
            unreliable,
        }
    }

    fn spread(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| lo + (i as f64 + 0.5) * (hi - lo) / n as f64).collect()
    }

    /// 守恒:每个参考事件的权重贡献之和 == 总 filler 数。每个 filler 都完整
    /// 归回它的源参考事件,不漏不重,所以两者必须相等。
    #[test]
    fn ref_weight_contributions_sum_to_filler_count() {
        // 目标盒 A:仅在标定窗(gap 前后 0.5s)有事件,gap 内饱和、无事件
        let target = make_box(
            [spread(0.5, 1.0, 50), spread(1.1, 1.6, 50)].concat(),
            vec![si(1.0, 1.1)],
        );
        // 两个参考盒:标定窗 + gap 内密集(每盒 gap 内 200 事件 → 全覆盖)
        let ref_events = || {
            [spread(0.5, 1.0, 50), spread(1.0, 1.1, 200), spread(1.1, 1.6, 50)].concat()
        };
        let ref_b = make_box(ref_events(), vec![]);
        let ref_c = make_box(ref_events(), vec![]);
        let refs: Vec<&BoxReconstructionData> = vec![&ref_b, &ref_c];

        let (gaps, ref_weights) = reconstruct_gaps(&target, &refs);

        let n_lost_total: usize = gaps.iter().map(|g| g.n_lost).sum();
        let weight_sum: f64 = ref_weights.iter().flatten().sum();

        assert!(n_lost_total > 0, "should have filled events");
        assert!(
            (weight_sum - n_lost_total as f64).abs() < 1.0,
            "Σ ref weight contributions = {weight_sum}, expected filler count = {n_lost_total}"
        );
    }

    /// 守恒必须在**插值**(Λ>0)下也成立:参考只覆盖 gap 前半,后半靠外推。
    #[test]
    fn ref_weight_conserved_with_interpolation() {
        let target = make_box(
            [spread(0.5, 1.0, 50), spread(1.1, 1.6, 50)].concat(),
            vec![si(1.0, 1.1)],
        );
        // 参考只在 gap 前半 [1.0,1.05) 密集,后半 [1.05,1.1) 空 → 触发插值/外推
        let ref_events = || {
            [spread(0.5, 1.0, 50), spread(1.0, 1.05, 100), spread(1.1, 1.6, 50)].concat()
        };
        let ref_b = make_box(ref_events(), vec![]);
        let ref_c = make_box(ref_events(), vec![]);
        let refs: Vec<&BoxReconstructionData> = vec![&ref_b, &ref_c];

        let (gaps, ref_weights) = reconstruct_gaps(&target, &refs);
        let n_lost: usize = gaps.iter().map(|g| g.n_lost).sum();
        let wsum: f64 = ref_weights.iter().flatten().sum();

        assert!(n_lost > 0, "should have filled events");
        assert!(
            (wsum - n_lost as f64).abs() < 1.0,
            "插值下 Σcontrib = {wsum}, 应等于 n_lost = {n_lost}"
        );
    }

    /// 比守恒更强:验证权重落在**正确的事件**上、且**数值正确**。
    /// 单参考盒、k=1、满覆盖、无插值 → gap 内每个参考事件贡献恰为 ρ=1,
    /// gap 外(标定窗)事件贡献恰为 0。
    #[test]
    fn weight_lands_on_in_gap_refs_with_correct_value() {
        let target = make_box(
            [spread(0.5, 1.0, 50), spread(1.1, 1.6, 50)].concat(),
            vec![si(1.0, 1.1)],
        );
        // gap [1.0,1.1) 内密集 300 事件(每 bin 多个 → 无空 bin、无插值 Λ=0)
        // + 标定窗使 k=1。于是 gap 内每事件贡献 = ρ·k/n_m·(1+Λ) = 1·1/1·1 = 1
        let ref_ev =
            [spread(0.5, 1.0, 50), spread(1.0, 1.1, 300), spread(1.1, 1.6, 50)].concat();
        let ref_b = make_box(ref_ev.clone(), vec![]);
        let (_gaps, rw) = reconstruct_gaps(&target, &[&ref_b]);
        let w = &rw[0];

        for (i, &t) in ref_ev.iter().enumerate() {
            if (1.0..1.1).contains(&t) {
                // gap 内:贡献 = ρ·k/n_m·(1+Λ) = 1·1/1·1 = 1
                assert!(
                    (w[i] - 1.0).abs() < 1e-6,
                    "gap 内 ref[{i}] t={t}: 贡献={} 应=1.0",
                    w[i]
                );
            } else {
                // gap 外(标定窗)的参考事件不参与填充,贡献必须为 0
                assert!(
                    w[i].abs() < 1e-9,
                    "gap 外 ref[{i}] t={t}: 贡献={} 应=0",
                    w[i]
                );
            }
        }
    }

    /// 验证 1/n_m 分摊:两个满覆盖参考盒 → gap 内每个参考事件贡献恰为 0.5。
    #[test]
    fn two_references_halve_the_weight() {
        let target = make_box(
            [spread(0.5, 1.0, 50), spread(1.1, 1.6, 50)].concat(),
            vec![si(1.0, 1.1)],
        );
        let ref_ev =
            || [spread(0.5, 1.0, 50), spread(1.0, 1.1, 300), spread(1.1, 1.6, 50)].concat();
        let rb = make_box(ref_ev(), vec![]);
        let rc = make_box(ref_ev(), vec![]);
        let (_gaps, rw) = reconstruct_gaps(&target, &[&rb, &rc]);
        let ev = ref_ev();
        for (i, &t) in ev.iter().enumerate() {
            if (1.0..1.1).contains(&t) {
                assert!(
                    (rw[0][i] - 0.5).abs() < 1e-6 && (rw[1][i] - 0.5).abs() < 1e-6,
                    "n_m=2 时 gap 内贡献应=0.5, 得 {} / {}",
                    rw[0][i], rw[1][i]
                );
            }
        }
    }

    /// 退化(fallback):无参考盒 → 用 target 自己 pre/post 包率恢复。退化 gap 的
    /// 每个 filler 携带仅方差权重 v,Σv² == σ²_gap;方差挂 filler(gap 内)、不碰
    /// 相邻真实观测事件(ref_weights 全 0)。
    #[test]
    fn degenerate_gap_filler_weight_conserved() {
        // target 有 pre 包 [0.9,1.0)、post 包 [1.1,1.2),中间 gap [1.0,1.1)
        let events = [spread(0.9, 1.0, 109), spread(1.1, 1.2, 109)].concat();
        let mut target = make_box(events, vec![si(1.0, 1.1)]);
        target.packets = vec![
            PacketInfo { pkt_idx: 0, min_met: 0.9, max_met: 1.0, n_events: 109 },
            PacketInfo { pkt_idx: 1, min_met: 1.1, max_met: 1.2, n_events: 109 },
        ];
        target.gaps[0].prev_pkt_idx = 0;
        target.gaps[0].next_pkt_idx = 1;

        // 无参考盒 → NO-REF fallback
        let refs: Vec<&BoxReconstructionData> = vec![];
        let (gaps, _rw) = reconstruct_gaps(&target, &refs);
        let g = &gaps[0];
        assert!(g.n_lost > 0, "退化 gap 应产生 filler");
        assert!(g.filler_weight > 0.0, "退化 gap 的 filler 应带正方差权重");

        // σ²_gap = (T/2)²(r_pre²+r_post²)/(N-1)，r=109/0.1=1090，T=0.1，N-1=108
        let r = 1090.0_f64;
        let t = 0.1_f64;
        let sigma2 = (t / 2.0).powi(2) * (r * r + r * r) / 108.0;
        // Σv² 必须用**实际生成的 filler 数**(filled_events.len(),逐 bin round 后可
        // ≠ n_lost),否则下游按 filled_events 求和时不守恒。
        let sv2 = g.filled_events.len() as f64 * g.filler_weight * g.filler_weight;
        assert!(
            (sv2 - sigma2).abs() < sigma2 * 0.02,
            "Σv²(按实际 filler 数)={sv2} 应≈σ²_gap={sigma2}"
        );
    }

    /// 守护 k:标定系数必须真进入贡献。target 率=2×ref → k=2 → 每个 in-gap
    /// 参考事件 contrib = k/n_m = 2。删掉 `*k` 或写反 k 会让此测试变红。
    #[test]
    fn weight_uses_calibration_k() {
        let target = make_box(
            [spread(0.5, 1.0, 100), spread(1.1, 1.6, 100)].concat(),
            vec![si(1.0, 1.1)],
        );
        let ref_ev =
            [spread(0.5, 1.0, 50), spread(1.0, 1.1, 300), spread(1.1, 1.6, 50)].concat();
        let ref_b = make_box(ref_ev.clone(), vec![]);
        let (_g, rw) = reconstruct_gaps(&target, &[&ref_b]);
        for (i, &t) in ref_ev.iter().enumerate() {
            if (1.0..1.1).contains(&t) {
                assert!(
                    (rw[0][i] - 2.0).abs() < 1e-6,
                    "k=2 时 in-gap 贡献应=2,得 {}",
                    rw[0][i]
                );
            }
        }
    }

    /// 守护跨盒路由:只有盒 A(idx0) 有 gap,B/C 作参考。编排后 A 自己不该收到
    /// 任何贡献(weights[0] 全 1)——若 gj 映射错位、把 B/C 的贡献落到 A 会红。
    #[test]
    fn contributions_route_to_correct_box() {
        let a = make_box(
            [spread(0.5, 1.0, 50), spread(1.1, 1.6, 50)].concat(),
            vec![si(1.0, 1.1)],
        );
        let ref_ev =
            || [spread(0.5, 1.0, 50), spread(1.0, 1.1, 300), spread(1.1, 1.6, 50)].concat();
        let boxes = [a, make_box(ref_ev(), vec![]), make_box(ref_ev(), vec![])];

        let mut weights: Vec<Vec<f64>> =
            boxes.iter().map(|bx| vec![1.0f64; bx.events.len()]).collect();
        for i in 0..boxes.len() {
            let refs_with_idx: Vec<(usize, &BoxReconstructionData)> = boxes
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(j, d)| (j, d))
                .collect();
            let refs: Vec<&BoxReconstructionData> =
                refs_with_idx.iter().map(|(_, d)| *d).collect();
            let (_g, rw) = reconstruct_gaps(&boxes[i], &refs);
            for (r, (gj, _)) in refs_with_idx.iter().enumerate() {
                for (ev, &cc) in rw[r].iter().enumerate() {
                    weights[*gj][ev] += cc;
                }
            }
        }
        assert!(
            weights[0].iter().all(|&w| (w - 1.0).abs() < 1e-9),
            "盒 A 不应收到任何贡献,却有 w≠1"
        );
        assert!(weights[1].iter().any(|&w| w > 1.0001), "B 应有被参考的事件");
        assert!(weights[2].iter().any(|&w| w > 1.0001), "C 应有被参考的事件");
    }

    /// 守护防双计:cross-ref gap 的 filler_weight 必须为 0(方差在参考事件,
    /// 不能同时挂到 filler)。
    #[test]
    fn cross_ref_gap_has_zero_filler_weight() {
        let target = make_box(
            [spread(0.5, 1.0, 50), spread(1.1, 1.6, 50)].concat(),
            vec![si(1.0, 1.1)],
        );
        let ref_ev =
            [spread(0.5, 1.0, 50), spread(1.0, 1.1, 300), spread(1.1, 1.6, 50)].concat();
        let (gaps, _rw) = reconstruct_gaps(&target, &[&make_box(ref_ev, vec![])]);
        assert!(gaps[0].has_cross_ref, "该 gap 应判为 cross-ref");
        assert!(gaps[0].filler_weight.abs() < 1e-12, "cross-ref filler_weight 必须为 0");
    }

    /// Bug:有参考但覆盖<30% 的 gap 走退化路径,不该标 has_cross_ref=true。
    #[test]
    fn low_coverage_gap_not_marked_cross_ref() {
        let mut target = make_box(
            [spread(0.5, 1.0, 50), spread(1.1, 1.6, 50)].concat(),
            vec![si(1.0, 1.1)],
        );
        target.packets = vec![
            PacketInfo { pkt_idx: 0, min_met: 0.9, max_met: 1.0, n_events: 109 },
            PacketInfo { pkt_idx: 1, min_met: 1.1, max_met: 1.2, n_events: 109 },
        ];
        target.gaps[0].prev_pkt_idx = 0;
        target.gaps[0].next_pkt_idx = 1;
        // 参考只在 gap 前 ~20% [1.0,1.02) 有事件 → 覆盖<30% → 退化分支
        let ref_ev =
            [spread(0.5, 1.0, 50), spread(1.0, 1.02, 60), spread(1.1, 1.6, 50)].concat();
        let (gaps, _rw) = reconstruct_gaps(&target, &[&make_box(ref_ev, vec![])]);
        assert!(
            !gaps[0].has_cross_ref,
            "覆盖<30% 的退化 gap 不应标 has_cross_ref=true"
        );
    }

    /// Bug:两侧包都无有效率 (None,None) 时用 MCU 地板率造 filler,但方差不应为 0
    /// (最不确定的纯猜测填充反而零方差、被当精确测量)。
    #[test]
    fn no_packet_rate_degenerate_gap_gets_variance() {
        let mut target = make_box(
            [spread(0.5, 1.0, 50), spread(1.1, 1.6, 50)].concat(),
            vec![si(1.0, 1.1)],
        );
        // 包 span=0 → packet_rate 返回 None → (None,None)
        target.packets = vec![
            PacketInfo { pkt_idx: 0, min_met: 1.0, max_met: 1.0, n_events: 1 },
            PacketInfo { pkt_idx: 1, min_met: 1.0, max_met: 1.0, n_events: 1 },
        ];
        target.gaps[0].prev_pkt_idx = 0;
        target.gaps[0].next_pkt_idx = 1;
        let refs: Vec<&BoxReconstructionData> = vec![];
        let (gaps, _rw) = reconstruct_gaps(&target, &refs);
        assert!(gaps[0].n_lost > 0, "地板率仍会造 filler");
        assert!(
            gaps[0].filler_weight > 0.0,
            "(None,None) 地板填充的 filler 方差不应为 0"
        );
    }

    /// M1:形状分母必须是『该格可用(可信)参考盒数』,不是『该格非零盒数』——饱和(被
    /// 排除)和 count=0(可信但泊松涨落到 0)是两回事。两盒都可信,B 只在偶数 1ms 格、
    /// C 只在奇数格各 1 事件(错开)→ 每格 1 非零、2 可用。真率=1/2=0.5/格 → n_lost≈10;
    /// bug(÷非零盒=1)给 shape=1、n_lost≈20。标定窗 ±0.5s 内 target 与两 ref 同率 → k=1。
    #[test]
    fn shape_divides_by_available_boxes_not_nonzero() {
        let calib: Vec<f64> = [spread(0.5, 1.0, 500), spread(1.02, 1.52, 500)].concat();
        let target = make_box(calib.clone(), vec![si(1.0, 1.02)]); // gap [1.0,1.02)=20 格
        let mut b = calib.clone();
        for i in (0..20).step_by(2) {
            b.push(1.0 + (i as f64 + 0.5) * 0.001); // 偶数格
        }
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let mut c = calib.clone();
        for i in (1..20).step_by(2) {
            c.push(1.0 + (i as f64 + 0.5) * 0.001); // 奇数格
        }
        c.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let (gaps, _) =
            reconstruct_gaps(&target, &[&make_box(b, vec![]), &make_box(c, vec![])]);
        assert_eq!(gaps.len(), 1);
        assert!(
            (gaps[0].n_lost as f64 - 10.0).abs() <= 1.5,
            "n_lost={} 应≈10(÷可用盒数);bug 会给≈20",
            gaps[0].n_lost
        );
        // 回归锁(审查 M1-fix#1):每格 avail=2、nonzero=1 → measured n_m 必须=2(可用盒),
        // 非 1(非零盒)。钉死 detect.rs 的 `n_m: avail[si2]`,防它被悄悄回退成 bin_refs.len()。
        assert!(
            gaps[0]
                .bins
                .iter()
                .filter(|b| b.kind == GapBinKind::Measured)
                .all(|b| b.n_m == 2),
            "measured 格 n_m 应=可用盒数 2,不是非零盒数 1"
        );
    }

    /// 回归锁(审查 M1-fix#2):真 0 格(两可用盒该格恰好都 count=0,非共饱和)必须判
    /// **Measured**(n_m=可用盒数、shape 保持 0、不插值),而非旧码的 Empty 插值——
    /// 可信参考观测到 0 = 源真没来,不发明计数。钉死 interpolate 的 `|| avail>0` 分支。
    #[test]
    fn true_zero_cell_is_measured_not_empty() {
        let calib: Vec<f64> = [spread(0.5, 1.0, 500), spread(1.03, 1.53, 500)].concat();
        let target = make_box(calib.clone(), vec![si(1.0, 1.03)]); // 30 格
        // 两参考盒都可信、每格 1 事件,唯独格 15 两盒都无事件 → 真 0(非共饱和)
        let mk = || {
            let mut e = calib.clone();
            for i in 0..30 {
                if i != 15 {
                    e.push(1.0 + (i as f64 + 0.5) * 0.001);
                }
            }
            e.sort_by(|a, b| a.partial_cmp(b).unwrap());
            e
        };
        let (gaps, _) =
            reconstruct_gaps(&target, &[&make_box(mk(), vec![]), &make_box(mk(), vec![])]);
        let b15 = &gaps[0].bins[15];
        assert_eq!(b15.kind, GapBinKind::Measured, "真 0 格应 Measured(有可用参考),非 Empty");
        assert_eq!(b15.n_m, 2, "真 0 格 n_m=可用盒数=2");
        assert!(b15.left_bin.is_none() && b15.tau.is_none(), "真 0 格不插值(无端点)");
    }

    /// 评审②:率的分子必须是该 packet 的**实际事例数**,不是名义 109。
    /// 残包(reset 前后常见)只有 50 个事例、span=1s → 率=50/s 而非 109/s。
    #[test]
    fn packet_rate_uses_actual_n_events_not_109() {
        let packets =
            vec![PacketInfo { pkt_idx: 0, min_met: 0.0, max_met: 1.0, n_events: 50 }];
        let r = packet_rate(&packets, 0).expect("有 span 应给率");
        assert!((r - 50.0).abs() < 1e-9, "率应=n_events/span=50，得 {r}");
    }

    /// 评审②:退化方差的自由度必须用各端点 packet 的**实际 n_events−1**,不是
    /// 名义 108;且两端率各用各的 n_events(残包 pre=40、post=60)。
    #[test]
    fn degenerate_variance_uses_actual_dof_not_108() {
        let packets = vec![
            PacketInfo { pkt_idx: 0, min_met: 0.9, max_met: 1.0, n_events: 40 },
            PacketInfo { pkt_idx: 1, min_met: 1.1, max_met: 1.2, n_events: 60 },
        ];
        let mut gap = si(1.0, 1.1);
        gap.prev_pkt_idx = 0;
        gap.next_pkt_idx = 1;
        let gap_dur = 0.1_f64;
        let got = degenerate_gap_variance(&packets, &gap, gap_dur);
        // r_pre=40/0.1=400, r_post=60/0.1=600
        // σ²=(T/2)²(r_pre²/(n_pre−1)+r_post²/(n_post−1))=0.05²(400²/39+600²/59)
        let (rp, rn) = (400.0_f64, 600.0_f64);
        let expected = (gap_dur / 2.0).powi(2) * (rp * rp / 39.0 + rn * rn / 59.0);
        assert!(
            (got - expected).abs() < expected * 1e-9,
            "σ²_gap 应用实际自由度:得 {got}, 期望 {expected}"
        );
    }

    /// spec §13:退化 gap 的协方差块记录端点率、实际事例数、可屏蔽标志与系统偏
    /// 代理。r_pre=40/0.1=400、r_post=60/0.1=600、sys_bias_scale=|600−400|/1000=0.2。
    #[test]
    fn degenerate_gap_block_records_rates_and_sys_bias() {
        let events = [spread(0.9, 1.0, 40), spread(1.1, 1.2, 60)].concat();
        let mut target = make_box(events, vec![si(1.0, 1.1)]);
        target.packets = vec![
            PacketInfo { pkt_idx: 0, min_met: 0.9, max_met: 1.0, n_events: 40 },
            PacketInfo { pkt_idx: 1, min_met: 1.1, max_met: 1.2, n_events: 60 },
        ];
        target.gaps[0].prev_pkt_idx = 0;
        target.gaps[0].next_pkt_idx = 1;
        let refs: Vec<&BoxReconstructionData> = vec![];
        let (gaps, _rw) = reconstruct_gaps(&target, &refs);
        let cov = &gaps[0].cov;
        assert!((cov.r_pre.unwrap() - 400.0).abs() < 1e-6, "r_pre={:?}", cov.r_pre);
        assert!((cov.r_post.unwrap() - 600.0).abs() < 1e-6, "r_post={:?}", cov.r_post);
        assert!((cov.n_pre.unwrap() - 40.0).abs() < 1e-9);
        assert!((cov.n_post.unwrap() - 60.0).abs() < 1e-9);
        assert!(!cov.maskable, "有率的退化段不可屏蔽");
        assert!((cov.sys_bias_scale - 0.2).abs() < 1e-6, "sys_bias={}", cov.sys_bias_scale);
        assert!(cov.refs.is_empty(), "退化 gap 无 cross-ref 标定");
    }

    /// spec §13/§5b:cross-ref gap 的块记录每参考盒的 k 与标定窗计数(算 σ_k 用)。
    /// target 率=2×ref → k=2;标定窗 ±0.5s 内 target 200 事例、ref 100 事例。
    #[test]
    fn cross_ref_gap_block_records_k_and_calib_counts() {
        let target = make_box(
            [spread(0.5, 1.0, 100), spread(1.1, 1.6, 100)].concat(),
            vec![si(1.0, 1.1)],
        );
        let ref_ev =
            [spread(0.5, 1.0, 50), spread(1.0, 1.1, 300), spread(1.1, 1.6, 50)].concat();
        let ref_b = make_box(ref_ev, vec![]);
        let (gaps, _rw) = reconstruct_gaps(&target, &[&ref_b]);
        let cov = &gaps[0].cov;
        assert_eq!(cov.refs.len(), 1, "应有一个参考盒");
        let rc = &cov.refs[0];
        assert_eq!(rc.ref_idx, 0);
        assert!((rc.k - 2.0).abs() < 1e-6, "k 应=2,得 {}", rc.k);
        assert!((rc.c_a_cal - 200.0).abs() < 1e-9, "C_a_cal 应=200,得 {}", rc.c_a_cal);
        assert!((rc.c_ref_cal - 100.0).abs() < 1e-9, "C_ref_cal 应=100,得 {}", rc.c_ref_cal);
        assert!(cov.r_pre.is_none(), "cross-ref 段无退化端点率");
    }

    /// spec ③:interpolate_empty_bins 除 lambda 外还须逐格报告插值端点/τ。
    /// shape=[10,0,0,20] → bin1,bin2 为空,端点 l=0,r=3,τ=(i−l)/(r−l)。
    #[test]
    fn interp_reports_endpoints_and_tau() {
        let mut shape = vec![10.0, 0.0, 0.0, 20.0];
        let avail = vec![0usize; shape.len()]; // 无可用参考 → 全按真 empty 插值
        let (_lambda, interp) = interpolate_empty_bins(&mut shape, &avail);
        assert!(interp[0].is_none(), "measured 格 interp 应为 None");
        assert!(interp[3].is_none(), "measured 格 interp 应为 None");
        let (l1, r1, t1) = interp[1].unwrap();
        assert_eq!((l1, r1), (0, 3), "端点应为 (0,3)");
        assert!((t1 - 1.0 / 3.0).abs() < 1e-9, "τ 应=(1−0)/(3−0)=1/3,得 {t1}");
        let (l2, r2, t2) = interp[2].unwrap();
        assert_eq!((l2, r2), (0, 3));
        assert!((t2 - 2.0 / 3.0).abs() < 1e-9, "τ 应=2/3,得 {t2}");
    }

    /// spec ③:单侧外推的空格也记端点(left=right=唯一有值格),τ 取 0/1 使
    /// (1−τ)、τ 两支全落到同一端点、权重和=1。
    #[test]
    fn interp_extrapolation_endpoints() {
        // bin0 左空(None,Some(1)) → 右外推;bin2 右空(Some(1),None) → 左外推
        let mut shape = vec![0.0, 5.0, 0.0];
        let avail = vec![0usize; shape.len()];
        let (_lambda, interp) = interpolate_empty_bins(&mut shape, &avail);
        assert_eq!(interp[0].map(|(l, r, t)| (l, r, t)), Some((1, 1, 1.0)));
        assert_eq!(interp[2].map(|(l, r, t)| (l, r, t)), Some((1, 1, 0.0)));
    }

    /// spec ③:cross-ref gap 全覆盖 → 每 1ms 格 measured、n_m=参考盒数、端点留空;
    /// bins 长度=n_sbins;t_lo 从 gap_start 起。
    #[test]
    fn cross_ref_gap_populates_bin_structure() {
        let target = make_box(
            [spread(0.5, 1.0, 50), spread(1.1, 1.6, 50)].concat(),
            vec![si(1.0, 1.1)],
        );
        // 两参考盒,gap 内 400 事件/盒 → 每 1ms 格都有事件
        let ref_ev =
            || [spread(0.5, 1.0, 50), spread(1.0, 1.1, 400), spread(1.1, 1.6, 50)].concat();
        let rb = make_box(ref_ev(), vec![]);
        let rc = make_box(ref_ev(), vec![]);
        let (gaps, _rw) = reconstruct_gaps(&target, &[&rb, &rc]);
        let g = &gaps[0];
        // n_sbins = ceil(gap_dur/1ms).max(1);gap_dur 用 stop−start(浮点误差使
        // (1.1−1.0)/0.001≈100.0000009 → 101,勿硬编码 100)
        let gap_dur = 1.1_f64 - 1.0_f64;
        let n_sbins = ((gap_dur / SHAPE_BIN_WIDTH).ceil() as usize).max(1);
        assert_eq!(g.bins.len(), n_sbins, "格数应=n_sbins={n_sbins},得 {}", g.bins.len());
        assert!(
            g.bins.iter().all(|b| b.kind == GapBinKind::Measured),
            "全覆盖应全 measured"
        );
        assert!(g.bins.iter().all(|b| b.n_m == 2), "两参考盒 → n_m=2");
        assert!(
            g.bins.iter().all(|b| b.left_bin.is_none() && b.tau.is_none()),
            "measured 格端点/τ 应留空"
        );
        assert!((g.bins[0].t_lo - 1.0).abs() < 1e-9, "首格 t_lo=gap_start");
        assert_eq!(g.bins[0].bin_index, 0);
        assert_eq!(g.bins[n_sbins - 1].bin_index, n_sbins - 1);
        // bin_index 与 t_lo 单调递增
        assert!(g.bins.windows(2).all(|w| w[1].bin_index == w[0].bin_index + 1));
    }

    /// spec ③:cross-ref gap 后半无参考 → 那些格 kind=empty、n_m 不用、带插值端点/τ;
    /// measured 格反之。
    #[test]
    fn cross_ref_gap_marks_empty_bins() {
        let target = make_box(
            [spread(0.5, 1.0, 50), spread(1.1, 1.6, 50)].concat(),
            vec![si(1.0, 1.1)],
        );
        // 参考前半 [1.0,1.05) 有事件、可用;后半 [1.05,1.1) **饱和(unreliable)** → 后半
        // 无可用参考 → 共饱和 empty(真 empty)。注:参考『可用但 count=0』是真 0、不算 empty。
        let ref_ev =
            [spread(0.5, 1.0, 50), spread(1.0, 1.05, 300), spread(1.1, 1.6, 50)].concat();
        let (gaps, _rw) =
            reconstruct_gaps(&target, &[&make_box(ref_ev, vec![si(1.05, 1.1)])]);
        let g = &gaps[0];
        assert!(g.has_cross_ref, "覆盖≥30% 应判 cross-ref");
        let n_empty = g.bins.iter().filter(|b| b.kind == GapBinKind::Empty).count();
        assert!(n_empty > 0, "后半参考饱和 → 应出现共饱和空格");
        for b in g.bins.iter().filter(|b| b.kind == GapBinKind::Empty) {
            assert!(
                b.left_bin.is_some() && b.right_bin.is_some() && b.tau.is_some(),
                "空格必须带插值端点/τ"
            );
        }
        for b in g.bins.iter().filter(|b| b.kind == GapBinKind::Measured) {
            assert!(b.n_m >= 1, "measured 格 n_m≥1");
            assert!(b.left_bin.is_none() && b.tau.is_none(), "measured 格端点/τ 留空");
        }
    }

    /// spec ③/§7:cross-ref 块须记录重整因子 ρ=N_lost/Σshape。k=1 全覆盖 → ρ≈1。
    #[test]
    fn cross_ref_block_records_rho() {
        let target = make_box(
            [spread(0.5, 1.0, 50), spread(1.1, 1.6, 50)].concat(),
            vec![si(1.0, 1.1)],
        );
        let ref_ev =
            [spread(0.5, 1.0, 50), spread(1.0, 1.1, 400), spread(1.1, 1.6, 50)].concat();
        let (gaps, _rw) = reconstruct_gaps(&target, &[&make_box(ref_ev, vec![])]);
        let rho = gaps[0].cov.rho;
        assert!(rho > 0.0, "cross-ref 块应记录正的 ρ,得 {rho}");
        assert!((rho - 1.0).abs() < 0.1, "k=1 全覆盖 ρ 应≈1,得 {rho}");
    }

    /// degenerate gap 无 S filler↔参考结构 → bins 为空、ρ=0。
    #[test]
    fn degenerate_gap_has_empty_bins_and_zero_rho() {
        let events = [spread(0.9, 1.0, 109), spread(1.1, 1.2, 109)].concat();
        let mut target = make_box(events, vec![si(1.0, 1.1)]);
        target.packets = vec![
            PacketInfo { pkt_idx: 0, min_met: 0.9, max_met: 1.0, n_events: 109 },
            PacketInfo { pkt_idx: 1, min_met: 1.1, max_met: 1.2, n_events: 109 },
        ];
        target.gaps[0].prev_pkt_idx = 0;
        target.gaps[0].next_pkt_idx = 1;
        let refs: Vec<&BoxReconstructionData> = vec![];
        let (gaps, _rw) = reconstruct_gaps(&target, &refs);
        assert!(gaps[0].bins.is_empty(), "退化 gap 不产 S 格结构");
        assert!(gaps[0].cov.rho.abs() < 1e-12, "退化 gap ρ=0");
    }

    fn dense_except(lo: f64, hi: f64, per_sec: usize, gap: (f64, f64)) -> Vec<f64> {
        let n = ((hi - lo) * per_sec as f64) as usize;
        (0..n)
            .map(|i| lo + (i as f64 + 0.5) * (hi - lo) / n as f64)
            .filter(|&t| t < gap.0 || t >= gap.1)
            .collect()
    }

    /// 端到端守恒:三盒各有一个 gap、彼此互为参考。模拟编排层(每观测事件
    /// 初值 1、跨所有 target 累加贡献),验证 Σ(观测权重) == 总事件数(观测+填充)。
    /// 这多覆盖了"一个参考事件被多个盒的 gap 引用、贡献跨盒累加"这一层。
    #[test]
    fn total_evt_weight_equals_total_event_count() {
        let ga = (1.0, 1.1);
        let gb = (1.5, 1.6);
        let gc = (2.0, 2.1);
        let a = make_box(dense_except(0.5, 3.5, 2000, ga), vec![si(ga.0, ga.1)]);
        let b = make_box(dense_except(0.5, 3.5, 2000, gb), vec![si(gb.0, gb.1)]);
        let c = make_box(dense_except(0.5, 3.5, 2000, gc), vec![si(gc.0, gc.1)]);
        let boxes = [a, b, c];

        // 编排:每观测事件初值 1.0,累加各 target gap 传来的贡献
        let mut weights: Vec<Vec<f64>> =
            boxes.iter().map(|bx| vec![1.0f64; bx.events.len()]).collect();
        let mut total_fill = 0usize;
        for i in 0..boxes.len() {
            let refs_with_idx: Vec<(usize, &BoxReconstructionData)> = boxes
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(j, d)| (j, d))
                .collect();
            let refs: Vec<&BoxReconstructionData> =
                refs_with_idx.iter().map(|(_, d)| *d).collect();
            let (gaps, ref_w) = reconstruct_gaps(&boxes[i], &refs);
            total_fill += gaps.iter().map(|g| g.n_lost).sum::<usize>();
            for (r, (gj, _)) in refs_with_idx.iter().enumerate() {
                for (ev, &c) in ref_w[r].iter().enumerate() {
                    weights[*gj][ev] += c;
                }
            }
        }

        let total_obs: usize = boxes.iter().map(|bx| bx.events.len()).sum();
        let w_sum: f64 = weights.iter().flatten().sum();
        let expected = (total_obs + total_fill) as f64;
        assert!(total_fill > 0, "should have filled events");
        assert!(
            (w_sum - expected).abs() < 1.0,
            "Σ EVT weights = {w_sum}, expected total events (obs {total_obs} + fill {total_fill}) = {expected}"
        );
    }
}

/// 线性插值填补空 bin，并返回每个（原本有值的）端点 bin 被空 bin 借用的
/// 系数和 Λ_l：空 bin i 以 l 为左端点则 l 得 (1-t)，为右端点则得 t，外推
/// 则整份 1。用于把插值 bin 的 filler 权重反算回端点的参考事件。
fn interpolate_empty_bins(
    shape: &mut [f64],
    avail: &[usize],
) -> (Vec<f64>, Vec<Option<(usize, usize, f64)>>) {
    let n = shape.len();
    let mut lambda = vec![0.0f64; n];
    // spec ③:逐格插值端点/τ。None = 该格 measured(有值);Some((l,r,τ)) = 该格空,
    // 借端点 l/r 插值。单侧外推记 (e,e,0/1) 使权重全落到唯一端点。
    let mut interp: Vec<Option<(usize, usize, f64)>> = vec![None; n];
    if n == 0 {
        return (lambda, interp);
    }

    // 预计算每个位置左边和右边最近的有值 bin 索引
    let mut left_filled: Vec<Option<usize>> = vec![None; n];
    let mut right_filled: Vec<Option<usize>> = vec![None; n];

    let mut last = None;
    for i in 0..n {
        if shape[i] > 0.0 {
            last = Some(i);
        }
        left_filled[i] = last;
    }

    last = None;
    for i in (0..n).rev() {
        if shape[i] > 0.0 {
            last = Some(i);
        }
        right_filled[i] = last;
    }

    for i in 0..n {
        // 只插『真 empty』(无可用参考=共饱和);真 0(有可用参考、恰好 count=0)保持 0、不插。
        if shape[i] > 0.0 || avail[i] > 0 {
            continue;
        }
        match (left_filled[i], right_filled[i]) {
            (Some(l), Some(r)) => {
                // 两侧都有值：线性插值
                let t = (i - l) as f64 / (r - l) as f64;
                shape[i] = shape[l] * (1.0 - t) + shape[r] * t;
                lambda[l] += 1.0 - t;
                lambda[r] += t;
                interp[i] = Some((l, r, t));
            }
            (Some(l), None) => {   // 右侧无值：常数外推
                shape[i] = shape[l];
                lambda[l] += 1.0;
                interp[i] = Some((l, l, 0.0));
            }
            (None, Some(r)) => {   // 左侧无值：常数外推
                shape[i] = shape[r];
                lambda[r] += 1.0;
                interp[i] = Some((r, r, 1.0));
            }
            (None, None) => {}                          // 全空，不应到达此处
        }
    }
    (lambda, interp)
}
