//! 从事例流推活时间段（GTI）。
//!
//! # 为什么需要它
//!
//! `search_new` 的本底窗时长由 `live_length()` 按传进去的 GTI 算，缺口不计入分母。
//! 那套机制是对的、也有测试，**HXMT 的问题只是一直传 `[[span[0], span[1]]]`
//! ——整小时当一段活时间**。于是本底窗落在数据缺口上时，分子少、分母按墙钟不变，
//! `mean` 被压低、显著性做高。
//!
//! **只有"缺口在 chunk 内部"这一类需要修**：本底窗伸出 chunk 边界那一类，
//! `live_length` 早就夹对了（`live_start = window_start.max(seg[0].max(start))`）。
//! 实测三分类，"搜索的 `mean` ÷ 用审计本底率重算的值"：正常候选 1.000、
//! 整点边界 4.6（搜索已修）、内部缺口 1.03（搜索没修）。见 OPEN-QUESTIONS 第 34 条。
//!
//! # 判据
//!
//! 泊松过程里相邻事例间隔服从指数分布，中位间隔 = `μ ln 2`，`μ = 1/ρ`。
//! 把"一小时里误判出的假缺口条数"定成预算 `ε`，阈值就有闭式：
//!
//! ```text
//! g = μ · ln(3600 / (μ · ε))
//! ```
//!
//! `ε` 是唯一的旋钮，**有量纲意义**（每小时假缺口条数），不是调出来的数。
//! `ε = 1e-3` 时：ρ = 700 c/s ⇒ g ≈ 31 ms；ρ = 20000 c/s ⇒ g ≈ 1.25 ms。
//!
//! **阈值落在数据的空谷里，不是拟出来的。** 实测 7541 个候选的
//! `本底窗最大间隔 ÷ 泊松预期`：中位 1.11（主体正好坐在预期上 ⇒ 标尺是准的），
//! 75 分位 1.34，**90 分位直接跳到 33.8**。预算跨几个数量级选出的是同样一批。
//!
//! # `μ` 为什么用中位数估
//!
//! 用 `(t_last - t_first)/(n-1)` 或事例数除以窗长都会被缺口本身拉偏——缺口越大、
//! 估出的 `μ` 越大、阈值越高，**越大的缺口越查不出来**。中位数对单个大间隔免疫。

use crate::types::{Event, HxmtHe};
use blink_core::traits::Event as _;
use blink_core::types::MissionElapsedTime;
use uom::si::time::second;

/// 每小时允许误判出的假缺口条数。
pub const DEFAULT_FALSE_GAP_BUDGET: f64 = 1e-3;

/// 估计本底率用的窗长（秒）。要远大于要查的缺口（约 1 s），又要短到跟得上
/// 进出 SAA 时的速率变化。
const RATE_WINDOW_SECONDS: f64 = 10.0;

/// 一个窗里少于这么多事例就不估速率，沿用上一个窗的阈值。
const MIN_EVENTS_PER_WINDOW: usize = 32;

/// 从事例流推 GTI。事例必须按时间升序（`search_new` 也这么要求）。
///
/// 返回的段两两不重叠、按时间排好；第一段从 `span[0]` 起、最后一段到 `span[1]` 止
/// ——**除非首尾本身就是缺口**，那时按同一条判据切掉。
pub fn gti_from_events(
    events: &[Event],
    span: [MissionElapsedTime<HxmtHe>; 2],
    budget: f64,
) -> Vec<[MissionElapsedTime<HxmtHe>; 2]> {
    // 全部换算成相对 chunk 起点的秒。绝对 MET 约 1.6e8 s，f64 在那里的分辨率是
    // 35 ns；相对量最多 3600 s，分辨率 8e-13 s。阈值在毫秒量级，两者都够用，
    // 但相对量没有"大数相减"的隐患。
    let t0 = span[0];
    let times: Vec<f64> = events
        .iter()
        .map(|e| (e.time() - t0).get::<second>())
        .collect();
    let span_end = (span[1] - t0).get::<second>();

    gti_from_times(&times, span_end, budget)
        .into_iter()
        .map(|[a, b]| [shift(t0, a), shift(t0, b)])
        .collect()
}

/// 纯函数版：输入是相对 chunk 起点的秒，输出同样。判据全在这里，便于单测。
///
/// 空 GTI 会让所有候选的本底时长变成 0，所以任何退化情形都回落到整段。
pub fn gti_from_times(times: &[f64], span_end: f64, budget: f64) -> Vec<[f64; 2]> {
    let whole = vec![[0.0, span_end]];
    if times.len() < MIN_EVENTS_PER_WINDOW || !(span_end > 0.0) {
        return whole;
    }
    let thresholds = window_thresholds(times, span_end, budget);

    // 首尾也按同一条判据看：文件开头就没数据的话，那一截不是活时间。
    let mut cuts: Vec<(f64, f64)> = Vec::new();
    if times[0] > threshold_at(&thresholds, 0.0) {
        cuts.push((0.0, times[0]));
    }
    for i in 1..times.len() {
        let dt = times[i] - times[i - 1];
        if dt > threshold_at(&thresholds, times[i - 1]) {
            cuts.push((times[i - 1], times[i]));
        }
    }
    let last = times[times.len() - 1];
    if span_end - last > threshold_at(&thresholds, last) {
        cuts.push((last, span_end));
    }
    if cuts.is_empty() {
        return whole;
    }

    let mut segments: Vec<[f64; 2]> = Vec::with_capacity(cuts.len() + 1);
    let mut cursor = 0.0f64;
    for (gap_start, gap_stop) in cuts {
        if gap_start > cursor {
            segments.push([cursor, gap_start]);
        }
        cursor = gap_stop;
    }
    if span_end > cursor {
        segments.push([cursor, span_end]);
    }
    if segments.is_empty() { whole } else { segments }
}

fn shift(t0: MissionElapsedTime<HxmtHe>, dt: f64) -> MissionElapsedTime<HxmtHe> {
    MissionElapsedTime::new(t0.met() + dt)
}

/// 每个速率窗一个 `(窗起点, 阈值)`，按窗起点升序。
fn window_thresholds(t: &[f64], span_end: f64, budget: f64) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::new();
    let mut scratch: Vec<f64> = Vec::new();
    let mut i = 0usize;
    let mut w_start = 0.0f64;
    // 开头的窗若事例太少就不切——宁可漏也不要在没有依据的地方切掉活时间。
    let mut last = f64::INFINITY;
    while w_start < span_end {
        let w_stop = w_start + RATE_WINDOW_SECONDS;
        let begin = i;
        while i < t.len() && t[i] < w_stop {
            i += 1;
        }
        let th = if i.saturating_sub(begin) >= MIN_EVENTS_PER_WINDOW {
            scratch.clear();
            scratch.extend((begin + 1..i).map(|k| t[k] - t[k - 1]));
            let mid = scratch.len() / 2;
            scratch.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap());
            gap_threshold(scratch[mid] / std::f64::consts::LN_2, budget)
        } else {
            last
        };
        out.push((w_start, th));
        last = th;
        w_start = w_stop;
    }
    out
}

/// `g = μ · ln(3600 / (μ · ε))`。`μ · ε >= 3600` 时右边非正，
/// 表示这个速率下一小时里本来就期望出现这么长的间隔，不能判成缺口。
fn gap_threshold(mu: f64, budget: f64) -> f64 {
    if !(mu > 0.0) || !(budget > 0.0) {
        return f64::INFINITY;
    }
    let ratio = 3600.0 / (mu * budget);
    if ratio <= 1.0 {
        return f64::INFINITY;
    }
    mu * ratio.ln()
}

fn threshold_at(thresholds: &[(f64, f64)], at: f64) -> f64 {
    if thresholds.is_empty() {
        return f64::INFINITY;
    }
    let idx = thresholds.partition_point(|(start, _)| *start <= at);
    thresholds[idx.saturating_sub(1)].1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 固定间隔的流：中位间隔就是它本身，阈值可以手算。
    fn uniform(rate: f64, from: f64, to: f64) -> Vec<f64> {
        let step = 1.0 / rate;
        let n = ((to - from) / step).floor() as usize;
        (0..n).map(|k| from + step * k as f64).collect()
    }

    fn live(segments: &[[f64; 2]]) -> f64 {
        segments.iter().map(|s| s[1] - s[0]).sum()
    }

    #[test]
    fn clean_stream_gives_one_segment() {
        let t = uniform(1000.0, 0.0, 30.0);
        assert_eq!(
            gti_from_times(&t, 30.0, DEFAULT_FALSE_GAP_BUDGET),
            vec![[0.0, 30.0]],
            "干净的流不该被切开"
        );
    }

    /// 验收标准第 4 条的单元版：没有缺口时结果逐位等于整段，
    /// 也就是新旧口径在正常数据上完全一样。
    #[test]
    fn without_gaps_the_result_is_bit_identical_to_the_whole_span() {
        for rate in [200.0, 1000.0, 20000.0] {
            let t = uniform(rate, 0.0, 25.0);
            assert_eq!(
                gti_from_times(&t, 25.0, DEFAULT_FALSE_GAP_BUDGET),
                vec![[0.0, 25.0]],
                "rate={rate} 不该被切"
            );
        }
    }

    #[test]
    fn an_interior_gap_splits_the_span() {
        let mut t = uniform(1000.0, 0.0, 10.0);
        t.extend(uniform(1000.0, 11.0, 30.0)); // 中间挖掉 1 s
        let gti = gti_from_times(&t, 30.0, DEFAULT_FALSE_GAP_BUDGET);
        assert_eq!(gti.len(), 2, "内部缺口该切成两段：{gti:?}");
        assert!((gti[0][1] - 10.0).abs() < 0.01, "第一段该到缺口前：{gti:?}");
        assert!(
            (gti[1][0] - 11.0).abs() < 0.01,
            "第二段该从缺口后起：{gti:?}"
        );
        assert!(
            (live(&gti) - 29.0).abs() < 0.05,
            "活时间该是 30 - 1 = 29 s，实得 {}",
            live(&gti)
        );
    }

    #[test]
    fn a_gap_at_the_leading_edge_is_cut_too() {
        let t = uniform(1000.0, 2.0, 30.0); // 前 2 s 没有数据
        let gti = gti_from_times(&t, 30.0, DEFAULT_FALSE_GAP_BUDGET);
        assert_eq!(gti.len(), 1, "{gti:?}");
        assert!((gti[0][0] - 2.0).abs() < 0.01, "该从第一个事例起：{gti:?}");
    }

    #[test]
    fn a_gap_at_the_trailing_edge_is_cut_too() {
        let t = uniform(1000.0, 0.0, 28.0);
        let gti = gti_from_times(&t, 30.0, DEFAULT_FALSE_GAP_BUDGET);
        assert_eq!(gti.len(), 1, "{gti:?}");
        assert!(
            (gti[0][1] - 28.0).abs() < 0.01,
            "该到最后一个事例止：{gti:?}"
        );
    }

    /// 阈值必须随本底率缩放：低速率下 20 ms 是正常间隔，高速率下是缺口。
    #[test]
    fn the_threshold_scales_with_the_rate() {
        let low = gap_threshold(1.0 / 200.0, DEFAULT_FALSE_GAP_BUDGET);
        let high = gap_threshold(1.0 / 20000.0, DEFAULT_FALSE_GAP_BUDGET);
        assert!(low > 0.02, "200 c/s 时 20 ms 不该算缺口，阈值 {low}");
        assert!(high < 0.02, "20000 c/s 时 20 ms 该算缺口，阈值 {high}");
    }

    /// 判据对预算不敏感——实测那个"空谷"的单元版。
    #[test]
    fn the_answer_does_not_move_across_eight_decades_of_budget() {
        let mut t = uniform(1000.0, 0.0, 10.0);
        t.extend(uniform(1000.0, 10.5, 30.0)); // 0.5 s 缺口，约 500 倍中位间隔
        let lens: Vec<usize> = [1e-1, 1e-3, 1e-6, 1e-9]
            .iter()
            .map(|&b| gti_from_times(&t, 30.0, b).len())
            .collect();
        assert!(
            lens.iter().all(|&l| l == 2),
            "预算跨 8 个数量级结果该一样：{lens:?}"
        );
    }

    #[test]
    fn a_single_large_gap_is_still_found() {
        let mut t = uniform(1000.0, 0.0, 4.0);
        t.extend(uniform(1000.0, 6.0, 10.0)); // 2 s 缺口占掉窗内五分之一
        let gti = gti_from_times(&t, 10.0, DEFAULT_FALSE_GAP_BUDGET);
        assert_eq!(gti.len(), 2, "{gti:?}");
    }

    /// 中位数估 μ 的**实测灵敏度**：只有断断续续的流才分得出中位数与均值。
    ///
    /// 单个大缺口分不出来——手算过：4 s + 2 s 缺口 + 4 s 那个例子里，
    /// 均值估给阈值 27 ms、中位数估给 31 ms，而缺口是 2 s，两者都查得出。
    /// **所以上面那个测试证明不了"该用中位数"，它只证明"缺口查得出"。**
    ///
    /// 真正分得开的是数据被切成许多段的情形：30 段各 10 ms、间隔 0.3 s。
    /// 均值估的 μ = 31 ms ⇒ 阈值 0.577 s > 0.3 s ⇒ **一个缺口都查不出**；
    /// 中位数估的 μ = 1.44 ms ⇒ 阈值 31 ms ⇒ 全查得出。
    #[test]
    fn a_chopped_stream_is_where_the_median_beats_the_mean() {
        let mut t = Vec::new();
        for k in 0..30 {
            t.extend(uniform(1000.0, k as f64 * 0.31, k as f64 * 0.31 + 0.01));
        }
        let gti = gti_from_times(&t, 30.0 * 0.31, DEFAULT_FALSE_GAP_BUDGET);
        assert!(
            gti.len() >= 25,
            "断续流该被切成许多段，均值估只会给 1 段；实得 {} 段",
            gti.len()
        );
    }

    #[test]
    fn too_few_events_falls_back_to_the_whole_span() {
        let t = uniform(1.0, 0.0, 10.0); // 10 个事例
        assert_eq!(
            gti_from_times(&t, 10.0, DEFAULT_FALSE_GAP_BUDGET),
            vec![[0.0, 10.0]]
        );
    }

    #[test]
    fn segments_are_sorted_and_disjoint_and_never_empty() {
        let mut t = uniform(1000.0, 0.0, 5.0);
        t.extend(uniform(1000.0, 6.0, 12.0));
        t.extend(uniform(1000.0, 14.0, 20.0));
        let gti = gti_from_times(&t, 20.0, DEFAULT_FALSE_GAP_BUDGET);
        assert_eq!(gti.len(), 3, "{gti:?}");
        for w in gti.windows(2) {
            assert!(w[0][1] <= w[1][0], "段之间不该重叠：{gti:?}");
        }
        for s in &gti {
            assert!(s[0] < s[1], "段必须非空：{gti:?}");
        }
    }

    /// 速率在窗之间变化时阈值要跟着走：前 10 s 高速率、后 10 s 低速率，
    /// 后半段里 5 ms 的间隔是正常的，不该被前半段的阈值切掉。
    #[test]
    fn the_threshold_follows_a_rate_change() {
        let mut t = uniform(20000.0, 0.0, 10.0);
        t.extend(uniform(300.0, 10.0, 20.0));
        let gti = gti_from_times(&t, 20.0, DEFAULT_FALSE_GAP_BUDGET);
        assert_eq!(gti, vec![[0.0, 20.0]], "速率变化不是缺口：{gti:?}");
    }
}
