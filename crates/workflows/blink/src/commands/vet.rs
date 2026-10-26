//! 候选复核（管线 S2）：回到候选前后的事例流，补上搜索现场没算、但判选与
//! 展示要用的量。只加字段、不删候选；删不删由 `blink catalog` 决定。
//!
//! 逐候选给出：
//!
//! * **GRB 判据**（`blink_algorithms::extended_emission`）：候选之外 [−0.1, +0.3) s
//!   的延展辐射 `Z` 与最大单机箱占比。被判掉者 1 s 内的候选算同一个暴，那一步
//!   要看整张候选表，留给目录生成。
//! * **ACD**：符合计数（与搜索、`acd-audit` 同一实现）与顶/侧二项检验，标出
//!   电子束。只做标记，不进判选。
//! * **单探测器占比**：按整个候选窗算，与 `acd-audit` 同口径。
//! * **光变曲线**：候选窗外扩 5 ms 内的逐事例（相对峰值的微秒、探测器、道址、
//!   ACD 位），以及 [−2, +2) s 的 10 ms 计数，给网页事件页用。
//!
//! 两种用法：
//!
//! * `blink vet <FROM> <TO>`：读 `data/<slug>/YYYY/MM/YYYYMMDD_signals.json`，
//!   复核 `fa ≤ --max-fa` 的候选，写同目录 `YYYYMMDD_vet.json`（空表也写，
//!   表示这天复核过）。实时管线走这条。
//! * `blink vet --list <CSV> -o <CSV>`：任意带 `start`/`stop` 列的表，追加判据列，
//!   不写光变曲线。用来和冻结的目录（如 v9）对账。
//!
//! 判据要候选前后 2 s 的事例，候选靠近整点时会读相邻小时（跨天也一样）。
//! 相邻小时缺失时本底有缺口，`Z` 给 `None`——这种候选在实时管线里不能定为 final。
//!
//! 仪器侧只需实现 [`VetSource`]：事例载入、GRB 判据的分组、逐路编号，有 ACD 的
//! 再给电子束检验。现有 Insight-HXMT/HE（三机箱分组、ACD）与 Fermi/GBM（不分组、
//! 无 ACD）。

use crate::commands::acd_audit::parse_time;
use blink_algorithms::detector_share::max_detector_fraction;
use blink_algorithms::extended_emission::{
    ExtendedEmission, ExtendedEmissionConfig, extended_emission,
};
use blink_core::traits::{Chunk as _, Event as _, Instrument};
use blink_core::types::{AcdCounts, MissionElapsedTime, UnifiedSignal};
use blink_fermi_gbm::types::FermiGbm;
use blink_hxmt_he::algorithms::acd::{
    BEAM_P_THRESHOLD, acd_counts, acd_top_side, beam_probability,
};
use blink_hxmt_he::io::level_1k::EventFile;
use blink_hxmt_he::types::HxmtHe;
use chrono::TimeDelta;
use chrono::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// 复核要的仪器侧接口。
pub trait VetSource: Instrument {
    type Event: blink_core::traits::Event<Instrument = Self> + Clone;
    /// GRB 判据的分组数（单组时只看 `Z`，见 `extended_emission`）
    const N_GROUPS: usize;
    fn group(event: &Self::Event) -> usize;
    /// 逐路探测器编号：单路占比与光变曲线用
    fn unit(event: &Self::Event) -> u8;
    fn lc_channel(event: &Self::Event) -> u16;
    /// 光变曲线里的 ACD 着火位；没有 ACD 的仪器为 0
    fn lc_acd(_event: &Self::Event) -> u32 {
        0
    }
    /// 一小时内搜索所用的事例（准入之后），按时间排序
    fn hour_events(hour: &DateTime<Utc>) -> Result<Vec<Self::Event>, String>;
    /// 这一小时数据文件的最后修改时间（数据重发会更新它）
    fn hour_modified(hour: &DateTime<Utc>) -> Option<DateTime<Utc>>;
    /// ACD 符合计数与电子束检验；没有 ACD 的仪器为 `None`
    fn acd(_events: &[Self::Event], _start: f64, _stop: f64) -> Option<(AcdCounts, ElectronBeam)> {
        None
    }
}

/// HE 三个机箱各 6 路：Box A = 0..=5，B = 6..=11，C = 12..=17
/// （见 `blink_hxmt_he::algorithms::detectors::N_DETECTORS` 的实测说明）。
impl VetSource for HxmtHe {
    type Event = blink_hxmt_he::types::Event;
    const N_GROUPS: usize = 3;
    fn group(event: &Self::Event) -> usize {
        event.detector.id as usize / 6
    }
    fn unit(event: &Self::Event) -> u8 {
        event.detector.id
    }
    fn lc_channel(event: &Self::Event) -> u16 {
        event.channel()
    }
    fn lc_acd(event: &Self::Event) -> u32 {
        event
            .acds
            .iter()
            .enumerate()
            .fold(0u32, |bits, (j, &fired)| bits | ((fired as u32) << j))
    }
    fn hour_events(hour: &DateTime<Utc>) -> Result<Vec<Self::Event>, String> {
        let file = EventFile::from_epoch(hour).map_err(|e| e.to_string())?;
        let mut events: Vec<Self::Event> = file.into_iter().filter(|e| e.keep()).collect();
        events.sort_by_key(|e| e.time());
        Ok(events)
    }
    fn hour_modified(hour: &DateTime<Utc>) -> Option<DateTime<Utc>> {
        EventFile::last_modified(hour).ok()
    }
    fn acd(events: &[Self::Event], start: f64, stop: f64) -> Option<(AcdCounts, ElectronBeam)> {
        let top_side = acd_top_side(events, start, stop);
        let p = beam_probability(top_side);
        Some((
            acd_counts(events, start, stop),
            ElectronBeam {
                top_only: top_side.top_only,
                side_only: top_side.side_only,
                p,
                flagged: p < BEAM_P_THRESHOLD,
            },
        ))
    }
}

/// GBM 不分组：12 个 NaI 与 2 个 BGO 朝向各不相同，GRB 的超出本来就不均匀，
/// 分组占比判据挡的是 HXMT 单机箱的时间标签错，这里没有对应物。没有 ACD。
impl VetSource for FermiGbm {
    type Event = blink_fermi_gbm::types::Event;
    const N_GROUPS: usize = 1;
    fn group(_event: &Self::Event) -> usize {
        0
    }
    fn unit(event: &Self::Event) -> u8 {
        event.unit
    }
    fn lc_channel(event: &Self::Event) -> u16 {
        // keep() 之后道址在 1..=126
        event.channel as u16
    }
    fn hour_events(hour: &DateTime<Utc>) -> Result<Vec<Self::Event>, String> {
        let chunk = <FermiGbm as Instrument>::Chunk::from_epoch(hour).map_err(|e| e.to_string())?;
        if let Some(reason) = chunk.exclusion() {
            return Err(format!("excluded: {reason:?}"));
        }
        Ok(chunk.events_in_gti().0)
    }
    fn hour_modified(hour: &DateTime<Utc>) -> Option<DateTime<Utc>> {
        <FermiGbm as Instrument>::Chunk::last_modified(hour).ok()
    }
}

/// 光变曲线逐事例的范围：候选窗两侧各外扩这么多（秒）
const LC_PAD_S: f64 = 0.005;
/// 10 ms 计数覆盖 [−2, +2) s（相对起点），400 格
const CONTEXT_BIN_S: f64 = 0.01;
const CONTEXT_HALF_S: f64 = 2.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElectronBeam {
    pub top_only: u32,
    pub side_only: u32,
    /// 单边二项 `P`（参考比例 `TOP_SHARE_REFERENCE`）
    pub p: f64,
    pub flagged: bool,
}

/// 光变曲线里的一个事例。时间相对峰值，微秒。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LcEvent {
    pub t_us: f64,
    pub det: u8,
    pub channel: u16,
    /// 18 块 ACD 的着火位，第 j 位即第 j 块
    pub acd: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LightCurve {
    /// 峰值时刻（UTC），`events[].t_us` 的零点
    pub peak: DateTime<Utc>,
    pub events: Vec<LcEvent>,
    /// [−2, +2) s（相对起点）10 ms 计数
    pub context_bin_s: f64,
    pub context_start_s: f64,
    pub context: Vec<u32>,
}

/// `YYYYMMDD_vet.json` 的一行。`start` 与 signals 的 `start` 逐字节一致，是连接键。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vet {
    pub start: DateTime<Utc>,
    pub stop: DateTime<Utc>,
    pub instrument: String,
    pub false_positive_per_year: f64,
    pub grb: ExtendedEmission,
    pub det_share_max: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acd: Option<AcdCounts>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub electron_beam: Option<ElectronBeam>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light_curve: Option<LightCurve>,
}

/// 逐候选的判据，不含时间戳与光变曲线。
struct Verdict {
    grb: ExtendedEmission,
    det_share_max: f64,
    acd: Option<AcdCounts>,
    beam: Option<ElectronBeam>,
}

fn verdict<I: VetSource>(
    events: &[I::Event],
    start: f64,
    stop: f64,
    config: &ExtendedEmissionConfig,
) -> Verdict {
    let grb = extended_emission(events, start, stop, I::N_GROUPS, I::group, config);
    let (acd, beam) = I::acd(events, start, stop).unzip();
    Verdict {
        grb,
        det_share_max: max_detector_fraction(
            events,
            MissionElapsedTime::new(start),
            MissionElapsedTime::new(stop),
            I::unit,
        ),
        acd,
        beam,
    }
}

fn light_curve<I: VetSource>(events: &[I::Event], start: f64, stop: f64, peak: f64) -> LightCurve {
    let at = |t: f64| events.partition_point(|e| e.time().met() < t);
    let near = &events[at(start - LC_PAD_S)..at(stop + LC_PAD_S)];
    let lc_events = near
        .iter()
        .map(|e| LcEvent {
            t_us: (e.time().met() - peak) * 1e6,
            det: I::unit(e),
            channel: I::lc_channel(e),
            acd: I::lc_acd(e),
        })
        .collect();
    let n_bins = (2.0 * CONTEXT_HALF_S / CONTEXT_BIN_S).round() as usize;
    let mut context = vec![0u32; n_bins];
    for e in &events[at(start - CONTEXT_HALF_S)..at(start + CONTEXT_HALF_S)] {
        let k = ((e.time().met() - start + CONTEXT_HALF_S) / CONTEXT_BIN_S).floor() as usize;
        if let Some(slot) = context.get_mut(k) {
            *slot += 1;
        }
    }
    LightCurve {
        peak: MissionElapsedTime::<I>::new(peak).to_utc(),
        events: lc_events,
        context_bin_s: CONTEXT_BIN_S,
        context_start_s: -CONTEXT_HALF_S,
        context,
    }
}

fn epoch_hour<I: Instrument>(met: f64) -> DateTime<Utc> {
    let utc = MissionElapsedTime::<I>::new(met).to_utc();
    utc.date_naive()
        .and_hms_opt(utc.hour(), 0, 0)
        .expect("valid hour")
        .and_utc()
}

/// 按小时缓存 kept 事例（时间有序）。候选按时间顺序来，只留最近几个小时。
struct HourCache<I: VetSource> {
    hours: BTreeMap<DateTime<Utc>, Option<Vec<I::Event>>>,
    missing: Vec<DateTime<Utc>>,
}

impl<I: VetSource> Default for HourCache<I> {
    fn default() -> Self {
        Self {
            hours: BTreeMap::new(),
            missing: Vec::new(),
        }
    }
}

impl<I: VetSource> HourCache<I> {
    fn hour(&mut self, hour: DateTime<Utc>) -> Option<&Vec<I::Event>> {
        if !self.hours.contains_key(&hour) {
            // 候选按时间排好，比当前早两小时以上的不会再用到
            let stale = hour - TimeDelta::hours(2);
            self.hours.retain(|&h, _| h >= stale);
            let events = match I::hour_events(&hour) {
                Ok(events) => Some(events),
                Err(error) => {
                    eprintln!("vet: hour {} unavailable: {error}", hour.format("%Y-%m-%dT%H"));
                    self.missing.push(hour);
                    None
                }
            };
            self.hours.insert(hour, events);
        }
        self.hours[&hour].as_ref()
    }

    /// [lo, hi)（MET）内的事例，跨小时拼接；相邻小时文件首尾可能重叠，去重。
    fn events(&mut self, lo: f64, hi: f64) -> Vec<I::Event> {
        let mut out: Vec<I::Event> = Vec::new();
        let mut hour = epoch_hour::<I>(lo);
        let last = epoch_hour::<I>(hi);
        while hour <= last {
            if let Some(events) = self.hour(hour) {
                let a = events.partition_point(|e| e.time().met() < lo);
                let b = events.partition_point(|e| e.time().met() < hi);
                out.extend_from_slice(&events[a..b]);
            }
            hour += TimeDelta::hours(1);
        }
        out.sort_by_key(|e| e.time());
        out.dedup_by(|a, b| {
            a.time() == b.time() && I::unit(a) == I::unit(b) && I::lc_channel(a) == I::lc_channel(b)
        });
        out
    }
}

/// 一个候选所需的事例：判据范围与光变曲线范围的并集
fn load<I: VetSource>(
    cache: &mut HourCache<I>,
    start: f64,
    stop: f64,
    config: &ExtendedEmissionConfig,
) -> Vec<I::Event> {
    let (lo, hi) = config.span();
    let lo = (start + lo).min(start - CONTEXT_HALF_S).min(start - LC_PAD_S);
    let hi = (start + hi).max(start + CONTEXT_HALF_S).max(stop + LC_PAD_S);
    cache.events(lo, hi)
}

fn utc_to_met<I: Instrument>(utc: DateTime<Utc>) -> f64 {
    MissionElapsedTime::<I>::from(utc).met()
}

fn day_dir<I: Instrument>(day: NaiveDate) -> PathBuf {
    PathBuf::from(format!(
        "data/{}/{:04}/{:02}",
        I::name().replace('/', "_"),
        day.year(),
        day.month()
    ))
}

fn modified(path: &Path) -> Option<DateTime<Utc>> {
    fs::metadata(path).and_then(|m| m.modified()).ok().map(DateTime::<Utc>::from)
}

#[derive(Default)]
struct Totals {
    days: usize,
    skipped_fresh: usize,
    no_signals: usize,
    candidates: usize,
    no_z: usize,
    grb: usize,
    beams: usize,
}

/// 复核一天，计数记进 `totals`。没有 signals 文件或输出已是最新时不写。
fn vet_day<I: VetSource>(
    day: NaiveDate,
    max_fa: f64,
    force: bool,
    config: &ExtendedEmissionConfig,
    cache: &mut HourCache<I>,
    totals: &mut Totals,
) {
    let dir = day_dir::<I>(day);
    let stem = day.format("%Y%m%d").to_string();
    let signals_path = dir.join(format!("{stem}_signals.json"));
    let vet_path = dir.join(format!("{stem}_vet.json"));
    let Ok(content) = fs::read_to_string(&signals_path) else {
        totals.no_signals += 1;
        return;
    };
    let signals: Vec<UnifiedSignal> =
        serde_json::from_str(&content).expect("failed to parse signals json");
    let mut selected: Vec<&UnifiedSignal> = signals
        .iter()
        .filter(|s| s.false_positive_per_year <= max_fa)
        .collect();
    selected.sort_by_key(|s| s.start);

    if !force && let Some(vetted) = modified(&vet_path) {
        // 输出须不旧于候选表与所用的每个小时数据文件（数据重发会更新它们）
        let mut sources = vec![modified(&signals_path)];
        for s in &selected {
            let met = utc_to_met::<I>(s.start);
            for dh in [-1, 0, 1] {
                let hour = epoch_hour::<I>(met) + TimeDelta::hours(dh);
                sources.push(I::hour_modified(&hour));
            }
        }
        if sources.into_iter().flatten().all(|t| t <= vetted) {
            totals.skipped_fresh += 1;
            return;
        }
    }

    let mut out = Vec::with_capacity(selected.len());
    for signal in selected {
        let start = utc_to_met::<I>(signal.start);
        let stop = utc_to_met::<I>(signal.stop);
        let peak = utc_to_met::<I>(signal.peak_time());
        let events = load(cache, start, stop, config);
        let v = verdict::<I>(&events, start, stop, config);
        totals.candidates += 1;
        totals.no_z += v.grb.z.is_none() as usize;
        totals.grb += v.grb.flagged as usize;
        totals.beams += v.beam.as_ref().is_some_and(|b| b.flagged) as usize;
        out.push(Vet {
            start: signal.start,
            stop: signal.stop,
            instrument: signal.instrument.clone(),
            false_positive_per_year: signal.false_positive_per_year,
            grb: v.grb,
            det_share_max: v.det_share_max,
            acd: v.acd,
            electron_beam: v.beam,
            light_curve: Some(light_curve::<I>(&events, start, stop, peak)),
        });
    }

    let json = serde_json::to_string(&out).expect("failed to serialize vet");
    let temp = vet_path.with_extension(format!("json.{}.tmp", std::process::id()));
    fs::write(&temp, json).expect("failed to write vet json");
    fs::rename(&temp, &vet_path).expect("failed to rename vet json");
    totals.days += 1;
}

pub fn cmd_vet_days<I: VetSource>(
    from: NaiveDate,
    to: NaiveDate,
    workers: usize,
    worker: usize,
    max_fa: f64,
    force: bool,
) {
    let config = ExtendedEmissionConfig::default();
    let mut cache = HourCache::<I>::default();
    let mut totals = Totals::default();
    for (i, day) in from.iter_days().take_while(|d| *d <= to).enumerate() {
        if i % workers == worker {
            vet_day::<I>(day, max_fa, force, &config, &mut cache, &mut totals);
        }
    }
    eprintln!(
        "vet {} {from} .. {to} (worker {worker}/{workers}, fa <= {max_fa:e}): \
         {} days written, {} up to date, {} without signals; \
         {} candidates: {} without Z (background gap), {} GRB-flagged, {} electron beams; \
         {} hours unavailable",
        I::name(),
        totals.days,
        totals.skipped_fresh,
        totals.no_signals,
        totals.candidates,
        totals.no_z,
        totals.grb,
        totals.beams,
        cache.missing.len(),
    );
}

/// 只对 HXMT：用来与冻结的 HXMT 目录对账。
pub fn cmd_vet_list(list: &Path, out: &Path) {
    let config = ExtendedEmissionConfig::default();
    let content = fs::read_to_string(list).expect("failed to read input csv");
    let mut lines = content.lines();
    let header = lines.next().expect("empty input csv");
    let columns: Vec<&str> = header.split(',').collect();
    let find = |name: &str| {
        columns
            .iter()
            .position(|c| c.trim() == name)
            .unwrap_or_else(|| panic!("input csv has no `{name}` column"))
    };
    let (i_start, i_stop) = (find("start"), find("stop"));

    let rows: Vec<(&str, Option<(f64, f64)>)> = lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            let window = match (fields.get(i_start), fields.get(i_stop)) {
                (Some(a), Some(b)) => parse_time(a).zip(parse_time(b)),
                _ => None,
            };
            (line, window)
        })
        .collect();

    // 按时间顺序处理，小时缓存才有效；输出保持输入行序
    let mut order: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].1.is_some()).collect();
    order.sort_by(|&a, &b| rows[a].1.unwrap().0.total_cmp(&rows[b].1.unwrap().0));
    let mut cache = HourCache::<HxmtHe>::default();
    let mut verdicts: Vec<Option<Verdict>> = (0..rows.len()).map(|_| None).collect();
    for (k, &i) in order.iter().enumerate() {
        let (start, stop) = rows[i].1.unwrap();
        let events = load(&mut cache, start, stop, &config);
        verdicts[i] = Some(verdict::<HxmtHe>(&events, start, stop, &config));
        if (k + 1) % 500 == 0 {
            eprintln!("vet: {}/{} rows", k + 1, order.len());
        }
    }

    let mut output = String::with_capacity(content.len() * 2);
    output.push_str(header);
    output.push_str(
        ",grb_z,grb_observed,grb_expected,grb_max_box_share,grb_flag,\
         acd_top_only,acd_side_only,beam_p,electron_beam,det_share_max\n",
    );
    let (mut n_grb, mut n_beam, mut n_no_z, mut n_unresolved) = (0, 0, 0, 0);
    for ((line, _), v) in rows.iter().zip(&verdicts) {
        output.push_str(line);
        match v {
            Some(v) => {
                n_grb += v.grb.flagged as usize;
                let beam = v.beam.as_ref().expect("HXMT has ACD");
                n_beam += beam.flagged as usize;
                n_no_z += v.grb.z.is_none() as usize;
                output.push_str(&format!(
                    ",{},{},{:.3},{:.3},{},{},{},{:.3e},{},{:.3}\n",
                    v.grb.z.map(|z| format!("{z:.2}")).unwrap_or_default(),
                    v.grb.observed,
                    v.grb.expected,
                    v.grb.max_group_share,
                    v.grb.flagged as u8,
                    beam.top_only,
                    beam.side_only,
                    beam.p,
                    beam.flagged as u8,
                    v.det_share_max,
                ));
            }
            None => {
                n_unresolved += 1;
                output.push_str(",,,,,,,,,,\n");
            }
        }
    }
    fs::write(out, output).expect("failed to write output csv");
    eprintln!(
        "vet: {} rows, {} unparsable; {} without Z, {} GRB-flagged, {} electron beams; \
         {} hours unavailable -> {}",
        rows.len(),
        n_unresolved,
        n_no_z,
        n_grb,
        n_beam,
        cache.missing.len(),
        out.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use blink_hxmt_he::types::{Detector, Event, Scintillator};

    fn event(met: f64, det: u8, acds: &[usize]) -> Event {
        let mut bits = [false; 18];
        acds.iter().for_each(|&j| bits[j] = true);
        Event::new(
            MissionElapsedTime::new(met),
            100,
            Detector {
                id: det,
                scintillator: Scintillator::Csi,
            },
            false,
            bits,
        )
    }

    /// 1e4/s 平本底，18 路轮流
    fn background(lo: f64, hi: f64) -> Vec<Event> {
        let n = ((hi - lo) * 1e4) as usize;
        (0..n)
            .map(|i| event(lo + (i as f64 + 0.5) * 1e-4, (i % 18) as u8, &[]))
            .collect()
    }

    fn sorted(mut events: Vec<Event>) -> Vec<Event> {
        events.sort_by_key(|e| e.time());
        events
    }

    #[test]
    fn a_tgf_is_neither_grb_nor_beam() {
        let start = 1000.0;
        let mut events = background(start - 3.0, start + 3.0);
        // 30 个事例的 TGF，ACD 只零星着火
        events.extend((0..30).map(|i| {
            let acds: &[usize] = if i % 10 == 0 { &[4] } else { &[] };
            event(start + 1e-5 * i as f64, (i % 18) as u8, acds)
        }));
        let events = sorted(events);
        let stop = start + 29e-5;
        let v = verdict::<HxmtHe>(&events, start, stop, &Default::default());
        assert!(v.grb.z.is_some() && !v.grb.flagged);
        assert!(!v.beam.as_ref().unwrap().flagged);
        assert!(v.det_share_max < 0.2);
        let lc = light_curve::<HxmtHe>(&events, start, stop, start + 15e-5);
        // 窗内 30 个 + 本底；时间相对峰值
        assert!(lc.events.len() >= 30);
        assert!(lc.events.iter().all(|e| e.t_us.abs() < 5.3e3));
        assert_eq!(lc.context.len(), 400);
        assert!(lc.context[200] > lc.context[100]);
    }

    #[test]
    fn a_beam_from_the_top_is_flagged() {
        let start = 1000.0;
        let mut events = background(start - 3.0, start + 3.0);
        // 80 个事例，几乎全部只有顶部 ACD 着火
        events.extend((0..80).map(|i| {
            let acds: &[usize] = if i % 20 == 0 { &[0] } else { &[4, 10] };
            event(start + 5e-5 * i as f64, (i % 18) as u8, acds)
        }));
        let events = sorted(events);
        let v = verdict::<HxmtHe>(&events, start, start + 79.0 * 5e-5, &Default::default());
        let beam = v.beam.as_ref().unwrap();
        assert!(beam.flagged, "p = {}", beam.p);
        assert_eq!(beam.top_only, 76);
        assert_eq!(beam.side_only, 4);
    }

    #[test]
    fn a_burst_lasting_beyond_the_candidate_is_flagged_as_grb() {
        let start = 1000.0;
        let mut events = background(start - 3.0, start + 3.0);
        // 起点后 200 ms 的延展辐射，各机箱都有
        events.extend((0..2000).map(|i| event(start + 0.01 + 1e-4 * i as f64, (i % 18) as u8, &[])));
        let events = sorted(events);
        let v = verdict::<HxmtHe>(&events, start, start + 3e-4, &Default::default());
        assert!(v.grb.flagged);
    }
}
