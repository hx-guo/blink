use blink_core::error::Error;
use blink_core::types::{Coverage, ExclusionReason, MissionElapsedTime};
use chrono::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::io::file::{find_cpd, find_grd, find_posatt};
use crate::io::{CpdFile, EvtFile, PosAttFile};
use crate::types::event::Event;
use crate::types::instrument::{Gecam, Satellite};

mod from_epoch;
mod search;

/// 判为「整段重复」的时间回跳幅度下限（秒），口径与 SVOM/GRM 一致：小于搜索
/// 窗长的回跳排序即可消化，大于它的说明同一段物理时间被记录了两次，会让窗长
/// 判据失效、计数虚涨，必须整小时排除。
pub const REVERSAL_THRESHOLD: f64 = 1e-3;

pub struct Chunk<S: Satellite> {
    pub span: [MissionElapsedTime<Gecam<S>>; 2],
    pub evt_file: EvtFile<S>,
    /// 带电粒子探测器。只统计不否决——判据还没定标，见 `search`。
    pub cpd_file: CpdFile<S>,
    pub posatt_file: PosAttFile,
    /// 峰值时刻取不到位置、只能丢掉的候选数，见 `search`。
    pub(super) dropped_no_ephemeris: AtomicUsize,
    /// 峰值时刻取不到姿态、姿态留空的候选数（位置有，候选照留）。
    pub(super) without_attitude: AtomicUsize,
    /// 候选窗里单路探测器占比过高（单路毛刺）而被否决的候选数。
    pub(super) dropped_single_detector: AtomicUsize,
    /// 落在 GTI 之外、搜索前就丢掉的事例数。
    pub(super) events_outside_gti: AtomicUsize,
}

impl<S: Satellite> blink_core::traits::Chunk for Chunk<S> {
    type Event = Event<S>;

    fn from_epoch(epoch: &DateTime<Utc>) -> Result<Self, Error> {
        from_epoch::from_epoch(epoch)
    }

    fn search(&self) -> Vec<blink_core::types::Signal<Self::Event>> {
        search::search(self)
    }

    fn exclusion(&self) -> Option<ExclusionReason> {
        // 大幅时间回跳＝同一段物理时间被记了两次。搜索假设输入有序，乱序会让
        // max_duration 判据失效，把本来只有十几个事例的窗报成几百计数。
        if self.evt_file.time_reversals().max_magnitude > REVERSAL_THRESHOLD {
            return Some(ExclusionReason::UnorderedEvents);
        }
        if self.evt_file.is_empty() {
            return Some(ExclusionReason::NoEvents);
        }
        None
    }

    /// 曝光走事例文件自带的 GTI。GECAM 的小时文件远不覆盖整点到整点（抽查
    /// GECAM-C 某小时 1763 s / 3700 s、GECAM-A 某小时 1303 s / 3600 s），
    /// 而且文件像 SVOM 一样越过整点，所以 GTI 要先截到本小时内再累加，
    /// 否则重叠段会被相邻两小时各记一次。
    fn coverage(&self) -> Coverage {
        let (start, stop) = (self.span[0].met(), self.span[1].met());
        let gti_seconds = self.evt_file.gti_seconds_within(start, stop);
        let span_seconds = stop - start;

        Coverage {
            span_seconds,
            masked_seconds: (span_seconds - gti_seconds).max(0.0),
        }
    }

    /// 在 `search` 之后取：几个计数器是搜索过程中才写入的。
    fn diagnostics(&self) -> Vec<(&'static str, f64)> {
        let reversals = self.evt_file.time_reversals();
        let mut diagnostics = vec![
            ("n_detectors", self.evt_file.detector_count() as f64),
            ("n_cpd_detectors", self.cpd_file.detector_count() as f64),
            ("n_events", self.evt_file.len() as f64),
            ("n_posatt_samples", self.posatt_file.len() as f64),
            ("time_reversals", reversals.count as f64),
            ("max_time_reversal", reversals.max_magnitude),
        ];
        for (name, counter) in [
            ("dropped_no_ephemeris", &self.dropped_no_ephemeris),
            ("without_attitude", &self.without_attitude),
            ("dropped_single_detector", &self.dropped_single_detector),
            ("events_outside_gti", &self.events_outside_gti),
        ] {
            let value = counter.load(Ordering::Relaxed);
            if value > 0 {
                diagnostics.push((name, value as f64));
            }
        }
        diagnostics
    }

    fn last_modified(epoch: &DateTime<Utc>) -> Result<DateTime<Utc>, Error> {
        let paths = [
            find_grd::<S>(epoch),
            find_cpd::<S>(epoch),
            find_posatt::<S>(epoch),
        ];

        paths
            .iter()
            .flatten()
            .map(|path| {
                let modified = std::fs::metadata(path)?.modified()?;
                Ok::<DateTime<Utc>, Error>(DateTime::<Utc>::from(modified))
            })
            .collect::<Result<Vec<_>, Error>>()?
            .into_iter()
            .max()
            .ok_or_else(|| Error::FileNotFound(format!("no GECAM files for {epoch}")))
    }
}
