//! 带电粒子探测器（CPD）的逐小时事例流。
//!
//! CPD 是 GECAM 上专门用来认带电粒子的塑料闪烁体（A/B 8 路、C 2 路），
//! 计数率只有 GRD 的百分之几。天格和 SVOM 上没有这个东西，粒子穿越只能靠
//! 「多路同一时戳」这类代理判据去猜；GECAM 上可以直接看同一时刻 CPD 有没有
//! 一起响。
//!
//! 这里只做统计、不做否决：CPD 符合到什么程度才算带电粒子，得拿真候选定标，
//! 定标之前先把计数随候选记下来（见 `AcdCounts` 与本 crate 的 OPEN-QUESTIONS）。
//! 只存计数不存比例，是为了保住泊松误差、让阈值日后可以复议而不必重跑全量。

use blink_core::error::Error;

use crate::types::instrument::Satellite;
use std::marker::PhantomData;

/// 认「两路 CPD 同时着火」的时间窗。GECAM 的时间量化步是亚微秒量级，取
/// 10 µs 是为了容纳读出抖动而不至于把无关的本底事例算成符合：CPD 单路
/// 本底率约 100–200 c/s，10 µs 里偶然撞上的概率是 1e-3 量级。
const COINCIDENCE_SECONDS: f64 = 1e-5;

/// 正常事例的 `EVT_TYPE`。与 GRD 同一套编码：2 是量程溢出。
const NORMAL_EVT_TYPE: u8 = 1;

const MAX_DETECTORS: u8 = 99;

pub struct CpdFile<S: Satellite> {
    /// 各路合并后按时间排好的事例时刻。
    time: Vec<f64>,
    /// 与 `time` 平行的探头号，用来区分「一路响」和「多路同时响」。
    unit: Vec<u8>,
    detector_count: usize,
    _satellite: PhantomData<S>,
}

impl<S: Satellite> CpdFile<S> {
    pub fn from_fits_file(path: &str) -> Result<Self, Error> {
        let mut fptr = fitsio::FitsFile::open(path)?;

        let mut rows: Vec<(f64, u8)> = Vec::new();
        let mut detector_count = 0usize;
        for id in 1..=MAX_DETECTORS {
            let Ok(hdu) = fptr.hdu(format!("EVENTS{id:02}").as_str()) else {
                break;
            };
            detector_count += 1;
            let time = hdu.read_col::<f64>(&mut fptr, "TIME")?;
            let evt_type = hdu.read_col::<u8>(&mut fptr, "EVT_TYPE")?;
            rows.extend(
                time.into_iter()
                    .zip(evt_type)
                    .filter(|(_, evt_type)| *evt_type == NORMAL_EVT_TYPE)
                    .map(|(time, _)| (time, id)),
            );
        }
        if detector_count == 0 {
            return Err(Error::InvalidData(format!("{path} 里一个 EVENTS 表也没有")));
        }

        rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let (time, unit) = rows.into_iter().unzip();

        Ok(Self {
            time,
            unit,
            detector_count,
            _satellite: PhantomData,
        })
    }

    pub fn detector_count(&self) -> usize {
        self.detector_count
    }

    fn range(&self, start: f64, stop: f64) -> std::ops::Range<usize> {
        let lower = self.time.partition_point(|t| *t < start);
        let upper = self.time.partition_point(|t| *t <= stop);
        lower..upper
    }

    /// `[start, stop]` 内的 CPD 事例数。
    pub fn count_within(&self, start: f64, stop: f64) -> u32 {
        self.range(start, stop).len() as u32
    }

    /// `[start, stop]` 内、`COINCIDENCE_SECONDS` 之内还有另一路 CPD 一起响的
    /// 事例数。一路响可能是本底，多路同时响基本只能是穿过整台仪器的带电粒子。
    pub fn count_multi_within(&self, start: f64, stop: f64) -> u32 {
        let window = self.range(start, stop);
        window
            .clone()
            .filter(|index| {
                let (time, unit) = (self.time[*index], self.unit[*index]);
                // 往两边各扫一小段：时间有序，越界即停
                let backward = (window.start..*index)
                    .rev()
                    .take_while(|other| time - self.time[*other] <= COINCIDENCE_SECONDS)
                    .any(|other| self.unit[other] != unit);
                let forward = (*index + 1..window.end)
                    .take_while(|other| self.time[*other] - time <= COINCIDENCE_SECONDS)
                    .any(|other| self.unit[other] != unit);
                backward || forward
            })
            .count() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::instrument::SatB;

    fn cpd(rows: &[(f64, u8)]) -> CpdFile<SatB> {
        let mut rows = rows.to_vec();
        rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (time, unit) = rows.into_iter().unzip();
        CpdFile {
            time,
            unit,
            detector_count: 8,
            _satellite: PhantomData,
        }
    }

    #[test]
    fn counting_is_inclusive_at_both_ends() {
        let file = cpd(&[(1.0, 1), (2.0, 2), (3.0, 3)]);
        assert_eq!(file.count_within(1.0, 3.0), 3);
        assert_eq!(file.count_within(1.5, 2.5), 1);
        assert_eq!(file.count_within(4.0, 5.0), 0);
    }

    #[test]
    fn two_units_within_the_coincidence_window_both_count_as_multi() {
        let file = cpd(&[(10.0, 1), (10.0 + 1e-6, 4)]);
        assert_eq!(file.count_multi_within(9.0, 11.0), 2);
    }

    #[test]
    fn the_same_unit_firing_twice_is_not_a_coincidence() {
        // 一路自己连响两次是本底涨落，不是穿仪器的粒子
        let file = cpd(&[(10.0, 3), (10.0 + 1e-6, 3)]);
        assert_eq!(file.count_multi_within(9.0, 11.0), 0);
    }

    #[test]
    fn units_further_apart_than_the_window_are_not_a_coincidence() {
        let file = cpd(&[(10.0, 1), (10.0 + 1e-3, 4)]);
        assert_eq!(file.count_multi_within(9.0, 11.0), 0);
    }

    #[test]
    fn coincidences_are_only_counted_inside_the_window() {
        // 窗外的伙伴不算数：窗内那一个是孤立的
        let file = cpd(&[(9.999_999, 1), (10.0, 4)]);
        assert_eq!(file.count_multi_within(10.0, 11.0), 0);
        assert_eq!(file.count_multi_within(9.0, 11.0), 2);
    }

    #[test]
    fn an_empty_file_counts_zero() {
        let file = cpd(&[]);
        assert_eq!(file.count_within(0.0, 1.0), 0);
        assert_eq!(file.count_multi_within(0.0, 1.0), 0);
    }
}
