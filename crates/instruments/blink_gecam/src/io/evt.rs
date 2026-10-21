/*
GECAM-B: gbg_evt_220615_00_v00.fits
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PrimaryHDU    1 PrimaryHDU            ()
  1  EBOUNDS       1 BinTableHDU           498R x 3C
  2  GTI           1 BinTableHDU           1R x 2C
  3  EVENTS01      1 BinTableHDU      6407659R x 7C   [D, I, B, E, B, B, B]
  ...                                                 到 EVENTS25
GECAM-C 同一套结构，只是探头到 EVENTS12，EBOUNDS 470 道——2022-08-03 .. 10-15
那 1,121 小时是 896 道，见 `ebounds_hdu`。
*/

use blink_core::error::Error;
use std::{cmp::Reverse, collections::BinaryHeap};

mod ebounds_hdu;
mod events_hdu;
mod gti_hdu;

use ebounds_hdu::EboundsHdu;
pub use events_hdu::TimeReversals;
use events_hdu::{EventsHdu, EventsHduIterator};
use gti_hdu::GtiHdu;

use crate::types::event::{ChannelWindow, MIN_ENERGY_KEV};
use crate::types::{Event, instrument::Satellite};


pub struct EvtFile<S: Satellite> {
    gti: GtiHdu,
    detectors: Vec<EventsHdu<S>>,
    time_reversals: TimeReversals,
    /// 本文件的事例准入道号窗，由本文件的 EBOUNDS 现算。**不是编译期常量**
    /// ——归档里有两把能量梯，见 [`ebounds_hdu`]。
    channels: ChannelWindow,
}

/// 文件里各 HDU 的 EXTNAME（转成大写）；没有 EXTNAME 的（主 HDU）跳过。
fn extnames(
    fptr: &fitsio_pure::compat::fitsfile::FitsFile,
) -> Result<Vec<String>, fitsio_pure::compat::errors::Error> {
    let mut names = Vec::new();
    for index in 0..fptr.num_hdus()? {
        let hdu = fptr.hdu(index)?;
        if let Ok(name) = hdu.read_key::<String>(fptr, "EXTNAME") {
            names.push(name.trim().to_uppercase());
        }
    }
    Ok(names)
}

/// GRD 事例表的路数。表名必须是从 `EVENTS01` 起不断号、不重号的一段：中间
/// 缺一张就是产品出了问题，不能只读到断号处、把后面几路悄悄丢掉。
fn grd_table_count(extnames: &[String]) -> Result<u8, String> {
    let mut ids: Vec<u8> = extnames
        .iter()
        .filter_map(|name| name.strip_prefix("EVENTS"))
        .filter(|digits| digits.len() == 2)
        .filter_map(|digits| digits.parse().ok())
        .collect();
    ids.sort_unstable();
    if ids.is_empty() {
        return Err("一个 EVENTS 表也没有".to_string());
    }
    if ids.iter().zip(1..).any(|(&id, expected)| id != expected) {
        return Err(format!("EVENTS 表不是从 01 起连续的一段：{ids:?}"));
    }
    Ok(ids.len() as u8)
}

impl<S: Satellite> EvtFile<S> {
    pub fn from_fits_file(path: &str) -> Result<Self, Error> {
        let fptr = fitsio_pure::compat::fitsfile::FitsFile::open(path)?;

        // 能量梯必须是标定过的那把尺子（ch0 = 2.00 keV、梯内逐道连续、梯顶
        // 10053.5 keV），梯长与能阈道由这张表自己给出——**归档里有两把梯子，
        // 同一个道号在它们上面差一倍能量**，照搬道号会把能窗整个搬错位。
        let ebounds = EboundsHdu::from_fptr(&fptr)?;
        let ladder = ebounds
            .ladder(MIN_ENERGY_KEV)
            .map_err(|error| Error::InvalidData(format!("{path}: {error}")))?;
        let channels = ChannelWindow::new(ladder.min_channel, ladder.length);
        let gti = GtiHdu::from_fptr(&fptr)?;

        // 路数随星而异（A/B 25 路、C 12 路），由文件里的表名定；每一路都必须读得出来
        let count = grd_table_count(&extnames(&fptr)?)
            .map_err(|error| Error::InvalidData(format!("{path}: {error}")))?;
        let mut detectors = Vec::new();
        for id in 1..=count {
            let hdu = EventsHdu::from_fptr(&fptr, id)
                .map_err(|error| Error::InvalidData(format!("{path}: EVENTS{id:02}: {error}")))?;
            detectors.push(hdu);
        }

        let time_reversals = detectors
            .iter()
            .map(EventsHdu::time_reversals)
            .fold(TimeReversals::default(), TimeReversals::merge);

        Ok(Self {
            gti,
            detectors,
            time_reversals,
            channels,
        })
    }

    /// 本文件的事例准入道号窗。搜索一律用它，不要用编译期常量。
    pub fn channels(&self) -> ChannelWindow {
        self.channels
    }

    /// 本文件里的探头路数。
    pub fn detector_count(&self) -> usize {
        self.detectors.len()
    }

    /// 各路合计的事例数（未过滤）。
    pub fn len(&self) -> usize {
        self.detectors.iter().map(EventsHdu::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 本文件 GTI 与 `[from, to]` 的交集长度（秒）——曝光核算的分子。
    pub fn gti_seconds_within(&self, from: f64, to: f64) -> f64 {
        self.gti.seconds_within(from, to)
    }

    /// `time` 是否在本文件的 GTI 内。
    pub fn gti_contains(&self, time: f64) -> bool {
        self.gti.contains(time)
    }

    /// GTI 段 `(start, stop)`。
    pub fn gti_segments(&self) -> Vec<(f64, f64)> {
        self.gti
            .start
            .iter()
            .copied()
            .zip(self.gti.stop.iter().copied())
            .collect()
    }

    /// 各路合计的时间回跳统计。
    pub fn time_reversals(&self) -> TimeReversals {
        self.time_reversals
    }
}

impl<'a, S: Satellite> IntoIterator for &'a EvtFile<S> {
    type Item = Event<S>;
    type IntoIter = EvtFileIter<'a, S>;

    /// 各路按时间归并成一条流。路数是运行期才知道的，所以用堆而不是定长数组。
    fn into_iter(self) -> Self::IntoIter {
        let mut detector_iters: Vec<EventsHduIterator<'a, S>> =
            self.detectors.iter().map(|hdu| hdu.into_iter()).collect();
        let mut buffer = BinaryHeap::with_capacity(detector_iters.len());
        for (index, iter) in detector_iters.iter_mut().enumerate() {
            if let Some(event) = iter.next() {
                buffer.push(Reverse((event, index)));
            }
        }
        EvtFileIter {
            detector_iters,
            buffer,
        }
    }
}

pub struct EvtFileIter<'a, S: Satellite> {
    detector_iters: Vec<EventsHduIterator<'a, S>>,
    buffer: BinaryHeap<Reverse<(Event<S>, usize)>>,
}

impl<S: Satellite> Iterator for EvtFileIter<'_, S> {
    type Item = Event<S>;

    fn next(&mut self) -> Option<Self::Item> {
        let Reverse((event, index)) = self.buffer.pop()?;
        if let Some(next_event) = self.detector_iters[index].next() {
            self.buffer.push(Reverse((next_event, index)));
        }
        Some(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn table_count_is_the_unbroken_run_from_01() {
        let mut list = vec!["EBOUNDS", "GTI"];
        let events: Vec<String> = (1..=25).rev().map(|id| format!("EVENTS{id:02}")).collect();
        list.extend(events.iter().map(String::as_str));
        assert_eq!(grd_table_count(&names(&list)), Ok(25));
    }

    #[test]
    fn missing_or_repeated_table_is_an_error() {
        assert!(grd_table_count(&names(&["EVENTS01", "EVENTS02", "EVENTS04"])).is_err());
        assert!(grd_table_count(&names(&["EVENTS02", "EVENTS03"])).is_err());
        assert!(grd_table_count(&names(&["EVENTS01", "EVENTS01", "EVENTS02"])).is_err());
        assert!(grd_table_count(&names(&["EBOUNDS", "GTI"])).is_err());
    }

    #[test]
    fn other_names_are_not_detector_tables() {
        let list = names(&["EVENTS01", "EVENTS02", "EVENTS", "EVENTS003", "EVENTSXY"]);
        assert_eq!(grd_table_count(&list), Ok(2));
    }
}
