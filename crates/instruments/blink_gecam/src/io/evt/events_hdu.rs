use blink_core::types::MissionElapsedTime;
use std::marker::PhantomData;

use crate::types::{Event, instrument::Satellite};

/// 一路 GRD 的事例表（`EVENTS01` .. `EVENTS25`）。
pub(super) struct EventsHdu<S: Satellite> {
    id: u8,
    time: Vec<f64>,
    pi: Vec<i16>,
    gain_type: Vec<u8>,
    dead_time: Vec<f32>,
    evt_type: Vec<u8>,
    flag: Vec<u8>,
    _satellite: PhantomData<S>,
}

impl<S: Satellite> EventsHdu<S> {
    /// `id` 从 1 数起，对应 HDU 名 `EVENTS01`、`EVENTS02` ……
    pub fn from_fptr(fptr: &mut fitsio::FitsFile, id: u8) -> Result<Self, fitsio::errors::Error> {
        let events = fptr.hdu(format!("EVENTS{id:02}").as_str())?;

        Ok(Self {
            id,
            time: events.read_col::<f64>(fptr, "TIME")?,
            pi: events.read_col::<i16>(fptr, "PI")?,
            gain_type: events.read_col::<u8>(fptr, "GAIN_TYPE")?,
            dead_time: events.read_col::<f32>(fptr, "DEAD_TIME")?,
            evt_type: events.read_col::<u8>(fptr, "EVT_TYPE")?,
            flag: events.read_col::<u8>(fptr, "FLAG")?,
            _satellite: PhantomData,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.time.len()
    }

    /// 本路事例表里的时间回跳。每路各自有序是 k 路归并的前提，也是搜索
    /// 窗长判据的前提。
    pub(super) fn time_reversals(&self) -> TimeReversals {
        self.time.windows(2).filter(|pair| pair[1] < pair[0]).fold(
            TimeReversals::default(),
            |acc, pair| TimeReversals {
                count: acc.count + 1,
                max_magnitude: acc.max_magnitude.max(pair[0] - pair[1]),
            },
        )
    }
}

/// 事例表内时间回跳的统计量，判据看最大幅度而不是处数：微米级的排序抖动
/// 排一下就没了，整段重复才是要整小时排除的。口径与 SVOM/GRM 一致。
#[derive(Default, Clone, Copy)]
pub struct TimeReversals {
    pub count: usize,
    pub max_magnitude: f64,
}

impl TimeReversals {
    pub fn merge(self, other: Self) -> Self {
        Self {
            count: self.count + other.count,
            max_magnitude: self.max_magnitude.max(other.max_magnitude),
        }
    }
}

impl<'a, S: Satellite> IntoIterator for &'a EventsHdu<S> {
    type Item = Event<S>;
    type IntoIter = EventsHduIterator<'a, S>;

    fn into_iter(self) -> Self::IntoIter {
        EventsHduIterator {
            hdu: self,
            index: 0,
        }
    }
}

pub struct EventsHduIterator<'a, S: Satellite> {
    hdu: &'a EventsHdu<S>,
    index: usize,
}

impl<S: Satellite> Iterator for EventsHduIterator<'_, S> {
    type Item = Event<S>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.hdu.time.len() {
            return None;
        }
        let event = Event {
            time: MissionElapsedTime::new(self.hdu.time[self.index]),
            channel: self.hdu.pi[self.index],
            detector_id: self.hdu.id,
            gain_type: self.hdu.gain_type[self.index],
            dead_time: self.hdu.dead_time[self.index],
            evt_type: self.hdu.evt_type[self.index],
            flag: self.hdu.flag[self.index],
        };
        self.index += 1;
        Some(event)
    }
}
