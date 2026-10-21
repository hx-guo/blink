//! 曝光统计（管线 S5）：搜索实际用到的活时间沿轨道的分布，给率估计当分母。
//!
//! 每天一个 `YYYYMMDD_exposure.json`，与 `_signals.json`、`_hours.json` 同目录。
//! 活时间由仪器给（`Chunk::exposure_samples`，与搜索同一口径），这里只按格累加秒数：
//!
//! * `map`：`[纬度格, 经度格, 秒]`，纬度格 = ⌊lat + 90⌋（0..179），经度格 = ⌊lon mod 360⌋（0..359）
//! * `lst`：`[纬度格, 视太阳时格, 秒]`，视太阳时格 = ⌊地方视太阳时 × 2⌋（0..47，半小时）
//!
//! 月份、季节、正午太阳天顶角都由日期加纬度定，按天的 `map` 已经够推，不另存。
//! 取不到星历的活时间计入 `live_seconds` 与 `no_ephemeris_seconds`，不进格子。
//!
//! 两处产出：`blink search` 搜一天时顺带写；`blink exposure` 给已有的存档补算
//! （不重跑搜索，载入同样的数据、走同样的体检）。

use blink_core::traits::{Chunk, Instrument};
use blink_core::types::{ExposureSample, MissionElapsedTime};
use chrono::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// 沿轨道的采样步长（秒）。7.6 km/s 下 1 s 走 7.6 km，远小于 1° 格。
pub const STEP_SECONDS: f64 = 1.0;

#[derive(Default)]
pub struct Grid {
    pub live_seconds: f64,
    pub no_ephemeris_seconds: f64,
    map: BTreeMap<(u16, u16), f64>,
    lst: BTreeMap<(u16, u16), f64>,
}

impl Grid {
    pub fn add<I: Instrument>(&mut self, samples: &[ExposureSample]) {
        for s in samples {
            self.live_seconds += s.seconds;
            let Some(p) = &s.position else {
                self.no_ephemeris_seconds += s.seconds;
                continue;
            };
            let lat = ((p.latitude + 90.0).floor() as i32).clamp(0, 179) as u16;
            let lon = p.longitude.rem_euclid(360.0);
            *self.map.entry((lat, (lon.floor() as u16).min(359))).or_default() += s.seconds;
            let utc = MissionElapsedTime::<I>::new(s.met).to_utc();
            let lst = blink_solar::apparent_solar_time(utc, lon);
            let half_hour = (lst.num_seconds_from_midnight() / 1800).min(47) as u16;
            *self.lst.entry((lat, half_hour)).or_default() += s.seconds;
        }
    }

    pub fn into_day(self, date: NaiveDate, instrument: &str) -> DayExposure {
        let cells = |m: BTreeMap<(u16, u16), f64>| {
            m.into_iter()
                .map(|((i, j), s)| (i, j, (s * 10.0).round() / 10.0))
                .collect()
        };
        DayExposure {
            date,
            instrument: instrument.to_string(),
            step_seconds: STEP_SECONDS,
            live_seconds: self.live_seconds,
            no_ephemeris_seconds: self.no_ephemeris_seconds,
            map: cells(self.map),
            lst: cells(self.lst),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DayExposure {
    pub date: NaiveDate,
    pub instrument: String,
    pub step_seconds: f64,
    pub live_seconds: f64,
    pub no_ephemeris_seconds: f64,
    pub map: Vec<(u16, u16, f64)>,
    pub lst: Vec<(u16, u16, f64)>,
}

pub fn day_path<I: Instrument>(root: &Path, day: NaiveDate, kind: &str) -> PathBuf {
    root.join(I::name().replace('/', "_"))
        .join(format!("{:04}", day.year()))
        .join(format!("{:02}", day.month()))
        .join(format!("{}_{kind}.json", day.format("%Y%m%d")))
}

pub fn write_day(path: &Path, exposure: &DayExposure) {
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).expect("failed to create output directory");
    }
    let temp = path.with_extension(format!("json.{}.tmp", std::process::id()));
    fs::write(&temp, serde_json::to_string(exposure).expect("serialize exposure"))
        .expect("failed to write exposure");
    fs::rename(&temp, path).expect("failed to rename exposure");
}

/// 一个小时的曝光：仪器没实现时 `None`
pub fn hour_samples<C: Chunk>(chunk: &C) -> Option<Vec<ExposureSample>> {
    chunk.exposure_samples(STEP_SECONDS)
}

/// 给已有存档补算一天。体检与搜索相同：载不进来或被排除的小时曝光为零。
/// 返回是否写出（仪器没实现曝光时不写）。
pub fn exposure_day<I: Instrument>(day: NaiveDate, root: &Path, force: bool) -> Option<f64> {
    let path = day_path::<I>(root, day, "exposure");
    let hours: Vec<DateTime<Utc>> = (0..24)
        .map(|h| day.and_hms_opt(h, 0, 0).expect("valid hour").and_utc())
        .collect();
    let last_modified = hours.iter().filter_map(|h| I::Chunk::last_modified(h).ok()).max()?;
    if !force
        && fs::metadata(&path)
            .and_then(|m| m.modified())
            .is_ok_and(|m| DateTime::<Utc>::from(m) >= last_modified)
    {
        return None;
    }
    let mut grid = Grid::default();
    let mut implemented = false;
    for hour in &hours {
        let Ok(chunk) = I::Chunk::from_epoch(hour) else {
            continue;
        };
        if chunk.exclusion().is_some() {
            continue;
        }
        if let Some(samples) = hour_samples(&chunk) {
            implemented = true;
            grid.add::<I>(&samples);
        }
    }
    if !implemented {
        return None;
    }
    let live = grid.live_seconds;
    write_day(&path, &grid.into_day(day, I::name()));
    Some(live)
}

pub fn exposure_range<I: Instrument>(
    from: NaiveDate,
    to: NaiveDate,
    workers: usize,
    worker: usize,
    force: bool,
) {
    let root = Path::new("data");
    let (mut written, mut live) = (0usize, 0.0);
    for (i, day) in from.iter_days().take_while(|d| *d <= to).enumerate() {
        if i % workers != worker {
            continue;
        }
        if let Some(seconds) = exposure_day::<I>(day, root, force) {
            written += 1;
            live += seconds;
            eprintln!("exposure {day}: {:.0} s live", seconds);
        }
    }
    eprintln!(
        "exposure {} {from} .. {to} (worker {worker}/{workers}): {written} days written, \
         {:.3} yr live",
        I::name(),
        live / (365.25 * 86400.0)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use blink_core::types::Position;
    use uom::si::f64::Length;

    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    struct T;
    struct C;
    #[derive(serde::Serialize, Clone, Debug)]
    struct E;
    impl blink_core::traits::Event for E {
        type Instrument = T;
        type ChannelType = u16;
        fn time(&self) -> MissionElapsedTime<T> { MissionElapsedTime::new(0.0) }
        fn channel(&self) -> u16 { 0 }
        fn group(&self) -> u8 { 0 }
        fn keep(&self) -> bool { true }
    }
    impl Chunk for C {
        type Event = E;
        fn from_epoch(_: &DateTime<Utc>) -> Result<Self, blink_core::error::Error> { Err(blink_core::error::Error::Unknown) }
        fn search(&self) -> Vec<blink_core::types::Signal<E>> { Vec::new() }
        fn last_modified(_: &DateTime<Utc>) -> Result<DateTime<Utc>, blink_core::error::Error> { Err(blink_core::error::Error::Unknown) }
        fn coverage(&self) -> blink_core::types::Coverage { blink_core::types::Coverage { span_seconds: 0.0, masked_seconds: 0.0 } }
    }
    impl Instrument for T {
        type Chunk = C;
        fn ref_time() -> &'static DateTime<Utc> {
            static R: std::sync::OnceLock<DateTime<Utc>> = std::sync::OnceLock::new();
            R.get_or_init(|| Utc.with_ymd_and_hms(2024, 6, 21, 0, 0, 0).unwrap())
        }
        fn launch_day() -> NaiveDate { NaiveDate::from_ymd_opt(2024, 6, 21).unwrap() }
        fn name() -> &'static str { "Test/X" }
    }

    fn at(met: f64, lon: f64, lat: f64) -> ExposureSample {
        ExposureSample {
            met,
            seconds: 1.0,
            position: Some(Position { longitude: lon, latitude: lat, altitude: Length::new::<uom::si::length::kilometer>(540.0) }),
        }
    }

    #[test]
    fn samples_land_in_their_cells() {
        let mut grid = Grid::default();
        grid.add::<T>(&[
            at(0.0, 0.5, 0.5),         // 00:00 UTC，经度 0.5° → 地方时约 0 点
            at(1.0, 0.7, 0.2),         // 同一格
            at(43200.0, -90.5, -10.0), // 12:00 UTC 在 90.5°W → 地方时约 6 点；经度折到 269
            ExposureSample { met: 2.0, seconds: 1.0, position: None },
        ]);
        let day = grid.into_day(NaiveDate::from_ymd_opt(2024, 6, 21).unwrap(), "Test/X");
        assert_eq!(day.live_seconds, 4.0);
        assert_eq!(day.no_ephemeris_seconds, 1.0);
        assert_eq!(day.map, vec![(80, 269, 1.0), (90, 0, 2.0)]);
        // 视太阳时：0 点附近（夏至前后时差约 −1.7 min，可能落在前一天的 23:30 格）与 6 点
        let lst: Vec<u16> = day.lst.iter().map(|c| c.1).collect();
        assert!(lst.contains(&12) || lst.contains(&11), "{lst:?}");
        assert!(lst.contains(&0) || lst.contains(&47), "{lst:?}");
    }
}
