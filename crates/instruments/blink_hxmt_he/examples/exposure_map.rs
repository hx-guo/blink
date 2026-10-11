//! 搜索的活时间在轨道上的分布：给 TGF 的地理、地方时、季节分布做曝光归一，给发生率估计当分母。
//!
//! 用法：从 stdin 逐行读 `YYYY-MM-DDTHH`，对每个小时取搜索实际用的活时间
//! （`gti_from_events_by_box`，与搜索同一口径；饱和掩膜不扣，全任务约 1.3%），
//! 在活时间里每秒取一个点、插值星下点，按下面四种格子累加秒数，stdout 输出稀疏 CSV
//! `kind,i,j,seconds`：
//!
//! * `map`：i = 纬度格（-90 起，1°），j = 经度格（0–360，1°）
//! * `lst`：i = 纬度格（1°），j = 视太阳时格（0.5 h，0–47）
//! * `month`：i = 纬度格（1°），j = 月份（1–12）
//! * `zenith`：i = 正午太阳天顶角格（1°，0–179），j = 0
//!
//! 视太阳时与正午天顶角用 `blink_solar`，与目录分析同一套公式。读不到的小时打到 stderr 并跳过。

use blink_core::traits::{Chunk as _, Event as _};
use blink_core::types::{MissionElapsedTime, Position, Trajectory};
use blink_hxmt_he::algorithms::live_time;
use blink_hxmt_he::types::HxmtHe;
use blink_hxmt_he::types::chunk::Chunk;
use chrono::{DateTime, Datelike, NaiveDateTime, TimeZone, Timelike, Utc};
use std::collections::BTreeMap;
use std::io::BufRead;

const STEP_SECONDS: f64 = 1.0;

fn main() {
    let mut cells: BTreeMap<(&'static str, i32, i32), f64> = BTreeMap::new();
    for line in std::io::stdin().lock().lines() {
        let epoch = line.expect("stdin");
        let epoch = epoch.trim();
        if epoch.is_empty() {
            continue;
        }
        let Ok(naive) = NaiveDateTime::parse_from_str(&format!("{epoch}:00:00"), "%Y-%m-%dT%H:%M:%S")
        else {
            eprintln!("{epoch}: bad epoch");
            continue;
        };
        let chunk = match Chunk::from_epoch(&Utc.from_utc_datetime(&naive)) {
            Ok(chunk) => chunk,
            Err(e) => {
                eprintln!("{epoch}: {e:?}");
                continue;
            }
        };
        let events = chunk
            .event_file
            .into_iter()
            .filter(|event| event.keep())
            .collect::<Vec<_>>();
        let gti = live_time::gti_from_events_by_box(
            &events,
            [chunk.span[0], chunk.span[1]],
            live_time::DEFAULT_FALSE_GAP_BUDGET,
        );
        let positions = Trajectory::<MissionElapsedTime<HxmtHe>, Position>::from(&chunk.orbit_file);
        let mut missing = 0usize;
        for seg in &gti {
            let (a, b) = (seg[0].met(), seg[1].met());
            let mut t = a;
            while t < b {
                // 每个采样点代表 [t, t + step) 与段的交集
                let weight = (b - t).min(STEP_SECONDS);
                let met = MissionElapsedTime::<HxmtHe>::new(t + weight / 2.0);
                let Some(p) = positions.interpolate(met) else {
                    missing += 1;
                    t += STEP_SECONDS;
                    continue;
                };
                let (lat, lon) = (p.state.latitude, p.state.longitude.rem_euclid(360.0));
                let utc: DateTime<Utc> = met.into();
                let lat_bin = (lat + 90.0).floor() as i32;
                *cells.entry(("map", lat_bin, lon.floor() as i32)).or_default() += weight;
                let lst = blink_solar::apparent_solar_time(utc, lon);
                let lst_hours = lst.num_seconds_from_midnight() as f64 / 3600.0;
                *cells.entry(("lst", lat_bin, (lst_hours * 2.0).floor() as i32)).or_default() += weight;
                *cells.entry(("month", lat_bin, utc.month() as i32)).or_default() += weight;
                let zenith = blink_solar::solar_zenith_angle_at_noon(utc, lat);
                *cells.entry(("zenith", zenith.floor() as i32, 0)).or_default() += weight;
                t += STEP_SECONDS;
            }
        }
        if missing > 0 {
            eprintln!("{epoch}: {missing} samples without ephemeris");
        }
    }
    println!("kind,i,j,seconds");
    for ((kind, i, j), s) in cells {
        println!("{kind},{i},{j},{s:.3}");
    }
}
