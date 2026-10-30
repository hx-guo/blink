//! 实时管线的目录生成（S3 列车 + S4 判选）：逐日 signals / hours / vet → 目录。
//!
//! 与 `catalog.rs`（读 `tgfs.json`、论文判选）的区别：
//!
//! * **不用闪电数据。** 判选只有一档：`fa ≤ 1e-5`。闪电数据总比卫星数据晚到，
//!   实时管线不等它。
//! * **列车在这里算。** 邻居数与阈值用 `blink_algorithms::train`，与 `blink wwlln`
//!   同一实现，池是磁盘上全部 `*_signals.json`。
//! * **GRB 剔除。** 读 `blink vet` 的逐候选判据，再把被判掉者 1 s 内的入选候选算作
//!   同一个暴一并剔除（HXMT 论文 4.6 节）。
//! * **事件状态。** 判据所需的数据都齐了才是 `final`，否则 `provisional` 并写明
//!   缺什么（`pending` 列）。编号只发给 final（见 docs/tgf-pipeline-design.md 第 3 节）。
//!
//! 产出到 `--out-dir`：
//!
//! * `catalog.csv`：入选事件（final 与 provisional），按 start 排序；
//! * `rejected.csv`：过了显著性但被列车或 GRB 判据剔除的候选及原因；
//! * `manifest.json`：代码版本、判据常数、逐步计数、数据范围与活时间；
//! * `exposure.json`：逐日 `*_exposure.json`（S5）的合计——1° 经纬格、纬度 × 视太阳时、
//!   纬度 × 月份的活时间（秒），率估计的分母。活时间与搜索同一口径。
//!
//! 逐步计数同时打印到 stderr，无静默过滤。

use crate::commands::vet::Vet;
use blink_algorithms::train::{TRAIN_HALF_WINDOW_US, neighbor_counts_us, train_threshold};
use blink_core::traits::Instrument;
use blink_core::types::{ExclusionReason, UnifiedSignal};
use blink_hxmt_he::algorithms::acd::{BEAM_P_THRESHOLD, TOP_SHARE_REFERENCE};
use blink_search::DayReport;
use blink_search::exposure::DayExposure;
use chrono::{DateTime, Datelike, NaiveDate, TimeDelta, Timelike, Utc};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

/// 判选：fa ≤ 1e-5（HXMT 论文的直接显著档，污染约 0.3%）
pub const SELECT_FA: f64 = 1e-5;
/// 被 GRB 判据判掉的候选这么近（秒）以内的入选候选算同一个暴
pub const SAME_BURST_S: f64 = 1.0;

/// 一天的磁盘产物
struct Day {
    signals: Vec<UnifiedSignal>,
    hours: Option<DayReport>,
    vets: Vec<Vet>,
    exposure: Option<DayExposure>,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    let content = fs::read_to_string(path).ok()?;
    Some(
        serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("failed to parse {}: {e}", path.display())),
    )
}

/// 扫 `<data>/<slug>/YYYY/MM/YYYYMMDD_signals.json`，按日期收齐三种文件。
fn load_days(root: &Path) -> BTreeMap<NaiveDate, Day> {
    let mut days = BTreeMap::new();
    let entries = |dir: &Path| -> Vec<std::path::PathBuf> {
        let mut paths: Vec<_> = fs::read_dir(dir)
            .map(|it| it.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        paths.sort();
        paths
    };
    for year in entries(root) {
        for month in entries(&year) {
            for file in entries(&month) {
                let Some(name) = file.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                let Some(stem) = name.strip_suffix("_signals.json") else {
                    continue;
                };
                let Ok(date) = NaiveDate::parse_from_str(stem, "%Y%m%d") else {
                    continue;
                };
                days.insert(
                    date,
                    Day {
                        signals: read_json(&file).unwrap_or_default(),
                        hours: read_json(&month.join(format!("{stem}_hours.json"))),
                        vets: read_json(&month.join(format!("{stem}_vet.json"))).unwrap_or_default(),
                        exposure: read_json(&month.join(format!("{stem}_exposure.json"))),
                    },
                );
            }
        }
    }
    days
}

/// 与 signals 内串一致的时间字符串（serde 同一格式，供下游按 start 精确 join）
fn time_string(t: &DateTime<Utc>) -> String {
    serde_json::to_string(t)
        .expect("serialize datetime")
        .trim_matches('"')
        .to_string()
}

fn floor_hour(t: DateTime<Utc>) -> DateTime<Utc> {
    t.date_naive()
        .and_hms_opt(t.hour(), 0, 0)
        .expect("valid hour")
        .and_utc()
}

/// 临时 ID：`<slug>-<YYYYMMDDTHHMMSS.ffffff>`（候选起点，UTC）
pub fn cand_id(slug: &str, start: &DateTime<Utc>) -> String {
    format!("{slug}-{}", start.format("%Y%m%dT%H%M%S%.6f"))
}

/// 经度折到 [−180, 180)
fn wrap_longitude(lon: f64) -> f64 {
    (lon + 180.0).rem_euclid(360.0) - 180.0
}

enum Rejection {
    Train,
    Grb,
    GrbSameBurst,
}

impl Rejection {
    fn as_str(&self) -> &'static str {
        match self {
            Rejection::Train => "train",
            Rejection::Grb => "grb",
            Rejection::GrbSameBurst => "grb_same_burst",
        }
    }
}

/// 数据缺失（`MissingData`）的小时要等多久才算定论：实时运行时文件可能只是还没到
pub const DEFAULT_SETTLE_DAYS: i64 = 7;

pub fn cmd_catalog_pipeline<I: Instrument>(
    data: &Path,
    out_dir: &Path,
    fixed_train_threshold: Option<u32>,
    settle_days: i64,
) {
    // 比它新的缺数小时不算"已有结论"：数据晚到时，相邻事件的列车计数还会变
    let settled_before = Utc::now() - TimeDelta::days(settle_days);
    let slug = I::name().replace('/', "_");
    let root = data.join(&slug);
    let days = load_days(&root);
    assert!(!days.is_empty(), "catalog: no *_signals.json under {}", root.display());

    // 账本：哪些小时已经有了结论（搜过或显式排除），以及活时间
    let mut accounted: BTreeSet<DateTime<Utc>> = BTreeSet::new();
    let mut searched_seconds = 0.0;
    let mut days_without_hours = 0usize;
    for (date, day) in &days {
        match &day.hours {
            Some(report) => {
                searched_seconds += report.searched_seconds;
                for record in &report.hours {
                    let hour = date.and_hms_opt(record.hour, 0, 0).expect("valid hour").and_utc();
                    let awaiting = record.reason == Some(ExclusionReason::MissingData) && hour >= settled_before;
                    if !awaiting {
                        accounted.insert(hour);
                    }
                }
            }
            None => days_without_hours += 1,
        }
    }

    // S5 曝光合计。纬度 × 月份由逐日地图按日期折出
    let mut live_seconds = 0.0;
    let mut no_ephemeris_seconds = 0.0;
    let mut days_with_exposure = 0usize;
    let mut map: BTreeMap<(u16, u16), f64> = BTreeMap::new();
    let mut lst: BTreeMap<(u16, u16), f64> = BTreeMap::new();
    let mut month: BTreeMap<(u16, u16), f64> = BTreeMap::new();
    for (date, day) in &days {
        let Some(e) = &day.exposure else { continue };
        days_with_exposure += 1;
        live_seconds += e.live_seconds;
        no_ephemeris_seconds += e.no_ephemeris_seconds;
        for &(i, j, s) in &e.map {
            *map.entry((i, j)).or_default() += s;
            *month.entry((i, date.month() as u16)).or_default() += s;
        }
        for &(i, j, s) in &e.lst {
            *lst.entry((i, j)).or_default() += s;
        }
    }
    let cells = |m: &BTreeMap<(u16, u16), f64>| -> Vec<(u16, u16, f64)> {
        m.iter().map(|(&(i, j), &s)| (i, j, (s * 10.0).round() / 10.0)).collect()
    };

    let mut pool: Vec<&UnifiedSignal> = days.values().flat_map(|d| d.signals.iter()).collect();
    pool.sort_by_key(|s| s.start);
    let vets: HashMap<DateTime<Utc>, &Vet> = days
        .values()
        .flat_map(|d| d.vets.iter())
        .map(|v| (v.start, v))
        .collect();
    let n_pool = pool.len();

    // S3 列车：全池邻居数，阈值按池或固定
    let times_us: Vec<i64> = pool.iter().map(|s| s.start.timestamp_micros()).collect();
    let neighbors = neighbor_counts_us(&times_us);
    let threshold = fixed_train_threshold.unwrap_or_else(|| train_threshold(&times_us, &neighbors));

    // S4 判选：显著性 → 列车 → GRB（逐候选） → 同一个暴
    let mut rejected: Vec<(usize, Rejection)> = Vec::new();
    let mut selected: Vec<usize> = Vec::new();
    let mut n_significant = 0usize;
    for (i, signal) in pool.iter().enumerate() {
        if signal.false_positive_per_year > SELECT_FA {
            continue;
        }
        n_significant += 1;
        if neighbors[i] > threshold {
            rejected.push((i, Rejection::Train));
        } else {
            selected.push(i);
        }
    }
    let n_after_train = selected.len();
    let flagged: Vec<i64> = selected
        .iter()
        .filter(|&&i| vets.get(&pool[i].start).is_some_and(|v| v.grb.flagged))
        .map(|&i| times_us[i])
        .collect();
    let same_burst_us = (SAME_BURST_S * 1e6) as i64;
    let mut kept: Vec<usize> = Vec::new();
    let (mut n_grb, mut n_same_burst) = (0usize, 0usize);
    for i in selected {
        let t = times_us[i];
        if flagged.binary_search(&t).is_ok() {
            n_grb += 1;
            rejected.push((i, Rejection::Grb));
        } else if flagged.iter().any(|&f| (f - t).abs() < same_burst_us) {
            n_same_burst += 1;
            rejected.push((i, Rejection::GrbSameBurst));
        } else {
            kept.push(i);
        }
    }

    // 状态：判据所需的东西齐了才 final
    let half_window = TimeDelta::microseconds(TRAIN_HALF_WINDOW_US);
    let pending_of = |signal: &UnifiedSignal| -> Vec<&'static str> {
        let mut pending = Vec::new();
        match vets.get(&signal.start) {
            None => pending.push("not_vetted"),
            Some(v) if v.grb.z.is_none() => pending.push("grb_background_gap"),
            Some(_) => {}
        }
        let mut hour = floor_hour(signal.start - half_window);
        let last = floor_hour(signal.start + half_window);
        while hour <= last {
            if !accounted.contains(&hour) {
                pending.push("neighbors_unsearched");
                break;
            }
            hour += TimeDelta::hours(1);
        }
        pending
    };

    fs::create_dir_all(out_dir).expect("failed to create output directory");
    let mut rows = String::from(
        "cand_id,instrument,status,pending,start,stop,peak,duration_us,count,\
         false_positive_per_year,longitude,latitude,altitude_km,solar_time,neighbors_10min,\
         grb_z,grb_max_box_share,electron_beam,beam_p,det_share_max,\
         n,n_acd,n_acd_multi,n_bg,n_acd_bg\n",
    );
    let (mut n_final, mut n_beams) = (0usize, 0usize);
    let mut pending_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for &i in &kept {
        let s = pool[i];
        let vet = vets.get(&s.start);
        let pending = pending_of(s);
        if pending.is_empty() {
            n_final += 1;
        }
        for p in &pending {
            *pending_counts.entry(p).or_default() += 1;
        }
        let beam = vet.and_then(|v| v.electron_beam.as_ref());
        n_beams += beam.is_some_and(|b| b.flagged) as usize;
        let acd = vet.and_then(|v| v.acd.as_ref()).or(s.acd.as_ref());
        rows.push_str(&format!(
            "{},{},{},{},{},{},{},{:.3},{},{:e},{:.4},{:.4},{:.1},{},{},{},{},{},{},{},{}\n",
            cand_id(&slug, &s.start),
            I::name(),
            if pending.is_empty() { "final" } else { "provisional" },
            pending.join("|"),
            time_string(&s.start),
            time_string(&s.stop),
            time_string(&s.peak_time()),
            // 持续时间 = 最佳区间（最显著窗口首末事例之间），同 HXMT 目录论文；不是合并后整个候选窗的宽度
            s.bin_size_best.get::<uom::si::time::microsecond>(),
            s.count,
            s.false_positive_per_year,
            wrap_longitude(s.position.longitude),
            s.position.latitude,
            s.position.altitude.get::<uom::si::length::kilometer>(),
            // 峰值时刻星下点的视太阳时，与曝光（S5）同一公式
            blink_solar::apparent_solar_time(s.peak_time(), s.position.longitude.rem_euclid(360.0))
                .format("%H:%M:%S"),
            neighbors[i],
            vet.and_then(|v| v.grb.z).map(|z| format!("{z:.2}")).unwrap_or_default(),
            vet.map(|v| format!("{:.3}", v.grb.max_group_share)).unwrap_or_default(),
            beam.map(|b| (b.flagged as u8).to_string()).unwrap_or_default(),
            beam.map(|b| format!("{:.3e}", b.p)).unwrap_or_default(),
            vet.map(|v| format!("{:.3}", v.det_share_max)).unwrap_or_default(),
            match acd {
                Some(a) => format!("{},{},{},{},{}", a.n, a.n_acd, a.n_acd_multi, a.n_bg, a.n_acd_bg),
                None => ",,,,".to_string(),
            },
        ));
    }
    fs::write(out_dir.join("catalog.csv"), rows).expect("failed to write catalog.csv");

    rejected.sort_by_key(|(i, _)| *i);
    let mut rejected_rows =
        String::from("cand_id,start,false_positive_per_year,reason,neighbors_10min,grb_z,grb_max_box_share\n");
    for (i, reason) in &rejected {
        let s = pool[*i];
        let vet = vets.get(&s.start);
        rejected_rows.push_str(&format!(
            "{},{},{:e},{},{},{},{}\n",
            cand_id(&slug, &s.start),
            time_string(&s.start),
            s.false_positive_per_year,
            reason.as_str(),
            neighbors[*i],
            vet.and_then(|v| v.grb.z).map(|z| format!("{z:.2}")).unwrap_or_default(),
            vet.map(|v| format!("{:.3}", v.grb.max_group_share)).unwrap_or_default(),
        ));
    }
    fs::write(out_dir.join("rejected.csv"), rejected_rows).expect("failed to write rejected.csv");

    let n_train = n_significant - n_after_train;
    let manifest = serde_json::json!({
        "instrument": I::name(),
        "slug": slug,
        "generated_at": time_string(&Utc::now()),
        "blink_git": env!("BLINK_GIT"),
        "data": {
            "first_day": days.keys().next().map(|d| d.to_string()),
            "last_day": days.keys().next_back().map(|d| d.to_string()),
            "days": days.len(),
            "days_without_hours": days_without_hours,
            "searched_seconds": searched_seconds,
            // 曝光的活时间（GTI 交集减屏蔽，与搜索同口径）；没有曝光文件的天不计
            "live_seconds": live_seconds,
            "no_ephemeris_seconds": no_ephemeris_seconds,
            "days_with_exposure": days_with_exposure,
        },
        "selection": {
            "max_false_positive_per_year": SELECT_FA,
            "train_half_window_s": TRAIN_HALF_WINDOW_US / 1_000_000,
            "train_threshold": threshold,
            "train_threshold_fixed": fixed_train_threshold.is_some(),
            "missing_data_settle_days": settle_days,
            "grb_rule": "Z > 5 in [-0.1, +0.3) s outside the candidate, largest box share < 0.6",
            "grb_same_burst_s": SAME_BURST_S,
            "electron_beam_p": BEAM_P_THRESHOLD,
            "electron_beam_top_share_reference": TOP_SHARE_REFERENCE,
        },
        "counts": {
            "pool": n_pool,
            "significant": n_significant,
            "train_removed": n_train,
            "grb_removed": n_grb,
            "grb_same_burst_removed": n_same_burst,
            "catalog": kept.len(),
            "final": n_final,
            "provisional": kept.len() - n_final,
            "pending": pending_counts,
            "electron_beams": n_beams,
        },
    });
    let exposure = serde_json::json!({
        "instrument": I::name(),
        "live_seconds": live_seconds,
        "no_ephemeris_seconds": no_ephemeris_seconds,
        "days_with_exposure": days_with_exposure,
        "days": days.len(),
        "map": { "i": "latitude cell, floor(lat + 90)", "j": "longitude cell, floor(lon mod 360)", "cells": cells(&map) },
        "lst": { "i": "latitude cell", "j": "apparent solar time, half hours", "cells": cells(&lst) },
        "month": { "i": "latitude cell", "j": "month (1-12)", "cells": cells(&month) },
    });
    fs::write(out_dir.join("exposure.json"), exposure.to_string()).expect("failed to write exposure.json");

    fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
    )
    .expect("failed to write manifest.json");

    eprintln!(
        "catalog {}: pool {n_pool} -> fa<={SELECT_FA:e} {n_significant} -> train removed {n_train} \
         (neighbors > {threshold}{}) -> GRB removed {n_grb} + same burst {n_same_burst} -> \
         catalog {} ({n_final} final, {} provisional {pending_counts:?}), {n_beams} electron beams; \
         {:.2} yr searched over {} days, exposure {:.2} yr live on {days_with_exposure} days -> {}",
        I::name(),
        if fixed_train_threshold.is_some() { ", fixed" } else { "" },
        kept.len(),
        kept.len() - n_final,
        searched_seconds / (365.25 * 86400.0),
        days.len(),
        live_seconds / (365.25 * 86400.0),
        out_dir.display(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use blink_hxmt_he::types::HxmtHe;

    fn signal(start: &str, fa: f64) -> String {
        format!(
            r#"{{"start": "{start}", "stop": "{start}",
                "bin_size_min": 0.0001, "bin_size_max": 0.001, "bin_size_best": 0.0005,
                "delay": 0.0, "count": 20, "mean": 0.5, "sf": 1e-12,
                "false_positive_per_year": {fa},
                "position": {{ "longitude": 270.0, "latitude": 10.0, "altitude": 540000.0 }},
                "instrument": "Insight-HXMT/HE"}}"#
        )
    }

    fn hours(date: &str, n: u32) -> String {
        let records: Vec<String> = (0..n)
            .map(|h| format!(r#"{{"hour": {h}, "status": "searched", "searched_seconds": 3600.0, "n_signals": 0}}"#))
            .collect();
        format!(
            r#"{{"date": "{date}", "instrument": "Insight-HXMT/HE", "searched_hours": {n},
                "excluded_hours": 0, "searched_seconds": {}, "excluded_by_reason": {{}},
                "n_signals": 0, "hours": [{}]}}"#,
            3600.0 * n as f64,
            records.join(",")
        )
    }

    fn vet(start: &str, z: Option<f64>, flagged: bool) -> String {
        format!(
            r#"{{"start": "{start}", "stop": "{start}", "instrument": "Insight-HXMT/HE",
                "false_positive_per_year": 1e-9,
                "grb": {{"observed": 10, "expected": 10.0, "z": {}, "group_excess": [0,0,0],
                         "max_group_share": 0.3, "flagged": {flagged}}},
                "det_share_max": 0.1,
                "electron_beam": {{"top_only": 1, "side_only": 1, "p": 0.8, "flagged": false}}}}"#,
            z.map(|z| z.to_string()).unwrap_or("null".into())
        )
    }

    #[test]
    fn train_grb_and_status() {
        let dir = std::env::temp_dir().join(format!("blink_catalog_pipeline_{}", std::process::id()));
        let month = dir.join("data/Insight-HXMT_HE/2024/10");
        fs::create_dir_all(&month).unwrap();
        let mut signals = vec![
            // 弱候选：不入选
            signal("2024-10-09T01:00:00.000000000Z", 0.5),
            // final
            signal("2024-10-09T02:00:00.000000000Z", 1e-9),
            // GRB 判掉，以及 0.5 s 后的同一个暴
            signal("2024-10-09T03:00:00.000000000Z", 1e-9),
            signal("2024-10-09T03:00:00.500000000Z", 1e-9),
            // 本底缺口：provisional
            signal("2024-10-09T04:00:00.000000000Z", 1e-9),
            // 没复核：provisional
            signal("2024-10-09T05:00:00.000000000Z", 1e-9),
            // 23:55 的候选：+10 min 落进次日 00 时，次日没搜：provisional
            signal("2024-10-09T23:55:00.000000000Z", 1e-9),
        ];
        // 06 时一串 40 个候选：列车
        signals.extend((0..40).map(|k| signal(&format!("2024-10-09T06:00:{:02}.000000000Z", k), 1e-9)));
        fs::write(month.join("20241009_signals.json"), format!("[{}]", signals.join(","))).unwrap();
        fs::write(month.join("20241009_hours.json"), hours("2024-10-09", 24)).unwrap();
        let vets = [
            vet("2024-10-09T02:00:00.000000000Z", Some(0.3), false),
            vet("2024-10-09T03:00:00.000000000Z", Some(30.0), true),
            vet("2024-10-09T03:00:00.500000000Z", Some(0.1), false),
            vet("2024-10-09T04:00:00.000000000Z", None, false),
            vet("2024-10-09T23:55:00.000000000Z", Some(0.1), false),
        ];
        fs::write(month.join("20241009_vet.json"), format!("[{}]", vets.join(","))).unwrap();
        fs::write(
            month.join("20241009_exposure.json"),
            r#"{"date": "2024-10-09", "instrument": "Insight-HXMT/HE", "step_seconds": 1.0,
                "live_seconds": 30.0, "no_ephemeris_seconds": 5.0,
                "map": [[90, 0, 10.0], [91, 1, 15.0]], "lst": [[90, 24, 25.0]]}"#,
        )
        .unwrap();

        let out = dir.join("out");
        cmd_catalog_pipeline::<HxmtHe>(&dir.join("data"), &out, Some(34), DEFAULT_SETTLE_DAYS);

        let csv = fs::read_to_string(out.join("catalog.csv")).unwrap();
        let row = |t: &str| csv.lines().find(|l| l.contains(t)).map(str::to_string);
        assert!(row("T02:00:00Z").unwrap().contains(",final,,"));
        // duration_us 是最佳区间 0.5 ms，不是 stop − start（这里为 0）
        assert!(row("T02:00:00Z").unwrap().contains(",500.000,20,"), "{csv}");
        assert!(row("T04:00:00Z").unwrap().contains(",provisional,grb_background_gap,"));
        assert!(row("T05:00:00Z").unwrap().contains(",provisional,not_vetted,"));
        assert!(row("T23:55:00Z").unwrap().contains(",provisional,neighbors_unsearched,"));
        assert!(row("T01:00:00Z").is_none() && row("T03:00:00").is_none() && row("T06:00").is_none());
        // 经度折到 [−180, 180)，ID 用 slug
        assert!(row("T02:00:00Z").unwrap().starts_with("Insight-HXMT_HE-20241009T020000.000000,"));
        assert!(row("T02:00:00Z").unwrap().contains(",-90.0000,"));
        // 02:00 UTC 在 90°W：地方平时 20:00，10 月上旬时差约 +12 min
        assert!(row("T02:00:00Z").unwrap().contains(",20:1"), "{csv}");

        let rejected = fs::read_to_string(out.join("rejected.csv")).unwrap();
        assert_eq!(rejected.matches(",train,").count(), 40);
        assert_eq!(rejected.matches(",grb,").count(), 1);
        assert_eq!(rejected.matches(",grb_same_burst,").count(), 1);

        let manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["counts"]["catalog"], 4);
        assert_eq!(manifest["counts"]["final"], 1);
        assert_eq!(manifest["selection"]["train_threshold"], 34);
        assert_eq!(manifest["data"]["live_seconds"], 30.0);
        let exposure: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("exposure.json")).unwrap()).unwrap();
        // 10 月：两个纬度格各进 month = 10
        assert_eq!(exposure["month"]["cells"], serde_json::json!([[90, 10, 10.0], [91, 10, 15.0]]));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_recently_missing_hour_keeps_its_neighbours_provisional() {
        let dir = std::env::temp_dir().join(format!("blink_catalog_settle_{}", std::process::id()));
        let month = dir.join("data/Insight-HXMT_HE/2024/10");
        fs::create_dir_all(&month).unwrap();
        fs::write(
            month.join("20241009_signals.json"),
            format!("[{}]", signal("2024-10-09T21:55:00.000000000Z", 1e-9)),
        )
        .unwrap();
        // 22 时缺数据，其余小时都搜过
        let mut hours: Vec<String> = (0..24)
            .map(|h| format!(r#"{{"hour": {h}, "status": "searched", "searched_seconds": 3600.0, "n_signals": 0}}"#))
            .collect();
        hours[22] = r#"{"hour": 22, "status": "excluded", "reason": "missing_data", "searched_seconds": 0.0, "n_signals": 0}"#.into();
        fs::write(
            month.join("20241009_hours.json"),
            format!(
                r#"{{"date": "2024-10-09", "instrument": "Insight-HXMT/HE", "searched_hours": 23, "excluded_hours": 1,
                    "searched_seconds": 82800.0, "excluded_by_reason": {{}}, "n_signals": 0, "hours": [{}]}}"#,
                hours.join(",")
            ),
        )
        .unwrap();
        fs::write(month.join("20241009_vet.json"), format!("[{}]", vet("2024-10-09T21:55:00.000000000Z", Some(0.1), false))).unwrap();
        let row = |settle: i64| {
            let out = dir.join(format!("out{settle}"));
            cmd_catalog_pipeline::<HxmtHe>(&dir.join("data"), &out, Some(34), settle);
            fs::read_to_string(out.join("catalog.csv")).unwrap().lines().nth(1).unwrap().to_string()
        };
        // 宽限期还没过（按 10 万天算）：等数据
        assert!(row(100_000).contains(",provisional,neighbors_unsearched,"));
        // 宽限期已过：缺数据是定论
        assert!(row(7).contains(",final,,"));
        fs::remove_dir_all(&dir).unwrap();
    }
}
