//! 归档里按小时找文件。
//!
//! 三类产品的文件名同一个模子：`{星代号}{类型}_{yymmdd}_{HH}_v{NN}.fits`，
//! 例如 `gbg_evt_220615_00_v00.fits`（GECAM-B 的 GRD）、`gbc_evt_...`（CPD）、
//! `gb_posatt_...`（位姿）。同一小时常有多个版本，取版本号最大的那个——
//! 高版本过境更全，这一条与天格的处理一致。

use chrono::prelude::*;
use std::fs;
use std::path::PathBuf;

use crate::types::instrument::Satellite;

fn find_latest_version(dir: &PathBuf, stem: &str) -> Result<PathBuf, std::io::Error> {
    let mut names: Vec<String> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        // 目录里混着 `*_quality.json`，只认 FITS
        .filter(|name| name.starts_with(stem) && name.ends_with(".fits"))
        .collect();
    names.sort();

    names.last().map(|name| dir.join(name)).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no GECAM file {stem}* in {}", dir.display()),
        )
    })
}

/// 文件名里版本号之前的那一截，例如 `gbg_evt_220615_07_v`。
fn stem<S: Satellite>(hour: &DateTime<Utc>, infix: &str) -> String {
    format!(
        "{}{}_{}_{}_v",
        S::CODE,
        infix,
        hour.format("%y%m%d"),
        hour.format("%H")
    )
}

fn find<S: Satellite>(
    hour: &DateTime<Utc>,
    subdir: &str,
    infix: &str,
) -> Result<PathBuf, std::io::Error> {
    find_latest_version(&S::day_dir(hour).join(subdir), &stem::<S>(hour, infix))
}

/// 伽马射线探测器（GRD）的逐小时事例文件。
pub fn find_grd<S: Satellite>(hour: &DateTime<Utc>) -> Result<PathBuf, std::io::Error> {
    find::<S>(hour, S::GRD_SUBDIR, "g_evt")
}

/// 带电粒子探测器（CPD）的逐小时事例文件。
pub fn find_cpd<S: Satellite>(hour: &DateTime<Utc>) -> Result<PathBuf, std::io::Error> {
    find::<S>(hour, S::CPD_SUBDIR, "c_evt")
}

/// 逐小时位姿文件（位置与姿态在同一张表里）。
pub fn find_posatt<S: Satellite>(hour: &DateTime<Utc>) -> Result<PathBuf, std::io::Error> {
    find::<S>(hour, S::POSATT_SUBDIR, "_posatt")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::instrument::{SatA, SatB, SatC};

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("blink_gecam_{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(dir: &PathBuf, name: &str) {
        fs::write(dir.join(name), b"").unwrap();
    }

    fn hour(h: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2022, 6, 15, h, 0, 0).unwrap()
    }

    // 归档根是进程级环境变量，改它会和并行跑的别的测试打架，所以这里分开验：
    // 文件名怎么拼（stem）和目录里怎么挑（find_latest_version）。

    #[test]
    fn the_stem_carries_the_satellite_code_the_product_and_the_hour() {
        assert_eq!(stem::<SatB>(&hour(7), "g_evt"), "gbg_evt_220615_07_v");
        assert_eq!(stem::<SatB>(&hour(7), "c_evt"), "gbc_evt_220615_07_v");
        assert_eq!(stem::<SatB>(&hour(7), "_posatt"), "gb_posatt_220615_07_v");
        assert_eq!(stem::<SatA>(&hour(0), "g_evt"), "gag_evt_220615_00_v");
        assert_eq!(stem::<SatC>(&hour(23), "g_evt"), "gcg_evt_220615_23_v");
    }

    #[test]
    fn the_highest_version_wins_and_json_siblings_are_ignored() {
        let dir = scratch("versions");
        touch(&dir, "gbg_evt_220615_07_v00.fits");
        touch(&dir, "gbg_evt_220615_07_v02.fits");
        touch(&dir, "gbg_evt_220615_07_v01.fits");
        touch(&dir, "gbg_evt_220615_07_0_1234_quality.json");

        let found = find_latest_version(&dir, "gbg_evt_220615_07_v").unwrap();
        assert_eq!(found.file_name().unwrap(), "gbg_evt_220615_07_v02.fits");
    }

    #[test]
    fn a_neighbouring_hour_does_not_stand_in_for_a_missing_one() {
        let dir = scratch("missing");
        touch(&dir, "gcg_evt_230615_03_v00.fits");
        assert!(find_latest_version(&dir, "gcg_evt_230615_04_v").is_err());
    }

    #[test]
    fn a_missing_directory_is_an_error_not_a_panic() {
        let dir = std::env::temp_dir().join("blink_gecam_no_such_dir");
        let _ = fs::remove_dir_all(&dir);
        assert!(find_latest_version(&dir, "gbg_evt_220615_07_v").is_err());
    }
}
