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

/// 文件名尾巴上的版本号：`..._v03.fits` 给出 3。认不出来就不是本小时的文件。
///
/// 归档里同一小时的时间字段有两种写法——多数是 `_07_`，2021 年有一批写成
/// `_070000_`（HHMMSS）。**两种都要认**：GECAM-B 有 2,384 个 GRD 文件是后者，
/// 其中 102 个小时只有这一种写法，而已发表的 147 个 TGF 里有 2 个（1.4%）正
/// 落在这种小时上——匹不上就等于连数据都读不到，完备性量的就成了文件名匹配
/// 而不是搜索算法。
///
/// 版本号要单独解析出来比大小，不能靠整串文件名排序：两种写法混在一起时
/// `_070000_v01` 会排在 `_07_v00` 前面（`'0' < '_'`），照字典序取最后一个就
/// 会拿到低版本。
fn version_of(name: &str, prefix: &str) -> Option<u32> {
    let rest = name.strip_prefix(prefix)?.strip_suffix(".fits")?;
    // `_07` 之后要么直接是 `_v..`，要么先补 MMSS 四位再 `_v..`
    let rest = match rest.strip_prefix("_v") {
        Some(digits) => digits,
        None => {
            let (minutes_seconds, rest) = rest.split_at_checked(4)?;
            if !minutes_seconds.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            rest.strip_prefix("_v")?
        }
    };
    rest.parse().ok()
}

fn find_latest_version(dir: &PathBuf, prefix: &str) -> Result<PathBuf, std::io::Error> {
    // 目录里混着 `*_quality.json`，`version_of` 只认 `..._v<数字>.fits`
    fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .filter_map(|name| version_of(&name, prefix).map(|version| (version, name)))
        // 版本相同时按文件名取最后一个，只为了结果可复现
        .max()
        .map(|(_, name)| dir.join(name))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no GECAM file {prefix}* in {}", dir.display()),
            )
        })
}

/// 文件名里时间字段之前（含时间字段）的那一截，例如 `gbg_evt_220615_07`。
/// 版本号那一段交给 `version_of`，因为时间字段有两种写法。
fn stem<S: Satellite>(hour: &DateTime<Utc>, infix: &str) -> String {
    format!(
        "{}{}_{}_{}",
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
        assert_eq!(stem::<SatB>(&hour(7), "g_evt"), "gbg_evt_220615_07");
        assert_eq!(stem::<SatB>(&hour(7), "c_evt"), "gbc_evt_220615_07");
        assert_eq!(stem::<SatB>(&hour(7), "_posatt"), "gb_posatt_220615_07");
        assert_eq!(stem::<SatA>(&hour(0), "g_evt"), "gag_evt_220615_00");
        assert_eq!(stem::<SatC>(&hour(23), "g_evt"), "gcg_evt_220615_23");
    }

    #[test]
    fn the_highest_version_wins_and_json_siblings_are_ignored() {
        let dir = scratch("versions");
        touch(&dir, "gbg_evt_220615_07_v00.fits");
        touch(&dir, "gbg_evt_220615_07_v02.fits");
        touch(&dir, "gbg_evt_220615_07_v01.fits");
        touch(&dir, "gbg_evt_220615_07_0_1234_quality.json");

        let found = find_latest_version(&dir, "gbg_evt_220615_07").unwrap();
        assert_eq!(found.file_name().unwrap(), "gbg_evt_220615_07_v02.fits");
    }

    /// 2021 年有一批文件把时间字段写成 HHMMSS。GECAM-B 上有 102 个小时只有这
    /// 一种写法，而已发表的 147 个 TGF 里有 2 个正落在这种小时上。
    #[test]
    fn the_hhmmss_spelling_of_the_hour_is_found_too() {
        let dir = scratch("hhmmss");
        touch(&dir, "gbg_evt_210228_190000_v00.fits");
        let found = find_latest_version(&dir, "gbg_evt_210228_19").unwrap();
        assert_eq!(found.file_name().unwrap(), "gbg_evt_210228_190000_v00.fits");
    }

    /// 两种写法混在一起时不能靠整串排序挑版本：`'0' < '_'`，字典序会把
    /// `_190000_v01` 排在 `_19_v00` 前面，取最后一个就拿到了低版本。
    #[test]
    fn the_version_number_wins_across_both_spellings() {
        let dir = scratch("mixed");
        touch(&dir, "gbg_evt_210228_19_v00.fits");
        touch(&dir, "gbg_evt_210228_190000_v01.fits");
        let found = find_latest_version(&dir, "gbg_evt_210228_19").unwrap();
        assert_eq!(found.file_name().unwrap(), "gbg_evt_210228_190000_v01.fits");
    }

    /// 放宽写法不能顺带放宽小时：`_19` 之后只许接 `_v` 或四位 MMSS 再 `_v`。
    #[test]
    fn a_longer_hour_field_is_not_mistaken_for_this_hour() {
        let dir = scratch("hourfield");
        touch(&dir, "gbg_evt_210228_19123_v00.fits"); // 五位，既非 HH 也非 HHMMSS
        touch(&dir, "gbg_evt_210228_19abcd_v00.fits"); // MMSS 位置不是数字
        assert!(find_latest_version(&dir, "gbg_evt_210228_19").is_err());
    }

    #[test]
    fn a_neighbouring_hour_does_not_stand_in_for_a_missing_one() {
        let dir = scratch("missing");
        touch(&dir, "gcg_evt_230615_03_v00.fits");
        assert!(find_latest_version(&dir, "gcg_evt_230615_04").is_err());
    }

    #[test]
    fn a_missing_directory_is_an_error_not_a_panic() {
        let dir = std::env::temp_dir().join("blink_gecam_no_such_dir");
        let _ = fs::remove_dir_all(&dir);
        assert!(find_latest_version(&dir, "gbg_evt_220615_07").is_err());
    }
}
