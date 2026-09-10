use blink_core::traits::Instrument;
use chrono::prelude::*;
use std::{marker::PhantomData, path::PathBuf, str::FromStr, sync::OnceLock};

/// GECAM 的一颗星。
///
/// 三颗星的事例产品是同一套格式（EBOUNDS + GTI + 每探头一个 `EVENTS%02d`，
/// 列 `TIME/PI/GAIN_TYPE/DEAD_TIME/EVT_TYPE/FLAG`），差别只在归档路径、
/// 文件名前缀和 MET 历元，所以共用一个 crate，按类型参数分星。
///
/// * GECAM-A/B 是 2020-12-10 一起发射的双星，25 路 GRD + 8 路 CPD，MET 历元
///   2019-01-01。A 星起初没有数据，归档里最早的事例产品是 2022-10-21。
/// * GECAM-C（载荷代号 HEBS，搭在 SATech-01 上）12 路 GRD + 2 路 CPD，
///   MET 历元 2021-01-01，归档在另一个挂载点、子目录名全大写。
pub trait Satellite:
    Clone + Copy + PartialEq + Eq + PartialOrd + Ord + std::fmt::Debug + Send + Sync + 'static
{
    /// 对外的名字，也是输出目录名。
    const NAME: &'static str;
    /// 文件名前缀里的星代号：`ga` / `gb` / `gc`。
    const CODE: &'static str;
    /// 事例产品的第一天。不是发射日：盲搜从有数据的那天起算。
    fn first_day() -> NaiveDate;
    /// 事例表 `TIME` 的零点。A/B 同为 2019-01-01（MJDREFI 58484），
    /// C 是 2021-01-01（59215）。
    fn ref_time() -> &'static DateTime<Utc>;
    /// 某一天的产品目录，例如 `.../daily/2023/06/15/GECAM_B`。C 星那一层
    /// 没有卫星子目录，日期目录下直接就是 `GRD_EVT/` 等。
    fn day_dir(day: &DateTime<Utc>) -> PathBuf;
    /// 三类产品的子目录名。A/B 用小写 `GRD_evt`，C 用大写 `GRD_EVT`。
    const GRD_SUBDIR: &'static str;
    const CPD_SUBDIR: &'static str;
    const POSATT_SUBDIR: &'static str;
}

/// A/B 共用的归档根。`GECAM_ARCHIVE` 可覆盖，为的是能拿一小时的样本文件在
/// 本地跑通整条链路（与 `SVOM_ARCHIVE`、`WWLLN_DB_PATH` 同一套做法）。
const AB_ARCHIVE: &str = "/gecamfs/Archived-DATA/GSDC/LEVEL1/daily";
/// C 星（HEBS）挂在另一个目录树下。`GECAM_C_ARCHIVE` 可覆盖。
const C_ARCHIVE: &str = "/gecamfs/hebs/Archived-DATA/GSDC/LEVEL1/daily";

fn archive_root(env_key: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(env_key).unwrap_or_else(|_| default.to_string()))
}

macro_rules! satellite {
    (
        $ty:ident, $name:literal, $code:literal,
        ($y:literal, $m:literal, $d:literal), $ref_time:literal,
        $subdirs:tt, $day_dir:expr, $doc:literal
    ) => {
        #[doc = $doc]
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
        pub struct $ty;

        impl Satellite for $ty {
            const NAME: &'static str = $name;
            const CODE: &'static str = $code;
            satellite!(@subdirs $subdirs);

            fn first_day() -> NaiveDate {
                NaiveDate::from_ymd_opt($y, $m, $d).unwrap()
            }

            fn ref_time() -> &'static DateTime<Utc> {
                static REF_TIME: OnceLock<DateTime<Utc>> = OnceLock::new();
                REF_TIME.get_or_init(|| DateTime::<Utc>::from_str($ref_time).unwrap())
            }

            fn day_dir(day: &DateTime<Utc>) -> PathBuf {
                #[allow(clippy::redundant_closure_call)]
                ($day_dir)(day)
            }
        }
    };
    (@subdirs (lower)) => {
        const GRD_SUBDIR: &'static str = "GRD_evt";
        const CPD_SUBDIR: &'static str = "CPD_evt";
        const POSATT_SUBDIR: &'static str = "posatt";
    };
    (@subdirs (upper)) => {
        const GRD_SUBDIR: &'static str = "GRD_EVT";
        const CPD_SUBDIR: &'static str = "CPD_EVT";
        const POSATT_SUBDIR: &'static str = "POSATT";
    };
}

satellite!(
    SatA,
    "GECAM-A",
    "ga",
    (2022, 10, 21),
    "2019-01-01T00:00:00.000000000 UTC",
    (lower),
    |day: &DateTime<Utc>| archive_root("GECAM_ARCHIVE", AB_ARCHIVE)
        .join(day.format("%Y/%m/%d").to_string())
        .join("GECAM_A"),
    "GECAM-A：25 GRD + 8 CPD，事例产品最早 2022-10-21。绝大多数数据在 2025 年\n\
     以后，落在 WWLLN 库（止于 2024-12-31）之外，闪电关联只覆盖早期一小段。"
);

satellite!(
    SatB,
    "GECAM-B",
    "gb",
    (2020, 12, 16),
    "2019-01-01T00:00:00.000000000 UTC",
    (lower),
    |day: &DateTime<Utc>| archive_root("GECAM_ARCHIVE", AB_ARCHIVE)
        .join(day.format("%Y/%m/%d").to_string())
        .join("GECAM_B"),
    "GECAM-B：25 GRD + 8 CPD，2020-12 起近连续到 2026。逐小时事例文件约 2.7 GB、\n\
     1.5 亿事例，是三颗星里数据量最大的一颗。"
);

satellite!(
    SatC,
    "GECAM-C",
    "gc",
    (2022, 7, 27),
    "2021-01-01T00:00:00.000000000 UTC",
    (upper),
    |day: &DateTime<Utc>| archive_root("GECAM_C_ARCHIVE", C_ARCHIVE)
        .join(day.format("%Y/%m/%d").to_string()),
    "GECAM-C（HEBS，搭 SATech-01）：12 GRD + 2 CPD，2022-07 .. 2025-02。\n\
     2022–2024 基本满 24 小时，且整段落在 WWLLN 覆盖内。"
);

/// 某颗 GECAM 星作为一台仪器。
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Gecam<S: Satellite>(PhantomData<S>);

impl<S: Satellite> Instrument for Gecam<S> {
    type Chunk = crate::types::Chunk<S>;

    fn ref_time() -> &'static DateTime<Utc> {
        S::ref_time()
    }

    fn launch_day() -> NaiveDate {
        S::first_day()
    }

    fn name() -> &'static str {
        S::NAME
    }
}

pub type GecamA = Gecam<SatA>;
pub type GecamB = Gecam<SatB>;
pub type GecamC = Gecam<SatC>;

#[cfg(test)]
mod tests {
    use super::*;

    fn hour(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 7, 0, 0).unwrap()
    }

    #[test]
    fn ab_carry_a_satellite_subdir_and_c_does_not() {
        // 不去动归档根的环境变量：那是进程级状态，会和并行跑的 io::file 测试打架
        let day = hour(2023, 6, 15);
        assert!(
            SatB::day_dir(&day).ends_with("2023/06/15/GECAM_B"),
            "{:?}",
            SatB::day_dir(&day)
        );
        assert!(
            SatA::day_dir(&day).ends_with("2023/06/15/GECAM_A"),
            "{:?}",
            SatA::day_dir(&day)
        );
        // C 星的日期目录下直接就是产品子目录
        assert!(
            SatC::day_dir(&day).ends_with("2023/06/15"),
            "{:?}",
            SatC::day_dir(&day)
        );
        assert_eq!(SatB::GRD_SUBDIR, "GRD_evt");
        assert_eq!(SatC::GRD_SUBDIR, "GRD_EVT");
    }

    #[test]
    fn met_epochs_match_the_file_headers() {
        // MJDREFI 58484 = 2019-01-01，59215 = 2021-01-01
        assert_eq!(SatA::ref_time(), SatB::ref_time());
        assert_eq!(
            SatB::ref_time().format("%Y-%m-%d").to_string(),
            "2019-01-01"
        );
        assert_eq!(
            SatC::ref_time().format("%Y-%m-%d").to_string(),
            "2021-01-01"
        );
    }
}
