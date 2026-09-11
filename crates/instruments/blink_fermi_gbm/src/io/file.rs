//! 归档里找文件。
//!
//! GBM 的镜像有五种并存的布局，而且 NaI 与 BGO 的连续数据不在一处：
//!
//! * `{ROOT}/{YYYY}/{MM}/{DD}/current/` —— 2015 年起的主目录。NaI 的逐小时
//!   TTE 只在这里，且要到 2017-10 才开始有；2020 年这里只剩 12 个 NaI。
//! * `{ROOT}/{YYYY}/{YYMMDD}/tte/` —— 2014 年及以前的主目录，里面是按触发
//!   命名的短文件，没有逐小时 TTE。poshist 在这里。
//! * `{ROOT}/{YYYY}/{MM}/{YYMMDD}/tte/` —— 上一种在 2015-05..07 这 92 天的
//!   变体：多了一层月份，日目录仍是 YYMMDD、里面仍叫 `tte`。这 92 天的
//!   poshist 只在这里，漏掉它整段会变成「有 TTE 却没星历」的零曝光。
//! * `{ROOT}/BGO/{YYYY}/{YYMMDD}/` —— BGO 的逐小时 TTE 单独存放，2012-11
//!   起一直到 2020，是覆盖最长的一份。
//! * `{ROOT}/poshist/` —— 2020-11 那 30 天的 poshist 平摊在这里，`{ROOT}/2020/11`
//!   整个目录都不存在。只有 poshist，没有 TTE。
//!
//! 这五种不是猜的，是把镜像里全部 3765 个 poshist 与 7526 个 TTE 目录的路径
//! 归一化后穷举出来的（`scripts/cluster/gbm_layout_scan.sh`）。前三种曾被当成
//! 全部，结果 2014-12 与 2015-05..07 两次出现「有 TTE 却找不到星历、整段静默
//! 零曝光」，所以这里的规则是：改动镜像覆盖范围前重跑一次穷举，别一种一种试。
//!
//! 所以查找一个探头时按上面的顺序逐个目录试，取版本号最大的那个文件。

use chrono::prelude::*;
use std::path::PathBuf;
use std::{env, fs, sync::LazyLock};

static ROOT: LazyLock<String> = LazyLock::new(|| {
    env::var("FERMI_GBM_DIR").unwrap_or_else(|_| "/hxmtfs/data/Fermi_GBM".to_string())
});

/// 一天可能落在的目录，按优先级排列。
fn day_dirs(time: &DateTime<Utc>) -> Vec<PathBuf> {
    let root = ROOT.as_str();
    let ymd = time.format("%y%m%d").to_string();
    vec![
        PathBuf::from(format!("{root}/{}/current", time.format("%Y/%m/%d"))),
        PathBuf::from(format!("{root}/{}/{ymd}/tte", time.format("%Y"))),
        PathBuf::from(format!("{root}/{}/{ymd}/tte", time.format("%Y/%m"))),
        PathBuf::from(format!("{root}/BGO/{}/{ymd}", time.format("%Y"))),
        PathBuf::from(format!("{root}/poshist")),
    ]
}

/// 目录里以 `prefix` 开头、版本号最大的文件。
fn latest_version(dir: &PathBuf, prefix: &str) -> Option<PathBuf> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().to_string())
        .filter(|name| name.starts_with(prefix))
        .collect();
    // 文件名尾部形如 `_vNN.fit.gz`，版本号定长，字典序即版本序
    names.sort();
    names.last().map(|name| dir.join(name))
}

fn find(time: &DateTime<Utc>, prefix: &str) -> Option<PathBuf> {
    day_dirs(time)
        .iter()
        .find_map(|dir| latest_version(dir, prefix))
}

/// 某探头某小时的 TTE。`detector` 是归档里的代号，如 `n0` / `b1`。
pub fn find_tte(time: &DateTime<Utc>, detector: &str) -> Option<PathBuf> {
    find(
        time,
        &format!(
            "glg_tte_{detector}_{}_{:02}z_v",
            time.format("%y%m%d"),
            time.hour()
        ),
    )
}

/// 当天的 poshist（整天一个文件，1 Hz）。
pub fn find_poshist(time: &DateTime<Utc>) -> Option<PathBuf> {
    find(
        time,
        &format!("glg_poshist_all_{}_v", time.format("%y%m%d")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 镜像的五种布局必须都在候选目录里。这个测试钉的是**条目数与拼法**，
    /// 不是镜像本身——布局是穷举出来的（`scripts/cluster/gbm_layout_scan.sh`），
    /// 少一种就会让落在那种布局里的天静默零曝光：有 TTE、找不到星历，整天
    /// 每小时都被记成 `missing_data`，而搜索照跑、退出码为零。2014-12 与
    /// 2015-05..07 各出过一次，第二次是拿"列表天数 vs 有曝光天数"这把尺子
    /// 才看见的。
    #[test]
    fn day_dirs_covers_every_known_mirror_layout() {
        // 2015-05-01 正是只有第三种布局才找得到 poshist 的那 92 天之一
        let time = Utc.with_ymd_and_hms(2015, 5, 1, 7, 0, 0).unwrap();
        let root = ROOT.as_str();
        let dirs: Vec<String> = day_dirs(&time)
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect();
        assert_eq!(
            dirs,
            vec![
                format!("{root}/2015/05/01/current"),
                format!("{root}/2015/150501/tte"),
                format!("{root}/2015/05/150501/tte"),
                format!("{root}/BGO/2015/150501"),
                format!("{root}/poshist"),
            ]
        );
    }

    /// 一天都找不到时返回 None 而不是 panic——搜索靠这个把缺数据的小时记成
    /// `missing_data`，而不是整天倒掉。
    #[test]
    fn a_day_outside_the_mirror_yields_none_rather_than_panicking() {
        let time = Utc.with_ymd_and_hms(1999, 1, 1, 0, 0, 0).unwrap();
        assert!(find_poshist(&time).is_none());
        assert!(find_tte(&time, "b0").is_none());
    }
}
