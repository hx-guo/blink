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
use fitsio_pure::compat::errors::Error;
use fitsio_pure::compat::fitsfile::FitsFile;
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

/// 打开 GBM 归档里的 FITS 文件（多为 `.fit.gz`）。
///
/// 镜像里有一批 gzip 文件在数据流之后还带着几百 kB 的尾部垃圾：2013-07 起的 BGO
/// 逐小时 TTE，如 `glg_tte_b0_130801_00z_v00.fit.gz` 解完第一个成员（83128320 字节，
/// 恰为 2880 的整数倍）后还剩 447287 字节。gzip(1) 报 "trailing garbage ignored" 照常
/// 解出，cfitsio 也读得了；fitsio-pure 却把它当下一个成员去解，报 invalid gzip header，
/// 这些小时整小时被记成 corrupt_data。这里逐个成员解压，后面不是 gzip 成员就停，与
/// gzip(1) 一致；每个成员照样校验 CRC-32 与长度，所以截断或损坏仍然报错。
pub fn open_fits(path: &str) -> Result<FitsFile, Error> {
    let data = fs::read(path)?;
    if !fitsio_pure::gzip::is_gzip(&data) {
        return Ok(FitsFile::from_bytes(data)?);
    }
    let mut out = Vec::new();
    let mut rest = &data[..];
    loop {
        let used = gunzip_member(rest, &mut out)
            .map_err(|e| Error::Message(format!("{path}: gzip: {e}")))?;
        rest = &rest[used..];
        if !fitsio_pure::gzip::is_gzip(rest) {
            break;
        }
    }
    Ok(FitsFile::from_bytes(out)?)
}

/// 解一个 gzip 成员（RFC 1952）接到 `out` 末尾，返回它占的字节数。
fn gunzip_member(data: &[u8], out: &mut Vec<u8>) -> Result<usize, &'static str> {
    use miniz_oxide::inflate::TINFLStatus;
    use miniz_oxide::inflate::core::{DecompressorOxide, decompress, inflate_flags};

    if data.len() < 18 || data[2] != 8 {
        return Err("invalid header");
    }
    let flags = data[3];
    let mut pos = 10;
    if flags & 0x04 != 0 {
        let xlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2 + xlen;
    }
    for bit in [0x08, 0x10] {
        // FNAME、FCOMMENT：以 0 结尾的字符串
        if flags & bit != 0 {
            pos += data.get(pos..).and_then(|d| d.iter().position(|&b| b == 0)).ok_or("truncated header")? + 1;
        }
    }
    if flags & 0x02 != 0 {
        pos += 2;
    }
    let start = out.len();
    let mut state = DecompressorOxide::new();
    let mut out_pos = start;
    out.resize(start + data.len().saturating_mul(2).max(64), 0);
    loop {
        let input = data.get(pos..).ok_or("truncated data")?;
        let (status, read, written) = decompress(
            &mut state,
            input,
            out,
            out_pos,
            inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF,
        );
        pos += read;
        out_pos += written;
        match status {
            TINFLStatus::Done => break,
            TINFLStatus::HasMoreOutput => {
                let grown = out.len() * 2;
                out.resize(grown, 0);
            }
            _ => return Err("inflate failed"),
        }
    }
    out.truncate(out_pos);
    let trailer = data.get(pos..pos + 8).ok_or("truncated trailer")?;
    let crc = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    let size = u32::from_le_bytes([trailer[4], trailer[5], trailer[6], trailer[7]]);
    let member = &out[start..];
    if crc32(member) != crc {
        return Err("CRC-32 mismatch");
    }
    if member.len() as u32 != size {
        return Err("length mismatch");
    }
    Ok(pos + 8)
}

fn crc32(data: &[u8]) -> u32 {
    static TABLE: LazyLock<[u32; 256]> = LazyLock::new(|| {
        let mut table = [0u32; 256];
        for (i, slot) in table.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *slot = c;
        }
        table
    });
    !data
        .iter()
        .fold(!0u32, |c, &b| TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8))
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

    /// 把一段字节包成一个 gzip 成员，给测试造文件
    fn gzip(data: &[u8]) -> Vec<u8> {
        let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 0xff];
        out.extend(miniz_oxide::deflate::compress_to_vec(data, 6));
        out.extend(crc32(data).to_le_bytes());
        out.extend((data.len() as u32).to_le_bytes());
        out
    }

    #[test]
    fn crc32_matches_the_reference_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn members_decompress_and_trailing_garbage_is_ignored() {
        let (a, b) = (vec![7u8; 5000], b"second member".to_vec());
        let mut data = gzip(&a);
        data.extend(gzip(&b));
        let mut out = Vec::new();
        let used = gunzip_member(&data, &mut out).unwrap();
        assert_eq!(out, a);
        let used2 = gunzip_member(&data[used..], &mut out).unwrap();
        assert_eq!(used + used2, data.len());
        assert_eq!(&out[a.len()..], &b[..]);
        // 尾部垃圾：open_fits 在下一个不是 gzip 成员处停，这里只验成员边界
        let mut junk = gzip(&a);
        let n = junk.len();
        junk.extend([0xf1, 0xe9, 0x71, 0xe4]);
        assert_eq!(gunzip_member(&junk, &mut Vec::new()).unwrap(), n);
        assert!(!fitsio_pure::gzip::is_gzip(&junk[n..]));
    }

    #[test]
    fn a_corrupt_member_is_an_error() {
        let mut data = gzip(&[1u8; 3000]);
        let k = data.len() - 6;
        data[k] ^= 0xff; // CRC 或长度
        assert!(gunzip_member(&data, &mut Vec::new()).is_err());
        assert!(gunzip_member(&data[..data.len() - 4], &mut Vec::new()).is_err());
    }
}
