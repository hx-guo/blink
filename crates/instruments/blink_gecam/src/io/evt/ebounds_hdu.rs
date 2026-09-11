//! EBOUNDS：道号到能量的对照表，以及溢出道的边界。
//!
//! 实测（GECAM-B 498 道、GECAM-C 470 道，两颗星逐道数值一致）：
//!
//! * 道 0..447 是一条连续单调的对数能量梯，2.00 keV 到 10053.5 keV，
//!   相邻道 `E_MIN[k+1] == E_MAX[k]`，每道宽约 1.34%。
//! * 道 448 及以上是溢出道：`E_MAX` 一律 20000 keV，`E_MIN` 不单调
//!   （每探头每增益档各占一行，B 有 50 行、C 有 22 行），能量不可信。
//!   事例表里 `EVT_TYPE == 2` 的事例 PI 正好全落在这一段。
//!
//! 两颗星共用同一条能量梯，所以 `Event::keep` 里的道号常量对三颗星通用。
//! 但"通用"是实测结论不是设计保证，跨五年的数据谁也不敢担保标定不改，
//! 所以这里在载入时逐条核对，对不上就报错，不静默沿用错的能阈。

use blink_core::error::Error;

/// 溢出道的起点。道号 ≥ 此值的事例能量不可信。
pub const OVERFLOW_CHANNEL: i16 = 448;

/// 能量梯的头尾，用来核对 EBOUNDS 是不是我们标定过的那一张表。
const LADDER_MIN_KEV: f64 = 2.0;
const LADDER_MAX_KEV: f64 = 10053.5;
/// 溢出道的 `E_MAX` 哨兵值，两个版本都认。
///
/// 标定库的生成器在 2022-01-14 改过一次：之前最后 50 道的上下阈一律填 0，
/// 之后下阈改成 4096 道对应的能量、上阈改成 20 MeV。GECAM-B/C 跟进了新版，
/// GECAM-A 的 `pi/` 只有 2023-01-10 那一版、一直在用旧生成器，所以 A 星
/// **全部** 4,609 小时的 ch448 `E_MAX` 都是 0.0（实测 9 个 epoch 横跨
/// 2022-10 到 2026-06，24 个文件无一例外）。0.0 和 20000 是同一个约定的
/// 两个版本号，不是哪一个填错了，所以两者都放行。
///
/// 放行的只有这一条。道 0..447 那把梯子的连续性、头尾、以及 40 keV 落在
/// ch54，A 星与 B 星逐道完全一致、全程都过——那几条才是真正护住能阈的闸，
/// 一条都没有放松。
const OVERFLOW_MAX_KEV: f64 = 20000.0;
const OVERFLOW_MAX_KEV_LEGACY: f64 = 0.0;

pub(super) struct EboundsHdu {
    e_min: Vec<f32>,
    e_max: Vec<f32>,
}

impl EboundsHdu {
    pub fn from_fptr(fptr: &mut fitsio::FitsFile) -> Result<Self, fitsio::errors::Error> {
        let ebounds = fptr.hdu("EBOUNDS")?;
        Ok(Self {
            e_min: ebounds.read_col::<f32>(fptr, "E_MIN")?,
            e_max: ebounds.read_col::<f32>(fptr, "E_MAX")?,
        })
    }

    /// 核对这张表就是标定过的那一条能量梯。对不上宁可整小时报错，也不能
    /// 让 `Event::keep` 拿着一个含义变了的道号继续切。
    pub fn verify(&self) -> Result<(), Error> {
        let ladder = OVERFLOW_CHANNEL as usize;
        if self.e_min.len() <= ladder {
            return Err(Error::InvalidData(format!(
                "GECAM EBOUNDS 只有 {} 道，不足以容纳 0..{} 的能量梯",
                self.e_min.len(),
                ladder - 1
            )));
        }
        let head = self.e_min[0] as f64;
        let tail = self.e_max[ladder - 1] as f64;
        if (head - LADDER_MIN_KEV).abs() > 0.01 || (tail - LADDER_MAX_KEV).abs() > 1.0 {
            return Err(Error::InvalidData(format!(
                "GECAM 能量梯变了：ch0 起于 {head:.2} keV（应为 {LADDER_MIN_KEV:.2}）、\
                 ch{} 止于 {tail:.1} keV（应为 {LADDER_MAX_KEV:.1}）",
                ladder - 1
            )));
        }
        // 梯内必须连续单调，否则道号与能量的对应关系就不是我们以为的那个
        for k in 1..ladder {
            let (previous, current) = (self.e_max[k - 1] as f64, self.e_min[k] as f64);
            if (current - previous).abs() > previous * 1e-4 {
                return Err(Error::InvalidData(format!(
                    "GECAM 能量梯在 ch{k} 处断开：{previous:.3} → {current:.3} keV"
                )));
            }
        }
        // 溢出道确实是溢出道：它不能是能量梯的延续。梯子在 ch447 收在
        // 10053.5 keV，接着往下走一道该是 10190 上下，两个哨兵值离得都很远。
        let overflow = self.e_max[ladder] as f64;
        let sentinel = (overflow - OVERFLOW_MAX_KEV).abs() <= 1.0
            || (overflow - OVERFLOW_MAX_KEV_LEGACY).abs() <= 1.0;
        if !sentinel {
            return Err(Error::InvalidData(format!(
                "GECAM ch{ladder} 的 E_MAX 是 {overflow:.1} keV，\
                 不是溢出道该有的 {OVERFLOW_MAX_KEV:.0}（新生成器）\
                 或 {OVERFLOW_MAX_KEV_LEGACY:.0}（2022-01-14 之前的旧生成器）"
            )));
        }
        Ok(())
    }

    /// 能量梯上第一个上边界超过 `energy_kev` 的道号——把能阈折成道号切。
    /// 只在梯内找；找不到（阈值高过梯顶）返回溢出道起点，等于全切掉。
    pub fn channel_above(&self, energy_kev: f64) -> i16 {
        let ladder = (OVERFLOW_CHANNEL as usize).min(self.e_max.len());
        self.e_max[..ladder]
            .iter()
            .position(|top| *top as f64 > energy_kev)
            .map_or(OVERFLOW_CHANNEL, |index| index as i16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 造一张与实测同形状的表：0..447 是对数梯，其后是溢出道。
    fn ebounds(ladder_top: f64) -> EboundsHdu {
        let n = OVERFLOW_CHANNEL as usize;
        let step = (ladder_top / LADDER_MIN_KEV).powf(1.0 / n as f64);
        let mut e_min = Vec::with_capacity(n + 4);
        let mut e_max = Vec::with_capacity(n + 4);
        let mut edge = LADDER_MIN_KEV;
        for _ in 0..n {
            e_min.push(edge as f32);
            edge *= step;
            e_max.push(edge as f32);
        }
        for _ in 0..4 {
            e_min.push(5000.0);
            e_max.push(OVERFLOW_MAX_KEV as f32);
        }
        EboundsHdu { e_min, e_max }
    }

    #[test]
    fn the_calibrated_table_passes() {
        assert!(ebounds(LADDER_MAX_KEV).verify().is_ok());
    }

    #[test]
    fn a_shifted_ladder_is_rejected_rather_than_used() {
        // 梯顶挪了一倍：道号的能量含义变了，不能拿旧阈值继续切
        let shifted = ebounds(LADDER_MAX_KEV * 2.0);
        assert!(shifted.verify().is_err());
    }

    #[test]
    fn a_broken_ladder_is_rejected() {
        let mut broken = ebounds(LADDER_MAX_KEV);
        broken.e_min[200] *= 1.5;
        assert!(broken.verify().is_err());
    }

    /// GECAM-A 的标定库停在 2022-01-14 之前的生成器上，最后 50 道填 0 而不是
    /// 20 MeV。梯子本身逐道与 B 星一致，所以这张表要照收。
    #[test]
    fn the_legacy_overflow_sentinel_is_accepted() {
        let mut legacy = ebounds(LADDER_MAX_KEV);
        for k in OVERFLOW_CHANNEL as usize..legacy.e_max.len() {
            legacy.e_min[k] = 0.0;
            legacy.e_max[k] = 0.0;
        }
        assert!(legacy.verify().is_ok());
    }

    /// 放行 0.0 不能顺带放行"溢出道其实是梯子的延续"这种表——那种表里道号
    /// 的能量含义就变了，正是这条校验要挡的东西。
    #[test]
    fn an_overflow_row_that_continues_the_ladder_is_still_rejected() {
        let mut continued = ebounds(LADDER_MAX_KEV);
        let ladder = OVERFLOW_CHANNEL as usize;
        continued.e_min[ladder] = continued.e_max[ladder - 1];
        continued.e_max[ladder] = continued.e_max[ladder - 1] * 1.0134;
        assert!(continued.verify().is_err());
    }

    #[test]
    fn a_table_without_the_overflow_block_is_rejected() {
        let mut short = ebounds(LADDER_MAX_KEV);
        short.e_min.truncate(OVERFLOW_CHANNEL as usize);
        short.e_max.truncate(OVERFLOW_CHANNEL as usize);
        assert!(short.verify().is_err());
    }

    #[test]
    fn channel_above_returns_the_first_channel_whose_top_clears_the_energy() {
        // 实测的能量梯低端疏、高端密，不是均匀对数，所以「40 keV 落在 ch54」
        // 这件事没法用合成表来验——那一条由 `EvtFile` 载入真文件时正面核对。
        // 这里验的是查表本身的语义。
        let table = EboundsHdu {
            e_min: vec![2.0, 10.0, 30.0, 60.0],
            e_max: vec![10.0, 30.0, 60.0, 100.0],
        };
        assert_eq!(table.channel_above(1.0), 0);
        assert_eq!(table.channel_above(10.0), 1, "上边界相等时要落到下一道");
        assert_eq!(table.channel_above(40.0), 2);
        assert_eq!(table.channel_above(99.0), 3);
    }

    #[test]
    fn a_threshold_above_the_ladder_keeps_nothing() {
        assert_eq!(
            ebounds(LADDER_MAX_KEV).channel_above(LADDER_MAX_KEV * 2.0),
            OVERFLOW_CHANNEL
        );
    }
}
