//! EBOUNDS：道号到能量的对照表，以及能量梯的形状。
//!
//! 归档里有**两把梯子**，不是一把：
//!
//! * **470 道版**（A/B/C 绝大多数时段）：道 0..447 是一条连续单调的对数能量梯，
//!   2.00 keV 到 10053.5 keV；道 448 及以上是溢出道，`E_MAX` 一律 20000 keV、
//!   `E_MIN` 不单调（每探头每增益档各占一行，B 有 50 行、C 有 22 行），
//!   能量不可信。事例表里 `EVT_TYPE == 2` 的事例 PI 正好全落在这一段。
//! * **896 道版**（GECAM-C 2022-08-03 .. 2022-10-15，实测 1,121 小时）：同样
//!   起于 2.00 keV、止于 10053.5 keV，但把同一段能量切成一倍细的 896 道，
//!   **一路连续到表尾、没有溢出块**。同一个道号在两把梯子上差一倍能量：
//!   ch54 在 470 版上是 39.1–40.2 keV，在 896 版上是 14.16–14.54 keV。
//!
//! 所以**道号阈值不能是编译期常量**，两个都得从这张表自己推：
//!
//! * **梯长** = 从 ch0 起 `E_MAX[k-1] == E_MIN[k]` 连续到断开为止的长度
//!   （470 版在 ch448 断，896 版不断到底）；
//! * **能阈道** = 梯上第一个上边界越过 [`crate::types::event::MIN_ENERGY_KEV`]
//!   的道（470 版 ch54，896 版 ch109）。
//!
//! 推不出来就整小时报错，不静默沿用一个含义变了的道号——照搬 ch54 会把能窗
//! 整个搬错位（14 keV 以上全收、514 keV 封顶）。

use blink_core::error::Error;

/// 能量梯的头尾，用来核对 EBOUNDS 是不是我们标定过的那把尺子。**两把梯子
/// 的头尾逐位相同**，差的只是中间切成多少道，所以这两条闸对两个版本一样紧。
const LADDER_MIN_KEV: f64 = 2.0;
const LADDER_MAX_KEV: f64 = 10053.5;
/// 梯内相邻道的对接容差（相对）。
const LADDER_JOIN_TOLERANCE: f64 = 1e-4;
/// 溢出道的 `E_MAX` 哨兵值，两个版本都认。
///
/// 标定库的生成器在 2022-01-14 改过一次：之前最后 50 道的上下阈一律填 0，
/// 之后下阈改成 4096 道对应的能量、上阈改成 20 MeV。GECAM-B/C 跟进了新版，
/// GECAM-A 的 `pi/` 只有 2023-01-10 那一版、一直在用旧生成器，所以 A 星
/// **全部** 4,609 小时的 ch448 `E_MAX` 都是 0.0（实测 9 个 epoch 横跨
/// 2022-10 到 2026-06，24 个文件无一例外）。0.0 和 20000 是同一个约定的
/// 两个版本号，不是哪一个填错了，所以两者都放行。
///
/// 896 道版压根没有溢出块（梯子一路到表尾），这一条在那一段没有输入。
const OVERFLOW_MAX_KEV: f64 = 20000.0;
const OVERFLOW_MAX_KEV_LEGACY: f64 = 0.0;

/// 这张表里的能量梯：长度与能阈道，两个都是**逐文件**现算的。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Ladder {
    /// 梯长：从 ch0 起连续单调的道数。道号 ≥ 此值的事例不在梯上，能量无从谈起。
    pub length: i16,
    /// 能阈落在的道号。
    pub min_channel: i16,
}

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

    /// 核对这张表就是标定过的那把尺子，并把梯长与能阈道算出来。对不上宁可
    /// 整小时报错，也不能让准入拿着一个含义变了的道号继续切。
    ///
    /// 放松的只有"梯子必须正好 448 道"这一条形状假定；**护住能量刻度的三道闸
    /// 一条没动**：ch0 起于 2.00 keV、梯内逐道连续、梯顶收在 10053.5 keV。
    pub fn ladder(&self, min_energy_kev: f64) -> Result<Ladder, Error> {
        if self.e_min.len() != self.e_max.len() {
            return Err(Error::InvalidData(format!(
                "GECAM EBOUNDS 的 E_MIN 有 {} 行、E_MAX 有 {} 行",
                self.e_min.len(),
                self.e_max.len()
            )));
        }
        // 梯长由连续性定出来，不认行数也不认版本号
        let length = self.continuous_prefix();
        if length < 2 {
            return Err(Error::InvalidData(format!(
                "GECAM EBOUNDS 的能量梯只有 {length} 道，连相邻道的对接都无从谈起"
            )));
        }
        // 道号是 i16（PI 列就是 `I`）。梯长溢出 i16 会绕成负数，而负的梯长
        // 不报错、只会让准入一个事例都不收——**静默清空**，正是这套校验要防的
        // 失效方式。已知的两把梯子是 448 与 896，离这条线还很远。
        if length > i16::MAX as usize {
            return Err(Error::InvalidData(format!(
                "GECAM EBOUNDS 的能量梯有 {length} 道，超出道号能表达的范围"
            )));
        }
        let head = self.e_min[0] as f64;
        let tail = self.e_max[length - 1] as f64;
        if (head - LADDER_MIN_KEV).abs() > 0.01 || (tail - LADDER_MAX_KEV).abs() > 1.0 {
            return Err(Error::InvalidData(format!(
                "GECAM 能量梯变了：ch0 起于 {head:.2} keV（应为 {LADDER_MIN_KEV:.2}）、\
                 ch{} 止于 {tail:.1} keV（应为 {LADDER_MAX_KEV:.1}）",
                length - 1
            )));
        }
        // 梯子没走到表尾时，后面那一段必须确实是溢出块，不能是梯子的延续。
        // 896 道版走到了表尾，没有这一段可查。
        if length < self.e_max.len() {
            let overflow = self.e_max[length] as f64;
            let sentinel = (overflow - OVERFLOW_MAX_KEV).abs() <= 1.0
                || (overflow - OVERFLOW_MAX_KEV_LEGACY).abs() <= 1.0;
            if !sentinel {
                return Err(Error::InvalidData(format!(
                    "GECAM ch{length} 的 E_MAX 是 {overflow:.1} keV，\
                     不是溢出道该有的 {OVERFLOW_MAX_KEV:.0}（新生成器）\
                     或 {OVERFLOW_MAX_KEV_LEGACY:.0}（2022-01-14 之前的旧生成器）"
                )));
            }
        }

        let min_channel = self.channel_above(min_energy_kev, length);
        if min_channel as usize >= length {
            return Err(Error::InvalidData(format!(
                "GECAM 能阈 {min_energy_kev} keV 高过梯顶 {tail:.1} keV，切完一个事例都不剩"
            )));
        }
        Ok(Ladder {
            length: length as i16,
            min_channel,
        })
    }

    /// 从 ch0 起连续单调的道数：`E_MAX[k-1] == E_MIN[k]` 一直成立到哪里。
    /// 470 版在 ch448 断（那里起是溢出块），896 版一路到表尾。
    fn continuous_prefix(&self) -> usize {
        for k in 1..self.e_min.len() {
            let (previous, current) = (self.e_max[k - 1] as f64, self.e_min[k] as f64);
            if (current - previous).abs() > previous * LADDER_JOIN_TOLERANCE {
                return k;
            }
        }
        self.e_min.len()
    }

    /// 能量梯上第一个上边界超过 `energy_kev` 的道号——把能阈折成道号切。
    /// 只在梯内找；找不到（阈值高过梯顶）返回梯长，等于全切掉。
    fn channel_above(&self, energy_kev: f64, ladder: usize) -> i16 {
        self.e_max[..ladder.min(self.e_max.len())]
            .iter()
            .position(|top| *top as f64 > energy_kev)
            .map_or(ladder as i16, |index| index as i16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 实测的 470 道梯子：ch0 = 2.00–2.183、ch54 = 39.118–40.177、ch447 收在
    /// 10053.5。合成表只能对上头尾和连续性，对不上"40 keV 落在 ch54"——真正的
    /// 梯子低端疏高端密，不是均匀对数。所以能阈道那一条由 `EvtFile` 载入真文件
    /// 时正面核对，这里的合成表只验形状闸与查表语义。
    const LADDER_470: usize = 448;
    const LADDER_896: usize = 896;

    /// 造一张与实测同形状的表：`ladder` 道对数梯，其后跟 `overflow` 行溢出道。
    fn ebounds(ladder: usize, ladder_top: f64, overflow: usize) -> EboundsHdu {
        let step = (ladder_top / LADDER_MIN_KEV).powf(1.0 / ladder as f64);
        let mut e_min = Vec::with_capacity(ladder + overflow);
        let mut e_max = Vec::with_capacity(ladder + overflow);
        let mut edge = LADDER_MIN_KEV;
        for k in 0..ladder {
            e_min.push(edge as f32);
            // 末道的上边界直接钉成梯顶，免得逐道累乘的浮点误差落到闸的容差外
            edge = if k + 1 == ladder {
                ladder_top
            } else {
                edge * step
            };
            e_max.push(edge as f32);
        }
        for _ in 0..overflow {
            e_min.push(5000.0);
            e_max.push(OVERFLOW_MAX_KEV as f32);
        }
        EboundsHdu { e_min, e_max }
    }

    /// 470 道版：梯长 448，后面 22 行是溢出块。
    fn ebounds_470() -> EboundsHdu {
        ebounds(LADDER_470, LADDER_MAX_KEV, 22)
    }

    /// 896 道版：同一段能量切成一倍细，**一路到表尾、没有溢出块**。
    fn ebounds_896() -> EboundsHdu {
        ebounds(LADDER_896, LADDER_MAX_KEV, 0)
    }

    #[test]
    fn the_calibrated_table_passes_and_reports_its_own_ladder_length() {
        let ladder = ebounds_470().ladder(40.0).expect("470 道版该过");
        assert_eq!(ladder.length, LADDER_470 as i16);
    }

    /// 896 道版整表连续、没有溢出块，照样得过——这是 2022-08-03 .. 10-15 那
    /// 1,121 小时能载入的全部理由。
    #[test]
    fn the_finer_ladder_without_an_overflow_block_passes_too() {
        let ladder = ebounds_896().ladder(40.0).expect("896 道版该过");
        assert_eq!(ladder.length, LADDER_896 as i16);
    }

    /// 两把梯子上的能阈道**必须不同**：同一个能量在细梯上落在约两倍的道号。
    /// 照搬道号就是把能窗整个搬错位，这条测试就是钉住那件事不许再发生。
    #[test]
    fn the_same_energy_lands_on_a_different_channel_on_each_ladder() {
        let coarse = ebounds_470().ladder(40.0).unwrap().min_channel;
        let fine = ebounds_896().ladder(40.0).unwrap().min_channel;
        assert!(fine > coarse, "细梯上的道号应更大：{fine} vs {coarse}");
        // 道数正好差一倍，能阈道也该差一倍上下（合成表是均匀对数，实测是 54 → 109）
        assert!(
            (fine as f64 / coarse as f64 - 2.0).abs() < 0.1,
            "{fine} / {coarse}"
        );
    }

    #[test]
    fn a_shifted_ladder_is_rejected_rather_than_used() {
        // 梯顶挪了一倍：道号的能量含义变了，不能拿旧阈值继续切
        assert!(
            ebounds(LADDER_470, LADDER_MAX_KEV * 2.0, 22)
                .ladder(40.0)
                .is_err()
        );
        // 细梯也要挡住，不能因为"支持第二把梯子"就把梯顶那道闸放了
        assert!(
            ebounds(LADDER_896, LADDER_MAX_KEV * 2.0, 0)
                .ladder(40.0)
                .is_err()
        );
    }

    /// 梯子中间断开：断点之前那截的梯顶不是 10053.5，整表拒收。
    #[test]
    fn a_broken_ladder_is_rejected() {
        let mut broken = ebounds_470();
        broken.e_min[200] *= 1.5;
        assert!(broken.ladder(40.0).is_err());
    }

    /// GECAM-A 的标定库停在 2022-01-14 之前的生成器上，最后 50 道填 0 而不是
    /// 20 MeV。梯子本身逐道与 B 星一致，所以这张表要照收。
    #[test]
    fn the_legacy_overflow_sentinel_is_accepted() {
        let mut legacy = ebounds_470();
        for k in LADDER_470..legacy.e_max.len() {
            legacy.e_min[k] = 0.0;
            legacy.e_max[k] = 0.0;
        }
        assert_eq!(legacy.ladder(40.0).unwrap().length, LADDER_470 as i16);
    }

    /// 放行 0.0 不能顺带放行"溢出道其实是梯子的延续"这种表——那种表里道号
    /// 的能量含义就变了，正是这条校验要挡的东西。梯长由连续性定出来之后，
    /// 挡住它的是梯顶那道闸：延续一道，梯顶就不再是 10053.5。
    #[test]
    fn an_overflow_row_that_continues_the_ladder_is_still_rejected() {
        let mut continued = ebounds_470();
        continued.e_min[LADDER_470] = continued.e_max[LADDER_470 - 1];
        continued.e_max[LADDER_470] = continued.e_max[LADDER_470 - 1] * 1.0134;
        assert!(continued.ladder(40.0).is_err());
    }

    /// 溢出块的 `E_MAX` 既不是 20000 也不是 0：不知道那一段是什么，整表拒收。
    #[test]
    fn an_unknown_overflow_sentinel_is_rejected() {
        let mut odd = ebounds_470();
        for k in LADDER_470..odd.e_max.len() {
            odd.e_max[k] = 1234.0;
        }
        assert!(odd.ladder(40.0).is_err());
    }

    #[test]
    fn a_table_that_is_not_a_ladder_at_all_is_rejected() {
        let table = EboundsHdu {
            e_min: vec![2.0],
            e_max: vec![10.0],
        };
        assert!(table.ladder(40.0).is_err());
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
        assert_eq!(table.channel_above(1.0, 4), 0);
        assert_eq!(table.channel_above(10.0, 4), 1, "上边界相等时要落到下一道");
        assert_eq!(table.channel_above(40.0, 4), 2);
        assert_eq!(table.channel_above(99.0, 4), 3);
        // 只在梯内找：梯长收到 2 就够不到 ch2 了
        assert_eq!(table.channel_above(40.0, 2), 2);
    }

    #[test]
    fn a_threshold_above_the_ladder_keeps_nothing_and_is_an_error() {
        assert!(ebounds_470().ladder(LADDER_MAX_KEV * 2.0).is_err());
    }
}
