use blink_core::types::MissionElapsedTime;
use serde::Serialize;

use crate::types::instrument::{Gecam, Satellite};

/// 正常事例的 `EVT_TYPE`。实测 GECAM-B 某小时单路 94.0%、GECAM-C 84.9% 是 1；
/// 其余是 2，PI 全部落在梯外的溢出道，能量不可信。
const NORMAL_EVT_TYPE: u8 = 1;

/// 低能阈，能量。
///
/// 这是个有依据的起点，不是定标结果。SVOM/GRM 上用 72 个闪电证实的 TGF 扫过
/// 能阈，结论是 42 keV（该仪器的 ch25）：证实 TGF 的窗内谱在更低的道上一律
/// 低于本底，从 42 keV 起才反超，切在这里少 15% 信号、少 43% 本底，信噪反升。
/// GECAM 的谱学响应不同（LaBr₃ 对 NaI/CsI，能量梯一路到 10 MeV），这条结论
/// 不能直接搬，得拿 GECAM 自己的闪电证实样本重扫一遍——见本 crate 的
/// OPEN-QUESTIONS。在那之前先取同一量级的 40 keV。
///
/// **阈值存能量、切的时候用道号，中间那一步逐文件算**：切用道号是为了不给每个
/// 事例都背一个能量值（GECAM-B 一小时 1.5 亿个事例，四个字节就是 600 MB），
/// 而落在哪一道由该文件的 EBOUNDS 决定，见 [`ChannelWindow`]。
///
/// **原先这里写死着「40 keV 落在 ch54」，并注明"三颗星逐道一致，所以这一个道号
/// 常量对 A/B/C 通用"——那句话已被 GECAM-C 2022-08-03 .. 10-15 的 896 道纪元
/// 证伪**：同一个 40 keV 在那把梯子上落在 ch109，ch54 只有 14.16 keV。
pub(crate) const MIN_ENERGY_KEV: f64 = 40.0;

/// 事例准入的道号窗，**逐文件**由该文件的 EBOUNDS 现算（[`crate::io::EvtFile`]
/// 载入时），不是编译期常量。
///
/// 归档里有两把能量梯，同一个道号在它们上面差一倍能量（470 道版 ch54 是
/// 39.1–40.2 keV，896 道版 ch54 是 14.16–14.54 keV），所以把道号阈值写死就等于
/// 在细梯那 1,121 小时上把能窗整个搬错位——14 keV 以上全收、514 keV 封顶。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelWindow {
    /// 能阈落在的道号，[`MIN_ENERGY_KEV`] 在本文件的梯子上的位置。
    min_channel: i16,
    /// 梯长。道号 ≥ 此值的不在梯上：470 道版那里起是溢出块
    /// （`E_MAX` 一律 20000 keV 的哨兵、`E_MIN` 逐探头逐增益各不相同），
    /// 896 道版梯子走到表尾、这一段是空的。
    ladder_length: i16,
}

impl ChannelWindow {
    pub(crate) fn new(min_channel: i16, ladder_length: i16) -> Self {
        Self {
            min_channel,
            ladder_length,
        }
    }

    /// 能阈道。
    pub fn min_channel(&self) -> i16 {
        self.min_channel
    }

    /// 梯长（= 梯外第一道的道号）。
    pub fn ladder_length(&self) -> i16 {
        self.ladder_length
    }

    /// 完整的事例准入：标志位那一半（[`Event::keep`]）**加上**道号窗。
    ///
    /// 搜索一律走这里。单独用 `keep()` 会把道号窗整个漏掉。
    pub fn admits<S: Satellite>(&self, event: &Event<S>) -> bool {
        use blink_core::traits::Event as _;
        event.keep() && event.channel >= self.min_channel && event.channel < self.ladder_length
    }
}

#[derive(Serialize, Debug, Clone)]
#[serde(bound(serialize = ""))]
pub struct Event<S: Satellite> {
    pub time: MissionElapsedTime<Gecam<S>>,
    pub channel: i16,
    /// 第几路 GRD，从 1 数起（`EVENTS01` .. `EVENTS25`）。
    pub detector_id: u8,
    /// 0 高增益、1 低增益。两档共用同一套 PI 刻度，所以道号可以直接比。
    pub gain_type: u8,
    /// 本事例的死时间（µs），留给事后的死时间修正。
    pub dead_time: f32,
    pub evt_type: u8,
    pub flag: u8,
}

impl<S: Satellite> blink_core::traits::Event for Event<S> {
    type Instrument = Gecam<S>;
    type ChannelType = i16;

    fn time(&self) -> MissionElapsedTime<Self::Instrument> {
        self.time
    }

    fn channel(&self) -> Self::ChannelType {
        self.channel
    }

    fn group(&self) -> u8 {
        0
    }

    /// 事例准入的**标志位那一半**：`EVT_TYPE == 1` 才是正常事例，2 是量程溢出。
    ///
    /// **另一半——道号窗——不在这里，在 [`ChannelWindow`]**，因为两端都是逐文件
    /// 现算的，而这个 trait 方法拿不到文件上下文。要完整准入请用
    /// `ChannelWindow::admits`；单独调 `keep()` 会把能阈和梯顶整个漏掉。
    ///
    /// 道号窗那一半的实测情况记在这里，免得散掉：溢出道的 `E_MAX` 一律是
    /// 20000 keV 的哨兵值、`E_MIN` 逐探头逐增益各不相同，能量无从谈起。
    /// **实测梯顶那一条一票都没否过**：`EVT_TYPE == 1` 的事例里 `PI >= 448`
    /// 占 0.00%（GECAM-A 25 路 × 3 个过境无一例外），溢出段与 `EVT_TYPE == 2`
    /// 是同一个集合。留着是因为两者一个来自标志位、一个来自能标，任一处出岔子
    /// 另一处还能兜住。
    ///
    /// **但梯顶不是真正的量程上限。** 实测低增益支路的高能端在
    /// **ch≈379（约 4.0 MeV，逐路 375–381）** 就是悬崖，之下还堆着一个
    /// 12–22 倍于平台的超量程沉积包（ch368–383），**这批事例现在以
    /// 3.9–4.2 MeV 硬光子的身份进搜索和谱学**。逐探头的量程上限该不该
    /// 进准入见本 crate 的 OPEN-QUESTIONS——那是准入口径问题，不是新判据。
    fn keep(&self) -> bool {
        self.evt_type == NORMAL_EVT_TYPE
    }
}

impl<S: Satellite> Ord for Event<S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.time.cmp(&other.time)
    }
}
impl<S: Satellite> PartialOrd for Event<S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<S: Satellite> PartialEq for Event<S> {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time
    }
}
impl<S: Satellite> Eq for Event<S> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::instrument::SatB;
    use blink_core::traits::Event as _;

    fn event(channel: i16, evt_type: u8) -> Event<SatB> {
        Event {
            time: MissionElapsedTime::new(0.0),
            channel,
            detector_id: 1,
            gain_type: 0,
            dead_time: 4.0,
            evt_type,
            flag: 0,
        }
    }

    /// 实测的 470 道梯子：能阈 ch54、梯长 448。
    const COARSE: ChannelWindow = ChannelWindow {
        min_channel: 54,
        ladder_length: 448,
    };
    /// 实测的 896 道梯子（GECAM-C 2022-08-03 .. 10-15）：同一个 40 keV 落在 ch109。
    const FINE: ChannelWindow = ChannelWindow {
        min_channel: 109,
        ladder_length: 896,
    };

    #[test]
    fn a_normal_event_above_the_threshold_is_admitted() {
        assert!(COARSE.admits(&event(COARSE.min_channel, NORMAL_EVT_TYPE)));
        assert!(COARSE.admits(&event(200, NORMAL_EVT_TYPE)));
        assert!(COARSE.admits(&event(COARSE.ladder_length - 1, NORMAL_EVT_TYPE)));
    }

    #[test]
    fn soft_events_below_the_threshold_are_dropped() {
        assert!(!COARSE.admits(&event(COARSE.min_channel - 1, NORMAL_EVT_TYPE)));
        assert!(!COARSE.admits(&event(0, NORMAL_EVT_TYPE)));
    }

    #[test]
    fn overflow_is_dropped_by_the_flag_and_by_the_channel_alike() {
        // 实测这两条同时成立；各留一条，任一处出岔子另一处还能兜住
        assert!(!COARSE.admits(&event(COARSE.ladder_length, 2)));
        assert!(!COARSE.admits(&event(COARSE.ladder_length, NORMAL_EVT_TYPE)));
        assert!(!COARSE.admits(&event(200, 2)));
    }

    /// 同一个道号在两把梯子上是两个能量，所以准入结果必须不同——**这正是
    /// 「支持第二把梯子」与「照搬道号阈值」的分水岭**：ch60 在 470 版上是
    /// 43 keV（收），在 896 版上是 15 keV（该切）。
    #[test]
    fn the_same_channel_is_admitted_on_one_ladder_and_cut_on_the_other() {
        let soft_on_the_fine_ladder = event(60, NORMAL_EVT_TYPE);
        assert!(COARSE.admits(&soft_on_the_fine_ladder));
        assert!(!FINE.admits(&soft_on_the_fine_ladder));
        // 细梯上 40 keV 起的那一道两边都收
        assert!(FINE.admits(&event(FINE.min_channel, NORMAL_EVT_TYPE)));
    }

    /// 细梯的梯顶是 ch895 而不是 ch447：470 版的梯顶在细梯上还是有效道。
    #[test]
    fn the_fine_ladder_reaches_past_where_the_coarse_one_stops() {
        assert!(FINE.admits(&event(COARSE.ladder_length, NORMAL_EVT_TYPE)));
        assert!(FINE.admits(&event(FINE.ladder_length - 1, NORMAL_EVT_TYPE)));
        assert!(!FINE.admits(&event(FINE.ladder_length, NORMAL_EVT_TYPE)));
    }

    /// `keep()` 只是标志位那一半，单独用会把道号窗漏掉——钉住这件事，
    /// 免得日后有人在别处 `.filter(|e| e.keep())` 就以为准入做完了。
    #[test]
    fn keep_alone_is_only_the_flag_half() {
        let too_soft = event(0, NORMAL_EVT_TYPE);
        assert!(too_soft.keep(), "标志位这一半通过");
        assert!(!COARSE.admits(&too_soft), "道号窗那一半要挡住它");
    }
}
