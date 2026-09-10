use blink_core::types::MissionElapsedTime;
use serde::Serialize;

use crate::io::evt::OVERFLOW_CHANNEL;
use crate::types::instrument::{Gecam, Satellite};

/// 正常事例的 `EVT_TYPE`。实测 GECAM-B 某小时单路 94.0%、GECAM-C 84.9% 是 1；
/// 其余是 2，PI 全部落在溢出道（≥ 448），能量不可信。
const NORMAL_EVT_TYPE: u8 = 1;

/// 低能阈，道号。ch54 = 39.1–40.2 keV。
///
/// 这是个有依据的起点，不是定标结果。SVOM/GRM 上用 72 个闪电证实的 TGF 扫过
/// 能阈，结论是 42 keV（该仪器的 ch25）：证实 TGF 的窗内谱在更低的道上一律
/// 低于本底，从 42 keV 起才反超，切在这里少 15% 信号、少 43% 本底，信噪反升。
/// GECAM 的谱学响应不同（LaBr₃ 对 NaI/CsI，能量梯一路到 10 MeV），这条结论
/// 不能直接搬，得拿 GECAM 自己的闪电证实样本重扫一遍——见本 crate 的
/// OPEN-QUESTIONS。在那之前先取同一量级的 40 keV。
///
/// 按能量取而不是按道号拍脑袋：三颗星的能量梯逐道一致（载入时核对，见
/// `EboundsHdu::verify`），所以这一个道号常量对 A/B/C 通用。切的时候用道号
/// 是为了不给每个事例都背一个能量值——GECAM-B 一小时 1.5 亿个事例，四个字节
/// 就是 600 MB。道号与能量的对应由 `EvtFile` 载入时核对。
pub(crate) const MIN_CHANNEL: i16 = 54;

/// `MIN_CHANNEL` 对应的能量。载入 EBOUNDS 时用它反查道号核对，见 `EvtFile`。
pub(crate) const MIN_ENERGY_KEV: f64 = 40.0;

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

    /// 事例准入。
    ///
    /// * `EVT_TYPE == 1` 才是正常事例，2 是量程溢出。
    /// * 道号要落在能量梯内。溢出道（≥ 448）的 `E_MAX` 一律是 20000 keV 的
    ///   哨兵值、`E_MIN` 逐探头逐增益各不相同，能量无从谈起。实测溢出事例
    ///   的 `EVT_TYPE` 都是 2，这一条与上一条重复——留着是因为两者一个来自
    ///   标志位、一个来自能标，任一处出岔子另一处还能兜住。
    /// * 低能截止见 `MIN_CHANNEL`。
    fn keep(&self) -> bool {
        self.evt_type == NORMAL_EVT_TYPE
            && self.channel >= MIN_CHANNEL
            && self.channel < OVERFLOW_CHANNEL
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

    #[test]
    fn a_normal_event_above_the_threshold_is_kept() {
        assert!(event(MIN_CHANNEL, NORMAL_EVT_TYPE).keep());
        assert!(event(200, NORMAL_EVT_TYPE).keep());
        assert!(event(OVERFLOW_CHANNEL - 1, NORMAL_EVT_TYPE).keep());
    }

    #[test]
    fn soft_events_below_the_threshold_are_dropped() {
        assert!(!event(MIN_CHANNEL - 1, NORMAL_EVT_TYPE).keep());
        assert!(!event(0, NORMAL_EVT_TYPE).keep());
    }

    #[test]
    fn overflow_is_dropped_by_the_flag_and_by_the_channel_alike() {
        // 实测这两条同时成立；各留一条，任一处出岔子另一处还能兜住
        assert!(!event(OVERFLOW_CHANNEL, 2).keep());
        assert!(!event(OVERFLOW_CHANNEL, NORMAL_EVT_TYPE).keep());
        assert!(!event(200, 2).keep());
    }
}
