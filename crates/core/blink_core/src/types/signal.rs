use chrono::{DateTime, TimeDelta, Utc};
use serde::{Deserialize, Serialize};
use uom::si::f64::*;

use crate::{
    traits::{Event, Instrument},
    types::{Attitude, MissionElapsedTime, Position},
};

/// 反符合探测器（ACD）符合计数，候选生成时从事例流现场统计——事后无法从
/// 候选表复原，必须随产物保存。只存原始计数不存比例：保泊松误差，阈值可
/// 复议而不必重跑全量。仅 HXMT HE 填写；无 ACD 的仪器留 `None`。
#[derive(Clone, Serialize, Deserialize)]
pub struct AcdCounts {
    /// 候选窗 [start, stop] 内 kept 事例总数。自带分母：不依赖 `count`
    /// 的分箱语义，占比 n_acd/n 自洽。
    pub n: u32,
    /// 候选窗内任意一块 ACD 着火的事例数
    pub n_acd: u32,
    /// 候选窗内 ≥2 块 ACD 同时着火的事例数（区分真带电粒子与偶然符合）
    pub n_acd_multi: u32,
    /// 邻域基线窗内 kept 事例总数
    pub n_bg: u32,
    /// 邻域基线窗内任意一块 ACD 着火的事例数
    pub n_acd_bg: u32,
}

/// 候选窗内逐路探测器的计数，下标即探头序（GECAM 是 `EVENTS01..NN` 的顺序）。
///
/// 和 `AcdCounts` 一样，这是候选生成时从事例流现场统计的量，事后无法从候选表
/// 复原——不当场存下来，日后要做方向分析就得重跑全量（GECAM-C 5.8 TB、
/// GECAM-B 110 TB）。
///
/// 用途是方向：探头朝向各不相同，逐路计数的相对高低就编码了入射方向。GECAM
/// 的 25 路 GRD 有 401 个方向的蒙卡响应（CALDB `mc_rsp`），拿这个向量去拟合
/// 就能定方向；判断"从天顶来还是从地球来"更是不需要拟合，看朝地那几路有没有
/// 亮就够——而这一条正是 TGF 与 GRB、TGF 与 TEB 的分界。探头少的仪器（天格 4
/// 路、GRM 3 路）填了也无妨，只是能问的问题少些。
///
/// 只存计数不存比例：保泊松误差，判据日后可复议而不必重跑。
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct DetectorCounts {
    /// 候选窗 `[start, stop]` 内逐路通过准入的事例数
    pub window: Vec<u32>,
    /// 邻域基线窗内逐路的计数，同样的下标。没有它就没法把"这一路亮"
    /// 和"这一路本来就快"区分开。
    pub baseline: Vec<u32>,
    /// 基线窗的总长（秒）。窗长不同，两组计数不能直接比。
    pub baseline_seconds: f64,
}

#[derive(Serialize, Deserialize)]
pub struct Signal<E: Event> {
    pub start: MissionElapsedTime<E::Instrument>,
    pub stop: MissionElapsedTime<E::Instrument>,
    pub bin_size_min: Time,
    pub bin_size_max: Time,
    pub bin_size_best: Time,
    pub delay: Time,
    pub count: u32,
    pub mean: f64,
    pub sf: f64,
    pub false_positive_per_year: f64,
    /// 峰值时刻的姿态。候选的实质是时间加位置，姿态只是方向分析用的元数据：
    /// 位姿表里姿态解整段缺失时（天格常见）候选照留，这里为 None、不写进 JSON。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attitude: Option<Attitude>,
    pub position: Position,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acd: Option<AcdCounts>,
    /// 逐路探测器计数，见 `DetectorCounts`。探头多的仪器填，其余留 `None`。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detectors: Option<DetectorCounts>,
}

impl<E: Event> Signal<E> {
    pub fn to_unified(&self) -> UnifiedSignal {
        UnifiedSignal {
            start: self.start.to_utc(),
            stop: self.stop.to_utc(),
            bin_size_min: self.bin_size_min,
            bin_size_max: self.bin_size_max,
            bin_size_best: self.bin_size_best,
            delay: self.delay,
            count: self.count,
            mean: self.mean,
            sf: self.sf,
            false_positive_per_year: self.false_positive_per_year,
            attitude: self.attitude.clone(),
            position: self.position.clone(),
            instrument: <E::Instrument as Instrument>::name().to_string(),
            acd: self.acd.clone(),
            detectors: self.detectors.clone(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct UnifiedSignal {
    pub start: DateTime<Utc>,
    pub stop: DateTime<Utc>,
    pub bin_size_min: Time,
    pub bin_size_max: Time,
    pub bin_size_best: Time,
    pub delay: Time,
    pub count: u32,
    pub mean: f64,
    pub sf: f64,
    pub false_positive_per_year: f64,
    /// 见 `Signal::attitude`；`default` 兼容旧文件（缺字段 → None）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attitude: Option<Attitude>,
    pub position: Position,
    pub instrument: String,
    /// `default` 兼容旧 signals.json（无此字段 → None），None 不序列化。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acd: Option<AcdCounts>,
    /// 见 `DetectorCounts`；`default` 兼容旧文件。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detectors: Option<DetectorCounts>,
}

impl UnifiedSignal {
    pub fn peak_time(&self) -> DateTime<Utc> {
        self.start
            + TimeDelta::nanoseconds(self.delay.get::<uom::si::time::nanosecond>().round() as i64)
            + TimeDelta::nanoseconds(
                (self.bin_size_best / 2.0)
                    .get::<uom::si::time::nanosecond>()
                    .round() as i64,
            )
    }
}
