use crate::types::Position;

/// 曝光沿轨道的一个采样：活时间 `[met − seconds/2, met + seconds/2)` 里卫星在
/// `position`。取不到星历时 `position` 为 `None`——这段活时间照样算进总数，但落不到
/// 地图上，调用方要单独计数，不能静默丢掉。
#[derive(Clone, Debug)]
pub struct ExposureSample {
    pub met: f64,
    pub seconds: f64,
    pub position: Option<Position>,
}
